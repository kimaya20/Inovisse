"""Chunking, embeddings, and FAISS search."""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import faiss
import numpy as np
from pypdf import PdfReader

from src import config

EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
HUB_REPO = 'sentence-transformers/all-MiniLM-L6-v2'
CHUNK_CHARS = 500
CHUNK_OVERLAP = 100
ITEM_HEADING_RE = re.compile(r'^(Item\s+\d+[A-Z]?\.\s*[^\n.]{1,80})', re.IGNORECASE)
MAX_SEQ_LEN = 256


def _offline() -> bool:
    return os.environ.get('HF_HUB_OFFLINE', '').strip().lower() in {'1', 'true', 'yes'}


class _OnnxMiniLM:
    """all-MiniLM-L6-v2 via ONNX Runtime when torch cannot load."""

    def __init__(self, local_files_only: bool) -> None:
        from huggingface_hub import hf_hub_download
        import onnxruntime as ort
        from tokenizers import Tokenizer

        tokenizer_path = hf_hub_download(
            HUB_REPO, 'tokenizer.json', local_files_only=local_files_only,
        )
        model_path = hf_hub_download(
            HUB_REPO, 'onnx/model.onnx', local_files_only=local_files_only,
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        self.session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider'],
        )

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        encodings = self.tokenizer.encode_batch(list(texts))
        input_ids = np.array([item.ids for item in encodings], dtype=np.int64)
        attention_mask = np.array(
            [item.attention_mask for item in encodings], dtype=np.int64,
        )
        token_type_ids = np.array([item.type_ids for item in encodings], dtype=np.int64)
        last_hidden = self.session.run(
            None,
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
            },
        )[0]
        mask = attention_mask.astype(np.float32)[:, :, None]
        pooled = (last_hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        if normalize_embeddings:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / np.clip(norms, 1e-12, None)
        return pooled.astype(np.float32)


def _load_embedder():
    """Load MiniLM from cache when possible; honor HF_HUB_OFFLINE.

    Prefers sentence-transformers. Falls back to the same Hub model's ONNX
    weights if torch is blocked (Windows Application Control).
    """
    offline = _offline()
    try:
        from sentence_transformers import SentenceTransformer
        try:
            return SentenceTransformer(
                EMBEDDING_MODEL, device='cpu', local_files_only=True,
            )
        except Exception:
            if offline:
                raise
            return SentenceTransformer(EMBEDDING_MODEL, device='cpu')
    except (OSError, ImportError):
        try:
            return _OnnxMiniLM(local_files_only=True)
        except Exception:
            if offline:
                raise
            return _OnnxMiniLM(local_files_only=False)


def _is_heading_only(paragraph: str) -> bool:
    line = paragraph.strip()
    if '|' in line:
        return False
    if ITEM_HEADING_RE.match(line) and line.count('.') <= 1:
        return True
    if len(line) > 80 or line.endswith('.'):
        return False
    words = line.split()
    return 1 <= len(words) <= 12 and line[0].isupper()


def _heading_from(paragraph: str) -> Optional[str]:
    match = ITEM_HEADING_RE.match(paragraph.strip())
    if match:
        return match.group(1).strip()
    if _is_heading_only(paragraph):
        return paragraph.strip()
    return None


def _split_overlap(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    step = size - overlap
    pieces = []
    start = 0
    while start < len(text):
        pieces.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += step
    return pieces


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r'\n\s*\n', text) if part.strip()]


def chunk_text(text: str) -> list[tuple[str, str]]:
    """Split into ~500-character chunks with 100-character overlap.

    Returns (chunk_text, page_section) pairs. page_section is the nearest
    heading, or 'chunk N' when there is no heading.
    """
    pieces: list[tuple[str, Optional[str]]] = []
    heading: Optional[str] = None
    for paragraph in _paragraphs(text):
        found = _heading_from(paragraph)
        if found:
            heading = found
        if _is_heading_only(paragraph):
            continue
        for part in _split_overlap(paragraph, CHUNK_CHARS, CHUNK_OVERLAP):
            pieces.append((part, heading))

    merged: list[tuple[str, Optional[str]]] = []
    buf = ''
    buf_heading: Optional[str] = None
    for part, part_heading in pieces:
        candidate = part if not buf else f'{buf}\n\n{part}'
        if buf and len(candidate) > CHUNK_CHARS:
            merged.append((buf, buf_heading))
            overlap = buf[-CHUNK_OVERLAP:] if len(buf) >= CHUNK_OVERLAP else ''
            buf = f'{overlap}\n\n{part}' if overlap else part
            buf_heading = part_heading
            if len(buf) > CHUNK_CHARS:
                merged.append((buf[:CHUNK_CHARS], buf_heading))
                buf = buf[CHUNK_CHARS - CHUNK_OVERLAP:]
        else:
            buf = candidate
            buf_heading = buf_heading or part_heading
    if buf.strip():
        merged.append((buf, buf_heading))

    labelled = []
    for index, (chunk, chunk_heading) in enumerate(merged, start=1):
        section = chunk_heading if chunk_heading else f'chunk {index}'
        labelled.append((chunk, section))
    return labelled


class Retriever:
    """In-memory MiniLM + FAISS passage retriever."""

    def __init__(self) -> None:
        self.model = _load_embedder()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[dict] = []

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype='float32')

    def _add_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        vectors = self._embed([chunk['text'] for chunk in chunks])
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def add_document(
        self,
        text: str,
        source_id: str,
        title: str,
        date: str,
        doc_type: str,
    ) -> None:
        """Chunk, embed, and index a plain-text document."""
        if not text or not text.strip():
            return
        chunks = []
        for offset, (chunk_text_value, page_section) in enumerate(chunk_text(text)):
            chunks.append({
                'text': chunk_text_value,
                'source_id': source_id,
                'title': title,
                'doc_type': doc_type,
                'date': date,
                'page_section': page_section,
                'chunk_id': f'{source_id}:{offset}',
            })
        self._add_chunks(chunks)

    def add_pdf(
        self,
        path: str,
        source_id: str,
        title: str,
        date: str,
        doc_type: str,
    ) -> None:
        """Extract PDF text with pypdf, then index it like any other document."""
        reader = PdfReader(path)
        pages = [(page.extract_text() or '') for page in reader.pages]
        self.add_document('\n\n'.join(pages), source_id, title, date, doc_type)

    def load_from_metadata(self, docs_dir: str) -> None:
        """Index each file listed in metadata.json; skip entries whose file is missing."""
        path = os.path.join(docs_dir, 'metadata.json')
        with open(path, encoding='utf-8') as handle:
            entries = json.load(handle)
        for entry in entries:
            file_path = os.path.join(docs_dir, entry['file'])
            if not os.path.isfile(file_path):
                continue
            if file_path.lower().endswith('.pdf'):
                self.add_pdf(
                    file_path,
                    entry['source_id'],
                    entry['title'],
                    entry['date'],
                    entry['doc_type'],
                )
                continue
            with open(file_path, encoding='utf-8') as handle:
                text = handle.read()
            self.add_document(
                text,
                entry['source_id'],
                entry['title'],
                entry['date'],
                entry['doc_type'],
            )

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        source_id: Optional[str] = None,
        doc_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """Return top passages. Filters apply before the top-k cut."""
        if k is None:
            k = config.TOP_K
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = self._embed([query])
        scores, ids = self.index.search(query_vec, self.index.ntotal)
        results: list[dict] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            if source_id is not None and chunk['source_id'] != source_id:
                continue
            if doc_types is not None and chunk['doc_type'] not in doc_types:
                continue
            results.append({
                'text': chunk['text'],
                'score': float(score),
                'source_id': chunk['source_id'],
                'title': chunk['title'],
                'page_section': chunk['page_section'],
                'date': chunk['date'],
                'doc_type': chunk['doc_type'],
            })
            if len(results) >= k:
                break
        return results
