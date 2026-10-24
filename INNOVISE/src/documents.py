"""Local document extraction and query-ranked evidence; no model required."""
from io import BytesIO
import re
from pypdf import PdfReader

MAX_BYTES = 25 * 1024 * 1024

def extract_pages(data: bytes, filename: str) -> list[dict]:
    if len(data) > MAX_BYTES:
        raise ValueError('Maximum file size is 25 MB.')
    if filename.lower().endswith('.pdf'):
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted and not reader.decrypt(''):
                raise ValueError('Password-protected PDF: upload an unlocked copy.')
            pages = [{'page': i, 'text': (p.extract_text() or '').strip()}
                     for i, p in enumerate(reader.pages, 1)]
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError('Unable to read this PDF. Try exporting it again.') from exc
    elif filename.lower().endswith('.txt'):
        pages = [{'page': 1, 'text': data.decode('utf-8-sig', errors='replace').strip()}]
    else:
        raise ValueError('Upload a PDF or UTF-8 text file.')
    if not any(p['text'] for p in pages):
        raise ValueError('No extractable text found. This may be a scanned PDF; run OCR and upload the searchable PDF.')
    return pages

def analyze(pages: list[dict], query: str = '') -> dict:
    words = set(re.findall(r'\w+', query.lower())) - {'the','a','of','in','and','is','what','for','to'}
    rows = []
    for page in pages:
        for part in re.split(r'\n\s*\n|(?<=[.!?])\s+', page['text']):
            part = part.strip()
            if not part:
                continue
            # Bound excerpts without silently losing long text blocks.
            for start in range(0, len(part), 1200):
                text = part[start:start+1200]
                score = len(words & set(re.findall(r'\w+', text.lower())))
                rows.append({'page': page['page'], 'text': text, 'score': score})
    matches = sorted((r for r in rows if not words or r['score']), key=lambda r: -r['score'])
    numeric = [r for r in rows if re.search(r'\d', r['text'])]
    return {'pages': len(pages), 'characters': sum(len(p['text']) for p in pages),
            'empty_pages': [p['page'] for p in pages if not p['text']],
            'matches': matches[:20], 'numeric_passages': numeric[:30],
            'match_count': len(matches), 'numeric_count': len(numeric)}
