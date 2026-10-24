# INNOVISE — repaired prototype

## Start on Windows (Python 3.11 or 3.12 recommended)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The default **Uploaded document analysis** workspace reads PDFs and TXT files immediately. Upload a file, then optionally enter a topic. Results include page counts, relevant source excerpts, numeric passages, full extracted text, and downloadable evidence/JSON files. This workspace does not need Ollama or downloaded embedding weights.

## Important scope

Uploaded document results are extractive evidence, not independent verification or automatic financial table extraction. Keyword searches may miss synonyms. Scanned pages require external OCR; encrypted PDFs must be unlocked. A 25 MB limit applies. Uploads stay in the current browser session; reselect after a new session. The UI shows the full extracted text when ranked results are limited.

The separate **Acme synthetic demonstration** runs the original fixed financial dataset. Its query does not select a company. It requires MiniLM weights and, for uncached support judgments, local Ollama with `qwen3:4b`. Missing judgment responses fail closed as unsupported. Download MiniLM once while online before using HF_HUB_OFFLINE=1. Optional ONNX fallback dependencies are in requirements-onnx.txt.

```powershell
ollama pull qwen3:4b
$env:LLM_BACKEND="ollama"
python -m streamlit run app.py
```

## Verification

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests/test_regressions.py tests/test_schema.py tests/test_ratios.py tests/test_llm.py -q
```

These local regression tests cover PDF page preservation, blank/scanned PDF errors, query matching, numeric/source mismatches, zero denominators and Streamlit startup. Full legacy integration tests additionally require the embedding model. Old cached snapshots/evaluation results predate the stricter verifier and are not current accuracy measurements; regenerate them with the updated code and review changed expectations.

## Changes

- `src/documents.py`, `src/document_ui.py`: separate immediate PDF/TXT evidence workspace, citations, input errors and exports.
- `app.py`: workspace selection, upload reindexing, unique safe upload filenames, date/finite-number validation, decision invalidation after re-verification, approval identity and exported dispositions.
- `src/verifier.py`: source support required alongside arithmetic, percentage/year consistency checks, invalid date handling, fail-closed judge errors.
- `src/ratios.py`: input validation and undefined ratio omission.
- `src/analyst.py`, `run_pipeline.py`: deterministic summaries based on supported claims; regenerate after correction.
- `src/retriever.py`: true PDF page citations and empty extraction error.
- `src/risk.py`: exclude unrelated reference risk document from demo risks, relevance threshold, no high confidence for empty/unverified claims.
- `tests/test_regressions.py`: focused regression checks.

## Remaining limitations

This is a repaired prototype, not a guarantee of no bugs. General text entailment still depends on the configured local model; deterministic consistency checks cover common percent/year cases, not arbitrary prose. Multi-user durable report versioning, authenticated reviewer identity, structured financial ingestion, OCR and independent PDF fact verification remain future work. Do not expose the prototype publicly as a production service.
