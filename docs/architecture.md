# Trustworthy Financial Research System: Architecture Plan

*PS1: Trustworthy Multi-Agent Financial Research and Decision Support (AIML). Solo build on one CPU-only Windows laptop (32GB RAM), with Cursor writing the code, local Ollama models, target build time about 7 hours.*

## 1. Goal and design philosophy

Build a system that helps a financial analyst evaluate a listed company, where **verification quality** matters more than the number of agents. The system retrieves evidence, computes metrics in code, checks every claim against sources, flags problems, and hands a reviewable report to a human who approves, rejects or corrects it.

Core principles:

- **Agents are plain Python modules** chained together. No framework to learn.
- **Numbers are always computed by pandas**, never by the LLM. The LLM only explains them.
- **Every claim is structured JSON**, and that schema is the contract between modules.
- **Unsupported claims are never silently dropped.** They show a visible "no evidence found" label.
- **The system never executes transactions or makes decisions.** The human analyst has the final say.

## 2. Pipeline overview

```
 Documents (synthetic + 1 real EDGAR filing)     Structured financials (CSV/JSON)
            |  chunk + metadata                           |
            v                                             v
   1. RETRIEVER (MiniLM + FAISS)  <-- query --   2. ANALYST (pandas ratios/trends;
      passages with source + date                    LLM writes narrative only)
            |                                             |
            |                                   draft report = list of CLAIMS
            v                                             v
   3. VERIFIER (core piece)
        a) Textual: does the cited passage support the claim?
        b) Numeric: recompute the figure and compare
        c) Conflict: same metric + period, two sources disagree -> flag both
        d) No source -> "no evidence found"
            |
            v
   4. RISK AGENT: stale sources, conflicts, evidence gaps, overall confidence
            |
            v
   5. REPORT BUILDER: findings, evidence, risks, limitations, unresolved questions
            |
            v
   6. STREAMLIT REVIEW UI: evidence panel, flagged-claims table, approve / reject / correct
            |
            v
   SQLite / JSON audit log
```

## 3. The claim schema (the contract between modules)

Every stage passes claims in this shape. Define it first and give it to Cursor as a rules file.

```json
{
  "claim_id": "c1",
  "claim": "Revenue grew 18% in FY2024",
  "metric": "revenue_growth",
  "period": "FY2024",
  "value": 0.18,
  "source_id": "press_release_2024",
  "page_section": "p.1",
  "date": "2025-02-10",
  "status": "unverified"
}
```

The verifier fills in `status`, an `evidence` list and a `reason`. Allowed statuses:

| Status | Meaning |
|---|---|
| supported | Source and recalculation back the claim |
| contradicted | A source or the math disagrees with the claim |
| conflict | Two sources disagree on the same metric and period; both are flagged |
| unsupported | No evidence found (never labelled "false") |
| outdated | Source date is too old relative to the report period |

The `metric` and `period` fields make conflict detection simple: compare claims sharing the same `(metric, period)` pair.

## 4. Component details

Each module is a specialized agent with a role name and a one-line contract. The README repeats this table, and the UI shows each claim's handoff trail (which agent set which status), so the workflow is visible.

| Agent | Module | Input to output |
|---|---|---|
| Research Agent | `retriever.py` | Query to passages with source and date |
| Analyst Agent | `analyst.py`, `ratios.py` | Financial data to claims with numbers |
| Verifier Agent | `verifier.py` | Claims and passages to claims with status, evidence and reason |
| Risk Agent | `risk.py` | Verified claims and ratios to risk flags and confidence |
| Report Agent | `report.py` | Claims and flags to a report for human review |

### 4.1 Retriever

- Chunk documents with overlap; store text, source_id, page/section and publication date with each chunk.
- Embeddings: `sentence-transformers` with `all-MiniLM-L6-v2`.
- Index: `faiss-cpu`.
- Include one real SEC EDGAR 10-K, using only its Item 1A Risk Factors copied into a text file in data/docs/ (no parsing needed, works offline). Use `pypdf` if it is a PDF.
- Returns top passages with similarity scores and metadata.

**Document intake:** the analyst can upload a filing or earnings report (PDF or text) in the UI and enter its source name and publication date. The file is then chunked and indexed on the fly. The one real EDGAR 10-K (Item 1A text) is loaded the same way.

### 4.2 Analyst

- `ratios.py` computes revenue growth, profit margin, debt-to-equity, and similar metrics (3 to 4 metrics total) across periods with pandas.
- Produces claims with numbers attached, so the LLM cannot invent figures.
- The LLM writes the narrative text from numbers already computed.

### 4.3 Verifier (the core piece)

1. **Numeric check:** recompute the figure from structured data and compare with the claimed value within ±0.5 percentage points for growth rates (written in .cursor/rules so Cursor does not invent a different tolerance).
2. **Textual check:** embedding similarity between claim and cited passage as a pre-filter, then an LLM yes/no judgment as a second opinion. The LLM is never the only check.
3. **Conflict detection:** group claims by `(metric, period)`; if values disagree, flag both rather than picking one.
4. **Unsupported detection:** no `source_id`, or retrieval finds nothing above the similarity threshold.
5. **Staleness:** compare source date with the report period and set the outdated status.

### 4.4 Risk agent

Covers three kinds of risk with deterministic rules rather than an LLM opinion, and combines them into an overall confidence indication.

- **Data-quality risks:** stale sources, conflicts and evidence gaps from the verifier.
- **Financial risks:** rule-based flags computed from the ratios, such as rising debt-to-equity or shrinking margins, each with the numbers behind it.
- **Business risks:** risk-factor passages retrieved from the real EDGAR 10-K, each shown with its source and date.

### 4.5 Report builder

Produces the report with findings, evidence, risks, limitations and unresolved questions. Every report carries the banner: **"AI-generated, requires human review."**

It also adds a clearly labelled **Considerations for the analyst (non-binding)** section, built only from verified findings and risk flags. The analyst can edit or delete each item, and the system never acts on them.

### 4.6 Streamlit review UI

The UI is Streamlit: one Python codebase and no separate backend. React is an optional later upgrade, only if everything else is finished.

- Evidence panel to inspect the source passage behind each claim.
- Flagged-claims table with status and reason.
- Approve, reject or correct actions, plus free-text feedback.
- Every analyst action written to the audit log (SQLite or JSON).

**Document upload:** a file uploader for filings and reports, with fields for source name and publication date, feeding the retriever.

## 5. Where the LLM is used and where it is not

| Task | Done by | ML involved? |
|---|---|---|
| Ratios, growth, trends | pandas | No |
| Finding relevant passages | Embeddings + FAISS | Yes (pretrained, no training) |
| Writing the narrative | LLM from computed numbers | Yes (pretrained) |
| "Does this passage support this claim?" | LLM yes/no after embedding pre-filter | Yes (pretrained) |
| Conflict, staleness, unsupported detection | Plain Python rules | No |
| Final decision | Human analyst | n/a |

No model is trained. The contribution is connecting retrieval, calculation and verification into one trustworthy workflow. Most verification is deterministic code, which makes it reliable and demo-safe.

## 6. LLM and model strategy

Gemini API gave problems, so the plan uses local Ollama models.

**Hardware:** CPU only, 32GB RAM. Small models are fast; 7 to 8B models run at a few tokens per second; 14B and above are too slow.

**Models to download:**

```
ollama pull qwen3:4b
ollama pull qwen3:8b
```

You already have `gemma4:e2b`. If a tag is not found, run `ollama list` or check the Ollama library for the nearest 4B and 8B instruct models.

**Default model:** `qwen3:4b`. It is faster and uses less memory, and the yes/no check is a small job. The 8B is the fallback: use it only for the verifier's yes/no step if the 4B misses eval cases that the 8B gets right, and keep the 4B for narrative text. Switching is a one-line change of the model name in `llm.py`.

**Choosing the model with a mini eval:**

- Write about 20 test cases: a claim plus a passage, labelled supported, contradicted or unrelated. Include the three planted errors and near-miss cases (for example, the right metric but the wrong year), because small models fail there.
- Run each candidate (`gemma4:e2b`, `qwen3:4b`, `qwen3:8b`) through the same yes/no prompt.
- Keep the smallest model that gets all cases right. This gives a number for the write-up.

**Similarity threshold:** tune the embedding pre-filter threshold on the same eval set rather than guessing. Twenty cases is a small set, so treat the result as a rough setting.

**Ollama settings:** temperature 0, JSON output mode, short prompts, and a cap on output tokens (a yes/no with a one-line reason needs very few). For Qwen3, disable thinking mode for the yes/no check (`/no_think` in the prompt or `think: false` in the API call), because long reasoning is very slow on CPU.

**The embedding model stays as is.** MiniLM runs separately from the LLM.

### `llm.py`: one wrapper, two backends

A single function `ask(prompt) -> str` with the model name as a config value.

| Backend | Use |
|---|---|
| `ollama` | Live LLM calls on your laptop |
| `cache` | Replays saved outputs for Acme Corp so the demo is fast and repeatable |

The cache is not for hosting. A 4B or 8B model on CPU is slow, so replaying saved outputs keeps the Acme Corp demo fast and repeatable.

## 7. Demo data and planted errors

Synthetic company **Acme Corp** with 3 years of financials. Three deliberate issues:

| # | Issue | How it is caught |
|---|---|---|
| 1 | **Conflict:** press release says revenue grew 18%, filing data gives 11% | Numeric recomputation gives 11%; mismatch beyond ±0.5 percentage points flags both sources |
| 2 | **Unsupported:** claim "debt levels are declining" with no source | No `source_id` or no retrieval hit above threshold, shown as "no evidence found" |
| 3 | **Outdated:** one source dated two years old | Verifier sets the outdated status from source date versus report period; the risk agent only lists it in the risk section |

**Demo flow:** run the query, the system catches all three, the analyst corrects in the UI, then approves the report.

The cross-source conflict (18% vs 11%) is a stronger demo than a simple wrong-arithmetic claim, because it shows reasoning across sources.

### Evaluation numbers for the write-up

Report three results: the model comparison from the mini eval, the similarity threshold chosen, and how many injected errors the verifier caught. Add a few extra synthetic errors beyond the three planted ones; `test_verifier.py` produces the counts.

| Error type | Injected | Caught |
|---|---|---|
| Cross-source conflict | TBD | TBD |
| Unsupported claim | TBD | TBD |
| Outdated source | TBD | TBD |
| Wrong year (near-miss) | TBD | TBD |
| Wrong arithmetic | TBD | TBD |

## 8. Running locally

The system runs only on your laptop and is not deployed online, so there is no hosting, Groq or Hugging Face setup.

- **What runs:** Ollama, retrieval, analysis, verification, risk checks and the Streamlit review UI, all on one laptop.
- **Demo:** start the app with `streamlit run app.py` and present from the laptop. The cache keeps Acme Corp results fast.
- **README:** state that the system runs locally and uses cached narratives for the Acme demo.

### Running offline on the laptop

The whole system runs with no internet once the models are downloaded.

- **Ollama:** once `qwen3:4b` and `qwen3:8b` are pulled, they run with no internet. The model files sit on your disk, and Ollama serves them at `localhost:11434`.
- **Embedding model:** MiniLM downloads from Hugging Face the first time you use it, then it is cached locally. Run the app once with internet, then set the environment variable `HF_HUB_OFFLINE=1` so it never tries to reach out.
- **Everything else:** pandas, FAISS, SQLite and Streamlit are all local, so the whole pipeline works offline, including live LLM calls.

Recommended demo setup: run live from the laptop with Ollama, and test once with Wi-Fi turned off.

## 9. Project structure

```
acme-research/
├── .cursor/rules/              # claim schema + project rules Cursor must follow
├── data/
│   ├── financials.csv          # structured numbers
│   ├── docs/                   # synthetic filings, press release
│   └── cache/                  # cached LLM outputs for the demo
├── src/
│   ├── schema.py               # Claim dataclass (written first)
│   ├── retriever.py            # chunking, embeddings, FAISS search
│   ├── ratios.py               # pandas ratio calculator
│   ├── analyst.py              # builds claims from the numbers
│   ├── verifier.py             # numeric, textual, conflict, staleness checks
│   ├── risk.py                 # risk flags and confidence
│   ├── report.py               # report builder
│   ├── llm.py                  # one wrapper, swap providers here
│   └── audit.py                # logging analyst actions
├── tests/
│   └── test_verifier.py        # asserts the 3 planted errors get flagged
├── eval/
│   └── mini_eval.py            # model comparison script
├── app.py                      # Streamlit UI
├── run_pipeline.py             # runs everything end to end
├── requirements.txt
└── README.md                   # includes NIST mapping
```

## 10. Build workflow with Cursor

One person builds on one laptop, with Cursor writing the code module by module. Your job is to keep Cursor on the contract and check each step.

1. Create the repo and add `schema.py` plus a `.cursor/rules` file stating the claim schema, "pandas computes all numbers" and "no trading or execution functionality".
2. Ask Cursor for one module at a time, in the order in section 11, naming each module's input and output claims.
3. Run the module's test before moving on, and keep `test_verifier.py` asserting that the 3 planted errors are flagged.
4. Read what Cursor wrote against the schema before integrating modules.

## 11. Build order (one laptop, roughly 7 hours)

| Step | Build | Check before moving on |
|---|---|---|
| 1 | Repo, venv, `.cursor/rules`, `schema.py`, Acme data with the 3 planted errors, real 10-K Item 1A text | Data loads and matches the schema |
| 2 | `llm.py` (ollama and cache) and `mini_eval.py` | Model and similarity threshold chosen from the eval score |
| 3 | `retriever.py` | A query returns the right passage with source and date |
| 4 | `ratios.py` and `analyst.py` | Acme revenue growth computes to 11% |
| 5 | `verifier.py` and `test_verifier.py` | All 3 planted errors and the extra synthetic errors are flagged, with counts recorded |
| 6 | `risk.py`, `report.py`, `audit.py` | Report shows flags, financial and business risks, non-binding considerations and the review banner |
| 7 | `app.py` and `run_pipeline.py` | An uploaded filing is indexed, a full query runs end to end and analyst actions are logged |
| 8 | Cache outputs, offline test, rehearse | Works with Wi-Fi off |

Build and test one module at a time; do not start the next until its check passes.

## 12. Responsible AI and NIST mapping

| Feature | NIST AI RMF function |
|---|---|
| Human approve / reject / correct | Govern, Manage |
| Audit log of every analyst action | Govern, Measure |
| Uncertainty, limitations and unresolved questions in the report | Map |
| Staleness, conflict and unsupported-claim flags | Measure |
| "AI-generated, requires human review" banner | Govern |
| No trading or execution functionality at all | Manage |

Refine this table against the NIST Generative AI Profile in the final write-up.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Crowded problem statement (many teams pick PS1) | Differentiate on verification quality and a clear demo of caught errors |
| Small local model is weak at nuanced judgment | Deterministic checks do the real work; LLM yes/no is only a second opinion |
| LLM slowness or failure during live demo | Cache outputs for Acme Corp; isolate LLM calls behind `llm.py` |
| Overscoping | One company, 3 to 4 metrics, 3 planted errors done well |
| Cursor-generated code drifting from the schema | Lock `schema.py` first and give it to Cursor as a rules file; run its test after each module |
| Laptop or Ollama fails on demo day | `cache` backend replays saved outputs; test the demo offline beforehand |

## 14. Open items and next step

- Confirm which model wins the mini eval.
- Choose the real company for the EDGAR 10-K and record its filing date.
- **Next:** create the repo, add `schema.py` and `.cursor/rules`, then build module by module in Cursor following section 11.