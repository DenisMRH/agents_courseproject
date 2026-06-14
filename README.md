# MSB lending support agent

Prototype support agent for the training case `Проект поддержки кредитования МСБ`.

## What is implemented

- LangGraph workflow: intent classification, safety check, RAG/tools, escalation gate, final self-check; sequential fallback is kept for environments without LangGraph.
- RAG over the 5 markdown regulations from `project_support_lending_msb_clean/Проект поддержки кредитования МСБ/data/documents`.
- SQLite tools over `clients.sqlite` in read-only mode: client profile, applications, loans and combined context.
- Escalation tickets in `runtime/support_tickets.sqlite`.
- Trace logs with RAG chunks and tool calls in `runtime/traces`.
- Streamlit demo UI in `app/streamlit_app.py`.
- E2E evaluation over `data/qa/qa.jsonl` with a generated report in `reports/eval_report.md`.

The QA file is used only for evaluation labels and is not indexed into RAG.

## Modes

The code supports two modes:

- Local mode: deterministic TF-IDF RAG and rule-based answer generation. This mode runs without network access and is used by scripts by default.
- Online mode: uses GigaChat if `langchain-gigachat` is installed and credentials are available in `.env`. The parser accepts both standard names such as `GIGACHAT_CREDENTIALS` and the local `Authorization_key` / `client_id` format. Guardrail-sensitive answers, calculations and client-specific transactional answers remain deterministic even in online mode.

## Install

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 scripts/ingest.py
python3 scripts/demo.py
python3 scripts/evaluate.py
streamlit run app/streamlit_app.py
```

For online LLM mode in demo:

```bash
python3 scripts/demo.py --online
```

## Runtime files

Generated files are placed under:

- `runtime/chroma` for the fallback index and optional Chroma collection;
- `runtime/traces` for request traces;
- `runtime/support_tickets.sqlite` for escalation tickets;
- `reports/eval_report.md` for evaluation output.

Source training data is not modified.


## Online smoke test

A safe online check that does not send training documents or client data is available:

```bash
.venv/bin/python scripts/online_smoke.py
```

Verified on 2026-06-15:

- GigaChat credentials are detected.
- GigaChat chat request returns a response.
- GigaChat embeddings return 1024-dimensional vectors.
- Chroma can store and query a synthetic document using GigaChat embeddings.

The full `scripts/demo.py --online` path sends retrieved regulation chunks and demo client context to GigaChat for non-sensitive generation paths. Use it only in an environment where that external data transfer is allowed.

## Last local verification

Run on 2026-06-15 in local mode:

- `.venv/bin/python scripts/ingest.py`: 5 documents, 289 chunks, required metadata present.
- `.venv/bin/python -m compileall src scripts app`: all modules compile.
- `.venv/bin/python scripts/demo.py`: 5 demo scenarios completed; sales and negative cases created tickets.
- `.venv/bin/python scripts/demo.py --online`: full online demo completed; GigaChat was available, while transactional and guardrail-sensitive answers used deterministic safeguards.
- `.venv/bin/python scripts/evaluate.py`: 180/180 cases passed with 100.0% outcome, escalation, source hit, tool calls and rejection/offtopic metrics.
- `.venv/bin/python scripts/online_smoke.py`: online GigaChat chat, embeddings and Chroma smoke passed on synthetic data.
- Streamlit UI verified with HTTP 200 at `http://localhost:8501`, then the test server was stopped.

Sensitive and generated local files are excluded via `.gitignore` (`.env`, `.venv`, caches, runtime indexes, traces and ticket DB).
