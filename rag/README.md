# Note Search (RAG) — semantic search over operator notes

Retrieval-augmented search over the operator downtime/curtailment notes, beside
the deterministic deferment pipeline. The keyword reason-code classifier answers
*"what bucket is this note?"*; this answers the questions a keyword match can't:

- *"find shutdowns that read like a slow ESP failure, not an instant trip"*
- *"what freeze-offs lasted more than three days last winter?"*
- *"have we seen this separator emulsion upset before?"*

## How it works

```
corpus.py   operator notes  → documents + metadata      (pure Python, no deps)
store.py    documents        → pgvector index            (Postgres + pgvector)
engine.py   query            → top-k retrieve + cited synthesis
                               (embeddings: fastembed/ONNX, local & free;
                                synthesis: Claude BYOK, extractive fallback)
```

Surfaces: the **Note Search (RAG)** Streamlit page, and **`mcp_server.py`** which
exposes the same retriever as MCP tools for any MCP client.

## Run it locally (≈2 min)

```bash
pip install -r requirements-rag.txt          # llama-index, fastembed, pgvector
docker compose -f docker-compose.rag.yml up -d   # pgvector on :5433
streamlit run app.py                         # → Loss Accounting → Note Search (RAG)
#   click "Build index" once (~10s), then ask questions
```

No API key needed: embeddings are local (fastembed ONNX), and without an
Anthropic key the answer is a deterministic extractive rollup. Add a key in the
sidebar for a narrated, [n]-cited answer.

## MCP server

```bash
pip install -r requirements-rag.txt -r requirements-mcp.txt
python mcp_server.py        # stdio; register in claude_desktop_config.json (see file header)
```

Tools: `search_operator_notes`, `answer_from_notes`, `index_status`.

## Tests

```bash
pytest tests/test_rag.py    # corpus tests always run; DB roundtrip runs when pgvector is up
```

## Going to the cloud / Kubernetes

The same code points at AWS RDS pgvector by changing only `OPS_PG_DSN`.
See [`../k8s/README.md`](../k8s/README.md) for the Kubernetes topology and
[`../docs/AWS_DEPLOY.md`](../docs/AWS_DEPLOY.md) for the AWS runbook (and a frank
take on whether the cloud step is worth it — short version: only for the résumé
line, the feature itself runs fine on a laptop).

## Design notes

- **fastembed, not torch** — ONNX runtime keeps the install light, no GPU, no key.
- **Optional everywhere** — RAG deps live in `requirements-rag.txt`, separate from
  the core app; the page degrades to a readable "extended install required"
  message when they're absent, so Streamlit Cloud stays light.
- **Synthetic corpus** — `corpus.py` unions the shipped reason-coded event log
  with a seeded generator (~360 notes) so retrieval has enough volume to be
  meaningful; generation is reproducible (tested).
