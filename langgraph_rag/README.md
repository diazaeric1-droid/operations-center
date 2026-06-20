# Agentic RAG — a self-correcting LangGraph flow over Note Search

One-shot RAG (the Note Search page) embeds the query, retrieves once, and
answers. That's great when the first retrieval is good — and stuck when it
isn't. This package wraps the **same** `rag.engine.NoteSearchEngine` in a
**LangGraph** state machine that grades its own retrieval and loops to fix a weak
query before answering.

```
   START ─► retrieve ─► grade ──(weak)──► rewrite ──┐
                          │                          │  (cycle, max 2)
                       (good / max iters)            ▼
                          ▼                       retrieve …
                       generate ─► END
```

## What it demonstrates (the LangGraph skills)

| Concept | Where |
|---|---|
| Typed shared **State** | `AgenticRAGState` (TypedDict, with a `trace` reducer) |
| **Nodes** | `retrieve`, `grade`, `rewrite`, `generate` |
| **Conditional edges** | `route_after_grade` → rewrite vs generate |
| **Cycle** (self-correction) | `rewrite → retrieve`, capped by `max_iterations` |
| **Checkpointing** | `MemorySaver` (swap `PostgresSaver` in prod) |
| **Observability** | a `trace` of every step, surfaced in the UI |

## Grading & rewriting — works with no API key

- **Grade** (`grading.grade_relevance`): deterministic by default — the top
  retrieval cosine score must clear `RELEVANCE_THRESHOLD` (0.68, calibrated to
  the bge-small scale on this corpus). With an Anthropic key it asks Claude for a
  yes/no relevance judgment, falling back to the score rule on any error.
- **Rewrite** (`grading.rewrite_query`): deterministic by default — appends
  field-vocabulary expansions for any lay terms ("power" → "substation breaker
  transformer …"), which measurably lifts retrieval (e.g. 0.63 → 0.71). With a
  key, Claude rephrases the query.

So the whole agent runs offline; the LLM only sharpens it.

## Run it

```bash
pip install -r requirements-rag.txt -r requirements-langgraph.txt
docker compose -f docker-compose.rag.yml up -d        # pgvector
python -m langgraph_rag.graph "the machinery quit unexpectedly overnight"
```

Example (the loop firing):

```
retrieve(round 0): "the machinery quit unexpectedly overnight" → 6 notes, top score 0.63
grade: WEAK — will rewrite
rewrite → "the machinery quit … production downtime curtailment operator note"
retrieve(round 1): … → 6 notes, top score 0.71
grade: relevant
generate: extractive answer from 6 notes
```

In the app: **Note Search (RAG)** → tick **"Self-correcting (agentic)"** to run
the graph and see the trace.

## Tests

```bash
pytest tests/test_langgraph_rag.py    # grading logic always; graph flow needs langgraph
```

The graph tests drive a **fake engine** with scripted retrieval scores, so the
cycle / routing / max-iteration guard are verified without a vector DB or a key.

## Design notes

- The package is `langgraph_rag`, **not** `langgraph` — a top-level `langgraph/`
  dir would shadow the installed library on `sys.path`.
- `graph.py` imports LangGraph lazily; the package imports cleanly without the
  extra, and nothing in the deployed app depends on it (the UI toggle disables
  itself when it's absent).
- Nodes close over an injected `engine`, so the same graph runs against the real
  `NoteSearchEngine` or a fake one in tests.
