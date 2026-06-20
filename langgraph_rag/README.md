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

## Second flow — `approval_agent.py` (the patterns reference)

A heavily-commented intervention-approval agent built to demonstrate the four
LangGraph capabilities an AI-engineer role expects, each isolated and runnable:

| Capability | Where in the agent |
|---|---|
| **Branch** | `route_after_assess` → one of four next nodes on state |
| **Cycle** | `gather → assess` loop to enrich missing inputs (capped) |
| **Durable persistence** | compiled with `SqliteSaver` — paused state is written to disk |
| **Human-in-the-loop** | `human_review` calls `interrupt()`; resume with `Command(resume=…)` |

```bash
python -m langgraph_rag.approval_agent
```

The demo runs all four: auto-approve, auto-reject, the gather→re-assess cycle,
and — the headline — a costly job that **pauses for a human, survives a simulated
process restart** (a fresh graph rebuilt from the same sqlite file is still
paused), then resumes from disk with the reviewer's decision. That "wait days,
survive a restart, resume" property is the thing prompt chains fundamentally
can't do, and the one to be able to speak to in an interview.

## Running on any model — `providers.py` (provider-agnostic)

The agent's logic doesn't care which model answers, so the same graph runs on
Claude, Gemini, Groq, GitHub Models (GPT-4o), OpenRouter, or OpenAI. One `chat()`
abstracts them: five speak the OpenAI Chat Completions protocol (so a single
OpenAI client with a different `base_url` covers them), and Claude uses the
Anthropic SDK.

```bash
pip install -r requirements-providers.txt        # the OpenAI universal client
python -m langgraph_rag.providers                 # which providers have a key set?

# get a FREE key, export it, and run the demo agent on that model:
export GEMINI_API_KEY=...        # aistudio.google.com  (free, no card)
python examples/prompt_vs_graph.py graph gemini
export GROQ_API_KEY=...          # console.groq.com     (free, no card — Llama)
python examples/prompt_vs_graph.py graph groq
```

| provider | env var | free key |
|---|---|---|
| `claude` | `ANTHROPIC_API_KEY` | console.anthropic.com ($5 trial) |
| `gemini` | `GEMINI_API_KEY` | aistudio.google.com (free) |
| `groq` | `GROQ_API_KEY` | console.groq.com (free) |
| `github` | `GITHUB_TOKEN` | a GitHub PAT with the Models scope (free) |
| `openrouter` | `OPENROUTER_API_KEY` | openrouter.ai/keys (free open models) |
| `openai` | `OPENAI_API_KEY` | platform.openai.com (paid) |

The résumé line this earns: *provider-agnostic LLM agents across Claude, GPT,
Gemini, and open models — with per-task model routing* (e.g. a cheap fast model
for the grade step, a premium model for the final answer).

## Design notes

- The package is `langgraph_rag`, **not** `langgraph` — a top-level `langgraph/`
  dir would shadow the installed library on `sys.path`.
- `graph.py` imports LangGraph lazily; the package imports cleanly without the
  extra, and nothing in the deployed app depends on it (the UI toggle disables
  itself when it's absent).
- Nodes close over an injected `engine`, so the same graph runs against the real
  `NoteSearchEngine` or a fake one in tests.
