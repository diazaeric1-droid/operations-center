"""Agentic, self-correcting RAG over the operator-note index — built with LangGraph.

The Note Search page does ONE-SHOT RAG: embed → retrieve → answer. That's fine
when the first retrieval is good, but it can't recover from a weak query. This
package wraps the *same* ``rag.engine.NoteSearchEngine`` backend in a LangGraph
state machine that grades its own retrieval and loops to fix it:

        ┌─────────────────────────────────────────────┐
        │                                             ▼
   START ─► retrieve ─► grade ──(weak)──► rewrite ──┘
                          │
                       (good / max iters)
                          ▼
                       generate ─► END

What this exercises (the LangGraph skills):
  * a typed shared **State** that flows through every node,
  * **conditional edges** (grade routes to rewrite vs generate),
  * a **cycle** (rewrite → retrieve), with a max-iteration guard,
  * **checkpointing** (MemorySaver; swap PostgresSaver in prod),
  * a **trace** of every step for observability.

Grading and rewriting are deterministic by default (retrieval-score threshold +
domain-synonym expansion) so the whole graph runs with NO API key — and use
Claude when a key is supplied. LangGraph is an OPTIONAL extra
(``requirements-langgraph.txt``); ``graph.py`` imports it lazily, so this package
imports cleanly without it and nothing in the deployed app depends on it.

NB: the package is ``langgraph_rag`` (not ``langgraph``) on purpose — a top-level
``langgraph`` dir would shadow the installed library on sys.path.
"""
from __future__ import annotations

__all__ = ["graph"]
