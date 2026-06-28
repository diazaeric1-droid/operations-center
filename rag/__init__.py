"""Operations Center — semantic search (RAG) over operator notes.

A retrieval-augmented layer that sits *beside* the deterministic deferment
pipeline. The existing reason-code classifier answers "what bucket is this
note?"; this answers the questions a keyword classifier can't:

    "find shutdowns that read like a slow ESP failure, not an instant trip"
    "what freeze-offs lasted more than three days last winter?"
    "have we seen this separator emulsion upset before, and what fixed it?"

Architecture (each layer is independently swappable — that's the point):

    corpus.py   operator notes  -> documents + metadata   (pure Python, no deps)
    store.py    documents       -> pgvector index          (Postgres + pgvector)
    engine.py   query           -> retrieved notes + cited LLM synthesis
                                   (embeddings: fastembed/ONNX, local & free;
                                    synthesis: Anthropic Claude, BYOK)

Nothing here is imported by the deterministic core; the RAG view loads it
lazily and degrades to a readable "extended install required" message when the
optional deps (see requirements-rag.txt) or the vector DB are absent.
"""
from __future__ import annotations

__all__ = ["corpus", "store", "engine"]
