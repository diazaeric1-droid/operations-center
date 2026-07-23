"""FastAPI service over the operator-note RAG engine.

A thin HTTP layer on top of ``rag.engine.NoteSearchEngine`` — the same engine the
Streamlit page and the MCP server wrap. The heavy embedding model is loaded once
as a lazy singleton (first request pays the cost; everything after is fast).

Run:
    .venv/bin/python -m uvicorn api.main:app --port 8000

Endpoints:
    GET  /health           -> index size + DB reachability
    POST /search           -> ranked operator notes (no synthesis)
    POST /answer           -> grounded answer + the cited source notes

The frontend (web/) calls /answer; CORS is opened for the Next.js dev origin.
"""
from __future__ import annotations

from dataclasses import asdict
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import store
from rag.engine import NoteSearchEngine

# Cause filter values the corpus understands. Mirrored in the frontend <select>.
VALID_CAUSES = {
    "artificial_lift",
    "surface_facility",
    "power",
    "gathering_thirdparty",
    "wellbore",
    "planned",
    "weather",
    "reservoir",
}

app = FastAPI(
    title="Operator Notes RAG API",
    description="Semantic search + grounded synthesis over the operator-note log.",
    version="1.0.0",
)

# The Next.js dev server origin. Add your deployed origin here in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- lazy singleton engine ---------------------------------------------------
# Constructing the engine loads a local ONNX embedding model (~seconds). Build it
# once, behind a lock so two concurrent first-requests can't race.
_engine: Optional[NoteSearchEngine] = None
_engine_lock = Lock()


def get_engine() -> NoteSearchEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = NoteSearchEngine()
    return _engine


# --- request / response models ----------------------------------------------
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language query.")
    top_k: int = Field(6, ge=1, le=12, description="How many notes to retrieve.")
    cause: Optional[str] = Field(
        None, description="Optional cause filter (one of the 8 valid causes).")


def _validate_cause(cause: Optional[str]) -> Optional[str]:
    """Normalize an optional cause filter; reject unknown values."""
    if cause is None or cause == "":
        return None
    if cause not in VALID_CAUSES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid cause '{cause}'. Valid: {sorted(VALID_CAUSES)}",
        )
    return cause


# --- endpoints ---------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Index size + DB reachability. Cheap; safe to poll."""
    ok, detail = store.ping()
    indexed = 0
    try:
        indexed = get_engine().index_size()
    except Exception:  # noqa: BLE001 — health must never raise
        pass
    return {"ok": ok, "indexed": indexed, "db": detail}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    """Semantic search — ranked operator notes, no synthesis."""
    cause = _validate_cause(req.cause)
    hits = get_engine().retrieve(req.query, top_k=req.top_k, cause=cause)
    return {"results": [asdict(h) for h in hits]}


@app.post("/answer")
def answer(req: SearchRequest) -> dict:
    """Retrieve + synthesize a cited answer (extractive; no API key)."""
    cause = _validate_cause(req.cause)
    ans = get_engine().answer(
        req.query, top_k=req.top_k, cause=cause, anthropic_key=None)
    return {
        "answer": ans.text,
        "used_llm": ans.used_llm,
        "sources": [asdict(s) for s in ans.sources],
    }
