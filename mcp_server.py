"""MCP server — expose operator-note retrieval as Model Context Protocol tools.

The same pgvector index the Streamlit page queries, surfaced over MCP so any MCP
client (Claude Desktop, an agent runtime, an IDE) can search the operator-note
log as a tool. This is the "multi-step tool-use / MCP" capability the AI-engineer
JDs ask for, pointed at a real retrieval backend rather than a toy.

Run (stdio transport — what Claude Desktop launches):
    pip install -r requirements-rag.txt -r requirements-mcp.txt
    python mcp_server.py            # or: mcp run mcp_server.py

Register with Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "operator-notes": {
          "command": "/abs/path/.venv/bin/python",
          "args": ["/abs/path/operations-center/mcp_server.py"],
          "env": { "OPS_PG_DSN": "postgresql://ops:ops@localhost:5433/opsrag" }
        }
      }
    }

Tools exposed:
    search_operator_notes(query, top_k, cause)  -> ranked notes (structured)
    answer_from_notes(query, top_k, cause)       -> grounded extractive answer
    index_status()                               -> vectors indexed + DB target
"""
from __future__ import annotations

import json
import os

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as e:  # noqa: BLE001
    raise SystemExit(
        "MCP SDK not installed. Run:\n"
        "    pip install -r requirements-rag.txt -r requirements-mcp.txt") from e

from rag.engine import NoteSearchEngine
from rag import store

mcp = FastMCP("operator-notes")

# One engine for the server's lifetime (loads the embedding model once).
_engine: NoteSearchEngine | None = None


def _eng() -> NoteSearchEngine:
    global _engine
    if _engine is None:
        _engine = NoteSearchEngine()
    return _engine


@mcp.tool()
def search_operator_notes(query: str, top_k: int = 6,
                          cause: str | None = None) -> str:
    """Semantic search over the operator-note / downtime-event log.

    Retrieves the notes most similar in MEANING to `query` from the pgvector
    index (not keyword match). Optionally restrict to one reason `cause`
    (artificial_lift, surface_facility, power, gathering_thirdparty, wellbore,
    planned, weather, reservoir). Returns JSON: a list of
    {score, well_id, cause, start_date, duration_days, deferred_bbl, note}.
    """
    hits = _eng().retrieve(query, top_k=top_k, cause=cause)
    return json.dumps([{
        "score": h.score, "well_id": h.well_id, "cause": h.cause,
        "start_date": h.start_date, "duration_days": h.duration_days,
        "deferred_bbl": h.deferred_bbl, "note": h.note,
    } for h in hits], indent=2)


@mcp.tool()
def answer_from_notes(query: str, top_k: int = 6,
                      cause: str | None = None) -> str:
    """Answer a question grounded ONLY in retrieved operator notes.

    Retrieves the most relevant notes and returns a concise answer plus the
    cited source notes. Deterministic/extractive (no external LLM call) so it's
    safe for an agent to call repeatedly; the calling model does the reasoning
    over the returned, citation-tagged evidence.
    """
    ans = _eng().answer(query, top_k=top_k, cause=cause, anthropic_key=None)
    return json.dumps({
        "answer": ans.text,
        "sources": [{
            "n": i, "well_id": h.well_id, "cause": h.cause,
            "start_date": h.start_date, "duration_days": h.duration_days,
            "deferred_bbl": h.deferred_bbl, "note": h.note,
        } for i, h in enumerate(ans.sources, 1)],
    }, indent=2)


@mcp.tool()
def index_status() -> str:
    """How many operator notes are indexed, and which vector DB is targeted."""
    ok, detail = store.ping()
    size = _eng().index_size() if ok else 0
    return json.dumps({
        "vector_db_reachable": ok, "target": detail,
        "vectors_indexed": size,
        "dsn_source": "OPS_PG_DSN" if os.environ.get("OPS_PG_DSN")
        else "default (local docker)",
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
