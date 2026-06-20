"""Tests for the LangGraph agentic-RAG flow.

Two tiers:
  * grading logic — pure Python, always runs.
  * graph control flow — needs langgraph; driven by a FAKE engine with scripted
    retrieval scores, so the cycle / conditional routing is verified WITHOUT a
    vector DB or any API key.
"""
from __future__ import annotations

import pytest

from langgraph_rag import grading
from langgraph_rag.graph import deps_available

needs_lg = pytest.mark.skipif(not deps_available()[0],
                              reason="langgraph not installed")


# --- grading logic (pure, always run) -----------------------------------------
def test_grade_relevance_threshold():
    strong = [{"score": 0.80, "note": "x"}]
    weak = [{"score": 0.40, "note": "x"}]
    assert grading.grade_relevance("q", strong) is True
    assert grading.grade_relevance("q", weak) is False
    assert grading.grade_relevance("q", []) is False


def test_rewrite_expands_domain_terms():
    out = grading.rewrite_query("lost power at the pad", [{"score": 0.3}])
    assert out != "lost power at the pad"          # always changes
    assert "substation" in out or "breaker" in out  # power-domain expansion


def test_rewrite_falls_back_when_no_terms_match():
    out = grading.rewrite_query("zzz qqq", [{"score": 0.3}])
    assert "downtime" in out                         # generic ops context appended


# --- graph control flow (needs langgraph; fake engine, no DB) ------------------
class _FakeEngine:
    """Returns notes whose top score follows a script across successive calls."""
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def retrieve(self, query, top_k=6, cause=None):
        from rag.engine import RetrievedNote
        s = self.scores[min(self.calls, len(self.scores) - 1)]
        self.calls += 1
        return [RetrievedNote(score=s, note="scripted", well_id="well_x",
                              cause="power", start_date="2024-01-01",
                              duration_days=3, deferred_bbl=100, source="synthetic")]

    def synthesize(self, query, hits, anthropic_key=None, model="x"):
        from rag.engine import Answer
        return Answer(f"answer from {len(hits)} notes", False, list(hits))


@needs_lg
def test_graph_one_shot_when_first_retrieval_is_strong():
    from langgraph_rag.graph import run
    eng = _FakeEngine([0.85])                      # strong immediately
    final = run("q", engine=eng, max_iterations=2)
    assert eng.calls == 1                           # retrieved once, no rewrite
    assert not any("rewrite" in t for t in final["trace"])
    assert final["answer"].startswith("answer from")


@needs_lg
def test_graph_loops_then_succeeds():
    from langgraph_rag.graph import run
    eng = _FakeEngine([0.40, 0.85])                # weak, then strong after rewrite
    final = run("q", engine=eng, max_iterations=2)
    assert eng.calls == 2                           # one rewrite -> second retrieve
    assert sum(t.startswith("rewrite") for t in final["trace"]) == 1
    assert any(t.startswith("grade: relevant") for t in final["trace"])
    assert final["answer"]


@needs_lg
def test_graph_respects_max_iterations():
    from langgraph_rag.graph import run
    eng = _FakeEngine([0.40])                       # never strong enough
    final = run("q", engine=eng, max_iterations=2)
    # round 0 + 2 rewrites = 3 retrievals, then forced to generate
    assert eng.calls == 3
    assert sum(t.startswith("rewrite") for t in final["trace"]) == 2
    assert final["answer"]                          # still answers from best effort
