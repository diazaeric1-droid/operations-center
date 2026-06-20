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


# --- approval agent: branch / cycle / durable HITL ----------------------------
def test_route_after_assess_branches():
    """The router is pure — test every branch directly (no langgraph)."""
    from langgraph_rag.approval_agent import route_after_assess
    base = {"failure_risk": 0.5, "risked_value": 100.0, "intervention_cost": 10_000,
            "gather_attempts": 0}
    assert route_after_assess({**base, "failure_risk": 0.0}) == "gather"
    assert route_after_assess({**base, "risked_value": -1.0}) == "auto_reject"
    assert route_after_assess({**base, "intervention_cost": 10_000}) == "auto_approve"
    assert route_after_assess({**base, "intervention_cost": 90_000}) == "human_review"
    # cycle is capped: unknown risk but already gathered -> don't loop forever
    assert route_after_assess({**base, "failure_risk": 0.0,
                               "gather_attempts": 1}) != "gather"


@needs_lg
def test_agent_auto_paths():
    from langgraph_rag.approval_agent import build_approval_graph, _initial
    app = build_approval_graph()
    s = app.invoke(_initial("w", 1800, 18_000, 0.55),
                   {"configurable": {"thread_id": "a"}})
    assert s["outcome"] == "auto-approved"
    s = app.invoke(_initial("w", 90, 60_000, 0.20),
                   {"configurable": {"thread_id": "b"}})
    assert s["outcome"] == "rejected"


@needs_lg
def test_agent_cycle_enriches_then_proceeds():
    from langgraph_rag.approval_agent import build_approval_graph, _initial
    app = build_approval_graph()
    s = app.invoke(_initial("w", 1200, 15_000, 0.0),    # unknown risk
                   {"configurable": {"thread_id": "c"}})
    assert any(t.startswith("gather") for t in s["trace"])   # the cycle fired
    assert s["trace"].count("assess: ") == 0 or \
        sum(t.startswith("assess") for t in s["trace"]) == 2  # assessed twice
    assert s["outcome"]                                   # reached a decision


@needs_lg
def test_agent_human_in_the_loop_survives_restart(tmp_path):
    """The flagship test: pause for a human, 'restart', resume from disk."""
    from langgraph.types import Command
    from langgraph_rag.approval_agent import (
        build_approval_graph, sqlite_saver, _initial, is_interrupted)
    db = str(tmp_path / "approvals.sqlite")
    cfg = {"configurable": {"thread_id": "AFE-1"}}

    app1 = build_approval_graph(sqlite_saver(db))
    s1 = app1.invoke(_initial("w", 2100, 140_000, 0.62), cfg)   # costly -> human
    assert is_interrupted(app1, cfg)            # paused, awaiting a human
    assert not s1["outcome"]
    assert s1.get("__interrupt__")              # interrupt payload surfaced

    # brand-new graph object from the SAME db file == a process restart
    app2 = build_approval_graph(sqlite_saver(db))
    assert is_interrupted(app2, cfg)            # state was persisted to disk
    s2 = app2.invoke(Command(resume="approve"), cfg)
    assert s2["outcome"] == "approved by human"


@needs_lg
def test_agent_human_can_reject(tmp_path):
    from langgraph.types import Command
    from langgraph_rag.approval_agent import (
        build_approval_graph, sqlite_saver, _initial)
    db = str(tmp_path / "approvals.sqlite")
    cfg = {"configurable": {"thread_id": "AFE-2"}}
    app = build_approval_graph(sqlite_saver(db))
    app.invoke(_initial("w", 2100, 140_000, 0.62), cfg)
    s = app.invoke(Command(resume="reject"), cfg)
    assert s["outcome"] == "rejected by human"
