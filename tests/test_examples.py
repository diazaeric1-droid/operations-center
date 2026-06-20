"""Keep the prompt-vs-graph teaching example working.

Forces the no-key stub so it's deterministic, and proves the contrast the example
is meant to show: one prompt overshoots the constraint; the graph loops until it
fits.
"""
from __future__ import annotations

import pytest

try:
    from langgraph_rag.graph import deps_available
    _LG = deps_available()[0]
except Exception:  # noqa: BLE001
    _LG = False

needs_lg = pytest.mark.skipif(not _LG, reason="langgraph not installed")


@needs_lg
def test_graph_loops_until_constraint_met(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)   # force the stub
    from examples.prompt_vs_graph import build, run_prompt, LIMIT

    app = build()
    final = app.invoke({"instruction": "In one short sentence: why?", "draft": "",
                        "rounds": 0, "provider": "stub", "trace": []})
    assert len(final["draft"]) <= LIMIT     # the loop satisfied the constraint
    assert final["rounds"] >= 1             # …and it actually had to loop

    out, ok = run_prompt("stub")            # the one-shot version overshoots
    assert not ok and len(out) > LIMIT


@needs_lg
def test_model_routing_runs_each_step_on_its_provider():
    """The routing agent: classify+draft on the 'cheap' model, polish on the
    'premium' model — forced to stub for a deterministic, key-free test."""
    from examples.model_routing import run
    final = run("why might a well make more water?", cheap="stub", premium="stub")
    assert final["category"] and final["draft"] and final["answer"]
    # the trace records which provider handled each step (the routing)
    assert any(t.startswith("classify [stub") for t in final["trace"])
    assert any(t.startswith("draft    [stub") for t in final["trace"])
    assert any(t.startswith("polish   [stub") for t in final["trace"])


def test_eval_harness_passes_good_answers():
    from examples.eval_harness import run_eval
    results = run_eval(["stub"])               # stub gives domain-correct answers
    assert all(r["passed"] for r in results["stub"])


def test_eval_harness_catches_domain_hallucination():
    """The must_avoid check flags the groundwater answer a generalist model gave."""
    from examples.eval_harness import run_eval

    def groundwater(provider, q):              # the real failure, reproduced
        return ("The well makes more water because rainfall recharges the aquifer "
                "and raises the water table.")
    rows = run_eval(["x"], answer_fn=groundwater)["x"]
    water = next(r for r in rows if r["id"] == "water_increase")
    assert not water["passed"]                 # the eval FAILS it automatically
    assert water["violations"]                 # …on banned domain terms


@needs_lg
def test_supervisor_dispatches_to_the_right_specialist():
    """The supervisor routes each question to the matching specialist agent."""
    from examples.supervisor import run
    cases = {
        "my ESP keeps tripping on underload": "production",
        "why is my well making more water and losing pressure": "reservoir",
        "the separator keeps tripping on high level": "facilities",
    }
    for question, expected in cases.items():
        final = run(question, supervisor_model="stub", expert_model="stub")
        assert final["route"] == expected, (question, final["route"])
        assert final["answer"]
