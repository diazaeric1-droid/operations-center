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


# --- eval harness: judge + multi-sample + parsing (review regressions) --------
def test_parse_judge_robust_to_casing_and_garbage():
    from examples.eval_harness import _parse_judge
    # lowercase 'reason:' must NOT crash (the HIGH-severity review bug)
    assert _parse_judge("SCORE: 5 | reason: correct") == {"score": 5, "reason": "correct"}
    assert _parse_judge("score: 4 | Reason: ok")["score"] == 4
    assert _parse_judge("SCORE: 7 | REASON: x")["score"] is None   # out of 1–5
    assert _parse_judge("no score at all")["score"] is None        # unparseable -> None
    assert _parse_judge("SCORE: 3")["reason"]                      # reason falls back


def test_stub_judge_scores_hallucination_low():
    from examples.eval_harness import judge_answer
    assert judge_answer("stub", "q", "rainfall recharges the aquifer")["score"] == 1
    assert judge_answer("stub", "q", "the reservoir depletes")["score"] == 4


def test_run_eval_with_stub_judge_reports_scores_and_coverage():
    from examples.eval_harness import run_eval
    rows = run_eval(["stub"], judge="stub")["stub"]
    for r in rows:
        assert isinstance(r["judge_score"], (int, float))
        assert r["judge_graded"] == r["judge_attempts"]   # all runs graded by stub


def test_word_boundary_matching_no_substring_collision():
    from examples.eval_harness import score_case
    case = {"id": "x", "must_include": ["level"], "must_avoid": [], "max_chars": 400}
    # 'levelheaded' must NOT satisfy the required term 'level'
    assert not score_case(case, "stay levelheaded today", 0.0)["passed"]
    assert score_case(case, "the liquid level is high", 0.0)["passed"]


def test_percentile_edges():
    from examples.eval_harness import _percentile
    assert _percentile([], 0.95) == 0.0
    assert _percentile([7.0], 0.95) == 7.0
    assert _percentile([1.0, 2.0], 0.95) == 2.0


def test_multisample_reports_pass_rate_for_a_flaky_model():
    import itertools
    from examples.eval_harness import run_eval
    # alternate pass/fail every call; run_eval calls this for all 3 cases × 2 runs,
    # so each case sees [pass, fail] -> a 0.5 pass-rate (multi-sampling's whole point)
    seq = itertools.cycle(["the liquid level is high",   # passes separator case
                           "stay levelheaded, unrelated"])  # fails (no 'level' word)
    rows = run_eval(["x"], answer_fn=lambda p, q: next(seq), n_runs=2)["x"]
    sep = next(r for r in rows if r["id"] == "separator_highlevel")
    assert sep["pass_rate"] == 0.5 and sep["passed"] is False and sep["n_runs"] == 2


def test_argv_rejects_flag_without_value():
    import pytest as _pt
    from examples.eval_harness import _parse_argv, self_judging
    with _pt.raises(SystemExit):
        _parse_argv(["groq", "--judge"])
    assert self_judging(["groq", "gemini"], "groq") is True
    assert self_judging(["groq"], "gemini") is False


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
