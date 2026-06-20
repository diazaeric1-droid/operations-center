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


# --- observability / cost tracing --------------------------------------------
def test_price_lookup_known_and_unknown():
    from langgraph_rag.tracing import price_for, cost_usd
    pin, pout, known = price_for("claude-sonnet-4-6")
    assert known and pin > 0 and pout > pin
    assert price_for("some-unknown-model-xyz") == (0.0, 0.0, False)
    # 1M in + 1M out at sonnet rates = $3 + $15
    assert round(cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000), 2) == 18.00


def test_summary_aggregates_per_provider():
    from langgraph_rag.tracing import TraceEvent, summary
    events = [
        TraceEvent("groq", "llama-3.3-70b", 50, 80, 0.3, 0.0001),
        TraceEvent("groq", "llama-3.3-70b", 60, 90, 0.5, 0.0002),
        TraceEvent("gemini", "gemini-2.5-flash", 40, 200, 2.9, 0.0006),
    ]
    s = summary(events)
    assert s["calls"] == 3
    assert s["total_tokens"] == (130 + 150 + 240)
    assert s["by_provider"]["groq"]["calls"] == 2
    assert s["by_provider"]["gemini"]["p95_latency"] == 2.9


def test_chat_traced_records_an_event(monkeypatch):
    from langgraph_rag import tracing, providers
    monkeypatch.setattr(providers, "chat_meta", lambda *a, **k: {
        "text": "hi", "provider": "groq", "model": "llama-3.3-70b",
        "prompt_tokens": 10, "completion_tokens": 20, "tokens_estimated": False})
    with tracing.trace() as events:
        out = tracing.chat_traced("q", provider="groq", label="answer")
    assert out == "hi"
    assert len(events) == 1 and events[0].total_tokens == 30
    assert events[0].cost_usd > 0 and events[0].label == "answer"


def test_observability_workload_traces_each_provider(monkeypatch):
    from langgraph_rag import providers
    from examples.observability import run_workload
    monkeypatch.setattr(providers, "chat_meta", lambda prompt, provider, **k: {
        "text": "a", "provider": provider, "model": "llama-3.3-70b",
        "prompt_tokens": 5, "completion_tokens": 5, "tokens_estimated": False})
    events = run_workload("q", ["groq", "gemini"])
    assert len(events) == 2 and {e.provider for e in events} == {"groq", "gemini"}


def test_summary_lists_all_models_per_provider():
    from langgraph_rag.tracing import TraceEvent, summary
    events = [TraceEvent("claude", "claude-haiku", 100, 100, 0.2, 0.001),
              TraceEvent("claude", "claude-opus", 100, 100, 0.3, 0.003)]
    # the cost must not be tagged to just one model when a provider serves several
    assert summary(events)["by_provider"]["claude"]["models"] == \
        ["claude-haiku", "claude-opus"]


def test_print_report_marks_unknown_price_not_as_free(capsys):
    from langgraph_rag.tracing import TraceEvent, print_report
    print_report([TraceEvent("x", "mystery-model", 50, 50, 0.4, 0.0,
                             price_known=False)])
    out = capsys.readouterr().out
    assert "? no price" in out          # the row is marked, not a bare $ amount
    assert "no list price" in out       # legend explains it
    assert "≥" in out                   # the total is flagged as a lower bound


def test_trace_restores_active_on_exception_and_record_is_safe():
    from langgraph_rag import tracing
    try:
        with tracing.trace():
            raise ValueError("boom")
    except ValueError:
        pass
    assert tracing._active is None                 # restored despite the exception
    tracing.record(object())                       # no active trace -> silent no-op


def test_pct_edges():
    from langgraph_rag.tracing import _pct
    assert _pct([], 0.95) == 0.0
    assert _pct([5.0], 0.95) == 5.0
    assert _pct([1.0, 2.0, 3.0], 1.0) == 3.0
    assert _pct([1.0, 2.0, 3.0], 0.0) == 1.0


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
