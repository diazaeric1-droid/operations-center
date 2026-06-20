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
