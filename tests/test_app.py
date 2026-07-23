"""App-layer tests: the designed navigation map, a full AppTest render smoke of
app.py, and per-view execution coverage through the AppTest harness."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "view_harness.py"

# The designed page map — section order and Title Case page titles.
EXPECTED_NAV = {
    "Today": ["Home", "Surveillance", "Morning Brief", "Optimization Board",
              "Ongoing Events"],
    "Loss Accounting": ["Deferment Overview", "Causes & Pareto",
                        "Recovery Work Queue", "Note Search (RAG)"],
    "Well File": ["Well 360", "Action Chain"],
    "Data": ["Sources & BYOD", "Methods & Limitations"],
}

ALL_TITLES = [t for ts in EXPECTED_NAV.values() for t in ts]


def test_navigation_page_map_is_as_designed():
    import views
    got = {section: [t for (t, _i, _s, _r, _d) in specs]
           for section, specs in views.PAGES.items()}
    assert got == EXPECTED_NAV
    # exactly one default page, and it's Home
    defaults = [t for specs in views.PAGES.values()
                for (t, _i, _s, _r, d) in specs if d]
    assert defaults == ["Home"]
    # url_paths unique; material icons only (no emoji in nav)
    slugs = [s for specs in views.PAGES.values() for (_t, _i, s, _r, _d) in specs]
    assert len(slugs) == len(set(slugs))
    icons = [i for specs in views.PAGES.values() for (_t, i, _s, _r, _d) in specs]
    assert all(i.startswith(":material/") for i in icons)


def test_app_render_smoke(bootstrapped):
    """Full app.py run (set_page_config, sidebar, bootstrap, navigation, Home)
    with zero exceptions. First cold run trains the ESP model (~30s) — the
    session bootstrap fixture has already paid that cost."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert not at.error, [str(e.value) for e in at.error]


def _run_view(title: str, monkeypatch, extra_state: dict | None = None):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("OPS_VIEW", title)
    at = AppTest.from_file(str(HARNESS), default_timeout=300)
    at.session_state["oil_price"] = 70.0
    at.session_state["nri"] = 0.80
    at.session_state["discount"] = 0.10
    at.session_state["well_id"] = "well_001"
    at.session_state["data_source"] = "real_co"
    at.session_state["anthropic_key"] = ""
    for k, v in (extra_state or {}).items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, f"{title}: {[str(e.value) for e in at.exception]}"
    return at


@pytest.mark.parametrize("title", ALL_TITLES)
def test_each_view_renders_without_exception(title, bootstrapped, monkeypatch):
    _run_view(title, monkeypatch)


@pytest.mark.parametrize("title", ["Deferment Overview", "Causes & Pareto",
                                   "Recovery Work Queue"])
def test_loss_views_render_on_synthetic_source(title, bootstrapped, monkeypatch):
    """The reason-coded synthetic source exercises the cause/queue branches the
    real default honestly N/A's out."""
    _run_view(title, monkeypatch, extra_state={"data_source": "synthetic"})


def test_well_360_renders_for_alerted_well(bootstrapped, monkeypatch):
    """well_013 carries today's top digest alert — exercises the alert-overlay
    branch (flagged path) of the Well File pages."""
    _run_view("Well 360", monkeypatch, extra_state={"well_id": "well_013"})
    _run_view("Action Chain", monkeypatch, extra_state={"well_id": "well_013"})
