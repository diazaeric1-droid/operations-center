"""Operations Center views package.

``PAGES`` is the product's navigation spec as PLAIN DATA (section → list of
(title, material icon, url_path, render callable, is_default)) so tests can
assert the designed page map without a Streamlit runtime. app.py turns it into
``st.Page`` objects and fills ``PAGE_OBJECTS`` (title → Page) so views can
``st.page_link`` to each other.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Repo root on sys.path so `import core / theme / product_theme / fleet_registry`
# resolve no matter how the package is imported (streamlit run, pytest, AppTest).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from views import (  # noqa: E402
    action_chain,
    causes_pareto,
    data_sources,
    deferment_overview,
    home,
    methods,
    morning_brief,
    note_search,
    ongoing_events,
    recovery_queue,
    surveillance,
    triage_board,
    well_360,
)

# (title, icon, url_path, render, default)
PAGES: dict[str, list[tuple[str, str, str, object, bool]]] = {
    "Today": [
        ("Home", ":material/home:", "home", home.render, True),
        ("Surveillance", ":material/insights:", "surveillance",
         surveillance.render, False),
        ("Morning Brief", ":material/article:", "morning-brief",
         morning_brief.render, False),
        # Renamed from the earlier "triage board" title per PE feedback (OC5) —
        # module filename kept. The slug changed too ("triage-board" →
        # "optimization-board"): old deep links 404, noted in the CHANGELOG.
        ("Optimization Board", ":material/monitoring:", "optimization-board",
         triage_board.render, False),
        ("Ongoing Events", ":material/event_repeat:", "ongoing-events",
         ongoing_events.render, False),
    ],
    "Loss Accounting": [
        ("Deferment Overview", ":material/waterfall_chart:", "deferment-overview",
         deferment_overview.render, False),
        ("Causes & Pareto", ":material/bar_chart:", "causes-pareto",
         causes_pareto.render, False),
        ("Recovery Work Queue", ":material/build:", "recovery-queue",
         recovery_queue.render, False),
        ("Note Search (RAG)", ":material/search:", "note-search",
         note_search.render, False),
    ],
    "Well File": [
        ("Well 360", ":material/oil_barrel:", "well-360", well_360.render, False),
        ("Action Chain", ":material/account_tree:", "action-chain",
         action_chain.render, False),
    ],
    "Data": [
        ("Sources & BYOD", ":material/database:", "sources",
         data_sources.render, False),
        ("Methods & Limitations", ":material/fact_check:", "methods",
         methods.render, False),
    ],
}

# Filled by app.py at navigation wiring: {page title: st.Page object}.
PAGE_OBJECTS: dict[str, object] = {}
