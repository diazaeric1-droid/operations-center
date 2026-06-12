"""AppTest harness: render ONE Operations Center view as a bare Streamlit script.

Driven by the OPS_VIEW env var (a page title from views.PAGES). Used by
tests/test_app.py for per-view execution coverage — each view is exercised
exactly as st.navigation would call it, with session state pre-set by the test.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import views  # noqa: E402

_RENDERS = {title: render
            for specs in views.PAGES.values()
            for (title, _icon, _slug, render, _default) in specs}

_RENDERS[os.environ["OPS_VIEW"]]()
