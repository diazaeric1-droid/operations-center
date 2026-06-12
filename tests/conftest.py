"""Shared test setup: repo root importable + one bootstrap per session."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def bootstrapped():
    """Regenerate every gitignored artifact once (idempotent; ~30s cold for the
    ESP model train, instant warm)."""
    import core
    core.bootstrap(log=lambda *_a, **_k: None)
    return True
