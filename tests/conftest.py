"""Shared test setup: repo root importable + one bootstrap per session."""
from __future__ import annotations

import os

# Native-lib coexistence guard (must be set BEFORE torch / onnxruntime load).
# The optional DL extras (torch) and RAG extras (fastembed -> onnxruntime) each
# bundle their own OpenMP runtime; with both installed, their intra-op thread
# pools collide and segfault at teardown when the full suite runs in one process.
# Pinning to a single OMP thread + allowing the duplicate libomp removes the
# clash. No effect in CI (neither extra is installed there) and none on the
# deployed app (which imports neither). Harmless when the extras are absent.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
