"""Tests for the RAG note-search layer.

Tiered by dependency so they run everywhere:
  * corpus tests        — pure Python, always run.
  * document tests      — skip if llama-index isn't installed.
  * end-to-end tests    — skip if the RAG extras OR a reachable pgvector are
                          absent (CI without a DB still passes; a dev with
                          `docker compose -f docker-compose.rag.yml up` gets the
                          full roundtrip).
"""
from __future__ import annotations

import pytest

from rag import corpus

# --- dependency probes --------------------------------------------------------
try:
    from rag.engine import deps_available, NoteSearchEngine
    from rag import store
    _DEPS_OK, _ = deps_available()
except Exception:  # noqa: BLE001
    _DEPS_OK = False

_DB_OK = False
if _DEPS_OK:
    try:
        _DB_OK, _ = store.ping()
    except Exception:  # noqa: BLE001
        _DB_OK = False

needs_deps = pytest.mark.skipif(not _DEPS_OK, reason="RAG extras not installed")
needs_db = pytest.mark.skipif(not (_DEPS_OK and _DB_OK),
                              reason="RAG extras + reachable pgvector required")


# --- corpus (pure, always run) ------------------------------------------------
def test_corpus_unions_real_and_synthetic():
    recs = corpus.build_note_records(n_synthetic=50, seed=7)
    sources = {r.source for r in recs}
    assert "synthetic" in sources
    assert sum(r.source == "synthetic" for r in recs) == 50
    # real events.csv rows ride along (the shipped synthetic event log)
    assert any(r.source == "events_csv" for r in recs)


def test_corpus_is_reproducible():
    a = corpus.build_note_records(n_synthetic=30, seed=11)
    b = corpus.build_note_records(n_synthetic=30, seed=11)
    assert [r.note for r in a] == [r.note for r in b]
    assert [r.well_id for r in a] == [r.well_id for r in b]


def test_record_text_and_metadata_roundtrip():
    rec = corpus.build_note_records(n_synthetic=5, seed=1)[-1]
    text = rec.to_text()
    assert rec.note in text
    assert rec.well_id in text
    md = rec.metadata()
    assert md["cause"] == rec.cause
    assert md["recoverable"] == rec.recoverable
    assert isinstance(md["recoverable"], bool)


def test_recoverable_flag_matches_taxonomy():
    # reservoir + planned are non-recoverable; the rest recoverable
    assert not corpus.NoteRecord(
        "x", "w", "2024-01-01", "2024-01-02", 1, 10, "reservoir",
        "watering out", "synthetic").recoverable
    assert corpus.NoteRecord(
        "x", "w", "2024-01-01", "2024-01-02", 1, 10, "power",
        "lost power", "synthetic").recoverable


# --- documents (needs llama-index) --------------------------------------------
@needs_deps
def test_records_convert_to_documents():
    recs = corpus.build_note_records(n_synthetic=8, seed=3)
    docs = corpus.records_to_documents(recs)
    assert len(docs) == len(recs)
    d0 = docs[0]
    assert recs[0].note in d0.text
    assert d0.metadata["well_id"] == recs[0].well_id


# --- end-to-end (needs pgvector) ----------------------------------------------
@needs_db
def test_build_and_retrieve_roundtrip():
    """Index a small corpus and prove semantic retrieval works.

    Uses a throwaway table so it never touches a real index.
    """
    eng = NoteSearchEngine(table="operator_notes_test")
    eng.reset_table()
    recs = corpus.build_note_records(n_synthetic=120, seed=5)
    n = eng.build_index(recs)
    assert n == len(recs)
    assert eng.index_size() == n

    # a query with NO keyword overlap with the target note must still retrieve it
    hits = eng.retrieve("the gas buyer cut our pipeline takeaway", top_k=5)
    assert hits
    assert any(h.cause == "gathering_thirdparty" for h in hits)

    # cause filter restricts results
    filtered = eng.retrieve("equipment problem", top_k=5, cause="power")
    assert filtered
    assert all(h.cause == "power" for h in filtered)

    eng.reset_table()  # clean up the throwaway table


@needs_db
def test_extractive_answer_without_key():
    eng = NoteSearchEngine(table="operator_notes_test2")
    eng.reset_table()
    eng.build_index(corpus.build_note_records(n_synthetic=60, seed=2))
    ans = eng.answer("freeze offs", top_k=4, anthropic_key=None)
    assert not ans.used_llm
    assert ans.sources
    assert "bbl deferred" in ans.text
    eng.reset_table()
