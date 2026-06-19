"""The RAG engine: query -> retrieved operator notes -> cited synthesis.

Pipeline
--------
    embed (fastembed / BAAI bge-small, local ONNX, no API key)
      -> retrieve top-k from pgvector (optional metadata filter on cause)
        -> synthesize a grounded answer with Claude (BYOK), citing [n] notes
           (no key -> deterministic extractive rollup; the product never *needs*
            a key, the LLM only makes the answer prettier)

Why this shape closes the résumé gaps in one place:
  * RAG / embeddings / semantic search ...... retrieve()
  * vector database (pgvector) .............. store.make_vector_store()
  * LlamaIndex framework .................... index + retriever below
  * LLM synthesis w/ citations ............. answer()
The MCP server (mcp_server.py) and the Streamlit page (views/note_search.py)
are both thin wrappers over this one class.
"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass

from . import corpus, store

_EMBED_MODEL = "BAAI/bge-small-en-v1.5"   # 384-dim; matches store.EMBED_DIM


def deps_available() -> tuple[bool, str]:
    """Are the RAG extras importable? (ok, missing-detail)."""
    missing = []
    for mod, pip in (("llama_index.core", "llama-index-core"),
                     ("llama_index.vector_stores.postgres",
                      "llama-index-vector-stores-postgres"),
                     ("llama_index.embeddings.fastembed",
                      "llama-index-embeddings-fastembed")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pip)
    if missing:
        return False, "pip install " + " ".join(missing)
    return True, ""


@dataclass
class RetrievedNote:
    score: float
    note: str
    well_id: str
    cause: str
    start_date: str
    duration_days: int
    deferred_bbl: int
    source: str

    @classmethod
    def from_node(cls, n) -> "RetrievedNote":
        m = n.node.metadata
        return cls(
            score=round(float(n.score or 0.0), 4),
            note=m.get("note", n.node.get_content()),
            well_id=m.get("well_id", "?"),
            cause=m.get("cause", "?"),
            start_date=m.get("start_date", "?"),
            duration_days=int(m.get("duration_days", 0)),
            deferred_bbl=int(m.get("deferred_bbl", 0)),
            source=m.get("source", "?"))


@dataclass
class Answer:
    text: str
    used_llm: bool
    sources: list[RetrievedNote]


class NoteSearchEngine:
    """Build/load the pgvector index and query it. Construct once, reuse."""

    def __init__(self, dsn: str | None = None, table: str = store.DEFAULT_TABLE):
        self.dsn = dsn or store.default_dsn()
        self.table = table
        self._index = None
        self._configure_embeddings()

    # --- setup ---------------------------------------------------------------
    @staticmethod
    def _configure_embeddings() -> None:
        from llama_index.core import Settings
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        # Local ONNX embeddings; do synthesis ourselves so llama-index needs no
        # LLM configured (keeps BYOK + the no-key fallback in our hands).
        Settings.embed_model = FastEmbedEmbedding(model_name=_EMBED_MODEL)
        Settings.llm = None

    def _vector_store(self):
        return store.make_vector_store(self.dsn, self.table)

    # --- index lifecycle -----------------------------------------------------
    def build_index(self, records: list[corpus.NoteRecord] | None = None,
                    show_progress: bool = False) -> int:
        """Embed the corpus into pgvector. Returns the document count.

        Idempotent-ish: the table is created if absent; re-running appends, so
        callers that want a clean rebuild should drop the table first
        (reset_table()).
        """
        from llama_index.core import VectorStoreIndex, StorageContext
        records = records if records is not None else corpus.build_note_records()
        documents = corpus.records_to_documents(records)
        ctx = StorageContext.from_defaults(vector_store=self._vector_store())
        self._index = VectorStoreIndex.from_documents(
            documents, storage_context=ctx, show_progress=show_progress)
        return len(documents)

    def load_index(self):
        from llama_index.core import VectorStoreIndex
        self._index = VectorStoreIndex.from_vector_store(self._vector_store())
        return self._index

    def index_size(self) -> int:
        """How many vectors are in the table (0 if it doesn't exist yet)."""
        import psycopg2
        p = store.PgParams.from_dsn(self.dsn)
        conn = psycopg2.connect(host=p.host, port=p.port, dbname=p.database,
                                user=p.user, password=p.password)
        try:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) FROM data_{self.table};")
                return int(cur.fetchone()[0])
        except Exception:  # noqa: BLE001 — table absent -> empty index
            return 0
        finally:
            conn.close()

    def reset_table(self) -> None:
        """Drop the vector table (for a clean rebuild)."""
        import psycopg2
        p = store.PgParams.from_dsn(self.dsn)
        conn = psycopg2.connect(host=p.host, port=p.port, dbname=p.database,
                                user=p.user, password=p.password)
        try:
            with conn, conn.cursor() as cur:
                # llama-index prefixes table names with "data_".
                cur.execute(f'DROP TABLE IF EXISTS data_{self.table} CASCADE;')
        finally:
            conn.close()

    # --- query ---------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 6,
                 cause: str | None = None) -> list[RetrievedNote]:
        if self._index is None:
            self.load_index()
        filters = None
        if cause:
            from llama_index.core.vector_stores import (
                MetadataFilter, MetadataFilters, FilterOperator)
            filters = MetadataFilters(filters=[
                MetadataFilter(key="cause", value=cause,
                               operator=FilterOperator.EQ)])
        retriever = self._index.as_retriever(similarity_top_k=top_k,
                                              filters=filters)
        return [RetrievedNote.from_node(n) for n in retriever.retrieve(query)]

    def answer(self, query: str, top_k: int = 6, cause: str | None = None,
               anthropic_key: str | None = None,
               model: str = "claude-sonnet-4-6") -> Answer:
        """Retrieve, then synthesize a cited answer (LLM if key, else extractive)."""
        hits = self.retrieve(query, top_k=top_k, cause=cause)
        if not hits:
            return Answer("No operator notes matched that query.", False, [])
        if anthropic_key:
            try:
                return Answer(self._llm_answer(query, hits, anthropic_key, model),
                              True, hits)
            except Exception:  # noqa: BLE001 — never fail the page on an API hiccup
                pass
        return Answer(self._extractive_answer(query, hits), False, hits)

    # --- synthesis backends --------------------------------------------------
    @staticmethod
    def _context_block(hits: list[RetrievedNote]) -> str:
        lines = []
        for i, h in enumerate(hits, 1):
            lines.append(
                f"[{i}] ({h.well_id}, {h.start_date}, {h.cause}, "
                f"{h.duration_days}d, {h.deferred_bbl} bbl) {h.note}")
        return "\n".join(lines)

    def _llm_answer(self, query: str, hits: list[RetrievedNote],
                    key: str, model: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        ctx = self._context_block(hits)
        sys = (
            "You are a production-operations analyst. Answer the question using "
            "ONLY the numbered operator notes provided. Cite the notes you use "
            "as [n]. Be concise and specific; if the notes don't support an "
            "answer, say so. Never invent wells, dates, or causes.")
        msg = client.messages.create(
            model=model, max_tokens=400, system=sys,
            messages=[{"role": "user",
                       "content": f"Question: {query}\n\nOperator notes:\n{ctx}"}])
        return "".join(b.text for b in msg.content if b.type == "text").strip()

    @staticmethod
    def _extractive_answer(query: str, hits: list[RetrievedNote]) -> str:
        """No-key deterministic rollup: the matches + a small aggregate."""
        total = sum(h.deferred_bbl for h in hits)
        by_cause: dict[str, int] = {}
        for h in hits:
            by_cause[h.cause] = by_cause.get(h.cause, 0) + 1
        cause_line = ", ".join(f"{k} ×{v}" for k, v in
                               sorted(by_cause.items(), key=lambda x: -x[1]))
        head = textwrap.dedent(f"""\
            {len(hits)} notes match — {total:,} bbl deferred across them ({cause_line}).
            Add an Anthropic key in the sidebar for a narrated answer; the matches:""")
        body = "\n".join(
            f"  [{i}] {h.well_id} · {h.start_date} · {h.cause} · "
            f"{h.duration_days}d · {h.deferred_bbl} bbl — {h.note}"
            for i, h in enumerate(hits, 1))
        return head + "\n" + body
