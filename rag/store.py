"""pgvector wiring — the vector database the notes are indexed into.

One Postgres table with a ``vector`` column (the pgvector extension). Locally
that's a Docker container (``docker-compose.rag.yml``); in the cloud the *exact
same code* points at an AWS RDS Postgres instance with pgvector enabled — only
``OPS_PG_DSN`` changes. That portability is the point of using a real SQL
vector store instead of an in-process toy.

Connection precedence:
    OPS_PG_DSN env var  ->  local docker default (localhost:5433)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

# Embedding dimension for BAAI/bge-small-en-v1.5 (the fastembed default we use).
# Must match the model in engine.py; the pgvector column is sized to it.
EMBED_DIM = 384
DEFAULT_TABLE = "operator_notes"

# Local Docker default (see docker-compose.rag.yml). Port 5433 avoids colliding
# with a system Postgres on 5432.
DEFAULT_DSN = "postgresql://ops:ops@localhost:5433/opsrag"


@dataclass(frozen=True)
class PgParams:
    host: str
    port: int
    database: str
    user: str
    password: str

    @classmethod
    def from_dsn(cls, dsn: str) -> "PgParams":
        u = urlparse(dsn)
        return cls(
            host=u.hostname or "localhost",
            port=u.port or 5432,
            database=(u.path or "/opsrag").lstrip("/") or "opsrag",
            user=u.username or "ops",
            password=u.password or "ops",
        )


def default_dsn() -> str:
    return os.environ.get("OPS_PG_DSN", DEFAULT_DSN)


def ping(dsn: str | None = None, timeout: float = 2.0) -> tuple[bool, str]:
    """Cheap reachability check so the UI can say 'DB down' instead of hanging.

    Returns (ok, detail). Never raises.
    """
    dsn = dsn or default_dsn()
    p = PgParams.from_dsn(dsn)
    try:
        import psycopg2  # noqa: PLC0415
    except ImportError:
        return False, "psycopg2 not installed (pip install -r requirements-rag.txt)"
    try:
        conn = psycopg2.connect(
            host=p.host, port=p.port, dbname=p.database,
            user=p.user, password=p.password, connect_timeout=int(timeout))
        conn.close()
        return True, f"{p.host}:{p.port}/{p.database}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def make_vector_store(dsn: str | None = None, table: str = DEFAULT_TABLE,
                      embed_dim: int = EMBED_DIM):
    """Build a LlamaIndex ``PGVectorStore`` (lazy import of the extras)."""
    dsn = dsn or default_dsn()
    p = PgParams.from_dsn(dsn)
    try:
        from llama_index.vector_stores.postgres import PGVectorStore
    except ImportError as e:  # noqa: BLE001
        raise ImportError(
            "llama-index pgvector store not installed. Run:\n"
            "    pip install -r requirements-rag.txt") from e
    return PGVectorStore.from_params(
        host=p.host, port=str(p.port), database=p.database,
        user=p.user, password=p.password,
        table_name=table, embed_dim=embed_dim,
        # IVFFlat is plenty for a few-hundred-to-millions-of-rows note corpus and
        # keeps the AWS RDS footprint small; swap to HNSW for very large fleets.
        hnsw_kwargs=None,
    )
