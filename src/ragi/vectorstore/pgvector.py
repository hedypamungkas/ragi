"""ragi/vectorstore/pgvector -- Postgres + pgvector-backed VectorStore.

Import-gated behind the ``vectorstore-pgvector`` extra (``psycopg`` 3). Uses the
``<=>`` cosine-distance operator from the ``vector`` extension. Requires a running
Postgres with ``vector`` installed.

Notes
-----
- The connection is **synchronous** (``psycopg.connect``); calls block the event loop.
  Wrap ``add``/``search`` in ``asyncio.to_thread`` at a higher layer if you need
  non-blocking behavior.
- ``CREATE EXTENSION vector`` is best-effort: it typically needs superuser, so if the
  ``dsn`` role lacks the privilege the call is swallowed with a warning and the operator
  is expected to install the extension out-of-band.
- The ``vector`` Python adapter (``pgvector.psycopg``) is **not** required: embeddings
  are serialized to pgvector's text literal form (``[1.0,2.0,...]``) and cast with
  ``::vector`` on the server side, which works with a plain psycopg3 install.
- ``filter`` is rendered into the WHERE clause as jsonb predicates on ``metadata``
  (text extraction via ``metadata->>%s``); semantics mirror
  :func:`ragi.ingest.filters.matches_filter` (scalar equality + ``$gte/$gt/
  $lte/$lt/$in``). Numeric comparisons are lexical on the text extraction -- sufficient
  for equality/``$in`` and same-year-precision ranges; if you need typed numeric
  ordering, cast the jsonb path to numeric in your own filter.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ragi.errors import LLMInvalidRequestError
from ragi.types import Chunk, RetrievalResult

if TYPE_CHECKING:
    from ragi.protocols import EmbeddingClient

_logger = logging.getLogger(__name__)
_METHOD = "vectorstore:pgvector"

try:
    import psycopg  # type: ignore[import-not-found]

    _HAS_PSYCOPG = True
except ImportError:  # pragma: no cover - exercised via the construct-time gate
    psycopg = None  # type: ignore[assignment]
    _HAS_PSYCOPG = False


_INSTALL_HINT = "pip install 'ragi[vectorstore-pgvector]'"

# SQL comparison operator by Mongo-style $op (mirrors filters._cmp).
_SQL_OP = {"$gte": ">=", "$gt": ">", "$lte": "<=", "$lt": "<"}


def _require_psycopg() -> None:
    if not _HAS_PSYCOPG:
        raise LLMInvalidRequestError(f"(vectorstore-pgvector) psycopg required: {_INSTALL_HINT}")


def _vec_str(vec: list[float]) -> str:
    """pgvector text literal: ``[1.0,2.0,3.0]``. Accepted by ``::vector`` on the server."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


class PgvectorStore:
    """Postgres + pgvector backend (sync psycopg3 connection)."""

    def __init__(
        self,
        embedder: EmbeddingClient | None = None,
        dsn: str | None = None,
        table: str = "ragi_chunks",
    ):
        _require_psycopg()
        if not dsn:
            raise LLMInvalidRequestError("(vectorstore-pgvector) dsn is required")
        self._embedder = embedder
        self._dsn = dsn
        # ``table`` is operator-supplied config (not user input) -- it cannot be
        # parameterized in psycopg, so it is interpolated verbatim. Same trust posture
        # as the DSN itself.
        self._table = table
        self._conn = psycopg.connect(dsn)  # type: ignore[union-attr]
        self._dim: int | None = None
        # Best-effort: CREATE EXTENSION typically needs superuser. Degrade + document.
        try:
            with self._conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._conn.commit()
        except Exception as exc:  # pragma: no cover - permission-dependent
            _logger.warning(
                "pgvector: CREATE EXTENSION vector failed (%s); install it out-of-band if the role lacks privilege",
                exc,
            )
            self._conn.rollback()
        # Table is created lazily once dim is inferred from the first add.

    def _ensure_table(self, dim: int) -> None:
        if self._dim == dim:
            return
        self._dim = dim
        with self._conn.cursor() as cur:
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                "id text PRIMARY KEY, "
                "doc_id text, "
                "content text, "
                "metadata jsonb, "
                f"embedding vector({dim})"
                ")"
            )
        self._conn.commit()

    async def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        to_embed = [c for c in chunks if c.embedding is None]
        if to_embed and self._embedder is not None:
            vectors = await self._embedder.embed_batch([c.content for c in to_embed])
            for c, v in zip(to_embed, vectors, strict=True):
                c.embedding = v
        usable = [c for c in chunks if c.embedding is not None]
        if not usable:
            return
        self._ensure_table(len(usable[0].embedding))  # type: ignore[arg-type]
        with self._conn.cursor() as cur:
            for c in usable:
                cur.execute(
                    f"INSERT INTO {self._table} "
                    "(id, doc_id, content, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s::vector) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "doc_id = EXCLUDED.doc_id, "
                    "content = EXCLUDED.content, "
                    "metadata = EXCLUDED.metadata, "
                    "embedding = EXCLUDED.embedding",
                    (
                        c.id,
                        c.doc_id,
                        c.content,
                        json.dumps(c.metadata),
                        _vec_str(c.embedding),  # type: ignore[arg-type]
                    ),
                )
        self._conn.commit()

    @staticmethod
    def _render_filter(filter: dict | None) -> tuple[str, list]:
        """Render a Mongo-style filter as a jsonb WHERE clause + params.

        Mirrors :func:`ragi.ingest.filters.matches_filter`: scalar -> equality
        (text), ``$gte/$gt/$lte/$lt`` -> comparison, ``$in`` -> membership. Unknown
        operators yield ``FALSE`` (strict no-match, same as ``matches_filter``).
        """
        if not filter:
            return ("", [])
        clauses: list[str] = []
        params: list = []
        for field, clause in filter.items():
            if isinstance(clause, dict):
                for op, val in clause.items():
                    if op == "$in":
                        if not val:
                            clauses.append("FALSE")  # empty $in -> no match
                            continue
                        placeholders = ",".join(["%s"] * len(val))
                        clauses.append(f"(metadata->>%s IN ({placeholders}))")
                        params.append(field)
                        params.extend(str(v) for v in val)
                    elif op in _SQL_OP:
                        clauses.append(f"(metadata->>%s {_SQL_OP[op]} %s)")
                        params.append(field)
                        params.append(str(val))
                    else:
                        # unknown operator -> strict no-match (matches filters.py)
                        clauses.append("FALSE")
            else:
                clauses.append("(metadata->>%s = %s)")
                params.append(field)
                params.append(str(clause))
        return (" AND ".join(clauses), params)

    async def search(
        self,
        query: str | list[float] | None,
        *,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[RetrievalResult]:
        if self._dim is None:
            return []  # nothing indexed yet (table not created)
        if isinstance(query, str):
            if self._embedder is None:
                return []
            query = await self._embedder.embed(query)
        if query is None:
            return []
        where_sql, where_params = self._render_filter(filter)
        where_clause = f"WHERE {where_sql}" if where_sql else ""
        sql = (
            f"SELECT id, doc_id, content, metadata, "
            f"embedding <=> %s::vector AS dist "
            f"FROM {self._table} {where_clause} "
            f"ORDER BY dist LIMIT %s"
        )
        # param order: query vector, then WHERE params, then LIMIT
        params: list = [_vec_str(query), *where_params, top_k]
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        out: list[RetrievalResult] = []
        for row in rows:
            cid, doc_id, content, meta_json, dist = row
            if isinstance(meta_json, str):
                metadata = json.loads(meta_json)
            elif meta_json is None:
                metadata = {}
            else:
                # psycopg3 returns jsonb as a Python object directly when adapted
                metadata = dict(meta_json)
            # cosine distance <=> in [0, 2]; similarity = 1 - dist (higher is better)
            score = 1.0 - float(dist) if dist is not None else 0.0
            out.append(
                RetrievalResult(
                    chunk=Chunk(
                        id=str(cid),
                        doc_id=str(doc_id or cid),
                        content=content or "",
                        metadata=metadata,
                    ),
                    score=score,
                    retrieval_method=_METHOD,
                )
            )
        return out

    @property
    def count(self) -> int:
        if self._dim is None:
            return 0  # table not created yet
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {self._table}")
            row = cur.fetchone()
        return int(row[0]) if row else 0
