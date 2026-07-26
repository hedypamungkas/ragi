"""ragworkbench/vectorstore/chroma -- ChromaDB-backed VectorStore.

Import-gated behind the ``vectorstore-chroma`` extra (``chromadb``). Delegates storage +
ANN to a Chroma collection. The client kind is chosen from ``path``:

- ``path`` is an ``http(s)://`` URL -> ``chromadb.HttpClient()`` (a separate chroma server)
- ``path`` is a local directory -> ``chromadb.PersistentClient(path=path)`` (on-disk)
- ``path`` is None -> ``chromadb.EphemeralClient()`` (in-process, lost on exit)

Chroma applies the Mongo-style ``where`` metadata filter **natively** at query time using
the same ``$``-operators as :func:`ragworkbench.ingest.filters.matches_filter`, so ``filter``
is forwarded straight through (no post-hoc Python filtering). ``doc_id`` is folded into the
stored metadata on add and lifted back out on search so round-trip identity is preserved.
Chroma returns L2/cosine **distances** (lower = closer); we map to a similarity-like score
via ``1.0 - dist`` for parity with the other backends' higher-is-better convention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ragworkbench.errors import LLMInvalidRequestError
from ragworkbench.types import Chunk, RetrievalResult

if TYPE_CHECKING:
    from ragworkbench.protocols import EmbeddingClient

_logger = logging.getLogger(__name__)
_METHOD = "vectorstore:chroma"

try:
    import chromadb  # type: ignore[import-not-found]

    _HAS_CHROMA = True
except ImportError:  # pragma: no cover - exercised via the construct-time gate
    chromadb = None  # type: ignore[assignment]
    _HAS_CHROMA = False


_INSTALL_HINT = "pip install 'ragworkbench[vectorstore-chroma]'"


def _require_chroma() -> None:
    if not _HAS_CHROMA:
        raise LLMInvalidRequestError(f"(vectorstore-chroma) chromadb required: {_INSTALL_HINT}")


def _looks_like_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


class ChromaVectorStore:
    """ChromaDB collection backend (persistent / HTTP / ephemeral)."""

    def __init__(
        self,
        embedder: EmbeddingClient | None = None,
        collection: str = "rwb",
        path: str | None = None,
    ):
        _require_chroma()
        self._embedder = embedder
        if path is not None and _looks_like_url(path):
            # Operator is expected to point chroma's HttpClient at the right host/port
            # via chromadb's own env/config; we just pick the client kind from the URL.
            client = chromadb.HttpClient()  # type: ignore[union-attr]
        elif path is not None:
            client = chromadb.PersistentClient(path=path)  # type: ignore[union-attr]
        else:
            client = chromadb.EphemeralClient()  # type: ignore[union-attr]
        self._collection = client.get_or_create_collection(name=collection)

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
        # Fold doc_id into metadata so it survives the round-trip (Chroma has no
        # first-class doc_id field); we lift it back out on search.
        self._collection.add(
            ids=[c.id for c in usable],
            embeddings=[c.embedding for c in usable],
            documents=[c.content for c in usable],
            metadatas=[{**c.metadata, "doc_id": c.doc_id} for c in usable],
        )

    async def search(
        self,
        query: str | list[float] | None,
        *,
        top_k: int = 5,
        filter: dict | None = None,
    ) -> list[RetrievalResult]:
        if isinstance(query, str):
            if self._embedder is None:
                return []
            query = await self._embedder.embed(query)
        if query is None:
            return []
        # Chroma's `where` uses the same Mongo-style $-ops as matches_filter, so `filter`
        # forwards directly. None means "no filter".
        res = self._collection.query(
            query_embeddings=[query],
            n_results=top_k,
            where=filter,  # type: ignore[arg-type]
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[RetrievalResult] = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
            meta_dict = dict(meta or {})
            doc_id = str(meta_dict.pop("doc_id", cid))
            chunk = Chunk(
                id=str(cid),
                doc_id=doc_id,
                content=doc or "",
                metadata=meta_dict,
            )
            # Chroma returns distances (lower = closer). Convert to similarity-like.
            score = 1.0 - float(dist) if dist is not None else 0.0
            out.append(RetrievalResult(chunk=chunk, score=score, retrieval_method=_METHOD))
        return out

    @property
    def count(self) -> int:
        return int(self._collection.count())
