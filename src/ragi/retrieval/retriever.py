"""ragi/retrieval/retriever -- lexical + embedding-based retrievers.

Lexical (:class:`KeywordRetriever` TF-IDF cosine + :class:`BM25Retriever`) lift from
``koboi/rag/retriever.py``; embedding-based (:class:`SemanticRetriever` +
:class:`HybridRetriever`) lift from the same source. Decoupled: **zero
``koboi.*`` imports**. All four share the :class:`BaseRetriever` ABC and
self-register on import via :func:`ragi.registry.register_retriever` so a
plain ``from ragi.retrieval.retriever import ...`` is enough to populate
:data:`retriever_registry`.

Optional Indonesian morphology (Sastrawi stemmer via the ``[indo-nlp]`` extra) is
try-excepted on the lexical legs -- absent the extra, a warning is logged and
stemming is skipped.

Semantic/Hybrid need an :class:`ragi.protocols.EmbeddingClient` (e.g.
:class:`ragi.embeddings.openai.OpenAIEmbeddingClient`); they degrade to
keyword retrieval when no embedder is supplied or the endpoint returns ``None``,
and reuse the process-level :data:`ragi.embeddings.cache._EMBEDDING_CACHE`
so the corpus is embedded once per process.
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from ragi.embeddings.cache import _EMBEDDING_CACHE
from ragi.ingest.filters import matches_filter
from ragi.registry import register_retriever, retriever_registry
from ragi.retrieval.stopwords import STOPWORDS_EN, STOPWORDS_ID, get_stopwords
from ragi.types import Chunk, RetrievalResult

if TYPE_CHECKING:
    from ragi.protocols import EmbeddingClient

_logger = logging.getLogger(__name__)


# Re-export the stopword sets for callers that import from this module (mirrors
# koboi's pre-extraction surface where these lived alongside the retrievers).
__all__ = [
    "BaseRetriever",
    "KeywordRetriever",
    "BM25Retriever",
    "SemanticRetriever",
    "HybridRetriever",
    "resolve_retriever",
    "STOPWORDS_EN",
    "STOPWORDS_ID",
]


def _normalize_stemmer(stemmer: str | bool | None) -> Callable[[str], str] | None:
    """Resolve a ``stemmer`` spec to a cached stem function (or ``None`` = off).

    Only the literal ``"id"`` is accepted (Indonesian, via the optional
    ``[indo-nlp]`` extra / Sastrawi); it normalizes morphology (meN-, ber-, -kan,
    -i, -an) so inflected forms match their root. Applied to BOTH index and query
    tokens (consistent). Unlike ``stopwords``, ``True`` is NOT a valid alias here
    (no English stemmer ships) -- ``True`` or any other value logs a warning and
    disables stemming. Falls back to ``None`` if the extra is absent. Default off.
    """
    if stemmer is None or stemmer is False:
        return None
    if stemmer == "id":
        try:
            from Sastrawi.Stemmer.StemmerFactory import (  # optional [indo-nlp] extra
                StemmerFactory,
            )
        except ImportError:
            _logger.warning(
                "stemmer 'id' needs the [indo-nlp] extra (Sastrawi): pip install 'ragi[indo-nlp]'. Skipping stemming."
            )
            return None

        sastrawi = StemmerFactory().create_stemmer()

        @lru_cache(maxsize=200_000)
        def _stem(word: str) -> str:
            return sastrawi.stem(word)

        return _stem
    _logger.warning("Unknown stemmer %r; skipping. Supported: 'id'.", stemmer)
    return None


def resolve_retriever(name: str, chunks: list[Chunk], **kwargs) -> BaseRetriever:
    """Build a registered retriever by name.

    Looks up ``name`` in :data:`retriever_registry` and passes ``chunks`` plus
    any additional ``kwargs`` to its constructor. Falls back to ``"keyword"``
    (with a warning) when ``name`` is unknown; raises :class:`ValueError` when
    no retriever is registered at all.
    """
    entry = retriever_registry.get(name)
    if entry is None:
        _logger.warning(
            "Unknown retriever %r, falling back to 'keyword'. Available: %s",
            name,
            retriever_registry.list_available(),
        )
        entry = retriever_registry.get("keyword")
        if entry is None:
            raise ValueError("No retrievers registered")
    return entry.cls(chunks=chunks, **kwargs)


class BaseRetriever(ABC):
    """ABC for lexical retrievers.

    Subclasses populate ``self._chunks`` and build their index in ``__init__``.
    The shared :meth:`_allowed_ids` helper applies a relevance-scoping metadata
    filter (NOT an ACL boundary -- see :mod:`ragi.ingest.filters`).
    """

    _chunks: list[Chunk]

    def _allowed_ids(self, metadata_filter: dict | None) -> set[str] | None:
        """Relevance-scoping filter -> the set of chunk ids that pass it.

        Returns ``None`` when no filter is set (all chunks eligible). NOT an ACL
        boundary -- a metadata typo here must never be the only thing keeping
        tenants apart; see :mod:`ragi.ingest.filters`.
        """
        if not metadata_filter:
            return None
        return {c.id for c in self._chunks if matches_filter(c.metadata, metadata_filter)}

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int = 3, metadata_filter: dict | None = None
    ) -> list[RetrievalResult]: ...


@register_retriever("keyword", description="TF-IDF cosine similarity retrieval")
class KeywordRetriever(BaseRetriever):
    """TF-IDF cosine-similarity lexical retrieval (pure stdlib).

    Options (all optional, off by default):

    - ``synonyms``: query-side lexical-bridge map (e.g. ``{"dog": ["pet"]}``).
      Applied to the QUERY only, so vocabulary that differs from the document
      still matches. Cheap (no re-indexing). Complements -- but does not replace
      -- semantic retrieval.
    - ``stopwords``: ``True``/``"en"``/``"id"``/custom set. Applied to BOTH index
      and query tokens so common function words stop producing spurious matches.
    - ``stemmer``: ``"id"`` (Sastrawi, the ``[indo-nlp]`` extra). Normalizes
      inflected forms to their root on BOTH index and query. ``True`` is NOT
      valid (no English stemmer ships).
    """

    def __init__(
        self,
        chunks: list[Chunk],
        synonyms: dict[str, list[str]] | None = None,
        stopwords: bool | set[str] | frozenset[str] | str | None = None,
        stemmer: str | bool | None = None,
    ):
        self._chunks = chunks
        # Optional lexical-bridge map applied to the QUERY only.
        self._synonyms: dict[str, list[str]] = {k.lower(): v for k, v in (synonyms or {}).items()}
        # Optional stopword filter applied to BOTH index and query tokens.
        self._stopwords: set[str] | None = get_stopwords(stopwords)
        # Optional morphological stemmer applied to BOTH index and query tokens.
        self._stemmer: Callable[[str], str] | None = _normalize_stemmer(stemmer)
        self._tfidf_index: dict[str, dict[str, float]] = {}
        self._idf: dict[str, float] = {}
        self._build_index()

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\w+", text.lower())
        if self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        if self._stemmer:
            tokens = [self._stemmer(t) for t in tokens]
        return tokens

    def _build_index(self) -> None:
        doc_freq: dict[str, int] = {}
        total_docs = len(self._chunks)

        term_counts_per_chunk: list[Counter] = []
        for chunk in self._chunks:
            terms = self._tokenize(chunk.content)
            counts = Counter(terms)
            term_counts_per_chunk.append(counts)
            for term in counts:
                doc_freq[term] = doc_freq.get(term, 0) + 1

        self._idf = {term: math.log((total_docs + 1) / (freq + 1)) + 1 for term, freq in doc_freq.items()}

        for i, counts in enumerate(term_counts_per_chunk):
            chunk_id = self._chunks[i].id
            total_terms = sum(counts.values()) or 1
            self._tfidf_index[chunk_id] = {
                term: (count / total_terms) * self._idf.get(term, 1.0) for term, count in counts.items()
            }

    def _score(self, query_terms: list[str], chunk_id: str) -> float:
        chunk_vec = self._tfidf_index.get(chunk_id, {})
        query_counts = Counter(query_terms)
        total = sum(query_counts.values()) or 1
        query_vec = {term: (count / total) * self._idf.get(term, 1.0) for term, count in query_counts.items()}

        dot = sum(query_vec.get(t, 0) * chunk_vec.get(t, 0) for t in query_vec)
        norm_q = sum(v * v for v in query_vec.values()) ** 0.5
        norm_c = sum(v * v for v in chunk_vec.values()) ** 0.5

        if norm_q == 0 or norm_c == 0:
            return 0.0
        return dot / (norm_q * norm_c)

    async def retrieve(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[RetrievalResult]:
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Expand the query with configured synonyms (query-side only; the index
        # is untouched). Bridges vocabulary gaps like "dog" vs document "pet".
        if self._synonyms:
            expanded = [a for term in query_terms for a in self._synonyms.get(term, [])]
            if expanded:
                query_terms = query_terms + expanded

        allowed = self._allowed_ids(metadata_filter)  # relevance scoping (NOT ACL)
        scored = [
            (chunk, self._score(query_terms, chunk.id))
            for chunk in self._chunks
            if allowed is None or chunk.id in allowed
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(chunk=chunk, score=score, retrieval_method="keyword")
            for chunk, score in scored[:top_k]
            if score > 0
        ]


@register_retriever(
    "bm25",
    description="BM25Okapi lexical retrieval with saturation + document-length normalization",
)
class BM25Retriever(BaseRetriever):
    """BM25Okapi lexical retrieval (saturation + document-length normalization).

    Unlike :class:`KeywordRetriever` (TF-IDF cosine), BM25 saturates term frequency
    (``k1``) and normalizes by document length (``b``) -- the standard lexical
    ranking used by search engines. Accepts the same ``synonyms`` / ``stopwords``
    / ``stemmer`` options as :class:`KeywordRetriever`.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        synonyms: dict[str, list[str]] | None = None,
        stopwords: bool | set[str] | frozenset[str] | str | None = None,
        stemmer: str | bool | None = None,
    ):
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._synonyms = {k.lower(): v for k, v in (synonyms or {}).items()}
        self._stopwords: set[str] | None = get_stopwords(stopwords)
        self._stemmer: Callable[[str], str] | None = _normalize_stemmer(stemmer)
        self._doc_tokens = [self._tokenize(c.content) for c in chunks]
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0
        n_docs = len(chunks)
        doc_freq: dict[str, int] = {}
        self._tf: list[Counter] = []
        for tokens in self._doc_tokens:
            counts = Counter(tokens)
            self._tf.append(counts)
            for term in counts:
                doc_freq[term] = doc_freq.get(term, 0) + 1
        # BM25 idf (the +1 inside the log keeps it non-negative for small corpora).
        self._idf = {term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1) for term, freq in doc_freq.items()}

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"\w+", text.lower())
        if self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        if self._stemmer:
            tokens = [self._stemmer(t) for t in tokens]
        return tokens

    async def retrieve(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[RetrievalResult]:
        query_terms = self._tokenize(query)
        if self._synonyms:
            query_terms = query_terms + [a for t in query_terms for a in self._synonyms.get(t, [])]
        if not query_terms:
            return []

        allowed = self._allowed_ids(metadata_filter)  # relevance scoping (NOT ACL)
        avgdl = self._avgdl or 1.0
        scored: list[tuple[Chunk, float]] = []
        for i, chunk in enumerate(self._chunks):
            if allowed is not None and chunk.id not in allowed:
                continue
            tf = self._tf[i]
            dl = self._doc_len[i] or 1
            score = 0.0
            for term in set(query_terms):
                f = tf.get(term, 0)
                if f == 0 or term not in self._idf:
                    continue
                idf = self._idf[term]
                score += idf * (f * (self._k1 + 1)) / (f + self._k1 * (1 - self._b + self._b * dl / avgdl))
            if score > 0:
                scored.append((chunk, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [RetrievalResult(chunk=chunk, score=score, retrieval_method="bm25") for chunk, score in scored[:top_k]]


# ---------------------------------------------------------------------------
# Embedding-based retrievers (need an EmbeddingClient; degrade to keyword)
# ---------------------------------------------------------------------------


@register_retriever(
    "semantic",
    description="Embedding-based retrieval with keyword fallback",
    inject=["embedder"],
)
class SemanticRetriever(BaseRetriever):
    """Embedding-cosine retrieval with automatic keyword fallback.

    ``embedder`` satisfies :class:`ragi.protocols.EmbeddingClient`
    (``await embedder.embed(text) -> list[float] | None``). The corpus is embedded
    once per process via the shared :data:`_EMBEDDING_CACHE` (keyed by content hash).

    Degrades to :class:`KeywordRetriever` on any of:

    1. No ``embedder`` supplied (``embedder=None``).
    2. The query embedding returned ``None`` (endpoint hiccup mid-query).
    3. The corpus cache build failed (``ok=False`` -- embedder unavailable).

    On fallback, results are stamped ``"semantic (fallback to keyword)"``; a
    successful semantic pass stamps ``"semantic"``. ``synonyms`` is propagated to
    the keyword fallback so vocabulary gaps still close under degradation.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: EmbeddingClient | None = None,
        synonyms: dict[str, list[str]] | None = None,
    ):
        self._chunks = chunks
        self._embedder = embedder
        # Query-side synonym bridge, propagated to every keyword fallback so that
        # when embeddings are unavailable (and semantic degrades to keyword) the
        # bridge still closes vocabulary gaps.
        self._synonyms = synonyms
        self._embedding_available = True
        self._fallback: KeywordRetriever | None = None
        self._chunk_embeddings: dict[str, list[float]] = {}
        self._index_built = False
        # Bounded query-embedding cache (the corpus is cached separately in
        # _EMBEDDING_CACHE). Avoids re-embedding the (often identical) query on
        # every retrieve() call.
        self._query_cache: dict[str, list[float]] = {}
        self._query_cache_size = 256
        self._build_index()

    async def _get_embedding(self, text: str) -> list[float] | None:
        if not self._embedder:
            return None
        cached = self._query_cache.get(text)
        if cached is not None:
            return cached
        result = await self._embedder.embed(text)
        if result is None:
            self._embedding_available = False
            return None
        if len(self._query_cache) >= self._query_cache_size:
            self._query_cache.pop(next(iter(self._query_cache)), None)  # FIFO evict
        self._query_cache[text] = result
        return result

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-10)

    def _build_index(self) -> None:
        if not self._embedder:
            self._embedding_available = False
            self._fallback = KeywordRetriever(self._chunks, synonyms=self._synonyms)
            _logger.warning(
                "SemanticRetriever: no embedder provided -- falling back to keyword retrieval for %d chunks",
                len(self._chunks),
            )
            return

    async def _ensure_index_built(self) -> None:
        if self._index_built:
            return
        if not self._embedder:
            # No embedder: ``_build_index`` already armed the keyword fallback.
            self._index_built = True
            return
        # Shared, process-level index: the corpus is embedded once per process,
        # not per retriever/session. Concurrent first-builds are deduped by the
        # cache's per-signature lock.
        emb_map, ok = await _EMBEDDING_CACHE.get_or_build(self._chunks, self._embedder.embed)
        if not ok:
            self._fallback = KeywordRetriever(self._chunks, synonyms=self._synonyms)
            self._embedding_available = False
            self._index_built = True
            _logger.warning(
                "SemanticRetriever: falling back to keyword retrieval for %d chunks "
                "(the embedder returned no embeddings). Provide an EmbeddingClient "
                "(e.g. OpenAIEmbeddingClient) to enable true semantic retrieval.",
                len(self._chunks),
            )
            return
        self._chunk_embeddings = emb_map
        for c in self._chunks:
            c.embedding = emb_map.get(c.id)
        self._index_built = True

    async def _fallback_retrieve(self, query: str, top_k: int, metadata_filter: dict | None) -> list[RetrievalResult]:
        if self._fallback is None:
            self._fallback = KeywordRetriever(self._chunks, synonyms=self._synonyms)
        results = await self._fallback.retrieve(query, top_k, metadata_filter=metadata_filter)
        for r in results:
            r.retrieval_method = "semantic (fallback to keyword)"
        return results

    async def retrieve(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[RetrievalResult]:
        await self._ensure_index_built()

        if not self._embedding_available:
            return await self._fallback_retrieve(query, top_k, metadata_filter)

        query_emb = await self._get_embedding(query)
        if query_emb is None:
            _logger.warning("SemanticRetriever: query embedding returned None; falling back to keyword retrieval")
            return await self._fallback_retrieve(query, top_k, metadata_filter)

        allowed = self._allowed_ids(metadata_filter)  # relevance scoping (NOT ACL)
        scored = [
            (chunk, self._cosine_similarity(query_emb, emb))
            for chunk, emb in zip(
                self._chunks,
                [self._chunk_embeddings.get(c.id, []) for c in self._chunks],
                strict=False,
            )
            if emb and (allowed is None or chunk.id in allowed)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(chunk=chunk, score=score, retrieval_method="semantic") for chunk, score in scored[:top_k]
        ]


@register_retriever(
    "hybrid",
    description="Reciprocal Rank Fusion of keyword + semantic retrieval",
    inject=["embedder"],
)
class HybridRetriever(BaseRetriever):
    """Combines keyword (TF-IDF) and semantic (embedding) retrieval via RRF.

    Reciprocal Rank Fusion merges two ranked lists by ``score = weight/(rrf_k + rank)``
    (rank 1-indexed) summed per leg, where ``rrf_k=60`` is the standard smoothing
    constant. Each leg over-fetches ``fetch_k = max(top_k*3, 10)`` so fusion has room
    to demote/spromote beyond the final ``top_k``.

    ``synonyms`` propagates to BOTH legs (keyword + the semantic leg's keyword
    fallback), so vocabulary gaps close even when the semantic leg degrades.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: EmbeddingClient | None = None,
        rrf_k: int = 60,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
        synonyms: dict[str, list[str]] | None = None,
    ):
        self._chunks = chunks
        self._rrf_k = rrf_k
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight
        # Propagate the query-side synonym bridge to the keyword leg so that
        # vocabulary gaps (e.g. user "dog" vs document "pet") are closed even
        # when embeddings are unavailable and the semantic leg falls back.
        self._keyword = KeywordRetriever(chunks, synonyms=synonyms)
        self._semantic = SemanticRetriever(chunks, embedder=embedder, synonyms=synonyms)

    async def retrieve(self, query: str, top_k: int = 3, metadata_filter: dict | None = None) -> list[RetrievalResult]:
        # Get results from both retrievers (fetch more than top_k for fusion).
        fetch_k = max(top_k * 3, 10)
        kw_results = await self._keyword.retrieve(query, top_k=fetch_k, metadata_filter=metadata_filter)
        sem_results = await self._semantic.retrieve(query, top_k=fetch_k, metadata_filter=metadata_filter)

        # Build RRF scores: score = weight / (rrf_k + rank), rank 1-indexed.
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, r in enumerate(kw_results, start=1):
            cid = r.chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self._keyword_weight / (self._rrf_k + rank)
            chunk_map[cid] = r.chunk

        for rank, r in enumerate(sem_results, start=1):
            cid = r.chunk.id
            rrf_scores[cid] = rrf_scores.get(cid, 0) + self._semantic_weight / (self._rrf_k + rank)
            chunk_map[cid] = r.chunk

        # Sort by combined RRF score.
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        return [
            RetrievalResult(
                chunk=chunk_map[cid],
                score=score,
                retrieval_method="hybrid",
            )
            for cid, score in sorted_items[:top_k]
        ]
