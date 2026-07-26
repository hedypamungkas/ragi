"""ragworkbench/protocols -- framework-agnostic contracts.

These are ``typing.Protocol`` (``@runtime_checkable``) so any framework's objects can
be adapted WITHOUT inheritance -- a LangChain ``Retriever``, a LlamaIndex
``BaseRetriever``, or a raw function can all satisfy ``Retriever``.

Design note: koboi conflates embedding + chat in one ``LLMClient`` (koboi/llm/base.py),
which forces every retriever to carry a chat-capable client even when it only embeds.
We split into two single-purpose protocols so a **lexical-only (keyword/BM25) stack
needs ZERO LLM clients** -- the v0.1 closed loop runs with no API key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ragworkbench.types import Chunk, Document, RetrievalResult

if TYPE_CHECKING:
    # Forward-only import: keeps the Protocols layer free of an eval dependency while
    # still resolving the ``Scorer.score`` annotation for type-checkers / ruff.
    from ragworkbench.eval.types import EvalCase, EvalScore


@runtime_checkable
class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class Parser(Protocol):
    def extract(self, name: str, data: bytes) -> tuple[str, dict]: ...


@runtime_checkable
class Retriever(Protocol):
    """Composed retrieval stage (lexical/semantic/hybrid, optionally rerank-wrapped)."""

    async def retrieve(
        self, query: str, *, top_k: int = 5, metadata_filter: dict | None = None
    ) -> list[RetrievalResult]: ...


@runtime_checkable
class EmbeddingClient(Protocol):
    """Embed text. Return None to signal unavailability (fail-soft -> lexical fallback)."""

    async def embed(self, text: str) -> list[float] | None: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float] | None]: ...


@runtime_checkable
class ChatClient(Protocol):
    """Single-turn text completion (used by rewrite / HyDE / LLM-judge / faithfulness)."""

    async def complete(self, messages: list[dict]) -> str: ...


@runtime_checkable
class VectorStore(Protocol):
    """Pluggable backend. v0.1 ships ``InMemoryVectorStore``; external backends arrive v0.3+."""

    async def add(self, chunks: list[Chunk]) -> None: ...

    async def search(
        self, query: str | list[float], *, top_k: int, filter: dict | None = None
    ) -> list[RetrievalResult]: ...

    @property
    def count(self) -> int: ...


@runtime_checkable
class Reranker(Protocol):
    provider: str

    async def rerank(self, query: str, documents: list[str], *, top_n: int) -> list[tuple[int, float]] | None: ...


@runtime_checkable
class Scorer(Protocol):
    name: str

    async def score(self, case: EvalCase, output: str, context: dict) -> EvalScore: ...
