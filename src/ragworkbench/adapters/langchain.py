"""ragworkbench/adapters/langchain -- LangChain ``BaseRetriever`` adapter.

Adapts a ragworkbench :class:`Retriever` as a LangChain
:class:`langchain_core.retrievers.BaseRetriever` so it drops into any LangChain
chain/agent. Import-gated behind the ``[adapters-langchain]`` extra.

Validated against langchain-core 1.4.9 (still the v0 ``_get_relevant_documents`` contract).
Both sync (``invoke``) and async (``ainvoke``) paths are implemented.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ragworkbench.errors import LLMInvalidRequestError

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except ImportError as e:
    raise LLMInvalidRequestError(
        "(adapters-langchain) langchain-core required: pip install 'ragworkbench[adapters-langchain]'"
    ) from e

_logger = logging.getLogger(__name__)


def _run_sync(coro):
    """Run an async retriever call from sync code (fails clearly if a loop is running)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "LangChainRetrieverAdapter.invoke() was called from a running event loop; "
        "use ainvoke() (the async path) instead."
    )


class LangChainRetrieverAdapter(BaseRetriever):
    """A ragworkbench Retriever exposed as a LangChain BaseRetriever."""

    rwb_retriever: Any
    top_k: int = 5

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        results = _run_sync(self.rwb_retriever.retrieve(query, top_k=self.top_k))
        return [self._to_document(r) for r in results]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = await self.rwb_retriever.retrieve(query, top_k=self.top_k)
        return [self._to_document(r) for r in results]

    @staticmethod
    def _to_document(r) -> Document:
        meta = dict(r.chunk.metadata)
        meta.update({"doc_id": r.chunk.doc_id, "score": r.score, "retrieval_method": r.retrieval_method})
        return Document(page_content=r.chunk.content, metadata=meta)
