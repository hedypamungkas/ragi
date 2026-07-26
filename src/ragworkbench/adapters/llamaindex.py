"""ragworkbench/adapters/llamaindex -- LlamaIndex ``BaseRetriever`` adapter.

Adapts a ragworkbench :class:`Retriever` as a LlamaIndex
:class:`llama_index.core.retrievers.BaseRetriever` (returns ``NodeWithScore``). Import-gated
behind the ``[adapters-llamaindex]`` extra.

NOTE: llama-index is heavy and NOT installed in the test env, so this adapter is written
against the documented ``_retrieve(query_bundle)`` / ``NodeWithScore`` / ``TextNode`` /
``QueryBundle`` API but is **not live-tested in CI** -- validate it with
``pip install 'ragworkbench[adapters-llamaindex]'`` when you adopt it. The import-gate itself
IS tested (constructing without the extra raises a clear error).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ragworkbench.errors import LLMInvalidRequestError

try:
    from llama_index.core import QueryBundle  # type: ignore[import-not-found]
    from llama_index.core.retrievers import BaseRetriever  # type: ignore[import-not-found]
    from llama_index.core.schema import NodeWithScore, TextNode  # type: ignore[import-not-found]
except ImportError as e:
    raise LLMInvalidRequestError(
        "(adapters-llamaindex) llama-index required: pip install 'ragworkbench[adapters-llamaindex]'"
    ) from e

_logger = logging.getLogger(__name__)


def _run_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "LlamaIndexRetrieverAdapter sync retrieve() was called from a running event loop; "
        "use the async path (artrieve) instead."
    )


class LlamaIndexRetrieverAdapter(BaseRetriever):
    """A ragworkbench Retriever exposed as a LlamaIndex BaseRetriever."""

    rwb_retriever: Any = None
    top_k: int = 5

    def _retrieve(self, query_bundle: QueryBundle, **kwargs: Any):  # noqa: ARG002
        query = getattr(query_bundle, "query_str", str(query_bundle))
        results = _run_sync(self.rwb_retriever.retrieve(query, top_k=self.top_k))
        nodes: list[NodeWithScore] = []
        for r in results:
            node = TextNode(text=r.chunk.content, id_=r.chunk.doc_id, metadata=dict(r.chunk.metadata))
            nodes.append(NodeWithScore(node=node, score=float(r.score)))
        return nodes

    async def _aretrieve(self, query_bundle: QueryBundle, **kwargs: Any):  # noqa: ARG002
        query = getattr(query_bundle, "query_str", str(query_bundle))
        results = await self.rwb_retriever.retrieve(query, top_k=self.top_k)
        return [
            NodeWithScore(
                node=TextNode(text=r.chunk.content, id_=r.chunk.doc_id, metadata=dict(r.chunk.metadata)),
                score=float(r.score),
            )
            for r in results
        ]
