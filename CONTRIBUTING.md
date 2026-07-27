# Contributing to ragi

Eval-first, agent-agnostic RAG toolkit. Contributions welcome — please run the gates locally
before opening a PR.

## Local gates (run all 5 before pushing)

```bash
pip install -e ".[dev]"
ruff check src/ tests/ scripts/          # HARD
ruff format --check src/ tests/ scripts/ # HARD
pytest --cov=ragi --cov-fail-under=90   # HARD (ratcheting floor; see pyproject)
bandit -r src/ -c pyproject.toml         # HARD (no HIGH/CRITICAL)
mypy src/ragi                    # SOFT (Protocol-heavy; promote later)
```

Coverage is a **ratcheting floor** (~95% today, gated at 90%): don't lower it, raise it as you
add tests. Optional backends (`vectorstore/{faiss,chroma,pgvector}.py`, `adapters/{llamaindex,langchain}.py`)
are `omit`-ed from coverage (import-gated; their deps aren't installed in CI).

## Extending the toolkit

The lib is built on a **registry + Protocol** pattern. To add a component, subclass the ABC /
satisfy the Protocol and register it:

**Add a chunker** (`ingest/chunker.py`):
```python
from ragi.ingest.chunker import BaseChunker
from ragi.registry import register_chunker

@register_chunker("my_chunker", description="...")
class MyChunker(BaseChunker):
    def chunk(self, document): ...   # -> list[Chunk]
```

**Add a retriever** (`retrieval/`):
```python
from ragi.retrieval.retriever import BaseRetriever
from ragi.registry import register_retriever

@register_retriever("my_retriever", inject=["embedder"])  # list deps to auto-inject
class MyRetriever(BaseRetriever):
    async def retrieve(self, query, *, top_k=5, metadata_filter=None): ...  # -> list[RetrievalResult]
```

**Add a VectorStore backend** (`vectorstore/`): implement the `VectorStore` Protocol
(`add`/`search`/`count`), import-gate the dep, register in `build_store()`.

**Add a framework adapter** (`adapters/`): wrap the ragi `Retriever` as the framework's
retriever type; import-gate the framework behind its `[adapters-*]` extra.

**Add a scorer** (`eval/`): subclass `BaseScorer` (`async score(case, output, context) -> EvalScore`),
register in `eval/registry.py`.

Components self-register on import (module-level `@register_*`); `ragi.register_builtins()`
imports the builtins. YAML-driven plugin loading via `custom_modules: [...]`.

## Conventions

- `from __future__ import annotations`; modern unions (`list[Chunk] | None`); stdlib logging.
- Module docstring: `"""ragi/<path> -- short description."""`.
- **Zero `koboi.*` imports** (the lib is a decoupled extraction). Couple only through Protocols.
- Fail-soft: optional components warn-and-fallback; never crash a retrieval/eval run.
- Tests: pytest, `asyncio_mode="auto"`; mock providers (`MockEmbeddingClient`/`MockChatClient`/
  `MockReranker`) keep the suite **zero-API-key**.

## Branching

Commit on a feature branch off `main`; the CI gates (lint/test/security) must be green before merge.
