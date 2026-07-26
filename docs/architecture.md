# rag-workbench architecture

**Eval-first, agent-agnostic RAG toolkit.** The closed loop:

```
define -> measure (offline, reproducible) -> iterate -> ship as a portable retrieval contract
```

A tuned retrieval stack is measured against a golden IR set with bootstrap CIs, iterated via
A/B compare, then shipped as a Python lib / MCP server / tool-schema / framework adapter that
**any** agent consumes. Mode A (retrieval-only) runs with **no API key**; Mode B (faithfulness)
needs a chat client.

## The 10-stage pipeline

```
ingest/source -> parse -> chunk -> (embed) -> index -> retrieve -> rerank -> rewrite/HyDE
                                                                     |
                                                          augmentation (agent seam)
                                                                     |
                                                          eval (Mode A/B + 9-dim rubric)
                                                                     |
                                                          export (MCP / tool-schema / adapters)
```

Each stage is a swappable, registered component behind a Protocol.

## Decoupling: Protocols, not inheritance

The lib has **zero `koboi.*` imports** — it's a clean extraction. Framework objects adapt
*without inheritance* via `typing.Protocol`:

- `Retriever.retrieve(query, *, top_k, metadata_filter)` — the composed retrieval stage.
- `EmbeddingClient.embed/embed_batch` + `ChatClient.complete` — split from a monolithic LLM
  client so a **lexical-only (BM25) stack needs zero LLM clients** (Mode A runs API-key-free).
- `VectorStore.add/search/count` — pluggable backends (memory/faiss/chroma/pgvector).
- `Reranker` / `Scorer` / `Chunker` / `Parser`.

## Registry + `build_pipeline`

`ComponentRegistry` (`registry.py`) introspects a class `__init__` to map YAML config keys to
kwargs (+ `config_aliases`), and an `inject` list wires named deps (e.g. `inject=["embedder"]`
for semantic retrievers). `@register_chunker/retriever/augmentation/parser` decorators
self-register on import; `register_builtins()` loads them. `build_pipeline(conf)` composes
chunker → retriever → (rerank wrapper) → (rewrite wrapper) from config.

**Wrappers are measurable stages**: `CrossEncoderReranker` + `RewritingRetriever` are
`Retriever`s, so the eval runner measures each stage's lift directly (`compare bm25 vs
bm25+rerank`). Augmentation is a separate agent seam (not auto-composed).

## Eval: Mode A (retrieval) + Mode B (answer)

- **Mode A** (`StandaloneEvalRunner`): drives the `Retriever` directly → IR metrics
  (recall@k / MRR / nDCG / precision / hit) vs golden qrels → seedable bootstrap CI. Zero key.
- **Mode B** (`EndToEndEvalRunner`): retrieve → generate via `ChatClient` → judge
  **faithfulness** (NLI claim-decomposition, not RAGAS) + answer-correctness + abstention.
- **9-dim rubric**: faithfulness 0.18 > ranking 0.17 > correctness 0.13 > … ; a dimension
  **FAILS if its CI lower bound < threshold** (no false greens); infra dims (ingestion/
  robustness/perf) stay NA.
- **Golden sets**: MS MARCO (EN), TyDi-id native (Indonesian, Apache-2.0), bring-your-own.

## Export: ship anywhere

- **MCP server** (`export/mcp.py`, FastMCP): read-only `search` tool over stdio (Claude
  Desktop/Cursor) or Streamable HTTP. SAFE-by-construction.
- **Tool-schema** (`export/toolschema.py`): MCP/OpenAI/Anthropic formats (no deps).
- **Adapters**: LangChain/LlamaIndex `BaseRetriever` + OpenAI/Claude SDK tool-call dispatch.

## Production posture

- **Fail-soft** everywhere (rerank outages → `rerank:failed(...)`, never silent).
- **SSRF guard** + **3-layer size cap** on remote sources.
- **CI gates** (`.github/workflows/ci.yml`): ruff/format (HARD) + pytest-cov (ratcheting floor) +
  bandit + pip-audit (HARD) + mypy (SOFT). See `CONTRIBUTING.md`.
