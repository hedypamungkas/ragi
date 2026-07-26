# Changelog

All notable changes to rag-workbench.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0: minor may include breaking changes).

## [0.3.0] - 2026-07-26

### Added — ship as a portable retrieval contract
- **MCP server** (`export/mcp.py`, `[mcp]` extra): expose a tuned Retriever as a read-only
  `search` tool via the official `mcp` SDK (FastMCP) — stdio (Claude Desktop/Cursor) +
  Streamable HTTP. SAFE-by-construction (only a read tool → no risk gate). `ragwb serve
  <config> --transport stdio|http`. Returns deterministic markdown with `[Source:]` citations.
- **Tool-schema generator** (`export/toolschema.py`, no deps): `search_tool_schema()` +
  `to_mcp` / `to_openai` / `to_anthropic` → wire the `search` tool into OpenAI function-calling
  or Claude `tool_use` directly. `ragwb export-tool-schema --format ...`.
- **Framework adapters**: `adapters/langchain.py` (`LangChainRetrieverAdapter`,
  `[adapters-langchain]`, sync + async) + `adapters/llamaindex.py` (`[adapters-llamaindex]`,
  import-gated).
- +9 tests (75 total, ruff clean): toolschema shape (no dep), a **real `mcp` client→server
  stdio round-trip** (protocol-compatible with Claude Desktop by construction), langchain
  adapter invoke/ainvoke, llamaindex import-gate.

### Notes
- `mcp` SDK validated at 1.28.1; langchain-core 1.4.9 still uses the v0
  `_get_relevant_documents` contract (both sync + async paths implemented).
- `ragwb serve` does not wire an embedder (v0.3) → semantic/hybrid configs degrade to keyword
  over MCP; lexical + rerank configs serve directly. HTTP transport is localhost/no-auth
  (production remote needs a Bearer reverse-proxy, documented).
- llama-index is heavy and not installed in CI → the LlamaIndex adapter is import-gate-tested
  only; validate with `pip install 'ragworkbench[adapters-llamaindex]'` when adopting.

## [0.2.0] - 2026-07-26

### Added — "beats the baseline" stages
- **Embedding-based retrieval**: `SemanticRetriever` (cosine) + `HybridRetriever` (RRF k=60) via
  an `EmbeddingClient` Protocol (split from koboi's monolithic `LLMClient`). Degrade-to-keyword
  when no embedder / embed returns None.
- **`embeddings/`**: `OpenAIEmbeddingClient` (httpx via `_internal/http`, fail-soft), the lifted
  `_EmbeddingIndexCache` (content-hash process singleton + JSON disk persistence), and a
  `MockEmbeddingClient` (bag-of-words) for offline tests.
- **Cross-encoder rerank** (`retrieval/rerank.py`): `JinaRerankBackend`/`CohereRerankBackend`
  (HTTP) + `LocalBGERerankBackend` (`[rerank-local]`) + `CrossEncoderReranker` wrapper (over-fetch
  × `fetch_multiplier`, rescore, **fail-soft** → `rerank:failed(...)` stamp on outage; success
  stamps `rerank:<provider>(<base>)`). `build_rerank_client` with `JINA_API_KEY`/`COHERE_API_KEY`
  env-var fallback.
- **Query rewrite / HyDE** (`retrieval/rewrite.py`): `QueryRewriter` (rule + LLM + HyDE via
  `ChatClient`) + a new `RewritingRetriever` **Retriever wrapper** (a measurable stage — koboi
  buried rewrite inside the augmentation).
- **Augmentation seam** (`retrieval/augmentation.py`): `AugmentationStrategy` (the 2-method agent
  boundary) + `InMemory`/`OnTheFly` with `[i] [Source: x]` citations + `ABSTENTION_MARKER`.
  Rewrite removed from here (moved to the wrapper) to avoid double-rewrite.
- **`_internal/http.py`**: inlined `HttpTransport` (POST + retry/backoff) + `BearerAuth` (~95 LOC).
- **`build_pipeline` composition**: base retriever → `RewritingRetriever` (outside) →
  `CrossEncoderReranker` (after). Both wrappers are `Retriever`s, so the eval runner measures each
  stage's lift directly.
- New configs: `bm25_rerank.yaml` (jina), `bm25_rerank_local.yaml` (BGE, no key), `semantic.yaml`,
  `hybrid_rerank.yaml`. +26 tests (66 total, ruff clean).

### Notes
- Augmentation is **not** auto-composed by `build_pipeline` (eval measures retrieval; the agent
  seam is for agent consumers). Wrap the returned retriever in an `AugmentationStrategy`.
- The hosted **~0.977** gate needs a `JINA_API_KEY`; the local BGE gate needs `[rerank-local]`.
  In-session verification is mock-based (no key, no model download).

## [0.1.0] - 2026-07-26

### Added
- Core types (`Chunk` / `Document` / `RetrievalResult`) + framework-agnostic Protocols
  (`Retriever`, `Chunker`, `Parser`, `EmbeddingClient`, `ChatClient`, `VectorStore`,
  `Reranker`, `Scorer`).
- `ComponentRegistry` + `@register_*` decorators + `build_pipeline()` (lifted & decoupled
  from koboi/rag/registry.py; zero `koboi.*` imports).
- Ingest: text/html/pdf/docx parsers, paragraph/sentence/fixed chunkers, Mongo-style
  metadata filters, file/HTTP sources with SSRF guard.
- Retrieval: `KeywordRetriever` (TF-IDF cosine) + `BM25Retriever` (BM25Okapi, k1/b),
  with optional synonyms / stopwords (en|id) / Indonesian stemmer (Sastrawi).
- Eval (the wedge): pure-stdlib IR metrics (`recall@_k` / `precision_at_k` / `hit_rate`
  / `mrr` / `ndcg_at_k`), citation precision, seedable bootstrap CI, a `ScorerRegistry`,
  and a `StandaloneEvalRunner` that drives a `Retriever` directly (Mode A — no LLM, no
  API key). Plus A/B stack comparison with paired bootstrap CI and a 9-dimension
  production-readiness rubric (CI-lower-bound gate).
- Golden sets: MS MARCO (EN) corpus builder + bring-your-own JSON loader.
- Workbench CLI: `ragwb ingest | eval | compare`.

### Notes
- SemanticChunker deliberately **not** lifted (known-broken upstream); reimplement in v0.4.
- Semantic/Hybrid retrievers, rerank, rewrite, augmentation seam, MCP export, and
  framework adapters are v0.2–v0.5.
