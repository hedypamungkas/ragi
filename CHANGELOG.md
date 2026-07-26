# Changelog

All notable changes to rag-workbench.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-1.0: minor may include breaking changes).

## [0.4.0] - 2026-07-26

### Added — production-readiness depth (Mode B + full 9-dim rubric + TyDi-id)
- **Mode B end-to-end eval** (`EndToEndEvalRunner`): retrieve → generate via `ChatClient` → judge.
  Answer-quality scorers:
  - `eval/faithfulness.py` — `FaithfulnessScorer` via **NLI claim-decomposition** (decompose →
    batch-NLI → coverage), adapted from koboi's `GroundingGuardrail` — **not RAGAS** (the doc
    indicts RAGAS for gateway multi-gen stalling). 2 side-LLM calls/answer; fail-soft.
  - `eval/generation_scorers.py` — `AnswerCorrectnessScorer` (judge + deterministic substring
    fallback) + `AbstentionScorer` (OOS refusal detection).
  - `eval/mock_chat.py` — `MockChatClient` for zero-key Mode B tests.
- **Full 9-dimension rubric** (`eval/rubric.py`): faithfulness 0.18 / ranking 0.17
  (recall+mrr+ndcg) / correctness 0.13 / abstention 0.09 / noise 0.09 + infra dims
  (ingestion/robustness/perf) NA + confidence (min_n). Weights from the production-readiness doc.
- **TyDi-id native Indonesian baseline** (`scripts/build_id_native_corpus.py` + `resolve_dataset("tydi-id")`):
  natively-collected (NOT translated) — closes the translation-inflation caveat. Apache-2.0 qrels.
- **`chat/openai.py`** — `OpenAIChatClient` (OpenAI-compatible gateway) for Mode B. CLI
  `ragwb eval --mode retrieval|end_to_end` (+ `--chat-model`).
- +22 tests (97 total, ruff clean): faithfulness claim-decomp + normalization + fail-soft,
  correctness judge/fallback, abstention, end-to-end Mode B over synthetic, 9-dim rubric weights,
  TyDi-id loader.

### Notes
- Mode B needs `OPENAI_API_KEY` (live); mock-based tests cover the wiring. Documented:
  `OPENAI_API_KEY=... ragwb eval configs/bm25_rerank.yaml --dataset msmarco --mode end_to_end -n 120`
  → full 9-dim report with live faithfulness/correctness.
- Native-ID BM25 recall@10 = **0.967** [0.867, 1.000] on TyDi-id (n=30) — the caveat-free ID baseline.
- Noise/ingestion/robustness/perf dims stay NA (need a noise fixture / infra harness — fast-follow).
- faiss + SemanticChunker deferred to v0.5 (backend-breadth slice).

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
