# ragi

[![CI](https://github.com/hedypamungkas/ragi/actions/workflows/ci.yml/badge.svg)](https://github.com/hedypamungkas/ragi/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

> ***Retrieval that rises.*** — named for the **sourdough starter** (*ragi*, Indonesian): a
> living culture you **cultivate**, **measure before you bake**, and **share**. Tend your RAG the
> same way — feed it data, measure it honestly, then ship the tuned stack to any agent.

**Eval-first, agent-agnostic RAG toolkit.** Define a retrieval stack, measure it on a
real IR golden set (offline, reproducible), iterate with statistical confidence, then
ship the tuned stack as a **portable retrieval contract** — a Python library, an MCP
server, or a tool-schema — that any agent can consume (koboi, LangChain, LlamaIndex,
OpenAI / Claude SDK, v0, …).

> *define → measure → iterate → ship*

## Why

Every existing RAG library is **build-first, eval-as-afterthought** and locks you into
its framework. ragi closes the loop the other way: eval is the spine, the stack
is portable, and a **lexical-only baseline runs the entire closed loop with zero API
key** — measure your retrieval quality for free *before* paying for embeddings.

Reproducible baselines on a real MS MARCO golden set (built locally via one script;
HF-cached, license-light): BM25 recall@10 ≈ **0.82–0.90** lexical, climbing to **~0.98**
with a jina cross-encoder rerank (that step needs a `JINA_API_KEY`). Numbers you reproduce
yourself with `ragi eval` — measured, not vibes.

## Install

```bash
pip install ragi-toolkit                  # core, from PyPI  ->  import ragi
pip install "ragi-toolkit[mcp,parsers]"   # + MCP server + text/html/pdf/docx parsers
```

From source (contributors):

```bash
pip install -e ".[dev,parsers]"           # editable + tests + linters
```

## Quickstart (v0.1 closed loop — no API key)

```bash
# 1. build the MS MARCO golden corpus (HF-cached after first run; license-light)
python scripts/build_ir_corpus.py

# 2. measure a BM25 stack (retrieval-only eval, deterministic, zero cost)
ragi eval configs/bm25_baseline.yaml --dataset msmarco --n 120

# 3. A/B compare two stacks with a paired bootstrap CI on the difference
ragi compare configs/bm25.yaml configs/bm25_stopwords.yaml --metric recall@10
```

As a library:

```python
import ragi
ragi.register_builtins()
retriever = ragi.build_pipeline({
    "enabled": True,
    "chunker": "paragraph",
    "retriever": "bm25",
    "documents": [{"path": "data/ir_corpus"}],
})
results = await retriever.retrieve("what is photosynthesis?", top_k=5)
```

## Status

**v1.0 — productionized (CI-gated).** The feature arc (v0.1–v0.5) is complete; v1.0 adds the
trust layer. See [CONTRIBUTING.md](CONTRIBUTING.md) (local gates) +
[docs/architecture.md](docs/architecture.md) (design). Phased roadmap:

| Slice | Ships |
|---|---|
| **v0.1** ✅ | lexical retrieval (keyword/BM25) + retrieval-only eval (recall@k / MRR / nDCG / precision + bootstrap CI) + A/B compare + 9-dim rubric |
| **v0.2** ✅ | semantic/hybrid + cross-encoder rerank (jina/cohere/local) + query-rewrite/HyDE + augmentation seam — all as composable Retriever wrappers |
| **v0.3** ✅ | MCP server export (read-only) + tool-schema (MCP/OpenAI/Anthropic) + LangChain/LlamaIndex adapters |
| **v0.4** ✅ | Mode B end-to-end eval (faithfulness NLI) + full 9-dim rubric + TyDi-id native baseline |
| **v0.5** ✅ | VectorStore (memory/faiss/chroma/pgvector) + SemanticChunker + s3/firecrawl + OpenAI/Claude SDK adapters — **feature-complete** |

## Ship to any agent (v0.3)

A tuned stack becomes a portable retrieval contract three ways:

**1. MCP server** (Claude Desktop / Cursor / any MCP client) — one read-only `search` tool:

```bash
pip install "ragi-toolkit[mcp]"
ragi serve configs/bm25_baseline.yaml --transport stdio
```

Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ragi": {
      "command": "ragi",
      "args": ["serve", "/abs/path/to/configs/bm25_baseline.yaml", "--transport", "stdio"]
    }
  }
}
```

**2. Tool-schema** (OpenAI function-calling / Claude `tool_use`):

```bash
ragi export-tool-schema --format openai      # or: mcp | anthropic | base
```

**3. Framework adapters** (LangChain / LlamaIndex):

```python
# pip install "ragi-toolkit[adapters-langchain]"
from ragi.adapters.langchain import LangChainRetrieverAdapter
lc = LangChainRetrieverAdapter(ragi_retriever=retriever, top_k=5)
docs = lc.invoke("your query")   # -> list[langchain_core.documents.Document]
```

## Measure answer quality (v0.4 — Mode B)

Mode A measures **retrieval**; Mode B measures the **answer** — is it faithful (grounded, not
hallucinated), correct, and does it abstain on out-of-scope queries?

```bash
# needs OPENAI_API_KEY (Mode B generates + judges answers via a chat model)
OPENAI_API_KEY=... ragi eval configs/bm25_rerank.yaml --dataset msmarco --mode end_to_end -n 120
```

Faithfulness uses **NLI claim-decomposition** (decompose the answer into atomic claims, NLI-check
each vs the retrieved context → coverage ratio) — not RAGAS, which stalls on OpenAI-compatible
gateways. Production targets: faithfulness ≥ 0.8, answer-correctness ≥ 0.75. The full **9-dimension
rubric** weights faithfulness highest (0.18) > ranking (0.17) > correctness (0.13) > …

**TyDi-id native Indonesian baseline** (`--dataset tydi-id`, Mode A needs no key): natively-collected
(not machine-translated) — BM25 recall@10 ≈ 0.97. Closes the translation-inflation caveat; the
SEA-aware differentiator.

## Scale + completeness (v0.5 — feature-complete)

**VectorStore backends** — the scale path for >100k-chunk corpora (`SemanticRetriever` stays the
in-memory-cosine path for smaller ones):

```python
ragi.build_pipeline({
    "retriever": "vectorstore",
    "vectorstore": {"backend": "faiss"},   # or chroma / pgvector / memory
    "documents": [{"path": "data/corpus"}],
}, embedder=OpenAIEmbeddingClient(api_key=...))
```

`memory` is always available; `faiss` (`[vector-faiss]`), `chroma` (`[vectorstore-chroma]`), and
`pgvector` (`[vectorstore-pgvector]`) are import-gated. **SemanticChunker** (embedding-aware greedy
merge), **s3/firecrawl** sources (`[rag-cloud]`), and **OpenAI/Claude SDK tool-call adapters**
(`adapters/openai.py` + `adapters/anthropic.py` — for non-MCP SDK agents) round out the stack.

v0.5 is the **capstone** — all 5 roadmap slices shipped. **v1.0** then added the trust layer:
CI gates (ruff / format / coverage / bandit / pip-audit), packaging, and dev docs. See
[CHANGELOG.md](CHANGELOG.md).

## Design

- **Framework-agnostic by Protocol, not inheritance** — adapt any framework's objects.
- **Eval-first** — a `StandaloneEvalRunner` drives the retriever directly (isolating
  retrieval quality from LLM variance), not an agent loop.
- **Fail-safe / fail-soft** — unknown components warn-and-fallback; rerank outages
  return base results, never crash a run.

## License

Apache-2.0.
