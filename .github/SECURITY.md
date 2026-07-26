# Security Policy

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities. Instead, use
**GitHub's private vulnerability reporting** (the *"Report a vulnerability"* button on the
Security tab) so the report stays private until a fix is coordinated.

## Response SLA

- **Acknowledgement**: within **3 business days**.
- **Fix or mitigation**: within **30 days** for high-severity issues (sooner for critical).

## Scope

This policy covers the `rag-workbench` PyPI package and its source on the `main` branch. The
optional external backends (faiss/chroma/pgvector) and the MCP/SDK adapters are in scope as
shipped code; third-party dependencies are not (report upstream).

## Supported versions

Only the latest minor release line receives security fixes.

## Hardening notes (already in place)

- **SSRF guard** on all remote document sources (`ingest/sources.py` `_check_url_ssrf`) —
  blocks loopback/private/link-local IPs.
- **Read-only MCP surface** — the MCP server exposes only a `search` tool (SAFE-by-construction,
  no write/mutate over MCP).
- **3-layer size cap** on s3/http fetches (CWE-400 defense).
- **Fail-soft** retrieval/rerank/eval — provider hiccups never crash a run; rerank outages are
  **observably** stamped (`rerank:failed(...)`), never silent.
