"""ragworkbench/_internal -- inlined decoupling primitives (not public API).

``http.py`` inlines the minimal HttpTransport + BearerAuth surface the rerank backends
and embedding client need, so the lib has zero dependency on any provider SDK.
"""
