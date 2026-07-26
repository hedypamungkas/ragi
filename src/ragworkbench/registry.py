"""ragworkbench/registry -- decorator-based component registry + build_pipeline().

Lifted from koboi/rag/registry.py and decoupled: zero ``koboi.*`` imports. The
``ComponentRegistry`` introspects a class ``__init__`` to map YAML config keys to
kwargs (with ``config_aliases``), and an ``inject`` list wires named dependencies
(e.g. ``inject=["embedder"]`` for semantic retrievers). ~200 LOC of pure stdlib.

``build_pipeline()`` composes chunker + retriever from config. v0.1 composes lexical
retrievers only; v0.2 will layer rerank + augmentation + rewrite on top.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ragworkbench.ingest.chunker import BaseChunker
    from ragworkbench.protocols import ChatClient, EmbeddingClient
    from ragworkbench.retrieval.retriever import BaseRetriever
    from ragworkbench.types import Chunk

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic component registry
# ---------------------------------------------------------------------------


class ComponentEntry:
    """Metadata for a registered component."""

    __slots__ = ("cls", "parameters", "description", "config_aliases", "inject")

    def __init__(
        self,
        cls: type,
        parameters: dict[str, dict[str, Any]],
        description: str = "",
        config_aliases: dict[str, str] | None = None,
        inject: list[str] | None = None,
    ):
        self.cls = cls
        self.parameters = parameters
        self.description = description
        self.config_aliases = config_aliases or {}
        self.inject = inject or []


class ComponentRegistry:
    """Generic registry for swappable components (chunkers, retrievers, parsers, ...)."""

    def __init__(self, component_type: str) -> None:
        self._component_type = component_type
        self._entries: dict[str, ComponentEntry] = {}

    def register(
        self,
        name: str,
        cls: type,
        *,
        description: str = "",
        config_aliases: dict[str, str] | None = None,
        inject: list[str] | None = None,
    ) -> None:
        params = _extract_parameters(cls)
        if config_aliases:
            valid_params = set(params.keys())
            for yaml_key, param_name in config_aliases.items():
                if param_name not in valid_params:
                    raise ValueError(
                        f"config_aliases maps '{yaml_key}' to '{param_name}', "
                        f"but {cls.__name__}.__init__ has no such parameter. "
                        f"Available: {sorted(valid_params)}"
                    )
        self._entries[name] = ComponentEntry(
            cls=cls,
            parameters=params,
            description=description,
            config_aliases=config_aliases,
            inject=inject,
        )

    def get(self, name: str) -> ComponentEntry | None:
        return self._entries.get(name)

    def list_available(self) -> list[str]:
        return sorted(self._entries.keys())

    def clear(self) -> None:
        self._entries.clear()


def _extract_parameters(cls: type) -> dict[str, dict[str, Any]]:
    """Extract constructor parameters via introspection.

    Returns ``{param_name: {"default": ..., "annotation": ...}}``. Skips ``self``
    and ``*args``/``**kwargs``.
    """
    sig = inspect.signature(cls.__init__)  # type: ignore[misc]
    params: dict[str, dict[str, Any]] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        entry: dict[str, Any] = {}
        if param.default is not inspect.Parameter.empty:
            entry["default"] = param.default
        if param.annotation is not inspect.Parameter.empty:
            entry["annotation"] = param.annotation
        params[name] = entry
    return params


# ---------------------------------------------------------------------------
# Module-level registries
# ---------------------------------------------------------------------------

chunker_registry = ComponentRegistry("chunker")
retriever_registry = ComponentRegistry("retriever")
augmentation_registry = ComponentRegistry("augmentation")  # v0.2 seam (unused in v0.1 build)
parser_registry = ComponentRegistry("parser")


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def register_chunker(name: str, description: str = "", *, config_aliases: dict[str, str] | None = None):
    def decorator(cls: type) -> type:
        chunker_registry.register(name, cls, description=description, config_aliases=config_aliases)
        return cls

    return decorator


def register_retriever(name: str, description: str = "", *, inject: list[str] | None = None):
    def decorator(cls: type) -> type:
        retriever_registry.register(name, cls, description=description, inject=inject)
        return cls

    return decorator


def register_augmentation(name: str, description: str = ""):
    def decorator(cls: type) -> type:
        augmentation_registry.register(name, cls, description=description)
        return cls

    return decorator


def register_parser(name: str, description: str = ""):
    def decorator(cls: type) -> type:
        parser_registry.register(name, cls, description=description)
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------


def _build_chunker(conf: dict[str, Any]) -> BaseChunker:
    chunker_name = conf.get("chunker", "paragraph")
    entry = chunker_registry.get(chunker_name)
    if entry is None:
        _logger.warning(
            "Unknown chunker '%s', falling back to 'paragraph'. Available: %s",
            chunker_name,
            chunker_registry.list_available(),
        )
        entry = chunker_registry.get("paragraph")
        if entry is None:
            raise ValueError("No chunkers registered")
    return entry.cls(**_resolve_kwargs(entry, conf))


def _build_retriever(
    chunks: list[Chunk],
    conf: dict[str, Any],
    *,
    embedder: EmbeddingClient | None = None,
) -> BaseRetriever:
    retriever_name = conf.get("retriever", "keyword")
    entry = retriever_registry.get(retriever_name)
    if entry is None:
        _logger.warning(
            "Unknown retriever '%s', falling back to 'keyword'. Available: %s",
            retriever_name,
            retriever_registry.list_available(),
        )
        entry = retriever_registry.get("keyword")
        if entry is None:
            raise ValueError("No retrievers registered")

    kwargs = _resolve_kwargs(entry, conf)
    kwargs["chunks"] = chunks
    if "embedder" in entry.inject:
        kwargs["embedder"] = embedder
    return entry.cls(**kwargs)


def _resolve_kwargs(entry: ComponentEntry, conf: dict[str, Any]) -> dict[str, Any]:
    """Resolve constructor kwargs from config using entry metadata + config_aliases."""
    config_aliases = entry.config_aliases
    kwargs: dict[str, Any] = {}
    for param_name in entry.parameters:
        yaml_key = param_name
        for yk, pn in config_aliases.items():
            if pn == param_name:
                yaml_key = yk
                break
        if yaml_key in conf:
            kwargs[param_name] = conf[yaml_key]
        elif "default" in entry.parameters[param_name]:
            kwargs[param_name] = entry.parameters[param_name]["default"]
    return kwargs


def _load_documents(conf: dict[str, Any]) -> tuple[BaseChunker, list[Chunk]]:
    """Load, parse, and chunk documents from config. Returns ``(chunker, chunks)``.

    Each ``documents[]`` entry selects a source:

    - ``{path: "..."}`` or a bare string -- local file / glob / directory (recursed).
    - ``{source: http, url: "..."}`` -- fetch over HTTP(S) via httpx (hard dep).
    - ``{source: s3, ...}`` -- S3/R2 via boto3 (``[rag-cloud]`` extra; raises
      ``LLMInvalidRequestError`` at call time when boto3 is missing).
    - ``{source: firecrawl, ...}`` -- site crawl via httpx (no extra).

    Fetched/loaded bytes are parsed by format via the parser registry; unreadable or
    oversized files are skipped. ``max_document_size_mb`` (default 10) bounds a single doc.
    """
    import glob as _glob
    from pathlib import Path as PathlibPath

    from ragworkbench.ingest.parsers import dispatch_parser
    from ragworkbench.ingest.sources import (
        DocumentCache,
        fetch_firecrawl_entry,
        fetch_http_entry,
        fetch_s3_entry,
    )
    from ragworkbench.types import Document

    chunker = _build_chunker(conf)
    doc_cache_path = conf.get("document_cache_path")
    doc_cache = DocumentCache(doc_cache_path) if doc_cache_path else None

    def _resolve_files(path: str) -> list[PathlibPath]:
        if any(ch in path for ch in "*?["):
            matches = sorted(PathlibPath(p) for p in _glob.glob(path, recursive=True) if PathlibPath(p).is_file())
            if not matches:
                _logger.warning("RAG glob pattern %r matched no files", path)
            return matches
        p = PathlibPath(path)
        if p.is_dir():
            matches = sorted(f for f in p.rglob("*") if f.is_file())
            if not matches:
                _logger.warning("RAG directory %r contains no files", path)
            return matches
        if not p.is_file():
            _logger.warning("RAG document path does not exist: %s", path)
        return [p] if p.is_file() else []

    def _resolve_entry(entry: Any):
        if isinstance(entry, str):
            for fp in _resolve_files(entry):
                try:
                    yield fp.name, fp.read_bytes()
                except OSError as exc:
                    _logger.warning("RAG: skipping unreadable file %s: %s", fp, exc)
            return
        if not isinstance(entry, dict):
            return
        source = (entry.get("source") or "file").lower()
        if source in ("file", "local"):
            path = entry.get("path", "")
            if path:
                for fp in _resolve_files(path):
                    try:
                        yield fp.name, fp.read_bytes()
                    except OSError as exc:
                        _logger.warning("RAG: skipping unreadable file %s: %s", fp, exc)
            return
        if source == "http" or "url" in entry:
            yield from fetch_http_entry(entry, doc_cache, max_bytes=max_bytes)
            return
        if source == "s3":
            yield from fetch_s3_entry(entry, doc_cache, max_bytes=max_bytes)
            return
        if source == "firecrawl":
            yield from fetch_firecrawl_entry(entry, doc_cache)
            return
        _logger.warning("Unknown/unsupported document source %r; skipping", source)

    max_mb = int(conf.get("max_document_size_mb", 10))
    max_bytes = max_mb * 1024 * 1024
    all_chunks: list[Chunk] = []
    for entry in conf.get("documents", []):
        fmt_hint = entry.get("format") if isinstance(entry, dict) else None
        for name, data in _resolve_entry(entry):
            if len(data) > max_bytes:
                _logger.warning("Skipping %s: %d bytes exceeds max_document_size_mb=%d", name, len(data), max_mb)
                continue
            text, meta = dispatch_parser(name, data, format_hint=fmt_hint)
            if not text or not text.strip():
                _logger.info(
                    "RAG: skipping %s (parsed to empty text; source_format=%s)", name, meta.get("source_format")
                )
                continue
            stem = PathlibPath(name).stem or name
            doc = Document(id=stem, title=stem, content=text, metadata={"source": stem, **meta})
            for chunk in chunker.chunk(doc):
                chunk.metadata["source"] = stem
                if meta.get("source_format"):
                    chunk.metadata["source_format"] = meta["source_format"]
                all_chunks.append(chunk)

    return chunker, all_chunks


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def build_pipeline(
    conf: dict[str, Any],
    *,
    embedder: EmbeddingClient | None = None,
    chat_client: ChatClient | None = None,
) -> BaseRetriever | None:
    """Build a retrieval pipeline from config: load docs -> chunk -> retriever (+ stage wrappers).

    Composes, in order: base retriever (keyword/bm25/semantic/hybrid) -> optional
    ``RewritingRetriever`` wrapper (``query_rewrite``/``hyde``, OUTSIDE so it rewrites the
    query before the rerank stage over-fetches) -> optional ``CrossEncoderReranker`` wrapper
    (``rerank`` dict). Returns the (possibly wrapped) retriever, or ``None`` if disabled /
    no documents. ``embedder`` is injected into semantic/hybrid retrievers; ``chat_client``
    into the rewriter. Augmentation is a separate agent seam -- NOT composed here (callers
    wrap the returned retriever in an ``AugmentationStrategy`` for agent integration).
    """
    if not conf or not conf.get("enabled", True):
        return None

    _, all_chunks = _load_documents(conf)
    if not all_chunks:
        _logger.warning("RAG enabled but no documents loaded")
        return None

    if conf.get("retriever") == "vectorstore":
        # Composite retriever: build the store from `vectorstore:` config + wrap it.
        # Not registered via the standard registry path (needs a store, not just kwargs).
        from ragworkbench.retrieval.vectorstore_retriever import VectorStoreRetriever
        from ragworkbench.vectorstore import build_store

        store = build_store(conf.get("vectorstore") or {}, embedder=embedder)
        retriever = VectorStoreRetriever(all_chunks, embedder=embedder, store=store)
    else:
        retriever = _build_retriever(all_chunks, conf, embedder=embedder)

    # Query-rewrite / HyDE wrapper -- runs BEFORE retrieval, so it sits OUTSIDE the rerank
    # wrapper (rewrite the query, then the rerank-wrapped retriever over-fetches + rescores).
    if conf.get("query_rewrite") or conf.get("hyde"):
        from ragworkbench.retrieval.rewrite import RewritingRetriever

        mode = "hyde" if conf.get("hyde") else "llm"
        retriever = RewritingRetriever(retriever, chat_client=chat_client, mode=mode, config=conf.get("rewrite") or {})

    # Cross-encoder rerank wrapper -- runs AFTER retrieval. ``build_rerank_client`` returns
    # None (with a warning) when the provider lacks an api_key, leaving the base unwrapped.
    rerank_conf = conf.get("rerank")
    if isinstance(rerank_conf, dict):
        from ragworkbench.retrieval.rerank import CrossEncoderReranker, build_rerank_client

        backend = build_rerank_client(rerank_conf)
        if backend is not None:
            retriever = CrossEncoderReranker(
                retriever,
                backend,
                fetch_multiplier=rerank_conf.get("fetch_multiplier", 3),
                score_threshold=rerank_conf.get("score_threshold"),
            )

    return retriever


# ---------------------------------------------------------------------------
# Custom module loading (YAML-driven extensibility)
# ---------------------------------------------------------------------------


def load_custom_components(custom_modules: list[str]) -> None:
    """Import modules to trigger their ``@register_*`` decorators.

    Config example::

        custom_modules:
          - my_package.rag.chunkers
    """
    for module_path in custom_modules:
        try:
            importlib.import_module(module_path)
        except ImportError as e:
            _logger.warning("Failed to import custom module '%s': %s", module_path, e)
