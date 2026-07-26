"""ragworkbench -- eval-first, agent-agnostic RAG toolkit.

Public API: ``build_pipeline()``, the component registries/decorators, core types,
and the framework-agnostic Protocols. Builtin chunkers/retrievers/parsers
self-register when their modules are imported; call :func:`register_builtins`
(or import the workbench CLI) to load them.
"""

from __future__ import annotations

from ragworkbench.errors import (
    ConfigError,
    EvalError,
    LLMError,
    LLMInvalidRequestError,
    RagError,
    RetrievalError,
)
from ragworkbench.registry import (
    ComponentEntry,
    ComponentRegistry,
    augmentation_registry,
    build_pipeline,
    chunker_registry,
    load_custom_components,
    parser_registry,
    register_augmentation,
    register_chunker,
    register_parser,
    register_retriever,
    retriever_registry,
)
from ragworkbench.types import Chunk, Document, RetrievalResult

__version__ = "0.4.0"

_BUILTINS_LOADED = False


def register_builtins() -> None:
    """Import builtin chunkers/retrievers/parsers so their ``@register_*`` decorators fire.

    Idempotent. The workbench CLI calls this automatically; user code may call it too.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Importing these modules executes their module-level @register_* decorators.
    from ragworkbench.ingest import chunker as _chunkers  # noqa: F401
    from ragworkbench.ingest import parsers as _parsers  # noqa: F401
    from ragworkbench.retrieval import augmentation as _augmentations  # noqa: F401
    from ragworkbench.retrieval import rerank as _rerank  # noqa: F401
    from ragworkbench.retrieval import retriever as _retrievers  # noqa: F401
    from ragworkbench.retrieval import rewrite as _rewrite  # noqa: F401

    _BUILTINS_LOADED = True


__all__ = [
    "__version__",
    "register_builtins",
    # types
    "Chunk",
    "Document",
    "RetrievalResult",
    # errors
    "RagError",
    "ConfigError",
    "RetrievalError",
    "EvalError",
    "LLMError",
    "LLMInvalidRequestError",
    # registry
    "ComponentRegistry",
    "ComponentEntry",
    "chunker_registry",
    "retriever_registry",
    "augmentation_registry",
    "parser_registry",
    "register_chunker",
    "register_retriever",
    "register_augmentation",
    "register_parser",
    "build_pipeline",
    "load_custom_components",
]
