"""Import-gate test for the LlamaIndex adapter (construct-without-dep raises cleanly).

llama-index is heavy and not installed in the test env, so we verify the gate behavior:
importing the adapter module without the extra raises ``LLMInvalidRequestError``. (If
llama-index IS installed, the gate can't be exercised -> skip.)
"""

from __future__ import annotations

import importlib
import sys

import pytest

from ragi.errors import LLMInvalidRequestError


def test_llamaindex_adapter_import_gated():
    try:
        import llama_index  # noqa: F401

        pytest.skip("llama_index is installed; cannot exercise the import gate")
    except ImportError:
        pass  # expected in the bare test env

    sys.modules.pop("ragi.adapters.llamaindex", None)
    with pytest.raises(LLMInvalidRequestError, match="adapters-llamaindex"):
        importlib.import_module("ragi.adapters.llamaindex")
