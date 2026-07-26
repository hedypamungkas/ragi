"""Shared test fixtures for rag-workbench.

Builtin chunkers/retrievers/parsers self-register on import of their modules; this session
fixture calls ``register_builtins()`` once so every test sees a populated registry.
"""

from __future__ import annotations

import pytest

import ragworkbench as rwb


@pytest.fixture(scope="session", autouse=True)
def _register_builtins() -> None:
    rwb.register_builtins()
