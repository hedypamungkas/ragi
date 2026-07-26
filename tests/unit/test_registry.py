"""Unit tests for the ComponentRegistry + decorators."""

from __future__ import annotations

import pytest

import ragi
from ragi.registry import ComponentRegistry, _extract_parameters


class TestComponentRegistry:
    def test_register_and_get(self):
        reg = ComponentRegistry("x")

        class C:
            def __init__(self, a: int = 1, b: str = "z") -> None:
                self.a = a
                self.b = b

        reg.register("c", C, description="demo")
        entry = reg.get("c")
        assert entry is not None and entry.cls is C
        assert "c" in reg.list_available()

    def test_extract_parameters_skips_self_and_varargs(self):
        class C:
            def __init__(self, a: int = 1, *args: object, **kwargs: object) -> None: ...

        params = _extract_parameters(C)
        assert "a" in params
        assert "self" not in params and "args" not in params and "kwargs" not in params

    def test_invalid_config_alias_raises(self):
        reg = ComponentRegistry("x")

        class C:
            def __init__(self, a: int = 1) -> None: ...

        with pytest.raises(ValueError):
            reg.register("c", C, config_aliases={"yaml_key": "nonexistent_param"})

    def test_builtins_registered(self):
        ragi.register_builtins()
        assert "bm25" in ragi.retriever_registry.list_available()
        assert "keyword" in ragi.retriever_registry.list_available()
        assert "paragraph" in ragi.chunker_registry.list_available()
        assert "text" in ragi.parser_registry.list_available()

    def test_load_custom_modules_missing_is_soft(self):
        # A nonexistent module logs a warning, does not raise.
        ragi.load_custom_components(["this.module.does.not.exist"])
