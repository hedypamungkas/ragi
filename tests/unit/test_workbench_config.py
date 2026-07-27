"""Unit tests for ragi.workbench.config -- pipeline config loading (dict / YAML)."""

from __future__ import annotations

import pytest

from ragi.workbench.config import load_config


class TestLoadConfig:
    def test_dict_passthrough_returns_copy(self):
        src = {"enabled": True, "retriever": "bm25"}
        cfg = load_config(src)
        assert cfg == src
        assert cfg is not src  # defensive copy

    def test_yaml_file_loads(self, tmp_path):
        f = tmp_path / "p.yaml"
        f.write_text("enabled: true\nretriever: keyword\n", encoding="utf-8")
        cfg = load_config(f)
        assert cfg == {"enabled": True, "retriever": "keyword"}

    def test_non_mapping_yaml_raises(self, tmp_path):
        f = tmp_path / "list.yaml"
        f.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="did not parse to a mapping"):
            load_config(f)

    def test_scalar_yaml_raises(self, tmp_path):
        f = tmp_path / "scalar.yaml"
        f.write_text("just a string\n", encoding="utf-8")
        with pytest.raises(ValueError, match="did not parse to a mapping"):
            load_config(f)
