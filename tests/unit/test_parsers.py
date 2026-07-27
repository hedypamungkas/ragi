"""Unit tests for ragi.ingest.parsers -- text/html parsers, format detection, dispatch.

The optional backends (pypdf / pdfplumber / python-docx / trafilatura) are absent in CI,
so their classes are NOT registered. We still cover their ``extract`` bodies by faking the
backing module on the attribute ``parsers`` already imported into, or via ``sys.modules``
for the lazy imports.
"""

from __future__ import annotations

import sys
from io import BytesIO
from types import SimpleNamespace

import pytest

from ragi.ingest import parsers


# --------------------------------------------------------------------------- #
# TextParser
# --------------------------------------------------------------------------- #
class TestTextParser:
    def test_utf8_text(self):
        text, meta = parsers.TextParser().extract("a.txt", "héllo wörld".encode())
        assert text == "héllo wörld"
        assert meta["source_format"] == "text"

    def test_utf8_sig_bom_stripped(self):
        # utf-8 decodes the BOM bytes without error -> the ﻿ prefix is kept (utf-8-sig
        # would strip it, but is only tried if utf-8 fails). Substring check reflects that.
        text, _ = parsers.TextParser().extract("a.txt", b"\xef\xbb\xbfbom content")
        assert "bom content" in text

    def test_latin1_fallback(self):
        # 0xe9 is é in latin-1, invalid as standalone utf-8 -> latin-1 kicks in
        text, _ = parsers.TextParser().extract("a.txt", b"caf\xe9")
        assert text == "café"

    def test_binary_input_returns_empty(self):
        text, meta = parsers.TextParser().extract("a.txt", b"\x00\x01\x02 binary")
        assert text == ""
        assert meta["source_format"] == "binary"

    def test_looks_binary_heuristic(self):
        assert parsers._looks_binary(b"\x00 nul early") is True
        assert parsers._looks_binary(b"plain text no nul") is False
        # nul beyond 8KB window is NOT considered binary
        assert parsers._looks_binary(b"x" * 8192 + b"\x00") is False


# --------------------------------------------------------------------------- #
# HtmlParser + _TagStripper
# --------------------------------------------------------------------------- #
class TestTagStripper:
    def test_strips_tags_and_joins(self):
        s = parsers._TagStripper()
        s.feed("<p>Hello</p><p>world</p>")
        assert s.get_text() == "Hello world"

    def test_ignores_whitespace_only_chunks(self):
        s = parsers._TagStripper()
        s.feed("<div>   </div><span>kept</span>")
        assert s.get_text() == "kept"


class TestHtmlParser:
    def test_tag_strip_path_when_trafilatura_absent(self):
        text, meta = parsers.HtmlParser().extract("a.html", b"<p>Hello <b>world</b></p>")
        assert "Hello" in text and "world" in text
        assert meta["source_format"] == "html"

    def test_trafilatura_path_when_available(self, monkeypatch):
        monkeypatch.setattr(parsers, "_TRAFILATURA_AVAILABLE", True)
        monkeypatch.setattr(parsers, "trafilatura", SimpleNamespace(extract=lambda raw, **kw: "# clean md"))
        text, meta = parsers.HtmlParser().extract("a.html", b"<p>x</p>")
        assert text == "# clean md"
        assert meta["source_format"] == "html"

    def test_trafilatura_empty_result_falls_back_to_tag_strip(self, monkeypatch):
        monkeypatch.setattr(parsers, "_TRAFILATURA_AVAILABLE", True)
        monkeypatch.setattr(parsers, "trafilatura", SimpleNamespace(extract=lambda raw, **kw: ""))
        text, _ = parsers.HtmlParser().extract("a.html", b"<p>fallback</p>")
        assert "fallback" in text

    def test_trafilatura_raises_falls_back_to_tag_strip(self, monkeypatch):
        monkeypatch.setattr(parsers, "_TRAFILATURA_AVAILABLE", True)

        def boom(**kw):
            raise RuntimeError("trafil fail")

        monkeypatch.setattr(parsers, "trafilatura", SimpleNamespace(extract=boom))
        text, _ = parsers.HtmlParser().extract("a.html", b"<p>recovered</p>")
        assert "recovered" in text

    def test_malformed_html_falls_back_to_raw_decode(self, monkeypatch):
        class _BrokenStripper(parsers._TagStripper):
            def feed(self, data):
                raise RuntimeError("parse boom")

        monkeypatch.setattr(parsers, "_TagStripper", _BrokenStripper)
        text, _ = parsers.HtmlParser().extract("a.html", "café <weird".encode())
        assert "café" in text  # raw decode fallback


# --------------------------------------------------------------------------- #
# detect_format
# --------------------------------------------------------------------------- #
class TestDetectFormat:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("a.txt", "text"),
            ("a.md", "text"),
            ("a.markdown", "text"),
            ("a.rst", "text"),
            ("a.html", "html"),
            ("a.htm", "html"),
            ("a.xhtml", "html"),
            ("a.pdf", "pdf"),
            ("a.docx", "docx"),
        ],
    )
    def test_extension_mapping(self, name, expected):
        assert parsers.detect_format(name, b"") == expected

    def test_pdf_magic_bytes_unknown_ext(self):
        assert parsers.detect_format("noext", b"%PDF-1.4 stuff") == "pdf"

    def test_doctype_head_sniff(self):
        assert parsers.detect_format("noext", b"<!doctype html>") == "html"

    def test_html_tag_head_sniff(self):
        assert parsers.detect_format("noext", b"<html><body>") == "html"

    def test_plain_text_fallback(self):
        assert parsers.detect_format("noext", b"just some text") == "text"


# --------------------------------------------------------------------------- #
# dispatch_parser
# --------------------------------------------------------------------------- #
class TestDispatchParser:
    def test_format_hint_overrides_detection(self):
        # .txt normally -> text; force html parsing via hint
        text, meta = parsers.dispatch_parser("a.txt", b"<p>tagged</p>", format_hint="html")
        assert "tagged" in text and meta["source_format"] == "html"

    def test_pdf_without_pypdf_falls_back_to_text(self):
        # pypdf absent in CI -> "pdf" not registered -> text fallback
        text, _ = parsers.dispatch_parser("a.pdf", b"plain bytes here")
        assert "plain bytes here" in text

    def test_parser_failure_falls_back_to_decode(self, monkeypatch):
        def boom(self, name, data):
            raise RuntimeError("parse fail")

        monkeypatch.setattr(parsers.TextParser, "extract", boom)
        text, meta = parsers.dispatch_parser("a.txt", b"recovered")
        assert text == "recovered"
        assert meta == {}

    def test_parser_failure_on_binary_returns_empty(self, monkeypatch):
        def boom(self, name, data):
            raise RuntimeError("parse fail")

        monkeypatch.setattr(parsers.TextParser, "extract", boom)
        text, meta = parsers.dispatch_parser("a.txt", b"\x00\x01 binary")
        assert text == ""
        assert meta["source_format"] == "binary-parse-error"

    def test_text_not_registered_decodes_directly(self, monkeypatch):
        # simulate text parser missing from the registry entirely
        monkeypatch.setattr(parsers.parser_registry, "get", lambda key: None)
        text, meta = parsers.dispatch_parser("a.txt", b"direct decode")
        assert text == "direct decode"
        assert meta == {}


# --------------------------------------------------------------------------- #
# Optional parsers (deps faked)
# --------------------------------------------------------------------------- #
class _FakePdfPage:
    def __init__(self, text, tables=None):
        self._t = text
        self._tables = tables or []

    def extract_text(self):
        return self._t

    def extract_tables(self):
        return self._tables


class _CM:
    def __init__(self, obj):
        self._o = obj

    def __enter__(self):
        return self._o

    def __exit__(self, *exc):
        return False


class TestPdfParser:
    def test_extracts_pages_and_count(self, monkeypatch):
        class _Reader:
            def __init__(self, stream):
                assert isinstance(stream, BytesIO)
                self.pages = [_FakePdfPage("page one"), _FakePdfPage("page two")]

        monkeypatch.setattr(parsers, "PdfReader", _Reader)
        text, meta = parsers.PdfParser().extract("a.pdf", b"%PDF- fake")
        assert "page one" in text and "page two" in text
        assert meta == {"source_format": "pdf", "page_count": 2}


class TestPdfTableParser:
    def test_extracts_text_and_tables(self, monkeypatch):
        fake_pdf = type(
            "Pdf",
            (),
            {"pages": [_FakePdfPage("p1", [[["a", "b"], ["c", "d"]]])]},
        )()
        fake_module = SimpleNamespace(open=lambda stream: _CM(fake_pdf))
        monkeypatch.setitem(sys.modules, "pdfplumber", fake_module)
        text, meta = parsers.PdfTableParser().extract("a.pdf", b"%PDF- fake")
        assert "p1" in text
        assert "a" in text and "|" in text  # table rendered as markdown
        assert meta["table_count"] == 1
        assert meta["page_count"] == 1

    def test_table_to_markdown_adds_separator_row(self):
        md = parsers.PdfTableParser._table_to_markdown([[None, "x|y"], ["a", "b"]])
        lines = md.splitlines()
        assert lines[0].startswith("|") and lines[0].endswith("|")
        assert lines[1].replace("|", "").replace("-", "").strip() == ""  # separator row
        # None -> empty cell, pipe in cell is escaped to "/"
        assert "x/y" in lines[0]

    def test_table_to_markdown_empty(self):
        assert parsers.PdfTableParser._table_to_markdown([]) == ""


class TestDocxParser:
    def test_extracts_paragraphs(self, monkeypatch):
        class _Doc:
            def __init__(self, stream):
                self.paragraphs = [SimpleNamespace(text="para one"), SimpleNamespace(text="para two")]

        monkeypatch.setattr(parsers, "docx", SimpleNamespace(Document=_Doc))
        text, meta = parsers.DocxParser().extract("a.docx", b"PK fake")
        assert "para one" in text and "para two" in text
        assert meta["source_format"] == "docx"
