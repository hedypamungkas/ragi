"""ragi/ingest/parsers -- document-format parsers (text/html/pdf/docx).

Lifted from ``koboi/rag/parsers.py`` and decoupled: zero ``koboi.*`` imports. Built-ins
``text`` and ``html`` self-register at module import (always available, stdlib-only).
``pdf`` / ``pdf_table`` / ``docx`` register conditionally -- only when their backing
library imports successfully (the ``[parsers]`` extra). A format whose parser isn't
registered falls back to ``TextParser`` inside ``dispatch_parser``.

Notes:
- ``pymupdf`` is intentionally excluded (AGPL-3.0 clashes with the Apache-2.0 core).
- OCR, table extraction beyond pdfplumber, and legacy ``.doc`` are out of scope.
- ``trafilatura`` is preferred for HTML when importable (readability/boilerplate
  removal); the stdlib ``_TagStripper`` is the always-available offline fallback.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from html.parser import HTMLParser
from pathlib import Path

from ragi.registry import parser_registry, register_parser

_logger = logging.getLogger(__name__)

# Extension -> format key. Custom parsers may claim additional extensions by
# registering under a format key the dispatch layer maps to.
_EXT_TO_FORMAT: dict[str, str] = {
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
    ".rst": "text",
    ".text": "text",
    ".html": "html",
    ".htm": "html",
    ".xhtml": "html",
    ".pdf": "pdf",
    ".docx": "docx",
}


class BaseParser(ABC):
    """Extract plain text + metadata from raw document bytes."""

    @abstractmethod
    def extract(self, name: str, data: bytes) -> tuple[str, dict]:
        """Return ``(text, metadata)``. ``metadata`` may carry ``source_format`` etc."""
        ...


def _looks_binary(data: bytes) -> bool:
    """Heuristic (git-style): a NUL byte in the first 8KB means binary, not text.

    Catches PDFs/images/executables routed to the text fallback (e.g. when pypdf is
    absent), so they are skipped instead of ingested as mojibake. Genuine latin-1 text
    has no NUL bytes and still decodes.
    """
    return b"\x00" in data[:8192]


@register_parser("text", description="Plain text / markdown (encoding-safe)")
class TextParser(BaseParser):
    """Plain text / markdown with an encoding fallback (never crashes on bytes).

    Genuine binary (NUL-byte) input returns empty text so the loader skips it.
    """

    def extract(self, name: str, data: bytes) -> tuple[str, dict]:
        if _looks_binary(data):
            return "", {"source_format": "binary"}
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return data.decode(enc), {"source_format": "text"}
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace"), {"source_format": "text"}


class _TagStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(c.strip() for c in self._chunks if c.strip())


@register_parser("html", description="HTML -> text via stdlib html.parser (no deps)")
class HtmlParser(BaseParser):
    """HTML -> text via stdlib ``html.parser`` (zero dependencies).

    Prefers ``trafilatura`` (readability/boilerplate removal) when importable; falls
    back to the stdlib tag-stripper (the offline default). Either path returns
    ``source_format: html``.
    """

    def extract(self, name: str, data: bytes) -> tuple[str, dict]:
        raw = data.decode("utf-8", errors="replace")
        if _TRAFILATURA_AVAILABLE:
            try:
                extracted = trafilatura.extract(  # type: ignore[union-attr]
                    raw,
                    output_format="markdown",
                    include_comments=False,
                    include_tables=True,
                    with_metadata=False,
                )
                if extracted and extracted.strip():
                    return extracted, {"source_format": "html"}
            except Exception as exc:  # noqa: BLE001 - readability is best-effort; fall back below
                _logger.debug("trafilatura failed on '%s' (%s); falling back to tag-strip", name, exc)
        stripper = _TagStripper()
        try:
            stripper.feed(raw)
            text = stripper.get_text() or raw
        except Exception as exc:  # malformed html -> fall back to the raw decode
            _logger.warning("HtmlParser tag-strip failed on '%s' (%s); using raw decode", name, exc)
            text = raw
        return text, {"source_format": "html"}


# --- optional-dependency guards ---------------------------------------------
#
# Each parser whose backing library lives behind the ``[parsers]`` extra is guarded by
# a try/except import. The class is always defined (so type references resolve), but it
# only enters ``parser_registry`` when its dep imported successfully. ``dispatch_parser``
# falls back to ``TextParser`` when a format isn't registered.

_PYPDF_AVAILABLE = False
try:  # pragma: no cover - dep guard
    from pypdf import PdfReader  # type: ignore[import-not-found]

    _PYPDF_AVAILABLE = True
except ImportError:
    PdfReader = None  # type: ignore[assignment,misc]

_PDFPLUMBER_AVAILABLE = False
try:  # pragma: no cover - dep guard
    import pdfplumber  # type: ignore[import-not-found]  # noqa: F401

    _PDFPLUMBER_AVAILABLE = True
except ImportError:
    pdfplumber = None  # type: ignore[assignment]

_DOCX_AVAILABLE = False
try:  # pragma: no cover - dep guard
    import docx  # type: ignore[import-not-found]  # python-docx

    _DOCX_AVAILABLE = True
except ImportError:
    docx = None  # type: ignore[assignment]

# Optional readability extraction for the html parser (boilerplate -> markdown). Lives
# behind the ``[parsers]`` extra (mirrors koboi's ``[web]``); absent in CI (falls back
# to the stdlib tag-stripper).
_TRAFILATURA_AVAILABLE = False
try:  # pragma: no cover - dep guard
    import trafilatura  # type: ignore[import-not-found]  # noqa: F401

    _TRAFILATURA_AVAILABLE = True
except ImportError:
    trafilatura = None  # type: ignore[assignment]


class PdfParser(BaseParser):
    """PDF text-layer extraction via ``pypdf`` (scanned/encrypted PDFs may yield '')."""

    def extract(self, name: str, data: bytes) -> tuple[str, dict]:
        from io import BytesIO

        reader = PdfReader(BytesIO(data))  # type: ignore[misc]
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages), {"source_format": "pdf", "page_count": len(reader.pages)}


class PdfTableParser(BaseParser):
    """PDF text + tables via ``pdfplumber`` (heavier than pypdf; opt-in per-document).

    Extracts page text AND tables (rendered as markdown rows). Select it for PDFs where
    table content matters, via a per-document ``format: pdf_table`` override.
    """

    def extract(self, name: str, data: bytes) -> tuple[str, dict]:
        from io import BytesIO

        import pdfplumber  # type: ignore[import-not-found]  # noqa: PLC0415 - lazy import behind the guard

        text_parts: list[str] = []
        table_count = 0
        page_count = 0
        with pdfplumber.open(BytesIO(data)) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
                for table in page.extract_tables() or []:
                    table_count += 1
                    text_parts.append(self._table_to_markdown(table))
        return "\n".join(t for t in text_parts if t), {
            "source_format": "pdf_table",
            "table_count": table_count,
            "page_count": page_count,
        }

    @staticmethod
    def _table_to_markdown(table: list[list]) -> str:
        rows = []
        for row in table:
            cells = ["" if c is None else str(c).replace("|", "/").strip() for c in row]
            rows.append("| " + " | ".join(cells) + " |")
        if rows:
            cols = max(1, rows[0].count("|") - 1)
            rows.insert(1, "| " + " | ".join("---" for _ in range(cols)) + " |")
        return "\n".join(rows)


class DocxParser(BaseParser):
    """DOCX (Office Open XML) paragraph extraction via ``python-docx``."""

    def extract(self, name: str, data: bytes) -> tuple[str, dict]:
        from io import BytesIO

        document = docx.Document(BytesIO(data))  # type: ignore[union-attr]
        return "\n".join(p.text for p in document.paragraphs), {"source_format": "docx"}


def detect_format(name: str, data: bytes) -> str:
    """Format key by extension, with magic-byte confirmation for unknown extensions."""
    ext = Path(name).suffix.lower()
    fmt = _EXT_TO_FORMAT.get(ext)
    if fmt:
        return fmt
    if data[:5] == b"%PDF-":
        return "pdf"
    if data[:4] == b"PK\x03\x04" and ext == ".docx":
        return "docx"
    head = data[:10].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        return "html"
    return "text"


def dispatch_parser(name: str, data: bytes, format_hint: str | None = None) -> tuple[str, dict]:
    """Pick a parser by format and extract text.

    ``format_hint`` overrides extension/magic detection (e.g. ``"pdf_table"`` for a PDF
    whose tables matter). Falls back to an encoding-safe text decode if the format is
    unknown, the parser isn't registered (optional dep missing), or it raises.
    """
    fmt = format_hint or detect_format(name, data)
    entry = parser_registry.get(fmt) or parser_registry.get("text")
    if entry is None:  # text not registered yet (import order) -> decode directly
        return data.decode("utf-8", errors="replace"), {}
    try:
        return entry.cls().extract(name, data)
    except Exception as exc:  # corrupt / unsupported variant  # noqa: BLE001
        _logger.warning("Parser '%s' failed on '%s' (%s); skipping", fmt, name, exc)
        if _looks_binary(data):
            return "", {"source_format": "binary-parse-error"}
        return data.decode("utf-8", errors="replace"), {}


# --- conditional registration of optional parsers ---------------------------
# text/html are always registered (decorators above). pdf/docx/pdf_table only when
# their backing library imported successfully -- otherwise the format is unknown to the
# registry and ``dispatch_parser`` transparently falls back to TextParser.

if _PYPDF_AVAILABLE:
    register_parser("pdf", description="PDF text extraction via pypdf (optional, [parsers] extra)")(PdfParser)

if _DOCX_AVAILABLE:
    register_parser("docx", description="DOCX text extraction via python-docx (optional, [parsers] extra)")(DocxParser)

if _PDFPLUMBER_AVAILABLE:
    register_parser("pdf_table", description="PDF text+tables via pdfplumber (optional, [parsers] extra)")(
        PdfTableParser
    )
