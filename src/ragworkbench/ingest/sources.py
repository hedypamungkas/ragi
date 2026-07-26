"""ragworkbench/ingest/sources -- remote document fetching (HTTP / S3 / Firecrawl) + cache.

Lifted from ``koboi/rag/sources.py`` and decoupled: zero ``koboi.*`` imports. Ships the
HTTP source (httpx, hard dep), the S3/R2 source (boto3, optional ``[rag-cloud]`` extra),
and the Firecrawl site-crawl source (httpx -- no extra).

The SSRF guard and retry constants are inlined from ``koboi.tools.builtin.web`` (the
upstream source imports them across that boundary). Inlining keeps the lib decoupled
while preserving the exact posture: ``hostname`` is resolved and every returned IP is
rejected when ``ipaddress`` flags it as loopback / private / link-local / unspecified /
multicast / reserved (covers 127.0.0.0/8, 10/8, 172.16/12, 192.168/16, 169.254/16,
``::1``, ``fc00::/7``, and ``::``).

CWE-400 / GHSA-qf8c-xp5r-p869 defense: bodies are read via a streaming request with a
hard ``max_bytes`` bound -- an oversized document is rejected up front by the
``Content-Length`` pre-check, or bounded mid-stream, instead of being fully buffered.
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import logging
import os
import socket
import time
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

import httpx  # hard dependency (pyproject)

from ragworkbench.errors import LLMInvalidRequestError

_logger = logging.getLogger(__name__)

# --- retry policy (inlined; koboi reuses koboi.tools.builtin.web's constants) ---
MAX_RETRIES = 3
MAX_TIMEOUT = 30
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

_MAX_REDIRECTS = 5
# Default hard cap (50 MB) for a single remote document when no ``max_bytes`` is
# threaded in from the loader. Defends against buffer-before-limit DoS; the registry
# passes the configured ``max_document_size_mb`` which overrides this for RAG ingestion.
_DEFAULT_MAX_DOC_BYTES = 50 * 1024 * 1024

# Secrets kept out of cache-key material and logs.
_SECRET_KEYS = frozenset({"access_key_id", "secret_access_key", "headers", "token"})


# ---------------------------------------------------------------------------
# SSRF guard (inlined from koboi.tools.builtin.web._check_url_ssrf)
# ---------------------------------------------------------------------------


def _check_url_ssrf(url: str) -> None:
    """Parse URL, resolve hostname, verify it does not point to a private network.

    Pure stdlib (~40 LOC). Resolves the hostname via ``socket.getaddrinfo`` and rejects
    any returned IP whose ``ipaddress`` properties mark it as loopback, private,
    link-local, unspecified (``::`` / ``0.0.0.0``), multicast, or reserved. That covers
    127.0.0.0/8, 10/8, 172.16/12, 192.168/16, 169.254/16, ``::1``, and ULA ``fc00::/7``.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("invalid URL -- hostname not found")
    try:
        addrs = socket.getaddrinfo(hostname, None)
    except OSError as exc:  # gaierror / DNS failure -> treat as SSRF rejection
        raise ValueError(f"DNS resolution failed for '{hostname}': {exc}") from exc
    if not addrs:
        raise ValueError(f"DNS resolution returned no addresses for '{hostname}'")
    for _, _, _, _, sa in addrs:
        ip = ipaddress.ip_address(sa[0])
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_unspecified
            or ip.is_multicast
            or ip.is_reserved
        ):
            raise ValueError("URL points to unauthorized internal IP address")


# ---------------------------------------------------------------------------
# Cache-key helpers
# ---------------------------------------------------------------------------


def name_from_url(url: str) -> str:
    path = urlparse(url).path
    name = path.rsplit("/", 1)[-1] if path else ""
    return name or "document"


def source_key(spec: dict) -> str:
    """Stable cache key for a source spec (secrets excluded from the material)."""
    safe = {k: v for k, v in spec.items() if k not in _SECRET_KEYS}
    blob = repr(sorted(safe.items()))
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# HTTP fetch (streaming + bounded read; manual redirect loop with SSRF per hop)
# ---------------------------------------------------------------------------


def _close_stream(stream_ctx: object) -> None:
    """Best-effort close a ``Client.stream(...)`` sync context manager."""
    if stream_ctx is None:
        return
    with contextlib.suppress(Exception):
        stream_ctx.__exit__(None, None, None)  # type: ignore[attr-defined]


def fetch_http(
    url: str,
    *,
    headers: dict | None = None,
    timeout: int | None = None,
    max_bytes: int | None = None,
) -> bytes:
    """Fetch a single HTTP(S) document. Raises on failure.

    SSRF-checked on EVERY redirect hop (``follow_redirects=False`` + manual loop).
    Retries on transient failures (``RETRYABLE_STATUS`` + transport errors).

    The body is read via a STREAMING request with a hard ``max_bytes`` bound, so an
    oversized document is rejected early (``Content-Length`` pre-check) or bounded
    (stream read) instead of being fully buffered. Raises ``ValueError`` when the
    payload exceeds ``max_bytes``; defaults to ``_DEFAULT_MAX_DOC_BYTES`` when unset.
    """
    if max_bytes is None:
        max_bytes = _DEFAULT_MAX_DOC_BYTES
    timeout_s = max(1, min(int(timeout or MAX_TIMEOUT), MAX_TIMEOUT))
    with httpx.Client(timeout=timeout_s, follow_redirects=False) as client:
        current_url = url
        resp = None
        stream_ctx = None
        try:
            for _ in range(_MAX_REDIRECTS + 1):
                _check_url_ssrf(str(current_url))
                resp = None
                for attempt in range(MAX_RETRIES + 1):
                    try:
                        cm = client.stream("GET", current_url, headers=headers)
                        resp = cm.__enter__()
                        stream_ctx = cm  # track only after a successful enter
                    except httpx.HTTPError:
                        if attempt >= MAX_RETRIES:
                            raise
                        continue  # transport error -> retry
                    if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                        _close_stream(stream_ctx)
                        stream_ctx = None
                        resp = None
                        continue
                    break  # success or non-retryable status

                if resp.status_code in (301, 302, 303, 307, 308):
                    redir_status = resp.status_code
                    loc = resp.headers.get("location")
                    _close_stream(stream_ctx)
                    stream_ctx = None
                    resp = None
                    if not loc:
                        raise RuntimeError(f"HTTP {redir_status} from {current_url} had no Location header")
                    current_url = str(httpx.URL(current_url).join(loc))
                    continue
                resp.raise_for_status()
                # Content-Length pre-check: reject before consuming the body.
                cl = resp.headers.get("Content-Length")
                if cl is not None:
                    try:
                        cl_int = int(cl)
                    except (TypeError, ValueError):
                        cl_int = None
                    if cl_int is not None and cl_int > max_bytes:
                        raise ValueError(
                            f"document too large: Content-Length {cl_int} bytes exceeds {max_bytes} byte limit"
                        )
                # Bounded streaming read: stop as soon as the cap is crossed.
                buf = bytearray()
                for chunk in resp.iter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        raise ValueError(f"document too large: exceeds {max_bytes} byte limit")
                return bytes(buf)
            raise RuntimeError(f"too many redirects for {url}")
        finally:
            _close_stream(stream_ctx)


def fetch_http_entry(
    entry: dict,
    cache: DocumentCache | None,
    *,
    max_bytes: int,
) -> Iterator[tuple[str, bytes]]:
    """Yield ``(name, bytes)`` for an HTTP ``documents[]`` entry, with cache.

    ``max_bytes`` is required -- the loader computes it from ``max_document_size_mb``.
    Network / SSRF / status / oversize failures log a warning and yield nothing (the
    pipeline keeps building with the other sources) -- they never raise out.
    """
    url = entry.get("url", "")
    if not url:
        return
    key = source_key({"source": "http", "url": url})
    if cache:
        hit = cache.get(key)
        if hit is not None:
            yield hit
            return
    try:
        data = fetch_http(
            url,
            headers=entry.get("headers"),
            timeout=entry.get("timeout"),
            max_bytes=max_bytes,
        )
    except Exception as exc:  # network / SSRF / status / oversize -> skip, keep building  # noqa: BLE001
        _logger.warning("HTTP fetch failed for %s: %s", url, exc)
        return
    name = name_from_url(url)
    # Cache only AFTER fetch_http succeeded -- it raises ValueError on an oversized
    # body, so a too-large document is never written to the cache.
    if cache:
        try:  # protect cache write -> never crash the build on disk error
            cache.put(key, name, data)
        except OSError as exc:
            _logger.warning("DocumentCache write failed for %s: %s", url, exc)
    yield name, data


# ---------------------------------------------------------------------------
# S3 / R2 fetch (boto3, optional [rag-cloud] extra)
# ---------------------------------------------------------------------------


def fetch_s3_entry(
    entry: dict,
    doc_cache: DocumentCache | None,
    *,
    max_bytes: int | None = None,
) -> Iterator[tuple[str, bytes]]:
    """Yield ``(name, bytes)`` for objects under an S3/R2 prefix, with per-object cache.

    Cloudflare R2 is S3-compatible (use ``endpoint_url``). Requires ``boto3`` (the
    ``[rag-cloud]`` extra); raises ``LLMInvalidRequestError`` at call time when the
    optional dependency is missing. Three-layer size cap: ``Size`` metadata pre-check
    -> bounded ``body.read(max_bytes+1)`` -> post-read re-check. Per-object try/except
    so one bad object doesn't abort the rest.
    """
    try:
        import boto3  # type: ignore[import-not-found]
    except ImportError as exc:
        raise LLMInvalidRequestError("(rag-cloud) boto3 required: pip install 'ragworkbench[rag-cloud]'") from exc
    if max_bytes is None:
        max_bytes = _DEFAULT_MAX_DOC_BYTES
    bucket = entry.get("bucket")
    prefix = entry.get("key", "")
    if not bucket:
        _logger.warning("s3 document source missing 'bucket'; skipping")
        return
    endpoint = entry.get("endpoint_url") or ""
    region = entry.get("region", "auto")
    try:
        client = boto3.client(
            "s3",
            endpoint_url=endpoint or None,
            region_name=region,
            aws_access_key_id=entry.get("access_key_id"),
            aws_secret_access_key=entry.get("secret_access_key"),
        )
        paginator = client.get_paginator("list_objects_v2")
        found = False
        skipped = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                okey = obj["Key"]
                if okey.endswith("/"):  # folder marker
                    continue
                found = True
                name = okey.rsplit("/", 1)[-1] or okey
                # Include endpoint_url + region in the key so two backends with the
                # same bucket+key don't collide in a shared cache dir.
                ckey = source_key(
                    {"source": "s3", "bucket": bucket, "key": okey, "endpoint_url": endpoint, "region": region}
                )
                cached = doc_cache.get(ckey) if doc_cache else None
                if cached is not None:
                    yield cached
                    continue
                # Size pre-check -- skip the download entirely when the object
                # metadata already declares an oversized payload (no get_object call).
                obj_size = obj.get("Size")
                if obj_size is not None and obj_size > max_bytes:
                    _logger.warning(
                        "s3: skipping object %r (bucket=%s): Size %s exceeds %s byte limit",
                        okey,
                        bucket,
                        obj_size,
                        max_bytes,
                    )
                    skipped += 1
                    continue
                # Per-object try so one bad object doesn't kill the rest.
                try:
                    body = client.get_object(Bucket=bucket, Key=okey)["Body"]
                    data = body.read(max_bytes + 1)  # bounded read
                except Exception as exc:  # noqa: BLE001
                    _logger.warning("s3: skipping object %r (bucket=%s): %s", okey, bucket, exc)
                    skipped += 1
                    continue
                # Reject when the bounded read still crossed the cap (the body is
                # larger than max_bytes even without / despite the Size metadata).
                if len(data) > max_bytes:
                    _logger.warning(
                        "s3: skipping object %r (bucket=%s): exceeds %s byte limit", okey, bucket, max_bytes
                    )
                    skipped += 1
                    continue
                if doc_cache:
                    try:  # only cache AFTER the size check passed
                        doc_cache.put(ckey, name, data)
                    except OSError as exc:
                        _logger.warning("DocumentCache write failed for s3://%s/%s: %s", bucket, okey, exc)
                yield name, data
        if not found:
            _logger.warning("s3 bucket=%s prefix=%r returned no objects", bucket, prefix)
        if skipped:
            _logger.warning("s3: %d object(s) skipped due to errors (bucket=%s)", skipped, bucket)
    except Exception as exc:  # credentials / endpoint / network -> skip, keep building  # noqa: BLE001
        _logger.warning("s3 fetch failed (bucket=%s): %s", bucket, exc)


# ---------------------------------------------------------------------------
# Firecrawl site-crawl fetch (httpx, no extra)
# ---------------------------------------------------------------------------


def _firecrawl_crawl(base_url: str, api_key: str, limit: int, endpoint: str | None) -> list[dict]:
    """Run a Firecrawl site crawl (``POST /v1/crawl`` -> poll ``GET /v1/crawl/{id}``).

    Returns the list of page dicts (each carries ``markdown``/``html`` +
    ``metadata.sourceURL``). Raises on transport/HTTP failure or poll timeout. Returns
    the ``data`` list directly when Firecrawl answers synchronously.
    """
    base = (endpoint or "https://api.firecrawl.dev").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"url": base_url, "limit": limit, "scrapeOptions": {"formats": ["markdown"]}}
    with httpx.Client(timeout=60) as client:
        resp = client.post(f"{base}/v1/crawl", json=payload, headers=headers)
        resp.raise_for_status()
        job = resp.json()
        # Sync response already carrying data, or an async job id to poll.
        if isinstance(job.get("data"), list):
            return job["data"]
        job_id = job.get("id")
        if not job_id:
            raise RuntimeError("firecrawl crawl returned no job id and no data")
        for _ in range(120):  # up to ~4 minutes
            r = client.get(f"{base}/v1/crawl/{job_id}", headers=headers)
            r.raise_for_status()
            status = r.json()
            if status.get("status") == "completed":
                return status.get("data") or []
            if status.get("status") == "failed":
                raise RuntimeError(f"firecrawl crawl job failed: {status.get('error', '')}")
            time.sleep(2)
        raise RuntimeError("firecrawl crawl job timed out")


def fetch_firecrawl_entry(entry: dict, doc_cache: DocumentCache | None) -> Iterator[tuple[str, bytes]]:
    """Yield ``(name, bytes)`` for pages of a Firecrawl site crawl, with per-page cache.

    Mirrors ``fetch_http_entry`` / ``fetch_s3_entry``: one bad page skips (does not abort
    the crawl); ``DocumentCache`` keys by ``source_key({"source":"firecrawl","url":page_url})``
    so remote crawls aren't re-run on every per-run rebuild.

    ``entry`` keys: ``url`` (seed), ``api_key`` (or ``$FIRECRAWL_API_KEY``), ``limit`` (max
    pages, default 50), ``endpoint_url`` (self-host override). Uses ``httpx`` (already a
    hard dependency -- no extra required).
    """
    url = entry.get("url", "")
    if not url:
        return
    api_key = entry.get("api_key") or os.getenv("FIRECRAWL_API_KEY", "")
    if not api_key:
        _logger.warning("firecrawl source %s missing api_key; skipping", url)
        return
    # Defense in depth: validate the seed URL client-side before handing it to the SaaS.
    # ValueError = private range; OSError = DNS failure (gaierror). Either -> skip the entry.
    try:
        _check_url_ssrf(url)
    except (ValueError, OSError) as exc:
        _logger.warning("firecrawl source %s rejected by SSRF guard: %s", url, exc)
        return

    limit = int(entry.get("limit", 50))
    try:
        pages = _firecrawl_crawl(url, api_key, limit, entry.get("endpoint_url"))
    except Exception as exc:  # noqa: BLE001 - network / API / poll -> skip, keep building
        _logger.warning("firecrawl crawl failed for %s: %s", url, exc)
        return

    found = False
    for page in pages:
        if not isinstance(page, dict):
            continue
        meta = page.get("metadata") or {}
        page_url = (meta.get("sourceURL") if isinstance(meta, dict) else "") or url
        name = name_from_url(page_url) or "page"
        ckey = source_key({"source": "firecrawl", "url": page_url})
        if doc_cache:
            cached = doc_cache.get(ckey)
            if cached is not None:
                found = True
                yield cached
                continue
        content = page.get("markdown", "") or page.get("html", "")
        if not content:
            continue
        found = True
        data = content.encode("utf-8")
        if doc_cache:
            try:
                doc_cache.put(ckey, name, data)
            except OSError as exc:
                _logger.warning("DocumentCache write failed for firecrawl %s: %s", page_url, exc)
        yield name, data
    if not found:
        _logger.warning("firecrawl crawl of %s returned no usable pages", url)


# ---------------------------------------------------------------------------
# On-disk document cache (atomic writes)
# ---------------------------------------------------------------------------


class DocumentCache:
    """Opt-in on-disk cache for fetched remote documents (bytes + original name).

    Per-session pipeline rebuilds re-run ``_load_documents``; this cache prevents
    re-fetching the corpus over the network on every run. Keyed by a content-spec hash
    (secrets excluded); invalidation is manual (delete the directory).
    """

    def __init__(self, dir_path: str) -> None:
        self.dir = Path(dir_path)

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.dir / key, self.dir / f"{key}.name"

    def get(self, key: str) -> tuple[str, bytes] | None:
        data_f, name_f = self._paths(key)
        if data_f.exists() and name_f.exists():
            return name_f.read_text(), data_f.read_bytes()
        return None

    def put(self, key: str, name: str, data: bytes) -> None:
        """Atomic writes (temp + ``os.replace``)."""
        self.dir.mkdir(parents=True, exist_ok=True)
        data_f, name_f = self._paths(key)
        self._atomic_write(data_f, data)
        self._atomic_write(name_f, name.encode())

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink()
            raise
