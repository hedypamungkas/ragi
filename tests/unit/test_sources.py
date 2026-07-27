"""Unit tests for ragi.ingest.sources -- SSRF guard, bounded HTTP/S3/Firecrawl fetch + cache.

Every network surface (httpx.Client, socket.getaddrinfo, boto3) is faked so the +/-/edge
branches run without network or optional deps. The fake objects mirror only the surface
``sources.py`` actually touches (status_code, headers.get, iter_bytes, raise_for_status,
__enter__/__exit__).
"""

from __future__ import annotations

import types

import httpx
import pytest

from ragi.errors import LLMInvalidRequestError
from ragi.ingest import sources


# --------------------------------------------------------------------------- #
# socket.getaddrinfo fakes for the SSRF guard
# --------------------------------------------------------------------------- #
def _ssrf_monkey(monkeypatch, addrs):
    """Make socket.getaddrinfo yield the given raw IP strings (as sockaddr tuples)."""

    def fake_getaddrinfo(host, *args, **kwargs):
        if isinstance(addrs, Exception):
            raise addrs
        return [(None, None, None, None, (ip, 0)) for ip in addrs]

    monkeypatch.setattr(sources.socket, "getaddrinfo", fake_getaddrinfo)


class TestSsrfGuard:
    def test_public_ip_allowed(self, monkeypatch):
        _ssrf_monkey(monkeypatch, ["8.8.8.8"])
        sources._check_url_ssrf("https://example.com/doc")  # no raise

    @pytest.mark.parametrize("ip", ["127.0.0.1", "10.1.2.3", "192.168.0.1", "172.16.0.1"])
    def test_private_ranges_rejected(self, monkeypatch, ip):
        _ssrf_monkey(monkeypatch, [ip])
        with pytest.raises(ValueError, match="unauthorized internal IP"):
            sources._check_url_ssrf("https://host/doc")

    @pytest.mark.parametrize("ip", ["169.254.169.254", "0.0.0.0", "::1", "fc00::1", "ff00::1"])
    def test_link_local_unspecified_ipv6_rejected(self, monkeypatch, ip):
        _ssrf_monkey(monkeypatch, [ip])
        with pytest.raises(ValueError):
            sources._check_url_ssrf("https://host/doc")

    def test_missing_hostname_rejected(self):
        with pytest.raises(ValueError, match="hostname not found"):
            sources._check_url_ssrf("not a url")

    def test_dns_failure_rejected(self, monkeypatch):
        _ssrf_monkey(monkeypatch, OSError("gaierror"))
        with pytest.raises(ValueError, match="DNS resolution failed"):
            sources._check_url_ssrf("https://missing.invalid/doc")

    def test_empty_addr_list_rejected(self, monkeypatch):
        _ssrf_monkey(monkeypatch, [])
        with pytest.raises(ValueError, match="returned no addresses"):
            sources._check_url_ssrf("https://host/doc")

    def test_any_private_in_multi_answer_rejected(self, monkeypatch):
        # one public + one private -> still rejected (defense in depth)
        _ssrf_monkey(monkeypatch, ["8.8.8.8", "127.0.0.1"])
        with pytest.raises(ValueError):
            sources._check_url_ssrf("https://host/doc")


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
class TestHelpers:
    def test_name_from_url_file(self):
        assert sources.name_from_url("https://h/a/b/report.pdf") == "report.pdf"

    def test_name_from_url_no_path_defaults_to_document(self):
        assert sources.name_from_url("https://h") == "document"

    def test_name_from_url_trailing_slash(self):
        assert sources.name_from_url("https://h/a/") == "document"

    def test_source_key_is_deterministic(self):
        a = sources.source_key({"source": "http", "url": "https://h/x"})
        b = sources.source_key({"url": "https://h/x", "source": "http"})
        assert a == b and len(a) == 32

    def test_source_key_excludes_secrets(self):
        public = sources.source_key({"source": "s3", "bucket": "b"})
        with_secret = sources.source_key(
            {"source": "s3", "bucket": "b", "access_key_id": "AKIA", "secret_access_key": "shh", "token": "t"}
        )
        assert public == with_secret  # secret material never changes the key

    def test_close_stream_none_is_noop(self):
        sources._close_stream(None)  # no raise

    def test_close_stream_closes_cm(self):
        closed = {"count": 0}

        class CM:
            def __exit__(self, *exc):
                closed["count"] += 1

        sources._close_stream(CM())
        assert closed["count"] == 1

    def test_close_stream_swallows_exit_error(self):
        class CM:
            def __exit__(self, *exc):
                raise RuntimeError("boom")

        sources._close_stream(CM())  # suppressed, no raise


# --------------------------------------------------------------------------- #
# DocumentCache
# --------------------------------------------------------------------------- #
class TestDocumentCache:
    def test_put_then_get_roundtrip(self, tmp_path):
        cache = sources.DocumentCache(str(tmp_path / "c"))
        key = sources.source_key({"source": "http", "url": "https://h/x"})
        assert cache.get(key) is None
        cache.put(key, "name.bin", b"payload")
        assert cache.get(key) == ("name.bin", b"payload")

    def test_get_returns_none_when_name_file_missing(self, tmp_path):
        cache = sources.DocumentCache(str(tmp_path / "c"))
        key = sources.source_key({"x": 1})
        cache.put(key, "n", b"d")
        # corrupt: remove the .name sidecar -> miss
        (cache.dir / f"{key}.name").unlink()
        assert cache.get(key) is None

    def test_put_creates_parent_dirs(self, tmp_path):
        deep = tmp_path / "a" / "b" / "c"
        cache = sources.DocumentCache(str(deep))
        cache.put("k", "n", b"d")
        assert deep.exists()

    def test_atomic_write_cleans_temp_on_failure(self, tmp_path, monkeypatch):
        cache = sources.DocumentCache(str(tmp_path / "c"))
        cache.dir.mkdir(parents=True)
        target = cache.dir / "k"

        def fail_replace(*a, **kw):
            raise OSError("busy")

        monkeypatch.setattr(sources.os, "replace", fail_replace)
        with pytest.raises(OSError, match="busy"):
            cache._atomic_write(target, b"d")
        # temp file cleaned up by the except branch
        assert not (cache.dir / "k.tmp").exists()


# --------------------------------------------------------------------------- #
# fetch_http -- fake httpx.Client (streaming)
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, status=200, chunks=None, headers=None, location=None, body=None):
        self.status_code = status
        self._chunks = chunks if chunks is not None else ([body] if body is not None else [b"ok"])
        h = dict(headers or {})
        if location is not None:
            h["location"] = location
        self.headers = h

    def iter_bytes(self):
        yield from self._chunks

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=httpx.Request("GET", "http://x"),
                response=httpx.Response(self.status_code),
            )


class _StreamCM:
    def __init__(self, resp_or_exc):
        self._payload = resp_or_exc

    def __enter__(self):
        if isinstance(self._payload, BaseException):
            raise self._payload
        return self._payload

    def __exit__(self, *exc):
        return False


class _ScriptedClient:
    """``httpx.Client`` stand-in that returns queued responses (or raises queued errors)."""

    def __init__(self, responses, ssrf_allow=True):
        self._responses = list(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream(self, method, url, headers=None):
        self.calls.append(url)
        item = self._responses.pop(0)
        if isinstance(item, _FakeResp):
            return _StreamCM(item)
        # an exception to raise on __enter__ (transport error)
        return _StreamCM(item)


def _patch_http_client(monkeypatch, responses):
    client = _ScriptedClient(responses)
    monkeypatch.setattr(sources.httpx, "Client", lambda *a, **kw: client)
    return client


@pytest.fixture
def allow_ssrf(monkeypatch):
    """Permit every URL through the SSRF guard so redirect/retry logic is exercised."""
    monkeypatch.setattr(sources, "_check_url_ssrf", lambda url: None)


class TestFetchHttp:
    def test_simple_200_returns_body(self, monkeypatch, allow_ssrf):
        _patch_http_client(monkeypatch, [_FakeResp(200, body=b"hello world")])
        assert sources.fetch_http("https://h/d") == b"hello world"

    def test_streams_chunks_and_concatenates(self, monkeypatch, allow_ssrf):
        _patch_http_client(monkeypatch, [_FakeResp(200, chunks=[b"ab", b"cd", b"ef"])])
        assert sources.fetch_http("https://h/d") == b"abcdef"

    def test_content_length_pre_check_rejects_oversize(self, monkeypatch, allow_ssrf):
        _patch_http_client(
            monkeypatch,
            [_FakeResp(200, body=b"x" * 10, headers={"Content-Length": "9999"})],
        )
        with pytest.raises(ValueError, match="Content-Length"):
            sources.fetch_http("https://h/d", max_bytes=100)

    def test_malformed_content_length_ignored(self, monkeypatch, allow_ssrf):
        _patch_http_client(
            monkeypatch,
            [_FakeResp(200, body=b"ok", headers={"Content-Length": "not-a-number"})],
        )
        assert sources.fetch_http("https://h/d", max_bytes=100) == b"ok"

    def test_streaming_oversize_rejected_mid_read(self, monkeypatch, allow_ssrf):
        _patch_http_client(monkeypatch, [_FakeResp(200, chunks=[b"x" * 50, b"y" * 50])])
        with pytest.raises(ValueError, match="exceeds"):
            sources.fetch_http("https://h/d", max_bytes=80)

    def test_default_max_bytes_applied_when_none(self, monkeypatch, allow_ssrf):
        # body smaller than the 50MB default -> ok
        _patch_http_client(monkeypatch, [_FakeResp(200, body=b"small")])
        assert sources.fetch_http("https://h/d") == b"small"

    def test_retry_on_429_then_success(self, monkeypatch, allow_ssrf):
        client = _patch_http_client(monkeypatch, [_FakeResp(429), _FakeResp(429), _FakeResp(200, body=b"win")])
        assert sources.fetch_http("https://h/d") == b"win"
        assert len(client.calls) == 3  # 2 retries + success

    def test_retryable_status_exhausts_then_raises(self, monkeypatch, allow_ssrf):
        _patch_http_client(monkeypatch, [_FakeResp(503)] * 5)
        with pytest.raises(httpx.HTTPStatusError):
            sources.fetch_http("https://h/d")

    def test_transport_error_retry_then_success(self, monkeypatch, allow_ssrf):
        client = _patch_http_client(
            monkeypatch,
            [httpx.ConnectError("transient"), _FakeResp(200, body=b"ok")],
        )
        assert sources.fetch_http("https://h/d") == b"ok"
        assert len(client.calls) == 2

    def test_transport_error_exhausts(self, monkeypatch, allow_ssrf):
        _patch_http_client(monkeypatch, [httpx.ConnectError("dead")] * 5)
        with pytest.raises(httpx.ConnectError):
            sources.fetch_http("https://h/d")

    def test_redirect_followed_per_hop(self, monkeypatch, allow_ssrf):
        client = _patch_http_client(
            monkeypatch,
            [_FakeResp(301, location="https://h/final"), _FakeResp(200, body=b"landed")],
        )
        assert sources.fetch_http("https://h/start") == b"landed"
        assert client.calls[0].endswith("/start")
        assert client.calls[1].endswith("/final")

    def test_redirect_without_location_raises(self, monkeypatch, allow_ssrf):
        _patch_http_client(monkeypatch, [_FakeResp(301, location=None)])
        with pytest.raises(RuntimeError, match="no Location header"):
            sources.fetch_http("https://h/d")

    def test_too_many_redirects_raises(self, monkeypatch, allow_ssrf):
        # every hop redirects -> exhausts the redirect loop
        _patch_http_client(monkeypatch, [_FakeResp(301, location="https://h/n")] * 10)
        with pytest.raises(RuntimeError, match="too many redirects"):
            sources.fetch_http("https://h/d")

    def test_non_retryable_4xx_raises(self, monkeypatch, allow_ssrf):
        _patch_http_client(monkeypatch, [_FakeResp(404)])
        with pytest.raises(httpx.HTTPStatusError):
            sources.fetch_http("https://h/d")

    def test_timeout_is_clamped(self, monkeypatch, allow_ssrf):
        captured = {}

        def factory(*a, **kw):
            captured["timeout"] = kw.get("timeout")
            return _ScriptedClient([_FakeResp(200, body=b"ok")])

        monkeypatch.setattr(sources.httpx, "Client", factory)
        sources.fetch_http("https://h/d", timeout=99999)
        assert captured["timeout"] == sources.MAX_TIMEOUT  # clamped to ceiling

    def test_ssrf_checked_per_redirect_hop(self, monkeypatch):
        hops = []
        monkeypatch.setattr(sources, "_check_url_ssrf", lambda u: hops.append(u))
        _patch_http_client(
            monkeypatch,
            [_FakeResp(301, location="https://h/final"), _FakeResp(200, body=b"ok")],
        )
        sources.fetch_http("https://h/start")
        assert len(hops) == 2  # one SSRF check per hop


# --------------------------------------------------------------------------- #
# fetch_http_entry -- cache + fail-soft wrapper
# --------------------------------------------------------------------------- #
class TestFetchHttpEntry:
    def test_empty_url_yields_nothing(self):
        assert list(sources.fetch_http_entry({}, None, max_bytes=100)) == []

    def test_success_yields_name_and_bytes(self, monkeypatch, allow_ssrf):
        _patch_http_client(monkeypatch, [_FakeResp(200, body=b"data")])
        out = list(sources.fetch_http_entry({"url": "https://h/f.pdf"}, None, max_bytes=100))
        assert out == [("f.pdf", b"data")]

    def test_cache_hit_skips_fetch(self, monkeypatch, tmp_path):
        cache = sources.DocumentCache(str(tmp_path / "c"))
        key = sources.source_key({"source": "http", "url": "https://h/f.pdf"})
        cache.put(key, "cached.pdf", b"cached")
        called = {"n": 0}

        def fail(*a, **kw):
            called["n"] += 1
            raise AssertionError("should not fetch on cache hit")

        monkeypatch.setattr(sources, "fetch_http", fail)
        out = list(sources.fetch_http_entry({"url": "https://h/f.pdf"}, cache, max_bytes=100))
        assert out == [("cached.pdf", b"cached")]
        assert called["n"] == 0

    def test_failure_is_swallowed_and_yields_nothing(self, monkeypatch, allow_ssrf):
        def boom(*a, **kw):
            raise ValueError("network down")

        monkeypatch.setattr(sources, "fetch_http", boom)
        assert list(sources.fetch_http_entry({"url": "https://h/f"}, None, max_bytes=100)) == []

    def test_writes_cache_after_success(self, monkeypatch, allow_ssrf, tmp_path):
        _patch_http_client(monkeypatch, [_FakeResp(200, body=b"data")])
        cache = sources.DocumentCache(str(tmp_path / "c"))
        list(sources.fetch_http_entry({"url": "https://h/f.pdf"}, cache, max_bytes=100))
        assert cache.get(sources.source_key({"source": "http", "url": "https://h/f.pdf"})) == ("f.pdf", b"data")

    def test_cache_write_failure_swallowed(self, monkeypatch, allow_ssrf, tmp_path):
        _patch_http_client(monkeypatch, [_FakeResp(200, body=b"data")])
        cache = sources.DocumentCache(str(tmp_path / "c"))

        def fail_put(*a, **kw):
            raise OSError("disk")

        monkeypatch.setattr(cache, "put", fail_put)
        # still yields despite cache write failure
        assert list(sources.fetch_http_entry({"url": "https://h/f.pdf"}, cache, max_bytes=100)) == [("f.pdf", b"data")]


# --------------------------------------------------------------------------- #
# S3 fetch -- fake boto3 injected into sys.modules
# --------------------------------------------------------------------------- #
class _FakeS3Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, n: int) -> bytes:
        return self._data[:n]


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kw):
        yield from self._pages


class _FakeS3Client:
    def __init__(self, pages, objects=None):
        self._paginator = _FakePaginator(pages)
        self._objects = objects or {}

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return self._paginator

    def get_object(self, Bucket, Key):  # noqa: N803 -- mirrors the boto3 kwarg names the caller uses
        return {"Body": _FakeS3Body(self._objects[(Bucket, Key)])}


def _inject_boto3(monkeypatch, client):
    import sys

    fake_module = types.SimpleNamespace(client=lambda *a, **kw: client)
    monkeypatch.setitem(sys.modules, "boto3", fake_module)


class TestFetchS3:
    def test_missing_boto3_raises(self, monkeypatch):
        import sys

        # sys.modules[name] = None forces `import boto3` to raise ImportError
        monkeypatch.setitem(sys.modules, "boto3", None)
        with pytest.raises(LLMInvalidRequestError, match="rag-cloud"):
            list(sources.fetch_s3_entry({"bucket": "b", "key": "k"}, None))

    def test_missing_bucket_skips(self, monkeypatch):
        _inject_boto3(monkeypatch, _FakeS3Client([], {}))
        assert list(sources.fetch_s3_entry({}, None)) == []

    def test_yields_objects_and_ignores_folder_markers(self, monkeypatch):
        pages = [{"Contents": [{"Key": "folder/", "Size": 0}, {"Key": "a.txt", "Size": 3}]}]
        client = _FakeS3Client(pages, {("b", "a.txt"): b"abc"})
        _inject_boto3(monkeypatch, client)
        out = list(sources.fetch_s3_entry({"bucket": "b", "key": ""}, None))
        assert out == [("a.txt", b"abc")]

    def test_size_precheck_skips_oversize(self, monkeypatch, caplog):
        pages = [{"Contents": [{"Key": "big.bin", "Size": 9999}]}]
        _inject_boto3(monkeypatch, _FakeS3Client(pages, {}))
        with caplog.at_level("WARNING"):
            out = list(sources.fetch_s3_entry({"bucket": "b"}, None, max_bytes=10))
        assert out == []
        assert any("exceeds" in r.message for r in caplog.records)

    def test_get_object_error_skips_object(self, monkeypatch, caplog):
        pages = [{"Contents": [{"Key": "a.txt", "Size": 3}]}]

        class _Broken(_FakeS3Client):
            def get_object(self, Bucket, Key):  # noqa: N803
                raise RuntimeError("boom")

        _inject_boto3(monkeypatch, _Broken(pages, {}))
        with caplog.at_level("WARNING"):
            assert list(sources.fetch_s3_entry({"bucket": "b"}, None, max_bytes=100)) == []

    def test_read_exceeds_max_bytes_skips(self, monkeypatch, caplog):
        pages = [{"Contents": [{"Key": "a.txt", "Size": 3}]}]
        # body larger than max_bytes despite Size metadata
        _inject_boto3(monkeypatch, _FakeS3Client(pages, {("b", "a.txt"): b"way-too-large-payload"}))
        with caplog.at_level("WARNING"):
            assert list(sources.fetch_s3_entry({"bucket": "b"}, None, max_bytes=5)) == []

    def test_cache_hit_yields_cached(self, monkeypatch, tmp_path):
        pages = [{"Contents": [{"Key": "a.txt", "Size": 3}]}]
        _inject_boto3(monkeypatch, _FakeS3Client(pages, {("b", "a.txt"): b"abc"}))
        cache = sources.DocumentCache(str(tmp_path / "c"))
        # pre-seed cache so the object is a hit
        entry = {"bucket": "b", "key": "a.txt", "endpoint_url": "", "region": "auto"}
        ckey = sources.source_key({"source": "s3", "bucket": "b", "key": "a.txt", "endpoint_url": "", "region": "auto"})
        cache.put(ckey, "a.txt", b"cached")
        out = list(sources.fetch_s3_entry(entry, cache, max_bytes=100))
        assert out == [("a.txt", b"cached")]

    def test_no_objects_logs_warning(self, monkeypatch, caplog):
        _inject_boto3(monkeypatch, _FakeS3Client([{"Contents": []}], {}))
        with caplog.at_level("WARNING"):
            list(sources.fetch_s3_entry({"bucket": "b", "key": "x"}, None))
        assert any("no objects" in r.message for r in caplog.records)

    def test_paginator_failure_is_swallowed(self, monkeypatch, caplog):
        class _Broken:
            def get_paginator(self, name):
                raise RuntimeError("cred fail")

            def get_object(self, **kw):
                raise AssertionError

        _inject_boto3(monkeypatch, _Broken())
        with caplog.at_level("WARNING"):
            assert list(sources.fetch_s3_entry({"bucket": "b"}, None)) == []


# --------------------------------------------------------------------------- #
# Firecrawl -- fake httpx.Client (post/get)
# --------------------------------------------------------------------------- #
class _FirecrawlClient:
    def __init__(self, crawl_resp, poll_resps=None):
        self._crawl = crawl_resp
        self._poll = list(poll_resps or [])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None):
        return self._crawl

    def get(self, url, headers=None):
        return self._poll.pop(0)


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "x", request=httpx.Request("GET", "http://x"), response=httpx.Response(self.status_code)
            )

    def json(self):
        return self._payload


class TestFirecrawlCrawl:
    def test_sync_data_returned_directly(self, monkeypatch):
        monkeypatch.setattr(
            sources.httpx, "Client", lambda *a, **kw: _FirecrawlClient(_Resp(200, {"data": [{"markdown": "# hi"}]}))
        )
        assert sources._firecrawl_crawl("https://s", "k", 5, None) == [{"markdown": "# hi"}]

    def test_async_poll_until_completed(self, monkeypatch):
        crawl = _Resp(200, {"id": "job1"})
        polls = [_Resp(200, {"status": "scraping"}), _Resp(200, {"status": "completed", "data": [{"markdown": "x"}]})]
        monkeypatch.setattr(sources.httpx, "Client", lambda *a, **kw: _FirecrawlClient(crawl, polls))
        assert sources._firecrawl_crawl("https://s", "k", 5, None) == [{"markdown": "x"}]

    def test_no_job_id_and_no_data_raises(self, monkeypatch):
        monkeypatch.setattr(sources.httpx, "Client", lambda *a, **kw: _FirecrawlClient(_Resp(200, {"status": "ok"})))
        with pytest.raises(RuntimeError, match="no job id"):
            sources._firecrawl_crawl("https://s", "k", 5, None)

    def test_job_failed_raises(self, monkeypatch):
        crawl = _Resp(200, {"id": "j"})
        polls = [_Resp(200, {"status": "failed", "error": "boom"})]
        monkeypatch.setattr(sources.httpx, "Client", lambda *a, **kw: _FirecrawlClient(crawl, polls))
        with pytest.raises(RuntimeError, match="failed"):
            sources._firecrawl_crawl("https://s", "k", 5, None)

    def test_transport_error_raises(self, monkeypatch):
        class _C:
            def __enter__(self):
                return self

            def __exit__(self, *e):
                return False

            def post(self, *a, **k):
                raise httpx.ConnectError("down")

            def get(self, *a, **k):
                raise AssertionError

        monkeypatch.setattr(sources.httpx, "Client", lambda *a, **kw: _C())
        with pytest.raises(httpx.ConnectError):
            sources._firecrawl_crawl("https://s", "k", 5, None)


class TestFetchFirecrawlEntry:
    def test_empty_url_yields_nothing(self):
        assert list(sources.fetch_firecrawl_entry({}, None)) == []

    def test_missing_api_key_skips(self, monkeypatch, caplog):
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        with caplog.at_level("WARNING"):
            assert list(sources.fetch_firecrawl_entry({"url": "https://s"}, None)) == []
        assert any("api_key" in r.message for r in caplog.records)

    def test_ssrf_rejected_seed_skips(self, monkeypatch, caplog):
        def reject(url):
            raise ValueError("private")

        monkeypatch.setattr(sources, "_check_url_ssrf", reject)
        with caplog.at_level("WARNING"):
            assert list(sources.fetch_firecrawl_entry({"url": "https://s", "api_key": "k"}, None)) == []

    def test_crawl_failure_skips_entry(self, monkeypatch, caplog):
        monkeypatch.setattr(sources, "_check_url_ssrf", lambda url: None)

        def fail(*a, **kw):
            raise RuntimeError("api down")

        monkeypatch.setattr(sources, "_firecrawl_crawl", fail)
        with caplog.at_level("WARNING"):
            assert list(sources.fetch_firecrawl_entry({"url": "https://s", "api_key": "k"}, None)) == []

    def test_yields_markdown_pages(self, monkeypatch):
        monkeypatch.setattr(sources, "_check_url_ssrf", lambda url: None)
        pages = [
            {"markdown": "# a", "metadata": {"sourceURL": "https://s/a"}},
            {"html": "<p>b</p>", "metadata": {"sourceURL": "https://s/b"}},
        ]
        monkeypatch.setattr(sources, "_firecrawl_crawl", lambda *a, **kw: pages)
        out = list(sources.fetch_firecrawl_entry({"url": "https://s", "api_key": "k"}, None))
        names = [n for n, _ in out]
        assert names == ["a", "b"]

    def test_non_dict_page_skipped(self, monkeypatch):
        monkeypatch.setattr(sources, "_check_url_ssrf", lambda url: None)
        monkeypatch.setattr(
            sources, "_firecrawl_crawl", lambda *a, **kw: ["junk", {"markdown": "# ok", "metadata": {}}]
        )
        out = list(sources.fetch_firecrawl_entry({"url": "https://s", "api_key": "k"}, None))
        assert len(out) == 1

    def test_empty_content_page_skipped(self, monkeypatch):
        monkeypatch.setattr(sources, "_check_url_ssrf", lambda url: None)
        monkeypatch.setattr(
            sources, "_firecrawl_crawl", lambda *a, **kw: [{"markdown": "", "html": "", "metadata": {}}]
        )
        assert list(sources.fetch_firecrawl_entry({"url": "https://s", "api_key": "k"}, None)) == []

    def test_cache_hit_yields_cached(self, monkeypatch, tmp_path):
        monkeypatch.setattr(sources, "_check_url_ssrf", lambda url: None)
        cache = sources.DocumentCache(str(tmp_path / "c"))
        ckey = sources.source_key({"source": "firecrawl", "url": "https://s/a"})
        cache.put(ckey, "a", b"cached")
        monkeypatch.setattr(
            sources,
            "_firecrawl_crawl",
            lambda *a, **kw: [{"markdown": "# a", "metadata": {"sourceURL": "https://s/a"}}],
        )
        out = list(sources.fetch_firecrawl_entry({"url": "https://s", "api_key": "k"}, cache))
        assert out == [("a", b"cached")]

    def test_no_usable_pages_logs_warning(self, monkeypatch, caplog):
        monkeypatch.setattr(sources, "_check_url_ssrf", lambda url: None)
        monkeypatch.setattr(sources, "_firecrawl_crawl", lambda *a, **kw: [])
        with caplog.at_level("WARNING"):
            list(sources.fetch_firecrawl_entry({"url": "https://s", "api_key": "k"}, None))
        assert any("no usable pages" in r.message for r in caplog.records)

    def test_env_var_api_key_used(self, monkeypatch):
        seen = {}

        def fake_crawl(base_url, api_key, limit, endpoint):
            seen["key"] = api_key
            return []

        monkeypatch.setattr(sources, "_check_url_ssrf", lambda url: None)
        monkeypatch.setattr(sources, "_firecrawl_crawl", fake_crawl)
        monkeypatch.setenv("FIRECRAWL_API_KEY", "ENVKEY")
        list(sources.fetch_firecrawl_entry({"url": "https://s"}, None))
        assert seen["key"] == "ENVKEY"
