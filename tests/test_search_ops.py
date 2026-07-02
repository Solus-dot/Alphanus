from __future__ import annotations

import importlib.util
import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from core.memory import LexicalMemory
from core.project import ProjectRuntime
from skills.runtime import SkillContext, SkillRuntime


class _Headers:
    def __init__(self, content_type: str = "application/json; charset=utf-8"):
        self._content_type = content_type

    def get(self, key, default=None):
        if key.lower() == "content-type":
            return self._content_type
        return default

    def get_content_charset(self):
        return "utf-8"


def _load_search_module():
    path = Path(__file__).resolve().parents[1] / "bundled-skills" / "search-ops" / "tools.py"
    spec = importlib.util.spec_from_file_location("search_ops_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(provider: str = "searxng"):
    return SimpleNamespace(
        config={
            "search": {
                "provider": provider,
                "fallback_provider": "",
                "searxng_base_url": "http://127.0.0.1:8888",
                "tavily_api_key_env": "TAVILY_API_KEY",
            },
            "retrieval": {"store_path": "/private/tmp/alphanus-test-retrieval.sqlite", "web_ttl_hours": 24},
        }
    )


def _env_with_store(path: Path, *, embeddings: bool = False):
    cfg = {
        "search": {"provider": "searxng", "fallback_provider": "", "searxng_base_url": "http://127.0.0.1:8888"},
        "retrieval": {"store_path": str(path), "web_ttl_hours": 24},
    }
    if embeddings:
        cfg["retrieval"]["embeddings"] = {
            "enabled": True,
            "base_url": "http://127.0.0.1:8080",
            "model": "local-embed",
            "api_key_env": "ALPHANUS_EMBEDDINGS_API_KEY",
        }
    return SimpleNamespace(config=cfg)


def _env_with_search_store(path: Path):
    return SimpleNamespace(
        config={
            "search": {
                "provider": "searxng",
                "fallback_provider": "",
                "searxng_base_url": "http://127.0.0.1:8888",
                "cache_first": True,
            },
            "retrieval": {"store_path": str(path), "web_ttl_hours": 24},
        }
    )


def _location_headers(location: str) -> Message:
    headers = Message()
    headers["Location"] = location
    return headers


def test_web_search_calls_searxng_and_normalizes_results(mocker):
    module = _load_search_module()

    payload = {
        "results": [
            {
                "title": "Meta Newsroom",
                "url": "https://about.fb.com/news/",
                "content": "Official Meta newsroom updates.",
            },
            {
                "title": "TechCrunch",
                "url": "https://techcrunch.com/example",
                "content": "Reporting on Meta acquisitions.",
            },
        ]
    }

    class _Resp:
        def __init__(self):
            self._payload = json.dumps(payload).encode("utf-8")
            self.headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._payload

    opened = {}

    def fake_urlopen(req, timeout=None):
        opened["url"] = req.full_url
        opened["accept"] = req.get_header("Accept")
        return _Resp()

    mocker.patch.object(module.urllib.request, "urlopen", side_effect=fake_urlopen)
    out = module.execute("web_search", {"query": "meta acquisitions", "limit": 2}, env=_env())

    assert opened["url"].startswith("http://127.0.0.1:8888/search?")
    assert "q=meta+acquisitions" in opened["url"]
    assert "format=json" in opened["url"]
    assert opened["accept"] == "application/json"
    assert out["search_engine"] == "searxng"
    assert out["results"][0]["domain"] == "about.fb.com"
    assert out["results"][1]["snippet"] == "Reporting on Meta acquisitions."
    assert out["results"][0]["source_type"] == "official"
    assert out["results"][0]["rank"] == 1
    assert out["attempts"][0]["provider"] == "searxng"
    assert out["attempts"][0]["status"] == "ok"
    assert out["evidence_quality"] in {"medium", "high"}
    assert out["failure_class"] == ""


def test_web_search_requires_searxng_base_url():
    module = _load_search_module()
    env = SimpleNamespace(config={"search": {"provider": "searxng"}})
    with pytest.raises(RuntimeError, match="SearXNG base URL not configured"):
        module.execute("web_search", {"query": "meta"}, env=env)


def test_web_search_rejects_unsupported_providers():
    module = _load_search_module()
    with pytest.raises(RuntimeError, match="Only SearXNG and Tavily search providers are supported"):
        module.execute("web_search", {"query": "latest cve", "limit": 2}, env=_env("brave"))


def test_fetch_url_extracts_title_and_text(mocker):
    module = _load_search_module()

    html = """
    <html>
      <head><title>Example Page</title><style>body { color: red; }</style></head>
      <body><article><h1>Hello</h1><p>Readable content.</p></article></body>
    </html>
    """

    class _Resp:
        def __init__(self, payload: str):
            self._payload = payload.encode("utf-8")
            self.headers = _Headers("text/html; charset=utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._payload

        def geturl(self):
            return "https://example.com/final"

    mocker.patch.object(module, "_open_no_redirect", return_value=_Resp(html))
    out = module.execute("fetch_url", {"url": "https://example.com/page", "max_chars": 1000}, env=_env())
    assert out["title"] == "Example Page"
    assert out["final_url"] == "https://example.com/final"
    assert "Readable content." in out["content"]
    assert out["truncated"] is False
    assert out["domain"] == "example.com"
    assert out["trust_score"] > 0
    assert out["excerpt"]
    assert out["best_passages"]
    assert out["usable_text"] is True


def test_fetch_url_marks_short_extraction_unusable(mocker):
    module = _load_search_module()

    html = "<html><head><title>Tiny</title></head><body><p>Short text.</p></body></html>"

    class _Resp:
        headers = _Headers("text/html; charset=utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return html.encode("utf-8")

        def geturl(self):
            return "https://example.com/tiny"

    env = _env()
    env.config["search"]["fetch_min_chars"] = 100
    mocker.patch.object(module, "_open_no_redirect", return_value=_Resp())

    out = module.execute("fetch_url", {"url": "https://example.com/tiny"}, env=env)

    assert out["usable_text"] is False
    assert out["failure_class"] == "fetch_unusable"


def test_fetch_url_respects_disabled_retrieval(mocker, tmp_path: Path):
    module = _load_search_module()
    db_path = tmp_path / "retrieval.sqlite"
    env = _env_with_store(db_path)
    env.config["retrieval"]["enabled"] = False
    html = """
    <html>
      <head><title>No Index</title></head>
      <body><article><p>This fetched page should not be persisted.</p></article></body>
    </html>
    """

    class _Resp:
        headers = _Headers("text/html; charset=utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return html.encode("utf-8")

        def geturl(self):
            return "https://example.com/no-index"

    mocker.patch.object(module, "_open_no_redirect", return_value=_Resp())

    out = module.execute("fetch_url", {"url": "https://example.com/no-index"}, env=env)
    stats = module.execute("retrieval_stats", {}, env=env)

    assert out["retrieval"] == {"indexed": False, "record_id": 0, "disabled": True}
    assert stats["backend"] == "disabled"
    assert stats["enabled"] is False
    assert not db_path.exists()


def test_fetch_url_indexes_embeddings_and_retrieve_uses_dense_query(mocker, tmp_path: Path):
    module = _load_search_module()
    env = _env_with_store(tmp_path / "retrieval.sqlite", embeddings=True)
    html = """
    <html>
      <head><title>Vector Page</title></head>
      <body><article><p>Semantic alpha release notes are available.</p></article></body>
    </html>
    """

    class _FetchResp:
        headers = _Headers("text/html; charset=utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return html.encode("utf-8")

        def geturl(self):
            return "https://example.com/vector"

    class _EmbeddingResp:
        headers = _Headers()

        def __init__(self, vector: list[float]):
            self._payload = json.dumps({"data": [{"embedding": vector}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._payload

    embedding_calls: list[dict[str, object]] = []

    def fake_urlopen(req, timeout=None):
        embedding_calls.append(json.loads(req.data.decode("utf-8")))
        return _EmbeddingResp([1.0, 0.0])

    mocker.patch.object(module, "_open_no_redirect", return_value=_FetchResp())
    mocker.patch.object(module.urllib.request, "urlopen", side_effect=fake_urlopen)

    fetched = module.execute("fetch_url", {"url": "https://example.com/vector"}, env=env)
    retrieved = module.execute("retrieve_knowledge", {"query": "meaning not lexically overlapping", "top_k": 1}, env=env)

    assert fetched["retrieval"]["embedding"]["stored"] == 1
    assert retrieved["backend"] == "sqlite_hybrid"
    assert retrieved["hits"][0]["title"] == "Vector Page"
    assert embedding_calls[0]["model"] == "local-embed"


def test_web_search_does_not_fall_back_without_fallback_provider(mocker):
    module = _load_search_module()
    mocker.patch.object(module.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline"))
    with pytest.raises(RuntimeError, match="SearXNG unreachable"):
        module.execute("web_search", {"query": "service status"}, env=_env())


def test_web_search_uses_fresh_cached_web_record_before_provider(mocker, tmp_path: Path):
    module = _load_search_module()
    env = _env_with_search_store(tmp_path / "retrieval.sqlite")
    store = module._retrieval_store(env)
    store.upsert_record(
        record_type="web_page",
        source="https://docs.example.com/status",
        canonical_source="https://docs.example.com/status",
        title="Official status documentation",
        text="Official status documentation says the service is healthy.",
        fetched_at=9999999999,
        ttl_seconds=3600,
        metadata={"domain": "docs.example.com", "source_type": "documentation"},
    )
    urlopen = mocker.patch.object(module.urllib.request, "urlopen")

    out = module.execute("web_search", {"query": "official status documentation", "limit": 3}, env=env)

    assert out["provider"] == "cache"
    assert out["attempts"] == []
    assert out["cache_hits"][0]["source"] == "https://docs.example.com/status"
    assert out["results"][0]["cached"] is True
    assert out["evidence_quality"] == "high"
    urlopen.assert_not_called()


def test_web_search_returns_stale_cache_as_degraded_when_provider_fails(mocker, tmp_path: Path):
    module = _load_search_module()
    env = _env_with_search_store(tmp_path / "retrieval.sqlite")
    store = module._retrieval_store(env)
    store.upsert_record(
        record_type="web_page",
        source="https://example.com/old",
        canonical_source="https://example.com/old",
        title="Old status page",
        text="Old status evidence.",
        fetched_at=1,
        ttl_seconds=1,
        metadata={"domain": "example.com", "source_type": "community"},
    )
    mocker.patch.object(module.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline"))

    out = module.execute("web_search", {"query": "old status evidence", "limit": 3}, env=env)

    assert out["provider"] == "cache"
    assert out["degraded"] is True
    assert out["failure_class"] == "network"
    assert out["attempts"][0]["provider"] == "searxng"
    assert out["attempts"][0]["status"] == "error"
    assert out["results"][0]["stale"] is True


def test_web_search_deduplicates_cached_chunks_before_cache_first_decision(mocker, tmp_path: Path):
    module = _load_search_module()
    env = _env_with_search_store(tmp_path / "retrieval.sqlite")
    env.config["search"]["min_usable_results"] = 2
    store = module._retrieval_store(env)
    store.upsert_record(
        record_type="web_page",
        source="https://docs.example.com/status",
        canonical_source="https://docs.example.com/status",
        title="Official status documentation",
        text=("Official status documentation says the service is healthy. " * 120),
        fetched_at=9999999999,
        ttl_seconds=3600,
        metadata={"domain": "docs.example.com", "source_type": "documentation"},
    )
    payload = {
        "results": [
            {
                "title": "Live status page",
                "url": "https://status.example.com/live",
                "content": "Live provider evidence.",
            }
        ]
    }

    class _Resp:
        headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    urlopen = mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp())

    out = module.execute("web_search", {"query": "official status documentation service healthy", "limit": 5}, env=env)

    assert urlopen.called
    assert len([item for item in out["results"] if item.get("cached")]) == 1
    assert len(out["cache_hits"]) == 1
    assert len({item["canonical_url"] for item in out["results"]}) == len(out["results"])


def test_web_search_preserves_stale_cache_metadata_when_provider_succeeds(mocker, tmp_path: Path):
    module = _load_search_module()
    env = _env_with_search_store(tmp_path / "retrieval.sqlite")
    store = module._retrieval_store(env)
    store.upsert_record(
        record_type="web_page",
        source="https://example.com/old",
        canonical_source="https://example.com/old",
        title="Old status page",
        text="Old status evidence.",
        fetched_at=1,
        ttl_seconds=1,
        metadata={"domain": "example.com", "source_type": "community"},
    )
    payload = {
        "results": [
            {
                "title": "Current official status",
                "url": "https://status.example.com/current",
                "content": "Current provider evidence.",
            }
        ]
    }

    class _Resp:
        headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp())

    out = module.execute("web_search", {"query": "old status evidence", "limit": 3}, env=env)

    stale = next(item for item in out["results"] if item["canonical_url"] == "https://example.com/old")
    assert stale["provider"] == "cache"
    assert stale["cached"] is True
    assert stale["stale"] is True
    assert stale["record_id"] > 0


def test_web_search_deduplicates_searxng_results(mocker):
    module = _load_search_module()
    payload = {
        "results": [
            {
                "title": "Official update",
                "url": "https://status.example.com/news?utm_source=feed",
                "content": "Primary source details.",
            },
            {
                "title": "Official update mirror",
                "url": "https://status.example.com/news",
                "content": "Same source without tracking params.",
            },
            {
                "title": "Independent coverage",
                "url": "https://example.org/coverage",
                "content": "Secondary source details.",
            },
        ]
    }

    class _Resp:
        def __init__(self):
            self._payload = json.dumps(payload).encode("utf-8")
            self.headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._payload

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp())
    out = module.execute("web_search", {"query": "status update", "limit": 5}, env=_env())

    assert out["search_engine"] == "searxng"
    assert out["provider"] == "searxng"
    assert len(out["results"]) == 2
    assert {item["canonical_url"] for item in out["results"]} == {
        "https://status.example.com/news",
        "https://example.org/coverage",
    }


def test_web_search_falls_back_to_tavily_when_searxng_unreachable(mocker, monkeypatch):
    module = _load_search_module()
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    payload = {
        "results": [
            {
                "title": "Fallback result",
                "url": "https://example.com/fallback",
                "content": "Tavily fallback snippet.",
            }
        ]
    }

    class _Resp:
        def __init__(self):
            self._payload = json.dumps(payload).encode("utf-8")
            self.headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._payload

    calls: list[str] = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if "/search?" in req.full_url:
            raise urllib.error.URLError("offline")
        assert req.full_url == "https://api.tavily.com/search"
        assert req.get_header("Authorization") == "Bearer tvly-test-key"
        return _Resp()

    env = _env()
    env.config["search"]["fallback_provider"] = "tavily"
    mocker.patch.object(module.urllib.request, "urlopen", side_effect=fake_urlopen)

    out = module.execute("web_search", {"query": "fallback status", "limit": 1}, env=env)

    assert calls[0].startswith("http://127.0.0.1:8888/search?")
    assert calls[1] == "https://api.tavily.com/search"
    assert out["provider"] == "tavily"
    assert out["attempts"][0]["provider"] == "searxng"
    assert out["attempts"][0]["status"] == "error"
    assert out["attempts"][0]["error"] == "SearXNG unreachable: offline"
    assert out["attempts"][1]["provider"] == "tavily"
    assert out["attempts"][1]["status"] == "ok"
    assert out["attempts"][0]["failure_class"] == "network"
    assert out["attempts"][1]["result_count"] == 1


def test_web_search_falls_back_to_tavily_when_searxng_url_missing(mocker, monkeypatch):
    module = _load_search_module()
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    payload = {"results": [{"title": "Fallback without SearXNG", "url": "https://example.com/fallback", "content": "Snippet."}]}

    class _Resp:
        headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    calls: list[str] = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return _Resp()

    env = _env()
    env.config["search"]["fallback_provider"] = "tavily"
    env.config["search"]["searxng_base_url"] = ""
    mocker.patch.object(module.urllib.request, "urlopen", side_effect=fake_urlopen)

    out = module.execute("web_search", {"query": "fallback status", "limit": 1}, env=env)

    assert calls == ["https://api.tavily.com/search"]
    assert out["provider"] == "tavily"
    assert out["attempts"][0]["provider"] == "searxng"
    assert out["attempts"][0]["status"] == "error"
    assert out["attempts"][0]["error"] == "SearXNG base URL not configured"
    assert out["attempts"][1]["provider"] == "tavily"
    assert out["attempts"][1]["status"] == "ok"


def test_web_search_can_use_tavily_as_primary(mocker, monkeypatch):
    module = _load_search_module()
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    payload = {"results": [{"title": "Primary Tavily", "url": "https://example.com/tavily", "content": "Snippet."}]}

    class _Resp:
        headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(payload).encode("utf-8")

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp())
    env = _env("tavily")

    out = module.execute("web_search", {"query": "primary", "limit": 1}, env=env)

    assert out["provider"] == "tavily"
    assert out["results"][0]["url"] == "https://example.com/tavily"
    assert out["attempts"][0]["provider"] == "tavily"
    assert out["attempts"][0]["status"] == "ok"


def test_web_search_classifies_invalid_provider_response(mocker):
    module = _load_search_module()

    class _Resp:
        headers = _Headers("text/html; charset=utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"<html>not json</html>"

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp())

    with pytest.raises(RuntimeError, match="non-JSON"):
        module.execute("web_search", {"query": "status update"}, env=_env())


def test_web_search_retries_retryable_http_error(mocker):
    module = _load_search_module()
    attempts = {"count": 0}
    payload = {
        "results": [
            {
                "title": "Service update",
                "url": "https://status.example.com/update",
                "content": "Recovered.",
            }
        ]
    }

    class _Resp:
        def __init__(self):
            self._payload = json.dumps(payload).encode("utf-8")
            self.headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._payload

    def fake_urlopen(req, timeout=None):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise urllib.error.HTTPError(
                req.full_url,
                503,
                "Service Unavailable",
                hdrs=cast(Message, Message()),
                fp=io.BytesIO(b"{}"),
            )
        return _Resp()

    mocker.patch.object(module.urllib.request, "urlopen", side_effect=fake_urlopen)
    out = module.execute("web_search", {"query": "status update"}, env=_env())

    assert attempts["count"] == 2
    assert out["results"][0]["url"] == "https://status.example.com/update"


def test_fetch_url_blocks_private_network_hosts(mocker):
    module = _load_search_module()
    called = {"value": False}

    def fake_open_no_redirect(_req, timeout_s=None):
        called["value"] = True
        raise AssertionError("open_no_redirect should not be called for private hosts")

    mocker.patch.object(module, "_open_no_redirect", side_effect=fake_open_no_redirect)

    with pytest.raises(RuntimeError, match="private or local network URL"):
        module.execute("fetch_url", {"url": "http://127.0.0.1/admin"}, env=_env())

    assert called["value"] is False


def test_fetch_url_blocks_redirect_to_private_network_host(mocker):
    module = _load_search_module()
    calls: list[str] = []

    def fake_open_no_redirect(req, timeout_s=None):
        calls.append(req.full_url)
        raise urllib.error.HTTPError(
            req.full_url,
            302,
            "Found",
            hdrs=cast(Message, _location_headers("http://127.0.0.1/admin")),
            fp=io.BytesIO(b""),
        )

    mocker.patch.object(module, "_open_no_redirect", side_effect=fake_open_no_redirect)

    with pytest.raises(RuntimeError, match="private or local network URL"):
        module.execute("fetch_url", {"url": "https://example.com/start"}, env=_env())

    assert calls == ["https://example.com/start"]


def test_fetch_url_follows_safe_redirect_chain(mocker):
    module = _load_search_module()
    calls: list[str] = []

    html = """
    <html>
      <head><title>Redirected Page</title></head>
      <body><p>Final redirected content.</p></body>
    </html>
    """

    class _Resp:
        def __init__(self, payload: str, final_url: str):
            self._payload = payload.encode("utf-8")
            self.headers = _Headers("text/html; charset=utf-8")
            self._final_url = final_url

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._payload

        def geturl(self):
            return self._final_url

    def fake_open_no_redirect(req, timeout_s=None):
        calls.append(req.full_url)
        if req.full_url == "https://example.com/start":
            raise urllib.error.HTTPError(
                req.full_url,
                302,
                "Found",
                hdrs=cast(Message, _location_headers("/final")),
                fp=io.BytesIO(b""),
            )
        if req.full_url == "https://example.com/final":
            return _Resp(html, "https://example.com/final")
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    mocker.patch.object(module, "_open_no_redirect", side_effect=fake_open_no_redirect)

    out = module.execute("fetch_url", {"url": "https://example.com/start", "max_chars": 1000}, env=_env())

    assert calls == ["https://example.com/start", "https://example.com/final"]
    assert out["final_url"] == "https://example.com/final"
    assert "Final redirected content." in out["content"]


def test_fetch_url_extracts_metadata_and_headings(mocker):
    module = _load_search_module()

    html = """
    <html>
      <head>
        <title>Example Page</title>
        <meta name="description" content="A useful page for testing." />
        <meta property="article:published_time" content="2026-03-27T12:00:00Z" />
        <meta name="author" content="Example Author" />
      </head>
      <body>
        <article>
          <h1>Hello</h1>
          <p>Readable content.</p>
          <p>More evidence here.</p>
        </article>
      </body>
    </html>
    """

    class _Resp:
        def __init__(self, payload: str):
            self._payload = payload.encode("utf-8")
            self.headers = _Headers("text/html; charset=utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._payload

        def geturl(self):
            return "https://example.com/final?utm_source=test"

    mocker.patch.object(module, "_open_no_redirect", return_value=_Resp(html))
    out = module.execute("fetch_url", {"url": "https://example.com/page", "max_chars": 1000}, env=_env())

    assert out["description"] == "A useful page for testing."
    assert out["author"] == "Example Author"
    assert out["published_at"].startswith("2026-03-27T12:00:00")
    assert out["headings"] == ["Hello"]
    assert out["canonical_url"] == "https://example.com/final"


def test_fetch_url_preserves_div_and_section_body_text(mocker):
    module = _load_search_module()

    html = """
    <html>
      <head><title>Mixed Layout</title></head>
      <body>
        <article>
          <h1>Situation Report</h1>
          <div>The body is rendered inside div tags.</div>
          <section>Further evidence appears inside a section block.</section>
        </article>
      </body>
    </html>
    """

    class _Resp:
        def __init__(self, payload: str):
            self._payload = payload.encode("utf-8")
            self.headers = _Headers("text/html; charset=utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._payload

        def geturl(self):
            return "https://example.com/article"

    mocker.patch.object(module, "_open_no_redirect", return_value=_Resp(html))
    out = module.execute("fetch_url", {"url": "https://example.com/article", "max_chars": 1000}, env=_env())

    assert "Situation Report" in out["content"]
    assert "The body is rendered inside div tags." in out["content"]
    assert "Further evidence appears inside a section block." in out["content"]


def test_source_type_does_not_treat_unofficial_or_officials_as_official():
    module = _load_search_module()

    assert module._source_type("mirror.example.com", "Unofficial mirror", "") == "community"
    assert module._source_type("news.example.com", "Officials warn about shortages", "") != "official"


def test_search_ops_skill_loads_and_executes_from_repo(tmp_path: Path, mocker):
    module = _load_search_module()
    home = tmp_path / "home"
    ws = home / "ws"
    mem = tmp_path / "mem.pkl"
    home.mkdir()
    ws.mkdir()

    payload = {
        "results": [
            {
                "title": "Example",
                "url": "https://example.com",
                "content": "Snippet",
            }
        ]
    }

    class _Resp:
        def __init__(self):
            self._payload = json.dumps(payload).encode("utf-8")
            self.headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self._payload

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp())

    runtime = SkillRuntime(
        skills_dir=str(Path(__file__).resolve().parents[1] / "bundled-skills"),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(mem)),
        config={
            "search": {"provider": "searxng", "searxng_base_url": "http://127.0.0.1:8888"},
            "retrieval": {"store_path": str(tmp_path / "retrieval.sqlite")},
        },
    )
    skill = runtime.get_skill("search-ops")
    assert skill is not None

    ctx = SkillContext(
        user_input="search the web for example",
        branch_labels=[],
        attachments=[],
        project_root=str(ws),
        memory_hits=[],
    )
    out = runtime.execute_tool_call("web_search", {"query": "example"}, selected=[skill], ctx=ctx)
    assert out["ok"] is True
    assert out["data"]["results"][0]["url"] == "https://example.com"
