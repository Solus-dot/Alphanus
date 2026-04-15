from __future__ import annotations

import io
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest

from core.memory import VectorMemory
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager


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
    path = Path(__file__).resolve().parents[1] / "skills" / "search-ops" / "tools.py"
    spec = importlib.util.spec_from_file_location("search_ops_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env(provider: str = "tavily"):
    return SimpleNamespace(config={"search": {"provider": provider}})


def test_web_search_calls_tavily_and_normalizes_results(mocker, monkeypatch):
    module = _load_search_module()
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")

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

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._payload

    opened = {}

    def fake_urlopen(req, timeout=None):
        opened["url"] = req.full_url
        opened["auth"] = req.get_header("Authorization")
        opened["body"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    mocker.patch.object(module.urllib.request, "urlopen", side_effect=fake_urlopen)
    out = module.execute("web_search", {"query": "meta acquisitions", "limit": 2}, env=_env())

    assert opened["url"] == "https://api.tavily.com/search"
    assert opened["auth"] == "Bearer tvly-test-key"
    assert opened["body"]["query"] == "meta acquisitions"
    assert out["search_engine"] == "tavily"
    assert out["results"][0]["domain"] == "about.fb.com"
    assert out["results"][1]["snippet"] == "Reporting on Meta acquisitions."
    assert out["results"][0]["source_type"] == "official"
    assert out["results"][0]["rank"] == 1
    assert out["provider_chain"] == [{"provider": "tavily", "status": "ok"}]


def test_web_search_requires_api_key(monkeypatch):
    module = _load_search_module()
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Tavily API key not configured"):
        module.execute("web_search", {"query": "meta"}, env=_env())


def test_web_search_calls_brave_and_normalizes_results(mocker, monkeypatch):
    module = _load_search_module()
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-test-key")

    payload = {
        "web": {
            "results": [
                {
                    "title": "NIST CVE",
                    "url": "https://nvd.nist.gov/vuln/detail/CVE-2026-0001",
                    "description": "Official advisory details.",
                },
                {
                    "title": "Vendor advisory",
                    "url": "https://example.com/advisory",
                    "extra_snippets": ["Additional verified context."],
                },
            ]
        }
    }

    class _Resp:
        def __init__(self):
            self._payload = json.dumps(payload).encode("utf-8")
            self.headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._payload

    opened = {}

    def fake_urlopen(req, timeout=None):
        opened["url"] = req.full_url
        opened["token"] = req.get_header("X-subscription-token")
        return _Resp()

    mocker.patch.object(module.urllib.request, "urlopen", side_effect=fake_urlopen)
    out = module.execute("web_search", {"query": "latest cve", "limit": 2}, env=_env("brave"))

    assert opened["url"].startswith("https://api.search.brave.com/res/v1/web/search?")
    assert opened["token"] == "brave-test-key"
    assert out["search_engine"] == "brave"
    assert out["results"][0]["domain"] == "nvd.nist.gov"
    assert out["results"][1]["snippet"] == "Additional verified context."
    assert out["results"][0]["source_type"] == "official"


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

        def __exit__(self, exc_type, exc, tb):
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


def test_web_search_falls_back_to_secondary_provider(mocker, monkeypatch):
    module = _load_search_module()
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-test-key")

    payload = {
        "web": {
            "results": [
                {
                    "title": "Official status page",
                    "url": "https://status.example.com/update",
                    "description": "The current service status update.",
                }
            ]
        }
    }

    class _Resp:
        def __init__(self):
            self._payload = json.dumps(payload).encode("utf-8")
            self.headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._payload

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp())
    out = module.execute("web_search", {"query": "service status"}, env=_env("tavily"))

    assert out["provider"] == "brave"
    assert out["provider_chain"][0]["provider"] == "tavily"
    assert out["provider_chain"][0]["status"] == "error"
    assert out["provider_chain"][1] == {"provider": "brave", "status": "ok"}


def test_web_search_merges_results_from_both_providers_when_configured(mocker, monkeypatch):
    module = _load_search_module()
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "brave-test-key")

    tavily_payload = {
        "results": [
            {
                "title": "Official update",
                "url": "https://status.example.com/news?utm_source=feed",
                "content": "Primary source details.",
            }
        ]
    }
    brave_payload = {
        "web": {
            "results": [
                {
                    "title": "Official update mirror",
                    "url": "https://status.example.com/news",
                    "description": "Same source without tracking params.",
                },
                {
                    "title": "Independent coverage",
                    "url": "https://example.org/coverage",
                    "description": "Secondary source details.",
                },
            ]
        }
    }

    class _Resp:
        def __init__(self, payload):
            self._payload = json.dumps(payload).encode("utf-8")
            self.headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._payload

    def fake_urlopen(req, timeout=None):
        if req.full_url == "https://api.tavily.com/search":
            return _Resp(tavily_payload)
        if req.full_url.startswith("https://api.search.brave.com/res/v1/web/search?"):
            return _Resp(brave_payload)
        raise AssertionError(f"Unexpected URL: {req.full_url}")

    mocker.patch.object(module.urllib.request, "urlopen", side_effect=fake_urlopen)
    out = module.execute("web_search", {"query": "status update", "limit": 5}, env=_env("tavily"))

    assert out["search_engine"] == "multi"
    assert out["provider"] == "multi"
    assert out["providers_used"] == ["tavily", "brave"]
    assert [item["provider"] for item in out["provider_chain"]] == ["tavily", "brave"]
    assert all(item["status"] == "ok" for item in out["provider_chain"])
    assert len(out["results"]) == 2
    assert {item["canonical_url"] for item in out["results"]} == {
        "https://status.example.com/news",
        "https://example.org/coverage",
    }


def test_web_search_retries_retryable_http_error(mocker, monkeypatch):
    module = _load_search_module()
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
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

        def __exit__(self, exc_type, exc, tb):
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
                hdrs=None,
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
            hdrs={"Location": "http://127.0.0.1/admin"},
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

        def __exit__(self, exc_type, exc, tb):
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
                hdrs={"Location": "/final"},
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

        def __exit__(self, exc_type, exc, tb):
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

        def __exit__(self, exc_type, exc, tb):
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


def test_search_ops_skill_loads_and_executes_from_repo(tmp_path: Path, mocker, monkeypatch):
    module = _load_search_module()
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test-key")
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

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._payload

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp())

    runtime = SkillRuntime(
        skills_dir=str(Path(__file__).resolve().parents[1] / "skills"),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=VectorMemory(storage_path=str(mem)),
        config={"search": {"provider": "tavily"}},
    )
    skill = runtime.get_skill("search-ops")
    assert skill is not None

    ctx = SkillContext(
        user_input="search the web for example",
        branch_labels=[],
        attachments=[],
        workspace_root=str(ws),
        memory_hits=[],
    )
    out = runtime.execute_tool_call("web_search", {"query": "example"}, selected=[skill], ctx=ctx)
    assert out["ok"] is True
    assert out["data"]["results"][0]["url"] == "https://example.com"
