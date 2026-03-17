from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp(html))
    out = module.execute("fetch_url", {"url": "https://example.com/page", "max_chars": 1000}, env=_env())
    assert out["title"] == "Example Page"
    assert out["final_url"] == "https://example.com/final"
    assert "Readable content." in out["content"]
    assert out["truncated"] is False
    assert out["domain"] == "example.com"
    assert out["trust_score"] > 0


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
