from __future__ import annotations

import importlib.util
from pathlib import Path

from core.memory import VectorMemory
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager


class _Headers:
    def get(self, key, default=None):
        if key.lower() == "content-type":
            return "text/html; charset=utf-8"
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


def test_web_search_parses_duckduckgo_results(mocker):
    module = _load_search_module()

    html = """
    <html><body>
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Falpha">Alpha Result</a>
      <a class="result__snippet">Alpha snippet text.</a>
      <a class="result__a" href="https://example.org/beta">Beta Result</a>
      <div class="result__snippet">Beta snippet text.</div>
    </body></html>
    """

    class _Resp:
        def __init__(self, payload: str):
            self._payload = payload.encode("utf-8")
            self.headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._payload

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp(html))
    out = module.execute("web_search", {"query": "alpha beta", "limit": 2}, env=None)
    assert out["query"] == "alpha beta"
    assert out["search_engine"] == "duckduckgo"
    assert out["results"][0]["url"] == "https://example.com/alpha"
    assert out["results"][0]["snippet"] == "Alpha snippet text."
    assert out["results"][1]["domain"] == "example.org"


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
            self.headers = _Headers()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return self._payload

        def geturl(self):
            return "https://example.com/final"

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp(html))
    out = module.execute("fetch_url", {"url": "https://example.com/page", "max_chars": 1000}, env=None)
    assert out["title"] == "Example Page"
    assert out["final_url"] == "https://example.com/final"
    assert "Readable content." in out["content"]
    assert out["truncated"] is False


def test_search_ops_skill_loads_and_executes_from_repo(tmp_path: Path, mocker):
    module = _load_search_module()
    home = tmp_path / "home"
    ws = home / "ws"
    mem = tmp_path / "mem.pkl"
    home.mkdir()
    ws.mkdir()

    class _Resp:
        def __init__(self):
            self._payload = (
                b'<a class="result__a" href="https://example.com">Example</a>'
                b'<div class="result__snippet">Snippet</div>'
            )
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
