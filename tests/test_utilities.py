from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path

from core.attachments import build_content
from core.memory import LexicalMemory
from core.skills import SkillContext, SkillRuntime
from core.workspace import WorkspaceManager


def _load_play_youtube_module():
    path = Path(__file__).resolve().parents[1] / "skills" / "utilities" / "tools.py"
    spec = importlib.util.spec_from_file_location("play_youtube_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_play_youtube_resolves_first_video_url(mocker):
    module = _load_play_youtube_module()

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"videoRenderer":{"videoId":"dQw4w9WgXcQ"}}'

    mocker.patch.object(module.urllib.request, "urlopen", return_value=_Resp())
    url, video_id, resolved = module._resolve_first_video_url("https://www.youtube.com/results?search_query=test")
    assert resolved is True
    assert video_id == "dQw4w9WgXcQ"
    assert url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&autoplay=1"


def test_play_youtube_falls_back_to_search_url_on_fetch_failure(mocker):
    module = _load_play_youtube_module()
    search_url = "https://www.youtube.com/results?search_query=test"
    mocker.patch.object(module.urllib.request, "urlopen", side_effect=RuntimeError("boom"))
    url, video_id, resolved = module._resolve_first_video_url(search_url)
    assert resolved is False
    assert video_id == ""
    assert url == search_url


def _runtime(tmp_path: Path) -> SkillRuntime:
    repo_root = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    return SkillRuntime(
        skills_dir=str(repo_root / "skills"),
        workspace=WorkspaceManager(str(ws), home_root=str(home)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )


def _ctx(workspace_root: str) -> SkillContext:
    return SkillContext(
        user_input="utility task",
        branch_labels=[],
        attachments=[],
        workspace_root=workspace_root,
        memory_hits=[],
    )


def test_get_weather_preserves_structured_network_error_in_runtime(mocker, tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("utilities")
    assert skill is not None

    reg = runtime._tool_registry["get_weather"]
    module_path = reg.module_path
    assert module_path is not None
    module = runtime._load_module(module_path, reg.module_name or "utilities_tools")  # noqa: SLF001
    assert module is not None
    reg.module = module
    mocker.patch.object(module.urllib.request, "urlopen", side_effect=urllib.error.URLError("offline"))

    out = runtime.execute_tool_call(
        "get_weather",
        {"city": "London"},
        selected=[skill],
        ctx=_ctx(str(runtime.workspace.workspace_root)),
    )

    assert out["ok"] is False
    assert out["error"]["code"] == "E_IO"
    assert out["error"]["message"] == "Weather service unreachable: offline"
    assert out["data"] == {"city": "London"}


def test_open_url_preserves_browser_failure_message_in_runtime(mocker, tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("utilities")
    assert skill is not None

    reg = runtime._tool_registry["open_url"]
    module_path = reg.module_path
    assert module_path is not None
    module = runtime._load_module(module_path, reg.module_name or "utilities_tools")  # noqa: SLF001
    assert module is not None
    reg.module = module
    mocker.patch.object(module.webbrowser, "open", return_value=False)

    out = runtime.execute_tool_call(
        "open_url",
        {"url": "https://example.com"},
        selected=[skill],
        ctx=_ctx(str(runtime.workspace.workspace_root)),
    )

    assert out["ok"] is False
    assert out["error"]["code"] == "E_IO"
    assert out["error"]["message"] == "Unable to open browser in this environment"
    assert out["data"] == {"url": "https://example.com"}


def test_build_content_places_user_text_before_image_parts(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"not-a-real-png")

    content = build_content("describe the image", [(str(image_path), "image")])

    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"].endswith("describe the image")
    assert content[1]["type"] == "image_url"


def test_open_url_accepts_file_urls_in_runtime(mocker, tmp_path: Path):
    runtime = _runtime(tmp_path)
    skill = runtime.get_skill("utilities")
    assert skill is not None

    reg = runtime._tool_registry["open_url"]
    module_path = reg.module_path
    assert module_path is not None
    module = runtime._load_module(module_path, reg.module_name or "utilities_tools")  # noqa: SLF001
    assert module is not None
    reg.module = module
    opened_urls: list[str] = []

    def _capture_open(url: str, new: int = 0):
        opened_urls.append(url)
        return True

    mocker.patch.object(module.webbrowser, "open", side_effect=_capture_open)
    file_url = "file:///Users/sohom/Desktop/Alphanus-Workspace/pomodoro-app/index.html"

    out = runtime.execute_tool_call(
        "open_url",
        {"url": file_url},
        selected=[skill],
        ctx=_ctx(str(runtime.workspace.workspace_root)),
    )

    assert out["ok"] is True
    assert out["data"] == {"url": file_url}
    assert opened_urls == [file_url]
