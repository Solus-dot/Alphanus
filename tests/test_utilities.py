from __future__ import annotations

import importlib.util
import urllib.error
import urllib.parse
from pathlib import Path

from core.attachments import build_content, classify_attachment, image_mime_type
from core.memory import LexicalMemory
from core.workspace import WorkspaceManager
from skills.runtime import SkillContext, SkillRuntime


def _load_play_youtube_module():
    path = Path(__file__).resolve().parents[1] / "bundled-skills" / "utilities" / "tools.py"
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

        def __exit__(self, *_args):
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
        skills_dir=str(repo_root / "bundled-skills"),
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


def test_classify_attachment_accepts_utf8_text_without_known_extension(tmp_path: Path) -> None:
    path = tmp_path / "module.cxx"
    path.write_text("#include <iostream>\nint main() { return 0; }\n", encoding="utf-8")

    assert classify_attachment(str(path)) == "text"


def test_attachment_image_mime_uses_standard_mimetype_registry(tmp_path: Path) -> None:
    path = tmp_path / "photo.png"
    path.write_bytes(b"not-a-real-png")

    assert classify_attachment(str(path)) == "image"
    assert image_mime_type(str(path)) == "image/png"


def test_classify_attachment_rejects_binary_and_invalid_utf8(tmp_path: Path) -> None:
    binary_path = tmp_path / "artifact.bin"
    invalid_path = tmp_path / "legacy.data"
    binary_path.write_bytes(b"\x00\x01valid utf-8 around a nul byte")
    invalid_path.write_bytes(b"\xff\xfe\xfd")

    assert classify_attachment(str(binary_path)) == "unknown"
    assert classify_attachment(str(invalid_path)) == "unknown"


def test_classify_attachment_rejects_late_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "late-invalid.cxx"
    path.write_bytes(("int main() { return 0; }\n" * 300).encode("utf-8") + b"\xff")

    assert classify_attachment(str(path)) == "unknown"


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

    def _capture_open(url: str, *_args, **_kwargs):
        opened_urls.append(url)
        return True

    mocker.patch.object(module.webbrowser, "open", side_effect=_capture_open)
    target = tmp_path / "pomodoro-app" / "index.html"
    target.parent.mkdir()
    target.write_text("<!doctype html>\n", encoding="utf-8")
    file_url = urllib.parse.urljoin("file:", urllib.parse.quote(str(target)))

    out = runtime.execute_tool_call(
        "open_url",
        {"url": file_url},
        selected=[skill],
        ctx=_ctx(str(runtime.workspace.workspace_root)),
    )

    assert out["ok"] is True
    assert out["data"] == {"url": file_url}
    assert opened_urls == [file_url]
