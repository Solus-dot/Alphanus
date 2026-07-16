from __future__ import annotations

import os
from pathlib import Path

from core.memory import LexicalMemory
from core.project import ProjectRuntime
from skills.runtime import SkillContext, SkillRuntime


def _runtime(tmp_path: Path) -> SkillRuntime:
    return _runtime_with_config(tmp_path, {})


def _runtime_with_config(tmp_path: Path, config: dict[str, object]) -> SkillRuntime:
    repo_root = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    return SkillRuntime(
        skills_dir=str(repo_root / "bundled-skills"),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config=config,
    )


def _ctx(runtime: SkillRuntime, text: str = "desktop task") -> SkillContext:
    return SkillContext(
        user_input=text,
        branch_labels=[],
        attachments=[],
        project_root=str(runtime.project.project_root),
        memory_hits=[],
    )


def _execute(runtime: SkillRuntime, skill_id: str, tool_name: str, args: dict[str, object]) -> dict[str, object]:
    skill = runtime.get_skill(skill_id)
    assert skill is not None
    return runtime.execute_tool_call(tool_name, args, selected=[skill], ctx=_ctx(runtime))


def _load_tool_module(runtime: SkillRuntime, tool_name: str):
    reg = runtime._tool_registry[tool_name]  # noqa: SLF001
    assert reg.module_path is not None
    module = runtime._load_module(reg.module_path, reg.module_name or f"{reg.skill_id}_tools")  # noqa: SLF001
    assert module is not None
    reg.module = module
    return module


def test_new_desktop_skills_expose_expected_tools(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    selected = runtime.skills_by_ids(["app-control", "browser-control", "local-search", "document-tools", "screenshot-ocr"])
    names = set(runtime.allowed_tool_names(selected, ctx=_ctx(runtime)))

    assert {
        "list_apps",
        "open_app",
        "focus_app",
        "quit_app",
        "open_browser_url",
        "browser_search",
        "get_current_browser_page",
        "search_local_files",
        "extract_document_text",
        "extract_document_tables",
        "capture_screenshot",
        "ocr_image",
        "capture_and_ocr",
    }.issubset(names)


def test_side_effecting_desktop_tools_count_as_mutating_for_plan_mode(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    assert runtime.tool_is_mutating("open_app") is True
    assert runtime.tool_is_mutating("focus_app") is True
    assert runtime.tool_is_mutating("quit_app") is True
    assert runtime.tool_is_mutating("open_browser_url") is True
    assert runtime.tool_is_mutating("browser_search") is True
    assert runtime.tool_is_mutating("capture_screenshot") is True
    assert runtime.tool_is_mutating("capture_and_ocr") is True

    assert runtime.tool_is_mutating("list_apps") is False
    assert runtime.tool_is_mutating("get_current_browser_page") is False
    assert runtime.tool_is_mutating("search_local_files") is False
    assert runtime.tool_is_mutating("extract_document_text") is False
    assert runtime.tool_is_mutating("ocr_image") is False


def test_bundled_tools_declare_mutability_and_actions(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    bundled_tools = [
        reg
        for reg in runtime._tool_registry.values()  # noqa: SLF001
        if reg.skill_id != "__runtime__" and runtime.get_skill(reg.skill_id) is not None
    ]

    assert bundled_tools
    missing_mutates = sorted(reg.name for reg in bundled_tools if reg.mutates is None)
    missing_actions = sorted(reg.name for reg in bundled_tools if not reg.actions)
    assert missing_mutates == []
    assert missing_actions == []


def test_app_control_mutating_actions_require_explicit_confirmation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    open_out = _execute(runtime, "app-control", "open_app", {"name": "Calculator"})
    focus_out = _execute(runtime, "app-control", "focus_app", {"name": "Calculator"})
    quit_out = _execute(runtime, "app-control", "quit_app", {"name": "Calculator"})

    assert open_out["ok"] is False
    assert open_out["error"]["code"] == "E_CONFIRMATION_REQUIRED"  # type: ignore[index]
    assert focus_out["ok"] is False
    assert focus_out["error"]["code"] == "E_CONFIRMATION_REQUIRED"  # type: ignore[index]
    assert quit_out["ok"] is False
    assert quit_out["error"]["code"] == "E_CONFIRMATION_REQUIRED"  # type: ignore[index]


def test_app_control_parses_macos_running_apps(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    module = _load_tool_module(runtime, "list_apps")
    mocker.patch.object(module.platform, "system", return_value="Darwin")

    class _Proc:
        returncode = 0
        stdout = """
 1) "Finder" ASN:0x0-0x1:
    bundleID="com.apple.finder"
    pid = 1 type="Foreground" flavor=3
 2) "backgroundd" ASN:0x0-0x2:
    pid = 2 type="BackgroundOnly" flavor=3
 3) "Terminal" ASN:0x0-0x3:
    bundleID="com.apple.Terminal"
    pid = 3 type="Foreground" flavor=3
"""
        stderr = ""

    mocker.patch.object(module.subprocess, "run", return_value=_Proc())

    out = _execute(runtime, "app-control", "list_apps", {})

    assert out["ok"] is True
    assert out["data"]["platform"] == "darwin"  # type: ignore[index]
    assert out["data"]["apps"] == ["Finder", "Terminal"]  # type: ignore[index]


def test_app_control_focus_uses_open_without_shell_interpolation(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    module = _load_tool_module(runtime, "focus_app")
    mocker.patch.object(module.platform, "system", return_value="Darwin")
    commands: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    def _capture_run(cmd: list[str], **_kwargs):
        commands.append(cmd)
        return _Proc()

    mocker.patch.object(module.subprocess, "run", side_effect=_capture_run)
    name = 'Bad" & do shell script "touch /tmp/pwned" & "'

    out = _execute(runtime, "app-control", "focus_app", {"name": name, "confirm_focus": True})

    assert out["ok"] is True
    assert commands[-1] == ["open", "-a", name]



def test_browser_open_and_search_require_confirmation_before_launching(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    open_out = _execute(runtime, "browser-control", "open_browser_url", {"url": "https://example.com"})
    search_out = _execute(runtime, "browser-control", "browser_search", {"query": "alphanus"})

    assert open_out["ok"] is False
    assert open_out["error"]["code"] == "E_CONFIRMATION_REQUIRED"  # type: ignore[index]
    assert search_out["ok"] is False
    assert search_out["error"]["code"] == "E_CONFIRMATION_REQUIRED"  # type: ignore[index]


def test_browser_search_opens_expected_engine_url(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    module = _load_tool_module(runtime, "browser_search")
    opened: list[str] = []
    mocker.patch.object(module.webbrowser, "open", side_effect=lambda url, *_args, **_kwargs: opened.append(url) or True)

    out = _execute(runtime, "browser-control", "browser_search", {"query": "alpha beta", "engine": "duckduckgo", "confirm_open": True})

    assert out["ok"] is True
    assert opened == ["https://duckduckgo.com/?q=alpha+beta"]
    assert out["data"]["url"] == opened[0]  # type: ignore[index]


def test_browser_current_page_rejects_unknown_browser_without_applescript(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    module = _load_tool_module(runtime, "get_current_browser_page")
    mocker.patch.object(module.platform, "system", return_value="Darwin")
    run = mocker.patch.object(module.subprocess, "run")

    out = _execute(runtime, "browser-control", "get_current_browser_page", {"browser": 'Safari"\ndo shell script "touch /tmp/pwned"\n"'})

    assert out["ok"] is False
    assert out["error"]["code"] == "E_VALIDATION"  # type: ignore[index]
    run.assert_not_called()


def test_browser_current_page_uses_linefeed_in_macos_applescript(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    module = _load_tool_module(runtime, "get_current_browser_page")
    mocker.patch.object(module.platform, "system", return_value="Darwin")
    commands: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = "Example\nhttps://example.com\n"
        stderr = ""

    def _capture_run(cmd: list[str], **_kwargs):
        commands.append(cmd)
        return _Proc()

    mocker.patch.object(module.subprocess, "run", side_effect=_capture_run)

    out = _execute(runtime, "browser-control", "get_current_browser_page", {"browser": "chrome"})

    assert out["ok"] is True
    script = commands[-1][2]
    assert "return pageTitle & linefeed & pageUrl" in script
    assert 'return pageTitle & "\n" & pageUrl' not in script


def test_browser_current_page_skips_timed_out_macos_candidate(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    module = _load_tool_module(runtime, "get_current_browser_page")
    mocker.patch.object(module.platform, "system", return_value="Darwin")

    class _Proc:
        returncode = 0
        stdout = "Example Domain\nhttps://example.com\n"
        stderr = ""

    run = mocker.patch.object(
        module.subprocess,
        "run",
        side_effect=[module.subprocess.TimeoutExpired(["osascript"], 5), _Proc()],
    )

    out = _execute(runtime, "browser-control", "get_current_browser_page", {})

    assert out["ok"] is True
    assert out["data"]["browser"] == "Google Chrome"  # type: ignore[index]
    assert run.call_count == 2


def test_local_search_finds_filename_and_content_matches(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    project = Path(runtime.project.project_root)
    (project / "notes").mkdir()
    (project / "notes" / "release-notes.txt").write_text("alpha beta gamma\nsecond line\n", encoding="utf-8")
    (project / "todo.txt").write_text("remember alpha release\n", encoding="utf-8")

    out = _execute(
        runtime,
        "local-search",
        "search_local_files",
        {"query": "alpha", "root": str(project), "mode": "both", "max_results": 10},
    )

    assert out["ok"] is True
    matches = out["data"]["matches"]  # type: ignore[index]
    paths = {Path(str(item["path"])).name for item in matches}  # type: ignore[index]
    assert {"release-notes.txt", "todo.txt"}.issubset(paths)
    assert len(matches) >= 2


def test_local_search_skips_symlinked_files_outside_allowed_roots(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    project = Path(runtime.project.project_root)
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("needle outside policy\n", encoding="utf-8")
    link = project / "linked-secret.txt"

    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        return

    out = _execute(
        runtime,
        "local-search",
        "search_local_files",
        {"query": "needle", "root": str(project), "mode": "content", "max_results": 10},
    )

    assert out["ok"] is True
    assert out["data"]["matches"] == []  # type: ignore[index]


def test_local_search_skips_sensitive_project_paths_and_ignores_home(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    project = Path(runtime.project.project_root)
    home = project.parent
    ssh_dir = home / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "notes.txt").write_text("needle from ssh\n", encoding="utf-8")
    (project / ".env.json").write_text('{"token": "needle"}\n', encoding="utf-8")
    (project / "safe.txt").write_text("needle from safe file\n", encoding="utf-8")

    content_out = _execute(runtime, "local-search", "search_local_files", {"query": "needle", "mode": "both", "max_results": 10})
    filename_out = _execute(runtime, "local-search", "search_local_files", {"query": "env", "mode": "filename", "max_results": 10})

    assert content_out["ok"] is True
    content_matches = content_out["data"]["matches"]  # type: ignore[index]
    assert [Path(str(item["path"])).name for item in content_matches] == ["safe.txt"]
    assert filename_out["ok"] is True
    assert filename_out["data"]["matches"] == []  # type: ignore[index]


def test_local_search_uses_configured_max_text_bytes(tmp_path: Path) -> None:
    runtime = _runtime_with_config(tmp_path, {"tools": {"local_search": {"max_text_bytes": 5}}})
    project = Path(runtime.project.project_root)
    (project / "small.txt").write_text("needle\n", encoding="utf-8")

    out = _execute(
        runtime,
        "local-search",
        "search_local_files",
        {"query": "needle", "root": str(project), "mode": "content", "max_results": 10},
    )

    assert out["ok"] is True
    assert out["data"]["matches"] == []  # type: ignore[index]
    assert out["data"]["skipped_large"] == 1  # type: ignore[index]


def test_local_search_rejects_roots_outside_project(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()

    out = _execute(runtime, "local-search", "search_local_files", {"query": "alpha", "root": str(outside)})

    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"  # type: ignore[index]


def test_document_tools_extract_text_and_csv_tables(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    project = Path(runtime.project.project_root)
    text_path = project / "doc.txt"
    csv_path = project / "data.csv"
    text_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    csv_path.write_text("name,count\nalpha,2\nbeta,5\n", encoding="utf-8")

    text_out = _execute(runtime, "document-tools", "extract_document_text", {"path": str(text_path), "max_chars": 8})
    table_out = _execute(runtime, "document-tools", "extract_document_tables", {"path": str(csv_path), "max_rows": 2})

    assert text_out["ok"] is True
    assert text_out["data"]["text"] == "alpha\nbe"  # type: ignore[index]
    assert text_out["data"]["truncated"] is True  # type: ignore[index]
    assert table_out["ok"] is True
    assert table_out["data"]["rows"] == [["name", "count"], ["alpha", "2"]]  # type: ignore[index]
    assert table_out["data"]["truncated"] is True  # type: ignore[index]


def test_document_tools_reject_external_and_sensitive_project_paths(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    project = Path(runtime.project.project_root)
    home = project.parent
    ssh_dir = home / ".ssh"
    ssh_dir.mkdir()
    home_secret = ssh_dir / "notes.txt"
    project_secret = project / ".env.json"
    home_secret.write_text("secret\n", encoding="utf-8")
    project_secret.write_text('{"secret": true}\n', encoding="utf-8")

    for path in (home_secret, project_secret):
        out = _execute(runtime, "document-tools", "extract_document_text", {"path": str(path)})

        assert out["ok"] is False
        assert out["error"]["code"] == "E_POLICY"  # type: ignore[index]


def test_document_tools_report_optional_dependency_for_pdf_when_missing(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    module = _load_tool_module(runtime, "extract_document_text")
    pdf_path = Path(runtime.project.project_root) / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    mocker.patch.object(module, "_read_pdf", return_value=module._err("E_DEPENDENCY", "PDF extraction requires pypdf", {}))

    out = _execute(runtime, "document-tools", "extract_document_text", {"path": str(pdf_path)})

    assert out["ok"] is False
    assert out["error"]["code"] == "E_DEPENDENCY"  # type: ignore[index]


def test_screenshot_capture_requires_confirmation(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    capture_out = _execute(runtime, "screenshot-ocr", "capture_screenshot", {})
    combined_out = _execute(runtime, "screenshot-ocr", "capture_and_ocr", {})

    assert capture_out["ok"] is False
    assert capture_out["error"]["code"] == "E_CONFIRMATION_REQUIRED"  # type: ignore[index]
    assert combined_out["ok"] is False
    assert combined_out["error"]["code"] == "E_CONFIRMATION_REQUIRED"  # type: ignore[index]


def test_screenshot_ocr_rejects_missing_image_path(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    out = _execute(runtime, "screenshot-ocr", "ocr_image", {"path": str(Path(runtime.project.project_root) / "missing.png")})

    assert out["ok"] is False
    assert out["error"]["code"] == "E_NOT_FOUND"  # type: ignore[index]


def test_screenshot_ocr_rejects_sensitive_image_paths(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    project = Path(runtime.project.project_root)
    home = project.parent
    ssh_dir = home / ".ssh"
    protected_dir = project / ".alphanus"
    ssh_dir.mkdir()
    protected_dir.mkdir()
    paths = [ssh_dir / "screen.png", project / ".env.png", protected_dir / "screen.png"]
    for path in paths:
        path.write_bytes(b"not actually a png")

    for path in paths:
        out = _execute(runtime, "screenshot-ocr", "ocr_image", {"path": str(path)})

        assert out["ok"] is False
        assert out["error"]["code"] == "E_POLICY"  # type: ignore[index]


def test_screenshot_capture_rejects_protected_output_path(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    protected_dir = Path(runtime.project.project_root) / ".alphanus"
    protected_dir.mkdir()

    out = _execute(
        runtime,
        "screenshot-ocr",
        "capture_screenshot",
        {"output_path": str(protected_dir / "screen.png"), "confirm_capture": True},
    )

    assert out["ok"] is False
    assert out["error"]["code"] == "E_POLICY"  # type: ignore[index]


def test_screenshot_capture_explains_macos_screen_recording_denial(mocker, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    module = _load_tool_module(runtime, "capture_screenshot")
    mocker.patch.object(module.platform, "system", return_value="Darwin")

    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "could not create image from display"

    mocker.patch.object(module.subprocess, "run", return_value=_Proc())
    out = _execute(
        runtime,
        "screenshot-ocr",
        "capture_screenshot",
        {"output_path": str(Path(runtime.project.project_root) / "screen.png"), "confirm_capture": True},
    )

    assert out["ok"] is False
    assert out["error"]["code"] == "E_PERMISSION"  # type: ignore[index]
    assert "Screen Recording" in out["error"]["message"]  # type: ignore[index]
