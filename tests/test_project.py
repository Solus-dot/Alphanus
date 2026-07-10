from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from core.project import ProjectRuntime


def _require_git() -> str:
    git_path = shutil.which("git")
    if git_path is None:
        pytest.skip("git is not available")
    return git_path


def _filesystem_is_case_insensitive(root: Path) -> bool:
    fd, probe_text = tempfile.mkstemp(prefix=".alphanusCaseProbe", dir=str(root))
    os.close(fd)
    probe = Path(probe_text)
    try:
        alt_probe = probe.with_name(probe.name.swapcase())
        if alt_probe.name == probe.name:
            return False
        return alt_probe.exists() and alt_probe.samefile(probe)
    except OSError:
        return False
    finally:
        try:
            probe.unlink()
        except OSError:
            pass


def test_write_path_traversal_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    mgr = ProjectRuntime(str(ws))

    with pytest.raises(PermissionError):
        mgr.create_file("../escape.txt", "x")


def test_absolute_write_outside_project_allowed(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    outside = home / "outside"
    home.mkdir()
    ws.mkdir()
    outside.mkdir()
    mgr = ProjectRuntime(str(ws))

    target = outside / "outside.txt"

    assert mgr.create_file(str(target), "x") == str(target.resolve())
    assert target.read_text(encoding="utf-8") == "x"


def test_absolute_read_outside_project_allowed(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    outside = home / "outside"
    home.mkdir()
    ws.mkdir()
    outside.mkdir()
    target = outside / "outside.txt"
    target.write_text("alpha", encoding="utf-8")

    mgr = ProjectRuntime(str(ws))

    assert mgr.read_file(str(target)) == "alpha"


def test_restricted_secret_reads_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    ssh = home / ".ssh"
    ssh.mkdir(parents=True)
    (ssh / "id_rsa").write_text("secret", encoding="utf-8")
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    with pytest.raises(PermissionError):
        mgr.read_file(str(ssh / "id_rsa"))


def test_symlink_escape_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    outside = home / "outside"
    home.mkdir()
    ws.mkdir()
    outside.mkdir()

    link = ws / "link"
    link.symlink_to(outside, target_is_directory=True)

    mgr = ProjectRuntime(str(ws))
    with pytest.raises(PermissionError):
        mgr.create_file("link/evil.txt", "boom")


def test_absolute_symlink_escape_under_project_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    outside = home / "outside"
    home.mkdir()
    ws.mkdir()
    outside.mkdir()
    target = outside / "secret.txt"
    target.write_text("secret", encoding="utf-8")

    link = ws / "link.txt"
    link.symlink_to(target)

    mgr = ProjectRuntime(str(ws))
    with pytest.raises(PermissionError):
        mgr.read_file(str(link))
    with pytest.raises(PermissionError):
        mgr.create_file(str(link), "boom")


def test_absolute_write_to_protected_external_state_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    state = home / ".alphanus"
    repo_git = home / "repo" / ".git"
    state.mkdir(parents=True)
    repo_git.mkdir(parents=True)
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))

    with pytest.raises(PermissionError):
        mgr.create_file(str(state / "secret.txt"), "x")
    with pytest.raises(PermissionError):
        mgr.create_file(str(repo_git / "config"), "x")


def test_missing_project_root_is_rejected_without_creation(tmp_path: Path):
    missing = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        ProjectRuntime(str(missing))

    assert not missing.exists()


def test_shell_command_requires_approval_for_shell_boundaries(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))

    assert mgr.shell_command_requires_approval("echo ok; rm -rf build") is True
    assert mgr.shell_command_requires_approval("true && git reset --hard") is True
    assert mgr.shell_command_requires_approval("printf 'b\\na\\n' | sort") is True
    assert mgr.shell_command_requires_approval("echo $(rm -rf build)") is True
    assert mgr.shell_command_requires_approval("echo `rm -rf build`") is True
    assert mgr.shell_command_requires_approval("sh -c 'rm -rf build'") is True
    assert mgr.shell_command_requires_approval("bash -lc 'echo ok'") is True
    assert mgr.shell_command_requires_approval("sudo rm -rf build") is True
    assert mgr.shell_command_requires_approval("env FOO=bar npm install") is True
    assert mgr.shell_command_requires_approval("python3 -c \"print('ok; still quoted')\"") is False
    assert mgr.shell_command_requires_approval("echo ok") is False


def test_shell_command_denies_unapproved_compound_command(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("echo ok; echo done")

    assert res["ok"] is False
    assert res["error"]["code"] == "E_POLICY"
    assert "requires approval" in res["error"]["message"]


def test_shell_command_allows_chaining(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("echo ok; echo done", approved=True)
    assert res["ok"] is True
    assert res["data"]["stdout"].splitlines() == ["ok", "done"]


def test_shell_command_allows_and_chaining_stop_on_failure(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("false && echo skipped", approved=True)
    assert res["ok"] is False
    assert res["error"]["code"] == "E_SHELL"
    assert "skipped" not in res["data"]["stdout"]


def test_project_reads_allowed_when_project_is_outside_user_home(tmp_path: Path):
    home = tmp_path / "home"
    ws_parent = tmp_path / "external"
    ws = ws_parent / "ws"
    home.mkdir()
    ws.mkdir(parents=True)
    target = ws / "notes.txt"
    target.write_text("alpha", encoding="utf-8")

    mgr = ProjectRuntime(str(ws))
    assert mgr.read_file(str(target)) == "alpha"
    assert mgr.list_files(str(ws)) == ["notes.txt"]


def test_resolve_text_section_accepts_reported_line_count_on_newline_terminated_files(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    mgr = ProjectRuntime(str(ws))

    content = "alpha\nbeta\n"
    resolved = mgr.resolve_text_section(content, start_line=3, end_line=3)

    assert resolved["total_line_count"] == 3
    assert resolved["start_line"] == 3
    assert resolved["end_line"] == 3
    assert resolved["content"] == ""


def test_protected_internal_state_is_hidden_from_project_views(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    (ws / ".alphanus" / "sessions").mkdir(parents=True)
    (ws / "notes.txt").write_text("alpha", encoding="utf-8")

    mgr = ProjectRuntime(str(ws))

    assert mgr.list_files(".") == ["notes.txt"]
    assert ".alphanus" not in mgr.project_tree()["tree"]


def test_symlink_alias_to_protected_state_is_hidden_from_project_views(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    (ws / ".alphanus" / "sessions").mkdir(parents=True)
    (ws / "alias").symlink_to(ws / ".alphanus", target_is_directory=True)
    (ws / "notes.txt").write_text("alpha", encoding="utf-8")

    mgr = ProjectRuntime(str(ws))

    assert mgr.list_files(".") == ["notes.txt"]
    assert "alias" not in mgr.project_tree()["tree"]


def test_read_protected_internal_state_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    target = ws / ".alphanus" / "sessions" / "manifest.json"
    home.mkdir()
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")

    mgr = ProjectRuntime(str(ws))

    with pytest.raises(PermissionError, match="protected internal state"):
        mgr.read_file(".alphanus/sessions/manifest.json")


def test_case_insensitive_protected_state_paths_are_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    target = ws / ".alphanus" / "secret.txt"
    home.mkdir()
    target.parent.mkdir(parents=True)
    target.write_text("secret", encoding="utf-8")
    if not _filesystem_is_case_insensitive(ws):
        pytest.skip("filesystem is case-sensitive")

    mgr = ProjectRuntime(str(ws))

    with pytest.raises(PermissionError, match="protected internal state"):
        mgr.read_file(".ALPHANUS/secret.txt")
    with pytest.raises(PermissionError, match="protected internal state"):
        mgr.create_file(".ALPHANUS/pwned.txt", "x")
    with pytest.raises(PermissionError, match="protected internal state"):
        mgr.delete_path(".ALPHANUS/secret.txt")


def test_shell_command_runs_with_user_shell(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("python3 -c \"print('ok')\"")
    assert res["ok"] is True
    assert res["data"]["stdout"].strip() == "ok"
    assert res["data"]["argv"]
    assert "python3 -c \"print('ok')\"" in res["data"]["argv"]
    assert res["meta"]["project_changed"] is False


def test_shell_command_pwd_does_not_mark_git_repo_changed(tmp_path: Path):
    git_path = _require_git()

    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    subprocess.run([git_path, "-C", str(ws), "init"], check=True, capture_output=True, text=True)

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("pwd")

    assert res["ok"] is True
    assert res["data"]["stdout"].strip() == str(ws.resolve())
    assert res["meta"]["project_changed"] is False


def test_shell_command_nonzero_exit_is_reported_as_failure(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command('python3 -c "raise SystemExit(7)"')
    assert res["ok"] is False
    assert res["error"]["code"] == "E_SHELL"
    assert "code 7" in res["error"]["message"]
    assert res["data"]["returncode"] == 7


def test_shell_command_allows_semicolon_inside_quoted_python_arg(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("python3 -c \"import sys; print('ok')\"")
    assert res["ok"] is True
    assert res["data"]["stdout"].strip() == "ok"


def test_shell_wrapper_with_dash_c_is_allowed(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("bash -lc 'echo ok'", approved=True)
    assert res["ok"] is True
    assert res["data"]["stdout"].strip() == "ok"


def test_shell_command_allows_pipes_and_redirection(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("printf 'beta\\nalpha\\n' | sort > sorted.txt; cat sorted.txt", approved=True)

    assert res["ok"] is True
    assert res["data"]["stdout"].splitlines() == ["alpha", "beta"]
    assert (ws / "sorted.txt").read_text(encoding="utf-8") == "alpha\nbeta\n"
    assert res["meta"]["project_changed"] is True


def test_shell_command_detects_failed_chain_project_change(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("touch changed.txt; false", approved=True)

    assert res["ok"] is False
    assert (ws / "changed.txt").exists()
    assert res["meta"]["project_changed"] is True


def test_shell_command_runs_in_requested_cwd(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    subdir = ws / "nested"
    home.mkdir()
    subdir.mkdir(parents=True)

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command('python3 -c "import os; print(os.getcwd())"', cwd=str(subdir))
    assert res["ok"] is True
    assert res["data"]["stdout"].strip() == str(subdir.resolve())
    assert res["data"]["cwd"] == str(subdir.resolve())


def test_shell_command_external_cwd_requires_approval(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    outside = home / "outside"
    home.mkdir()
    ws.mkdir()
    outside.mkdir()

    mgr = ProjectRuntime(str(ws))
    denied = mgr.run_shell_command('python3 -c "import os; print(os.getcwd())"', cwd=str(outside))
    assert denied["ok"] is False
    assert denied["error"]["code"] == "E_POLICY"

    approved = mgr.run_shell_command('python3 -c "import os; print(os.getcwd())"', cwd=str(outside), approved=True)
    assert approved["ok"] is True
    assert approved["data"]["stdout"].strip() == str(outside.resolve())


def test_shell_command_known_mutator_sets_project_changed_true(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("mkdir demo")

    assert res["ok"] is True
    assert res["meta"]["project_changed"] is True
    assert (ws / "demo").is_dir()


def test_shell_command_ambiguous_command_uses_git_snapshot_when_repo_present(tmp_path: Path):
    git_path = _require_git()

    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    subprocess.run([git_path, "-C", str(ws), "init"], check=True, capture_output=True, text=True)

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("python3 -c \"from pathlib import Path; Path('note.txt').write_text('x', encoding='utf-8')\"")

    assert res["ok"] is True
    assert res["meta"]["project_changed"] is True
    assert (ws / "note.txt").read_text(encoding="utf-8") == "x"


def test_shell_command_detects_changes_to_existing_untracked_file_in_git_repo(tmp_path: Path):
    git_path = _require_git()

    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    subprocess.run([git_path, "-C", str(ws), "init"], check=True, capture_output=True, text=True)
    scratch = ws / "scratch.txt"
    scratch.write_text("before", encoding="utf-8")

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("python3 -c \"from pathlib import Path; Path('scratch.txt').write_text('after', encoding='utf-8')\"")

    assert res["ok"] is True
    assert res["meta"]["project_changed"] is True
    assert scratch.read_text(encoding="utf-8") == "after"


def test_shell_command_detects_git_branch_creation_in_git_repo(tmp_path: Path):
    git_path = _require_git()

    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    subprocess.run([git_path, "-C", str(ws), "init"], check=True, capture_output=True, text=True)
    (ws / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run([git_path, "-C", str(ws), "add", "tracked.txt"], check=True, capture_output=True, text=True)
    subprocess.run(
        [
            git_path,
            "-C",
            str(ws),
            "-c",
            "user.name=Test User",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-m",
            "init",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("git branch feature/test")

    assert res["ok"] is True
    assert res["meta"]["project_changed"] is True


def test_shell_command_detects_ignored_output_changes_in_git_repo(tmp_path: Path):
    git_path = _require_git()

    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    subprocess.run([git_path, "-C", str(ws), "init"], check=True, capture_output=True, text=True)
    (ws / ".gitignore").write_text("dist/\n", encoding="utf-8")

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command(
        "python3 -c \"from pathlib import Path; Path('dist').mkdir(exist_ok=True); Path('dist/out.txt').write_text('x', encoding='utf-8')\""
    )

    assert res["ok"] is True
    assert res["meta"]["project_changed"] is True
    assert (ws / "dist" / "out.txt").read_text(encoding="utf-8") == "x"


def test_move_path_renames_project_file(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    source = ws / "index.html"
    source.write_text("<h1>hello</h1>\n", encoding="utf-8")

    mgr = ProjectRuntime(str(ws))
    moved = mgr.move_path("index.html", "site/index.html")

    assert moved == str((ws / "site" / "index.html").resolve())
    assert not source.exists()
    assert (ws / "site" / "index.html").read_text(encoding="utf-8") == "<h1>hello</h1>\n"


def test_delete_project_root_denied_for_dot_path(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    with pytest.raises(PermissionError):
        mgr.delete_path(".", recursive=True)


def test_delete_project_root_denied_for_empty_path(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    with pytest.raises(PermissionError):
        mgr.delete_path("", recursive=True)


def test_delete_alphanus_state_denied(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    state_dir = ws / ".alphanus" / "sessions"
    home.mkdir()
    state_dir.mkdir(parents=True)

    mgr = ProjectRuntime(str(ws))
    with pytest.raises(PermissionError, match="protected internal state"):
        mgr.delete_path(".alphanus", recursive=True)


def test_shell_command_hides_protected_state_name(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("ls .alphanus")

    assert res["ok"] is False
    assert res["error"]["code"] == "E_POLICY"
    assert "protected internal state" in res["error"]["message"]
    assert ".alphanus" not in res["error"]["message"]


def test_shell_command_blocks_unknown_command_with_standalone_protected_dir_operand(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    (ws / ".alphanus").mkdir()

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("tar -C .alphanus -cf /tmp/out.tar secret.txt")

    assert res["ok"] is False
    assert res["error"]["code"] == "E_POLICY"


def test_shell_command_blocks_glob_expanded_protected_state_path(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    (ws / ".alphanus").mkdir()
    (ws / ".alphanus" / "sessions.json").write_text("secret\n", encoding="utf-8")

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("cat .alpha*/sessions.json")

    assert res["ok"] is False
    assert res["error"]["code"] == "E_POLICY"
    assert "protected internal state" in res["error"]["message"]


def test_shell_command_blocks_symlink_alias_to_protected_state(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    (ws / ".alphanus" / "sessions").mkdir(parents=True)
    (ws / "alias").symlink_to(ws / ".alphanus", target_is_directory=True)

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("ls alias")

    assert res["ok"] is False
    assert res["error"]["code"] == "E_POLICY"
    assert "protected internal state" in res["error"]["message"]


def test_shell_command_blocks_glob_expanded_symlink_to_protected_state(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    (ws / ".alphanus" / "sessions").mkdir(parents=True)
    (ws / "alias").symlink_to(ws / ".alphanus", target_is_directory=True)

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("ls al*")

    assert res["ok"] is False
    assert res["error"]["code"] == "E_POLICY"
    assert "protected internal state" in res["error"]["message"]


def test_shell_command_does_not_treat_git_subcommand_as_path_operand(tmp_path: Path):
    git_path = _require_git()

    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    (ws / ".alphanus").mkdir()
    (ws / "status").symlink_to(ws / ".alphanus", target_is_directory=True)
    subprocess.run([git_path, "-C", str(ws), "init"], check=True, capture_output=True, text=True)

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("git status")

    assert res["ok"] is True


def test_shell_command_blocks_case_insensitive_protected_state_path(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    home.mkdir()
    ws.mkdir()
    (ws / ".alphanus").mkdir()
    if not _filesystem_is_case_insensitive(ws):
        pytest.skip("filesystem is case-sensitive")

    mgr = ProjectRuntime(str(ws))
    res = mgr.run_shell_command("ls .ALPHANUS")

    assert res["ok"] is False
    assert res["error"]["code"] == "E_POLICY"
