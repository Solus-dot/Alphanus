from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.skills import ToolExecutionEnv

TOOL_SPECS = {
    "git_status": {
        "capability": "workspace_read",
        "description": "Return structured Git status for a repository inside the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    "git_log": {
        "capability": "workspace_read",
        "description": "Return recent Git commits for a repository inside the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_count": {"type": "integer"},
                "ref": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    "git_diff": {
        "capability": "workspace_read",
        "description": "Return Git diff output for a repository inside the workspace.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "staged": {"type": "boolean"},
                "ref": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
            },
            "additionalProperties": False,
        },
    },
    "git_show": {
        "capability": "workspace_read",
        "description": "Show a Git object or revision inside a workspace repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "rev": {"type": "string"},
            },
            "required": ["rev"],
            "additionalProperties": False,
        },
    },
    "git_branch_list": {
        "capability": "workspace_read",
        "description": "List local and optional remote Git branches.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "all": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    "git_branch_create": {
        "capability": "workspace_write",
        "description": "Create a Git branch in a workspace repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "name": {"type": "string"},
                "start_point": {"type": "string"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    "git_branch_switch": {
        "capability": "workspace_write",
        "description": "Switch Git branches when the working tree is clean.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "name": {"type": "string"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    "git_add": {
        "capability": "workspace_write",
        "description": "Stage explicit paths in a workspace Git repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "paths": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["paths"],
            "additionalProperties": False,
        },
    },
    "git_commit": {
        "capability": "workspace_write",
        "description": "Commit currently staged changes with a non-empty message.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["message"],
            "additionalProperties": False,
        },
    },
    "git_fetch": {
        "capability": "workspace_write",
        "description": "Fetch from a Git remote in a workspace repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "remote": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    "git_pull": {
        "capability": "workspace_write",
        "description": "Pull from a Git remote, defaulting to rebase.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "remote": {"type": "string"},
                "branch": {"type": "string"},
                "rebase": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    "git_push": {
        "capability": "workspace_write",
        "description": "Push to a Git remote after explicit confirmation. Force push modes are rejected.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "remote": {"type": "string"},
                "branch": {"type": "string"},
                "confirm_push": {"type": "boolean"},
                "force": {"type": "boolean"},
            },
            "required": ["confirm_push"],
            "additionalProperties": False,
        },
    },
    "git_init": {
        "capability": "workspace_write",
        "description": "Initialize a Git repository in an explicit descendant folder inside the workspace, never at workspace root.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "create_if_missing": {"type": "boolean"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}

MAX_OUTPUT_BYTES = 20000


def _ok(data: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "data": data, "error": None, "meta": {}}


def _err(code: str, message: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"ok": False, "data": data, "error": {"code": code, "message": message}, "meta": {}}


def _git() -> str:
    git_path = shutil.which("git")
    if not git_path:
        raise FileNotFoundError("git executable not found")
    return git_path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_inside_workspace(env: ToolExecutionEnv, path: str | object = ".") -> Path:
    root = Path(env.workspace.workspace_root).resolve()
    raw_text = str(path or ".").strip() or "."
    raw = Path(os.path.expanduser(raw_text))
    candidate = raw if raw.is_absolute() else root / raw
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        resolved = Path(os.path.abspath(str(candidate)))
    if not _is_under(resolved, root):
        raise PermissionError("Git path escapes workspace root")
    return resolved


def _clip(text: str, max_bytes: int = MAX_OUTPUT_BYTES) -> tuple[str, bool]:
    encoded = str(text or "").encode("utf-8")
    if len(encoded) <= max_bytes:
        return str(text or ""), False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


def _run_git(cwd: Path, args: list[str], *, timeout_s: int = 30) -> dict[str, Any]:
    proc = subprocess.run(
        [_git(), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    stdout, stdout_truncated = _clip(proc.stdout)
    stderr, stderr_truncated = _clip(proc.stderr)
    return {
        "argv": ["git", *args],
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "returncode": proc.returncode,
        "cwd": str(cwd),
    }


def _git_success(cwd: Path, args: list[str], *, timeout_s: int = 30) -> tuple[bool, dict[str, Any]]:
    run = _run_git(cwd, args, timeout_s=timeout_s)
    return run["returncode"] == 0, run


def _git_failure_message(run: dict[str, Any], fallback: str = "Git command failed") -> str:
    detail = str(run.get("stderr") or run.get("stdout") or "").strip()
    if detail:
        return detail
    return f"{fallback} with exit code {run.get('returncode')}"


def _repo_root_for(cwd: Path, workspace_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    ok, run = _git_success(cwd, ["rev-parse", "--show-toplevel"])
    if not ok:
        return None, run
    root_text = str(run["stdout"]).strip()
    if not root_text:
        return None, run
    repo_root = Path(root_text).resolve()
    if not _is_under(repo_root, workspace_root):
        raise PermissionError("Git repository root escapes workspace root")
    return repo_root, run


def _require_repo(env: ToolExecutionEnv, path: object = ".") -> tuple[Path, Path]:
    cwd = _resolve_inside_workspace(env, path)
    if not cwd.exists():
        raise FileNotFoundError(str(cwd))
    if cwd.is_file():
        cwd = cwd.parent
    if not cwd.is_dir():
        raise FileNotFoundError(str(cwd))
    workspace_root = Path(env.workspace.workspace_root).resolve()
    repo_root, run = _repo_root_for(cwd, workspace_root)
    if repo_root is None:
        message = _git_failure_message(run or {}, "Path is not inside a Git repository")
        raise ValueError(f"Path is not inside a Git repository: {message}")
    return cwd, repo_root


def _clean_arg(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    if text.startswith("-") or text.startswith("+") or any(ch in text for ch in ("\n", "\r", "\0")):
        raise PermissionError(f"{field_name} is not a safe Git argument")
    force_tokens = {"--force", "-f", "--force-with-lease", "--force-if-includes"}
    if text in force_tokens or text.startswith("--force"):
        raise PermissionError("Force push modes are rejected")
    return text


def _optional_clean_arg(args: dict[str, object], key: str) -> str:
    value = args.get(key)
    if value is None or str(value).strip() == "":
        return ""
    return _clean_arg(value, key)


def _pathspecs(env: ToolExecutionEnv, repo_root: Path, raw_paths: object) -> list[str]:
    if not isinstance(raw_paths, list) or not raw_paths:
        raise ValueError("paths must be a non-empty array")
    workspace_root = Path(env.workspace.workspace_root).resolve()
    out: list[str] = []
    for item in raw_paths:
        text = str(item or "").strip()
        if not text:
            raise ValueError("paths must not contain empty values")
        raw = Path(os.path.expanduser(text))
        if raw.is_absolute():
            resolved = raw.resolve(strict=False)
        else:
            repo_relative = (repo_root / raw).resolve(strict=False)
            workspace_relative = (workspace_root / raw).resolve(strict=False)
            resolved = workspace_relative if _is_under(workspace_relative, repo_root) else repo_relative
        if not _is_under(resolved, workspace_root):
            raise PermissionError("Git pathspec escapes workspace root")
        if not _is_under(resolved, repo_root):
            raise PermissionError("Git pathspec must remain inside the target repository")
        try:
            out.append(resolved.relative_to(repo_root).as_posix())
        except ValueError as exc:
            raise PermissionError("Git pathspec must remain inside the target repository") from exc
    return out


def _parse_status(output: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in output.splitlines():
        if not line or line.startswith("##"):
            continue
        if len(line) < 4:
            continue
        entries.append({"index": line[0], "worktree": line[1], "path": line[3:]})
    return entries


def _branch_current(repo_root: Path) -> str:
    ok, run = _git_success(repo_root, ["branch", "--show-current"])
    return str(run.get("stdout", "")).strip() if ok else ""


def _working_tree_clean(repo_root: Path) -> bool:
    ok, run = _git_success(repo_root, ["status", "--porcelain=v1"])
    return ok and str(run.get("stdout", "")).strip() == ""


def _has_staged_changes(repo_root: Path) -> bool:
    run = _run_git(repo_root, ["diff", "--cached", "--quiet"])
    if run["returncode"] == 0:
        return False
    if run["returncode"] == 1:
        return True
    raise RuntimeError(_git_failure_message(run, "Unable to inspect staged changes"))


def _normalize_run(repo_root: Path, run: dict[str, Any]) -> dict[str, Any]:
    out = dict(run)
    out["repo_root"] = str(repo_root)
    out["current_branch"] = _branch_current(repo_root)
    return out


def _status(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    run = _run_git(repo_root, ["status", "--short", "--branch"])
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), _normalize_run(repo_root, run))
    return _ok({**_normalize_run(repo_root, run), "entries": _parse_status(str(run["stdout"]))})


def _log(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    max_count = max(1, min(100, int(args.get("max_count") or 20)))
    argv = ["log", f"--max-count={max_count}", "--date=iso-strict", "--pretty=format:%H%x00%h%x00%an%x00%ae%x00%ad%x00%s"]
    ref = _optional_clean_arg(args, "ref")
    if ref:
        argv.append(ref)
    run = _run_git(repo_root, argv)
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), _normalize_run(repo_root, run))
    commits = []
    for line in str(run["stdout"]).splitlines():
        parts = line.split("\x00")
        if len(parts) == 6:
            commits.append(
                {
                    "hash": parts[0],
                    "short_hash": parts[1],
                    "author_name": parts[2],
                    "author_email": parts[3],
                    "date": parts[4],
                    "subject": parts[5],
                }
            )
    return _ok({**_normalize_run(repo_root, run), "commits": commits, "count": len(commits)})


def _diff(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    argv = ["diff"]
    if bool(args.get("staged", False)):
        argv.append("--cached")
    ref = _optional_clean_arg(args, "ref")
    if ref:
        argv.append(ref)
    raw_paths = args.get("paths")
    if raw_paths is not None:
        argv.append("--")
        argv.extend(_pathspecs(env, repo_root, raw_paths))
    run = _run_git(repo_root, argv)
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), _normalize_run(repo_root, run))
    return _ok({**_normalize_run(repo_root, run), "diff": run["stdout"]})


def _show(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    rev = _clean_arg(args.get("rev"), "rev")
    run = _run_git(repo_root, ["show", "--stat", "--patch", rev])
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), _normalize_run(repo_root, run))
    return _ok({**_normalize_run(repo_root, run), "content": run["stdout"]})


def _branch_list(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    argv = ["branch", "--format=%(refname:short)%00%(HEAD)%00%(upstream:short)"]
    if bool(args.get("all", False)):
        argv.insert(1, "--all")
    run = _run_git(repo_root, argv)
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), _normalize_run(repo_root, run))
    branches = []
    for line in str(run["stdout"]).splitlines():
        parts = line.split("\x00")
        if len(parts) == 3:
            branches.append({"name": parts[0], "current": parts[1] == "*", "upstream": parts[2]})
    return _ok({**_normalize_run(repo_root, run), "branches": branches})


def _branch_create(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    name = _clean_arg(args.get("name"), "name")
    argv = ["branch", name]
    start_point = _optional_clean_arg(args, "start_point")
    if start_point:
        argv.append(start_point)
    run = _run_git(repo_root, argv)
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), _normalize_run(repo_root, run))
    return _ok({**_normalize_run(repo_root, run), "created_branch": name})


def _branch_switch(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    if not _working_tree_clean(repo_root):
        return _err("E_DIRTY_WORKTREE", "Dirty working tree blocks branch switching", {"repo_root": str(repo_root)})
    name = _clean_arg(args.get("name"), "name")
    run = _run_git(repo_root, ["switch", name])
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), _normalize_run(repo_root, run))
    return _ok({**_normalize_run(repo_root, run), "switched_to": name})


def _add(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    paths = _pathspecs(env, repo_root, args.get("paths"))
    run = _run_git(repo_root, ["add", "--", *paths])
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), _normalize_run(repo_root, run))
    return _ok({**_normalize_run(repo_root, run), "staged_paths": paths})


def _commit(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    message = str(args.get("message") or "").strip()
    if not message:
        raise ValueError("message must be non-empty")
    if not _has_staged_changes(repo_root):
        return _err("E_NOOP", "No staged changes to commit", {"repo_root": str(repo_root)})
    run = _run_git(repo_root, ["commit", "-m", message])
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), _normalize_run(repo_root, run))
    return _ok(_normalize_run(repo_root, run))


def _fetch(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    argv = ["fetch"]
    remote = _optional_clean_arg(args, "remote")
    if remote:
        argv.append(remote)
    run = _run_git(repo_root, argv, timeout_s=120)
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), _normalize_run(repo_root, run))
    return _ok(_normalize_run(repo_root, run))


def _pull(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    argv = ["pull", "--rebase" if bool(args.get("rebase", True)) else "--no-rebase"]
    remote = _optional_clean_arg(args, "remote")
    branch = _optional_clean_arg(args, "branch")
    if remote:
        argv.append(remote)
    if branch:
        argv.append(branch)
    run = _run_git(repo_root, argv, timeout_s=120)
    if run["returncode"] != 0:
        message = _git_failure_message(run)
        code = "E_CONFLICT" if any(token in message.lower() for token in ("conflict", "could not apply", "merge failed")) else "E_GIT"
        return _err(code, message, _normalize_run(repo_root, run))
    return _ok(_normalize_run(repo_root, run))


def _push(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    _, repo_root = _require_repo(env, args.get("path", "."))
    if bool(args.get("force", False)):
        return _err("E_POLICY", "Force push modes are rejected", {"repo_root": str(repo_root)})
    if not bool(args.get("confirm_push", False)):
        return _err("E_CONFIRMATION_REQUIRED", "git_push requires confirm_push=true", {"repo_root": str(repo_root)})
    argv = ["push"]
    remote = _optional_clean_arg(args, "remote")
    branch = _optional_clean_arg(args, "branch")
    if remote:
        argv.append(remote)
    if branch:
        argv.append(branch)
    if any(str(part).startswith(("-", "+")) or str(part).startswith("--force") for part in argv[1:]):
        return _err("E_POLICY", "Force push modes are rejected", {"repo_root": str(repo_root)})
    run = _run_git(repo_root, argv, timeout_s=120)
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), _normalize_run(repo_root, run))
    return _ok(_normalize_run(repo_root, run))


def _nearest_existing_parent(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _init(args: dict[str, object], env: ToolExecutionEnv) -> dict[str, Any]:
    if "path" not in args or str(args.get("path") or "").strip() == "":
        raise ValueError("path is required")
    workspace_root = Path(env.workspace.workspace_root).resolve()
    target = _resolve_inside_workspace(env, args["path"])
    created = False

    if target == workspace_root:
        return _err(
            "E_POLICY",
            "git_init is not allowed at the workspace root; choose an explicit nested folder inside the workspace",
            {"path": str(target), "workspace_root": str(workspace_root), "created": False, "block_reason": "workspace_root"},
        )
    if not _is_under(target, workspace_root):
        return _err(
            "E_POLICY",
            "git_init path escapes workspace root",
            {"path": str(target), "workspace_root": str(workspace_root), "created": False, "block_reason": "outside_workspace"},
        )
    if target.exists() and not target.is_dir():
        return _err(
            "E_NOT_DIRECTORY",
            "git_init target must be a directory",
            {"path": str(target), "workspace_root": str(workspace_root), "created": False, "block_reason": "not_directory"},
        )

    existing_parent = _nearest_existing_parent(target)
    if existing_parent.exists():
        repo_root, _ = _repo_root_for(existing_parent if existing_parent.is_dir() else existing_parent.parent, workspace_root)
        if repo_root is not None:
            return _err(
                "E_POLICY",
                "git_init target is already inside an existing Git repository; nested repository initialization is blocked",
                {
                    "path": str(target),
                    "workspace_root": str(workspace_root),
                    "existing_repo_root": str(repo_root),
                    "created": False,
                    "block_reason": "nested_repo",
                },
            )

    if not target.exists():
        if not bool(args.get("create_if_missing", False)):
            return _err(
                "E_NOT_FOUND",
                "git_init target directory does not exist; set create_if_missing=true to create it",
                {"path": str(target), "workspace_root": str(workspace_root), "created": False, "block_reason": "missing"},
            )
        target.mkdir(parents=True, exist_ok=True)
        created = True

    run = _run_git(target, ["init"])
    if run["returncode"] != 0:
        return _err("E_GIT", _git_failure_message(run), {**run, "path": str(target), "created": created})
    return _ok({**run, "path": str(target), "workspace_root": str(workspace_root), "created": created, "initialized": True})


def execute(tool_name: str, args: dict[str, object], env: ToolExecutionEnv):
    handlers = {
        "git_status": _status,
        "git_log": _log,
        "git_diff": _diff,
        "git_show": _show,
        "git_branch_list": _branch_list,
        "git_branch_create": _branch_create,
        "git_branch_switch": _branch_switch,
        "git_add": _add,
        "git_commit": _commit,
        "git_fetch": _fetch,
        "git_pull": _pull,
        "git_push": _push,
        "git_init": _init,
    }
    handler = handlers.get(tool_name)
    if handler is None:
        raise ValueError(f"Unsupported tool: {tool_name}")
    return handler(args, env)
