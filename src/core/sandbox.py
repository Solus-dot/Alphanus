from __future__ import annotations

import os
import platform
import shlex
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.project_command_policy import shell_has_boundary, unwrap_shell_command

MAX_SANDBOX_OUTPUT_BYTES = 20000
MACOS_READ_ROOTS = (
    "/bin",
    "/sbin",
    "/usr",
    "/System",
    "/Library",
    "/opt",
    "/opt/homebrew",
    "/opt/local",
)
SHELL_WRAPPER_EXECUTABLES = {"env", "command", "builtin", "exec", "time", "nice", "nohup", "sudo"}


@dataclass(frozen=True, slots=True)
class SandboxConfig:
    mode: str = "project-write"
    network: bool = False
    backend: str = "auto"
    fail_closed: bool = True


@dataclass(frozen=True, slots=True)
class SandboxCommand:
    command: str
    cwd: Path
    project_root: Path
    timeout_s: int
    config: SandboxConfig
    extra_roots: tuple[Path, ...] = ()


def _clip_text(text: str, max_bytes: int = MAX_SANDBOX_OUTPUT_BYTES) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False
    clipped = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return clipped, True


def _shell() -> str:
    configured_shell = os.environ.get("SHELL") or ""
    return configured_shell if configured_shell and (Path(configured_shell).exists() or shutil.which(configured_shell)) else "/bin/sh"


def _seatbelt_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _seatbelt_read_ancestor_rules(path: Path) -> str:
    resolved = path.resolve()
    ancestors = [resolved.anchor.rstrip("/") or "/"]
    ancestors.extend(str(parent) for parent in reversed(resolved.parents) if str(parent) != "/")
    return "\n".join(f'(allow file-read* (literal "{_seatbelt_escape(ancestor)}"))' for ancestor in dict.fromkeys(ancestors))


def _unique_resolved_paths(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = Path(os.path.abspath(str(path)))
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return tuple(unique)


def _is_single_git_command(command: str) -> bool:
    if shell_has_boundary(command):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    executable, _ = unwrap_shell_command(argv, SHELL_WRAPPER_EXECUTABLES)
    return executable == "git"


def run_bounded_process(
    argv: list[str],
    *,
    cwd: Path,
    timeout_s: int,
    env: dict[str, str] | None = None,
    stdin: str = "",
) -> dict[str, Any]:
    start = time.perf_counter()
    safe_env = {
        key: value for key, value in os.environ.items() if key in {"HOME", "LANG", "LC_ALL", "PATH", "SHELL", "TERM", "TMPDIR", "TZ"}
    }
    proc = subprocess.Popen(
        argv,
        shell=False,
        cwd=str(cwd),
        stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env or safe_env,
        start_new_session=True,
    )
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}

    def drain(name: str, stream) -> None:
        try:
            while chunk := stream.read(65536):
                remaining = MAX_SANDBOX_OUTPUT_BYTES - len(captured[name])
                if remaining > 0:
                    captured[name].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated[name] = True
        finally:
            stream.close()

    readers = [
        threading.Thread(target=drain, args=("stdout", proc.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", proc.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()
    if stdin and proc.stdin is not None:
        proc.stdin.write(stdin.encode("utf-8"))
        proc.stdin.close()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
            proc.wait()
        raise
    finally:
        for reader in readers:
            reader.join(timeout=2)
    stdout = bytes(captured["stdout"]).decode("utf-8", errors="replace")
    stderr = bytes(captured["stderr"]).decode("utf-8", errors="replace")
    stdout_truncated = truncated["stdout"]
    stderr_truncated = truncated["stderr"]
    return {
        "argv": argv,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "returncode": proc.returncode,
        "cwd": str(cwd),
        "duration_ms": int((time.perf_counter() - start) * 1000),
    }


class SandboxRunner:
    def run(self, spec: SandboxCommand) -> dict[str, Any]:
        cfg = spec.config
        if cfg.mode == "danger-full-access":
            return self._run_unsandboxed(spec)
        backend = self._resolve_backend(cfg.backend)
        if backend == "macos-seatbelt":
            return self._run_macos(spec)
        if backend == "linux-bubblewrap":
            return self._run_linux(spec)
        if backend == "windows-native":
            return self._setup_error(
                "Native Windows sandbox backend is not installed. Run Alphanus from WSL or install the Windows sandbox backend package."
            )
        return self._setup_error(f"Unsupported sandbox backend: {backend}")

    def preflight(self, config: SandboxConfig) -> dict[str, Any]:
        if config.mode == "danger-full-access":
            return {"ok": True, "backend": "none", "message": "danger-full-access bypasses sandboxing"}
        backend = self._resolve_backend(config.backend)
        if backend == "macos-seatbelt":
            ok = shutil.which("sandbox-exec") is not None
            message = "sandbox-exec is available" if ok else "sandbox-exec is required on macOS"
            return {"ok": ok, "backend": backend, "message": message}
        if backend == "linux-bubblewrap":
            ok = shutil.which("bwrap") is not None or shutil.which("bubblewrap") is not None
            message = "bubblewrap is available" if ok else "Install bubblewrap: apt install bubblewrap / dnf install bubblewrap"
            return {"ok": ok, "backend": backend, "message": message}
        if backend == "windows-native":
            return {
                "ok": False,
                "backend": backend,
                "message": "Native Windows sandbox backend is required and not bundled in this build.",
            }
        return {"ok": False, "backend": backend, "message": f"Unsupported sandbox backend: {backend}"}

    def _resolve_backend(self, requested: str) -> str:
        requested = (requested or "auto").strip().lower()
        if requested != "auto":
            return requested
        system = platform.system().lower()
        if system == "darwin":
            return "macos-seatbelt"
        if system == "linux":
            return "linux-bubblewrap"
        if system == "windows":
            return "windows-native"
        return "unsupported"

    def _run_unsandboxed(self, spec: SandboxCommand) -> dict[str, Any]:
        return run_bounded_process([_shell(), "-c", spec.command], cwd=spec.cwd, timeout_s=spec.timeout_s)

    def _run_macos(self, spec: SandboxCommand) -> dict[str, Any]:
        sandbox_exec = shutil.which("sandbox-exec")
        if not sandbox_exec:
            return self._setup_error("sandbox-exec is required for macOS project-write mode")
        profile_path = self._macos_profile(spec)
        try:
            with tempfile.TemporaryDirectory(prefix="alphanus-sandbox-home-") as sandbox_home:
                sandbox_home_path = str(Path(sandbox_home).resolve())
                argv = [
                    sandbox_exec,
                    "-f",
                    str(profile_path),
                    "/usr/bin/env",
                    f"HOME={sandbox_home_path}",
                    f"TMPDIR={Path(tempfile.gettempdir()).resolve()}",
                    _shell(),
                    "-c",
                    spec.command,
                ]
                return run_bounded_process(argv, cwd=spec.cwd, timeout_s=spec.timeout_s)
        finally:
            try:
                profile_path.unlink()
            except OSError:
                pass

    def _macos_profile(self, spec: SandboxCommand) -> Path:
        project = _seatbelt_escape(str(spec.project_root))
        temp_root_path = Path(tempfile.gettempdir()).resolve()
        temp_root = _seatbelt_escape(str(temp_root_path))
        extra_roots = _unique_resolved_paths(spec.extra_roots)
        ancestor_rules = "\n".join(
            dict.fromkeys(
                [
                    _seatbelt_read_ancestor_rules(spec.project_root),
                    _seatbelt_read_ancestor_rules(temp_root_path),
                    *(_seatbelt_read_ancestor_rules(root) for root in extra_roots),
                ]
            )
        )
        extra_read_rules = "\n".join(
            f'(allow file-read* (literal "{_seatbelt_escape(str(root))}"))\n(allow file-read* (subpath "{_seatbelt_escape(str(root))}"))'
            for root in extra_roots
        )
        read_roots = "\n".join(
            f'(allow file-read* (literal "{_seatbelt_escape(root)}"))\n(allow file-read* (subpath "{_seatbelt_escape(root)}"))'
            for root in MACOS_READ_ROOTS
        )
        protected_rules = f'(deny file-read* file-write* (subpath "{project}/.alphanus"))'
        if not _is_single_git_command(spec.command):
            protected_rules = "\n".join([f'(deny file-write* (subpath "{project}/.git"))', protected_rules])
        write_rules = ""
        if spec.config.mode == "project-write":
            roots = [f'(allow file-write* (subpath "{project}"))']
            roots.extend(f'(allow file-write* (subpath "{_seatbelt_escape(str(root))}"))' for root in extra_roots)
            write_rules = "\n".join(roots)
        network_rule = "(allow network*)" if spec.config.network else "(deny network*)"
        profile_text = f"""
(version 1)
(deny default)
(allow process*)
(allow signal (target same-sandbox))
(allow sysctl-read)
{ancestor_rules}
(allow file-read* (subpath "{project}"))
(allow file-read* (subpath "{temp_root}"))
(allow file-read* (literal "/dev/null"))
{extra_read_rules}
{read_roots}
(allow file-write* (literal "/dev/null"))
(allow file-write* (subpath "{temp_root}"))
{write_rules}
{protected_rules}
{network_rule}
""".strip()
        fd, raw_path = tempfile.mkstemp(prefix="alphanus-sandbox-", suffix=".sb")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(profile_text)
        return Path(raw_path)

    def _run_linux(self, spec: SandboxCommand) -> dict[str, Any]:
        bwrap = shutil.which("bwrap") or shutil.which("bubblewrap")
        if not bwrap:
            return self._setup_error("bubblewrap is required for Linux project-write mode")
        project = str(spec.project_root)
        cwd = str(spec.cwd)
        extra_roots = _unique_resolved_paths(spec.extra_roots)
        argv = [
            bwrap,
            "--die-with-parent",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind-try",
            "/lib64",
            "/lib64",
            "--ro-bind-try",
            "/etc",
            "/etc",
        ]
        if not spec.config.network:
            argv.extend(["--unshare-net"])
        if spec.config.mode == "project-write":
            argv.extend(["--bind", project, project])
            for root in extra_roots:
                root_text = str(root)
                argv.extend(["--bind", root_text, root_text])
        else:
            argv.extend(["--ro-bind", project, project])
            for root in extra_roots:
                root_text = str(root)
                argv.extend(["--ro-bind", root_text, root_text])
        argv.extend(["--chdir", cwd, _shell(), "-c", spec.command])
        return run_bounded_process(argv, cwd=spec.project_root, timeout_s=spec.timeout_s)

    @staticmethod
    def _setup_error(message: str) -> dict[str, Any]:
        return {
            "argv": [],
            "stdout": "",
            "stderr": "",
            "stdout_truncated": False,
            "stderr_truncated": False,
            "returncode": 126,
            "cwd": "",
            "duration_ms": 0,
            "sandbox_error": {"code": "E_SANDBOX_SETUP", "message": message},
        }


def shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return []
