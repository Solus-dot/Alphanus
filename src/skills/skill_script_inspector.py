from __future__ import annotations

import ast
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

from skills.skill_parser import SkillManifest

SCRIPT_INTERPRETER_BY_EXT = {
    ".py": [sys.executable],
    ".sh": ["bash"],
    ".js": ["node"],
    ".mjs": ["node"],
}


class SkillScriptInspector:
    def __init__(self, runtime) -> None:
        self.runtime = runtime

    @staticmethod
    def is_skill_script_candidate(relpath: str) -> bool:
        normalized = str(relpath or "").strip()
        if not normalized:
            return False
        name = Path(normalized).name
        if name in {"tools.py", "hooks.py", "__init__.py"}:
            return False
        if normalized.startswith("scripts/"):
            return True
        return "/" not in normalized and Path(normalized).suffix.lower() in SCRIPT_INTERPRETER_BY_EXT

    def skill_runnable_scripts(self, skill: SkillManifest) -> list[str]:
        runtime = self.runtime
        cache_key = (skill.id, str(skill.path or ""))
        cached = runtime._runnable_scripts_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        runnable: list[str] = []
        for rel in skill.bundled_files:
            if not self.is_skill_script_candidate(rel):
                continue
            if not self.script_is_cli_entry(skill, rel):
                continue
            ext = Path(rel).suffix.lower()
            interpreter = self.script_interpreter(ext)
            if not interpreter:
                continue
            if (
                not Path(interpreter[0]).exists()
                and interpreter[0] != Path(sys.executable).resolve().as_posix()
                and shutil.which(interpreter[0]) is None
            ):
                continue
            if self.python_script_missing_modules(skill, rel):
                continue
            runnable.append(rel)
        deduped = tuple(sorted(dict.fromkeys(runnable)))
        runtime._runnable_scripts_cache[cache_key] = deduped
        return list(deduped)

    def script_block_reason(self, skill: SkillManifest, rel_script: str) -> str:
        runtime = self.runtime
        if not skill.path:
            return "skill root is unavailable"
        script_path = (skill.path / rel_script).resolve()
        if not runtime._is_relative_to(script_path, skill.path.resolve()):
            return "script path escapes skill root"
        if not script_path.exists():
            return "script file is missing"
        ext = script_path.suffix.lower()
        interpreter = self.script_interpreter(ext)
        if not interpreter:
            return f"unsupported script type: {script_path.suffix}"
        if (
            not Path(interpreter[0]).exists()
            and interpreter[0] != Path(sys.executable).resolve().as_posix()
            and shutil.which(interpreter[0]) is None
        ):
            return f"missing interpreter: {interpreter[0]}"
        missing_modules = self.python_script_missing_modules(skill, rel_script)
        if missing_modules:
            return f"missing python modules: {', '.join(missing_modules)}"
        return ""

    def blocked_skill_scripts(self, skill: SkillManifest) -> list[dict[str, str]]:
        blocked: list[dict[str, str]] = []
        for rel in sorted(rel for rel in skill.bundled_files if self.is_skill_script_candidate(rel)):
            if not self.script_is_cli_entry(skill, rel):
                continue
            if Path(rel).suffix.lower() not in SCRIPT_INTERPRETER_BY_EXT:
                continue
            reason = self.script_block_reason(skill, rel)
            if reason:
                blocked.append({"script": rel, "reason": reason})
        return blocked

    @staticmethod
    def script_is_cli_entry(skill: SkillManifest, rel_script: str) -> bool:
        if not skill.path:
            return False
        script_path = (skill.path / rel_script).resolve()
        ext = script_path.suffix.lower()
        if ext in {".sh", ".js", ".mjs"}:
            return True
        if ext != ".py" or not script_path.exists():
            return False
        return script_path.name != "__init__.py"

    @staticmethod
    def script_has_local_module(skill: SkillManifest, script_path: Path, module_name: str) -> bool:
        if not skill.path:
            return False
        bases = [script_path.parent, skill.path / "scripts", skill.path]
        for base in bases:
            if (base / f"{module_name}.py").exists():
                return True
            module_dir = base / module_name
            if module_dir.exists() and module_dir.is_dir():
                return True
        return False

    def python_script_missing_modules(self, skill: SkillManifest, rel_script: str) -> list[str]:
        runtime = self.runtime
        if not skill.path:
            return []
        script_path = (skill.path / rel_script).resolve()
        if script_path.suffix.lower() != ".py" or not script_path.exists():
            return []
        try:
            tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            runtime._append_unique(
                skill.validation_warnings,
                f"script '{rel_script}' import inspection failed: {exc.__class__.__name__}: {exc}",
            )
            return []

        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = str(alias.name or "").split(".", 1)[0].strip()
                    if name:
                        imported.add(name)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                name = str(node.module).split(".", 1)[0].strip()
                if name:
                    imported.add(name)

        missing: list[str] = []
        stdlib = getattr(sys, "stdlib_module_names", set())
        for name in sorted(imported):
            if name in stdlib:
                continue
            if self.script_has_local_module(skill, script_path, name):
                continue
            if not self.python_module_available(name):
                missing.append(name)
        return missing

    def script_interpreter(self, ext: str) -> list[str] | None:
        normalized = str(ext or "").lower()
        if normalized == ".py":
            return [self.runtime.python_executable]
        base = SCRIPT_INTERPRETER_BY_EXT.get(normalized)
        return list(base) if base else None

    def python_module_available(self, module_name: str) -> bool:
        runtime = self.runtime
        cached = runtime._python_module_probe_cache.get(module_name)
        if cached is not None:
            return cached

        python_exec = str(runtime.python_executable or sys.executable)
        if python_exec == sys.executable:
            try:
                available = importlib.util.find_spec(module_name) is not None
            except (ImportError, AttributeError, ValueError):
                available = False
            runtime._python_module_probe_cache[module_name] = available
            return available

        try:
            proc = subprocess.run(
                [
                    python_exec,
                    "-c",
                    "import importlib.util,sys; raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
                    module_name,
                ],
                capture_output=True,
                text=True,
                timeout=5,
                env=runtime._proc_env_base,
            )
            available = proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            available = False
        runtime._python_module_probe_cache[module_name] = available
        return available
