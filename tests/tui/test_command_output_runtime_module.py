from __future__ import annotations

from types import SimpleNamespace

from tui.command_output_runtime import cmd_doctor, cmd_skills


class _Skill:
    def __init__(self) -> None:
        self.id = "skill-1"
        self.version = "1.0.0"
        self.description = "Example skill"
        self.user_invocable = True
        self.disable_model_invocation = False
        self.execution_allowed = True
        self.adapter = "agentskills"
        self.available = False
        self.availability_reason = "blocked by policy"
        self.availability_code = "blocked"
        self.validation_errors = ["missing field"]


class _SkillRuntime:
    def __init__(self) -> None:
        self._skill = _Skill()

    def list_skills(self):
        return [self._skill]

    def skill_status_label(self, _skill):
        return "enabled", "#10b981"

    def skill_source_label(self, _skill):
        return "local"

    def skill_provenance_label(self, _skill):
        return "filesystem"

    def _reported_skill_tools(self, _skill):
        return ["tool-a"]

    def _reported_skill_scripts(self, _skill):
        return ["script-a"]

    def _reported_skill_entrypoints(self, _skill):
        return [SimpleNamespace(name="main")]


class _SkillsApp:
    def __init__(self) -> None:
        self.agent = SimpleNamespace(skill_runtime=_SkillRuntime())
        self._loaded_skill_ids = ["skill-1"]
        self.lines: list[str] = []

    def _write_section_heading(self, text: str) -> None:
        self.lines.append(f"heading:{text}")

    def _write(self, text: str) -> None:
        self.lines.append(text)


class _DoctorApp:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.agent = SimpleNamespace(
            doctor_report=lambda: {
                "agent": {"ready": True},
                "workspace": {"path": "/workspace", "writable": True},
                "memory": {},
                "search": {},
                "skills": [
                    {
                        "id": "skill-1",
                        "provenance": "filesystem",
                        "availability_code": "blocked",
                        "status": "enabled",
                        "availability_reason": "blocked by policy",
                        "execution_allowed": True,
                        "adapter": "agentskills",
                        "tools": ["tool-a"],
                        "scripts": ["script-a"],
                        "entrypoints": [{"name": "main"}],
                        "user_invocable": True,
                        "model_invocable": True,
                        "validation_errors": ["missing field"],
                    }
                ],
            }
        )

    def _write_section_heading(self, text: str) -> None:
        self.lines.append(f"heading:{text}")

    def _write_detail_line(self, key: str, value: str, *, value_markup: bool = False) -> None:
        suffix = ":markup" if value_markup else ""
        self.lines.append(f"detail:{key}{suffix}={value}")

    def _write(self, text: str) -> None:
        self.lines.append(text)


def test_cmd_skills_renders_shared_skill_capability_summary() -> None:
    app = _SkillsApp()

    cmd_skills(app, accent_color="#6366f1")

    assert any("heading:Skills" == line for line in app.lines)
    assert any("execution=yes · adapter=agentskills · user=yes · model=yes · tools=1 · scripts=1 · entrypoints=1" in line for line in app.lines)
    assert any("validation:" in line and "missing field" in line for line in app.lines)


def test_cmd_doctor_renders_shared_skill_capability_summary() -> None:
    app = _DoctorApp()

    cmd_doctor(app, accent_color="#6366f1")

    assert any("heading:Skills" == line for line in app.lines)
    assert any("execution=yes · adapter=agentskills · user=yes · model=yes · tools=1 · scripts=1 · entrypoints=1" in line for line in app.lines)
    assert any("validation:" in line and "missing field" in line for line in app.lines)
