from __future__ import annotations

from core.skill_parser import SkillManifest


class SkillInventoryLoader:
    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def load_skills(self) -> None:
        runtime = self.runtime
        previous_enabled = {skill_id: skill.enabled for skill_id, skill in runtime.skills.items()}
        runtime.generation += 1
        runtime.skill_roots = runtime._discover_skill_roots()
        runtime.skills = {}
        runtime._all_skills = []
        runtime._skill_index = {}
        runtime._tool_registry = {}
        runtime._invalidate_skill_caches()
        runtime._register_runtime_tools()
        if not any(root.exists() for root in runtime.skill_roots):
            return

        for root in runtime.skill_roots:
            if not root.exists():
                continue
            for child in runtime._discover_skill_dirs(root):
                manifest: SkillManifest | None = None
                try:
                    manifest = runtime._load_manifest(child)
                    if manifest is None:
                        continue

                    if manifest.id in previous_enabled:
                        manifest.enabled = previous_enabled[manifest.id]

                    (
                        manifest.available,
                        manifest.availability_code,
                        manifest.availability_reason,
                    ) = runtime._check_skill_availability(manifest)

                    existing = runtime.skills.get(manifest.id)
                    if existing is not None:
                        source = runtime.skill_source_label(existing) or existing.id
                        incoming = runtime.skill_source_label(manifest) or manifest.id
                        runtime._append_unique(
                            existing.validation_errors,
                            f"duplicate skill id '{manifest.id}' ignored from {incoming}; using {source}",
                        )
                        continue

                    if manifest.available and manifest.execution_allowed and not runtime._load_skill_tools(manifest):
                        manifest.available = False
                        manifest.execution_allowed = False
                        if not manifest.availability_code or manifest.availability_code == "ready":
                            manifest.availability_code = "invalid"
                        if not manifest.availability_reason:
                            manifest.availability_reason = manifest.validation_errors[0] if manifest.validation_errors else "skill load failed"

                    runtime.skills[manifest.id] = manifest
                    runtime._all_skills.append(manifest)
                except Exception as exc:
                    runtime._remove_skill_tools(manifest.id if manifest else child.name)
                    if manifest is not None:
                        runtime._append_unique(manifest.validation_errors, str(exc))
                        manifest.available = False
                        manifest.execution_allowed = False
                        manifest.availability_code = "invalid"
                        manifest.availability_reason = str(exc)
                        runtime.skills[manifest.id] = manifest
                        runtime._all_skills.append(manifest)
                    elif runtime.debug:
                        print(f"[skill] failed to load {child.name}: {exc}")
        runtime._rebuild_skill_index()

