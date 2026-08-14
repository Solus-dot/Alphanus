from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from agent.provider import LLMClient
from agent.telemetry import TelemetryEmitter
from core.message_types import ChatMessage, JSONValue, MessageContentPart
from core.types import TurnClassification
from skills.runtime import SkillContext, SkillRuntime

_EXPLICIT_PATH_PATTERN = re.compile(
    r'(?P<quoted>(?P<quote>["\'`])(?P<quoted_path>(?:~/|/)[^"\'`]+?)(?P=quote))'
    r"|(?P<plain>(?<![:/\w])(?P<plain_path>(?:~/|/)[^\s\"'`]+))"
)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_NON_MUTATING_DONE_WORDS = frozenset(
    "opened ran running executed launched read listed shown showed displayed inspected checked verified".split()
)
_MUTATING_WORDS = frozenset("create make write save edit update modify delete remove rename move copy scaffold generate".split())
_NON_MUTATING_ACTION_PATTERNS = {
    "open": re.compile(r"\b(?:open|opened|launch|launched)\b"),
    "run": re.compile(r"\b(?:run|ran|running|execute|executed)\b"),
    "read": re.compile(r"\b(?:read|show|showed|display|displayed)\b"),
    "list": re.compile(r"\b(?:list|listed)\b"),
    "check": re.compile(r"\b(?:inspect|inspected|check|checked|verify|verified)\b"),
}
_PROJECT_FILE_TOKEN_RE = re.compile(r"(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\.[a-z0-9]{1,16}\b", re.IGNORECASE)
_PROJECT_ABS_PATH_RE = re.compile(r"(?<![:/\w])(?:~/|/)[^\s\"'`]+")
_WELL_KNOWN_DIRECTORY_RE = re.compile(
    r"\b(?:in|into|to|on|under)\s+(?:my\s+|the\s+)?(?P<directory>desktop|downloads|documents)\b",
    re.IGNORECASE,
)
_FILESYSTEM_DIRECTORY_CONTEXT_RE = re.compile(
    r"\b(?:file|folder|directory|script|code|document|archive|project|repo(?:sitory)?|shortcut)s?\b|\b(?:save|write)\b",
    re.IGNORECASE,
)


class TurnClassifier:
    def __init__(
        self,
        _config: object,
        skill_runtime: SkillRuntime,
        llm_client: LLMClient,
        telemetry: TelemetryEmitter | None = None,
    ) -> None:
        self.skill_runtime = skill_runtime
        self.llm_client = llm_client
        self.telemetry = telemetry or TelemetryEmitter()

    @staticmethod
    def message_text(value: JSONValue | list[MessageContentPart]) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts = [str(item.get("text", "")).strip() for item in value if isinstance(item, dict) and item.get("type") == "text"]
            return "\n".join(part for part in parts if part).strip()
        return str(value or "").strip()

    def recent_routing_context(self, history_messages: list[ChatMessage]) -> tuple[str, list[str]]:
        last_user = next(
            (index for index in range(len(history_messages) - 1, -1, -1) if history_messages[index].get("role") == "user"),
            -1,
        )
        if last_user < 0:
            return "", []
        recent = history_messages[last_user:]
        tool_names: list[str] = []
        skill_ids: list[str] = []
        assistant_text = ""
        registry = getattr(self.skill_runtime, "_tool_registry", {})

        def remember_tool(name: str) -> None:
            if not name or name in tool_names:
                return
            tool_names.append(name)
            registration = registry.get(name)
            if registration and registration.skill_id not in skill_ids:
                skill_ids.append(registration.skill_id)

        for msg in recent[1:]:
            role = str(msg.get("role", "")).lower()
            if role == "assistant":
                assistant_text = self.message_text(msg.get("content", "")) or assistant_text
                for call in msg.get("tool_calls", []) or []:
                    remember_tool(str(((call or {}).get("function") or {}).get("name", "")).strip())
            elif role == "tool":
                remember_tool(str(msg.get("name", "")).strip())
                try:
                    payload = json.loads(self.message_text(msg.get("content", "")) or "{}")
                except json.JSONDecodeError:
                    payload = {}
                data = payload.get("data") if isinstance(payload, dict) else {}
                loaded_skill_id = str(data.get("skill_id", "")).strip() if isinstance(data, dict) else ""
                if loaded_skill_id and loaded_skill_id not in skill_ids:
                    skill_ids.append(loaded_skill_id)
        parts = [f"previous user request: {self.message_text(recent[0].get('content', ''))}"]
        if assistant_text:
            compact_assistant = " ".join(assistant_text.split())
            compact_assistant = compact_assistant if len(compact_assistant) <= 240 else compact_assistant[:237].rstrip() + "..."
            parts.append(f"assistant just said: {compact_assistant}")
        if tool_names:
            parts.append(f"tools just used: {', '.join(tool_names[:4])}")
        return "\n".join(parts), skill_ids[:3]

    def build_skill_context(
        self,
        user_input: str,
        branch_labels: list[str],
        attachments: list[str],
        history_messages: list[ChatMessage] | None = None,
        loaded_skill_ids: list[str] | None = None,
    ) -> SkillContext:
        recent_hint, sticky_skill_ids = self.recent_routing_context(history_messages or [])
        return SkillContext(
            user_input=user_input,
            branch_labels=branch_labels,
            attachments=attachments,
            project_root=str(self.skill_runtime.project.project_root),
            loaded_skill_ids=loaded_skill_ids if loaded_skill_ids is not None else [],
            recent_routing_hint=recent_hint,
            sticky_skill_ids=sticky_skill_ids,
        )

    def _explicit_path_outside_project(self, text: str) -> str:
        project_root = Path(self.skill_runtime.project.project_root)
        for match in _EXPLICIT_PATH_PATTERN.finditer(text or ""):
            raw = match.group("quoted_path") or match.group("plain_path") or ""
            cleaned = raw if match.group("quoted_path") else raw.rstrip(".,:;!?)]}")
            expanded = Path(os.path.expanduser(cleaned))
            if not expanded.is_absolute():
                continue
            resolved = expanded.resolve(strict=False)
            try:
                resolved.relative_to(project_root)
            except ValueError:
                return str(resolved)
        known_directory = _WELL_KNOWN_DIRECTORY_RE.search(text or "")
        has_filesystem_context = bool(_FILESYSTEM_DIRECTORY_CONTEXT_RE.search(text or "") or _PROJECT_FILE_TOKEN_RE.search(text or ""))
        if known_directory and has_filesystem_context:
            directory_name = known_directory.group("directory").capitalize()
            resolved = (Path.home() / directory_name).resolve(strict=False)
            try:
                resolved.relative_to(project_root)
            except ValueError:
                return str(resolved)
        return ""

    def _should_model_classify(self) -> bool:
        return self.llm_client.enable_structured_classification

    @staticmethod
    def _parse_json_object(content: str) -> dict[str, JSONValue]:
        stripped = str(content or "").strip()
        if not stripped:
            return {}
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_RE.search(stripped)
            if not match:
                return {}
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _normalized_text(text: str) -> str:
        return " ".join(str(text or "").strip().lower().split())

    @staticmethod
    def _words(text: str) -> set[str]:
        return set(re.findall(r"[a-z]+", text))

    @classmethod
    def _request_requires_project_mutation(cls, current_user_input: str, recent_routing_hint: str = "") -> bool:
        text = cls._normalized_text(current_user_input)
        if text and cls._words(text) & _MUTATING_WORDS and "make sure" not in text:
            return True
        if cls._non_mutating_actions_in_text(text):
            return False
        hint = cls._normalized_text(recent_routing_hint)
        return bool(hint and cls._words(hint) & _MUTATING_WORDS and "make sure" not in hint)

    @classmethod
    def _non_mutating_actions_in_text(cls, text: str) -> set[str]:
        lowered = cls._normalized_text(text)
        if not lowered:
            return set()
        return {action for action, pattern in _NON_MUTATING_ACTION_PATTERNS.items() if pattern.search(lowered)}

    @classmethod
    def _text_targets_project_artifacts(cls, text: str) -> bool:
        raw = str(text or "")
        lowered = cls._normalized_text(raw)
        if not lowered:
            return False
        if _PROJECT_FILE_TOKEN_RE.search(raw):
            return True
        if _PROJECT_ABS_PATH_RE.search(raw):
            return True
        if re.search(r"\b(?:file|files|folder|folders|filename|filenames)\b", lowered):
            return True
        action_pattern = r"(?:create|make|write|save|edit|update|modify|delete|remove|rename|move|read|open|list|show|inspect|find|copy|scaffold|generate)"
        target_pattern = r"(?:directory|directories|project|repo|repository|project)"
        return bool(
            re.search(rf"\b{action_pattern}\b[^.\n]{{0,40}}\b{target_pattern}\b", lowered)
            or re.search(rf"\b{target_pattern}\b[^.\n]{{0,40}}\b{action_pattern}\b", lowered)
        )

    def _supports_local_project_preference(self, ctx: SkillContext, classification: TurnClassification) -> bool:
        if self._text_targets_project_artifacts(ctx.user_input):
            return True
        if classification.followup_kind in {"confirmation", "contextual_followup"} and self._text_targets_project_artifacts(
            getattr(ctx, "recent_routing_hint", "")
        ):
            return True
        return False

    def _supports_project_action_requirement(self, ctx: SkillContext, classification: TurnClassification) -> bool:
        if self._supports_local_project_preference(ctx, classification):
            return True
        if self._request_requires_project_mutation(ctx.user_input, getattr(ctx, "recent_routing_hint", "")):
            return True
        hint = self._normalized_text(getattr(ctx, "recent_routing_hint", ""))
        if (
            classification.followup_kind in {"confirmation", "contextual_followup"}
            and "project" in hint
            and self._non_mutating_actions_in_text(hint)
        ):
            return True
        return False

    def _structured_classification(
        self,
        prompt: str,
        user_lines: list[str],
        *,
        max_tokens: int,
        pass_id: str,
        failure_event: str,
        stop_event: Any,
    ) -> dict[str, JSONValue]:
        payload = self.llm_client.build_payload(
            [{"role": "system", "content": prompt}, {"role": "user", "content": "\n\n".join(user_lines)}],
            thinking=False,
            tools=None,
            max_tokens_override=max_tokens,
            model_override=self.llm_client.classifier_model if not self.llm_client.classifier_use_primary_model else "",
        )
        try:
            result = self.llm_client.call_with_retry(payload, stop_event, None, pass_id=pass_id)
        except Exception as exc:
            self.telemetry.emit(failure_event, error=str(exc))
            return {}
        return self._parse_json_object(result.content) if result is not None else {}

    def classify(self, ctx: SkillContext, stop_event=None) -> TurnClassification:
        explicit_external_path = self._explicit_path_outside_project(ctx.user_input)
        rule_requires_action = bool(explicit_external_path) and self._request_requires_project_mutation(
            ctx.user_input,
            getattr(ctx, "recent_routing_hint", ""),
        )
        seed = TurnClassification(
            requires_project_action=rule_requires_action,
            prefer_local_project_tools=False,
            explicit_external_path=explicit_external_path,
            source="rules",
        )
        if not self._should_model_classify():
            return seed
        prompt = (
            "Classify the next local assistant turn.\n"
            "Return strict JSON only with these fields:\n"
            '{"time_sensitive":false,"requires_project_action":false,'
            '"prefer_local_project_tools":false,"followup_kind":"new_request"}\n'
            "Allowed followup_kind values: new_request, confirmation, contextual_followup.\n"
            "Set requires_project_action only for actions on project files, folders, projects, or repository state. "
            "A request to create or save a file in the Desktop directory is a filesystem action. "
            "Do not set it for desktop applications, browser actions, screenshots, OCR, package managers, or general system checks.\n"
            "Do not explain."
        )
        user_lines = [f"User request:\n{ctx.user_input}"]
        if ctx.recent_routing_hint:
            user_lines.append(f"Immediate prior exchange:\n{ctx.recent_routing_hint}")
        parsed = self._structured_classification(
            prompt,
            user_lines,
            max_tokens=self.llm_client.max_classifier_tokens,
            pass_id="turn_classify",
            failure_event="turn_classification_failed",
            stop_event=stop_event,
        )
        if not parsed:
            return seed
        followup_kind = str(parsed.get("followup_kind", seed.followup_kind)).strip().lower() or seed.followup_kind
        if followup_kind not in {"new_request", "confirmation", "contextual_followup"}:
            followup_kind = seed.followup_kind
        merged = TurnClassification(
            time_sensitive=bool(parsed.get("time_sensitive", seed.time_sensitive)),
            requires_project_action=seed.requires_project_action or bool(parsed.get("requires_project_action", False)),
            prefer_local_project_tools=seed.prefer_local_project_tools or bool(parsed.get("prefer_local_project_tools", False)),
            explicit_external_path=seed.explicit_external_path,
            followup_kind=followup_kind,
            used_model=True,
            source="model",
        )
        if merged.prefer_local_project_tools and not self._supports_local_project_preference(ctx, merged):
            merged.prefer_local_project_tools = False
        if merged.requires_project_action and not self._supports_project_action_requirement(ctx, merged):
            merged.requires_project_action = False
        self.telemetry.emit(
            "turn_classified",
            source=merged.source,
            followup_kind=merged.followup_kind,
            time_sensitive=merged.time_sensitive,
        )
        return merged

    def reload_skills(self) -> int:
        self.skill_runtime.load_skills()
        return int(getattr(self.skill_runtime, "generation", 0))
