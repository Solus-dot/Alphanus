from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.skills import SkillContext, SkillRuntime

from agent.llm_client import LLMClient
from agent.telemetry import TelemetryEmitter
from agent.types import TurnClassification


class TurnClassifier:
    def __init__(self, config: Dict[str, Any], skill_runtime: SkillRuntime, llm_client: LLMClient, telemetry: Optional[TelemetryEmitter] = None) -> None:
        self.config = config
        self.skill_runtime = skill_runtime
        self.llm_client = llm_client
        self.telemetry = telemetry or TelemetryEmitter()
        self.call_with_retry = llm_client.call_with_retry

    def reload_config(self, config: Dict[str, Any]) -> None:
        self.config = config

    @staticmethod
    def message_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")).strip())
            return "\n".join(part for part in parts if part).strip()
        return str(value or "").strip()

    @staticmethod
    def is_confirmation_like(text: str) -> bool:
        lowered = " ".join(text.strip().lower().split())
        if not lowered:
            return False
        direct = {
            "yes",
            "yeah",
            "yep",
            "ok",
            "okay",
            "sure",
            "do it",
            "go ahead",
            "continue",
            "proceed",
            "run it",
            "do that",
            "delete them",
            "delete it",
            "all of them",
        }
        if lowered in direct:
            return True
        if len(lowered.split()) <= 3 and lowered in {"please do", "please continue", "yes please"}:
            return True
        affirmative_prefixes = ("yes ", "yeah ", "yep ", "ok ", "okay ", "sure ")
        return len(lowered.split()) <= 6 and any(lowered.startswith(prefix) for prefix in affirmative_prefixes)

    def recent_routing_context(self, history_messages: List[Dict[str, Any]]) -> tuple[str, List[str]]:
        if not history_messages:
            return "", []
        last_user_index = -1
        for idx in range(len(history_messages) - 1, -1, -1):
            if str(history_messages[idx].get("role", "")).lower() == "user":
                last_user_index = idx
                break
        if last_user_index < 0:
            return "", []
        recent = history_messages[last_user_index:]
        recent_user = self.message_text(recent[0].get("content", ""))
        tool_names: List[str] = []
        assistant_text = ""
        seen_tools = set()
        sticky_skill_ids: List[str] = []
        seen_skills = set()
        registry = getattr(self.skill_runtime, "_tool_registry", {})
        for msg in recent[1:]:
            role = str(msg.get("role", "")).lower()
            if role == "assistant":
                text = self.message_text(msg.get("content", ""))
                if text:
                    assistant_text = text
                for call in msg.get("tool_calls", []) or []:
                    name = str(((call or {}).get("function") or {}).get("name", "")).strip()
                    if not name or name in seen_tools:
                        continue
                    tool_names.append(name)
                    seen_tools.add(name)
                    reg = registry.get(name)
                    if reg and reg.skill_id not in seen_skills:
                        sticky_skill_ids.append(reg.skill_id)
                        seen_skills.add(reg.skill_id)
            elif role == "tool":
                name = str(msg.get("name", "")).strip()
                if not name or name in seen_tools:
                    continue
                tool_names.append(name)
                seen_tools.add(name)
                reg = registry.get(name)
                if reg and reg.skill_id not in seen_skills:
                    sticky_skill_ids.append(reg.skill_id)
                    seen_skills.add(reg.skill_id)
                try:
                    payload = json.loads(self.message_text(msg.get("content", "")) or "{}")
                except Exception as exc:
                    logging.debug("Failed to parse tool result content as JSON: %s", exc)
                    payload = {}
                data = payload.get("data") if isinstance(payload, dict) else {}
                loaded_skill_id = str(data.get("skill_id", "")).strip() if isinstance(data, dict) else ""
                if loaded_skill_id and loaded_skill_id not in seen_skills:
                    sticky_skill_ids.append(loaded_skill_id)
                    seen_skills.add(loaded_skill_id)
        parts: List[str] = []
        if recent_user:
            parts.append(f"previous user request: {recent_user}")
        if assistant_text:
            compact_assistant = " ".join(assistant_text.split())
            if len(compact_assistant) > 240:
                compact_assistant = compact_assistant[:237].rstrip() + "..."
            parts.append(f"assistant just said: {compact_assistant}")
        if tool_names:
            parts.append(f"tools just used: {', '.join(tool_names[:4])}")
        return "\n".join(parts), sticky_skill_ids[:3]

    def build_skill_context(
        self,
        user_input: str,
        branch_labels: List[str],
        attachments: List[str],
        history_messages: Optional[List[Dict[str, Any]]] = None,
    ) -> SkillContext:
        explicit_skill_id = ""
        explicit_skill_args = ""
        lowered = user_input.strip()
        match = re.match(r"(?is)^\s*use\s+skill\s+([a-z0-9][a-z0-9-]{0,63})(?::|\s+|$)(.*)$", lowered)
        if match:
            explicit_skill_id = str(match.group(1) or "").strip()
            explicit_skill_args = str(match.group(2) or "").strip()
        hits = self.skill_runtime.memory.search(user_input, top_k=3, min_score=0.45)
        recent_hint, sticky_skill_ids = self.recent_routing_context(history_messages or [])
        return SkillContext(
            user_input=user_input,
            branch_labels=branch_labels,
            attachments=attachments,
            workspace_root=str(self.skill_runtime.workspace.workspace_root),
            memory_hits=hits,
            recent_routing_hint=recent_hint,
            sticky_skill_ids=sticky_skill_ids,
            explicit_skill_id=explicit_skill_id,
            explicit_skill_args=explicit_skill_args,
        )

    def _explicit_path_outside_workspace(self, text: str) -> str:
        workspace_root = Path(self.skill_runtime.workspace.workspace_root)
        seen: set[str] = set()
        path_pattern = re.compile(
            r'(?P<quoted>(?P<quote>["\'`])(?P<quoted_path>(?:~/|/)[^"\'`]+?)(?P=quote))'
            r"|(?P<plain>(?<![:/\w])(?P<plain_path>(?:~/|/)[^\s\"'`]+))"
        )
        for match in path_pattern.finditer(text or ""):
            raw = match.group("quoted_path") or match.group("plain_path") or ""
            cleaned = raw if match.group("quoted_path") else raw.rstrip(".,:;!?)]}")
            expanded = Path(os.path.expanduser(cleaned))
            if not expanded.is_absolute():
                continue
            resolved = expanded.resolve(strict=False)
            resolved_str = str(resolved)
            if resolved_str in seen:
                continue
            seen.add(resolved_str)
            try:
                resolved.relative_to(workspace_root)
            except ValueError:
                return resolved_str
        return ""

    def _should_model_classify(self, ctx: SkillContext, seed: TurnClassification) -> bool:
        """Model classification is the default path.

        Only skipped when structured classification is explicitly disabled
        in the LLM client configuration.
        """
        return bool(self.llm_client.enable_structured_classification)

    @staticmethod
    def _parse_json_object(content: str) -> Dict[str, Any]:
        stripped = str(content or "").strip()
        if not stripped:
            return {}
        try:
            payload = json.loads(stripped)
            return payload if isinstance(payload, dict) else {}
        except Exception:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                return {}
            try:
                payload = json.loads(match.group(0))
            except Exception:
                return {}
            return payload if isinstance(payload, dict) else {}

    def classify(self, ctx: SkillContext, stop_event=None) -> TurnClassification:
        seed = TurnClassification(
            explicit_external_path=self._explicit_path_outside_workspace(ctx.user_input),
            source="fallback",
        )
        if not self._should_model_classify(ctx, seed):
            return seed
        prompt = (
            "Classify the next local assistant turn.\n"
            "Return strict JSON only with these fields:\n"
            '{"time_sensitive":false,"requires_workspace_action":false,'
            '"prefer_local_workspace_tools":false,"followup_kind":"new_request"}\n'
            "Allowed followup_kind values: new_request, confirmation, contextual_followup.\n"
            "Do not explain."
        )
        user_lines = [f"User request:\n{ctx.user_input}"]
        if ctx.recent_routing_hint:
            user_lines.append(f"Immediate prior exchange:\n{ctx.recent_routing_hint}")
        payload = self.llm_client.build_payload(
            [{"role": "system", "content": prompt}, {"role": "user", "content": "\n\n".join(user_lines)}],
            thinking=False,
            tools=None,
            max_tokens_override=self.llm_client.max_classifier_tokens,
            model_override=self.llm_client.classifier_model if not self.llm_client.classifier_use_primary_model else "",
        )
        try:
            result = self.call_with_retry(payload, stop_event, None, pass_id="turn_classify")
        except Exception as exc:
            self.telemetry.emit("turn_classification_failed", error=str(exc))
            return seed
        parsed = self._parse_json_object(result.content)
        if not parsed:
            return seed
        followup_kind = str(parsed.get("followup_kind", seed.followup_kind)).strip().lower() or seed.followup_kind
        if followup_kind not in {"new_request", "confirmation", "contextual_followup"}:
            followup_kind = seed.followup_kind
        merged = TurnClassification(
            time_sensitive=bool(parsed.get("time_sensitive", seed.time_sensitive)),
            requires_workspace_action=bool(parsed.get("requires_workspace_action", seed.requires_workspace_action)),
            prefer_local_workspace_tools=bool(parsed.get("prefer_local_workspace_tools", seed.prefer_local_workspace_tools)),
            explicit_external_path=seed.explicit_external_path,
            followup_kind=followup_kind,
            used_model=True,
            source="model",
        )
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
