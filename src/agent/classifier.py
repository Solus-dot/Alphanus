from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from core.message_types import ChatMessage, JSONValue, MessageContentPart
from core.skills import SkillContext, SkillRuntime

from agent.llm_client import LLMClient
from agent.telemetry import TelemetryEmitter
from core.types import JsonObject, TurnClassification


class TurnClassifier:
    def __init__(self, config: dict[str, object], skill_runtime: SkillRuntime, llm_client: LLMClient, telemetry: Optional[TelemetryEmitter] = None) -> None:
        self.config = config
        self.skill_runtime = skill_runtime
        self.llm_client = llm_client
        self.telemetry = telemetry or TelemetryEmitter()
        self.call_with_retry = llm_client.call_with_retry

    def reload_config(self, config: dict[str, object]) -> None:
        self.config = config

    @staticmethod
    def message_text(value: JSONValue | list[MessageContentPart]) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")).strip())
            return "\n".join(part for part in parts if part).strip()
        return str(value or "").strip()

    def recent_routing_context(self, history_messages: list[ChatMessage]) -> tuple[str, list[str]]:
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
        tool_names: list[str] = []
        assistant_text = ""
        seen_tools = set()
        sticky_skill_ids: list[str] = []
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
                except json.JSONDecodeError as exc:
                    logging.debug("Failed to parse tool result content as JSON: %s", exc)
                    payload = {}
                data = payload.get("data") if isinstance(payload, dict) else {}
                loaded_skill_id = str(data.get("skill_id", "")).strip() if isinstance(data, dict) else ""
                if loaded_skill_id and loaded_skill_id not in seen_skills:
                    sticky_skill_ids.append(loaded_skill_id)
                    seen_skills.add(loaded_skill_id)
        parts: list[str] = []
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
        history_messages: Optional[list[ChatMessage]] = None,
        loaded_skill_ids: Optional[list[str]] = None,
    ) -> SkillContext:
        hits = self.skill_runtime.memory.search(user_input, top_k=3, min_score=0.45)
        recent_hint, sticky_skill_ids = self.recent_routing_context(history_messages or [])
        return SkillContext(
            user_input=user_input,
            branch_labels=branch_labels,
            attachments=attachments,
            workspace_root=str(self.skill_runtime.workspace.workspace_root),
            memory_hits=hits,
            loaded_skill_ids=[
                str(item).strip()
                for item in (loaded_skill_ids or [])
                if str(item).strip()
            ],
            recent_routing_hint=recent_hint,
            sticky_skill_ids=sticky_skill_ids,
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
    def _parse_json_object(content: str) -> dict[str, JSONValue]:
        stripped = str(content or "").strip()
        if not stripped:
            return {}
        try:
            payload = json.loads(stripped)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
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

    @classmethod
    def _draft_requests_clarification(cls, assistant_reply: str) -> bool:
        lowered = cls._normalized_text(assistant_reply)
        if not lowered:
            return False
        if "?" in assistant_reply:
            return True
        return bool(
            re.search(
                r"\b(?:which|what|where|when|who|clarify|confirm|should i|do you want|would you like|can you tell me)\b",
                lowered,
            )
        )

    @classmethod
    def _draft_defers_workspace_action_to_user(cls, assistant_reply: str) -> bool:
        lowered = cls._normalized_text(assistant_reply)
        if not lowered:
            return False
        if "yourself" in lowered or "manually" in lowered:
            return True
        return bool(
            re.search(
                r"\b(?:you can|please|you should)\b[^.\n]{0,100}\b(?:create|delete|remove|rename|move|edit|update|write|save|copy|paste|type|run|execute|use)\b",
                lowered,
            )
        )

    @classmethod
    def _draft_claims_workspace_completion_without_evidence(cls, assistant_reply: str) -> bool:
        lowered = cls._normalized_text(assistant_reply)
        if not lowered:
            return False
        if re.search(
            r"\b(?:i|we)\s+(?:have\s+)?(?:already\s+)?(?:successfully\s+)?(?:deleted|removed|created|updated|edited|modified|renamed|moved|wrote|saved)\b",
            lowered,
        ):
            return True
        return bool(re.search(r"\b(?:workspace is now empty|done with workspace tools)\b", lowered))

    @classmethod
    def _draft_reports_supported_limitation(cls, assistant_reply: str) -> bool:
        lowered = cls._normalized_text(assistant_reply)
        if not lowered:
            return False
        if "no workspace tool actually ran" in lowered:
            return True
        return bool(
            re.search(
                r"\b(?:could not|couldn't|cannot|can't|unable|not allowed|blocked|permission|permissions|unsupported|unavailable|disabled)\b",
                lowered,
            )
        )

    @staticmethod
    def _evidence_shows_blocked_or_unavailable_tool(evidence: dict[str, JSONValue]) -> bool:
        recent_tools = evidence.get("recent_tools")
        if not isinstance(recent_tools, list):
            return False
        for item in recent_tools:
            if not isinstance(item, dict):
                continue
            if bool(item.get("policy_blocked")):
                return True
            error_code = str(item.get("error_code", "")).strip().upper()
            if error_code in {"E_POLICY", "E_PERMISSION", "E_PERMISSIONS", "E_UNSUPPORTED", "E_DISABLED", "E_UNAVAILABLE"}:
                return True
        return False

    @classmethod
    def _text_targets_workspace_artifacts(cls, text: str) -> bool:
        raw = str(text or "")
        lowered = cls._normalized_text(raw)
        if not lowered:
            return False
        if re.search(r"(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\.[a-z0-9]{1,16}\b", raw, flags=re.IGNORECASE):
            return True
        if re.search(r"(?<![:/\w])(?:~/|/)[^\s\"'`]+", raw):
            return True
        if re.search(r"\b(?:file|files|folder|folders|filename|filenames)\b", lowered):
            return True
        action_pattern = (
            r"(?:create|make|write|save|edit|update|modify|delete|remove|rename|move|read|open|list|show|inspect|find|copy|scaffold|generate)"
        )
        target_pattern = r"(?:directory|directories|workspace|repo|repository|project)"
        return bool(
            re.search(rf"\b{action_pattern}\b[^.\n]{{0,40}}\b{target_pattern}\b", lowered)
            or re.search(rf"\b{target_pattern}\b[^.\n]{{0,40}}\b{action_pattern}\b", lowered)
        )

    def _supports_local_workspace_preference(self, ctx: SkillContext, classification: TurnClassification) -> bool:
        if self._text_targets_workspace_artifacts(ctx.user_input):
            return True
        if classification.followup_kind in {"confirmation", "contextual_followup"} and self._text_targets_workspace_artifacts(
            getattr(ctx, "recent_routing_hint", "")
        ):
            return True
        return False

    def classify_workspace_action_outcome(
        self,
        *,
        current_user_input: str,
        recent_routing_hint: str,
        assistant_reply: str,
        evidence: JsonObject,
        pass_id: str,
        stop_event=None,
    ) -> str:
        fallback = self._fallback_workspace_action_outcome(assistant_reply, evidence)
        if not bool(self.llm_client.enable_structured_classification):
            return fallback
        prompt = (
            "Classify an assistant draft for a local workspace action request.\n"
            "Return strict JSON only with this field:\n"
            '{"outcome":"not_completed"}\n'
            "Allowed outcome values: completed_with_evidence, declined_or_blocked, needs_clarification, not_completed.\n"
            "Use the provided tool evidence as the source of truth.\n"
            "- Choose completed_with_evidence only if the evidence shows at least one successful mutating workspace tool.\n"
            "- Choose declined_or_blocked only when the reply transparently reports a real limitation supported by the evidence, such as a policy-blocked tool, unavailable tooling, or an explicit statement that no successful workspace tool actually ran.\n"
            "- Choose needs_clarification when the assistant is explicitly asking the user for information needed before acting.\n"
            "- Choose not_completed for drafts that hand the requested action back to the user, unsupported success claims, or deflections/refusals that are not supported by the evidence.\n"
            "Do not explain."
        )
        user_lines = [f"Current user input:\n{current_user_input}", f"Assistant draft:\n{assistant_reply}", f"Tool evidence:\n{json.dumps(evidence, ensure_ascii=False, default=str)}"]
        if recent_routing_hint:
            user_lines.insert(1, f"Immediate prior exchange:\n{recent_routing_hint}")
        payload = self.llm_client.build_payload(
            [{"role": "system", "content": prompt}, {"role": "user", "content": "\n\n".join(user_lines)}],
            thinking=False,
            tools=None,
            max_tokens_override=min(self.llm_client.max_classifier_tokens, 120),
            model_override=self.llm_client.classifier_model if not self.llm_client.classifier_use_primary_model else "",
        )
        try:
            result = self.call_with_retry(payload, stop_event, None, pass_id=f"{pass_id}_workspace_action_outcome")
        except Exception as exc:
            self.telemetry.emit("workspace_action_outcome_classification_failed", error=str(exc))
            return fallback
        parsed = self._parse_json_object(result.content)
        outcome = str(parsed.get("outcome", "")).strip().lower()
        if outcome in {"completed_with_evidence", "declined_or_blocked", "needs_clarification", "not_completed"}:
            return outcome
        return fallback

    @staticmethod
    def _fallback_workspace_action_outcome(assistant_reply: str, evidence: JsonObject) -> str:
        if bool(evidence.get("has_successful_mutation")):
            return "completed_with_evidence"
        lowered = TurnClassifier._normalized_text(assistant_reply)
        if not lowered:
            return "not_completed"
        if TurnClassifier._draft_requests_clarification(assistant_reply):
            return "needs_clarification"
        if TurnClassifier._draft_defers_workspace_action_to_user(assistant_reply):
            return "not_completed"
        if TurnClassifier._draft_claims_workspace_completion_without_evidence(assistant_reply):
            return "not_completed"
        if TurnClassifier._evidence_shows_blocked_or_unavailable_tool(evidence) and TurnClassifier._draft_reports_supported_limitation(assistant_reply):
            return "declined_or_blocked"
        return "not_completed"

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
        if merged.prefer_local_workspace_tools and not self._supports_local_workspace_preference(ctx, merged):
            merged.prefer_local_workspace_tools = False
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
