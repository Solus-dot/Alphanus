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
from agent.types import SkillRouteDecision, SkillRoutingSnapshot, TurnClassification

_TIME_SENSITIVE_MARKERS: tuple[str, ...] = (
    "latest",
    "recent",
    "current",
    "today",
    "right now",
    "up to date",
    "as of",
    "news",
    "this week",
    "this month",
)
_WORKSPACE_ACTION_RE = re.compile(r"\b(?:create|make|build|generate|write|save|edit|modify|update|rename|move|delete|remove|clear|wipe|fix|patch)\b")
_WORKSPACE_TARGET_MARKERS: tuple[str, ...] = (
    "workspace",
    "folder",
    "directory",
    "file",
    "files",
    "repo",
    "project",
    "module",
    "package",
    "component",
    "script",
    "code",
    "landing page",
    "scaffold",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".html",
    ".css",
    ".json",
    ".md",
)
_LOCAL_TOOL_BLOCKING_MARKERS: tuple[str, ...] = (
    "shell",
    "terminal",
    "bash",
    "zsh",
    "powershell",
    "run this command",
    "http://",
    "https://",
    "website",
    "web ",
    "search ",
    "google ",
    "fetch ",
)
_FOLLOWUP_STANDALONE_RE = re.compile(r"\b(?:create|make|build|write|edit|modify|update|rename|delete|remove|search|find|open|play|run|install)\b")
_FOLLOWUP_CONTEXT_RE = re.compile(r"\b(?:where|what|which|why|how|it|that|those|them|there|also|then|js|css|html)\b")


class TurnClassifier:
    def __init__(self, config: Dict[str, Any], skill_runtime: SkillRuntime, llm_client: LLMClient, telemetry: Optional[TelemetryEmitter] = None) -> None:
        self.config = config
        self.skill_runtime = skill_runtime
        self.llm_client = llm_client
        self.telemetry = telemetry or TelemetryEmitter()
        self._skill_snapshot: Optional[SkillRoutingSnapshot] = None
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

    @classmethod
    def should_use_recent_routing_hint(cls, ctx: SkillContext) -> bool:
        if not (ctx.recent_routing_hint or ctx.sticky_skill_ids):
            return False
        return cls.is_confirmation_like(ctx.user_input)

    def _combined_request_text(self, ctx: SkillContext, *, include_recent: bool = False) -> str:
        text = str(ctx.user_input or "").strip()
        if include_recent and ctx.recent_routing_hint:
            text = " ".join(part for part in (ctx.recent_routing_hint, text) if part).strip()
        return text.lower()

    def _looks_time_sensitive(self, text: str) -> bool:
        return any(marker in text for marker in _TIME_SENSITIVE_MARKERS)

    def _looks_local_workspace_task(self, text: str, *, explicit_external_path: str) -> bool:
        if explicit_external_path:
            return False
        if any(marker in text for marker in _LOCAL_TOOL_BLOCKING_MARKERS):
            return False
        return bool(_WORKSPACE_ACTION_RE.search(text) or any(marker in text for marker in _WORKSPACE_TARGET_MARKERS))

    def _looks_contextual_followup(self, ctx: SkillContext) -> bool:
        if self.is_confirmation_like(ctx.user_input):
            return False
        if not (ctx.recent_routing_hint or ctx.sticky_skill_ids):
            return False
        text = str(ctx.user_input or "").strip().lower()
        if not text:
            return False
        if _FOLLOWUP_STANDALONE_RE.search(text):
            return False
        tokens = re.findall(r"[a-z0-9']+", text)
        if len(tokens) > 8 and not text.endswith("?"):
            return False
        return bool(text.endswith("?") or _FOLLOWUP_CONTEXT_RE.search(text) or len(tokens) <= 4)

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

    def _deterministic_classification(self, ctx: SkillContext) -> TurnClassification:
        """Minimal seed classification for safety-critical fallback behavior."""
        explicit_external_path = self._explicit_path_outside_workspace(ctx.user_input)
        is_confirmation = self.is_confirmation_like(ctx.user_input)
        is_contextual_followup = self._looks_contextual_followup(ctx)
        followup_kind = "confirmation" if is_confirmation else "contextual_followup" if is_contextual_followup else "new_request"
        merged_text = self._combined_request_text(ctx, include_recent=is_confirmation or is_contextual_followup)
        recent_hint_active = bool(ctx.recent_routing_hint or ctx.sticky_skill_ids)

        candidate_skill_ids: List[str] = []
        if recent_hint_active and followup_kind in {"confirmation", "contextual_followup"}:
            candidate_skill_ids.extend(ctx.sticky_skill_ids[:3])

        return TurnClassification(
            time_sensitive=self._looks_time_sensitive(merged_text),
            requires_workspace_action=is_confirmation and recent_hint_active and self._looks_local_workspace_task(merged_text, explicit_external_path=explicit_external_path),
            prefer_local_workspace_tools=self._looks_local_workspace_task(merged_text, explicit_external_path=explicit_external_path),
            explicit_external_path=explicit_external_path,
            followup_kind=followup_kind,
            candidate_skill_ids=candidate_skill_ids,
        )

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
        seed = self._deterministic_classification(ctx)
        if not self._should_model_classify(ctx, seed):
            return seed

        shortlist = self.skill_runtime.propose_skill_candidates(ctx, top_n=6)
        prompt = (
            "Classify the next local assistant turn.\n"
            "Return strict JSON only with these fields:\n"
            '{"time_sensitive":false,"requires_workspace_action":false,'
            '"prefer_local_workspace_tools":false,"followup_kind":"new_request","candidate_skill_ids":[]}\n'
            "Allowed followup_kind values: new_request, confirmation, contextual_followup.\n"
            "Use candidate_skill_ids only from the supplied shortlist.\n"
            "Do not explain."
        )
        user_lines = [f"User request:\n{ctx.user_input}"]
        if ctx.recent_routing_hint:
            user_lines.append(f"Immediate prior exchange:\n{ctx.recent_routing_hint}")
        if shortlist:
            user_lines.append("Skill shortlist:\n" + self.skill_runtime.skill_cards_text([item.skill for item in shortlist]))
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
        allowed_skills = {item.skill.id for item in shortlist}
        candidate_skill_ids = [
            skill_id for skill_id in [str(item).strip() for item in parsed.get("candidate_skill_ids", []) if str(item).strip()]
            if skill_id in allowed_skills
        ]
        followup_kind = str(parsed.get("followup_kind", seed.followup_kind)).strip().lower() or seed.followup_kind
        if followup_kind not in {"new_request", "confirmation", "contextual_followup"}:
            followup_kind = seed.followup_kind
        merged = TurnClassification(
            time_sensitive=bool(parsed.get("time_sensitive", seed.time_sensitive)),
            requires_workspace_action=bool(parsed.get("requires_workspace_action", seed.requires_workspace_action)),
            prefer_local_workspace_tools=bool(parsed.get("prefer_local_workspace_tools", seed.prefer_local_workspace_tools)),
            explicit_external_path=seed.explicit_external_path,
            followup_kind=followup_kind,
            candidate_skill_ids=candidate_skill_ids or seed.candidate_skill_ids,
            used_model=True,
            source="model",
        )
        self.telemetry.emit(
            "turn_classified",
            source=merged.source,
            followup_kind=merged.followup_kind,
            time_sensitive=merged.time_sensitive,
            candidate_skill_ids=merged.candidate_skill_ids,
        )
        return merged

    def get_skill_snapshot(self, force_refresh: bool = False) -> SkillRoutingSnapshot:
        generation = int(getattr(self.skill_runtime, "generation", 0))
        if not force_refresh and self._skill_snapshot and self._skill_snapshot.generation == generation:
            return self._skill_snapshot
        skills = list(self.skill_runtime.enabled_skills())
        catalog = self.skill_runtime.skill_cards_text(skills)
        self._skill_snapshot = SkillRoutingSnapshot(generation=generation, skills=skills, catalog=catalog)
        return self._skill_snapshot

    def reload_skills(self) -> int:
        self.skill_runtime.load_skills()
        self._skill_snapshot = None
        return int(getattr(self.skill_runtime, "generation", 0))

    @staticmethod
    def _parse_skill_route(content: str, valid_ids: List[str], limit: int) -> Optional[List[str]]:
        try:
            parsed = json.loads(content.strip())
            raw = parsed.get("skills")
            parsed_ids: List[str] = []
            parsed_explicitly = False
            if isinstance(raw, list):
                parsed_explicitly = True
                parsed_ids = [str(item).strip() for item in raw]
            elif raw == []:
                parsed_explicitly = True
        except Exception:
            parsed_ids = []
            parsed_explicitly = False

        lowered = content.lower()
        if parsed_explicitly and not parsed_ids:
            return []
        if lowered in {"none", "no skills", "[]", "{\"skills\": []}", "{\"skills\":[]}"}:
            return []
        if not parsed_ids:
            parsed_ids = [skill_id for skill_id in valid_ids if skill_id.lower() in lowered]
        out: List[str] = []
        seen = set()
        valid = set(valid_ids)
        for skill_id in parsed_ids:
            if skill_id in valid and skill_id not in seen:
                out.append(skill_id)
                seen.add(skill_id)
            if len(out) >= limit:
                break
        return out or None

    def _contextual_skill_fallback(self, ctx: SkillContext) -> List[Any]:
        if not (self.is_confirmation_like(ctx.user_input) and (ctx.recent_routing_hint or ctx.sticky_skill_ids)):
            return []
        merged_text = "\n".join(part for part in (ctx.user_input, ctx.recent_routing_hint) if part).strip()
        if not merged_text:
            return []
        merged_ctx = SkillContext(
            user_input=merged_text,
            branch_labels=list(ctx.branch_labels),
            attachments=list(ctx.attachments),
            workspace_root=ctx.workspace_root,
            memory_hits=list(ctx.memory_hits),
            recent_routing_hint=ctx.recent_routing_hint,
            sticky_skill_ids=list(ctx.sticky_skill_ids),
        )
        return self.skill_runtime.select_skills(merged_ctx)

    def route_skills(self, ctx: SkillContext, classification: TurnClassification, stop_event=None) -> SkillRouteDecision:
        skills_cfg = self.config.get("skills", {}) if isinstance(self.config, dict) else {}
        mode = str(skills_cfg.get("selection_mode", getattr(self.skill_runtime, "selection_mode", ""))).strip().lower()
        configured_limit = int(skills_cfg.get("max_active_skills", getattr(self.skill_runtime, "max_active_skills", 2)))
        limit = configured_limit if configured_limit > 0 else 2

        if ctx.explicit_skill_id:
            selected = self.skill_runtime.select_skills(ctx, top_n=limit)
            return SkillRouteDecision(
                selected_skill_ids=[skill.id for skill in selected],
                shortlisted_skill_ids=[skill.id for skill in selected],
                candidate_skill_ids=[ctx.explicit_skill_id],
                used_model=False,
                source="explicit",
            )

        if mode == "all_enabled":
            selected = self.skill_runtime.select_skills(ctx, top_n=limit)
            return SkillRouteDecision(
                selected_skill_ids=[skill.id for skill in selected],
                shortlisted_skill_ids=[skill.id for skill in selected],
                candidate_skill_ids=list(classification.candidate_skill_ids),
                used_model=False,
                source="all_enabled",
            )

        candidate_items = self.skill_runtime.propose_skill_candidates(ctx, top_n=max(limit * 3, 6))
        heuristic_fallback = self.skill_runtime.rerank_skill_candidates(ctx, candidate_items, limit)
        if mode == "heuristic":
            return SkillRouteDecision(
                selected_skill_ids=[skill.id for skill in heuristic_fallback],
                shortlisted_skill_ids=[item.skill.id for item in candidate_items],
                candidate_skill_ids=list(classification.candidate_skill_ids),
                used_model=False,
                source="heuristic",
            )

        classified_selection = self.skill_runtime.skills_by_ids(classification.candidate_skill_ids[:limit])
        if classified_selection:
            shortlisted_ids = [item.skill.id for item in candidate_items]
            return SkillRouteDecision(
                selected_skill_ids=[skill.id for skill in classified_selection],
                shortlisted_skill_ids=shortlisted_ids,
                candidate_skill_ids=[skill.id for skill in classified_selection],
                used_model=classification.used_model,
                source="classification" if classification.used_model else "contextual",
            )

        snapshot = self.get_skill_snapshot()
        enabled = snapshot.skills
        if not enabled:
            return SkillRouteDecision(source="empty")

        use_recent_hint = classification.followup_kind in {"confirmation", "contextual_followup"} and bool(ctx.recent_routing_hint or ctx.sticky_skill_ids)
        shortlisted_skills = [item.skill for item in candidate_items[: min(len(candidate_items), max(limit * 2, self.skill_runtime.shortlist_size))]]
        if classification.candidate_skill_ids:
            preferred = [skill for skill in shortlisted_skills if skill.id in set(classification.candidate_skill_ids)]
            remainder = [skill for skill in shortlisted_skills if skill.id not in set(classification.candidate_skill_ids)]
            shortlisted_skills = preferred + remainder
        if not shortlisted_skills:
            fallback_ids = [skill.id for skill in heuristic_fallback] or classification.candidate_skill_ids
            return SkillRouteDecision(
                selected_skill_ids=fallback_ids[:limit],
                shortlisted_skill_ids=[],
                candidate_skill_ids=list(classification.candidate_skill_ids),
                used_model=False,
                source="fallback",
            )

        routing_system = (
            "You are selecting local skills for the next assistant turn.\n"
            "Choose the smallest set of skills needed for the user's request.\n"
            "Use the skill descriptions, tags, produced artifact hints, available tool names, and any supplied candidate skills.\n"
            f"Return strict JSON only in the form {{\"skills\": [\"skill-id\"]}} with at most {limit} ids.\n"
            "Return an empty list when no skill is needed.\n"
            "Do not explain your choice."
        )
        if use_recent_hint:
            routing_system += (
                "\nFor short confirmations like yes/continue/do it, use only the immediate prior exchange summary below."
                "\nDo not infer tools from older conversation context."
            )
        user_content = f"User request:\n{ctx.user_input}"
        if use_recent_hint and ctx.recent_routing_hint:
            user_content += f"\n\nImmediate prior exchange:\n{ctx.recent_routing_hint}"
        if classification.candidate_skill_ids:
            user_content += f"\n\nCandidate skills from turn classification:\n{', '.join(classification.candidate_skill_ids[:limit])}"
        user_content += "\n\nSkill shortlist:\n" + self.skill_runtime.skill_cards_text(shortlisted_skills)

        payload = self.llm_client.build_payload(
            [{"role": "system", "content": routing_system}, {"role": "user", "content": user_content}],
            thinking=False,
            tools=None,
            max_tokens_override=self.llm_client.max_classifier_tokens,
            model_override=self.llm_client.classifier_model if not self.llm_client.classifier_use_primary_model else "",
        )
        try:
            result = self.call_with_retry(payload, stop_event, None, pass_id="skill_route")
        except Exception as exc:
            self.telemetry.emit("skill_route_failed", error=str(exc))
            fallback_ids = [skill.id for skill in heuristic_fallback]
            return SkillRouteDecision(
                selected_skill_ids=fallback_ids[:limit],
                shortlisted_skill_ids=[skill.id for skill in shortlisted_skills],
                candidate_skill_ids=list(classification.candidate_skill_ids),
                used_model=False,
                source="fallback",
            )
        skill_ids = self._parse_skill_route(result.content, [skill.id for skill in enabled], limit)
        if skill_ids is None:
            skill_ids = [skill.id for skill in heuristic_fallback[:limit]]
            source = "fallback"
            used_model = False
        else:
            source = "model"
            used_model = True
        self.telemetry.emit("skill_routed", source=source, selected_skill_ids=skill_ids, shortlisted=[skill.id for skill in shortlisted_skills])
        return SkillRouteDecision(
            selected_skill_ids=skill_ids,
            shortlisted_skill_ids=[skill.id for skill in shortlisted_skills],
            candidate_skill_ids=list(classification.candidate_skill_ids),
            used_model=used_model,
            source=source,
        )
