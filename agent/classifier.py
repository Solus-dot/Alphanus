from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.skills import SkillContext, SkillRuntime

from agent.llm_client import LLMClient
from agent.telemetry import TelemetryEmitter
from agent.types import SkillRouteDecision, SkillRoutingSnapshot, TurnClassification


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

    @classmethod
    def is_contextual_followup_like(cls, text: str) -> bool:
        lowered = " ".join(text.strip().lower().split())
        if not lowered or cls.is_confirmation_like(lowered):
            return False
        words = lowered.split()
        if len(words) > 6:
            return False
        prefixes = (
            "where is",
            "where's",
            "what about",
            "and ",
            "also ",
            "add ",
            "include ",
            "make ",
            "put ",
            "now ",
        )
        if any(lowered.startswith(prefix) for prefix in prefixes):
            return True
        return bool(re.search(r"\b(?:js|css|html|script|style|that|it)\b", lowered))

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
                except Exception:
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
        return cls.is_confirmation_like(ctx.user_input) or cls.is_contextual_followup_like(ctx.user_input)

    def _combined_request_text(self, ctx: SkillContext) -> str:
        if self.is_confirmation_like(ctx.user_input):
            text = " ".join(part for part in (ctx.recent_routing_hint, ctx.user_input) if part).strip()
        else:
            text = str(ctx.user_input or "").strip()
        return text.lower()

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
        recent_hint_active = self.should_use_recent_routing_hint(ctx)
        merged_text = " ".join(part for part in (ctx.user_input, ctx.recent_routing_hint if recent_hint_active else "") if part).lower()
        if not merged_text:
            merged_text = str(ctx.user_input or "").lower()
        request_text = self._combined_request_text(ctx)
        explicit_external_path = self._explicit_path_outside_workspace(ctx.user_input)
        followup_kind = "new_request"
        if self.is_confirmation_like(ctx.user_input):
            followup_kind = "confirmation"
        elif self.is_contextual_followup_like(ctx.user_input):
            followup_kind = "contextual_followup"

        time_sensitive = any(
            phrase in merged_text
            for phrase in (
                "latest",
                "recent",
                "current",
                "today",
                "right now",
                "up to date",
                "current situation",
                "news",
                "this week",
                "this month",
                "as of",
            )
        )
        requires_workspace_action = self.is_confirmation_like(ctx.user_input) and recent_hint_active and any(
            marker in merged_text
            for marker in (
                "delete ",
                "remove ",
                "edit ",
                "write ",
                "create ",
                "rename ",
                "workspace",
                "file",
                "folder",
                "directory",
            )
        )
        batch_workspace_action = any(term in merged_text for term in ("delete", "remove", "wipe", "clear")) and (
            "all files" in merged_text
            or "all the files" in merged_text
            or "everything" in merged_text
            or "all contents" in merged_text
            or "entire workspace" in merged_text
            or "whole workspace" in merged_text
            or ("workspace" in merged_text and any(term in merged_text for term in ("all", "everything", "files", "contents")))
        )
        prefer_local_workspace_tools = (
            not explicit_external_path
            and not any(term in merged_text for term in ("shell", "terminal", "bash", "zsh", "cmd ", "powershell", "run this command"))
            and not any(term in merged_text for term in ("http://", "https://", "website", "web ", "search ", "google ", "fetch "))
            and any(
                marker in merged_text
                for marker in (
                    "workspace",
                    "folder",
                    "directory",
                    "file",
                    "files",
                    ".py",
                    "python",
                    "program",
                    "package",
                    "module",
                    "html",
                    "css",
                    "js",
                    "javascript",
                    "script",
                    "style",
                    "landing page",
                    "scaffold",
                    "component",
                )
            )
        )
        workspace_readback_required = any(
            marker in request_text
            for marker in (
                "read it back",
                "read back",
                "read the file back",
                "confirm the contents",
                "summarize what was written",
                "summarise what was written",
                "tell me how many",
                "count the",
                "verify the contents",
            )
        )
        strict_output_requested = any(
            marker in request_text
            for marker in (
                "reply with exactly",
                "respond with exactly",
                "answer with exactly",
                "reply with one line",
                "exactly one line",
                "reply with just",
                "respond with just",
                'exactly: "',
                "exactly: '",
            )
        )
        requires_post_tool_reasoning = strict_output_requested or any(
            marker in request_text
            for marker in (
                " then ",
                " after ",
                "read it back",
                "read back",
                "reply with",
                "respond with",
                "answer with",
                "summarize",
                "summarise",
                "confirm the contents",
                "tell me how many",
                "count the",
                "verify",
                "should i",
                "should we",
                "recommend",
                "umbrella",
                "do i need",
                "need to bring",
                "worth bringing",
            )
        )
        workspace_scaffold_action = any(term in request_text for term in ("make ", "create ", "build ", "generate ", "save it in", "save them in")) and any(
            term in request_text
            for term in (
                "landing page",
                "html",
                "css",
                "javascript",
                "js",
                "python",
                ".py",
                "program",
                "package",
                "module",
                "website",
                "web page",
                "page",
                "scaffold",
            )
        ) and any(
            term in request_text
            for term in (
                "folder",
                "directory",
                "files",
                " file",
                ".py",
                "main.py",
                "html css",
                "html, css",
                "css and javascript",
                "html css and javascript",
            )
        )
        workspace_materialization_target = 0
        if any(term in request_text for term in ("make ", "create ", "build ", "generate ", "write ", "save it in", "save them in")):
            requested_files = 0
            requested_exts = self.skill_runtime.requested_artifact_extensions(ctx)
            if "html" in request_text:
                requested_files += 1
            if "css" in request_text:
                requested_files += 1
            if "javascript" in request_text or " js " in f" {request_text} " or "script.js" in request_text:
                requested_files += 1
            if requested_exts:
                requested_files = max(requested_files, len(requested_exts))
            if requested_files == 0 and any(term in request_text for term in ("landing page", "website", "web page")):
                requested_files = 1
            if requested_files == 0 and any(term in request_text for term in (" file", "files", ".py", ".js", ".html", ".css", "python", "html", "css", "javascript", "js", "script", "code", "program")):
                requested_files = 1
            if "files" in request_text and requested_files < 2:
                requested_files = 2
            workspace_materialization_target = requested_files

        candidate_skill_ids: List[str] = []
        if self.should_use_recent_routing_hint(ctx):
            candidate_skill_ids.extend(ctx.sticky_skill_ids[:3])
        return TurnClassification(
            time_sensitive=time_sensitive,
            requires_workspace_action=requires_workspace_action,
            batch_workspace_action=batch_workspace_action,
            prefer_local_workspace_tools=prefer_local_workspace_tools,
            workspace_scaffold_action=workspace_scaffold_action,
            workspace_materialization_target=workspace_materialization_target,
            workspace_readback_required=workspace_readback_required,
            strict_output_requested=strict_output_requested,
            requires_post_tool_reasoning=requires_post_tool_reasoning,
            explicit_external_path=explicit_external_path,
            followup_kind=followup_kind,
            candidate_skill_ids=candidate_skill_ids,
        )

    def _should_model_classify(self, ctx: SkillContext, seed: TurnClassification) -> bool:
        if not self.llm_client.enable_structured_classification:
            return False
        has_execution_context = bool(
            ctx.sticky_skill_ids
            or "assistant just said:" in (ctx.recent_routing_hint or "")
            or "tools just used:" in (ctx.recent_routing_hint or "")
        )
        return bool(
            has_execution_context
            or seed.requires_workspace_action
            or seed.explicit_external_path
        )

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
            '{"time_sensitive":false,"requires_workspace_action":false,"batch_workspace_action":false,'
            '"prefer_local_workspace_tools":false,"workspace_scaffold_action":false,"workspace_materialization_target":0,'
            '"workspace_readback_required":false,"strict_output_requested":false,"requires_post_tool_reasoning":false,'
            '"followup_kind":"new_request","candidate_skill_ids":[]}\n'
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
            batch_workspace_action=bool(parsed.get("batch_workspace_action", seed.batch_workspace_action)),
            prefer_local_workspace_tools=bool(parsed.get("prefer_local_workspace_tools", seed.prefer_local_workspace_tools)),
            workspace_scaffold_action=bool(parsed.get("workspace_scaffold_action", seed.workspace_scaffold_action)),
            workspace_materialization_target=max(0, int(parsed.get("workspace_materialization_target", seed.workspace_materialization_target) or 0)),
            workspace_readback_required=bool(parsed.get("workspace_readback_required", seed.workspace_readback_required)),
            strict_output_requested=bool(parsed.get("strict_output_requested", seed.strict_output_requested)),
            requires_post_tool_reasoning=bool(parsed.get("requires_post_tool_reasoning", seed.requires_post_tool_reasoning)),
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

    def _prefer_deterministic_skill_selection(self, ctx: SkillContext) -> bool:
        if ctx.explicit_skill_id:
            return True
        ranked = [(score, skill) for score, skill in self.skill_runtime.score_skills(ctx) if score > 0]
        return self.skill_runtime.should_use_deterministic_selection(ranked)

    def _contextual_skill_fallback(self, ctx: SkillContext) -> List[Any]:
        if not self.should_use_recent_routing_hint(ctx):
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
        if mode == "hybrid_lazy" and self._prefer_deterministic_skill_selection(ctx):
            return SkillRouteDecision(
                selected_skill_ids=[skill.id for skill in heuristic_fallback],
                shortlisted_skill_ids=[item.skill.id for item in candidate_items],
                candidate_skill_ids=list(classification.candidate_skill_ids),
                used_model=False,
                source="deterministic",
            )

        snapshot = self.get_skill_snapshot()
        enabled = snapshot.skills
        if not enabled:
            return SkillRouteDecision(source="empty")

        use_recent_hint = self.should_use_recent_routing_hint(ctx)
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
