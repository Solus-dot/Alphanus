from __future__ import annotations

import re

from core.skill_parser import SkillManifest
from core.skills import SkillContext, SkillRuntime
from core.types import TurnPolicySnapshot


def search_rule(*lines: str) -> str:
    return "Search completion rule:\n" + "\n".join(f"- {line}" for line in lines)


_THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)
_TOOL_CALL_TAG_RE = re.compile(r"</?tool_call>", flags=re.IGNORECASE)
_FUNCTION_TAG_RE = re.compile(r"</?function(?:=[^>]+)?>", flags=re.IGNORECASE)
_PARAMETER_TAG_RE = re.compile(r"</?parameter(?:=[^>]+)?>", flags=re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", flags=re.IGNORECASE | re.DOTALL)
_THINK_LINE_RE = re.compile(r"</?think>", flags=re.IGNORECASE)
_TOOL_CALL_LINE_RE = re.compile(r"</?tool_call>", flags=re.IGNORECASE)
_FUNCTION_OPEN_RE = re.compile(r"<function=[^>]+>", flags=re.IGNORECASE)
_FUNCTION_CLOSE_RE = re.compile(r"</function>", flags=re.IGNORECASE)
_PARAMETER_OPEN_RE = re.compile(r"<parameter=[^>]+>", flags=re.IGNORECASE)
_PARAMETER_CLOSE_RE = re.compile(r"</parameter>", flags=re.IGNORECASE)
_TOOL_MARKUP_RE = re.compile(
    r"<tool_call\b|</tool_call>|<function=[^>]+>|</function>|<parameter=[^>]+>|</parameter>",
    flags=re.IGNORECASE,
)


class OutputSanitizer:
    def __init__(self, max_reasoning_chars: int) -> None:
        self.max_reasoning_chars = max_reasoning_chars

    def append_reasoning(self, full_reasoning: str, delta_reasoning: str) -> str:
        if not delta_reasoning:
            return full_reasoning
        delta_reasoning = self.sanitize_reasoning_markup(delta_reasoning)
        if not delta_reasoning:
            return full_reasoning
        if self.max_reasoning_chars <= 0:
            return ""
        if len(full_reasoning) >= self.max_reasoning_chars:
            return full_reasoning
        remaining = self.max_reasoning_chars - len(full_reasoning)
        if len(delta_reasoning) <= remaining:
            return full_reasoning + delta_reasoning
        return full_reasoning + delta_reasoning[:remaining]

    @staticmethod
    def sanitize_reasoning_markup(text: str) -> str:
        if not text:
            return ""
        text = _THINK_TAG_RE.sub("", text)
        text = _TOOL_CALL_TAG_RE.sub("", text)
        text = _FUNCTION_TAG_RE.sub("", text)
        text = _PARAMETER_TAG_RE.sub("", text)
        return text

    @staticmethod
    def sanitize_final_content(text: str) -> str:
        if not text:
            return ""
        text = _THINK_BLOCK_RE.sub("", text)
        text = _TOOL_CALL_BLOCK_RE.sub("", text)
        text = _THINK_TAG_RE.sub("", text)
        kept: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if _THINK_LINE_RE.fullmatch(stripped):
                continue
            if _TOOL_CALL_LINE_RE.fullmatch(stripped):
                continue
            if _FUNCTION_OPEN_RE.fullmatch(stripped):
                continue
            if _FUNCTION_CLOSE_RE.fullmatch(stripped):
                continue
            if _PARAMETER_OPEN_RE.fullmatch(stripped):
                continue
            if _PARAMETER_CLOSE_RE.fullmatch(stripped):
                continue
            kept.append(line)
        deduped: list[str] = []
        previous = None
        for line in kept:
            stripped = line.strip()
            if stripped and stripped == previous:
                continue
            deduped.append(line)
            previous = stripped if stripped else previous
        return "\n".join(deduped).strip()

    @staticmethod
    def contains_tool_markup(text: str) -> bool:
        if not text:
            return False
        return bool(_TOOL_MARKUP_RE.search(text))


class PromptPolicyRenderer:
    def __init__(self, system_prompt: str, skill_runtime: SkillRuntime, context_limit: int = 8192) -> None:
        self.system_prompt = system_prompt
        self.skill_runtime = skill_runtime
        self.context_limit = max(1, int(context_limit))

    def compose_system_content(self, selected: list[SkillManifest], ctx: SkillContext) -> str:
        parts = [self.system_prompt]
        skill_index = self.skill_runtime.compose_skill_index()
        if skill_index:
            parts.append(skill_index)
        if selected:
            skill_block = self.skill_runtime.compose_skill_block(
                selected,
                ctx,
                context_limit=self.context_limit,
            )
            if skill_block:
                parts.append("Loaded skill guidance:\n" + skill_block)
        return "\n\n".join(part.strip() for part in parts if part and part.strip())

    def render_policy_rules(self, snapshot: TurnPolicySnapshot) -> str:
        blocks: list[str] = []
        if str(getattr(snapshot, "collaboration_mode", "execute") or "").strip().lower() == "plan":
            blocks.append(
                "Plan mode rule:\n"
                "- This turn is in plan mode.\n"
                "- You may use only non-mutating tools to inspect context and gather facts.\n"
                "- Do not perform workspace mutations, run shell commands, or execute skill scripts.\n"
                "- If key intent or implementation details are missing, ask a concise follow-up question.\n"
                "- Conclude with a concrete implementation plan that can be executed later."
            )
        if snapshot.search_mode and snapshot.time_sensitive_query and snapshot.forced_search_retry:
            blocks.append(
                "Mandatory retrieval rule:\n"
                "- This user request is time-sensitive.\n"
                "- You must call web_search before answering.\n"
                "- Do not answer from memory cutoff or prior knowledge alone.\n"
                "- If web_search fails, say you could not verify the answer."
            )
        if snapshot.requires_workspace_action and snapshot.forced_action_retry:
            blocks.append(
                "Mandatory action rule:\n"
                "- This is a confirmation of an immediate prior workspace action request.\n"
                "- Use the available workspace tools to perform the requested action if policy allows.\n"
                "- Do not replace an available workspace tool action with manual terminal instructions.\n"
                "- Only decline if the required workspace tool is unavailable or policy blocks the action."
            )
        if snapshot.explicit_external_path:
            block = (
                "Explicit path rule:\n"
                f"- The user explicitly named a filesystem path outside the current workspace: {snapshot.explicit_external_path}\n"
                "- Do not silently substitute the current workspace root for that path.\n"
                "- Acknowledge the mismatch if you need to reference the current workspace.\n"
                "- If a tool can safely operate on the explicit path, pass that path directly.\n"
            )
            if snapshot.shell_tool_exposed:
                block += (
                    "- If command execution in that directory is required, use the exposed shell tool with the explicit path.\n"
                    "- Do not assume commands run from the current workspace when the user named a different path."
                )
            else:
                block += (
                    "- If command execution in that directory is required, first note that no shell tool is exposed in this turn.\n"
                    "- Ask to enable or load the needed skill before attempting command execution there."
                )
            blocks.append(block)
        if snapshot.prefer_local_workspace_tools:
            block = (
                "Local workspace tool rule:\n"
                "- This request is about local workspace files or folders.\n"
                "- Prefer native workspace tools for local file creation, reading, editing, and folder creation whenever they can directly do the job.\n"
                "- Do not use web_search, fetch_url, open_url, or play_youtube for local workspace file tasks."
            )
            if snapshot.shell_tool_exposed:
                block += (
                    "\n- A shell tool is exposed in this turn; use it only when workspace tools cannot directly accomplish the task or when shell output itself is required.\n"
                    "- Do not use a shell tool for generic folder creation or local file inspection when workspace tools already cover it."
                )
            else:
                block += "\n- No shell tool is exposed in this turn; stay within the available workspace tools."
            blocks.append(block)
        return "\n\n".join(blocks)
