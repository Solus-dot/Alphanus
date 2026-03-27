from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.skills import SkillContext, SkillRuntime

from agent.types import TurnPolicySnapshot


def search_rule(*lines: str) -> str:
    return "Search completion rule:\n" + "\n".join(f"- {line}" for line in lines)


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
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?tool_call>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?function(?:=[^>]+)?>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</?parameter(?:=[^>]+)?>", "", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def sanitize_final_content(text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        kept: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if re.fullmatch(r"</?think>", stripped, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"</?tool_call>", stripped, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"<function=[^>]+>", stripped, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"</function>", stripped, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"<parameter=[^>]+>", stripped, flags=re.IGNORECASE):
                continue
            if re.fullmatch(r"</parameter>", stripped, flags=re.IGNORECASE):
                continue
            kept.append(line)
        deduped: List[str] = []
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
        return bool(
            re.search(
                r"<tool_call\b|</tool_call>|<function=[^>]+>|</function>|<parameter=[^>]+>|</parameter>",
                text,
                flags=re.IGNORECASE,
            )
        )


class PromptPolicyRenderer:
    def __init__(self, system_prompt: str, skill_runtime: SkillRuntime) -> None:
        self.system_prompt = system_prompt
        self.skill_runtime = skill_runtime

    def compose_system_content(self, selected: List[Any], ctx: SkillContext) -> str:
        parts = [self.system_prompt]
        if selected:
            skill_block = self.skill_runtime.compose_skill_block(
                selected,
                ctx,
                context_limit=8192,
            )
            if skill_block:
                parts.append("Active skill guidance:\n" + skill_block)
        return "\n\n".join(part.strip() for part in parts if part and part.strip())

    def render_policy_rules(self, snapshot: TurnPolicySnapshot) -> str:
        blocks: List[str] = []
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
            blocks.append(
                "Explicit path rule:\n"
                f"- The user explicitly named a filesystem path outside the current workspace: {snapshot.explicit_external_path}\n"
                "- Do not silently substitute the current workspace root for that path.\n"
                "- Acknowledge the mismatch if you need to reference the current workspace.\n"
                "- If a tool can safely operate on the explicit path, pass that path directly.\n"
                "- If the task requires running a command in that other directory, use a single shell command that targets the explicit path instead of assuming the current workspace."
            )
        if snapshot.prefer_local_workspace_tools:
            block = (
                "Local workspace tool rule:\n"
                "- This request is about local workspace files or folders.\n"
                "- Prefer native workspace tools for local file creation, reading, editing, and folder creation whenever they can directly do the job.\n"
                "- shell_command is still available, but use it only when workspace tools cannot directly accomplish the task or when shell output itself is the requested result.\n"
                "- Do not use web_search, fetch_url, open_url, or play_youtube for local workspace file tasks."
            )
            if snapshot.selected_shell_workflow_skills:
                block += (
                    "\n- Prefer documented selected-skill shell/python workflows when this artifact task genuinely requires shell execution.\n"
                    f"- Skills requiring shell workflow here: {', '.join(snapshot.selected_shell_workflow_skills[:4])}\n"
                    "- Use documented shell/python commands from the selected skill to install missing dependencies and create the requested artifact.\n"
                    "- Each shell_command must be exactly one plain command.\n"
                    "- Do not use shell control operators, chaining, or redirection fallbacks such as `||`, `&&`, `;`, `|`, or `2>&1`.\n"
                    "- Do not use run_checks for dependency probing or installation on this task.\n"
                    "- Do not use python -c import probes before trying the documented install workflow.\n"
                    "- If a dependency is required or uncertain, run the skill's documented install command first after approval.\n"
                    "- Do not create helper .py files until required dependencies are installed.\n"
                    "- Do not use shell_command for generic local file inspection or folder creation when workspace tools already cover it."
                )
            else:
                block += "\n- Do not use shell_command for generic folder creation or local file inspection when workspace tools already cover it."
            blocks.append(block)
        if snapshot.requested_opaque_artifact_extensions and not snapshot.has_selected_materializers:
            blocks.append(
                "Opaque artifact capability rule:\n"
                f"- The request asks for a real opaque artifact: {', '.join(snapshot.requested_opaque_artifact_extensions[:3])}\n"
                "- None of the selected skills expose an executable creation path for that artifact in this runtime.\n"
                "- Do not invent script names.\n"
                "- Do not use shell_command or run_checks to probe or install dependencies for this local workspace file task.\n"
                "- Do not attempt create_file/create_files as a surrogate for the opaque artifact.\n"
                "- Say directly that no executable creation path is available."
            )
        return "\n\n".join(blocks)
