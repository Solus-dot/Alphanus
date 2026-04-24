from __future__ import annotations

from typing import Protocol

from core.message_types import ChatMessage
from core.types import JsonObject


class TurnRuntimeHooks(Protocol):
    def call_with_retry(self, payload: JsonObject, stop_event, on_event, pass_id: str):
        ...

    def build_skill_context(
        self,
        user_input: str,
        branch_labels: list[str],
        attachments: list[str],
        history_messages: list[ChatMessage] | None = None,
        loaded_skill_ids: list[str] | None = None,
    ):
        ...

    def classify_context(self, ctx, stop_event=None):
        ...

    def select_skills(self, ctx, stop_event):
        ...


class AgentTurnRuntimeHooks:
    def __init__(self, agent) -> None:
        self.agent = agent

    def call_with_retry(self, payload: JsonObject, stop_event, on_event, pass_id: str):
        return self.agent._call_with_retry(payload, stop_event, on_event, pass_id)

    def build_skill_context(
        self,
        user_input: str,
        branch_labels: list[str],
        attachments: list[str],
        history_messages: list[ChatMessage] | None = None,
        loaded_skill_ids: list[str] | None = None,
    ):
        return self.agent._build_skill_context(
            user_input,
            branch_labels,
            attachments,
            history_messages,
            loaded_skill_ids,
        )

    def classify_context(self, ctx, stop_event=None):
        return self.agent._classify_turn(ctx, stop_event=stop_event)

    def select_skills(self, ctx, stop_event):
        return self.agent._select_turn(ctx, stop_event)

