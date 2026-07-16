from __future__ import annotations

from typing import Protocol

from core.message_types import ChatMessage
from core.types import JsonObject


class TurnRuntimeHooks(Protocol):
    def call_with_retry(self, payload: JsonObject, stop_event, on_event, pass_id: str): ...

    def build_skill_context(
        self,
        user_input: str,
        branch_labels: list[str],
        attachments: list[str],
        history_messages: list[ChatMessage] | None = None,
        loaded_skill_ids: list[str] | None = None,
    ): ...

    def classify_context(self, ctx, stop_event=None): ...

    def select_skills(self, ctx, stop_event) -> object: ...
