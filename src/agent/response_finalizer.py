from __future__ import annotations

from collections.abc import Callable

from agent.policies import search_rule
from core.message_types import JsonObject
from core.types import AgentTurnResult, TurnState


class ResponseFinalizer:
    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator

    @property
    def sanitizer(self):
        return self.orchestrator.sanitizer

    def _is_stop_requested(self, stop_event) -> bool:
        return self.orchestrator._is_stop_requested(stop_event)

    def _is_plan_mode(self, state: TurnState) -> bool:
        return self.orchestrator._is_plan_mode(state)

    def emit(self, on_event, event: JsonObject) -> None:
        self.orchestrator.emit(on_event, event)

    def project_mutation_count(self, state: TurnState) -> int:
        return self.orchestrator.project_mutation_count(state)

    def project_action_outcome(self, state: TurnState, text: str, *, stop_event, pass_id: str) -> str:
        return self.orchestrator.project_action_outcome(state, text, stop_event=stop_event, pass_id=pass_id)

    def coerce_project_action_failure(self, state: TurnState, result: AgentTurnResult, *, stop_event, pass_id: str) -> AgentTurnResult:
        return self.orchestrator.coerce_project_action_failure(state, result, stop_event=stop_event, pass_id=pass_id)

    def needs_fetch_evidence(self, state: TurnState) -> bool:
        return self.orchestrator.needs_fetch_evidence(state)

    def finalize_turn(
        self, system_content: str, state: TurnState, stop_event, on_event, pass_id: str, extra_rules: str = ""
    ) -> AgentTurnResult:
        return self.orchestrator.finalize_turn(system_content, state, stop_event, on_event, pass_id, extra_rules)

    def finalize_response(
        self,
        *,
        system_content: str,
        state: TurnState,
        pass_id: str,
        stream_result,
        stop_event=None,
        on_event: Callable[[JsonObject], None] | None = None,
    ) -> tuple[str, AgentTurnResult | None]:
        if self._is_stop_requested(stop_event):
            return (
                "result",
                AgentTurnResult(
                    status="cancelled",
                    content="",
                    reasoning=state.full_reasoning,
                    skill_exchanges=state.skill_exchanges,
                ),
            )
        raw_final = stream_result.content
        final = self.sanitizer.sanitize_final_content(raw_final)
        if self.sanitizer.contains_tool_markup(raw_final):
            finalized = self.finalize_turn(
                system_content,
                state,
                stop_event,
                on_event,
                pass_id,
                "Output correction rule:\n- The previous reply emitted raw tool markup instead of a user-facing answer.\n- Rewrite it as a normal assistant response.\n- Do not emit tool markup, XML-like tags, or pseudo-function calls.",
            )
            if finalized.status != "done":
                return "result", finalized
            state.full_reasoning = finalized.reasoning
            final = finalized.content

        if state.search_mode and state.time_sensitive_query and state.completion.tool_counts.get("web_search", 0) == 0:
            if not state.forced_search_retry:
                state.forced_search_retry = True
                self.emit(on_event, {"type": "discard_pass_output", "pass_id": pass_id, "reason": "forced_search_retry"})
                return "continue", None
            finalized = self.finalize_turn(
                system_content,
                state,
                stop_event,
                on_event,
                pass_id,
                search_rule(
                    "This time-sensitive request never performed web_search.",
                    "State plainly that you could not verify the answer from reliable web results in this turn.",
                    "Do not answer from prior knowledge.",
                ),
            )
            return "finalized", finalized

        if not self._is_plan_mode(state) and state.requires_project_action and self.project_mutation_count(state) == 0:
            if not final.strip():
                if not state.forced_action_retry:
                    state.forced_action_retry = True
                    self.emit(on_event, {"type": "discard_pass_output", "pass_id": pass_id, "reason": "forced_action_retry"})
                    return "continue", None
            elif self.project_action_outcome(state, final, stop_event=stop_event, pass_id=pass_id) == "not_completed":
                if not state.forced_action_retry:
                    state.forced_action_retry = True
                    self.emit(on_event, {"type": "discard_pass_output", "pass_id": pass_id, "reason": "forced_action_retry"})
                    return "continue", None
                finalized = self.finalize_turn(
                    system_content,
                    state,
                    stop_event,
                    on_event,
                    pass_id,
                    "Project tool usage rule:\n- No project tool was used to complete the requested action.\n- Say plainly that the action was not completed.\n- Do not provide manual shell deletion advice.\n- Do not claim success.",
                )
                finalized = self.coerce_project_action_failure(state, finalized, stop_event=stop_event, pass_id=pass_id)
                return "finalized", finalized

        if self.needs_fetch_evidence(state):
            finalized = self.finalize_turn(
                system_content,
                state,
                stop_event,
                on_event,
                pass_id,
                search_rule(
                    "This time-sensitive request does not yet have fetched source content.",
                    "State plainly that you could not verify the answer from reliable fetched evidence in this turn.",
                    "Do not speculate or answer from prior knowledge.",
                ),
            )
            return "finalized", finalized

        if not final.strip():
            finalized = self.finalize_turn(system_content, state, stop_event, on_event, pass_id)
            if finalized.status != "done":
                return "result", finalized
            state.full_reasoning = finalized.reasoning
            final = finalized.content
            if self.needs_fetch_evidence(state):
                finalized = self.finalize_turn(
                    system_content,
                    state,
                    stop_event,
                    on_event,
                    pass_id,
                    search_rule(
                        "The prior finalization still lacked fetched source evidence.",
                        "State plainly that you could not verify the answer from reliable fetched evidence in this turn.",
                        "Do not speculate or answer from prior knowledge.",
                    ),
                )
                return "finalized", finalized

        return (
            "result",
            AgentTurnResult(
                status="done",
                content=final,
                reasoning=state.full_reasoning,
                skill_exchanges=state.skill_exchanges,
            ),
        )
