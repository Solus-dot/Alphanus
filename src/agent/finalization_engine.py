from __future__ import annotations


class FinalizationEngine:
    def __init__(self, orchestrator) -> None:
        self.orchestrator = orchestrator

    def finalize_turn(self, system_content: str, state, stop_event, on_event, pass_id: str, extra_rules: str = ""):
        return self.orchestrator._finalize_turn_core(
            system_content=system_content,
            state=state,
            stop_event=stop_event,
            on_event=on_event,
            pass_id=pass_id,
            extra_rules=extra_rules,
        )
