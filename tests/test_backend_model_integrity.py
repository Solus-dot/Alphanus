from __future__ import annotations

from agent.llm_client import LLMClient
from core.types import ModelStatus


def test_strict_local_backend_rejects_model_mismatch(mocker) -> None:
    llm_client = LLMClient({"agent": {"backend_profile": "llamacpp"}})
    mocker.patch.object(
        llm_client.provider,
        "_status_allows_immediate_send",
        return_value=ModelStatus(state="online", model_name="qwen-3"),
    )
    mocker.patch.object(llm_client.provider, "stream_completion")

    try:
        llm_client.call_with_retry(
            {"messages": [{"role": "user", "content": "hello"}], "model": "llava-1.5b"},
            stop_event=None,
            on_event=None,
            pass_id="pass_strict",
        )
        raise AssertionError("Expected model integrity mismatch error")
    except RuntimeError as exc:
        assert "Backend model mismatch" in str(exc)
