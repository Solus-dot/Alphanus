from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any, cast

import pytest

from agent.core import Agent
from core.memory import LexicalMemory
from core.project import ProjectRuntime
from core.types import TurnClassification
from skills.runtime import SkillRuntime
from tests.support import build_skill_runtime

pytestmark = pytest.mark.usefixtures("disable_model_classification")

TEST_BASE_URL = "http://127.0.0.1:8080"
TEST_MODEL_ENDPOINT = f"{TEST_BASE_URL}/v1/chat/completions"
TEST_MODELS_ENDPOINT = f"{TEST_BASE_URL}/v1/models"
TEST_SLOTS_ENDPOINT = f"{TEST_BASE_URL}/slots"
TEST_PROPS_ENDPOINT = f"{TEST_BASE_URL}/props"


def agent_config(*, sections=None, **overrides):
    agent = {
        "model_endpoint": TEST_MODEL_ENDPOINT,
        "models_endpoint": TEST_MODELS_ENDPOINT,
        "request_timeout_s": 5,
        "readiness_timeout_s": 1,
        "readiness_poll_s": 0.05,
        "enable_thinking": True,
        "tls_verify": True,
        "max_tokens": 256,
    }
    agent.update(overrides)
    config = {"agent": agent}
    config.update(sections or {})
    return config


class FakeResponse:
    def __init__(self, lines):
        self.lines = [line.encode("utf-8") for line in lines]
        self.status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)

    def read(self):
        return b"".join(self.lines)


@pytest.fixture
def runtime(tmp_path: Path) -> SkillRuntime:
    return build_skill_runtime(
        tmp_path,
        manifest="""
---
name: project-ops
description: project
version: 1.0.0
tools:
  allowed-tools:
    - create_directory
    - create_file
---
project
""",
        tools="""
TOOL_SPECS = {
  "create_directory": {
    "capability": "project_write",
    "description": "Create directory",
    "parameters": {
      "type": "object",
      "properties": {"path": {"type": "string"}},
      "required": ["path"]
    }
  },
  "create_file": {
    "capability": "project_write",
    "description": "Create file",
    "parameters": {
      "type": "object",
      "properties": {"filepath": {"type": "string"}, "content": {"type": "string"}},
      "required": ["filepath", "content"]
    }
  },
}

def execute(tool_name, args, env):
    if tool_name == "create_directory":
        path = env.project.create_directory(args["path"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    if tool_name == "create_file":
        path = env.project.create_file(args["filepath"], args["content"])
        return {"ok": True, "data": {"filepath": path}, "error": None, "meta": {}}
    return {"ok": False, "data": None, "error": {"code": "E_UNSUPPORTED", "message": "nope"}, "meta": {}}
""",
    )


def test_agent_records_model_usage_from_stream(mocker, runtime: SkillRuntime):
    cfg = agent_config()
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"Done"}}]}',
                'data: {"choices":[],"usage":{"prompt_tokens":321,"completion_tokens":12,"total_tokens":333}}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "say done"}],
        user_input="say done",
        thinking=True,
    )

    assert result.status == "done"
    usage = cast(Any, result.journal["model_usage"])
    assert usage["prompt_tokens"] == 321
    assert usage["completion_tokens"] == 12


def test_agent_journal_contains_turn_trace_payload_and_tools(mocker, runtime: SkillRuntime):
    cfg = agent_config()
    agent = Agent(cfg, runtime)

    chat_reqs = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\"trace.txt\\", \\"content\\": \\"hello\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"Done"}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "write trace file"}],
        user_input="write trace file",
        thinking=True,
    )

    assert result.status == "done"
    trace = cast(Any, result.journal.get("turn_trace", {}))
    assert isinstance(trace, dict)
    passes = trace.get("passes", [])
    assert isinstance(passes, list) and passes
    first_pass = passes[0]
    assert isinstance(first_pass, dict)
    assert first_pass.get("payload")
    assert first_pass.get("system_prompt")
    tool_calls = trace.get("tool_calls", [])
    assert isinstance(tool_calls, list) and tool_calls
    assert tool_calls[0].get("name") == "create_file"
    tool_results = trace.get("tool_results", [])
    assert isinstance(tool_results, list) and tool_results
    assert tool_results[0].get("name") == "create_file"


def test_image_turn_without_selected_skill_tools_omits_tool_schemas(mocker, runtime: SkillRuntime):
    cfg = agent_config(enable_structured_classification=True)
    agent = Agent(cfg, runtime)
    requests = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])

        body = json.loads(req.data.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"{\\"time_sensitive\\":false,\\"requires_project_action\\":false,\\"prefer_local_project_tools\\":false,\\"followup_kind\\":\\"new_request\\"}"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"I see an image."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Attachments: image.png (image)]\n\nWhat do you see?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
                ],
            }
        ],
        user_input="What do you see?",
        thinking=True,
    )

    assert result.status == "done"
    assert len(requests) == 1
    assert "tools" not in requests[0]


def test_image_turn_keeps_model_exposed_core_tools_for_project_actions(
    mocker,
    runtime: SkillRuntime,
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = agent_config(enable_structured_classification=True)
    agent = Agent(cfg, runtime)
    requests = []

    monkeypatch.setattr(
        runtime,
        "model_exposed_tool_names",
        lambda: [
            "create_file",
            "request_user_input",
            "skill_view",
            "skills_list",
        ],
    )
    runtime._tools_schema_cache.clear()

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])

        body = json.loads(req.data.decode("utf-8"))
        requests.append(body)
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"Saved the summary."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "[Attachments: screenshot.png (image)]\n\nLook at this screenshot and save a summary to notes.md",
                    },
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
                ],
            }
        ],
        user_input="Look at this screenshot and save a summary to notes.md",
        thinking=True,
    )

    assert result.status == "done"
    assert len(requests) == 1
    assert {tool["function"]["name"] for tool in requests[0]["tools"]} >= {"create_file"}


def test_leading_system_messages_stop_at_first_non_system(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    messages = cast(
        Any,
        [
            {"role": "system", "content": "base prompt"},
            {"role": "system", "content": "policy rules"},
            {"role": "user", "content": "What do you see?"},
            {"role": "system", "content": "ignored trailing system"},
        ],
    )

    assert agent.orchestrator._leading_system_messages(messages) == messages[:2]


def test_image_turn_reports_clear_error_when_backend_rejects_multimodal_prompt(mocker, runtime: SkillRuntime):
    cfg = agent_config(enable_structured_classification=True)
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        raise urllib.error.HTTPError(
            req.full_url,
            400,
            "Bad Request",
            hdrs=Message(),
            fp=io.BytesIO(b'{"error":{"code":400,"message":"Failed to tokenize prompt","type":"invalid_request_error"}}'),
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Attachments: image.png (image)]\n\nWhat do you see?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
                ],
            }
        ],
        user_input="What do you see?",
        thinking=True,
    )

    assert result.status == "error"
    assert result.error == (
        "The current model endpoint rejected this image attachment while tokenizing the prompt. "
        "Use a vision-capable model/template for image inputs, or remove the image attachment."
    )


def test_image_turn_reports_mmproj_hint_when_backend_has_no_image_support(mocker, runtime: SkillRuntime):
    cfg = agent_config(enable_structured_classification=True)
    agent = Agent(cfg, runtime)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        raise urllib.error.HTTPError(
            req.full_url,
            500,
            "Internal Server Error",
            hdrs=Message(),
            fp=io.BytesIO(
                b'{"error":{"code":500,"message":"image input is not supported - hint: if this is unexpected, you may need to provide the mmproj","type":"server_error"}}'
            ),
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Attachments: image.png (image)]\\n\\nWhat do you see?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
                ],
            }
        ],
        user_input="What do you see?",
        thinking=True,
    )

    assert result.status == "error"
    assert result.error == (
        "The current model endpoint does not support image inputs. If you are using llama.cpp, start the "
        "server with a vision-capable model and matching --mmproj file. Otherwise remove the image "
        "attachment or switch to a vision-capable endpoint."
    )


def test_image_turn_retries_with_latest_user_only_after_tokenize_failure(mocker, runtime: SkillRuntime):
    cfg = agent_config(enable_structured_classification=True)
    agent = Agent(cfg, runtime)
    requests = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        body = json.loads(req.data.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            raise urllib.error.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs=Message(),
                fp=io.BytesIO(b'{"error":{"code":400,"message":"Failed to tokenize prompt","type":"invalid_request_error"}}'),
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"I see a status bar screenshot."}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "[Attachments: image.png (image)]\n\nWhat do you see?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
                ],
            }
        ],
        user_input="What do you see?",
        thinking=True,
    )

    assert result.status == "done"
    assert len(requests) == 2
    assert requests[0]["messages"][0]["role"] == "system"
    assert requests[1]["messages"] == [
        requests[0]["messages"][0],
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "[Attachments: image.png (image)]\n\nWhat do you see?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,ZmFrZQ=="}},
            ],
        },
    ]


def test_agent_run_turn_exercises_structured_classification_path(mocker, runtime: SkillRuntime, monkeypatch: pytest.MonkeyPatch):
    cfg = agent_config(enable_structured_classification=True)
    agent = Agent(cfg, runtime)
    requests = []

    monkeypatch.setattr(agent.classifier, "_should_model_classify", lambda: True)

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])

        body = json.loads(req.data.decode("utf-8"))
        requests.append(body)
        system_prompt = body["messages"][0]["content"]
        if "Classify the next local assistant turn." in system_prompt:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"{\\"time_sensitive\\":false,\\"requires_project_action\\":false,\\"prefer_local_project_tools\\":false,\\"followup_kind\\":\\"new_request\\"}"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        return FakeResponse(
            [
                'data: {"choices":[{"delta":{"content":"Done"}}]}',
                'data: {"choices":[{"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "say done"}],
        user_input="say done",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "Done"
    assert len(requests) == 2
    assert requests[0]["chat_template_kwargs"]["enable_thinking"] is False
    assert "tools" not in requests[0]
    assert requests[1]["chat_template_kwargs"]["enable_thinking"] is True


def test_agent_requests_final_answer_if_post_tool_content_empty(mocker, runtime: SkillRuntime):
    cfg = agent_config()
    agent = Agent(cfg, runtime)

    chat_reqs = []
    events = []

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\"a.txt\\", \\"content\\": \\"hello\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"reasoning_content":"I should now answer the user."}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 3:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"Done"}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        raise AssertionError("Unexpected extra completion call")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "write a file"}],
        user_input="write a file",
        thinking=True,
        on_event=events.append,
    )

    assert result.status == "done"
    assert result.content == "Done"
    assert len(chat_reqs) == 3

    finalize_payload = json.loads(chat_reqs[2].data.decode("utf-8"))
    assert finalize_payload.get("chat_template_kwargs", {}).get("enable_thinking") is False
    assert "tools" not in finalize_payload
    assert "tool_choice" not in finalize_payload
    assert any(evt.get("type") == "reasoning_token" for evt in events)
    assert any(
        evt.get("type") == "pass_end"
        and evt.get("finish_reason") == "stop"
        and not evt.get("has_content")
        and not evt.get("has_tool_calls")
        for evt in events
    )


def test_agent_does_not_finish_after_helper_file_for_opaque_artifact_request(mocker, runtime: SkillRuntime):
    cfg = agent_config()
    agent = Agent(cfg, runtime)
    chat_reqs = []

    mocker.patch.object(agent.skill_runtime, "select_skills", return_value=runtime.enabled_skills())

    def fake_urlopen(req, timeout=None, context=None):
        if req.full_url.endswith("/v1/models"):
            return FakeResponse([])
        if req.full_url.endswith("/slots"):
            return FakeResponse(['{"id":0,"n_ctx":40960}'])
        chat_reqs.append(req)
        if len(chat_reqs) == 1:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"create_file","arguments":"{\\"filepath\\": \\"northstar-proposal.js\\", \\"content\\": \\"console.log(1)\\"}"}}]}}]}',
                    'data: {"choices":[{"finish_reason":"tool_calls"}]}',
                    "data: [DONE]",
                ]
            )
        if len(chat_reqs) == 2:
            return FakeResponse(
                [
                    'data: {"choices":[{"delta":{"content":"Continuing with the DOCX workflow."}}]}',
                    'data: {"choices":[{"finish_reason":"stop"}]}',
                    "data: [DONE]",
                ]
            )
        raise AssertionError("Unexpected extra completion call")

    mocker.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "create northstar-proposal.docx"}],
        user_input="create northstar-proposal.docx",
        thinking=True,
    )

    assert result.status == "done"
    assert result.content == "Continuing with the DOCX workflow."
    assert len(chat_reqs) == 2


def test_runtime_list_skills_refreshes_after_generation_changes(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "alpha").mkdir(parents=True)

    (skills / "alpha" / "SKILL.md").write_text(
        """
---
name: alpha
description: Alpha skill.
---
Alpha.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )
    assert [skill.id for skill in runtime.enabled_skills()] == ["alpha"]

    (skills / "beta").mkdir(parents=True)
    (skills / "beta" / "SKILL.md").write_text(
        """
---
name: beta
description: Beta skill.
---
Beta.
""".strip(),
        encoding="utf-8",
    )
    runtime.load_skills()

    assert runtime.generation >= 2
    assert [skill.id for skill in runtime.enabled_skills()] == ["alpha", "beta"]


def test_agent_reload_skills_returns_latest_generation(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "alpha").mkdir(parents=True)

    (skills / "alpha" / "SKILL.md").write_text(
        """
---
name: alpha
description: Alpha skill.
---
Alpha.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )
    agent = Agent({"agent": {}}, runtime)

    generation = agent.reload_skills()

    assert generation == runtime.generation
    assert [skill.id for skill in runtime.enabled_skills()] == ["alpha"]


def test_skill_toggle_bumps_generation_and_updates_enabled_skills(tmp_path: Path):
    home = tmp_path / "home"
    ws = home / "ws"
    skills = tmp_path / "skills"
    home.mkdir()
    ws.mkdir()
    (skills / "alpha").mkdir(parents=True)

    (skills / "alpha" / "SKILL.md").write_text(
        """
---
name: alpha
description: Alpha skill.
---
Alpha.
""".strip(),
        encoding="utf-8",
    )

    runtime = SkillRuntime(
        skills_dir=str(skills),
        project=ProjectRuntime(str(ws)),
        memory=LexicalMemory(storage_path=str(tmp_path / "mem.pkl")),
        config={},
    )
    generation_before = runtime.generation

    assert runtime.set_enabled("alpha", False) is True

    assert runtime.generation == generation_before + 1
    assert runtime.enabled_skills() == []


def test_confirmation_turn_reuses_immediate_prior_skill_context(mocker, runtime: SkillRuntime):
    runtime.config = {}
    agent = Agent({"agent": {}}, runtime)

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        return type("R", (), {"finish_reason": "stop", "content": '{"skills":["project-ops"]}'})()

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)

    history_messages = [
        {"role": "user", "content": "delete all files in the project"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "create_file",
                        "arguments": '{"filepath":"a.txt","content":"hello"}',
                    }
                }
            ],
        },
        {"role": "tool", "name": "create_file", "content": '{"ok": true, "data": {"filepath": "a.txt"}}'},
    ]

    ctx = agent.classifier.build_skill_context("yes", [], [], history_messages)

    assert "previous user request: delete all files in the project" in ctx.recent_routing_hint
    assert "tools just used: create_file" in ctx.recent_routing_hint
    assert ctx.sticky_skill_ids == ["project-ops"]

    assert agent.skill_runtime.select_skills(ctx) == []


def test_contextual_followup_reuses_immediate_prior_skill_context(mocker, runtime: SkillRuntime):
    runtime.config = {}
    agent = Agent({"agent": {}}, runtime)

    history_messages = [
        {"role": "user", "content": "create a bakery landing page in a folder called 1738 with html css js"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "create_file",
                        "arguments": '{"filepath":"1738/index.html","content":"<html></html>"}',
                    }
                }
            ],
        },
        {"role": "tool", "name": "create_file", "content": '{"ok": true, "data": {"filepath": "1738/index.html"}}'},
    ]

    ctx = agent.classifier.build_skill_context("Where is JS?", [], [], history_messages)

    assert agent.skill_runtime.select_skills(ctx) == []


def test_contextual_followup_seed_reuses_immediate_prior_skill_context(runtime: SkillRuntime):
    runtime.config = {}
    agent = Agent({"agent": {}}, runtime)

    history_messages = [
        {"role": "user", "content": "create a bakery landing page in a folder called 1738 with html css js"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "create_file",
                        "arguments": '{"filepath":"1738/index.html","content":"<html></html>"}',
                    }
                }
            ],
        },
        {"role": "tool", "name": "create_file", "content": '{"ok": true, "data": {"filepath": "1738/index.html"}}'},
    ]

    ctx = agent.classifier.build_skill_context("Where is JS?", [], [], history_messages)

    assert agent.skill_runtime.select_skills(ctx) == []


def test_project_action_keeps_project_ops_visible_when_other_skill_loaded(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(
        agent.classifier,
        "classify",
        return_value=TurnClassification(requires_project_action=True, prefer_local_project_tools=True),
    )

    state = agent.orchestrator.prepare_turn(
        history_messages=[],
        user_input="write hello world rust code and save it to the desktop",
        loaded_skill_ids=["memory-rag"],
    )
    system_content = agent.prompt_renderer.compose_system_content(state.selected, state.ctx)

    assert "project-ops" in system_content
    assert "load it with skill_view(name)" in system_content


def test_desktop_file_write_is_deterministic_even_when_model_classifier_misses_it(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    agent.llm_client.enable_structured_classification = True
    mocker.patch.object(
        agent.llm_client,
        "call_with_retry",
        return_value=type(
            "Result",
            (),
            {
                "content": (
                    '{"time_sensitive":false,"requires_project_action":false,'
                    '"prefer_local_project_tools":false,"followup_kind":"new_request"}'
                )
            },
        )(),
    )
    ctx = agent.classifier.build_skill_context(
        "write sorting code in Python to the desktop",
        [],
        [],
    )

    classification = agent.classifier.classify(ctx)

    assert classification.requires_project_action is True
    assert classification.prefer_local_project_tools is False
    assert classification.explicit_external_path == str((Path.home() / "Desktop").resolve())


def test_desktop_ui_language_is_not_mistaken_for_a_filesystem_target(runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    ctx = agent.classifier.build_skill_context(
        "move the browser window to the desktop and make the desktop app responsive",
        [],
        [],
    )

    classification = agent.classifier.classify(ctx)

    assert classification.requires_project_action is False
    assert classification.explicit_external_path == ""


def test_confirmation_project_action_retries_instead_of_accepting_manual_terminal_advice(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent.classifier, "classify", return_value=TurnClassification(requires_project_action=True, followup_kind="confirmation")
    )

    calls = []

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1_project_action_outcome":
            return type("R", (), {"finish_reason": "stop", "content": '{"outcome":"not_completed"}'})()
        if pass_id == "pass_1":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "I cannot create files directly here. You can run touch notes.txt manually in the terminal.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        if pass_id == "pass_2":
            return type(
                "R",
                (),
                {
                    "finish_reason": "tool_calls",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [
                        type(
                            "Call",
                            (),
                            {
                                "stream_id": "2",
                                "index": 0,
                                "id": "call_2",
                                "name": "create_file",
                                "arguments": {"filepath": "notes.txt", "content": "hello"},
                            },
                        )()
                    ],
                },
            )()
        if pass_id == "pass_3":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "Done with project tools.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    def fake_execute_tool_call(tool_name, args, selected, ctx, request_approval=None, **_kwargs):
        assert tool_name == "create_file"
        return {"ok": True, "data": {"filepath": "notes.txt"}, "error": None, "meta": {}}

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)
    mocker.patch.object(runtime, "execute_tool_call", side_effect=fake_execute_tool_call)

    events = []
    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "create notes.txt in the project"}],
        user_input="yes",
        thinking=True,
        on_event=events.append,
    )

    assert result.status == "done"
    assert result.content == "Done with project tools."
    assert calls[-4:] == ["pass_1", "pass_1_project_action_outcome", "pass_2", "pass_3"]
    assert {"type": "discard_pass_output", "pass_id": "pass_1", "reason": "forced_action_retry"} in events


def test_confirmation_project_action_rejects_manual_terminal_advice_after_retry(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent.classifier, "classify", return_value=TurnClassification(requires_project_action=True, followup_kind="confirmation")
    )

    calls = []
    outcome_calls = {"pass_1_project_action_outcome": 0, "pass_2_project_action_outcome": 0}

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1_project_action_outcome":
            outcome_calls[pass_id] += 1
            return type("R", (), {"finish_reason": "stop", "content": '{"outcome":"not_completed"}'})()
        if pass_id == "pass_2_project_action_outcome":
            outcome_calls[pass_id] += 1
            outcome = "not_completed" if outcome_calls[pass_id] == 1 else "declined_or_blocked"
            return type("R", (), {"finish_reason": "stop", "content": f'{{"outcome":"{outcome}"}}'})()
        if pass_id == "pass_2_final":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "No project tool actually ran.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        return type(
            "R",
            (),
            {
                "finish_reason": "stop",
                "content": "I cannot delete files directly. Please run rm -rf manually in your terminal.",
                "reasoning": "",
                "tool_calls": [],
            },
        )()

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "delete all files in the project"}],
        user_input="yes",
        thinking=True,
    )

    assert result.status == "done"
    assert result.error is None
    assert result.content == "No project tool actually ran."
    assert "rm -rf" not in result.content
    assert "pass_2_final" in calls


def test_confirmation_project_action_rejects_claimed_completion_without_tool_use(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent.classifier, "classify", return_value=TurnClassification(requires_project_action=True, followup_kind="confirmation")
    )

    calls = []
    outcome_calls = {"pass_1_project_action_outcome": 0, "pass_2_project_action_outcome": 0}

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1_project_action_outcome":
            outcome_calls[pass_id] += 1
            return type("R", (), {"finish_reason": "stop", "content": '{"outcome":"not_completed"}'})()
        if pass_id == "pass_2_project_action_outcome":
            outcome_calls[pass_id] += 1
            outcome = "not_completed" if outcome_calls[pass_id] == 1 else "declined_or_blocked"
            return type("R", (), {"finish_reason": "stop", "content": f'{{"outcome":"{outcome}"}}'})()
        if pass_id == "pass_2_final":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "No project tool actually ran.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        return type(
            "R",
            (),
            {
                "finish_reason": "stop",
                "content": "I have deleted the following files from your project: a.txt and b.txt. The project is now empty.",
                "reasoning": "",
                "tool_calls": [],
            },
        )()

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "delete all files in the project"}],
        user_input="yes",
        thinking=True,
    )

    assert result.status == "done"
    assert result.error is None
    assert result.content == "No project tool actually ran."
    assert "project is now empty" not in result.content.lower()
    assert "pass_2_final" in calls


def test_confirmation_project_action_requires_mutating_tool_before_accepting_success_claim(mocker, runtime: SkillRuntime):
    agent = Agent({"agent": {}}, runtime)
    mocker.patch.object(agent, "ensure_ready", return_value=True)
    mocker.patch.object(
        agent.classifier, "classify", return_value=TurnClassification(requires_project_action=True, followup_kind="confirmation")
    )

    calls = []
    outcome_calls = {"pass_2_project_action_outcome": 0, "pass_3_project_action_outcome": 0}

    def fake_call_with_retry(payload, stop_event, on_event, pass_id):
        calls.append(pass_id)
        if pass_id == "pass_1":
            return type(
                "R",
                (),
                {
                    "finish_reason": "tool_calls",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [
                        type(
                            "Call",
                            (),
                            {"stream_id": "1", "index": 0, "id": "call_1", "name": "read_file", "arguments": {"filepath": "foo.txt"}},
                        )()
                    ],
                },
            )()
        if pass_id == "pass_2_project_action_outcome":
            outcome_calls[pass_id] += 1
            return type("R", (), {"finish_reason": "stop", "content": '{"outcome":"not_completed"}'})()
        if pass_id == "pass_2":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "I renamed foo.txt to bar.txt.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        if pass_id == "pass_3":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "I renamed foo.txt to bar.txt.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        if pass_id == "pass_3_project_action_outcome":
            outcome_calls[pass_id] += 1
            outcome = "not_completed" if outcome_calls[pass_id] == 1 else "declined_or_blocked"
            return type("R", (), {"finish_reason": "stop", "content": f'{{"outcome":"{outcome}"}}'})()
        if pass_id == "pass_3_final":
            return type(
                "R",
                (),
                {
                    "finish_reason": "stop",
                    "content": "No project tool actually ran.",
                    "reasoning": "",
                    "tool_calls": [],
                },
            )()
        raise AssertionError(f"Unexpected pass id: {pass_id}")

    def fake_execute_tool_call(tool_name, args, selected, ctx, request_approval=None, **_kwargs):
        assert tool_name == "read_file"
        return {"ok": True, "data": {"filepath": "foo.txt", "content": "alpha"}, "error": None, "meta": {}}

    mocker.patch.object(agent.llm_client, "call_with_retry", side_effect=fake_call_with_retry)
    mocker.patch.object(runtime, "execute_tool_call", side_effect=fake_execute_tool_call)

    result = agent.run_turn(
        history_messages=[{"role": "user", "content": "rename foo.txt to bar.txt"}],
        user_input="yes",
        thinking=True,
    )

    assert result.status == "done"
    assert result.error is None
    assert result.content == "No project tool actually ran."
    assert "pass_3" in calls
