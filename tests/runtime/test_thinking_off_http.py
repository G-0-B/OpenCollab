"""Public reasoning overrides reach official SDK request bodies."""

from __future__ import annotations

import copy
import json

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.openai_provider import _build_request_kwargs
from opencollab.adapters.llm.responses_provider import _build_request_kwargs as _build_responses_kwargs
from opencollab.application.event_bus import EventBus
from opencollab.bootstrap import get_config
from opencollab.bootstrap.config import DEFAULT_THINKING_PARAMS, OpenCollabConfig, build_config
from opencollab.bootstrap.container import build_session
from opencollab.bootstrap.context_builder import ContextBuilder, SpawnConfig
from opencollab.bootstrap.team_config import RoleConfig, TeamConfig
from opencollab.domain.agent import Agent
from tests.support.provider_sdk_http import completion_http_response, http_response, install_sdk_transport


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["workflow", "team", "agent", "session", "client"])
@pytest.mark.parametrize("params,disabled", [
    ({"enable_thinking": True}, {"enable_thinking": False}),
    ({"thinking": {"type": "enabled", "keep": "all"}}, {"thinking": {"type": "disabled"}}),
])
async def test_explicit_thinking_off_sends_native_disable(monkeypatch, tmp_path, entry, params, disabled):
    requests = []
    original = copy.deepcopy(params)

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    install_sdk_transport(monkeypatch, "openai", handler)
    app = OpenCollab(
        tmp_path, model="configured-hybrid", provider="openai",
        api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"thinking": entry in {"workflow", "team"}, "thinking_params": params, "llm_max_retries": 0},
    )
    if entry == "workflow":
        async def flow(ctx, _inputs):
            output = await ctx.agent("answer", tools=(), thinking=False)
            assert ctx.sessions[0].agent.thinking is False
            return output

        result = await app.workflow(flow, trace=False, max_steps=1)
        assert result.ok
    elif entry == "team":
        team = tmp_path / "team.yaml"
        team.write_text("entry: reader\nroles:\n  reader:\n    prompt: Reader\n    tools: []\n    thinking: false\n")
        result = await app.team("answer", config=team, use_worktrees=False, trace=False, max_steps=1)
        assert result.ok
    elif entry == "agent":
        result = await app.agent("answer", tools=(), trace=False, max_steps=1)
        assert result.ok
    elif entry == "session":
        session = build_session(
            agent=Agent(name="reader", system_prompt="Reader", model="configured-hybrid", thinking=False,
                        thinking_params=params, api_key="fake-key-for-test",  # pragma: allowlist secret
                        base_url="https://controlled.invalid/v1", llm_max_retries=0),
        )
        await session.add_user_message("answer")
        await session.run_loop()
        await session.aclose()
        assert session.agent.thinking is False
    else:
        async with LLMClient(model="configured-hybrid", api_key="fake-key-for-test",  # pragma: allowlist secret
                             base_url="https://controlled.invalid/v1", max_retries=0) as client:
            await client.complete([{"role": "user", "content": "answer"}], thinking=False, thinking_params=params)

    assert len(requests) == 1
    assert {key: requests[0].get(key) for key in disabled} == disabled
    assert params == original

@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["chat_completions", "responses"])
@pytest.mark.parametrize("default", [False, True])
@pytest.mark.parametrize("override", [False, True, None])
async def test_workflow_thinking_override_and_inheritance(monkeypatch, tmp_path, wire, default, override):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    install_sdk_transport(monkeypatch, "openai", handler)
    app = OpenCollab(
        tmp_path, model="qwen3.8-flash", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"wire_protocol": wire, "thinking": default,
                "thinking_params": {"enable_thinking": True}, "llm_max_retries": 0},
    )
    expected = default if override is None else override

    async def flow(ctx, _inputs):
        output = await ctx.agent("answer", tools=(), thinking=override)
        assert ctx.sessions[0].agent.thinking is expected
        return output

    assert (await app.workflow(flow, trace=False, max_steps=1)).ok
    assert requests[0]["enable_thinking"] is expected

@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["workflow", "team", "agent"])
async def test_default_ordinary_model_keeps_native_protocol_absent(monkeypatch, tmp_path, entry):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    install_sdk_transport(monkeypatch, "openai", handler)
    app = OpenCollab(
        tmp_path, model="gpt-4o", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", config={"llm_max_retries": 0},
    )
    if entry == "workflow":
        async def flow(ctx, _inputs):
            return await ctx.agent("answer", tools=(), thinking=False)

        result = await app.workflow(flow, trace=False, max_steps=1)
    elif entry == "team":
        team = tmp_path / "team.yaml"
        team.write_text("entry: reader\nroles:\n  reader:\n    prompt: Reader\n    tools: []\n    thinking: false\n")
        result = await app.team("answer", config=team, use_worktrees=False, trace=False, max_steps=1)
    else:
        result = await app.agent("answer", tools=(), trace=False, max_steps=1)
    assert result.ok
    assert len(requests) == 1
    assert "enable_thinking" not in requests[0]
    assert "thinking" not in requests[0]

@pytest.mark.parametrize("env", [None, '{"thinking":{"type":"enabled"}}'])
@pytest.mark.parametrize("override", ["absent", None, {}, {"enable_thinking": True}])
def test_merged_configuration_tracks_protocol_source(monkeypatch, tmp_path, env, override):
    if env is None:
        monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    else:
        monkeypatch.setenv("OPENCOLLAB_THINKING_PARAMS", env)
    overrides = {} if override == "absent" else {"thinking_params": override}
    config = build_config(str(tmp_path), overrides=overrides)
    explicit = override not in ("absent", None) or env is not None
    assert config.thinking_params_explicit is explicit
    assert config.model_dump()["thinking_params_explicit"] is explicit
    if not explicit:
        assert config.thinking_params == DEFAULT_THINKING_PARAMS
    if isinstance(override, dict):
        assert config.thinking_params == override
    if override is None and env is not None:
        assert config.thinking_params == json.loads(env)

@pytest.mark.parametrize("params,explicit", [(None, False), ({}, True), ({"enable_thinking": True}, True)])
def test_config_type_preserves_default_and_explicit_sources(params, explicit):
    assert OpenCollabConfig(thinking_params=params).thinking_params_explicit is explicit
    assert OpenCollabConfig().thinking_params_explicit is False

@pytest.mark.asyncio
async def test_known_hybrid_model_without_declared_params_can_disable(monkeypatch, tmp_path):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    app = OpenCollab(
        tmp_path, model="gateway/qwen3.8-flash-2026-09-21",
        api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", config={"thinking": False, "llm_max_retries": 0},
    )
    assert app.configuration["thinking_params"] == DEFAULT_THINKING_PARAMS
    assert (await app.agent("answer", tools=(), trace=False, max_steps=1)).ok
    assert requests[0]["enable_thinking"] is False

@pytest.mark.asyncio
async def test_fixed_thinking_workflow_keeps_capability_protection_and_effort(monkeypatch, tmp_path):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    app = OpenCollab(
        tmp_path, model="gpt-5.6-luna", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"thinking": True, "thinking_params": {"thinking": {"type": "enabled"}},
                "reasoning_effort": "max", "llm_max_retries": 0},
    )

    async def flow(ctx, _inputs):
        output = await ctx.agent("answer", tools=(), thinking=False)
        assert ctx.sessions[0].agent.thinking is True
        assert ctx.sessions[0].agent.reasoning_effort == "max"
        return output

    assert (await app.workflow(flow, trace=False, max_steps=1)).ok
    assert requests[0]["thinking"] == {"type": "enabled"}
    assert requests[0]["reasoning_effort"] == "max"

@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
async def test_restored_session_keeps_callers_protocol_source(monkeypatch, tmp_path, legacy, explicit):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    params = {"enable_thinking": True}
    agent = Agent(
        name="reader", system_prompt="Reader", model="configured-hybrid", thinking=False,
        thinking_params=params, thinking_params_explicit=explicit,
        api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", llm_max_retries=0,
    )
    source = build_session(agent=agent)
    await source.add_user_message("saved request")
    path = tmp_path / "snapshot.json"
    source.save(str(path))
    await source.aclose()
    if legacy:
        data = json.loads(path.read_text())
        for key in ("snapshot_version", "session_state"):
            data.pop(key, None)
        path.write_text(json.dumps(data))
    restored = build_session(agent=agent)
    restored.restore(str(path))
    await restored.run_loop()
    await restored.aclose()
    assert restored.agent.thinking is False
    assert restored.agent.thinking_params_explicit is explicit
    assert ("enable_thinking" in requests[0]) is explicit
    if explicit:
        assert requests[0]["enable_thinking"] is False
    assert agent.thinking_params == params == {"enable_thinking": True}

@pytest.mark.asyncio
@pytest.mark.parametrize("params", [None, {}, {"enable_thinking": True}, {"thinking": {"type": "enabled"}}])
async def test_direct_spawn_config_tracks_declared_protocol(monkeypatch, tmp_path, params):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    config = SpawnConfig(
        model="configured-hybrid", provider="openai", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", llm_timeout=600, tracer=None,
        event_bus=EventBus(None), permission_policy=None, thinking_params=params,
    )
    assert config.thinking_params_explicit is (params is not None)
    assert config.thinking_params == (DEFAULT_THINKING_PARAMS if params is None else params)
    team = TeamConfig(roles={"reader": RoleConfig(prompt="Reader", tools=[])})
    agent = ContextBuilder(team, config).build_agent("reader")
    session = build_session(agent=agent)
    await session.add_user_message("answer")
    await session.run_loop()
    await session.aclose()
    assert len(requests) == 1
    assert ("enable_thinking" in requests[0]) is (isinstance(params, dict) and "enable_thinking" in params)
    assert ("thinking" in requests[0]) is (isinstance(params, dict) and "thinking" in params)

def test_native_disable_preserves_required_reasoning_replay_and_explicit_effort():
    params = {"thinking": {"type": "enabled", "keep": "all"}}
    kwargs = _build_request_kwargs(
        "deepseek-v4.1-flash", [{"role": "assistant", "content": "answer", "reasoning_content": "prior reasoning"}],
        None, 0.2, thinking=False, thinking_params=params, reasoning_effort="max",
    )
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kwargs["reasoning_effort"] == "max"
    assert kwargs["messages"][0]["reasoning_content"] == "prior reasoning"
    assert params == {"thinking": {"type": "enabled", "keep": "all"}}

@pytest.mark.asyncio
@pytest.mark.parametrize("role_thinking", [False, True, None])
async def test_team_role_thinking_override_and_inheritance(monkeypatch, tmp_path, role_thinking):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    app = OpenCollab(
        tmp_path, model="configured-hybrid", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"thinking": True, "thinking_params": {"enable_thinking": True}, "llm_max_retries": 0},
    )
    team = tmp_path / "team.yaml"
    fields = "" if role_thinking is None else f"    thinking: {str(role_thinking).lower()}\n"
    team.write_text("entry: reader\nroles:\n  reader:\n    prompt: Reader\n    tools: []\n" + fields)
    assert (await app.team("answer", config=team, use_worktrees=False, trace=False, max_steps=1)).ok
    assert requests[0]["enable_thinking"] is (role_thinking is not False)

@pytest.mark.parametrize("model", ["qwen3.8-flash-preview", "qwen-unmeasured", "gpt-4o"])
def test_unknown_protocol_keeps_provider_default(model):
    kwargs = _build_request_kwargs(model, [{"role": "user", "content": "answer"}], None, 0.2)
    assert "extra_body" not in kwargs

@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [False, True])
async def test_get_config_roundtrip_preserves_protocol_source_in_sdk_request(monkeypatch, tmp_path, declared):
    requests = []

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if not declared and ("enable_thinking" in body or "thinking" in body):
            return http_response(request, 400, json={"error": {
                "message": "ordinary provider rejects unsupported thinking extensions",
                "type": "invalid_request_error", "code": "unknown_parameter",
            }})
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("OPENCOLLAB_THINKING", "false")
    if declared:
        monkeypatch.setenv("OPENCOLLAB_THINKING_PARAMS", '{"enable_thinking":true}')
    else:
        monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    config = get_config(str(tmp_path))
    assert config["thinking_params_explicit"] is declared
    original = copy.deepcopy(config)
    app = OpenCollab(
        tmp_path, model="configured-hybrid" if declared else "gpt-4o",
        api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", config=config | {"llm_max_retries": 0},
    )
    assert (await app.agent("answer", tools=(), trace=False, max_steps=1)).ok
    assert app._config["thinking_params_explicit"] is declared
    assert ("enable_thinking" in requests[0]) is declared
    if declared:
        assert requests[0]["enable_thinking"] is False
    assert config == original

@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["workflow", "team", "agent", "session", "client"])
async def test_responses_thinking_off_reaches_all_public_entries(monkeypatch, tmp_path, entry):
    requests = []
    params = {"enable_thinking": True}

    async def handler(request):
        assert request.url.path.endswith("/responses")
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    app = OpenCollab(
        tmp_path, model="qwen3.8-flash", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"wire_protocol": "responses", "thinking": entry in {"workflow", "team"},
                "thinking_params": params, "llm_max_retries": 0},
    )
    if entry == "workflow":
        async def flow(ctx, _inputs):
            output = await ctx.agent("answer", tools=(), thinking=False)
            assert ctx.sessions[0].agent.thinking is False
            return output

        assert (await app.workflow(flow, trace=False, max_steps=1)).ok
    elif entry == "team":
        team = tmp_path / "team.yaml"
        team.write_text("entry: reader\nroles:\n  reader:\n    prompt: Reader\n    tools: []\n    thinking: false\n")
        assert (await app.team("answer", config=team, use_worktrees=False, trace=False, max_steps=1)).ok
    elif entry == "agent":
        assert (await app.agent("answer", tools=(), trace=False, max_steps=1)).ok
    elif entry == "session":
        agent = Agent(
            name="reader", system_prompt="Reader", model="qwen3.8-flash", wire_protocol="responses",
            thinking=False, thinking_params=params, api_key="fake-key-for-test",  # pragma: allowlist secret
            base_url="https://controlled.invalid/v1", llm_max_retries=0,
        )
        session = build_session(agent=agent)
        await session.add_user_message("answer")
        await session.run_loop()
        await session.aclose()
        assert session.agent.thinking is False
    else:
        async with LLMClient(
            model="qwen3.8-flash", wire_protocol="responses",
            api_key="fake-key-for-test",  # pragma: allowlist secret
            base_url="https://controlled.invalid/v1", max_retries=0,
        ) as client:
            await client.complete([{"role": "user", "content": "answer"}], thinking=False, thinking_params=params)
    assert len(requests) == 1
    assert requests[0]["enable_thinking"] is False
    assert params == {"enable_thinking": True}

@pytest.mark.asyncio
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("entry", ["workflow", "team", "agent"])
@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("effort", [None, "high"])
async def test_gpt_responses_default_thinking_preserves_native_request(
    monkeypatch, tmp_path, native, entry, thinking, effort,
):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    app = OpenCollab(
        tmp_path, model="gpt-5.6-luna", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://api.openai.com/v1" if native else "https://controlled.invalid/v1",
        config={"wire_protocol": "responses", "thinking": thinking, "reasoning_effort": effort,
                "llm_max_retries": 0},
    )
    if entry == "workflow":
        async def flow(ctx, _inputs):
            output = await ctx.agent("answer", tools=())
            assert ctx.sessions[0].agent.thinking_params_explicit is False
            return output

        result = await app.workflow(flow, trace=False, max_steps=1)
    elif entry == "team":
        team = tmp_path / "team.yaml"
        team.write_text("entry: reader\nroles:\n  reader:\n    prompt: Reader\n    tools: []\n")
        result = await app.team("answer", config=team, use_worktrees=False, trace=False, max_steps=1)
    else:
        result = await app.agent("answer", tools=(), trace=False, max_steps=1)
    assert result.ok
    assert len(requests) == 1
    assert "enable_thinking" not in requests[0]
    assert "thinking" not in requests[0]
    assert requests[0].get("reasoning") == (None if effort is None else {"effort": effort})

@pytest.mark.asyncio
async def test_fixed_gpt_responses_workflow_keeps_effort_during_off_override(monkeypatch, tmp_path):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    app = OpenCollab(
        tmp_path, model="gpt-5.6-luna", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"wire_protocol": "responses", "thinking": True, "reasoning_effort": "max", "llm_max_retries": 0},
    )

    async def flow(ctx, _inputs):
        output = await ctx.agent("answer", tools=(), thinking=False)
        assert ctx.sessions[0].agent.thinking is True
        assert ctx.sessions[0].agent.reasoning_effort == "max"
        return output

    assert (await app.workflow(flow, trace=False, max_steps=1)).ok
    assert requests[0]["reasoning"] == {"effort": "max"}
    assert "enable_thinking" not in requests[0]

@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("field", [
    "model", "input", "instructions", "tools", "tool_choice", "temperature", "top_p", "store",
    "stream", "include", "reasoning", "max_output_tokens", "prompt_cache_key", "text",
    "previous_response_id", "conversation",
])
def test_responses_thinking_rejects_framework_fields(thinking, field):
    with pytest.raises(ValueError, match=field):
        _build_responses_kwargs(
            "qwen3.8-flash", [{"role": "user", "content": "answer"}], None, 0.2,
            thinking=thinking, thinking_params={"enable_thinking": True, field: "override"}, native_openai=False,
        )

@pytest.mark.asyncio
async def test_chat_off_override_keeps_known_enabled_default_protocol(monkeypatch, tmp_path):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    app = OpenCollab(
        tmp_path, model="qwen3.8-flash", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", config={"thinking": True, "llm_max_retries": 0},
    )
    assert app._config["thinking_params_explicit"] is False

    async def flow(ctx, _inputs):
        return await ctx.agent("answer", tools=(), thinking=False)

    assert (await app.workflow(flow, trace=False, max_steps=1)).ok
    assert requests[0]["enable_thinking"] is False

@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["chat_completions", "responses"])
@pytest.mark.parametrize("global_model", ["qwen3.8-flash", "gpt-4o"])
@pytest.mark.parametrize("role_model", ["gpt-4o", "qwen3.8-flash", "deepseek-v4-flash"])
@pytest.mark.parametrize("clear", [False, True])
async def test_mixed_team_default_protocol_follows_role_model(
    monkeypatch, tmp_path, wire, global_model, role_model, clear,
):
    requests = []

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if ("enable_thinking" in body and body["model"] != "qwen3.8-flash") or (
            "thinking" in body and (wire != "chat_completions" or body["model"] != "deepseek-v4-flash")
        ):
            return http_response(request, 400, json={"error": {
                "message": "role provider rejects an unrelated thinking protocol",
                "type": "invalid_request_error", "code": "unknown_parameter",
            }})
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    config = {"wire_protocol": wire, "thinking": True, "llm_max_retries": 0}
    original = copy.deepcopy(config)
    app = OpenCollab(
        tmp_path, model=global_model, api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", config=config,
    )
    team = tmp_path / "team.yaml"
    params = "    thinking_params: {}\n" if clear else ""
    team.write_text("entry: reader\nroles:\n  reader:\n    prompt: Reader\n    tools: []\n"
                    f"    model: {role_model}\n    thinking: false\n" + params)
    assert (await app.team("answer", config=team, use_worktrees=False, trace=False, max_steps=1)).ok
    assert len(requests) == 1
    assert requests[0]["model"] == role_model
    assert ("enable_thinking" in requests[0]) is (role_model == "qwen3.8-flash")
    if role_model == "qwen3.8-flash":
        assert requests[0]["enable_thinking"] is False
    if wire == "chat_completions" and role_model == "deepseek-v4-flash":
        assert requests[0]["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in requests[0]
    assert config == original
    assert app.configuration["thinking_params"] == DEFAULT_THINKING_PARAMS

@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["chat_completions", "responses"])
@pytest.mark.parametrize("model", ["qwen3.8-flash", "gpt-4o"])
async def test_workflow_default_protocol_uses_declared_source_and_known_model(monkeypatch, tmp_path, wire, model):
    requests = []

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if body["model"] == "gpt-4o" and ("enable_thinking" in body or "thinking" in body):
            return http_response(request, 400, json={"error": {
                "message": "ordinary provider rejects unsupported thinking extensions",
                "type": "invalid_request_error", "code": "unknown_parameter",
            }})
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    app = OpenCollab(
        tmp_path, model=model, api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"wire_protocol": wire, "thinking": True, "llm_max_retries": 0},
    )

    async def flow(ctx, _inputs):
        output = await ctx.agent("answer", tools=(), thinking=False)
        assert ctx.sessions[0].agent.thinking is False
        assert ctx.sessions[0].agent.thinking_params_explicit is False
        return output

    assert (await app.workflow(flow, trace=False, max_steps=1)).ok
    assert len(requests) == 1
    assert ("enable_thinking" in requests[0]) is (model == "qwen3.8-flash")
    if model == "qwen3.8-flash":
        assert requests[0]["enable_thinking"] is False

@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["object", "json"])
@pytest.mark.parametrize("source", [False, True, "absent"])
async def test_get_config_edited_protocol_reaches_sdk_without_mutating_input(monkeypatch, tmp_path, encoding, source):
    requests = []

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        disabled = body.get("thinking") == {"type": "disabled"}
        return completion_http_response(request, text="thinking-off" if disabled else "thinking-on")

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("OPENCOLLAB_THINKING", "false")
    monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    config = get_config(str(tmp_path))
    assert config["thinking_params_explicit"] is False
    params = {"thinking": {"type": "enabled"}}
    config.update(model="deepseek-v4.1-flash", thinking=False,
                  thinking_params=json.dumps(params) if encoding == "json" else params, llm_max_retries=0)
    if source == "absent":
        del config["thinking_params_explicit"]
    else:
        config["thinking_params_explicit"] = source
    original = copy.deepcopy(config)
    app = OpenCollab(
        tmp_path, api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", config=config,
    )
    result = await app.agent("answer", tools=(), trace=False, max_steps=1)
    assert result.ok and result.output == "thinking-off"
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert app._config["thinking_params_explicit"] is True
    assert config == original

@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["object", "json"])
async def test_get_config_same_value_protocol_assignment_preserves_default_source(monkeypatch, tmp_path, encoding):
    requests = []

    async def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        if "enable_thinking" in body or "thinking" in body:
            return http_response(request, 400, json={"error": {
                "message": "ordinary provider rejects unsupported thinking extensions",
                "type": "invalid_request_error", "code": "unknown_parameter",
            }})
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.setenv("OPENCOLLAB_THINKING", "false")
    monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    config = get_config(str(tmp_path))
    config["thinking_params"] = (
        json.dumps(DEFAULT_THINKING_PARAMS) if encoding == "json" else dict(DEFAULT_THINKING_PARAMS)
    )
    original = copy.deepcopy(config)
    app = OpenCollab(
        tmp_path, model="gpt-4o", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", config=config | {"llm_max_retries": 0},
    )
    assert (await app.agent("answer", tools=(), trace=False, max_steps=1)).ok
    assert app._config["thinking_params_explicit"] is False
    assert "enable_thinking" not in requests[0]
    assert "thinking" not in requests[0]
    assert config == original

@pytest.mark.asyncio
@pytest.mark.parametrize("wire", ["chat_completions", "responses"])
@pytest.mark.parametrize("role_model", ["gpt-5.6-luna", "deepseek-v4-flash"])
async def test_mixed_fixed_role_preserves_independent_effort(monkeypatch, tmp_path, wire, role_model):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    app = OpenCollab(
        tmp_path, model="gpt-5.6-luna", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1",
        config={"wire_protocol": wire, "thinking": True, "reasoning_effort": "max", "llm_max_retries": 0},
    )
    team = tmp_path / "team.yaml"
    team.write_text("entry: reader\nroles:\n  reader:\n    prompt: Reader\n    tools: []\n"
                    f"    model: {role_model}\n    thinking: false\n")
    assert (await app.team("answer", config=team, use_worktrees=False, trace=False, max_steps=1)).ok
    assert len(requests) == 1
    assert "enable_thinking" not in requests[0]
    if wire == "chat_completions" and role_model == "deepseek-v4-flash":
        assert requests[0]["thinking"] == {"type": "disabled"}
    else:
        assert "thinking" not in requests[0]
    effort = (
        requests[0].get("reasoning", {}).get("effort")
        if wire == "responses" else requests[0].get("reasoning_effort")
    )
    assert effort == "max"

@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["object", "json"])
async def test_get_config_empty_protocol_stays_explicit_and_empty(monkeypatch, tmp_path, encoding):
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request)

    install_sdk_transport(monkeypatch, "openai", handler)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    config = get_config(str(tmp_path))
    config.update(thinking=True, thinking_params="{}" if encoding == "json" else {}, llm_max_retries=0)
    original = copy.deepcopy(config)
    app = OpenCollab(
        tmp_path, model="gpt-4o", api_key="fake-key-for-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", config=config,
    )
    assert (await app.agent("answer", tools=(), trace=False, max_steps=1)).ok
    assert app._config["thinking_params_explicit"] is True
    assert app.configuration["thinking_params"] == {}
    assert "enable_thinking" not in requests[0]
    assert "thinking" not in requests[0]
    assert config == original

@pytest.mark.parametrize("raw", ["not-json", "[]", '{"thinking":'])
def test_get_config_edited_invalid_json_keeps_existing_validation(monkeypatch, tmp_path, raw):
    monkeypatch.delenv("OPENCOLLAB_THINKING_PARAMS", raising=False)
    config = get_config(str(tmp_path))
    config["thinking_params"] = raw
    original = copy.deepcopy(config)
    with pytest.raises(ValueError, match="OPENCOLLAB_THINKING_PARAMS"):
        OpenCollab(
            tmp_path, model="gpt-4o", api_key="fake-key-for-test",  # pragma: allowlist secret
            base_url="https://controlled.invalid/v1", config=config,
        )
    assert config == original
