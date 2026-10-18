"""Unset optional overrides preserve configured provider requests."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from opencollab import OpenCollab
from opencollab.adapters.cli.main import app
from tests.support.provider_sdk_http import completion_http_response, install_sdk_transport

FILE_KEY = "fake-config-file-key"  # pragma: allowlist secret
FALLBACK_KEY = "fake-exported-fallback-key"  # pragma: allowlist secret
EXPLICIT_KEY = "fake-explicit-override-key"  # pragma: allowlist secret
MAPPING_KEY = "fake-valid-mapping-key"  # pragma: allowlist secret
EXPLICIT_BASE_URL = "https://explicit.example/custom"

@pytest.fixture(params=["openai", "anthropic", "dashscope"])
def configured_request(request, monkeypatch, tmp_path):
    for name in tuple(os.environ):
        if name.startswith(("OPENCOLLAB_", "OPENAI_", "ANTHROPIC_", "DASHSCOPE_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    route = request.param
    provider = "anthropic" if route == "anthropic" else "openai"
    base_url = {
        "openai": "https://configured.example/v1",
        "anthropic": "https://anthropic.example",
        "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    }[route]
    key_variable = {
        "openai": "OPENCOLLAB_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "dashscope": "DASHSCOPE_API_KEY",
    }[route]
    base_variable = "ANTHROPIC_BASE_URL" if provider == "anthropic" else "OPENAI_BASE_URL"
    config = tmp_path / "configured.env"
    config.write_text(
        "OPENCOLLAB_PROVIDER=openai\nOPENCOLLAB_MODEL=fixture-model\n"  # pragma: allowlist secret
        f"{key_variable}={FILE_KEY}\n{base_variable}={base_url}\n"
        "OPENCOLLAB_LLM_MAX_RETRIES=0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCOLLAB_CONFIG_FILE", str(config))
    fallback_variable = "OPENAI_API_KEY" if route == "openai" else "OPENCOLLAB_API_KEY"
    monkeypatch.setenv(fallback_variable, FALLBACK_KEY)
    captured = []

    def handler(request):
        captured.append({
            "url": str(request.url),
            "key": request.headers.get("x-api-key")
            or request.headers["authorization"].removeprefix("Bearer "),
        })
        return completion_http_response(request, text="connected")

    install_sdk_transport(monkeypatch, provider, handler)
    return SimpleNamespace(
        provider=provider, route=route, base_url=base_url, requests=captured,
        config_file=config, fallback_variable=fallback_variable,
        mapping_base_url=f"https://mapping.{route}.example" + ("" if provider == "anthropic" else "/v1"),
    )

def _request(provider, base_url, key):
    path = "v1/messages" if provider == "anthropic" else "chat/completions"
    return {"url": f"{base_url}/{path}", "key": key}

def _expected_request(configured, field=None):
    base_url = EXPLICIT_BASE_URL if field == "base_url" else configured.base_url
    key = EXPLICIT_KEY if field == "api_key" else FILE_KEY
    if field == "base_url" and configured.route == "dashscope":
        key = FALLBACK_KEY
    return _request(configured.provider, base_url, key)

async def _complete(client, entry="model"):
    if entry == "agent":
        result = await client.agent("Connect once.", tools=(), trace=False)
        assert result.ok
        assert result.output == "connected"
    else:
        async with client.create_model_client() as model:
            response = await model.complete([{"role": "user", "content": "Connect once."}])
        assert response.content == "connected"

@pytest.mark.parametrize("field", ["api_key", "base_url"])
@pytest.mark.parametrize("value", ["", " \t\n ", None], ids=["empty", "whitespace", "none"])
@pytest.mark.parametrize("source", ["config", "argument"])
async def test_sdk_unset_optional_override_keeps_file_request(
    tmp_path, configured_request, field, value, source,
):
    kwargs = {"provider": configured_request.provider}
    if source == "config":
        kwargs["config"] = {field: value}
    else:
        kwargs[field] = value
    await _complete(OpenCollab(tmp_path, **kwargs))
    assert configured_request.requests == [_expected_request(configured_request)]

@pytest.mark.parametrize("field", ["api_key", "base_url"])
@pytest.mark.parametrize("source", ["config", "argument"])
async def test_sdk_explicit_optional_override_controls_request(
    tmp_path, configured_request, field, source,
):
    value = EXPLICIT_KEY if field == "api_key" else EXPLICIT_BASE_URL
    kwargs = {"provider": configured_request.provider}
    if source == "config":
        kwargs["config"] = {field: f" {value} "}
    else:
        kwargs[field] = f" {value} "
    await _complete(OpenCollab(tmp_path, **kwargs))
    assert configured_request.requests == [_expected_request(configured_request, field)]

def _mapping_config(configured, monkeypatch, fallback):
    if not fallback:
        configured.config_file.write_text("", encoding="utf-8")
        monkeypatch.delenv(configured.fallback_variable)
    return {
        "provider": configured.provider, "model": "fixture-model",
        "api_key": MAPPING_KEY, "base_url": configured.mapping_base_url,
        "llm_max_retries": 0,
    }

@pytest.mark.parametrize("field", ["api_key", "base_url"])
@pytest.mark.parametrize("value", ["", " \t\n ", None], ids=["empty", "whitespace", "none"])
@pytest.mark.parametrize("fallback", [False, True], ids=["no-fallback", "with-fallback"])
@pytest.mark.parametrize("entry", ["model", "agent"])
async def test_sdk_unset_argument_keeps_mapping_request(
    tmp_path, monkeypatch, configured_request, field, value, fallback, entry,
):
    config = _mapping_config(configured_request, monkeypatch, fallback)
    await _complete(OpenCollab(tmp_path, config=config, **{field: value}), entry)
    assert configured_request.requests == [
        _request(configured_request.provider, configured_request.mapping_base_url, MAPPING_KEY),
    ]

@pytest.mark.parametrize("field", ["api_key", "base_url"])
@pytest.mark.parametrize("fallback", [False, True], ids=["no-fallback", "with-fallback"])
@pytest.mark.parametrize("entry", ["model", "agent"])
async def test_sdk_explicit_argument_overrides_mapping_request(
    tmp_path, monkeypatch, configured_request, field, fallback, entry,
):
    config = _mapping_config(configured_request, monkeypatch, fallback)
    value = EXPLICIT_KEY if field == "api_key" else EXPLICIT_BASE_URL
    await _complete(OpenCollab(tmp_path, config=config, **{field: f" {value} "}), entry)
    base_url = EXPLICIT_BASE_URL if field == "base_url" else configured_request.mapping_base_url
    key = EXPLICIT_KEY if field == "api_key" else MAPPING_KEY
    assert configured_request.requests == [_request(configured_request.provider, base_url, key)]

def _invoke_cli(tmp_path, configured, field, value):
    team = tmp_path / "team.yaml"
    team.write_text(
        "entry: reader\nroles:\n  reader:\n    prompt: Connect once.\n    tools: []\n",
        encoding="utf-8",
    )
    arguments = [
        "--workspace", str(tmp_path), "--provider", configured.provider,
        "--team-config", str(team), "--no-worktrees", "--prompt", "Connect once.",
    ]
    if value is not None:
        arguments.extend([f"--{field.replace('_', '-')}", value])
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "connected" in result.output

@pytest.mark.parametrize("field", ["api_key", "base_url"])
@pytest.mark.parametrize("value", ["", " \t ", None], ids=["empty", "whitespace", "omitted"])
def test_cli_unset_optional_override_keeps_file_request(
    tmp_path, configured_request, field, value,
):
    _invoke_cli(tmp_path, configured_request, field, value)
    assert configured_request.requests == [_expected_request(configured_request)]

@pytest.mark.parametrize("field", ["api_key", "base_url"])
def test_cli_explicit_optional_override_controls_request(tmp_path, configured_request, field):
    value = EXPLICIT_KEY if field == "api_key" else EXPLICIT_BASE_URL
    _invoke_cli(tmp_path, configured_request, field, f" {value} ")
    assert configured_request.requests == [_expected_request(configured_request, field)]
