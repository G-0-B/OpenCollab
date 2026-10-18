"""Public offline model inspection and profile tool attribution."""

import json

import pytest

from opencollab import OpenCollab
from opencollab.models import inspect_model_runtime
from opencollab.tools import profile_tool_names
from tests.support.provider_sdk_http import completion_http_response, install_sdk_transport


@pytest.mark.parametrize("top_p", [None, 0.85])
@pytest.mark.parametrize(
    ("model", "sampling", "reasoning"),
    [
        ("gpt-4o", True, False),
        ("gateway/gpt-4o-2024-08-06", True, False),
        ("gpt-5", False, True),
        ("gateway/gpt-5-2025-08-07", False, True),
        ("gpt-5.1", True, True),
        ("provider/gpt-5.1", True, True),
        ("gateway/gpt-5.1-2025-11-13", True, True),
        ("gpt-5.2", True, True),
        ("provider/gpt-5.2", True, True),
        ("gateway/gpt-5.2-2025-12-11", True, True),
        ("gpt-5.4", True, True),
        ("provider/gpt-5.4", True, True),
        ("gateway/gpt-5.4-2026-03-05", True, True),
        ("o1-preview", False, True),
        ("gateway/o3-2025-04-16", False, True),
        ("o4-mini", False, True),
    ],
)
async def test_inspection_sampling_defaults_match_sdk_chat_request(
    tmp_path, monkeypatch, model, sampling, reasoning, top_p
):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request, text="connected")

    install_sdk_transport(monkeypatch, "openai", handler)
    app = OpenCollab(
        tmp_path, model=model, provider="openai", api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://fixture.invalid/v1", config={"llm_max_retries": 0},
    )
    async with app.create_model_client() as client:
        await client.complete(
            [{"role": "user", "content": "Confirm the default request."}],
            temperature=0.4, top_p=top_p, max_output_tokens=123,
        )
    assert len(requests) == 1
    request = requests[0]
    snapshot = inspect_model_runtime(model)
    assert snapshot["sends_temperature"] is sampling
    assert snapshot["sends_temperature"] == ("temperature" in request)
    assert snapshot["sends_top_p"] is sampling
    assert ("top_p" in request) is (sampling and top_p is not None)
    assert snapshot["reasoning_request_fields"] is reasoning
    assert request[snapshot["max_output_token_field"]] == 123

@pytest.mark.parametrize("model", ["gpt-5.1", "gpt-5.2", "gpt-5.4"])
async def test_inspection_defaults_are_separate_from_call_reasoning_effort(tmp_path, monkeypatch, model):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion_http_response(request, text="connected")

    install_sdk_transport(monkeypatch, "openai", handler)
    app = OpenCollab(
        tmp_path, model=model, provider="openai", api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://fixture.invalid/v1", config={"llm_max_retries": 0},
    )
    async with app.create_model_client() as client:
        await client.complete(
            [{"role": "user", "content": "Confirm the low-effort request."}],
            temperature=0.4, top_p=0.85, reasoning_effort="low",
        )
    assert len(requests) == 1
    assert requests[0]["reasoning_effort"] == "low"
    assert "temperature" not in requests[0]
    assert "top_p" not in requests[0]
    snapshot = inspect_model_runtime(model)
    assert snapshot["sends_temperature"] is True
    assert snapshot["sends_top_p"] is True

def test_model_inspection_reports_main_luna_settings_without_a_client():
    snapshot = inspect_model_runtime("gateway/gpt-5.6-luna")
    assert snapshot["capabilities"]["context_window"] == 1_048_576
    assert snapshot["capabilities"]["honors_workflow_thinking_override"] is False
    assert snapshot["history"] == {"trigger": 1_015_576, "target": 761_682, "from": "context_window"}
    assert snapshot["reasoning_request_fields"] is True
    assert snapshot["max_output_token_field"] == "max_completion_tokens"
    assert snapshot["sends_temperature"] is False
    assert snapshot["sends_top_p"] is False
    assert snapshot["pricing_mode"] == "unset"

def test_model_inspection_classifies_local_sample_exceptions():
    snapshot = inspect_model_runtime(
        "unknown-model",
        overflow_samples=[
            {"label": "context", "message": "context length exceeded", "status": 400},
            {"label": "metadata", "message": "String too long in metadata", "status": 400},
        ],
        retry_samples=[
            {"label": "saturation", "message": "Concurrency limit exceeded", "class_name": "APIError"},
            {"label": "application", "message": "Concurrency limit exceeded", "class_name": "ValueError"},
            {"label": "auth", "message": "Concurrency limit exceeded", "class_name": "APIError", "status": 401},
        ],
    )
    assert snapshot["history"]["from"] == "fixed_default"
    assert snapshot["overflow"] == {"context": True, "metadata": False}
    assert snapshot["retry"] == {"saturation": True, "application": False, "auth": False}

def test_profile_default_tools_follow_base_single2_composition():
    assert profile_tool_names("Base") == profile_tool_names("single2")
    assert profile_tool_names("Base") == ("bash", "file_read", "file_write", "apply_patch", "git_diff", "grep")
