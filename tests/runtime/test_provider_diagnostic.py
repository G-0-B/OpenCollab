from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from opencollab.bootstrap.config import OpenCollabConfig
from scripts import check_dashscope


async def test_provider_diagnostic_uses_framework_configuration(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        async def complete(self, messages, **kwargs):
            captured["messages"] = messages
            captured["request"] = kwargs
            return SimpleNamespace(content="connected")

        async def close(self):
            captured["closed"] = True

    monkeypatch.setattr(
        check_dashscope,
        "build_config",
        lambda workspace: OpenCollabConfig(
            model="configured-model",
            provider="configured-provider",
            api_key="configured-key",
            base_url="https://provider.invalid/v1",
            llm_timeout=12.0,
            max_output_tokens=4096,
            wire_protocol="responses",
            llm_max_retries=1,
            context_window=32768,
            llm_connect_timeout=3.0,
            llm_first_event_timeout=4.0,
            llm_stream_idle_timeout=5.0,
            llm_stream_chat=True,
            provider_error_time_budget=6.0,
        ),
    )

    result = await check_dashscope.request_completion(
        "probe",
        workspace=Path("/workspace"),
        client_type=FakeClient,
    )

    assert result == "connected"
    assert captured["client"] == {
        "model": "configured-model",
        "provider": "configured-provider",
        "api_key": "configured-key",
        "base_url": "https://provider.invalid/v1",
        "request_timeout": 12.0,
        "wire_protocol": "responses",
        "max_retries": 1,
        "context_window": 32768,
        "connect_timeout": 3.0,
        "first_event_timeout": 4.0,
        "stream_idle_timeout": 5.0,
        "stream_chat": True,
        "provider_error_time_budget": 6.0,
    }
    assert captured["messages"][-1] == {"role": "user", "content": "probe"}
    assert captured["request"] == {
        "temperature": 0.0, "max_output_tokens": 256,
        "thinking": False, "thinking_params": None,
        "reasoning_effort": None,
    }
    assert captured["closed"] is True

async def test_provider_diagnostic_closes_injected_client_on_failure(monkeypatch):
    clients = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.closed = False
            clients.append(self)

        async def complete(self, messages, **kwargs):
            raise RuntimeError("provider failed")

        async def close(self):
            self.closed = True

    monkeypatch.setattr(
        check_dashscope, "build_config",
        lambda workspace: OpenCollabConfig(api_key="test-placeholder"),  # pragma: allowlist secret
    )
    with pytest.raises(RuntimeError, match="provider failed"):
        await check_dashscope.request_completion("connect", client_type=FakeClient)
    assert clients[0].closed is True
