"""Resource cleanup reports the outcome of every close task."""

from __future__ import annotations

import asyncio

import pytest

from opencollab import OpenCollab, RunError
from opencollab.application.session_lifecycle import close_session_resources
from opencollab.bootstrap import container
from tests.support.session_run_test_support import llm_response


class DelayedCloseSession:
    def __init__(self, cancellation_outcome):
        self.cancellation_outcome = cancellation_outcome
        self.closed = False
        self.cancelled = False
        self.close_calls = 0

    async def aclose(self):
        self.close_calls += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            if self.cancellation_outcome == "cancel":
                raise
            if self.cancellation_outcome == "error":
                raise RuntimeError("close failed after cancellation")
        self.closed = True

@pytest.mark.parametrize("cancellation_outcome", ["cancel", "error", "complete"])
async def test_close_checks_the_result_after_timeout_cancellation(cancellation_outcome):
    session = DelayedCloseSession(cancellation_outcome)
    succeeded = await close_session_resources([session, session], timeout=0.01)
    assert succeeded is (cancellation_outcome == "complete")
    assert session.closed is (cancellation_outcome == "complete")
    assert session.cancelled
    assert session.close_calls == 1

@pytest.mark.parametrize("fail", [False, True])
async def test_close_checks_a_result_returned_within_the_initial_deadline(fail):
    class Session:
        async def aclose(self):
            if fail:
                raise RuntimeError("close failed immediately")

    assert await close_session_resources([Session()], timeout=0.1) is (not fail)

async def test_close_still_reports_failure_when_cancellation_has_not_terminated():
    release = asyncio.Event()
    owners = []

    class Session:
        async def aclose(self):
            owners.append(asyncio.current_task())
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

    try:
        assert await close_session_resources([Session()], timeout=0.01) is False
        assert len(owners) == 1 and not owners[0].done()
    finally:
        release.set()
        await asyncio.gather(*owners)

@pytest.mark.parametrize("cancellation_outcome", ["cancel", "error", "complete"])
async def test_sdk_reports_actual_owned_client_close_result(
    tmp_path, monkeypatch, cancellation_outcome,
):
    clients = []

    class Model(DelayedCloseSession):
        def __init__(self, **_kwargs):
            super().__init__(cancellation_outcome)
            clients.append(self)

        async def complete(self, messages, **_kwargs):
            return llm_response(content="finished", total_tokens=5)

        async def close(self):
            await self.aclose()

    monkeypatch.setattr(container, "LLMClient", Model)
    client = OpenCollab(
        tmp_path, model="fixture", provider="openai",
        api_key="fixture-value",  # pragma: allowlist secret
    )
    if cancellation_outcome == "complete":
        result = await client.agent("finish once", tools=[], trace=False, cleanup_timeout=0.01)
        assert result.status == "completed"
        assert result.output == "finished"
        assert result.metrics["cleanup_quiesced"] is True
    else:
        with pytest.raises(RunError, match="cleanup or persistence"):
            await client.agent("finish once", tools=[], trace=False, cleanup_timeout=0.01)
    assert len(clients) == 1
    assert clients[0].cancelled
    assert clients[0].closed is (cancellation_outcome == "complete")
    assert clients[0].close_calls == 1
