"""Keep structured facility evidence when a workflow contains a failed role."""

from __future__ import annotations

import builtins
import errno
import json
from types import SimpleNamespace

import pytest

from opencollab import OpenCollab
from opencollab.application.workflow import WorkflowContext
from opencollab.bootstrap.programmatic import ProgrammaticResult
from opencollab.sdk.client import _public_result
from tests.support.workflow_context_test_support import FakeFactory, FakeSession


def record(error):
    context = WorkflowContext(FakeFactory([]))
    context._record_agent_failure("solver", error)
    return context.agent_failures[0]

def wrapped_storage_failure():
    try:
        raise OSError(errno.ENOSPC, "private storage message", "/private/caller-token")
    except OSError as storage:
        try:
            raise RuntimeError("private wrapped message") from storage
        except RuntimeError as wrapped:
            return wrapped

@pytest.mark.asyncio
async def test_role_preserves_wrapped_storage_errno_without_sensitive_text():
    class FailedSession(FakeSession):
        async def run_loop(self, cancel_event=None):
            raise wrapped_storage_failure()

    context = WorkflowContext(FakeFactory([FailedSession()]))
    assert await context.agent("do the task", label="solver") is None
    failure = context.agent_failures[0]
    assert failure["exception_type"] == "RuntimeError"
    assert failure["exception_chain"] == [
        {"type": "RuntimeError", "module": "builtins"},
        {"type": "OSError", "module": "builtins", "errno": errno.ENOSPC},
    ]
    assert "private" not in json.dumps(failure)

def test_implicit_context_and_exception_group_branches_are_preserved():
    group_type = getattr(builtins, "ExceptionGroup", None)
    if group_type is None:
        pytest.skip("exception groups require Python 3.11")
    error = RuntimeError("outer")
    error.__context__ = group_type("group", [ValueError("first"), OSError(errno.EDQUOT, "second")])
    chain = record(error)["exception_chain"]
    assert [item["type"] for item in chain] == ["RuntimeError", "ExceptionGroup", "ValueError", "OSError"]
    assert chain[-1]["errno"] == errno.EDQUOT

def test_explicit_cause_excludes_suppressed_unrelated_context():
    error = RuntimeError("outer")
    error.__context__ = OSError(errno.ENOSPC, "unrelated handled exception")
    error.__cause__ = ValueError("explicit cause")
    assert [item["type"] for item in record(error)["exception_chain"]] == ["RuntimeError", "ValueError"]

def test_cyclic_and_deep_chains_are_bounded():
    first, second = RuntimeError("first"), OSError(errno.EIO, "second")
    first.__cause__, second.__cause__ = second, first
    assert len(record(first)["exception_chain"]) == 2
    root = current = RuntimeError("root")
    for _ in range(100):
        current.__cause__ = RuntimeError("next")
        current = current.__cause__
    assert len(record(root)["exception_chain"]) == 32

def test_failure_snapshots_cannot_mutate_later_observations():
    context = WorkflowContext(FakeFactory([]))
    context._record_agent_failure("solver", wrapped_storage_failure())
    snapshot = context.agent_failures
    snapshot[0]["exception_chain"][-1]["errno"] = errno.EIO
    assert context.agent_failures[0]["exception_chain"][-1]["errno"] == errno.ENOSPC

def test_nested_http_response_records_numeric_status_without_body():
    class ProviderError(RuntimeError):
        status_code = True
        response = SimpleNamespace(status_code=503)
        body = {"error": {"message": "sensitive credential", "type": "provider_unavailable"}}

    root = RuntimeError("wrapped")
    root.__cause__ = ProviderError("secret")
    failure = record(root)
    assert failure["exception_chain"][-1]["status_code"] == 503
    assert "secret" not in json.dumps(failure)
    assert "credential" not in json.dumps(failure)

def test_plain_existing_record_shape_and_non_numeric_codes_are_preserved():
    error = RuntimeError("ordinary")
    error.errno = True
    error.status_code = "503"
    assert record(error) == {
        "label": "solver", "exception_type": "RuntimeError", "status_code": None,
        "provider_error_type": None,
    }

@pytest.mark.asyncio
async def test_public_workflow_result_exposes_role_error_chain(tmp_path):
    async def failing_role_workflow(context, _arguments):
        class BrokenFactory:
            def build_workflow_session(self, **kwargs):
                raise wrapped_storage_failure()

        context._factory = BrokenFactory()
        await context.agent("preserve a failed role", label="solver")
        return {"status": "incomplete"}

    result = await OpenCollab(tmp_path, model="unused", provider="openai").workflow(
        failing_role_workflow, {}, trace=False,
    )
    assert result.status == "completed"
    assert result.agent_failures[0]["exception_chain"][-1]["errno"] == errno.ENOSPC
    assert "private" not in json.dumps(result.agent_failures)

def test_public_projection_filters_extra_chain_payload_and_copies_nodes():
    original = [{"type": "OSError", "module": "builtins", "errno": errno.ENOSPC,
                 "message": "private", "filename": "/private/credential", "status_code": True}]
    result = _public_result(ProgrammaticResult(
        output=None, status="completed", reason=None, tokens=0, artifacts=None,
        agent_failures=({"label": "solver", "exception_type": "OSError", "exception_chain": original},),
    ))
    assert result.agent_failures[0]["exception_chain"] == [
        {"type": "OSError", "module": "builtins", "errno": errno.ENOSPC},
    ]
    original[0]["errno"] = errno.EIO
    assert result.agent_failures[0]["exception_chain"][0]["errno"] == errno.ENOSPC
