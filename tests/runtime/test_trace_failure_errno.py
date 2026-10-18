"""Keep the actual persistence errno across sticky tracing failures."""

from __future__ import annotations

import errno
from types import SimpleNamespace

import pytest

from opencollab import OpenCollab, RunError
from opencollab.adapters import trace
from opencollab.bootstrap import programmatic
from opencollab.bootstrap._workflow_runtime_cleanup import _sticky_tracer_failure


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EDQUOT, errno.EIO])
def test_real_trace_writer_retains_first_storage_errno(tmp_path, monkeypatch, code):
    def fail(*_args, **_kwargs):
        raise OSError(code, "write failed")

    monkeypatch.setattr(trace, "write_locked_text", fail)
    tracer = trace.Tracer("failure", str(tmp_path))
    try:
        tracer.log_step("tool_exec", {"result": "done"})
        tracer.flush()
        tracer.log_step("tool_exec", {"result": "another step"})
        tracer.flush()
        assert tracer.write_error_errno == code
        assert tracer.dropped_steps == 2
        failure = programmatic._close_tracer(tracer)
        assert isinstance(failure, OSError)
        assert failure.errno == code
    finally:
        tracer.close()

def test_agent_evidence_failure_preserves_errno_as_cause():
    tracer = SimpleNamespace(
        flush=lambda: None, write_error="write failed", dropped_steps=1, write_error_errno=errno.ENOSPC,
    )
    result = SimpleNamespace(cleanup_quiesced=True, persistence_errors=())
    with pytest.raises(programmatic.ProgrammaticLifecycleError) as error:
        programmatic._require_agent_evidence(result, None, None, tracer)
    assert error.value.__cause__.errno == errno.ENOSPC

@pytest.mark.asyncio
async def test_public_workflow_keeps_actual_trace_storage_failure(tmp_path, monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, "storage full")

    monkeypatch.setattr(trace, "write_locked_text", fail)

    async def flow(context, _arguments):
        await context.log("a real workflow event")
        return {"status": "done"}

    with pytest.raises(RunError) as caught:
        await OpenCollab(tmp_path, model="unused", provider="openai").workflow(
            flow, {}, artifacts=tmp_path / "artifacts", trace=True,
        )
    error = caught.value
    seen, found = set(), False
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        found = found or getattr(error, "errno", None) == errno.ENOSPC
        error = error.__cause__ or error.__context__
    assert found

def test_unknown_tracer_errno_stays_unknown():
    assert _sticky_tracer_failure("storage failed", 1).errno is None
    assert _sticky_tracer_failure("storage failed", 1, error_number=True).errno is None
