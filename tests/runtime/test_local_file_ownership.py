"""Cancellation retains local I/O ownership and native edit ordering."""

from __future__ import annotations

import asyncio
import os
import threading

import pytest

from opencollab.adapters import _env_local as local_module
from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.tools.fs import FileWriteTool
from opencollab.application import tool_execution_runtime
from opencollab.application.event_bus import EventBus
from opencollab.application.tool_execution import ToolExecutionUseCase, ToolRuntime
from opencollab.domain.agent import Agent
from opencollab.domain.session import SessionState
from tests.support.asyncio_test_support import assert_cancel_note


async def test_cancelled_native_write_keeps_lock_until_atomic_write_finishes(tmp_path, monkeypatch):
    target = tmp_path / "value.txt"
    target.write_text("initial\n")
    started, release, later_started = threading.Event(), threading.Event(), threading.Event()
    original_write = local_module.write_regular_bytes_atomic

    def controlled_write(*args, **kwargs):
        if args[3] == b"earlier\n":
            started.set()
            assert release.wait(5)
        else:
            later_started.set()
        return original_write(*args, **kwargs)

    monkeypatch.setattr(local_module, "write_regular_bytes_atomic", controlled_write)
    env = LocalEnvironment(str(tmp_path))
    runtime = ToolRuntime(environment=env, safety_policy=None, permission_policy=None)
    tool = FileWriteTool()
    earlier = asyncio.create_task(tool.execute_with_runtime({
        "path": "value.txt", "mode": "create", "content": "earlier\n", "overwrite": True,
    }, runtime))
    later = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        earlier.cancel()
        await asyncio.sleep(0)
        earlier.cancel()
        later = asyncio.create_task(tool.execute_with_runtime({
            "path": "value.txt", "mode": "create", "content": "later\n", "overwrite": True,
        }, runtime))
        await asyncio.sleep(0.05)
        assert not earlier.done()
        assert not later.done()
        assert not later_started.is_set()
        assert target.read_text() == "initial\n"
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await earlier
        assert "Created/wrote" in await later
        assert target.read_text() == "later\n"
    finally:
        release.set()
        await asyncio.gather(earlier, *([later] if later is not None else []), return_exceptions=True)
        await env.cleanup()

@pytest.mark.parametrize("operation", ["read", "range", "temporary-write", "temporary-remove"])
async def test_repeated_file_cancellation_keeps_cleanup_owned(tmp_path, monkeypatch, operation):
    env = LocalEnvironment(str(tmp_path))
    workspace_fd = env._workspace_fd
    source = tmp_path / "value.txt"
    source.write_text("payload\n")
    temporary = None
    if operation == "temporary-remove":
        temporary = await env.write_temp_file("temporary", prefix="owned-", suffix=".txt")
    seam = {
        "read": "read_regular_bytes",
        "range": "read_regular_text_range_at",
        "temporary-write": "create_regular_bytes_atomic_at",
        "temporary-remove": "unlink_regular_file_durable_at",
    }[operation]
    original = getattr(local_module, seam)
    started, release, completed = threading.Event(), threading.Event(), threading.Event()

    def controlled_io(*args, **kwargs):
        started.set()
        assert release.wait(5)
        result = original(*args, **kwargs)
        completed.set()
        return result

    monkeypatch.setattr(local_module, seam, controlled_io)
    if operation == "read":
        pending = env.read_file("value.txt")
    elif operation == "range":
        pending = env.read_text_range("value.txt", offset=1, limit=1, max_chars=20)
    elif operation == "temporary-write":
        pending = env.write_temp_file("temporary", prefix="owned-", suffix=".txt")
    else:
        pending = env.remove_file(temporary)
    owner = asyncio.create_task(pending)
    cleanup = None
    try:
        assert await asyncio.to_thread(started.wait, 2)
        owner.cancel()
        await asyncio.sleep(0)
        owner.cancel()
        cleanup = asyncio.create_task(env.cleanup())
        await asyncio.sleep(0.02)
        assert not owner.done()
        assert not cleanup.done()
        os.fstat(workspace_fd)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert completed.is_set()
        await cleanup
        assert env._workspace_fd is None
        assert not list(tmp_path.glob("owned-*.txt"))
        assert source.read_text() == "payload\n"
        with pytest.raises(OSError):
            os.fstat(workspace_fd)
    finally:
        release.set()
        await asyncio.gather(owner, *([cleanup] if cleanup is not None else []), return_exceptions=True)
        await env.cleanup()

async def test_cancelled_file_io_retains_underlying_failure(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def failing_write(*_args, **_kwargs):
        started.set()
        assert release.wait(5)
        raise OSError("controlled atomic write failed")

    monkeypatch.setattr(local_module, "write_regular_bytes_atomic", failing_write)
    env = LocalEnvironment(str(tmp_path))
    failures = []

    async def write():
        try:
            await env.write_file("value.txt", "payload")
        except asyncio.CancelledError as exc:
            # Python 3.10 replaces cancellation metadata at the Task boundary.
            failures.append(exc)
            raise

    writer = asyncio.create_task(write())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        writer.cancel()
        await asyncio.sleep(0)
        assert not writer.done()
        release.set()
        with pytest.raises(asyncio.CancelledError) as captured:
            await writer
        assert_cancel_note(captured.value, "controlled atomic write failed")
        assert len(failures) == 1
        assert any("controlled atomic write failed" in note for note in getattr(failures[0], "__notes__", ()))
        assert not (tmp_path / "value.txt").exists()
    finally:
        release.set()
        await asyncio.gather(writer, return_exceptions=True)
        await env.cleanup()

async def test_native_file_timeout_retains_owned_io_after_bounded_abort(tmp_path, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original_write = local_module.write_regular_bytes_atomic

    def controlled_write(*args, **kwargs):
        started.set()
        assert release.wait(5)
        return original_write(*args, **kwargs)

    monkeypatch.setattr(local_module, "write_regular_bytes_atomic", controlled_write)
    monkeypatch.setattr(tool_execution_runtime, "TOOL_EXECUTION_TIMEOUT_GRACE", 0.001)
    env = LocalEnvironment(str(tmp_path))
    workspace_fd = env._workspace_fd
    tool = FileWriteTool()
    tool.default_timeout = 0.02
    executor = ToolExecutionUseCase(
        agent=Agent(name="writer", system_prompt="Write the file.", tools=[tool]),
        state=SessionState(messages=[]), environment=env, event_publisher=EventBus(),
        cancellation_cleanup_timeout=0.01, cancellation_force_timeout=0.01, environment_abort_timeout=0.01,
    )
    execution = asyncio.create_task(executor.execute_tool(tool, {
        "path": "value.txt", "mode": "create", "content": "settled write\n", "overwrite": True,
    }))
    try:
        assert await asyncio.to_thread(started.wait, 2)
        done, _pending = await asyncio.wait({execution}, timeout=0.5)
        assert done, "outer tool timeout waited for an unbounded file worker"
        output, _latency = execution.result()
        assert "Tool cancellation cleanup failed" in output
        assert "environment abort did not quiesce" in output
        assert env.revoked
        assert executor.pending_cleanup_tasks
        os.fstat(workspace_fd)
        assert not (tmp_path / "value.txt").exists()
        release.set()
        await asyncio.gather(*executor.pending_cleanup_tasks, return_exceptions=True)
        assert (tmp_path / "value.txt").read_text() == "settled write\n"
        assert executor.pending_cleanup_tasks == ()
        assert env._workspace_fd is None
    finally:
        release.set()
        await asyncio.gather(execution, *executor.pending_cleanup_tasks, return_exceptions=True)
        await env.cleanup()
