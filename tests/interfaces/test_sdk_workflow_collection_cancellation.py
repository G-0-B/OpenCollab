"""A cancelled collection settles its workers before a workflow can recover."""

from __future__ import annotations

import asyncio
import json

import pytest

from opencollab import OpenCollab


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["parallel", "pipeline"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("cancellation", ["upstream", "recovered", "caller"])
async def test_cancelled_collection_stops_owned_writers_before_return(
    tmp_path, kind, nested, cancellation,
):
    candidate = tmp_path / "candidate.txt"
    candidate.write_text("initial candidate\n")
    queued_marker = tmp_path / "queued.txt"
    upstream = asyncio.get_running_loop().create_future()
    writer_started = asyncio.Event()
    release_writer = asyncio.Event()
    writer_finished = asyncio.Event()
    workers = set()

    async def wait_for_upstream():
        workers.add(asyncio.current_task())
        await writer_started.wait()
        return await upstream

    async def write_candidate():
        workers.add(asyncio.current_task())
        writer_started.set()
        try:
            await release_writer.wait()
            candidate.write_text("outdated candidate\n")
            return "written"
        finally:
            writer_finished.set()

    async def queued_write():
        workers.add(asyncio.current_task())
        queued_marker.write_text("queued work started\n")
        return "queued"

    async def flow(ctx, _inputs):
        async def collection():
            units = [wait_for_upstream, write_candidate, queued_write]
            if kind == "parallel":
                return await ctx.parallel(units)

            async def stage(_previous, operation, _index):
                return await operation()

            return await ctx.pipeline(units, stage)

        try:
            if nested:
                return await ctx.parallel([collection])
            return await collection()
        except asyncio.CancelledError:
            if cancellation == "recovered":
                candidate.write_text("fallback candidate\n")
                return {"recovered": True}
            raise

    artifacts = tmp_path / "artifacts"
    running = asyncio.create_task(OpenCollab(tmp_path, model="test-model").workflow(
        flow, concurrency=2, task_concurrency=2, timeout=30,
        cleanup_timeout=2, trace=False, artifacts=artifacts,
    ))
    try:
        await asyncio.wait_for(writer_started.wait(), timeout=10)
        if cancellation == "caller":
            running.cancel()
        else:
            upstream.cancel()
        if cancellation == "recovered":
            result = await asyncio.wait_for(asyncio.shield(running), timeout=10)
            assert result.status == "completed"
            assert result.output == {"recovered": True}
            assert result.tokens == 0
            assert result.metrics["execution_quiesced"] is True
            manifest = json.loads((artifacts / "workflow.json").read_text())
            assert manifest["status"] == "completed"
            assert manifest["evidence_complete"] is True
        else:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(running), timeout=10)

        finished_before_delivery = writer_finished.is_set()
        release_writer.set()
        await asyncio.wait_for(writer_finished.wait(), timeout=10)
        await asyncio.gather(*workers, return_exceptions=True)

        assert finished_before_delivery
        expected = "fallback candidate\n" if cancellation == "recovered" else "initial candidate\n"
        assert candidate.read_text() == expected
        assert not queued_marker.exists()
        assert all(task.done() for task in workers)
    finally:
        upstream.cancel()
        release_writer.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, *workers, return_exceptions=True)

@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["parallel", "pipeline"])
@pytest.mark.parametrize("nested", [False, True])
async def test_caller_cancellation_preserves_awaited_worker_cleanup(tmp_path, kind, nested):
    active_lease = tmp_path / "active-lease.txt"
    closed = tmp_path / "resource-closed.txt"
    active_lease.write_text("open\n")
    writer_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    blocked = asyncio.Event()
    workers = set()

    async def peer():
        workers.add(asyncio.current_task())
        await writer_started.wait()
        await blocked.wait()

    async def writer():
        workers.add(asyncio.current_task())
        writer_started.set()
        try:
            await blocked.wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
            active_lease.unlink()
            closed.write_text("closed\n")

    async def flow(ctx, _inputs):
        async def collection():
            if kind == "parallel":
                return await ctx.parallel([peer, writer])

            async def stage(_previous, operation, _index):
                return await operation()

            return await ctx.pipeline([peer, writer], stage)

        if nested:
            return await ctx.parallel([collection])
        return await collection()

    running = asyncio.create_task(OpenCollab(tmp_path, model="test-model").workflow(
        flow, concurrency=2, task_concurrency=2, timeout=30,
        cleanup_timeout=2, trace=False,
    ))
    try:
        await asyncio.wait_for(writer_started.wait(), timeout=10)
        running.cancel()
        await asyncio.wait_for(cleanup_started.wait(), timeout=10)
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(running), timeout=10)
        assert closed.read_text() == "closed\n"
        assert not active_lease.exists()
        assert all(task.done() for task in workers)
    finally:
        release_cleanup.set()
        if not running.done():
            running.cancel()
        await asyncio.gather(running, *workers, return_exceptions=True)
