"""The runtime owns its default source environment through final cleanup."""

import asyncio

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.bootstrap import _workflow_runtime_session as wiring
from opencollab.bootstrap.workflow_runtime import run_workflow


@pytest.fixture
def source_environments(monkeypatch):
    environments = []

    class RecordedEnvironment(LocalEnvironment):
        def __init__(self, workspace="."):
            super().__init__(workspace)
            self.cleanup_calls = 0
            environments.append(self)

        async def cleanup(self):
            self.cleanup_calls += 1
            await super().cleanup()

    monkeypatch.setattr(wiring, "LocalEnvironment", RecordedEnvironment)
    return environments

@pytest.mark.parametrize("ending", ["success", "error", "cancel"])
async def test_owned_source_closes_after_workflow_exit(tmp_path, source_environments, ending):
    started = asyncio.Event()

    async def workflow(ctx, args):
        started.set()
        if ending == "error":
            raise ValueError("workflow failed")
        if ending == "cancel":
            await asyncio.Event().wait()
        return "done"

    task = asyncio.create_task(run_workflow(
        workflow, {}, cfg={"model": "test-model", "provider": "openai"},
        workspace=str(tmp_path), trace=False,
    ))
    try:
        await asyncio.wait_for(started.wait(), 10)
        if ending == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif ending == "error":
            with pytest.raises(ValueError, match="workflow failed"):
                await task
        else:
            assert await task == "done"
        assert len(source_environments) == 1
        assert source_environments[0]._workspace_fd is None
        assert source_environments[0].cleanup_calls == 1
    finally:
        for environment in source_environments:
            await environment.cleanup()

async def test_repeated_default_workflows_release_every_source(tmp_path, source_environments):
    async def workflow(ctx, args):
        return "done"

    try:
        for _ in range(24):
            assert await run_workflow(
                workflow, {}, cfg={"model": "test-model", "provider": "openai"},
                workspace=str(tmp_path), trace=False,
            ) == "done"
        assert len(source_environments) == 24
        assert all(environment._workspace_fd is None for environment in source_environments)
    finally:
        for environment in source_environments:
            await environment.cleanup()

async def test_supplied_source_environment_remains_owned_by_caller(tmp_path, source_environments):
    environment = LocalEnvironment(str(tmp_path))

    async def workflow(ctx, args):
        return "done"

    try:
        assert await run_workflow(
            workflow, {}, cfg={"model": "test-model", "provider": "openai"},
            env=environment, trace=False,
        ) == "done"
        assert source_environments == []
        assert environment._workspace_fd is not None
        await environment.write_file("still-owned.txt", "usable")
        assert await environment.read_file("still-owned.txt") == "usable"
    finally:
        await environment.cleanup()

@pytest.mark.parametrize("kwargs", [{"max_concurrency": 0}, {"task_concurrency": 0}])
async def test_invalid_context_configuration_opens_no_source(tmp_path, source_environments, kwargs):
    try:
        with pytest.raises(ValueError):
            wiring.build_workflow_context(
                cfg={"model": "test-model", "provider": "openai"}, workspace=str(tmp_path), **kwargs,
            )
        assert source_environments == []
    finally:
        for environment in source_environments:
            await environment.cleanup()

async def test_source_cleanup_is_idempotent(tmp_path, source_environments):
    ctx = wiring.build_workflow_context(
        cfg={"model": "test-model", "provider": "openai"}, workspace=str(tmp_path),
    )
    try:
        await ctx.release_isolated_workspaces()
        await ctx.release_isolated_workspaces()
        assert source_environments[0]._workspace_fd is None
        assert source_environments[0].cleanup_calls == 1
    finally:
        for environment in source_environments:
            await environment.cleanup()
