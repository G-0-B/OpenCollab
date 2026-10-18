"""Candidates keep the source Git revision through execution and adoption."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from opencollab.application.workflow import WorkflowContext
from opencollab.application.workflow_candidates import CandidateWorkspaceTrackingError
from tests.workflows.test_workflow_candidate_workspace import _EditingSession, _Factory, _git, _repository


@pytest.mark.parametrize("mode", ["agent", "workflow"])
@pytest.mark.parametrize("when", ["during", "before-adoption"])
@pytest.mark.parametrize("change", ["different-head", "same-head", "dirty-note"])
async def test_candidate_tracks_source_revision_at_execution_and_adoption(tmp_path, monkeypatch, mode, when, change):
    repo = _repository(tmp_path)
    (repo / "flags.py").write_text("ENABLED = True\n")
    _git(repo, "add", "flags.py")
    _git(repo, "commit", "-m", "\u8bb0\u5f55\u5019\u9009\u4f9d\u8d56\u914d\u7f6e")
    original_branch = _git(repo, "branch", "--show-current").strip()
    initial_revision = _git(repo, "rev-parse", "HEAD").strip()
    _git(repo, "branch", "alias")
    _git(repo, "checkout", "-qb", "other")
    (repo / "flags.py").write_text("ENABLED = False\n")
    _git(repo, "commit", "-am", "\u4fee\u6539\u5019\u9009\u4f9d\u8d56\u914d\u7f6e")
    _git(repo, "checkout", "-q", original_branch)
    environment = LocalEnvironment(str(repo))
    backend = EnvCandidateWorkspace(environment)
    leases = []
    acquire = backend.acquire
    checked = []

    def change_source():
        if change == "dirty-note":
            (repo / "note.txt").write_text("user note\n")
        else:
            _git(repo, "checkout", "-q", "other" if change == "different-head" else "alias")

    async def record_lease(label):
        lease = await acquire(label)
        leases.append(lease)
        return lease

    class CheckedSession(_EditingSession):
        async def run_loop(self, cancel_event=None):
            output = await super().run_loop(cancel_event)
            command = (
                f"PYTHONDONTWRITEBYTECODE=1 {shlex.quote(sys.executable)} "
                "-c 'from flags import ENABLED; from source import value; assert ENABLED and value == 2'"
            )
            result = await self.environment.exec_cmd(command)
            checked.append(result.returncode)
            assert result.returncode == 0
            if when == "during":
                change_source()
            return output

    class CheckedFactory(_Factory):
        def build_workflow_session(self, **kwargs):
            return CheckedSession(kwargs["env"])

    monkeypatch.setattr(backend, "acquire", record_lease)
    context = WorkflowContext(CheckedFactory(environment), candidate_workspace=backend)

    async def nested(child, _args):
        return await child.agent("edit and check")

    async def generate():
        if mode == "agent":
            return await context.candidate_agent("edit and check", label="revision")
        return await context.candidate_workflow(nested, {}, label="revision")

    expected_error = change == "different-head" or (when == "during" and change == "dirty-note")
    try:
        if when == "during" and expected_error:
            with pytest.raises(CandidateWorkspaceTrackingError, match="worktree preserved"):
                await generate()
            assert Path(leases[0].candidate_workspace).exists()
            assert await leases[0].environment.read_file("source.py") == "value = 2\n"
        else:
            candidate = await generate()
            assert candidate.output == "done"
            if when == "before-adoption":
                change_source()
            if expected_error:
                with pytest.raises(CandidateWorkspaceTrackingError, match="before candidate adoption"):
                    await context.adopt_candidate(candidate)
                assert "+value = 2" in candidate.diff
                assert (repo / "source.py").read_text() == "value = 1\n"
            else:
                await context.adopt_candidate(candidate)
                assert (repo / "source.py").read_text() == "value = 2\n"
            assert candidate.source_revision == initial_revision
        assert checked == [0]
        if change == "different-head":
            assert (repo / "flags.py").read_text() == "ENABLED = False\n"
        if change == "dirty-note":
            assert (repo / "note.txt").read_text() == "user note\n"
    finally:
        for lease in leases:
            await lease.cleanup()
        await environment.cleanup()

@pytest.mark.parametrize("mode", ["agent", "workflow"])
async def test_candidate_backend_without_source_revision_remains_compatible(tmp_path, mode):
    repo = _repository(tmp_path)
    environment = LocalEnvironment(str(repo))
    backend = EnvCandidateWorkspace(environment)
    legacy = SimpleNamespace(acquire=backend.acquire, source_diff=backend.source_diff, adopt=backend.adopt)
    context = WorkflowContext(_Factory(environment), candidate_workspace=legacy)

    async def nested(child, _args):
        return await child.agent("edit")

    try:
        if mode == "agent":
            candidate = await context.candidate_agent("edit", label="legacy")
        else:
            candidate = await context.candidate_workflow(nested, {}, label="legacy")
        assert getattr(candidate, "source_revision", None) is None
        await context.adopt_candidate(candidate)
        assert (repo / "source.py").read_text() == "value = 2\n"
    finally:
        await environment.cleanup()
