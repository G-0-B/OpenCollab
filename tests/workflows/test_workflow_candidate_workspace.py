from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from opencollab.adapters._env_base import ExecResult
from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import (
    CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS,
    EnvCandidateWorkspace,
    _raw_diff_at,
)
from opencollab.application.workflow import WorkflowContext
from opencollab.application.workflow_candidates import (
    CandidateWorkspaceTrackingError,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    (repo / "source.py").write_text("value = 1\n")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-m", "initial")
    return repo

class _RecordingDiffEnvironment:
    def __init__(self) -> None:
        self.timeouts: list[float] = []
        self.results = [
            ExecResult(0, "", ""),
            ExecResult(0, "", ""),
        ]

    async def exec_cmd(self, _command: str, timeout: float = 120.0) -> ExecResult:
        self.timeouts.append(timeout)
        return self.results.pop(0)

@pytest.mark.asyncio
async def test_candidate_source_diff_uses_pre_model_workspace_timeout() -> None:
    environment = _RecordingDiffEnvironment()

    assert await _raw_diff_at(environment, "/app") == ""
    assert environment.timeouts == [CANDIDATE_WORKSPACE_GIT_TIMEOUT_SECONDS] * 2

class _EditingSession:
    used_tokens = 7
    pending_cleanup_tasks: tuple[Any, ...] = ()

    def __init__(self, environment: Any, *, fail: bool = False) -> None:
        self.environment = environment
        self.fail = fail
        self.state = type("State", (), {"messages": []})()

    async def add_user_message(self, content: str) -> None:
        self.state.messages.append({"role": "user", "content": content})

    async def run_loop(self, _cancel_event: Any = None) -> str:
        await self.environment.write_file("source.py", "value = 2\n")
        if self.fail:
            raise RuntimeError("role failed after writing")
        return "done"

class _Factory:
    def __init__(self, fallback: Any, *, fail: bool = False, ignore_override: bool = False):
        self.fallback = fallback
        self.fail = fail
        self.ignore_override = ignore_override
        self.environments: list[Any] = []

    def build_workflow_session(self, **kwargs: Any) -> _EditingSession:
        environment = self.fallback if self.ignore_override else kwargs.get("env")
        if environment is None:
            environment = self.fallback
        self.environments.append(environment)
        return _EditingSession(environment, fail=self.fail)

@pytest.mark.asyncio
async def test_candidate_agent_writes_only_candidate_worktree(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    factory = _Factory(base)
    ctx = WorkflowContext(
        factory,
        budget_total=None,
        candidate_workspace=EnvCandidateWorkspace(base),
    )

    candidate = await ctx.candidate_agent("edit", label="candidate-a")

    assert "value = 2" in candidate.diff
    assert (repo / "source.py").read_text() == "value = 1\n"
    assert factory.environments[0].workspace != str(repo)

@pytest.mark.asyncio
async def test_candidate_agent_preserves_partial_diff_after_role_error(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    ctx = WorkflowContext(
        _Factory(base, fail=True),
        budget_total=None,
        candidate_workspace=EnvCandidateWorkspace(base),
    )

    candidate = await ctx.candidate_agent("edit then fail", label="candidate-a")

    assert candidate.output is None
    assert "value = 2" in candidate.diff
    assert (repo / "source.py").read_text() == "value = 1\n"

@pytest.mark.asyncio
async def test_candidate_workflow_binds_nested_agent_to_candidate(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    factory = _Factory(base)
    ctx = WorkflowContext(
        factory,
        budget_total=None,
        candidate_workspace=EnvCandidateWorkspace(base),
    )

    async def nested(child: Any, _args: dict[str, Any]) -> str:
        return await child.agent("nested edit")

    candidate = await ctx.candidate_workflow(
        nested,
        {},
        label="candidate-workflow",
        budget=None,
    )

    assert "value = 2" in candidate.diff
    assert (repo / "source.py").read_text() == "value = 1\n"
    assert factory.environments[0].workspace != str(repo)

@pytest.mark.asyncio
async def test_source_worktree_leak_is_preserved_and_reported(tmp_path: Path, monkeypatch) -> None:
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(base)
    leases = []
    acquire = workspace.acquire

    async def recorded_acquire(label):
        lease = await acquire(label)
        leases.append(lease)
        return lease

    monkeypatch.setattr(workspace, "acquire", recorded_acquire)
    ctx = WorkflowContext(
        _Factory(base, ignore_override=True),
        budget_total=None,
        candidate_workspace=workspace,
    )

    try:
        with pytest.raises(CandidateWorkspaceTrackingError, match="worktree preserved") as captured:
            await ctx.candidate_agent("leak", label="candidate-a")
        assert (repo / "source.py").read_text() == "value = 2\n"
        assert Path(leases[0].candidate_workspace).is_dir()
        assert leases[0].base_revision in str(captured.value)
        assert _git(repo, "show", f"{leases[0].base_revision}:source.py") == "value = 1\n"
    finally:
        for lease in leases:
            await lease.cleanup()

@pytest.mark.asyncio
async def test_base_environment_captures_relative_untracked_candidate_diff(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(base)
    lease = await workspace.acquire("candidate-a")
    try:
        await lease.environment.write_file("new_file.py", "created = True\n")
        diff = await lease.diff()
    finally:
        await lease.cleanup()

    assert "b/new_file.py" in diff
    assert lease.candidate_workspace not in diff

@pytest.mark.parametrize("nested_wrapper", [False, True])
@pytest.mark.asyncio
async def test_candidate_isolated_role_reads_current_candidate_and_owns_cleanup(tmp_path, monkeypatch, nested_wrapper):
    from opencollab.bootstrap import _workflow_runtime_session as runtime
    from tests.support.workflow_context_test_support import FakeSession

    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    built = []

    class ReadingSession(FakeSession):
        def __init__(self, environment):
            super().__init__(tokens=3)
            self.environment = environment
            self.closes = 0

        async def run_loop(self, cancel_event=None):
            if self.prompt == "edit candidate":
                await self.environment.write_file("source.py", "value = 2\n")
                await self.environment.write_file("untracked.txt", "candidate content\n")
                return "edited"
            value = await self.environment.read_file("source.py")
            extra = await self.environment.read_file("untracked.txt")
            await self.environment.write_file("source.py", "value = 9\n")
            return value + extra

        async def aclose(self):
            self.closes += 1

    def build(**kwargs):
        session = ReadingSession(kwargs["env"])
        built.append(session)
        return session

    monkeypatch.setattr(runtime, "build_session", build)
    factory = runtime.WorkflowSessionFactory(
        model="gpt-fake", provider="openai", api_key=None, base_url=None,
        workspace=str(repo), env=base,
    )
    if nested_wrapper:
        from opencollab.application.workflow_candidates import _CandidateWorkflowSessionFactory

        factory = _CandidateWorkflowSessionFactory(factory, base)
    parent = WorkflowContext(factory, budget_total=100, candidate_workspace=EnvCandidateWorkspace(base))

    async def nested(child, _args):
        await child.agent("edit candidate")
        reply = await child.agent("inspect candidate", isolation=True, label="scout")
        assert await built[0].environment.read_file("source.py") == "value = 2\n"
        return reply

    candidate = await parent.candidate_workflow(nested, {}, label="candidate")
    assert candidate.output == "value = 2\ncandidate content\n"
    assert parent.tokens_spent() == 6
    assert len(parent.sessions) == 2
    assert built[1].environment.workspace != built[0].environment.workspace
    assert built[1].closes == 1
    assert built[0].closes == 0
    assert base.revoked is False
    assert (repo / "source.py").read_text() == "value = 1\n"
    assert len(_git(repo, "worktree", "list", "--porcelain").split("worktree ")) == 2
    await factory.release_isolated_envs()
    assert base.revoked is False

@pytest.mark.parametrize("source_edit", [False, True])
@pytest.mark.asyncio
async def test_candidate_source_changed_honors_excluded_paths(tmp_path, source_edit):
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    factory = _Factory(base)
    parent = WorkflowContext(factory, candidate_workspace=EnvCandidateWorkspace(base))

    async def nested(child, _args):
        environment = child._factory._environment
        await environment.write_file("generated_test.py", "test_case = 1\n")
        if source_edit:
            await environment.write_file("source.py", "value = 2\n")
        return {"all": await child.tree_changed(), "source": await child.source_changed(["generated_test.py"])}

    candidate = await parent.candidate_workflow(nested, {}, label="candidate")
    assert candidate.output == {"all": True, "source": source_edit}
    assert "generated_test.py" in candidate.diff

@pytest.mark.parametrize("failure_kind", ["close", "cleanup"])
@pytest.mark.asyncio
async def test_candidate_isolation_cleanup_failure_keeps_budget_and_other_resources(
    tmp_path, monkeypatch, failure_kind,
):
    from opencollab.bootstrap import _workflow_runtime_session as runtime
    from tests.support.workflow_context_test_support import FakeSession

    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    built = []
    fault = {"lease": None, "enabled": True}

    class ClosingSession(FakeSession):
        def __init__(self, environment):
            super().__init__(tokens=3)
            self.environment = environment
            self.closes = 0

        async def aclose(self):
            self.closes += 1
            if failure_kind == "close" and self.prompt == "first":
                raise OSError("isolated close failed")

    def build(**kwargs):
        session = ClosingSession(kwargs["env"])
        built.append(session)
        return session

    monkeypatch.setattr(runtime, "build_session", build)
    factory = runtime.WorkflowSessionFactory(
        model="gpt-fake", provider="openai", api_key=None, base_url=None, workspace=str(repo), env=base,
    )
    parent = WorkflowContext(factory, budget_total=100, candidate_workspace=EnvCandidateWorkspace(base))
    source_owner = []

    async def nested(child, _args):
        await child.agent("first", isolation=True)
        await child.agent("second", isolation=True)
        owner, lease = factory._candidate_isolation_leases[0]
        source_owner.append(owner)
        if failure_kind == "cleanup":
            fault["lease"] = lease
            real_cleanup = type(lease).cleanup

            async def cleanup(current):
                if current is fault["lease"] and fault["enabled"]:
                    raise OSError("isolated worktree cleanup failed")
                await real_cleanup(current)

            monkeypatch.setattr(type(lease), "cleanup", cleanup)
        return "done"

    candidate = await parent.candidate_workflow(nested, {}, label="candidate")
    assert candidate.output == "done"
    assert candidate.lifecycle_errors
    assert parent.tokens_spent() == 6
    assert parent.tokens_remaining() == 94
    assert parent.budget._leases == []
    assert [session.closes for session in built] == [1, 1]
    assert built[1].environment.revoked is True
    assert base.revoked is False
    if failure_kind == "cleanup":
        assert len(factory._candidate_isolation_leases) == 1
        assert source_owner[0].revoked is False
        fault["enabled"] = False
        await factory.release_isolated_envs()
        await source_owner[0].cleanup()
        _git(repo, "worktree", "remove", "--force", source_owner[0].workspace)
    else:
        assert factory._candidate_isolation_leases == []
    assert len(_git(repo, "worktree", "list", "--porcelain").split("worktree ")) == 2

@pytest.mark.parametrize("nested_wrapper", [False, True])
@pytest.mark.asyncio
async def test_candidate_isolation_rejects_legacy_factory_before_allocating(tmp_path, nested_wrapper):
    from opencollab.application.workflow_candidates import _CandidateWorkflowSessionFactory
    from tests.support.workflow_context_test_support import FakeFactory, FakeSession

    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    legacy = FakeFactory([FakeSession(tokens=7)])
    factory = _CandidateWorkflowSessionFactory(legacy, base) if nested_wrapper else legacy
    parent = WorkflowContext(factory, budget_total=100, candidate_workspace=EnvCandidateWorkspace(base))

    async def nested(child, _args):
        return await child.agent("legacy isolation", isolation=True, label="scout")

    candidate = await parent.candidate_workflow(nested, {}, label="candidate")
    assert candidate.output is None
    assert parent.tokens_spent() == 0
    assert parent.agent_failures[0]["exception_type"] == "TypeError"
    assert legacy.handed_out == []
    assert base.revoked is False

@pytest.mark.asyncio
async def test_candidate_capture_includes_committed_and_uncommitted_edits(tmp_path):
    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(base)
    lease = await workspace.acquire("candidate")
    try:
        candidate_repo = Path(lease.candidate_workspace)
        await lease.environment.write_file("source.py", "value = 2\n")
        _git(candidate_repo, "add", "source.py")
        _git(candidate_repo, "-c", "commit.gpgsign=false", "commit", "-m", "candidate change")
        await lease.environment.write_file("source.py", "value = 3\n")
        await lease.environment.write_file("new_file.py", "new_value = 4\n")
        diff = await lease.diff()
        assert "-value = 1" in diff
        assert "+value = 3" in diff
        assert "+new_value = 4" in diff
        await workspace.adopt(diff)
        assert (repo / "source.py").read_text() == "value = 3\n"
        assert (repo / "new_file.py").read_text() == "new_value = 4\n"
    finally:
        await lease.cleanup()

@pytest.mark.parametrize("name", [
    "protected.txt", "protected file.txt", "protected_\u6570\u636e.txt", "protected\tfile.txt",
])
@pytest.mark.asyncio
async def test_candidate_adoption_preserves_git_quoted_file_paths(tmp_path, name):
    repo = _repository(tmp_path)
    _git(repo, "config", "core.quotePath", "true")
    protected = repo / name
    protected.write_text("protected = True\n")
    _git(repo, "add", "--", name)
    _git(repo, "commit", "-m", "protected file")
    workspace = EnvCandidateWorkspace(LocalEnvironment(str(repo)))
    lease = await workspace.acquire("candidate")
    try:
        await lease.environment.write_file(name, "protected = False\n")
        patch = await lease.diff()
        with pytest.raises(ValueError, match="preserved"):
            await workspace.adopt(patch, preserve_paths=[name])
        assert protected.read_text() == "protected = True\n"
    finally:
        await lease.cleanup()

@pytest.mark.asyncio
async def test_candidate_cleanup_preserves_cancellation_without_add_note(tmp_path, monkeypatch):
    import asyncio

    from opencollab.bootstrap import _workflow_runtime_session as runtime
    from tests.support.workflow_context_test_support import FakeSession

    class LegacyCancelledError(asyncio.CancelledError):
        add_note = None

    class ClosingSession(FakeSession):
        async def aclose(self):
            raise OSError("isolated close failed")

    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))
    monkeypatch.setattr(runtime, "build_session", lambda **kwargs: ClosingSession(tokens=7))
    factory = runtime.WorkflowSessionFactory(
        model="gpt-fake", provider="openai", api_key=None, base_url=None, workspace=str(repo), env=base,
    )
    parent = WorkflowContext(factory, budget_total=100, candidate_workspace=EnvCandidateWorkspace(base))
    original = LegacyCancelledError("original cancellation")

    async def nested(child, _args):
        await child.agent("finished role", isolation=True)
        raise original

    with pytest.raises(LegacyCancelledError) as captured:
        await parent.candidate_workflow(nested, {}, label="candidate")
    assert captured.value is original
    assert any("isolated close failed" in note for note in original.__notes__)
    assert parent.tokens_spent() == 7
    assert parent.tokens_remaining() == 93
    assert parent.budget._leases == []

@pytest.mark.asyncio
async def test_candidate_rejects_old_factory_without_releasing_other_owner(tmp_path, monkeypatch):
    from opencollab.bootstrap import _workflow_runtime_session as runtime
    from tests.support.workflow_context_test_support import FakeSession

    repo = _repository(tmp_path)
    base = LocalEnvironment(str(repo))

    class LegacyFactory(runtime.WorkflowSessionFactory):
        allocations = 0
        releases = 0

        async def acquire_isolated_env(self, *, label=None):
            self.allocations += 1
            return await super().acquire_isolated_env(label=label)

        async def release_isolated_envs(self):
            self.releases += 1
            await super().release_isolated_envs()

    class ReadingSession(FakeSession):
        def __init__(self, environment):
            super().__init__(tokens=3)
            self.environment = environment

        async def run_loop(self, cancel_event=None):
            return await self.environment.read_file("source.py")

        async def aclose(self):
            pass

    monkeypatch.setattr(runtime, "build_session", lambda **kwargs: ReadingSession(kwargs["env"]))
    factory = LegacyFactory(
        model="gpt-fake", provider="openai", api_key=None, base_url=None, workspace=str(repo), env=base,
    )
    other = await factory.acquire_isolated_env(label="other-owner")
    try:
        parent = WorkflowContext(factory, candidate_workspace=EnvCandidateWorkspace(base))

        async def nested(child, _args):
            return await child.agent("inspect", isolation=True)

        candidate = await parent.candidate_workflow(nested, {}, label="candidate")
        assert candidate.output is None
        assert parent.agent_failures[0]["exception_type"] == "TypeError"
        assert factory.allocations == 1
        assert factory.releases == 0
        assert other.revoked is False
        assert await other.read_file("source.py") == "value = 1\n"
        ordinary = WorkflowContext(factory)
        assert await ordinary.agent("ordinary legacy role", isolation=True) == "value = 1\n"
        assert ordinary.agent_failures == ()
        assert factory.allocations == 2
    finally:
        await factory.release_isolated_envs()
    assert base.revoked is False

@pytest.mark.parametrize("names", [
    ("source.txt", "target.txt"),
    ("source file.txt", "target file.txt"),
    ("source_\u6570\u636e.txt", "target_\u6570\u636e.txt"),
])
@pytest.mark.parametrize("preserved_endpoint", ["source", "target", None])
@pytest.mark.asyncio
async def test_candidate_rename_preserves_both_endpoints(tmp_path, names, preserved_endpoint):
    source_name, target_name = names
    repo = _repository(tmp_path)
    (repo / source_name).write_text("kept content\n")
    _git(repo, "add", "--", source_name)
    _git(repo, "commit", "-m", "rename source")
    workspace = EnvCandidateWorkspace(LocalEnvironment(str(repo)))
    lease = await workspace.acquire("candidate")
    try:
        candidate_repo = Path(lease.candidate_workspace)
        _git(candidate_repo, "mv", "--", source_name, target_name)
        patch = await lease.diff()
        assert "rename from" in patch
        if preserved_endpoint is None:
            await workspace.adopt(patch)
            assert not (repo / source_name).exists()
            assert (repo / target_name).read_text() == "kept content\n"
        else:
            preserved = source_name if preserved_endpoint == "source" else target_name
            with pytest.raises(ValueError, match="preserved"):
                await workspace.adopt(patch, preserve_paths=[preserved])
            assert (repo / source_name).read_text() == "kept content\n"
            assert not (repo / target_name).exists()
    finally:
        await lease.cleanup()

@pytest.mark.asyncio
async def test_dirty_source_enters_candidate_and_only_candidate_increment_is_adopted(tmp_path):
    repo = _repository(tmp_path)
    (repo / "source.py").write_text("value = 11\n")
    _git(repo, "add", "source.py")
    (repo / "source.py").write_text("value = 12\n")
    (repo / "draft.txt").write_text("user draft\n")
    source_index = _git(repo, "diff", "--cached", "--binary")
    workspace = EnvCandidateWorkspace(LocalEnvironment(str(repo)))
    lease = await workspace.acquire("candidate")
    try:
        assert await lease.environment.read_file("source.py") == "value = 12\n"
        assert await lease.environment.read_file("draft.txt") == "user draft\n"
        assert await lease.diff() == ""
        await lease.environment.write_file("source.py", "value = 13\n")
        await lease.environment.write_file("draft.txt", "user draft\ncandidate addition\n")
        candidate_repo = Path(lease.candidate_workspace)
        _git(candidate_repo, "add", "source.py", "draft.txt")
        _git(candidate_repo, "-c", "commit.gpgsign=false", "commit", "-m", "candidate edit")
        await lease.environment.write_file("new.txt", "candidate file\n")
        patch = await lease.diff()
        assert "-value = 12" in patch
        assert "+value = 13" in patch
        assert "-value = 1\n" not in patch
        (repo / "notes.txt").write_text("later user draft\n")

        await workspace.adopt(patch)

        assert (repo / "source.py").read_text() == "value = 13\n"
        assert (repo / "draft.txt").read_text() == "user draft\ncandidate addition\n"
        assert (repo / "notes.txt").read_text() == "later user draft\n"
        assert (repo / "new.txt").read_text() == "candidate file\n"
        assert _git(repo, "diff", "--cached", "--binary") == source_index
    finally:
        await lease.cleanup()

@pytest.mark.asyncio
async def test_candidate_adoption_keeps_unrelated_dirty_source_and_preserved_paths(tmp_path):
    repo = _repository(tmp_path)
    (repo / "notes.txt").write_text("committed note\n")
    _git(repo, "add", "notes.txt")
    _git(repo, "commit", "-m", "notes")
    (repo / "notes.txt").write_text("user note\n")
    (repo / "draft.txt").write_text("private draft\n")
    workspace = EnvCandidateWorkspace(LocalEnvironment(str(repo)))
    lease = await workspace.acquire("candidate")
    try:
        await lease.environment.write_file("source.py", "value = 2\n")
        await workspace.adopt(await lease.diff(), preserve_paths=["notes.txt", "draft.txt"])
        assert (repo / "source.py").read_text() == "value = 2\n"
        assert (repo / "notes.txt").read_text() == "user note\n"
        assert (repo / "draft.txt").read_text() == "private draft\n"
    finally:
        await lease.cleanup()

@pytest.mark.asyncio
async def test_candidate_adoption_conflict_preserves_all_source_files_and_index(tmp_path):
    repo = _repository(tmp_path)
    workspace = EnvCandidateWorkspace(LocalEnvironment(str(repo)))
    lease = await workspace.acquire("candidate")
    try:
        await lease.environment.write_file("source.py", "value = 2\n")
        await lease.environment.write_file("new.txt", "candidate new file\n")
        patch = await lease.diff()
        (repo / "source.py").write_text("value = 99\n")
        _git(repo, "add", "source.py")
        (repo / "draft.txt").write_text("keep my draft\n")
        before = _git(repo, "status", "--porcelain")
        with pytest.raises(RuntimeError, match="candidate adoption failed"):
            await workspace.adopt(patch)
        assert (repo / "source.py").read_text() == "value = 99\n"
        assert (repo / "draft.txt").read_text() == "keep my draft\n"
        assert not (repo / "new.txt").exists()
        assert _git(repo, "status", "--porcelain") == before
    finally:
        await lease.cleanup()

@pytest.mark.asyncio
async def test_candidate_adoption_preserves_edit_arriving_during_apply(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    workspace = EnvCandidateWorkspace(LocalEnvironment(str(repo)))
    lease = await workspace.acquire("candidate")
    try:
        await lease.environment.write_file("source.py", "value = 2\n")
        patch = await lease.diff()
        execute = workspace._environment.exec_cmd

        async def edited_before_apply(command, **kwargs):
            if " apply --binary " in command:
                (repo / "new-user-note.txt").write_text("concurrent note\n")
            return await execute(command, **kwargs)

        monkeypatch.setattr(workspace._environment, "exec_cmd", edited_before_apply)
        await workspace.adopt(patch)
        assert (repo / "source.py").read_text() == "value = 2\n"
        assert (repo / "new-user-note.txt").read_text() == "concurrent note\n"
    finally:
        await lease.cleanup()

@pytest.mark.asyncio
async def test_candidate_workflow_source_drift_preserves_current_source_and_candidate(tmp_path, monkeypatch):
    repo = _repository(tmp_path)
    (repo / "draft.txt").write_text("initial draft\n")
    workspace = EnvCandidateWorkspace(LocalEnvironment(str(repo)))
    leases = []
    acquire = workspace.acquire

    async def recorded_acquire(label):
        lease = await acquire(label)
        leases.append(lease)
        return lease

    monkeypatch.setattr(workspace, "acquire", recorded_acquire)
    context = WorkflowContext(_Factory(workspace._environment), candidate_workspace=workspace)

    async def workflow(child, _args):
        await leases[0].environment.write_file("source.py", "value = 2\n")
        (repo / "draft.txt").write_text("concurrent draft\n")
        return "done"

    try:
        with pytest.raises(CandidateWorkspaceTrackingError, match="worktree preserved") as captured:
            await context.candidate_workflow(workflow, {}, label="candidate")
        assert (repo / "draft.txt").read_text() == "concurrent draft\n"
        assert (repo / "source.py").read_text() == "value = 1\n"
        assert await leases[0].environment.read_file("source.py") == "value = 2\n"
        assert leases[0].base_revision in str(captured.value)
        assert _git(repo, "show", f"{leases[0].base_revision}:draft.txt") == "initial draft\n"
        candidate_repo = Path(leases[0].candidate_workspace)
        assert _git(candidate_repo, "show", "refs/worktree/opencollab-source:draft.txt") == "initial draft\n"
        assert not (repo / ".git" / "refs" / "worktree" / "opencollab-source").exists()
    finally:
        for lease in leases:
            await lease.cleanup()

@pytest.mark.asyncio
async def test_dirty_candidate_capture_uses_contents_after_index_reset_and_commit(tmp_path):
    repo = _repository(tmp_path)
    (repo / "draft.txt").write_text("initial draft\n")
    workspace = EnvCandidateWorkspace(LocalEnvironment(str(repo)))
    lease = await workspace.acquire("candidate")
    try:
        candidate_repo = Path(lease.candidate_workspace)
        _git(candidate_repo, "reset", "--mixed", "HEAD")
        assert await lease.diff() == ""
        await lease.environment.write_file("draft.txt", "initial draft\ncandidate addition\n")
        await lease.environment.write_file("source.py", "value = 2\n")
        _git(candidate_repo, "add", "source.py")
        _git(candidate_repo, "-c", "commit.gpgsign=false", "commit", "-m", "candidate edit")
        index_before = _git(candidate_repo, "ls-files", "--stage")
        patch = await lease.diff()

        assert patch.count("diff --git a/draft.txt b/draft.txt") == 1
        assert "+candidate addition" in patch
        assert _git(candidate_repo, "ls-files", "--stage") == index_before
        await workspace.adopt(patch)
        assert (repo / "draft.txt").read_text() == "initial draft\ncandidate addition\n"
        assert (repo / "source.py").read_text() == "value = 2\n"
    finally:
        await lease.cleanup()
