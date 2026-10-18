"""Candidate tools retain the configured directory inside the Git worktree."""

from __future__ import annotations

from pathlib import Path

import pytest

from opencollab import OpenCollab
from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from opencollab.application.workflow import WorkflowContext
from tests.workflows.test_workflow_candidate_workspace import _Factory, _git, _repository


async def test_candidate_repository_path_retains_trailing_spaces(tmp_path):
    original = _repository(tmp_path)
    repo = original.with_name("repo ")
    original.rename(repo)
    environment = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(environment)
    lease = None
    try:
        lease = await workspace.acquire("space-suffixed-repository")
        await lease.environment.write_file("source.py", "value = 2\n")
        await workspace.adopt(await lease.diff())
        assert (repo / "source.py").read_text() == "value = 2\n"
    finally:
        if lease is not None:
            await lease.cleanup()
        await environment.cleanup()
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1

@pytest.mark.parametrize("prefix", ["package", "package/deep", "package with spaces"])
@pytest.mark.parametrize("tracked", [False, True])
async def test_subdirectory_candidate_inherits_edits_and_adopts_at_repository_root(tmp_path, prefix, tracked):
    repo = _repository(tmp_path)
    package = repo / prefix
    package.mkdir(parents=True)
    (package / "source.py").write_text("value = 10\n")
    (package / "binary.bin").write_bytes(b"source\x00\xff\n")
    if tracked:
        _git(repo, "add", ".")
        _git(repo, "commit", "-m", "\u8bb0\u5f55\u5b50\u76ee\u5f55\u6587\u4ef6")
    (repo / "source.py").write_text("value = 100\n")
    source_index = _git(repo, "ls-files", "--stage")
    environment = LocalEnvironment(str(package))
    workspace = EnvCandidateWorkspace(environment)
    lease = None
    try:
        lease = await workspace.acquire("package-edit")
        candidate_root = Path(lease.candidate_workspace)
        assert Path(lease.environment.workspace) == (candidate_root / prefix).resolve()
        assert await lease.environment.read_file("source.py") == "value = 10\n"
        assert (candidate_root / prefix / "binary.bin").read_bytes() == b"source\x00\xff\n"
        assert (candidate_root / "source.py").read_text() == "value = 100\n"
        assert await lease.diff() == ""
        await lease.environment.write_file("source.py", "value = 20\n")
        (candidate_root / prefix / "binary.bin").write_bytes(b"candidate\x00\xfe\n")
        patch = await lease.diff()
        assert "GIT binary patch" in patch
        await workspace.adopt(patch)
        assert (package / "source.py").read_text() == "value = 20\n"
        assert (package / "binary.bin").read_bytes() == b"candidate\x00\xfe\n"
        assert (repo / "source.py").read_text() == "value = 100\n"
        assert _git(repo, "ls-files", "--stage") == source_index
    finally:
        if lease is not None:
            await lease.cleanup()
        await environment.cleanup()
    assert not candidate_root.exists()
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1

@pytest.mark.parametrize("mode", ["agent", "workflow"])
async def test_candidate_session_uses_subdirectory_relative_paths(tmp_path, mode):
    repo = _repository(tmp_path)
    package = repo / "package"
    package.mkdir()
    (package / "source.py").write_text("value = 10\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "\u8bb0\u5f55\u5b50\u76ee\u5f55\u6587\u4ef6")
    environment = LocalEnvironment(str(package))
    context = WorkflowContext(_Factory(environment), candidate_workspace=EnvCandidateWorkspace(environment))

    async def nested(child, _args):
        assert Path(child.workspace_root).name == "package"
        return await child.agent("edit source.py")

    try:
        if mode == "agent":
            candidate = await context.candidate_agent("edit source.py", label="package")
        else:
            candidate = await context.candidate_workflow(nested, {}, label="package")
        assert candidate.output == "done"
        assert "+value = 2" in candidate.diff
        assert (package / "source.py").read_text() == "value = 10\n"
        await context.adopt_candidate(candidate)
        assert (package / "source.py").read_text() == "value = 2\n"
        assert (repo / "source.py").read_text() == "value = 1\n"
    finally:
        await environment.cleanup()

async def test_linked_subdirectory_candidate_preserves_paths_and_source_edits(tmp_path):
    main = _repository(tmp_path)
    package = main / "package"
    package.mkdir()
    (package / "source.py").write_text("value = 10\n")
    _git(main, "add", ".")
    _git(main, "commit", "-m", "\u8bb0\u5f55\u5b50\u76ee\u5f55\u6587\u4ef6")
    repo = tmp_path / "linked"
    _git(main, "worktree", "add", "--detach", str(repo))
    environment = LocalEnvironment(str(repo / "package"))
    workspace = EnvCandidateWorkspace(environment)
    lease = None
    try:
        lease = await workspace.acquire("linked-package")
        await lease.environment.write_file("source.py", "value = 20\n")
        patch = await lease.diff()
        with pytest.raises(ValueError, match="preserved path"):
            await workspace.adopt(patch, preserve_paths=["source.py"])
        (repo / "user-note.txt").write_text("new note\n")
        await workspace.adopt(patch)
        assert (repo / "package/source.py").read_text() == "value = 20\n"
        assert (repo / "user-note.txt").read_text() == "new note\n"
        assert (main / "package/source.py").read_text() == "value = 10\n"
    finally:
        if lease is not None:
            await lease.cleanup()
        await environment.cleanup()
        _git(main, "worktree", "remove", "--force", str(repo))

@pytest.mark.parametrize("prefix", ["", "package", "package/deep", "package with spaces"])
@pytest.mark.parametrize("preserved", ["tests", "./tests/", "tests/inner/..", "tests/check.py", "."])
async def test_public_subdirectory_candidate_preserves_workspace_paths(tmp_path, prefix, preserved):
    repo = _repository(tmp_path)
    workspace = repo / prefix
    (workspace / "tests").mkdir(parents=True)
    protected = workspace / "tests/check.py"
    protected.write_text("protected = True\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "record protected tests")

    async def edit_candidate(context, _inputs):
        Path(context.workspace_root, "tests/check.py").write_text("protected = False\n")
        return "edited"

    async def flow(context, _inputs):
        candidate = await context.candidate_workflow(edit_candidate, {}, label="edit-tests")
        with pytest.raises(ValueError, match="preserved path"):
            await context.adopt_candidate(candidate, preserve_paths=[preserved])
        return "protected"

    result = await OpenCollab(workspace).workflow(flow, trace=False)
    assert result.ok, (result.reason, result.error)
    assert result.output == "protected"
    assert protected.read_text() == "protected = True\n"
    assert _git(repo, "status", "--porcelain") == ""

@pytest.mark.parametrize("prefix", ["", "package"])
@pytest.mark.parametrize("preserved", ["", "../tests", "tests/../../escape", "/tests"])
async def test_subdirectory_candidate_rejects_invalid_preserved_paths(tmp_path, prefix, preserved):
    repo = _repository(tmp_path)
    source = repo / prefix
    source.mkdir(exist_ok=True)
    environment = LocalEnvironment(str(source))
    workspace = EnvCandidateWorkspace(environment)
    lease = await workspace.acquire("invalid-preserved-path")
    try:
        await lease.environment.write_file("source.py", "value = 2\n")
        with pytest.raises(ValueError, match="candidate preserve path"):
            await workspace.adopt(await lease.diff(), preserve_paths=[preserved])
    finally:
        await lease.cleanup()
        await environment.cleanup()
