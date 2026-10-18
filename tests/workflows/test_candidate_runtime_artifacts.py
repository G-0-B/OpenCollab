"""Runtime record ownership preserves real Git source and recovery contents."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from opencollab.adapters._env_docker import DockerEnvironment
from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from tests.workflows.test_workflow_candidate_workspace import _git, _repository


@pytest.mark.parametrize("nested", [False, True])
async def test_candidate_source_and_restoration_share_runtime_record_filter(tmp_path, nested):
    repo = _repository(tmp_path)
    source = repo / "package" if nested else repo
    source.mkdir(exist_ok=True)
    owned = repo / "records [current]"
    owned.mkdir()
    tracked = owned / "tracked.txt"
    tracked.write_text("tracked user data\n")
    (source / "source.py").write_text("value = 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Add user file in runtime directory")
    tracked.write_text("tracked user edit\n")
    (source / "source.py").write_text("value = 10\n")
    _git(repo, "add", ".")
    user = source / "user.txt"
    user.write_text("untracked user data\n")
    adjacent = repo / "records [current]-older"
    adjacent.mkdir()
    (adjacent / "note.txt").write_text("adjacent user data\n")
    record = owned / "session.json"
    record.write_text("runtime record one\n")
    index = (repo / ".git" / "index").read_bytes()
    environment = LocalEnvironment(str(source))
    workspace = EnvCandidateWorkspace(environment, owned_save_dir=str(owned))
    lease = None
    try:
        original = await workspace.source_diff()
        assert "session.json" not in original
        assert "+tracked user edit" in original
        assert "+untracked user data" in original
        assert "+adjacent user data" in original
        lease = await workspace.acquire("edit")
        candidate = Path(lease.candidate_workspace)
        assert not (candidate / owned.name / "session.json").exists()
        assert (candidate / owned.name / "tracked.txt").read_text() == "tracked user edit\n"
        assert (candidate / adjacent.name / "note.txt").read_text() == "adjacent user data\n"
        assert await lease.environment.read_file("user.txt") == "untracked user data\n"
        assert await lease.environment.read_file("source.py") == "value = 10\n"
        assert await lease.diff() == ""
        record.write_text("runtime record two\n")
        assert await workspace.source_diff() == original
        await lease.environment.write_file("source.py", "value = 20\n")
        await workspace.adopt(await lease.diff())
        assert (source / "source.py").read_text() == "value = 20\n"
        await workspace.restore_source(original)
        assert (source / "source.py").read_text() == "value = 10\n"
        assert tracked.read_text() == "tracked user edit\n"
        assert user.read_text() == "untracked user data\n"
        assert record.read_text() == "runtime record two\n"
        assert await workspace.source_diff() == original
        assert (repo / ".git" / "index").read_bytes() == index
    finally:
        if lease is not None:
            await lease.cleanup()
        await environment.cleanup()

@pytest.mark.parametrize("mutation", ["modified", "staged", "deleted", "renamed"])
async def test_owned_runtime_directory_keeps_tracked_user_changes(tmp_path, mutation):
    repo = _repository(tmp_path)
    owned = repo / "records"
    owned.mkdir()
    tracked = owned / "tracked.txt"
    tracked.write_text("user data\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Add tracked file in runtime directory")
    (owned / "session.json").write_text("runtime data\n")
    environment = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(environment, owned_save_dir=str(owned))
    try:
        assert await workspace.source_diff() == ""
        assert await workspace._source_changed() is False
        if mutation in {"modified", "staged"}:
            tracked.write_text("user edit\n")
            if mutation == "staged":
                _git(repo, "add", "records/tracked.txt")
        elif mutation == "deleted":
            tracked.unlink()
        else:
            _git(repo, "mv", "records/tracked.txt", "records/renamed.txt")
        index = (repo / ".git" / "index").read_bytes()
        patch = await workspace.source_diff()
        assert "tracked.txt" in patch
        assert "session.json" not in patch
        assert await workspace._source_changed() is True
        assert await workspace._source_changed(["records/tracked.txt", "records/renamed.txt"]) is False
        assert (repo / ".git" / "index").read_bytes() == index
    finally:
        await environment.cleanup()

@pytest.mark.parametrize("location", ["source", "ancestor"])
async def test_owned_save_directory_covering_source_keeps_user_contents(tmp_path, location):
    repo = _repository(tmp_path)
    (repo / "user.txt").write_text("user data\n")
    environment = LocalEnvironment(str(repo))
    owned = repo if location == "source" else tmp_path
    workspace = EnvCandidateWorkspace(environment, owned_save_dir=str(owned))
    lease = None
    try:
        assert "+user data" in await workspace.source_diff()
        assert await workspace._source_changed() is True
        lease = await workspace.acquire("covering-save")
        assert await lease.environment.read_file("user.txt") == "user data\n"
    finally:
        if lease is not None:
            await lease.cleanup()
        await environment.cleanup()

async def test_container_source_keeps_paths_matching_host_runtime_directory(tmp_path):
    repo = _repository(tmp_path)
    records = repo / "records"
    records.mkdir()
    (records / "user.txt").write_text("container user data\n")
    local = LocalEnvironment(str(repo))

    class ContainerCommands(DockerEnvironment):
        # Real Git commands stand in for the container transport. This adapter
        # has no claim over the host output supplied by the workflow runtime.
        async def exec_cmd(self, command, **kwargs):
            return await local.exec_cmd(command, **kwargs)

    workspace = EnvCandidateWorkspace(
        ContainerCommands(workspace=str(repo)), owned_save_dir=str(records),
    )
    try:
        assert "+container user data" in await workspace.source_diff()
        assert await workspace._source_changed() is True
    finally:
        await local.cleanup()

async def test_owned_source_status_preserves_index_when_clean_file_stat_changes(tmp_path):
    repo = _repository(tmp_path)
    owned = repo / "records"
    owned.mkdir()
    tracked = owned / "tracked.txt"
    tracked.write_text("user data\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Add index refresh control file")
    clean = repo / "source.py"
    stat = clean.stat()
    os.utime(clean, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    tracked.write_text("user edit\n")
    (owned / "session.json").write_text("runtime data\n")
    index = (repo / ".git" / "index").read_bytes()
    environment = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(environment, owned_save_dir=str(owned))
    try:
        assert await workspace._source_changed() is True
        assert clean.read_text() == "value = 1\n"
        assert tracked.read_text() == "user edit\n"
        assert (repo / ".git" / "index").read_bytes() == index
    finally:
        await environment.cleanup()
