"""Persist content evidence for repository paths from subdirectory workspaces."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from opencollab.adapters._env_container_worktree import ContainerWorktreeEnvironment
from opencollab.adapters.env import WorktreeEnvironment
from opencollab.adapters.trace import Tracer
from opencollab.adapters.worktree_pool import WorktreePool
from opencollab.application.event_bus import EventBus
from opencollab.application.scheduler import Scheduler
from tests.runtime.worktree_test_support import _ChildSession, _Factory
from tests.support.docker_native_edits_support import LocalDockerTransport


def _git(repository: Path | str, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        text=True, capture_output=True, check=True,
    ).stdout.strip()

@pytest.fixture
def repository(tmp_path, monkeypatch):
    config = tmp_path / "gitconfig"
    config.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    root = tmp_path / "source repository"
    root.mkdir()
    for path in (
        "item.py", "ordinary.py", "gone.py",
        "app/item.py", "app/ordinary.py", "app/gone.py", "app/app/item.py",
        "application/outside.py", "app/application/outside.py",
    ):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"original = {path!r}\n", encoding="utf-8")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.name", "OpenCollab Tests")
    _git(root, "config", "user.email", "tests@example.invalid")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", ".")
    _git(
        root, "commit", "-qm",
        "\u4fdd\u5b58\u8f68\u8ff9\u8def\u5f84\u6d4b\u8bd5\u521d\u59cb\u6587\u4ef6",
    )
    return root

def _environment(source: Path, tmp_path: Path, backend: str):
    if backend == "local":
        return WorktreeEnvironment(str(source))
    env = ContainerWorktreeEnvironment(
        container_id="c" * 64, repository_root=str(source),
        worktree_root=str(tmp_path / "worktrees"),
    )
    LocalDockerTransport(source, pair_reads=False).attach(env)
    return env

async def _capture(tmp_path, env) -> dict:
    tracer = Tracer(run_id="workspace-paths", output_dir=str(tmp_path / "traces"))
    scheduler = Scheduler(
        session_factory=_Factory(_ChildSession("coder", "done", env)),
        worktree_pool=WorktreePool(env.workspace, use_worktrees=False),
        event_sink=EventBus(None), tracer=tracer,
    )
    try:
        delivered = await scheduler._append_worktree_diff(env, "done", aid=1, role="coder")
        assert "[Changes made in worktree]" in delivered
        tracer.flush()
    finally:
        tracer.close()
    records = [json.loads(line) for line in Path(tracer.path).read_text(encoding="utf-8").splitlines()]
    payloads = [record["payload"] for record in records if record["type"] == "worktree_changes"]
    assert len(payloads) == 1
    assert payloads[0]["aid"] == 1
    assert payloads[0]["role"] == "coder"
    return {entry["path"]: entry for entry in payloads[0]["files"]}

@pytest.mark.parametrize("backend", ["local", "docker-shell"])
@pytest.mark.parametrize("subdirectory", ["", "app"], ids=["repository-root", "subdirectory"])
async def test_trace_content_matches_actual_modified_and_added_files(
    tmp_path, repository, backend, subdirectory,
):
    env = _environment(repository / subdirectory, tmp_path, backend)
    workspace = Path(await env.setup())
    prefix = f"{subdirectory}/" if subdirectory else ""
    try:
        for name in ("item.py", "ordinary.py", "new.py"):
            await env.write_file(name, f"changed = {name!r}\n")
        (workspace / "gone.py").unlink()

        entries = await _capture(tmp_path, env)

        assert set(entries) == {prefix + name for name in ("item.py", "ordinary.py", "new.py", "gone.py")}
        for name in ("item.py", "ordinary.py", "new.py"):
            path = prefix + name
            assert entries[path] == {
                "path": path, "op": "added" if name == "new.py" else "modified",
                "content_sha": hashlib.sha256((workspace / name).read_bytes()).hexdigest(),
            }
        assert entries[prefix + "gone.py"] == {
            "path": prefix + "gone.py", "op": "deleted", "content_sha": None,
        }
        if subdirectory:
            decoy = workspace / "app/item.py"
            assert entries["app/item.py"]["content_sha"] != hashlib.sha256(decoy.read_bytes()).hexdigest()
        assert _git(repository, "status", "--porcelain") == ""
    finally:
        await env.cleanup()

@pytest.mark.parametrize("backend", ["local", "docker-shell"])
async def test_similar_prefix_outside_workspace_records_permission_error(
    tmp_path, repository, backend,
):
    env = _environment(repository / "app", tmp_path, backend)
    workspace = Path(await env.setup())
    try:
        root = Path(_git(workspace, "rev-parse", "--show-toplevel"))
        (root / "application/outside.py").write_text("outside = 'changed'\n", encoding="utf-8")

        entries = await _capture(tmp_path, env)

        assert entries == {
            "application/outside.py": {
                "path": "application/outside.py", "op": "modified",
                "content_sha": None, "sha_error": "PermissionError",
            },
        }
        assert await env.read_file("application/outside.py") == "original = 'app/application/outside.py'\n"
        if backend == "local":
            with pytest.raises(PermissionError):
                await env.read_file("../application/outside.py")
    finally:
        await env.cleanup()
