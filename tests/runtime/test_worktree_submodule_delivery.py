"""Parent patches retain module contents they cannot deliver."""

import subprocess
from pathlib import Path

import pytest

from opencollab.adapters import worktree_pool
from opencollab.adapters.worktree_pool import WorktreePool
from opencollab.environments import attach_container
from tests.support.worktree_delivery_support import CONTAINER_ID, git
from tests.support.worktree_delivery_support import worktree_transport as worktree_transport
from tests.workflows.test_candidate_submodule_delivery import _source_with_dependency


async def _acquire(source, tmp_path, monkeypatch, backend):
    monkeypatch.setattr(worktree_pool, "CONTAINER_WORKTREE_ROOT", str(tmp_path / "worktrees"))
    base = attach_container(container_id=CONTAINER_ID, workspace=str(source)) if backend == "container" else None
    pool = WorktreePool(str(source), use_worktrees=True, base_environment=base)
    env = await pool.acquire("coder")
    # Establish an initialized checkout for export tests independently of the
    # container setup path. Every Git command still runs through its transport.
    initialized = await env.exec_cmd("git -c protocol.file.allow=always submodule update --init --recursive")
    assert initialized.returncode == 0, initialized.stderr
    return pool, env

def _remove_retained(source, env):
    git(source, "worktree", "remove", "--force", "--force", env.workspace)
    git(source, "update-ref", "-d", f"refs/heads/{env._branch}", env._base_commit)

@pytest.mark.parametrize("backend", ["local", "container"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("mutation", ["dirty", "untracked", "ignored", "commit", "gitlink", "remove", "replace", "add"])
async def test_unpatchable_module_changes_survive_capture_and_cleanup(
    worktree_transport, tmp_path, monkeypatch, backend, nested, mutation,
):
    source, module_path = _source_with_dependency(tmp_path, nested)
    pool, env = await _acquire(source, tmp_path, monkeypatch, backend)
    workspace = Path(env.workspace)
    module = workspace / module_path
    changed_file = module / ("new_source.py" if mutation in {"untracked", "ignored"} else "source.py")
    try:
        git(workspace, "config", "diff.ignoreSubmodules", "all")
        git(workspace, "config", "submodule.vendor/dependency.ignore", "all")
        if nested:
            git(workspace / "vendor/dependency", "config", "submodule.vendor/leaf.ignore", "all")
        if mutation in {"dirty", "untracked", "ignored", "commit"}:
            if mutation == "ignored":
                git(module, "config", "core.excludesFile", "/dev/null")
                exclude = Path(git(module, "rev-parse", "--git-path", "info/exclude").strip())
                exclude.write_text("new_source.py\n")
            changed_file.write_text("value = 2\n")
            if mutation == "commit":
                git(module, "-c", "user.name=Test User", "-c", "user.email=test@example.invalid",
                    "commit", "-am", "\u4fdd\u5b58\u4f9d\u8d56\u4fee\u6539")
        elif mutation == "gitlink":
            # The real index can differ even while the checkout remains clean.
            oid = git(source, "rev-parse", "HEAD").strip()
            git(workspace, "update-index", "--cacheinfo", f"160000,{oid},vendor/dependency")
        elif mutation in {"remove", "replace"}:
            git(workspace, "rm", "-f", "vendor/dependency")
            if mutation == "replace":
                module = workspace / "vendor/dependency"
                module.mkdir(parents=True)
                changed_file = module / "source.py"
                changed_file.write_text("replacement = True\n")
        else:
            git(workspace, "-c", "protocol.file.allow=always", "submodule", "add",
                str(source / "vendor/dependency"), "vendor/added")
            changed_file = workspace / "vendor/added/source.py"
        (workspace / "source.py").write_text("parent = 2\n")
        index = Path(git(workspace, "rev-parse", "--git-path", "index").strip())
        index_before = index.read_bytes()
        for _ in range(2):
            with pytest.raises(RuntimeError, match="submodule changes"):
                await env.get_diff()
            assert str(workspace) in env.recovery_location
            assert index.read_bytes() == index_before
            with pytest.raises(OSError, match="retry state retained"):
                await pool.release()
            assert workspace.is_dir()
            assert (workspace / "source.py").read_text() == "parent = 2\n"
            assert f"worktree {workspace.resolve()}" in git(source, "worktree", "list", "--porcelain")
            if mutation != "remove":
                assert changed_file.is_file()
                expected = {
                    "dirty": "value = 2\n", "untracked": "value = 2\n", "ignored": "value = 2\n",
                    "commit": "value = 2\n",
                    "gitlink": "value = 1\n", "replace": "replacement = True\n", "add": "value = 1\n",
                }[mutation]
                assert changed_file.read_text() == expected
            if mutation == "commit":
                assert git(module, "show", "HEAD:source.py") == "value = 2\n"
        assert (source / "source.py").read_text() == "value = 1\n"
        assert (source / module_path / "source.py").read_text() == "value = 1\n"
    finally:
        _remove_retained(source, env)

@pytest.mark.parametrize("backend", ["local", "container"])
async def test_parent_commit_of_module_gitlink_retains_module_objects(
    worktree_transport, tmp_path, monkeypatch, backend,
):
    source, module_path = _source_with_dependency(tmp_path, False)
    pool, env = await _acquire(source, tmp_path, monkeypatch, backend)
    workspace = Path(env.workspace)
    module = workspace / module_path
    try:
        (module / "source.py").write_text("value = 2\n")
        for repository in (module, workspace):
            git(repository, "-c", "user.name=Test User", "-c", "user.email=test@example.invalid",
                "commit", "-am", "\u4fdd\u5b58\u4f9d\u8d56\u63d0\u4ea4")
        assert git(workspace, "status", "--porcelain") == ""
        with pytest.raises(RuntimeError, match="submodule changes"):
            await env.get_diff()
        assert env.own_commit_count == 1
        with pytest.raises(OSError, match="retry state retained"):
            await pool.release()
        assert git(module, "show", "HEAD:source.py") == "value = 2\n"
        assert str(workspace) in env.recovery_location
        assert (source / module_path / "source.py").read_text() == "value = 1\n"
    finally:
        _remove_retained(source, env)

@pytest.mark.parametrize("backend", ["local", "container"])
async def test_clean_modules_and_parent_patch_export_and_cleanup(
    worktree_transport, tmp_path, monkeypatch, backend,
):
    source, module_path = _source_with_dependency(tmp_path, True)
    pool, env = await _acquire(source, tmp_path, monkeypatch, backend)
    workspace = Path(env.workspace)
    try:
        assert await env.get_diff() == ""
        await env.write_file("source.py", "value = 2\n")
        patch = await env.get_diff()
        assert "+value = 2" in patch
        assert env.recovery_location is None
        result = await env.exec_cmd("git diff --quiet -- vendor/dependency")
        assert result.returncode == 0
        subprocess.run(["git", "-C", str(source), "apply", "-"], input=patch, text=True, check=True)
        assert (source / "source.py").read_text() == "value = 2\n"
        assert (source / module_path / "source.py").read_text() == "value = 1\n"
    finally:
        await pool.release()
    assert not workspace.exists()
