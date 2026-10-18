"""A worktree inside the task container reports the same evidence as one outside.

The Team arm's per-agent attribution rests on four values a worktree reports --
the base its diff was measured against, where its HEAD stands, and the commits
it made itself -- and a handoff is read as ``tester.diff_base in
coder.commits``. Those readings have to mean the same thing when the repository
under test lives inside a container, because that is where the evaluation
harness keeps it and the host is never allowed to see it.

These tests run real ``git`` against real linked worktrees, with ``docker exec``
replaced by running the same argv locally. What is deliberately not covered here
is the Docker plumbing itself -- ``tests/runtime/test_docker_env.py`` owns that. What is
covered is everything this class decides: which commands it builds, where it
puts the worktree, and what it concludes from what Git answers.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from opencollab.adapters import _env_docker as docker_module
from opencollab.adapters._env_container_worktree import ContainerWorktreeEnvironment
from opencollab.adapters._env_process import ProcessResult
from opencollab.adapters._env_process import run_process as real_run_process
from opencollab.application._scheduler_team import _parse_worktree_diff

CONTAINER_ID = "c" * 64

def _git(cwd, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

def _repo(path, *, object_format: str = "sha1"):
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "main", f"--object-format={object_format}")
    _git(path, "config", "user.email", "tests@example.com")
    _git(path, "config", "user.name", "OpenCollab Tests")
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-qm", "base")
    return path

def _changed_files(diff: str) -> list[str]:
    """What the scheduler's ``worktree_changes`` record would list."""
    return [path for path, _ in _parse_worktree_diff(diff)]

@pytest.fixture
def local_docker(monkeypatch):
    """Run what would go to ``docker exec`` as a local process instead.

    The container is the only thing faked. Everything the command does -- the
    Git it runs, the files it touches, the worktrees it registers -- is real, so
    a wrong command or a wrong conclusion still fails the test.
    """

    async def shim(command, **kwargs):
        argv = list(command)
        assert argv[0] == "docker", argv
        if argv[1] == "inspect":
            return ProcessResult(0, f"{CONTAINER_ID}\t/oc-test\ttrue".encode(), b"")
        assert argv[1] == "exec", argv
        rest = argv[2:]
        workdir = None
        while rest and rest[0] != "--":
            if rest[0] == "-w":
                workdir = rest[1]
                rest = rest[2:]
            else:  # -i, and anything else Docker takes before the container
                rest = rest[1:]
        assert rest[0] == "--" and rest[1] == CONTAINER_ID, rest
        return await real_run_process(
            tuple(rest[2:]),
            shell=False,
            cwd=workdir,
            timeout=kwargs.get("timeout", 60.0),
            input_bytes=kwargs.get("input_bytes"),
        )

    monkeypatch.setattr(docker_module, "run_process", shim)
    return shim

def _env(repo, tmp_path, branch: str) -> ContainerWorktreeEnvironment:
    return ContainerWorktreeEnvironment(
        container_id=CONTAINER_ID,
        repository_root=str(repo),
        worktree_root=str(tmp_path / "worktrees"),
        branch_name=branch,
    )

async def test_verified_write_accepts_bsd_wc_padding(local_docker, monkeypatch, tmp_path):
    """A BSD-style padded ``wc -c`` count still verifies the write."""
    repo = _repo(tmp_path / "testbed")
    env = _env(repo, tmp_path, "container-bsd-wc")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_wc = fake_bin / "wc"
    fake_wc.write_text("#!/bin/sh\nprintf '      5\\n'\n", encoding="utf-8")
    fake_wc.chmod(0o755)
    try:
        await env.setup()
        monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
        await env.write_file("padded.txt", "hello")
        assert Path(env.workspace, "padded.txt").read_text(encoding="utf-8") == "hello"
    finally:
        await env.cleanup()

async def test_an_agents_own_commits_stay_in_its_own_diff(local_docker, tmp_path):
    repo = _repo(tmp_path / "testbed")
    env = _env(repo, tmp_path, "container-coder")
    try:
        await env.setup()
        creation_base = _git(env.workspace, "rev-parse", "HEAD")
        await env.write_file("f1.txt", "coder work\n")
        assert (await env.exec_cmd("git add -A && git commit -qm coder")).returncode == 0

        diff = await env.get_diff()

        assert _changed_files(diff) == ["f1.txt"]
        assert env.diff_base == creation_base
        assert env.head_commit != creation_base
        assert env.own_commits == (env.head_commit,)
        assert env.own_commit_count == 1
    finally:
        await env.cleanup()

async def test_a_checked_out_teammate_commit_is_not_this_agents_work(
    local_docker, tmp_path
):
    """The join the information-flow reading depends on, inside a container."""
    repo = _repo(tmp_path / "testbed")
    coder = _env(repo, tmp_path, "container-handoff-coder")
    tester = _env(repo, tmp_path, "container-handoff-tester")
    try:
        await coder.setup()
        await coder.write_file("f1.txt", "coder work\n")
        assert (await coder.exec_cmd("git add -A && git commit -qm coder")).returncode == 0
        handoff_sha = _git(coder.workspace, "rev-parse", "HEAD")
        assert _changed_files(await coder.get_diff()) == ["f1.txt"]

        await tester.setup()
        assert (await tester.exec_cmd(f"git checkout -q {handoff_sha}")).returncode == 0
        await tester.write_file("f2.txt", "tester work\n")

        diff = await tester.get_diff()

        assert _changed_files(diff) == ["f2.txt"]
        assert "f1.txt" not in diff
        assert tester.diff_base == handoff_sha
        assert tester.diff_base in coder.own_commits
    finally:
        await tester.cleanup()
        await coder.cleanup()

async def test_a_sibling_cannot_see_work_that_was_not_committed(local_docker, tmp_path):
    """What isolation buys: an edit nobody committed reaches nobody else."""
    repo = _repo(tmp_path / "testbed")
    coder = _env(repo, tmp_path, "container-solo-coder")
    tester = _env(repo, tmp_path, "container-solo-tester")
    try:
        await coder.setup()
        await tester.setup()
        await coder.write_file("only_here.txt", "uncommitted\n")

        listing = await tester.exec_cmd("ls only_here.txt")

        assert listing.returncode != 0
        assert not (repo / "only_here.txt").exists()
    finally:
        await tester.cleanup()
        await coder.cleanup()

async def test_the_worktree_is_not_inside_the_repository_it_checks_out(
    local_docker, tmp_path
):
    """An archive of the repository root is what the harness reads its patch from.

    A worktree nested inside that root would arrive in the archive as a
    directory full of files no agent wrote.
    """
    repo = _repo(tmp_path / "testbed")
    env = _env(repo, tmp_path, "container-outside")
    try:
        await env.setup()
        await env.write_file("f1.txt", "coder work\n")

        assert not str(env.workspace).startswith(f"{repo}/")
        assert _git(repo, "status", "--porcelain") == ""
    finally:
        await env.cleanup()

async def test_an_agents_commits_do_not_move_the_branch_it_leased(
    local_docker, tmp_path
):
    """The claimed ref is an ownership lease, not the worktree's HEAD.

    A detached worktree keeps its own commits off that ref, which is what lets
    cleanup delete it only if it still stands where it was claimed -- so a ref
    something else advanced stays visible instead of being quietly removed.
    """
    repo = _repo(tmp_path / "testbed")
    env = _env(repo, tmp_path, "container-lease")
    try:
        await env.setup()
        leased_at = _git(repo, "rev-parse", "refs/heads/container-lease")
        await env.write_file("f1.txt", "coder work\n")
        assert (await env.exec_cmd("git add -A && git commit -qm coder")).returncode == 0

        assert _git(env.workspace, "rev-parse", "HEAD") != leased_at
        assert _git(repo, "rev-parse", "refs/heads/container-lease") == leased_at
        assert _git(env.workspace, "branch", "--show-current") == ""
    finally:
        await env.cleanup()
    assert "container-lease" not in _git(repo, "branch", "--list")

async def test_cleanup_gives_back_the_worktree_and_its_branch(local_docker, tmp_path):
    repo = _repo(tmp_path / "testbed")
    env = _env(repo, tmp_path, "container-released")
    await env.setup()
    workspace = env.workspace
    assert "container-released" in _git(repo, "worktree", "list")

    await env.cleanup()

    assert "container-released" not in _git(repo, "worktree", "list")
    assert "container-released" not in _git(repo, "branch", "--list")
    assert not (tmp_path / "worktrees" / "container-released").exists()
    assert workspace.endswith("container-released")

async def test_a_handoff_is_read_in_a_sha256_repository_too(local_docker, tmp_path):
    """The harness rebuilds some repositories with 64-character object ids."""
    repo = _repo(tmp_path / "testbed", object_format="sha256")
    coder = _env(repo, tmp_path, "container-sha256-coder")
    tester = _env(repo, tmp_path, "container-sha256-tester")
    try:
        await coder.setup()
        await coder.write_file("f1.txt", "coder work\n")
        assert (await coder.exec_cmd("git add -A && git commit -qm coder")).returncode == 0
        handoff_sha = _git(coder.workspace, "rev-parse", "HEAD")
        assert len(handoff_sha) == 64
        assert _changed_files(await coder.get_diff()) == ["f1.txt"]

        await tester.setup()
        assert (await tester.exec_cmd(f"git checkout -q {handoff_sha}")).returncode == 0
        await tester.write_file("f2.txt", "tester work\n")

        assert _changed_files(await tester.get_diff()) == ["f2.txt"]
        assert tester.diff_base == handoff_sha
        assert tester.diff_base in coder.own_commits
    finally:
        await tester.cleanup()
        await coder.cleanup()

async def test_a_repository_hardened_by_its_harness_still_reports_a_diff(
    local_docker, tmp_path
):
    """The three local settings the evaluation harness writes into its snapshot."""
    repo = _repo(tmp_path / "testbed")
    _git(repo, "config", "core.attributesFile", "/dev/null")
    _git(repo, "config", "core.fsmonitor", "false")
    _git(repo, "config", "diff.ignoreSubmodules", "all")
    env = _env(repo, tmp_path, "container-hardened")
    try:
        await env.setup()
        await env.write_file("f1.txt", "coder work\n")

        assert _changed_files(await env.get_diff()) == ["f1.txt"]
    finally:
        await env.cleanup()

@pytest.mark.parametrize(
    ("repository_root", "worktree_root"),
    [("relative/testbed", "/worktrees"), ("/testbed", "worktrees"), ("/", "/worktrees")],
)
def test_container_paths_must_be_absolute_and_not_the_root(repository_root, worktree_root):
    with pytest.raises(ValueError):
        ContainerWorktreeEnvironment(
            container_id=CONTAINER_ID,
            repository_root=repository_root,
            worktree_root=worktree_root,
        )

async def test_workflow_factory_isolates_inside_the_supplied_container(
    local_docker, tmp_path, monkeypatch
):
    from opencollab.adapters import worktree_pool
    from opencollab.adapters.env import DockerEnvironment
    from opencollab.bootstrap.workflow_runtime import WorkflowSessionFactory

    repo = _repo(tmp_path / 'testbed')
    anchor = _repo(tmp_path / 'host')
    monkeypatch.setattr(worktree_pool, 'CONTAINER_WORKTREE_ROOT', str(tmp_path / 'worktrees'))
    prefix = 'export WORKFLOW_IMAGE_MARKER=task-image'
    base = DockerEnvironment(
        workspace=str(repo), container_id=CONTAINER_ID, command_prefix=prefix
    )
    factory = WorkflowSessionFactory(
        model='test-model', provider='openai', api_key=None, base_url=None,
        workspace=str(anchor), env=base,
    )
    isolated = await factory.acquire_isolated_env(label='coder')
    try:
        assert isinstance(isolated, ContainerWorktreeEnvironment)
        assert isolated.container_reference == CONTAINER_ID
        assert isolated.source_workspace == str(repo)
        assert isolated.command_prefix == prefix
        await isolated.write_file('candidate.txt', 'isolated change')
        assert not (repo / 'candidate.txt').exists()
        assert not (anchor / 'candidate.txt').exists()
        assert (await isolated.exec_cmd('echo "$WORKFLOW_IMAGE_MARKER"')).stdout.strip() == 'task-image'
    finally:
        await factory.release_isolated_envs()
    assert base.revoked is False
    assert not Path(isolated.workspace).exists()

@pytest.mark.parametrize("command", ["git reset --hard HEAD", "git checkout -b fix-topic"])
async def test_noop_reset_does_not_hide_a_committed_candidate(local_docker, tmp_path, command):
    repo = _repo(tmp_path / 'testbed')
    env = _env(repo, tmp_path, 'candidate-reset')
    try:
        await env.setup()
        await env.write_file('fix.txt', 'candidate change')
        assert (await env.exec_cmd('git add -A && git commit -qm fix')).returncode == 0
        before = await env.get_diff()
        assert 'fix.txt' in before
        assert (await env.exec_cmd(command)).returncode == 0
        assert await env.get_diff() == before
    finally:
        await env.cleanup()

@pytest.mark.parametrize("symlink_root", [False, True])
async def test_locked_container_worktree_reports_failure_and_can_retry(local_docker, tmp_path, symlink_root):
    repo = _repo(tmp_path / 'testbed')
    if symlink_root:
        physical = tmp_path / "physical worktrees\n\u5de5\u4f5c"
        physical.mkdir()
        (tmp_path / "worktrees").symlink_to(physical, target_is_directory=True)
    env = _env(repo, tmp_path, 'candidate-locked')
    await env.setup()
    _git(repo, 'worktree', 'lock', env.workspace)
    try:
        with pytest.raises(OSError, match='worktree'):
            await env.cleanup()
        assert Path(env.workspace).exists()
        assert env._worktree_registered
        assert 'candidate-locked' in _git(repo, 'branch', '--list')
    finally:
        _git(repo, 'worktree', 'unlock', env.workspace)
        await env.cleanup()
    assert not Path(env.workspace).exists()

@pytest.mark.parametrize('root', ['/app/children', '/app', '/app/../app/worktrees'])
def test_worktree_root_cannot_contaminate_the_exported_repository(root):
    with pytest.raises(ValueError, match='outside'):
        ContainerWorktreeEnvironment(
            container_id=CONTAINER_ID, repository_root='/app', worktree_root=root,
        )

@pytest.mark.parametrize(('repository', 'worktrees'), [('//', '/tmp/worktrees'), ('//app', '/app/children')])
def test_redundant_slashes_do_not_bypass_repository_boundaries(repository, worktrees):
    with pytest.raises(ValueError):
        ContainerWorktreeEnvironment(
            container_id=CONTAINER_ID, repository_root=repository, worktree_root=worktrees,
        )

@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="requires enforced POSIX permissions")
async def test_container_permission_failure_can_retry_after_git_unregisters(local_docker, tmp_path):
    repo = _repo(tmp_path / "testbed")
    env = _env(repo, tmp_path, "permission-recovery")
    workspace = Path(await env.setup())
    protected = workspace / "protected"
    protected.mkdir()
    (protected / "output.txt").write_text("output\n", encoding="utf-8")
    protected.chmod(0o500)
    try:
        with pytest.raises(OSError, match="Permission denied"):
            await env.cleanup()
        assert f"worktree {workspace}" not in _git(repo, "worktree", "list", "--porcelain", "-z").split("\0")
        assert protected.is_dir()
        assert not env._worktree_registered
        assert "permission-recovery" not in _git(repo, "branch", "--list")
    finally:
        protected.chmod(0o700)
        await env.cleanup()
    assert not workspace.exists()
    await env.cleanup()

@pytest.fixture
def container_namespace(local_docker, monkeypatch, tmp_path):
    """Translate an absent host path into the simulated container namespace."""
    virtual_root = f"/oc-container-{tmp_path.name}"
    physical_root = str(tmp_path.resolve())
    assert not Path(virtual_root).exists()

    async def translated_docker(command, **kwargs):
        translated = tuple(arg.replace(virtual_root, physical_root) for arg in command)
        result = await local_docker(translated, **kwargs)
        return ProcessResult(
            result.returncode,
            result.stdout.replace(physical_root.encode(), virtual_root.encode()),
            result.stderr.replace(physical_root.encode(), virtual_root.encode()),
            result.stdout_dropped_bytes,
            result.stderr_dropped_bytes,
        )

    monkeypatch.setattr(docker_module, "run_process", translated_docker)

    def container_path(path):
        return str(path.resolve()).replace(physical_root, virtual_root)

    return container_path

def _source_with_submodules(tmp_path, *, nested):
    leaf = _repo(tmp_path / "leaf")
    dependency = leaf
    dependency_path = "vendor/dependency"
    if nested:
        dependency = _repo(tmp_path / "middle")
        _git(
            dependency, "-c", "protocol.file.allow=always", "submodule", "add",
            "-q", str(leaf), "vendor/leaf",
        )
        _git(
            dependency, "config", "-f", ".gitmodules", "submodule.vendor/leaf.url",
            "https://example.invalid/leaf.git",
        )
        _git(dependency, "add", ".gitmodules", "vendor/leaf")
        _git(dependency, "commit", "-qm", "add nested leaf")
        dependency_path += "/vendor/leaf"
    source = _repo(tmp_path / "testbed")
    _git(
        source, "-c", "protocol.file.allow=always", "submodule", "add", "-q",
        str(dependency), "vendor/dependency",
    )
    if nested:
        _git(
            source / "vendor/dependency", "-c", "protocol.file.allow=always",
            "-c", f"submodule.vendor/leaf.url={leaf}", "submodule", "update",
            "--init", "--no-fetch", "--", "vendor/leaf",
        )
    _git(
        source, "config", "-f", ".gitmodules", "submodule.vendor/dependency.url",
        "https://example.invalid/dependency.git",
    )
    (source / "check.py").write_text(
        f"from pathlib import Path\nprint(Path('{dependency_path}/tracked.txt').read_text())\n",
        encoding="utf-8",
    )
    _git(source, "add", ".gitmodules", "vendor/dependency", "check.py")
    _git(source, "commit", "-qm", "add source dependency")
    return source, dependency_path

@pytest.mark.parametrize("nested", [False, True])
async def test_container_initializes_available_submodules_in_its_own_namespace(
    container_namespace, monkeypatch, tmp_path, nested,
):
    source, dependency_path = _source_with_submodules(tmp_path, nested=nested)
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    env = _env(container_namespace(source), Path(container_namespace(tmp_path)), "container-submodules")
    try:
        await env.setup()
        assert not Path(env.workspace).exists()
        result = await env.exec_cmd(f"{shlex.quote(sys.executable)} check.py")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "base\n\n"
        assert await env.read_file(f"{dependency_path}/tracked.txt") == "base\n"
        assert await env.get_diff() == ""
        assert _git(source, "status", "--porcelain") == ""
    finally:
        await env.cleanup()
    assert _git(source, "worktree", "list", "--porcelain").count("worktree ") == 1

async def test_container_keeps_uncommitted_source_changes_isolated(local_docker, tmp_path):
    source = _repo(tmp_path / "testbed")
    (source / "tracked.txt").write_text("sibling edit\n", encoding="utf-8")
    (source / "sibling.txt").write_text("sibling new file\n", encoding="utf-8")
    env = _env(source, tmp_path, "container-dirty-source")
    try:
        await env.setup()
        assert await env.read_file("tracked.txt") == "base\n"
        assert (await env.exec_cmd("test -e sibling.txt")).returncode != 0
        assert (source / "tracked.txt").read_text(encoding="utf-8") == "sibling edit\n"
        assert (source / "sibling.txt").read_text(encoding="utf-8") == "sibling new file\n"
        assert await env.get_diff() == ""
    finally:
        await env.cleanup()

async def test_container_uses_committed_submodules_despite_sibling_edits(
    container_namespace, tmp_path,
):
    source, dependency_path = _source_with_submodules(tmp_path, nested=False)
    (source / dependency_path / "tracked.txt").write_text("sibling edit\n", encoding="utf-8")
    _git(source, "config", "-f", ".gitmodules", "submodule.vendor/dependency.path", "missing")
    original_status = _git(source, "status", "--porcelain")
    env = _env(container_namespace(source), Path(container_namespace(tmp_path)), "container-dirty-modules")
    try:
        await env.setup()
        assert await env.read_file(f"{dependency_path}/tracked.txt") == "base\n"
        assert await env.get_diff() == ""
        assert _git(source, "status", "--porcelain") == original_status
    finally:
        await env.cleanup()

async def test_container_leaves_uninitialized_optional_submodule_empty(
    container_namespace, tmp_path,
):
    source, dependency_path = _source_with_submodules(tmp_path, nested=False)
    _git(source, "submodule", "deinit", "-f", "--", dependency_path)
    env = _env(container_namespace(source), Path(container_namespace(tmp_path)), "container-optional-module")
    try:
        await env.setup()
        assert (await env.exec_cmd(f"test -e {dependency_path}/tracked.txt")).returncode != 0
        assert await env.get_diff() == ""
    finally:
        await env.cleanup()

@pytest.mark.parametrize("sibling_edit", ["rename", "staged_removal"])
async def test_container_pool_keeps_optional_modules_isolated_from_sibling_configuration(
    container_namespace, monkeypatch, tmp_path, sibling_edit,
):
    from opencollab.adapters import worktree_pool
    from opencollab.adapters.env import DockerEnvironment
    from opencollab.adapters.worktree_pool import WorktreePool

    source, dependency_path = _source_with_submodules(tmp_path, nested=False)
    committed_configuration = (source / ".gitmodules").read_text(encoding="utf-8")
    _git(source, "submodule", "deinit", "-f", "--", dependency_path)
    monkeypatch.setattr(
        worktree_pool, "CONTAINER_WORKTREE_ROOT", container_namespace(tmp_path / "worktrees"),
    )
    base = DockerEnvironment(workspace=container_namespace(source), container_id=CONTAINER_ID)
    pool = WorktreePool(str(source), use_worktrees=True, base_environment=base)
    first = await pool.acquire("first")
    try:
        if sibling_edit == "rename":
            _git(source, "config", "-f", ".gitmodules", "submodule.vendor/dependency.path", "renamed")
        else:
            _git(source, "rm", "--cached", "--", dependency_path)
            _git(source, "config", "-f", ".gitmodules", "--remove-section", "submodule.vendor/dependency")
            _git(source, "add", ".gitmodules")
        source_status = _git(source, "status", "--porcelain")
        second = await pool.acquire("second")
        for environment in (first, second):
            assert await environment.read_file("tracked.txt") == "base\n"
            assert await environment.read_file(".gitmodules") == committed_configuration
            assert (await environment.exec_cmd(f"test -e {dependency_path}/tracked.txt")).returncode != 0
            assert await environment.get_diff() == ""
        assert _git(source, "status", "--porcelain") == source_status
    finally:
        await pool.release()
    assert _git(source, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert _git(source, "branch", "--list", "opencollab-*") == ""
    assert base.revoked is False

@pytest.mark.parametrize("metadata", ["invalid_gitfile", "missing_directory"])
async def test_container_optional_module_retains_source_metadata_validation(
    container_namespace, monkeypatch, tmp_path, metadata,
):
    source, dependency_path = _source_with_submodules(tmp_path, nested=False)
    _git(source, "submodule", "deinit", "-f", "--", dependency_path)
    source_module = source / dependency_path
    if metadata == "invalid_gitfile":
        (source_module / ".git").write_text("invalid Git metadata\n", encoding="utf-8")
    else:
        source_module.rmdir()
        original_transport = docker_module.run_process

        async def missing_directory_transport(command, **kwargs):
            try:
                return await original_transport(command, **kwargs)
            except FileNotFoundError as error:
                # Docker reports an absent exec working directory as a failed
                # command rather than a host subprocess creation exception.
                return ProcessResult(127, b"", str(error).encode())

        monkeypatch.setattr(docker_module, "run_process", missing_directory_transport)
    env = _env(container_namespace(source), Path(container_namespace(tmp_path)), "metadata-control")
    try:
        if metadata == "invalid_gitfile":
            with pytest.raises(RuntimeError, match="not initialized in source workspace"):
                await env.setup()
            assert (source_module / ".git").read_text(encoding="utf-8") == "invalid Git metadata\n"
        else:
            await env.setup()
            assert await env.get_diff() == ""
    finally:
        await env.cleanup()
    assert _git(source, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert _git(source, "branch", "--list", "metadata-control") == ""

async def test_container_pool_preserves_nested_workspace_and_captures_from_repository_root(
    container_namespace, monkeypatch, tmp_path,
):
    from opencollab.adapters import worktree_pool
    from opencollab.adapters.env import DockerEnvironment
    from opencollab.adapters.worktree_pool import WorktreePool

    source = _repo(tmp_path / "testbed")
    package = source / "packages/worker"
    package.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 7\n", encoding="utf-8")
    _git(source, "add", "packages/worker/module.py")
    _git(source, "commit", "-qm", "add worker package")
    monkeypatch.setattr(
        worktree_pool, "CONTAINER_WORKTREE_ROOT", container_namespace(tmp_path / "worktrees"),
    )
    base = DockerEnvironment(workspace=container_namespace(package), container_id=CONTAINER_ID)
    pool = WorktreePool(str(package), use_worktrees=True, base_environment=base)
    isolated = await pool.acquire("coder")
    try:
        assert isolated.workspace.endswith("/packages/worker")
        assert isolated.source_workspace == container_namespace(package)
        assert not Path(isolated.workspace).exists()
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote('from module import VALUE; print(VALUE)')}"
        result = await isolated.exec_cmd(command)
        assert result.returncode == 0, result.stderr
        assert result.stdout == "7\n"
        await isolated.write_file("module.py", "VALUE = 8\n")
        assert (await isolated.exec_cmd("printf 'root addition\\n' > ../../root-added.txt")).returncode == 0
        diff = await isolated.get_diff()
        assert "packages/worker/module.py" in diff
        assert "root-added.txt" in diff
        assert "+VALUE = 8" in diff
        assert "+root addition" in diff
        assert _git(source, "status", "--porcelain") == ""
    finally:
        await pool.release()
    assert _git(source, "worktree", "list", "--porcelain").count("worktree ") == 1
    assert _git(source, "branch", "--list", "opencollab-coder-*") == ""
    assert base.revoked is False

async def test_container_initializes_submodules_within_nested_workspace(
    container_namespace, tmp_path,
):
    dependency = _repo(tmp_path / "dependency")
    source = _repo(tmp_path / "testbed")
    package = source / "packages/worker"
    package.mkdir(parents=True)
    _git(
        source, "-c", "protocol.file.allow=always", "submodule", "add", "-q",
        str(dependency), "packages/worker/vendor/dependency",
    )
    _git(source, "commit", "-qam", "add worker dependency")
    env = _env(container_namespace(package), Path(container_namespace(tmp_path)), "container-nested-module")
    try:
        assert await env.setup() == env.workspace
        assert env.workspace.endswith("/packages/worker")
        assert await env.read_file("vendor/dependency/tracked.txt") == "base\n"
        assert await env.get_diff() == ""
    finally:
        await env.cleanup()
    assert _git(source, "status", "--porcelain") == ""
    assert _git(source, "worktree", "list", "--porcelain").count("worktree ") == 1

async def test_nested_source_rejects_worktree_root_inside_true_repository(
    container_namespace, tmp_path,
):
    source = _repo(tmp_path / "testbed")
    package = source / "packages/worker"
    package.mkdir(parents=True)
    (package / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(source, "add", "packages/worker/module.py")
    _git(source, "commit", "-qm", "add worker")
    env = ContainerWorktreeEnvironment(
        container_id=CONTAINER_ID,
        repository_root=container_namespace(package),
        worktree_root=container_namespace(source / "children"),
        branch_name="nested-contained",
    )
    try:
        with pytest.raises(ValueError, match="outside"):
            await env.setup()
    finally:
        await env.cleanup()
    assert not (source / "children").exists()
    assert _git(source, "branch", "--list", "nested-contained") == ""
