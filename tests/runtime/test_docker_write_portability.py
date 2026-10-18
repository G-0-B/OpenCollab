"""BSD byte-count compatibility preserved from the maintenance release."""

from __future__ import annotations

import os
import stat
import subprocess

import pytest

from opencollab.adapters._env_base import ExecResult
from opencollab.adapters.env import DockerEnvironment

CONTAINER_ID = "a" * 64

async def test_verified_write_accepts_bsd_wc_padding(monkeypatch, tmp_path) -> None:
    """A BSD-style padded ``wc -c`` count still verifies the write."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_wc = fake_bin / "wc"
    fake_wc.write_text("#!/bin/sh\nprintf '      5\\n'\n", encoding="utf-8")
    fake_wc.chmod(0o755)
    child_env = {"PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    async def run_locally(command, *, timeout, input_bytes=None):
        completed = subprocess.run(
            ["bash", "-c", command],
            input=input_bytes,
            capture_output=True,
            cwd=workspace,
            env=child_env,
            check=False,
        )
        return ExecResult(
            completed.returncode,
            completed.stdout.decode(),
            completed.stderr.decode(),
        )

    async def discard_temporary(_temporary):
        return None

    env = DockerEnvironment(workspace=str(workspace), container_id=CONTAINER_ID)
    env._attached_bound = True
    monkeypatch.setattr(env, "_exec", run_locally)
    monkeypatch.setattr(env, "_discard_write_temporary", discard_temporary)

    await env.write_file(str(workspace / "padded.txt"), "hello")
    assert (workspace / "padded.txt").read_text(encoding="utf-8") == "hello"

@pytest.fixture
def local_shell_docker(monkeypatch, tmp_path):
    async def run_locally(command, *, timeout, input_bytes=None):
        completed = subprocess.run(
            ["bash", "-c", command], input=input_bytes, capture_output=True, cwd=tmp_path, check=False,
        )
        return ExecResult(completed.returncode, completed.stdout.decode(), completed.stderr.decode())

    env = DockerEnvironment(workspace=str(tmp_path), container_id=CONTAINER_ID)
    env._attached_bound = True
    monkeypatch.setattr(env, "_exec", run_locally)
    return env

@pytest.mark.parametrize("mode", [0o755, 0o640, 0o444])
async def test_atomic_write_preserves_existing_regular_file_permissions(local_shell_docker, tmp_path, mode):
    target = tmp_path / "script.sh"
    target.write_text("#!/bin/sh\nprintf old\n")
    target.chmod(mode)

    await local_shell_docker.write_file(str(target), "#!/bin/sh\nprintf updated\n")

    assert stat.S_IMODE(target.stat().st_mode) == mode
    assert target.read_text() == "#!/bin/sh\nprintf updated\n"
    if mode & stat.S_IXUSR:
        executed = subprocess.run([str(target)], capture_output=True, check=True, text=True)
        assert executed.stdout == "updated"

async def test_atomic_write_keeps_new_file_private(local_shell_docker, tmp_path):
    target = tmp_path / "new.txt"

    await local_shell_docker.write_file(str(target), "new content")

    assert stat.S_IMODE(target.stat().st_mode) == 0o600

async def test_atomic_write_replaces_symlink_without_changing_referent(local_shell_docker, tmp_path):
    referent = tmp_path / "referent.sh"
    referent.write_text("original")
    referent.chmod(0o755)
    target = tmp_path / "alias.sh"
    target.symlink_to(referent)

    await local_shell_docker.write_file(str(target), "replacement")

    assert not target.is_symlink()
    assert target.read_text() == "replacement"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert referent.read_text() == "original"
    assert stat.S_IMODE(referent.stat().st_mode) == 0o755
