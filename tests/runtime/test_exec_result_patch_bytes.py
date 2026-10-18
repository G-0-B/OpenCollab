"""Opaque Git stdout survives the existing local and Docker result wrappers."""

from __future__ import annotations

import shlex
import sys

from opencollab.adapters import _env_docker as docker_module
from opencollab.adapters._env_base import ExecResult
from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters._env_process import ProcessResult
from opencollab.adapters.env import DockerEnvironment
from tests.support.docker_env_test_support import CONTAINER_ID, FakeDocker, _patch

_PAYLOAD = b"id,label\n1,caf\xe9\n"

async def test_local_command_keeps_patch_stdout_bytes(tmp_path):
    environment = LocalEnvironment(str(tmp_path))
    code = f"import sys; sys.stdout.buffer.write(bytes.fromhex({_PAYLOAD.hex()!r}))"
    try:
        result = await environment.exec_cmd(shlex.join((sys.executable, "-c", code)), timeout=5)
        assert result.returncode == 0
        assert result.stdout_bytes == _PAYLOAD
        assert result.stdout == _PAYLOAD.decode("utf-8", errors="replace")
        assert result == ExecResult(0, result.stdout, "")
    finally:
        await environment.cleanup()

async def test_docker_command_keeps_patch_stdout_bytes(monkeypatch):
    def respond(command, kwargs):
        assert docker_module._EXEC_WRAPPER in command
        assert kwargs["output_limit"] > len(_PAYLOAD)
        return ProcessResult(0, _PAYLOAD, b"")

    fake = FakeDocker(respond)
    _patch(monkeypatch, fake)
    environment = DockerEnvironment(container_id=CONTAINER_ID)
    environment._attached_bound = True
    result = await environment.exec_cmd("git diff --binary", timeout=5)
    assert result.returncode == 0
    assert result.stdout_bytes == _PAYLOAD
    assert result.stdout == _PAYLOAD.decode("utf-8", errors="replace")
    assert len(fake.calls) == 1
