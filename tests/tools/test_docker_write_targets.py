"""Docker create writes replace file targets and reject directory targets."""

from __future__ import annotations

import pytest

from opencollab.adapters.tools.fs import FileWriteTool
from opencollab.application.tool_execution import ToolRuntime
from tests.support.docker_native_edits_support import LocalDockerTransport, docker_environment


@pytest.mark.parametrize("kind", ["directory", "directory_symlink"])
@pytest.mark.parametrize("overwrite", [False, True])
async def test_create_rejects_directory_target_without_hidden_write(tmp_path, kind, overwrite):
    target = tmp_path / "requested.txt"
    directory = target if kind == "directory" else tmp_path / "referent"
    directory.mkdir()
    (directory / "user-data.txt").write_text("retained data")
    if kind == "directory_symlink":
        target.symlink_to(directory, target_is_directory=True)
    transport = LocalDockerTransport(tmp_path, pair_reads=False)
    environment = docker_environment(tmp_path, transport)

    with pytest.raises(OSError, match="write verification failed"):
        await FileWriteTool().execute_with_runtime(
            {"path": str(target), "mode": "create", "content": "expected content", "overwrite": overwrite},
            ToolRuntime(environment, None, None),
        )

    assert target.is_dir()
    assert target.is_symlink() == (kind == "directory_symlink")
    assert list(directory.iterdir()) == [directory / "user-data.txt"]
    assert (directory / "user-data.txt").read_text() == "retained data"
    assert not list(tmp_path.rglob("*.opencollab-write-*.tmp"))
    assert not environment._active_execs

@pytest.mark.parametrize("kind", ["missing", "regular_file", "file_symlink", "dangling_symlink"])
@pytest.mark.parametrize("overwrite", [False, True])
async def test_create_replaces_target_file_and_keeps_symlink_referent(tmp_path, kind, overwrite):
    target = tmp_path / "requested.txt"
    referent = tmp_path / "referent.txt"
    if kind == "regular_file":
        target.write_text("prior content")
    elif kind == "file_symlink":
        referent.write_text("referent content")
        target.symlink_to(referent)
    elif kind == "dangling_symlink":
        target.symlink_to(referent)
    transport = LocalDockerTransport(tmp_path, pair_reads=False)
    environment = docker_environment(tmp_path, transport)

    output = await FileWriteTool().execute_with_runtime(
        {"path": str(target), "mode": "create", "content": "expected content", "overwrite": overwrite},
        ToolRuntime(environment, None, None),
    )

    assert output.startswith("Created/wrote"), output
    assert target.is_file() and not target.is_symlink()
    assert target.read_text() == "expected content"
    if kind == "file_symlink":
        assert referent.read_text() == "referent content"
    else:
        assert not referent.exists()
    assert not list(tmp_path.rglob("*.opencollab-write-*.tmp"))
    assert not environment._active_execs

async def test_write_receipt_verifies_final_target_content(tmp_path):
    transport = LocalDockerTransport(tmp_path, pair_reads=False)
    environment = docker_environment(tmp_path, transport)
    environment._command_prefix = 'mv() { command mv "$@" && printf tampered > "$4"; }; export -f mv'

    with pytest.raises(OSError, match="write verification failed"):
        await environment.write_file(str(tmp_path / "requested.txt"), "expected content")

    assert (tmp_path / "requested.txt").read_text() == "tampered"
    assert not environment._active_execs

async def test_failed_publish_keeps_original_and_discards_temporary(tmp_path):
    target = tmp_path / "requested.txt"
    target.write_text("original content")
    transport = LocalDockerTransport(tmp_path, pair_reads=False)
    environment = docker_environment(tmp_path, transport)
    environment._command_prefix = 'mv() { return 1; }; export -f mv'

    with pytest.raises(OSError, match="write verification failed"):
        await environment.write_file(str(target), "replacement content")

    assert target.read_text() == "original content"
    assert not list(tmp_path.rglob("*.opencollab-write-*.tmp"))
    assert not environment._active_execs
