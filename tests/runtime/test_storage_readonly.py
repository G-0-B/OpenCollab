"""Read-only session exports and stable journal coordination."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from opencollab.adapters import storage as storage_module
from opencollab.adapters.storage import SessionStore
from opencollab.bootstrap import load_session
from opencollab.domain.agent import Agent
from opencollab.sdk import OpenCollab
from tests.support.session_characterization_test_support import FakeLLMClient

pytestmark = pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses file permissions")

_MESSAGES = [
    {"role": "system", "content": "system"},
    {"role": "user", "content": "restore this export"},
]

@pytest.mark.parametrize("file_mode", [0o444, 0o644])
def test_public_read_and_restore_accept_readonly_standalone_export(
    tmp_path: Path, file_mode: int,
) -> None:
    source = tmp_path / "source.json"
    SessionStore().save(str(source), _MESSAGES, meta={"snapshot_version": 1, "aid": 7})
    export = tmp_path / "export"
    export.mkdir()
    path = export / "session.json"
    shutil.copyfile(source, path)
    path.chmod(file_mode)
    original = path.read_bytes()
    export.chmod(0o555)
    try:
        assert OpenCollab.read_session_snapshot(path)["messages"] == _MESSAGES
        session = load_session(
            str(path), agent=Agent(name="lead", system_prompt="system"),
            llm=FakeLLMClient(),
        )
        try:
            assert session.messages == _MESSAGES
            assert session.state.aid == 7
        finally:
            asyncio.run(session.aclose())
        assert list(export.iterdir()) == [path]
        assert path.read_bytes() == original
        assert stat.S_IMODE(path.stat().st_mode) == file_mode
        assert stat.S_IMODE(export.stat().st_mode) == 0o555
    finally:
        export.chmod(0o755)

def test_readonly_existing_lock_replays_journal_without_changing_files(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    store = SessionStore()
    store.save(str(path), _MESSAGES)
    store.append_snapshot_delta(
        str(path), sequence=1, replace_from=2,
        messages=[{"role": "assistant", "content": "journal update"}], meta={},
    )
    files = list(tmp_path.iterdir())
    before = {file.name: (file.read_bytes(), file.stat().st_ino) for file in files}
    for file in files:
        file.chmod(0o444)
    tmp_path.chmod(0o555)
    try:
        snapshot = OpenCollab.read_session_snapshot(path)
        assert snapshot["messages"] == [
            *_MESSAGES, {"role": "assistant", "content": "journal update"},
        ]
        assert {file.name: (file.read_bytes(), file.stat().st_ino) for file in tmp_path.iterdir()} == before
        assert all(stat.S_IMODE(file.stat().st_mode) == 0o444 for file in files)
        assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o555
    finally:
        tmp_path.chmod(0o755)

def test_existing_journal_without_lock_still_requires_coordination(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    store = SessionStore()
    store.save(str(path), _MESSAGES)
    store.append_snapshot_delta(
        str(path), sequence=1, replace_from=2,
        messages=[{"role": "assistant", "content": "journal update"}], meta={},
    )
    lock = tmp_path / "session.json.journal.lock"
    lock.unlink()
    tmp_path.chmod(0o555)
    try:
        with pytest.raises(PermissionError):
            OpenCollab.read_session_snapshot(path)
        assert not lock.exists()
    finally:
        tmp_path.chmod(0o755)

@pytest.mark.parametrize("journal_mode", [0o444, 0o000])
def test_readonly_lock_preserves_journal_errors(tmp_path: Path, journal_mode: int) -> None:
    path = tmp_path / "session.json"
    SessionStore().save(str(path), _MESSAGES)
    journal = tmp_path / "session.json.journal"
    journal.write_text("not-json\n", encoding="utf-8")
    journal.chmod(journal_mode)
    lock = tmp_path / "session.json.journal.lock"
    lock.chmod(0o444)
    tmp_path.chmod(0o555)
    try:
        if journal_mode:
            with pytest.raises(ValueError, match="journal record at line 1"):
                OpenCollab.read_session_snapshot(path)
        else:
            with pytest.raises(PermissionError):
                OpenCollab.read_session_snapshot(path)
    finally:
        tmp_path.chmod(0o755)
        journal.chmod(0o644)

def test_standalone_readonly_export_preserves_missing_and_invalid_errors(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"messages": None}), encoding="utf-8")
    tmp_path.chmod(0o555)
    try:
        with pytest.raises(ValueError, match="messages.*list"):
            OpenCollab.read_session_snapshot(invalid)
        with pytest.raises(FileNotFoundError):
            OpenCollab.read_session_snapshot(tmp_path / "missing.json")
        assert list(tmp_path.iterdir()) == [invalid]
    finally:
        tmp_path.chmod(0o755)

@pytest.mark.parametrize("lock_kind", ["unreadable", "symlink", "fifo"])
def test_existing_invalid_lock_cannot_fall_back_to_standalone_read(
    tmp_path: Path, lock_kind: str,
) -> None:
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"messages": _MESSAGES}), encoding="utf-8")
    lock = tmp_path / "session.json.journal.lock"
    if lock_kind == "unreadable":
        lock.touch(mode=0o000)
    elif lock_kind == "symlink":
        lock.symlink_to(path)
    else:
        os.mkfifo(lock)
    tmp_path.chmod(0o555)
    try:
        with pytest.raises(OSError):
            OpenCollab.read_session_snapshot(path)
    finally:
        tmp_path.chmod(0o755)
        if lock_kind == "unreadable":
            lock.chmod(0o600)

def test_existing_write_only_lock_retains_read_access(tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    SessionStore().save(str(path), _MESSAGES)
    lock = tmp_path / "session.json.journal.lock"
    lock.chmod(0o200)
    try:
        assert OpenCollab.read_session_snapshot(path)["messages"] == _MESSAGES
    finally:
        lock.chmod(0o600)

@pytest.mark.parametrize("error_number", [errno.EROFS, errno.EIO, errno.ENOSPC])
def test_standalone_fallback_requires_a_lock_creation_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_number: int,
) -> None:
    path = tmp_path / "session.json"
    path.write_text(json.dumps({"messages": _MESSAGES}), encoding="utf-8")

    def fail_open(_path: str) -> None:
        raise OSError(error_number, "lock creation failed")

    monkeypatch.setattr(storage_module, "open_regular_text_append", fail_open)
    if error_number == errno.EROFS:
        assert OpenCollab.read_session_snapshot(path)["messages"] == _MESSAGES
    else:
        with pytest.raises(OSError) as error:
            OpenCollab.read_session_snapshot(path)
        assert error.value.errno == error_number
    assert list(tmp_path.iterdir()) == [path]
