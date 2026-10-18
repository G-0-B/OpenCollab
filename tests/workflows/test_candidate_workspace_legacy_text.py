"""Git candidate patches preserve legacy text bytes and readable UTF-8 diffs."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from opencollab import OpenCollab
from opencollab.adapters._env_local import LocalEnvironment
from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.tools import builtin_tools
from tests.workflows.test_workflow_candidate_workspace import _git, _repository

_FILENAME = "\u5ba2\u6237\u8bb0\u5f55.csv"
_LEGACY = b"id,label\n1,caf\xe9 \x80\n"

@pytest.mark.parametrize("tracked", [False, True], ids=["untracked", "tracked"])
async def test_candidate_inherits_and_adopts_legacy_text_bytes(tmp_path, tracked):
    repo = _repository(tmp_path)
    source = repo / _FILENAME
    if tracked:
        source.write_bytes(b"original caf\xe9\n")
        _git(repo, "add", _FILENAME)
        _git(repo, "commit", "-m", "legacy text fixture")
    source.write_bytes(_LEGACY)
    source_index = _git(repo, "ls-files", "--stage")
    info_attributes = repo / ".git" / "info" / "attributes"
    info_attributes.write_text("* diff\n", encoding="utf-8")
    environment = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(environment)
    lease = None
    try:
        lease = await workspace.acquire("legacy-text")
        candidate = Path(lease.candidate_workspace)
        assert (candidate / _FILENAME).read_bytes() == _LEGACY
        assert source.read_bytes() == _LEGACY
        assert await lease.diff() == ""
        updated = b"id,label\n1,pi\xf1ata \x80\n"
        (candidate / _FILENAME).write_bytes(updated)
        await lease.environment.write_file("source.py", "value = 2\n")
        patch = await lease.diff()
        assert "GIT binary patch" in patch
        assert "+value = 2" in patch
        patch.encode("utf-8")
        await workspace.adopt(patch)
        assert source.read_bytes() == updated
        assert (repo / "source.py").read_text() == "value = 2\n"
        assert _git(repo, "ls-files", "--stage") == source_index
        assert info_attributes.read_text() == "* diff\n"
    finally:
        if lease is not None:
            await lease.cleanup()
        await environment.cleanup()

class _CsvWriter:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    async def complete(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            code = f"from pathlib import Path; Path({_FILENAME!r}).write_bytes(bytes.fromhex({self.payload.hex()!r}))"
            command = shlex.join((sys.executable, "-c", code))
            return LLMResponse(tool_calls=[{
                "id": "export-csv", "type": "function", "function": {
                    "name": "bash", "arguments": json.dumps({"command": command}),
                },
            }], usage=Usage(10, 5), finish_reason="tool_calls")
        return LLMResponse(content="CSV exported.", usage=Usage(10, 5), finish_reason="stop")

@pytest.mark.parametrize("payload", [_LEGACY, "id,label\n1,café\n".encode(), b"id,label\x00\xff\n"],
                         ids=["legacy-text", "utf-8-text", "nul-binary"])
async def test_public_candidate_adoption_preserves_exported_csv_bytes(tmp_path, payload):
    repo = _repository(tmp_path)
    model = _CsvWriter(payload)

    async def flow(ctx, inputs):
        candidate = await ctx.candidate_agent(
            "Export a CSV using the required encoding.", tools=builtin_tools("bash", headless=False), label="csv",
        )
        await ctx.adopt_candidate(candidate)
        return {"output": candidate.output, "patch": candidate.diff, "lifecycle_errors": candidate.lifecycle_errors}

    result = await OpenCollab(repo, model="local-csv-model").workflow(
        flow, llm=model, budget=20_000, max_steps=4, timeout=15, trace=False,
    )
    assert result.ok
    assert result.output["output"] == "CSV exported."
    assert not result.output["lifecycle_errors"]
    assert (repo / _FILENAME).read_bytes() == payload
    assert model.calls == 2
    if payload == "id,label\n1,café\n".encode():
        assert "+1,café" in result.output["patch"]

@pytest.mark.parametrize("change", ["delete", "rename", "executable"])
async def test_legacy_text_patch_keeps_file_change_metadata(tmp_path, change):
    repo = _repository(tmp_path)
    source = repo / _FILENAME
    original = b"id,label\n" + b"1,caf\xe9 \x80\n" * 20
    source.write_bytes(original)
    _git(repo, "add", _FILENAME)
    _git(repo, "commit", "-m", "legacy metadata fixture")
    environment = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(environment)
    lease = None
    try:
        lease = await workspace.acquire("legacy-metadata")
        candidate = Path(lease.candidate_workspace)
        target = candidate / _FILENAME
        name = _FILENAME
        if change == "delete":
            target.unlink()
        else:
            if change == "rename":
                name = "\u65b0\u5ba2\u6237\u8bb0\u5f55.csv"
                target = target.rename(candidate / name)
            target.write_bytes(original + b"2,pi\xf1ata\n")
            if change == "executable":
                target.chmod(0o755)
        patch = await lease.diff()
        assert "GIT binary patch" in patch
        if change == "rename":
            assert "rename from" in patch and "rename to" in patch
        if change == "executable":
            assert "new mode 100755" in patch
        await workspace.adopt(patch)
        if change == "delete":
            assert not source.exists()
        else:
            assert (repo / name).read_bytes() == original + b"2,pi\xf1ata\n"
            if change == "rename":
                assert not source.exists()
            if change == "executable":
                assert (repo / name).stat().st_mode & 0o111 == 0o111
    finally:
        if lease is not None:
            await lease.cleanup()
        await environment.cleanup()

async def test_legacy_text_patch_uses_git_line_ending_normalization(tmp_path):
    repo = _repository(tmp_path)
    attributes = b"*.csv text eol=crlf\n"
    (repo / ".gitattributes").write_bytes(attributes)
    source = repo / _FILENAME
    source.write_bytes(b"id,label\r\n1,caf\xe9\r\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "normalized legacy text fixture")
    source.write_bytes(b"id,label\r\n1,pi\xf1ata\r\n")
    source_index = _git(repo, "ls-files", "--stage")
    environment = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(environment)
    lease = None
    try:
        lease = await workspace.acquire("normalized-legacy")
        candidate = Path(lease.candidate_workspace)
        assert (candidate / _FILENAME).read_bytes() == source.read_bytes()
        (candidate / _FILENAME).write_bytes(b"id,label\r\n1,pi\xf1ata\r\n2,caf\xe9\r\n")
        await workspace.adopt(await lease.diff())
        assert source.read_bytes() == b"id,label\r\n1,pi\xf1ata\r\n2,caf\xe9\r\n"
        assert (repo / ".gitattributes").read_bytes() == attributes
        assert _git(repo, "ls-files", "--stage") == source_index
    finally:
        if lease is not None:
            await lease.cleanup()
        await environment.cleanup()

@pytest.mark.parametrize("tracked", [False, True], ids=["untracked", "tracked"])
async def test_legacy_text_patch_keeps_git_clean_and_smudge_filters(tmp_path, tracked):
    repo = _repository(tmp_path)
    filter_script = tmp_path / "csv_filter.py"
    filter_script.write_text(
        "import sys\n"
        "payload = sys.stdin.buffer.read()\n"
        "sys.stdout.buffer.write(payload.removeprefix(b'WORK:') if sys.argv[1] == 'clean' else b'WORK:' + payload)\n",
        encoding="utf-8",
    )
    for mode in ("clean", "smudge"):
        _git(repo, "config", f"filter.csv.{mode}", shlex.join((sys.executable, str(filter_script), mode)))
    attributes = b"*.csv filter=csv\n"
    (repo / ".gitattributes").write_bytes(attributes)
    source = repo / _FILENAME
    source.write_bytes(b"WORK:original caf\xe9\n")
    _git(repo, "add", ".gitattributes")
    if tracked:
        _git(repo, "add", _FILENAME)
    _git(repo, "commit", "-m", "filtered legacy text fixture")
    source.write_bytes(b"WORK:source pi\xf1ata\n")
    source_index = _git(repo, "ls-files", "--stage")
    environment = LocalEnvironment(str(repo))
    workspace = EnvCandidateWorkspace(environment)
    lease = None
    try:
        lease = await workspace.acquire("filtered-legacy")
        candidate = Path(lease.candidate_workspace)
        assert (candidate / _FILENAME).read_bytes() == source.read_bytes()
        (candidate / _FILENAME).write_bytes(b"WORK:candidate caf\xe9\n")
        await workspace.adopt(await lease.diff())
        assert source.read_bytes() == b"WORK:candidate caf\xe9\n"
        assert (repo / ".gitattributes").read_bytes() == attributes
        assert _git(repo, "ls-files", "--stage") == source_index
    finally:
        if lease is not None:
            await lease.cleanup()
        await environment.cleanup()
