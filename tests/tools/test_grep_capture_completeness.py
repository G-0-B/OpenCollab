"""Search capture loss cannot splice records into a fabricated matching line."""

from __future__ import annotations

import shutil

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.application.tool_execution import ToolRuntime
from opencollab.tools import builtin_tools


@pytest.fixture(params=["rg", "grep"])
def backend(request, tmp_path, monkeypatch):
    if request.param == "rg":
        if shutil.which("rg") is None:
            pytest.skip("ripgrep is unavailable")
    else:
        grep = shutil.which("grep")
        if grep is None:
            pytest.skip("grep is unavailable")
        binaries = tmp_path / "binaries"
        binaries.mkdir()
        (binaries / "grep").symlink_to(grep)
        monkeypatch.setenv("PATH", str(binaries))
    return request.param

@pytest.mark.parametrize("payload_chars", [100_000, 700_000])
async def test_large_matching_records_preserve_capture_completeness(tmp_path, backend, payload_chars):
    first = 'NEEDLE first ' + 'a' * payload_chars + ' FIRST_RECORD_TAIL'
    second = 'NEEDLE second ' + 'b' * payload_chars + ' SECOND_RECORD_TAIL'
    (tmp_path / "records.jsonl").write_text(first + "\n" + second + "\n", encoding="utf-8")
    environment = LocalEnvironment(str(tmp_path))
    tool, = builtin_tools("grep")
    try:
        output = await tool.execute_with_runtime(
            {"path": "records.jsonl", "pattern": "NEEDLE", "max_results": 1},
            ToolRuntime(environment, None, None),
        )
        if payload_chars == 700_000:
            assert output.startswith(f"Error: {backend} search output was incomplete")
            assert "bytes dropped" in output
            assert "path or glob" in output
            assert "FIRST_RECORD_TAIL" not in output
            assert "SECOND_RECORD_TAIL" not in output
        else:
            assert "FIRST_RECORD_TAIL" in output
            assert "SECOND_RECORD_TAIL" not in output
            assert "1 more matching lines" in output
    finally:
        await environment.cleanup()

async def test_many_short_matches_report_capture_loss(tmp_path, backend):
    (tmp_path / "records.txt").write_text("NEEDLE record\n" * 100_000, encoding="utf-8")
    environment = LocalEnvironment(str(tmp_path))
    tool, = builtin_tools("grep")
    try:
        output = await tool.execute_with_runtime(
            {"path": "records.txt", "pattern": "NEEDLE", "max_results": 50},
            ToolRuntime(environment, None, None),
        )
        assert output.startswith(f"Error: {backend} search output was incomplete")
        assert "bytes dropped" in output
        assert "more matching lines not shown" not in output
    finally:
        await environment.cleanup()

async def test_narrowed_search_keeps_complete_short_output(tmp_path, backend):
    (tmp_path / "records.txt").write_text("NEEDLE first\nother\nNEEDLE second\n", encoding="utf-8")
    environment = LocalEnvironment(str(tmp_path))
    tool, = builtin_tools("grep")
    try:
        output = await tool.execute_with_runtime(
            {"path": "records.txt", "pattern": "NEEDLE", "max_results": 2},
            ToolRuntime(environment, None, None),
        )
        prefix = "records.txt:" if output.startswith("records.txt:") else ""
        assert output == f"{prefix}1:NEEDLE first\n{prefix}3:NEEDLE second"
    finally:
        await environment.cleanup()
