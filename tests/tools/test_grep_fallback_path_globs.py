"""The optional grep backend reports unavailable path-glob semantics explicitly."""

from __future__ import annotations

import shutil

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.application.tool_execution import ToolRuntime
from opencollab.tools import builtin_tools


@pytest.fixture
def grep_only(tmp_path, monkeypatch):
    grep = shutil.which("grep")
    if grep is None:
        pytest.skip("grep is unavailable")
    binaries = tmp_path / "binaries"
    binaries.mkdir()
    (binaries / "grep").symlink_to(grep)
    monkeypatch.setenv("PATH", str(binaries))
    source = tmp_path / "src" / "main.py"
    source.parent.mkdir()
    source.write_text("NEEDLE source\n", encoding="utf-8")

@pytest.mark.parametrize(
    "glob",
    ["src/*.py", "src/**/*.py", "source files/*.py", "**/", "!**/*.py", "**/.*", "**/[^a]*"],
)
async def test_unavailable_path_glob_is_an_error_instead_of_no_matches(tmp_path, grep_only, glob):
    environment = LocalEnvironment(str(tmp_path))
    tool, = builtin_tools("grep")
    try:
        output = await tool.execute_with_runtime(
            {"pattern": "NEEDLE", "glob": glob},
            ToolRuntime(environment, None, None),
        )
    finally:
        await environment.cleanup()

    assert output.startswith("Error:")
    assert "ripgrep" in output
    assert "path" in output and "filename glob" in output
    assert "No matches" not in output

@pytest.mark.parametrize(
    "glob, pattern, found",
    [
        ("*.py", "NEEDLE", True),
        ("**/*.py", "NEEDLE", True),
        ("**/main.py", "NEEDLE", True),
        (None, "NEEDLE", True),
        ("*.py", "ABSENT", False),
        ("**/*.py", "ABSENT", False),
        ("**/main.py", "ABSENT", False),
    ],
)
async def test_supported_fallback_search_retains_real_match_status(tmp_path, grep_only, glob, pattern, found):
    environment = LocalEnvironment(str(tmp_path))
    tool, = builtin_tools("grep")
    try:
        output = await tool.execute_with_runtime(
            {"pattern": pattern, "path": "src", "glob": glob},
            ToolRuntime(environment, None, None),
        )
    finally:
        await environment.cleanup()

    if found:
        assert "main.py:1:NEEDLE source" in output
    else:
        assert output == "No matches found for pattern: ABSENT"

@pytest.mark.parametrize("glob", ["*.{py,pyi}", "!*.pyi", "*.{py,{pyi,pyx}}"])
async def test_rg_only_filename_globs_report_missing_backend(tmp_path, grep_only, glob):
    (tmp_path / "src" / "types.pyi").write_text("NEEDLE types\n", encoding="utf-8")
    environment = LocalEnvironment(str(tmp_path))
    tool, = builtin_tools("grep")
    try:
        output = await tool.execute_with_runtime(
            {"pattern": "NEEDLE", "path": "src", "glob": glob},
            ToolRuntime(environment, None, None),
        )
    finally:
        await environment.cleanup()

    assert output.startswith("Error:")
    assert "ripgrep" in output and "unavailable" in output
    assert "separate searches" in output
    assert "No matches" not in output

async def test_escaped_braces_remain_a_literal_filename_glob(tmp_path, grep_only):
    (tmp_path / "src" / "literal{name}.py").write_text("NEEDLE literal\n", encoding="utf-8")
    environment = LocalEnvironment(str(tmp_path))
    tool, = builtin_tools("grep")
    try:
        output = await tool.execute_with_runtime(
            {"pattern": "NEEDLE", "path": "src", "glob": r"literal\{name\}.py"},
            ToolRuntime(environment, None, None),
        )
    finally:
        await environment.cleanup()

    assert "literal{name}.py:1:NEEDLE literal" in output
