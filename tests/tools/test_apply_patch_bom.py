"""Line insertions keep a UTF-8 signature at the start of executable files."""

import codecs
import subprocess
import sys

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.tools.apply_patch import ApplyPatchTool
from opencollab.application.tool_execution import ToolRuntime


@pytest.fixture
async def runtime(tmp_path):
    environment = LocalEnvironment(str(tmp_path))
    yield ToolRuntime(environment, None, None)
    await environment.cleanup()

async def insert(runtime, **params):
    return await ApplyPatchTool().execute_with_runtime({
        "path": "app.py", "mode": "line_replace", "start_line": 1, "end_line": 0,
        "expected_str": "", "new_str": "import os", **params,
    }, runtime)

def execute(target):
    return subprocess.run(
        [sys.executable, "-B", str(target)], capture_output=True, text=True,
    )

@pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
@pytest.mark.parametrize("terminated", [False, True])
@pytest.mark.parametrize("prefix", [b"", codecs.BOM_UTF8])
async def test_first_line_insertion_keeps_python_executable(
    tmp_path, runtime, newline, terminated, prefix,
):
    target = tmp_path / "app.py"
    body = newline.join([
        "value = '\ufeffinside'".encode(), "print(value == '\ufeffinside')".encode(),
    ]) + (newline if terminated else b"")
    target.write_bytes(prefix + body)
    target.chmod(0o755)
    before = execute(target)
    assert before.returncode == 0, before.stderr

    output = await insert(runtime)

    assert output.startswith("Applied line_replace"), output
    after = execute(target)
    assert after.returncode == 0, after.stderr
    assert after.stdout == before.stdout == "True\n"
    assert target.read_bytes() == prefix + b"import os" + newline + body
    assert target.stat().st_mode & 0o777 == 0o755

@pytest.mark.parametrize("prefix", [b"", codecs.BOM_UTF8])
async def test_insertion_into_empty_body_keeps_the_signature(tmp_path, runtime, prefix):
    target = tmp_path / "app.py"
    target.write_bytes(prefix)

    output = await insert(runtime, new_str="print(42)")

    assert output.startswith("Applied line_replace"), output
    after = execute(target)
    assert after.returncode == 0, after.stderr
    assert after.stdout == "42\n"
    assert target.read_bytes() == prefix + b"print(42)"

@pytest.mark.parametrize("body", [b"", b"print(42)\r\n"])
async def test_empty_insertion_is_rejected_without_moving_the_signature(tmp_path, runtime, body):
    target = tmp_path / "app.py"
    original = codecs.BOM_UTF8 + body
    target.write_bytes(original)
    before = target.stat()

    output = await insert(runtime, new_str="")

    assert "patch produced no changes" in output
    assert target.read_bytes() == original
    assert target.stat().st_mtime_ns == before.st_mtime_ns

@pytest.mark.parametrize("prefix", ["", "\ufeff"])
@pytest.mark.parametrize("relocate", [False, True])
async def test_exact_first_line_guard_still_matches_and_replaces_the_signature(
    tmp_path, runtime, prefix, relocate,
):
    target = tmp_path / "app.py"
    target.write_bytes(codecs.BOM_UTF8 + b"print(42)\r\n")
    updated = prefix + "print(43)"

    output = await insert(
        runtime, end_line=0 if relocate else 1, expected_str="\ufeffprint(42)",
        new_str=updated, relocate_expected=relocate,
    )

    assert output.startswith("Applied line_replace"), output
    assert target.read_bytes() == (updated + "\r\n").encode()
    after = execute(target)
    assert after.returncode == 0, after.stderr
    assert after.stdout == "43\n"

@pytest.mark.parametrize("params", [
    {"expected_str": "print(42)"},
    {"end_line": 1, "expected_str": "print(42)"},
    {"start_line": 0},
    {"end_line": 2},
])
async def test_rejected_bom_edits_preserve_all_bytes(tmp_path, runtime, params):
    target = tmp_path / "app.py"
    original = codecs.BOM_UTF8 + b"print(42)\n"
    target.write_bytes(original)

    output = await insert(runtime, **params)

    assert output.startswith("Error applying patch"), output
    assert target.read_bytes() == original

async def test_later_insertion_keeps_a_literal_bom_inside_the_body(tmp_path, runtime):
    target = tmp_path / "app.py"
    original = codecs.BOM_UTF8 + "print(42)\n\ufeffsecond\n".encode()
    target.write_bytes(original)

    output = await insert(runtime, start_line=2, end_line=1)

    assert output.startswith("Applied line_replace"), output
    assert target.read_bytes() == codecs.BOM_UTF8 + "print(42)\nimport os\n\ufeffsecond\n".encode()
