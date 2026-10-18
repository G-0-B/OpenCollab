"""Execute the README's workflow example against a real repository and tests."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.tools import builtin_tools

README = Path(__file__).resolve().parents[2] / "opencollab" / "README.md"
CODING_TOOLS = {"bash", "file_read", "file_write", "apply_patch", "git_diff", "grep"}
INITIAL = "def divide(left, right):\n    return 0\n"
DRAFT = "def divide(left, right):\n    return abs(left / right)\n"
REVIEWED = "def divide(left, right):\n    return left / right\n"

@pytest.fixture
def documented_workflow():
    section = README.read_text().split("### Workflow authoring\n", 1)[1]
    source = section.split("```python\n", 1)[1].split("```", 1)[0]
    namespace = {"__name__": "readme_workflow_example"}
    exec(compile(source, str(README), "exec"), namespace)
    return namespace["implement_and_review"]

@pytest.fixture
def project(tmp_path):
    (tmp_path / "calc.py").write_text(INITIAL)
    (tmp_path / "test_calc.py").write_text(
        "from calc import divide\n\n"
        "def test_positive():\n    assert divide(6, 2) == 3\n\n"
        "def test_negative():\n    assert divide(-6, 2) == -3\n"
    )
    for args in (
        ("init", "-q"),
        ("config", "user.email", "tests@example.com"),
        ("config", "user.name", "OpenCollab Tests"),
        ("add", "."),
        ("commit", "-qm", "\u6d4b\u8bd5\u521d\u59cb\u72b6\u6001"),
    ):
        subprocess.run(["git", *args], cwd=tmp_path, capture_output=True, check=True)
    return tmp_path

def _pytest_command(target):
    return (
        f"PYTHONDONTWRITEBYTECODE=1 {shlex.quote(sys.executable)} -B "
        f"-m pytest -rA -q -p no:cacheprovider {target}"
    )

def _run_native_tests(project):
    return subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-rA", "-q", "-p", "no:cacheprovider", "test_calc.py"],
        cwd=project,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
    )

class EditingLLM:
    """Use only advertised tools and leave a negative-input gap for review."""

    def __init__(self):
        self.round_tools = {}
        self.prompts = {}
        self.steps = {}
        self.outputs = {}

    async def complete(self, messages, tools=None, **_kwargs):
        prompt = next(message["content"] for message in messages if message["role"] == "user")
        stage = "review" if prompt.startswith("Review the implementation") else "implement"
        names = {tool["function"]["name"] for tool in tools or ()}
        self.round_tools.setdefault(stage, names)
        self.prompts.setdefault(stage, prompt)
        self.outputs[stage] = [
            message["content"] for message in messages if message["role"] == "tool"
        ]
        step = self.steps.get(stage, 0)
        self.steps[stage] = step + 1
        actions = []
        if "file_read" in names:
            actions.append(("file_read", {"path": "calc.py"}))
        if "file_write" in names:
            actions.append(("file_write", {
                "path": "calc.py", "mode": "create", "overwrite": True,
                "content": DRAFT if stage == "implement" else REVIEWED,
            }))
        if "bash" in names:
            target = "test_calc.py::test_positive" if stage == "implement" else "test_calc.py"
            actions.append(("bash", {"command": _pytest_command(target)}))
        if step >= len(actions):
            return LLMResponse(
                content=f"{stage} finished", finish_reason="stop", usage=Usage(4, 2),
            )
        name, arguments = actions[step]
        return LLMResponse(
            tool_calls=[{
                "id": f"{stage}-{step}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
            finish_reason="tool_calls", usage=Usage(4, 2),
        )

async def _run_example(project, documented_workflow, llm, **inputs):
    return await OpenCollab(project, model="test-model").workflow(
        documented_workflow,
        {"task": "Fix division for positive and negative inputs.", **inputs},
        llm=llm,
        budget=10_000,
        trace=False,
    )

@pytest.mark.parametrize("caller_tools", [False, True], ids=["default-tools", "caller-tools"])
async def test_readme_workflow_implements_then_reviews_with_native_tests(
    project, documented_workflow, caller_tools,
):
    before = _run_native_tests(project)
    assert before.returncode == 1
    assert "2 failed" in before.stdout
    llm = EditingLLM()
    inputs = {"allow_unisolated_shell": True}
    if caller_tools:
        inputs["tools"] = builtin_tools("file_write", "bash", headless=False)
    result = await _run_example(project, documented_workflow, llm, **inputs)

    assert result.ok, result.reason
    assert (project / "calc.py").read_text() == REVIEWED
    after = _run_native_tests(project)
    assert after.returncode == 0, after.stdout + after.stderr
    assert "2 passed" in after.stdout
    expected = {"file_write", "bash"} if caller_tools else CODING_TOOLS
    assert llm.round_tools == {"implement": expected, "review": expected}
    assert "1 passed" in llm.outputs["implement"][-1]
    assert "2 passed" in llm.outputs["review"][-1]
    assert "Exit code: 0" in llm.outputs["review"][-1]
    assert "implement finished" in llm.prompts["review"]
    assert "+    return abs(left / right)" in llm.prompts["review"]
    assert not (project / ".pytest_cache").exists()
    assert not (project / "__pycache__").exists()

@pytest.mark.parametrize("allow_host_shell", [None, False, "true", 1])
async def test_readme_workflow_requires_explicit_true_for_host_shell(
    project, documented_workflow, allow_host_shell,
):
    llm = EditingLLM()
    inputs = {} if allow_host_shell is None else {"allow_unisolated_shell": allow_host_shell}
    result = await _run_example(project, documented_workflow, llm, **inputs)

    assert result.ok, result.reason
    assert llm.round_tools == {"implement": CODING_TOOLS, "review": CODING_TOOLS}
    for stage in ("implement", "review"):
        assert "bash is disabled" in llm.outputs[stage][-1]
        assert "OS process sandbox" in llm.outputs[stage][-1]

async def test_readme_workflow_preserves_empty_caller_tools(project, documented_workflow):
    llm = EditingLLM()
    result = await _run_example(
        project, documented_workflow, llm, tools=[], allow_unisolated_shell=True,
    )

    assert result.ok, result.reason
    assert llm.round_tools == {"implement": set(), "review": set()}
    assert (project / "calc.py").read_text() == INITIAL
