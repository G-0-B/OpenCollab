"""A workflow reads the same excluded Git paths inside a candidate lease."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.tools import builtin_tools


def _repository(root: Path) -> None:
    root.mkdir()
    (root / "source.py").write_text("value = 1\n")
    for args in (
        ("init", "-q"), ("add", "source.py"),
        ("-c", "user.name=OpenCollab Tests", "-c", "user.email=tests@example.invalid", "commit", "-qm", "base"),
    ):
        subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)

@pytest.mark.parametrize("injected_path", ["test_probe.py", "test [probe]*.py"])
@pytest.mark.parametrize("source_changed", [False, True])
async def test_public_workflow_diff_exclusions_survive_candidate_execution(
    tmp_path, monkeypatch, injected_path, source_changed,
):
    edits = [
        ("source.py", "value = 2\n" if source_changed else "value = 1\n"),
        (injected_path, "def test_probe():\n    assert True\n"),
        ("test pother.py", "def test_other():\n    assert True\n"),
    ]

    class ControlledModel:
        def __init__(self, **kwargs):
            self.calls = 0

        async def complete(self, messages, **kwargs):
            self.calls += 1
            calls = []
            if self.calls <= len(edits):
                path, content = edits[self.calls - 1]
                calls = [{"id": f"call-{self.calls}", "type": "function", "function": {
                    "name": "file_write", "arguments": json.dumps({
                        "path": path, "mode": "create", "content": content,
                    }),
                }}]
            return LLMResponse(
                content=None if calls else "finished", tool_calls=calls,
                usage=Usage(4, 2), finish_reason="tool_calls" if calls else "stop",
            )

        async def aclose(self):
            pass

    monkeypatch.setattr("opencollab.bootstrap.container.LLMClient", ControlledModel)
    exclusions = (injected_path, "test pother.py")

    async def flow(ctx, _inputs):
        await ctx.agent("edit source and add tests", tools=builtin_tools("file_write"))
        return {
            "all": await ctx.diff(),
            "filtered": await ctx.diff(exclude_paths=exclusions),
            "literal_filtered": await ctx.diff(exclude_paths=(injected_path,)),
            "changed": await ctx.source_changed(exclusions),
        }

    async def outer(ctx, inputs):
        return (await ctx.candidate_workflow(flow, inputs, label="candidate")).output

    observations = []
    for name, entry in (("direct", flow), ("candidate", outer)):
        root = tmp_path / name
        _repository(root)
        result = await OpenCollab(
            root, model="controlled", provider="openai",
            api_key="local-fixture",  # pragma: allowlist secret
        ).workflow(entry, budget=100_000, trace=False, timeout=20)
        assert result.ok, result.reason
        observations.append(result.output)

    direct, candidate = observations
    for observation in observations:
        assert injected_path in observation["all"]
        assert "test pother.py" in observation["all"]
        assert "test pother.py" in observation["literal_filtered"]
        assert injected_path not in observation["literal_filtered"]
        assert observation["changed"] is source_changed
        if source_changed:
            assert "+value = 2" in observation["filtered"]
            assert "test_probe" not in observation["filtered"]
            assert "test pother.py" not in observation["filtered"]
            assert "+value = 2" in observation["all"]
        else:
            assert observation["filtered"] == ""
    assert direct["filtered"] == candidate["filtered"]
    assert direct["literal_filtered"] == candidate["literal_filtered"]
