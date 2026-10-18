"""Public source exclusions use the exposed workspace within a full repo patch."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.patches import patch_paths
from opencollab.tools import builtin_tools
from tests.workflows.test_workflow_candidate_workspace import _git, _repository


class _WriteFiles:
    def __init__(self, files):
        self.files = files
        self.calls = 0

    async def complete(self, messages, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(tool_calls=[{
                "id": f"write-{index}", "type": "function", "function": {
                    "name": "file_write", "arguments": json.dumps({
                        "path": path, "mode": "create", "content": content,
                    }),
                },
            } for index, (path, content) in enumerate(self.files.items())],
                usage=Usage(4, 2), finish_reason="tool_calls")
        return LLMResponse(content="Files written.", usage=Usage(4, 2), finish_reason="stop")

@pytest.mark.parametrize("subdirectory", [False, True])
@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("tracked_test", [False, True])
@pytest.mark.parametrize("source_edit", [None, "source.py", "tests/injected 1x.py"])
async def test_public_exclusions_keep_source_changes_and_literal_neighbors(
    tmp_path, monkeypatch, subdirectory, candidate, tracked_test, source_edit,
):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", "")
    repo = _repository(tmp_path)
    workspace = repo / "package" if subdirectory else repo
    workspace.mkdir(exist_ok=True)
    (workspace / "source.py").write_text("value = 1\n")
    injected = "tests/injected [1]*?.py"
    if tracked_test:
        (workspace / "tests").mkdir()
        (workspace / injected).write_text("def test_original():\n    assert True\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "--allow-empty", "-m", "Add workspace fixtures")
    index_before = (repo / ".git/index").read_bytes()
    files = {injected: "def test_injected():\n    assert True\n"}
    if source_edit:
        files[source_edit] = "value = 2\n"
    model = _WriteFiles(files)

    async def flow(ctx, inputs):
        await ctx.agent("Write the requested files.", tools=builtin_tools("file_write"))
        return {
            "changed": await ctx.source_changed(),
            "source_changed": await ctx.source_changed([injected]),
            "patch": await ctx.diff(exclude_paths=[injected]),
        }

    async def outer(ctx, inputs):
        result = await ctx.candidate_workflow(flow, inputs, label="source-exclusion")
        assert result.lifecycle_errors == ()
        return {**result.output, "candidate_patch": result.diff}

    result = await OpenCollab(workspace, model="local-scripted-model").workflow(
        outer if candidate else flow, llm=model, trace=False, budget=10000,
        artifacts=tmp_path / "records" if tracked_test else None, max_steps=4, timeout=20,
    )
    assert result.ok, result.reason
    assert model.calls == 2
    assert result.output["changed"] is True
    assert result.output["source_changed"] is bool(source_edit)
    prefix = "package/" if subdirectory else ""
    assert patch_paths(result.output["patch"]) == ([prefix + source_edit] if source_edit else [])
    if candidate:
        assert prefix + injected in patch_paths(result.output["candidate_patch"])
        assert (workspace / "source.py").read_text() == "value = 1\n"
        assert (workspace / injected).exists() is tracked_test
    else:
        assert (workspace / injected).read_text() == files[injected]
    assert (repo / ".git/index").read_bytes() == index_before
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1

@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("artifacts", [False, True])
@pytest.mark.parametrize("outside_path", ["shared.py", "sibling/new.py"])
async def test_exclusions_keep_root_and_sibling_changes(tmp_path, candidate, artifacts, outside_path):
    repo = _repository(tmp_path)
    workspace = repo / "package"
    workspace.mkdir()
    (workspace / "source.py").write_text("value = 1\n")
    (repo / "shared.py").write_text("shared = 1\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Add repository fixtures")
    index_before = (repo / ".git/index").read_bytes()

    async def flow(ctx, inputs):
        current = Path(ctx.workspace_root)
        test = current / "tests/injected.py"
        test.parent.mkdir()
        test.write_text("def test_injected():\n    assert True\n")
        outside = current.parent / outside_path
        outside.parent.mkdir(exist_ok=True)
        outside.write_text("shared = 2\n")
        return {
            "changed": await ctx.source_changed(),
            "source_changed": await ctx.source_changed(["tests/injected.py"]),
            "patch": await ctx.diff(exclude_paths=["tests/injected.py"]),
        }

    async def outer(ctx, inputs):
        result = await ctx.candidate_workflow(flow, inputs, label="outside-source")
        assert result.lifecycle_errors == ()
        return result.output

    result = await OpenCollab(workspace).workflow(
        outer if candidate else flow, trace=False,
        artifacts=tmp_path / "records" if artifacts else None,
    )
    assert result.ok, result.reason
    assert result.output["changed"] is True
    assert result.output["source_changed"] is True
    assert patch_paths(result.output["patch"]) == [outside_path]
    assert (repo / ".git/index").read_bytes() == index_before
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    if candidate:
        assert (repo / "shared.py").read_text() == "shared = 1\n"
        assert not (workspace / "tests/injected.py").exists()

@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("artifacts", [False, True])
@pytest.mark.parametrize("invalid_path", ["../shared.py", "/shared.py", "tests/\0injected.py"])
async def test_public_exclusions_reject_paths_outside_workspace(tmp_path, candidate, artifacts, invalid_path):
    repo = _repository(tmp_path)
    workspace = repo / "package"
    workspace.mkdir()

    async def flow(ctx, inputs):
        return {
            "source_changed": await ctx.source_changed([invalid_path]),
            "patch": await ctx.diff(exclude_paths=[invalid_path]),
        }

    async def outer(ctx, inputs):
        result = await ctx.candidate_workflow(flow, inputs, label="invalid-exclusion")
        return result.output

    result = await OpenCollab(workspace).workflow(
        outer if candidate else flow, trace=False, artifacts=tmp_path / "records" if artifacts else None,
    )
    assert result.ok, result.reason
    assert result.output == {"source_changed": None, "patch": None}

@pytest.mark.parametrize("subdirectory", [False, True])
@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("artifacts", [False, True])
@pytest.mark.parametrize("names", [
    ("original.py", "created.py"),
    ("original [1]*? \u96ea.py", "created [1]*? \u96ea.py"),
])
async def test_default_diff_uses_repository_paths_for_all_source_changes(
    tmp_path, monkeypatch, subdirectory, candidate, artifacts, names,
):
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", "")
    repo = _repository(tmp_path)
    workspace = repo / "package with spaces" if subdirectory else repo
    workspace.mkdir(exist_ok=True)
    tracked, untracked = names
    (workspace / tracked).write_text("original = 1\n")
    _git(repo, "add", ".")
    _git(repo, "-c", "commit.gpgsign=false", "commit", "-m", "Add diagnostic fixtures")
    index_before = (repo / ".git/index").read_bytes()
    model = _WriteFiles({tracked: "original = 2\n", untracked: "created = 1\n"})
    outside_paths = ["root [1]*? \u96ea.py", "sibling/new [1]*? \u96ea.py"]

    async def flow(ctx, inputs):
        await ctx.agent("Update the tracked file and create the new file.", tools=builtin_tools("file_write"))
        current = Path(ctx.workspace_root)
        root = current.parent if subdirectory else current
        for outside_path in outside_paths:
            outside = root / outside_path
            outside.parent.mkdir(exist_ok=True)
            outside.write_text("external_created = 1\n")
        return {"diagnostic": await ctx.diff(), "filtered": await ctx.diff(["unused-harness.py"])}

    async def outer(ctx, inputs):
        result = await ctx.candidate_workflow(flow, inputs, label="diagnostic-paths")
        assert result.lifecycle_errors == ()
        return {**result.output, "candidate_patch": result.diff}

    result = await OpenCollab(workspace, model="local-scripted-model").workflow(
        outer if candidate else flow, llm=model, trace=False, budget=10000,
        artifacts=tmp_path / "records" if artifacts else None, max_steps=4, timeout=20,
    )
    assert result.ok, result.reason
    assert model.calls == 2
    prefix = "package with spaces/" if subdirectory else ""
    expected = {prefix + tracked, prefix + untracked, *outside_paths}
    for key in ("diagnostic", "filtered", "candidate_patch"):
        if key in result.output:
            assert set(patch_paths(result.output[key])) == expected
            assert "+original = 2" in result.output[key]
            assert "+created = 1" in result.output[key]
            assert "+external_created = 1" in result.output[key]
    assert ("[Working tree status]" in result.output["diagnostic"]) is (not candidate and not artifacts)
    assert (repo / ".git/index").read_bytes() == index_before
    assert _git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    if candidate:
        assert (workspace / tracked).read_text() == "original = 1\n"
        assert not (workspace / untracked).exists()
