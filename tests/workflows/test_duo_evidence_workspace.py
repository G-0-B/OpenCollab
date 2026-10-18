"""Duo's own retained files preserve source changes made by the caller."""

from __future__ import annotations

import json
import subprocess

import pytest

from opencollab import OpenCollab
from opencollab.adapters.llm.client import LLMClient
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.bootstrap._workflow_runtime_session import build_workflow_context
from opencollab.builtin_workflows.duo import _source_evidence_root


@pytest.mark.parametrize("subdirectory", [False, True])
@pytest.mark.parametrize("evidence_location", ["repository", "workspace"])
@pytest.mark.parametrize("mutation", [
    None, "source", "evidence-sibling", "root-source", "sibling-source", "similar-prefix", "same-name", "cross-rename",
])
async def test_duo_excludes_only_its_owned_evidence_directory(
    tmp_path, monkeypatch, subdirectory, evidence_location, mutation,
):
    monkeypatch.delenv("OPENCOLLAB_UNBOUNDED_LIMITS", raising=False)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", "")
    repo = tmp_path / "repo"
    repo.mkdir()
    workspace = repo / "src" if subdirectory else repo
    workspace.mkdir(exist_ok=True)
    source = workspace / "source.txt"
    source.write_text("initial\n")
    evidence = (repo if evidence_location == "repository" else workspace) / "evidence[run]"
    evidence.mkdir()
    sibling = evidence / "caller.txt"
    sibling.write_text("caller evidence\n")
    shared = repo / "shared.txt"
    shared.write_text("shared source\n")
    other = repo / "other/source.txt"
    other.parent.mkdir()
    other.write_text("sibling source\n")
    for args in [("init", "-q"), ("add", "."),
                 ("-c", "user.name=Test User", "-c", "user.email=test@example.invalid",
                  "-c", "commit.gpgsign=false", "commit", "-qm", "\u521d\u59cb\u5316\u6d4b\u8bd5\u4ed3\u5e93")]:
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    note = repo / "notes.txt"
    note.write_text("existing untracked user note\n")
    calls = 0

    async def complete(llm, messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls in (1, 3):
            content = "candidate A\n" if calls == 1 else "candidate B\n"
            name, arguments = "file_write", {"path": "source.txt", "mode": "create", "content": content}
        elif calls in (2, 4):
            return LLMResponse(content="finished", usage=Usage(1, 1), finish_reason="stop")
        else:
            if mutation == "source":
                source.write_text("caller source change\n")
            elif mutation == "evidence-sibling":
                sibling.write_text("new caller evidence\n")
            elif mutation == "root-source":
                shared.write_text("new root source\n")
            elif mutation == "sibling-source":
                other.unlink()
            elif mutation in {"similar-prefix", "same-name"}:
                owned = next(evidence.glob("duo-evidence-*"))
                adjacent = (owned.with_name(owned.name + "-caller") if mutation == "similar-prefix"
                            else repo / "other" / owned.name)
                adjacent.mkdir()
                (adjacent / "caller.txt").write_text("new caller evidence\n")
            elif mutation == "cross-rename":
                owned = next(evidence.glob("duo-evidence-*"))
                subprocess.run(["git", "mv", "--", str(shared), str(owned / "moved.txt")],
                               cwd=repo, check=True, capture_output=True)
            name, arguments = "structured_output", {
                "winner": "B", "requirements_complete": True,
                "requirements": [{
                    "requirement": "source.txt must contain candidate B",
                    "a_coverage": "not_covered", "b_coverage": "covered",
                    "a_evidence": ["source.txt contains candidate A"],
                    "b_evidence": ["source.txt contains candidate B"],
                }],
                "rationale": "B satisfies the requested source value",
            }
        return LLMResponse(tool_calls=[{
            "id": f"call-{calls}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }], usage=Usage(1, 1), finish_reason="tool_calls")

    monkeypatch.setattr(LLMClient, "complete", complete)
    client = OpenCollab(
        workspace, model="test-model", provider="openai",
        api_key="test-key",  # pragma: allowlist secret
    )
    result = await client.workflow("duo", {
        "goal": "source.txt must contain candidate B", "candidate_evidence_dir": str(evidence),
    }, budget=100_000, trace=False)

    directories = [path for path in evidence.glob("duo-evidence-*") if (path / "A/candidate.diff").is_file()]
    assert len(directories) == 1
    assert (directories[0] / "A/candidate.diff").is_file()
    assert (directories[0] / "B/candidate.diff").is_file()
    assert "duo-evidence-" not in (directories[0] / "A/candidate.diff").read_text()
    assert "duo-evidence-" not in (directories[0] / "B/candidate.diff").read_text()
    assert note.read_text() == "existing untracked user note\n"
    assert calls == 5
    if mutation is None:
        assert result.status == "completed"
        assert result.output["adopted"] == "B"
        assert source.read_text() == "candidate B\n"
        assert sibling.read_text() == "caller evidence\n"
    else:
        assert result.status == "failed"
        assert "source worktree changed before candidate adoption" in result.reason
        assert source.read_text() == ("caller source change\n" if mutation == "source" else "initial\n")
        expected_sibling = "new caller evidence\n" if mutation == "evidence-sibling" else "caller evidence\n"
        assert sibling.read_text() == expected_sibling
        if mutation == "root-source":
            assert shared.read_text() == "new root source\n"
        elif mutation == "sibling-source":
            assert not other.exists()
        elif mutation == "cross-rename":
            assert not shared.exists()
            assert (directories[0] / "moved.txt").read_text() == "shared source\n"
        elif mutation in {"similar-prefix", "same-name"}:
            assert next(repo.rglob("duo-evidence-*/caller.txt")).read_text() == "new caller evidence\n"

def test_remote_workspace_name_does_not_identify_a_host_directory(tmp_path):
    from types import SimpleNamespace

    (tmp_path / ".git").mkdir()
    context = SimpleNamespace(workspace_root=str(tmp_path), host_workspace=None)
    assert _source_evidence_root(context, str(tmp_path / "evidence")) is None

@pytest.mark.parametrize("injected", [False, True])
async def test_host_source_metadata_requires_the_known_source_backend(tmp_path, injected):
    context = build_workflow_context(
        cfg={"model": "test-model", "provider": "openai"}, workspace=str(tmp_path),
        candidate_workspace=object() if injected else None,
    )
    try:
        assert context.host_workspace == (None if injected else str(tmp_path))
    finally:
        await context.release_isolated_workspaces()

async def test_unknown_source_snapshot_stops_duo_before_candidate_generation(tmp_path):
    from types import SimpleNamespace

    from opencollab.builtin_workflows.duo import duo

    (tmp_path / ".git").mkdir()
    calls = []

    async def diff(**options):
        calls.append(options)
        return None

    context = SimpleNamespace(host_workspace=str(tmp_path), diff=diff)
    with pytest.raises(RuntimeError, match="source diff unavailable for Duo evidence"):
        await duo(context, {"goal": "update source", "candidate_evidence_dir": str(tmp_path / "evidence")})
    assert len(calls) == 1

@pytest.mark.parametrize("available", [False, True])
async def test_outside_workspace_evidence_keeps_unknown_source_visible(tmp_path, available):
    from types import SimpleNamespace

    from opencollab.builtin_workflows.duo import duo

    (tmp_path / ".git").mkdir()
    workspace = tmp_path / "package"
    workspace.mkdir()
    reads = []

    async def source_diff():
        reads.append("source")
        return None

    context = SimpleNamespace(
        host_workspace=str(workspace),
        _candidate_workspace=SimpleNamespace(source_diff=source_diff) if available else None,
    )
    with pytest.raises(RuntimeError, match="source diff unavailable for Duo evidence"):
        await duo(context, {"goal": "update source", "candidate_evidence_dir": str(tmp_path / "evidence")})
    assert reads == (["source"] if available else [])
