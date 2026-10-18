"""Machine-consumed candidate patches ignore Git display preferences."""

from pathlib import Path

import pytest

from opencollab.adapters.candidate_workspace import EnvCandidateWorkspace
from opencollab.adapters.env import LocalEnvironment
from opencollab.application.workflow import WorkflowContext
from tests.workflows.test_workflow_candidate_workspace import _Factory, _git, _repository


@pytest.mark.parametrize("settings", [
    {},
    {"diff.noprefix": "true"},
    {"color.ui": "always"},
    {"diff.srcPrefix": "before/custom/", "diff.dstPrefix": "after/custom/"},
    {"diff.noprefix": "true", "diff.mnemonicPrefix": "true", "color.ui": "always"},
], ids=["default", "no-prefix", "color", "custom-prefix", "combined"])
@pytest.mark.parametrize("dirty", [False, True])
async def test_candidate_preserves_delivery_with_display_settings(tmp_path, settings, dirty):
    repo = _repository(tmp_path)
    for key, value in settings.items():
        _git(repo, "config", key, value)
    binary = b"\x00\xffexisting untracked bytes\n"
    if dirty:
        (repo / "source.py").write_text("value = 10\n")
        (repo / "notes.bin").write_bytes(binary)
    original = (repo / "source.py").read_bytes()
    env = LocalEnvironment(str(repo))
    factory = _Factory(env)
    context = WorkflowContext(factory, candidate_workspace=EnvCandidateWorkspace(env))
    try:
        candidate = await context.candidate_agent("edit source", label="format")
        assert candidate.output == "done"
        assert "diff --git a/source.py b/source.py" in candidate.diff
        assert "\x1b[" not in candidate.diff
        assert (repo / "source.py").read_bytes() == original
        await context.adopt_candidate(candidate)
        assert (repo / "source.py").read_text() == "value = 2\n"
        if dirty:
            assert (repo / "notes.bin").read_bytes() == binary
        for key, value in settings.items():
            assert _git(repo, "config", "--get", key).strip() == value
    finally:
        for candidate_env in factory.environments:
            await candidate_env.cleanup()
            if Path(candidate_env.workspace).exists():
                _git(repo, "worktree", "remove", "--force", candidate_env.workspace)
        await env.cleanup()

@pytest.mark.parametrize("dirty", [False, True])
async def test_candidate_keeps_context_when_git_display_context_is_zero(tmp_path, dirty):
    repo = _repository(tmp_path)
    original = "alpha\nbeta\ngamma\ndelta\n"
    (repo / "source.py").write_text(original)
    _git(repo, "commit", "-am", "record multiline source")
    _git(repo, "config", "diff.context", "0")
    if dirty:
        original = original.replace("beta", "source edit")
        (repo / "source.py").write_text(original)
    updated = original.replace("gamma", "candidate edit")
    env = LocalEnvironment(str(repo))
    backend = EnvCandidateWorkspace(env)
    lease = None
    try:
        lease = await backend.acquire("context")
        assert await lease.environment.read_file("source.py") == original
        await lease.environment.write_file("source.py", updated)
        patch = await lease.diff()
        assert (repo / "source.py").read_text() == original
        await backend.adopt(patch)
        assert (repo / "source.py").read_text() == updated
        assert _git(repo, "config", "--get", "diff.context").strip() == "0"
    finally:
        if lease is not None:
            await lease.cleanup()
        await env.cleanup()
