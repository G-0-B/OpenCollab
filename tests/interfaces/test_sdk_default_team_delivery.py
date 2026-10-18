"""The shipped team adopts real completed changes before fresh verification."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import opencollab.adapters.cli.main as cli_main
from opencollab import OpenCollab
from opencollab.adapters.worktree_pool import WorktreePool
from opencollab.application._scheduler_constants import WORKTREE_DIFF_MAX_CHARS
from tests.support.provider_sdk_http import completion_http_response, http_response, install_sdk_transport
from tests.support.worktree_delivery_support import CONTAINER_ID, git, make_repo
from tests.support.worktree_delivery_support import worktree_transport as worktree_transport


@pytest.mark.parametrize("backend", ["local", "container"])
async def test_default_team_executes_with_initialized_dependency(
    worktree_transport, tmp_path, monkeypatch, backend,
):
    from opencollab.environments import attach_container, local_environment

    source = make_repo(tmp_path / "source")
    dependency = make_repo(tmp_path / "dependency")
    git(source, "-c", "protocol.file.allow=always", "submodule", "add", str(dependency), "vendor/library")
    git(source, "config", "-f", ".gitmodules", "submodule.vendor/library.url", "https://example.invalid/library.git")
    git(source, "commit", "-qam", "record dependency")
    monkeypatch.setenv("GIT_ALLOW_PROTOCOL", "file")
    steps = {"analyst": 0, "coder": 0}
    executions = []
    workspaces = []
    acquire = WorktreePool.acquire

    async def observe_acquire(pool, role):
        child = await acquire(pool, role)
        workspaces.append(Path(child.workspace))
        code = "from pathlib import Path; assert Path('vendor/library/initial.txt').read_text() == 'initial\\n'"
        result = await child.exec_cmd(f"python3 -B -c {shlex.quote(code)} && echo DEPENDENCY_PASS")
        assert result.returncode == 0, result.stderr
        assert result.stdout == "DEPENDENCY_PASS\n"
        executions.append(result.stdout)
        return child

    monkeypatch.setattr(WorktreePool, "acquire", observe_acquire)

    def handler(request):
        messages = json.loads(request.content)["messages"]
        role = "analyst" if "OpenCollab's Analyst" in str(messages[0]["content"]) else "coder"
        steps[role] += 1
        if steps[role] == 2:
            latest = next(str(message["content"]) for message in reversed(messages) if message["role"] == "tool")
            assert ("initial" if role == "coder" else "DEPENDENCY_PASS") in latest, latest
            return completion_http_response(request, text="DEPENDENCY_PASS")
        if role == "analyst":
            name, arguments = "spawn_agent", {"role": "coder", "task": "Read and execute the initialized dependency."}
        else:
            name, arguments = "file_read", {"path": "vendor/library/initial.txt"}
        return http_response(request, 200, json={
            "id": f"{role}-1", "object": "chat.completion", "created": 0, "model": "offline",
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": role, "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments),
                }}],
            }, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    install_sdk_transport(monkeypatch, "openai", handler)
    environment = (
        attach_container(container_id=CONTAINER_ID, workspace=str(source))
        if backend == "container" else local_environment(source)
    )
    result = await OpenCollab(
        source, environment=environment, model="offline", api_key="controlled-test",  # pragma: allowlist secret
        base_url="https://controlled.invalid/v1", config={"llm_max_retries": 0},
    ).team(
        "Read and execute the initialized dependency.", use_worktrees=True, trace=False,
        artifacts=tmp_path / "artifacts", timeout=20, budget=20_000,
    )
    assert result.ok, (result.reason, result.error)
    assert len(executions) == 1, (steps, result.output)
    assert all(not path.exists() for path in workspaces)

@pytest.mark.parametrize("mode", ["git", "plain", "shared", "exported", "custom"])
async def test_default_team_delivers_two_rounds_through_real_tools(tmp_path, monkeypatch, mode):
    source = tmp_path / "project"
    source.mkdir()
    (source / "answer.py").write_text("def answer():\n    return 1\n")
    if mode != "plain":
        for args in (("init", "-q"), ("config", "user.name", "Tests"),
                     ("config", "user.email", "tests@example.invalid"),
                     ("add", "."), ("-c", "commit.gpgsign=false", "commit", "-qm", "\u521d\u59cb\u5316")):
            subprocess.run(["git", *args], cwd=source, check=True, capture_output=True)
    team_file = None
    if mode in {"exported", "custom"}:
        team_file = tmp_path / "team.yaml"
        initialized = CliRunner().invoke(cli_main.app, ["team", "init", str(team_file)])
        assert initialized.exit_code == 0, initialized.output
        if mode == "custom":
            import yaml
            team = yaml.safe_load(team_file.read_text())
            team["roles"]["analyst"]["tools"].remove("adopt_agent")
            team_file.write_text(yaml.safe_dump(team))
    rounds = {"analyst": 0, "coder": 0, "tester": 0}
    reads = []
    adopted = []
    source_before_adoption = []

    def handler(request):
        body = json.loads(request.content)
        messages = body["messages"]
        system = str(messages[0]["content"])
        role = (
            "analyst" if "OpenCollab's Analyst" in system
            else "coder" if "OpenCollab's Coder" in system else "tester"
        )
        rounds[role] += 1
        n = rounds[role]
        latest = next((str(m["content"]) for m in reversed(messages) if m["role"] == "tool"), "")
        tools = {item["function"]["name"] for item in body.get("tools", [])}
        name = None
        args = {}
        if role == "analyst":
            assert ("adopt_agent" in tools) == (mode != "custom")
            if n in (1, 4):
                name, args = "spawn_agent", {"role": "coder", "task": f"Implement round {1 if n == 1 else 2}."}
            elif n in (2, 5):
                if mode == "custom":
                    return completion_http_response(request, text="Returned complete candidate diff.\n" + latest)
                match = re.search(r"Completed coder aid=(\d+)", latest)
                assert match, latest
                aid = int(match.group(1))
                adopted.append(aid)
                source_before_adoption.append((source / "answer.py").read_text())
                name, args = "adopt_agent", {"aid": aid}
            elif n in (3, 6):
                assert ("Adopted coder" in latest or "shared delivery workspace" in latest), latest
                name, args = "spawn_agent", {
                    "role": "tester", "task": f"Verify round {1 if n == 3 else 2} from actual files.",
                }
            else:
                assert "PASS" in latest
                return completion_http_response(request, text="Completed two verified rounds.")
        elif role == "coder":
            step = (n - 1) % 3 + 1
            iteration = (n - 1) // 3 + 1
            if step == 1:
                name, args = "file_read", {"path": "answer.py"}
            elif step == 2:
                before, after = (1, 42) if iteration == 1 else (42, 43)
                assert f"return {before}" in latest, latest
                name, args = "apply_patch", {
                    "path": "answer.py", "mode": "line_replace", "start_line": 2, "end_line": 2,
                    "expected_str": f"    return {before}\n", "new_str": f"    return {after}\n",
                }
            else:
                assert "Error" not in latest, latest
                return completion_http_response(request, text="Implemented answer.py.")
        else:
            if n % 2:
                name, args = "file_read", {"path": "answer.py"}
            else:
                value = 42 if n == 2 else 43
                assert f"return {value}" in latest, latest
                reads.append(latest)
                return completion_http_response(request, text=f"PASS\nRead the adopted return {value} implementation.")
        return http_response(request, 200, json={
            "id": f"response-{role}-{n}", "object": "chat.completion", "created": 1, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": f"{role}-{n}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}],
            }}], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        })

    install_sdk_transport(monkeypatch, "openai", handler)
    result = await OpenCollab(source, model="local-fixture", api_key="controlled-test",  # pragma: allowlist secret
                             base_url="https://controlled.invalid/v1", config={"llm_max_retries": 0}).team(
        "Implement and verify two rounds.", config=team_file, use_worktrees=mode != "shared", trace=False,
        artifacts=tmp_path / "artifacts", max_steps=15, timeout=30, budget=100_000,
    )
    assert result.ok, (result.reason, result.error)
    if mode == "custom":
        assert (source / "answer.py").read_text() == "def answer():\n    return 1\n"
        assert "return 42" in result.output and "diff --git" in result.output
    else:
        assert len(reads) == 2 and len(adopted) == 2 and adopted[0] != adopted[1]
        expected_before = [1, 42] if mode != "shared" else [42, 43]
        assert all(f"return {value}" in text for value, text in zip(expected_before, source_before_adoption))
        executed = subprocess.run(
            [sys.executable, "-B", "-c",
             "from answer import answer; assert answer() == 43; print('SDK_DELIVERY_PASS')"],
            cwd=source, check=True, capture_output=True, text=True,
        )
        assert executed.stdout == "SDK_DELIVERY_PASS\n"

@pytest.mark.parametrize("save_artifacts", [False, True])
async def test_large_conflict_is_recoverable_after_a_fresh_coder_succeeds(tmp_path, monkeypatch, save_artifacts):
    source = tmp_path / "project"
    source.mkdir()
    (source / "answer.py").write_text("def answer():\n    return 1\n")
    (source / "second.py").write_text("value = 0\n")
    for args in (("init", "-q"), ("add", "."), ("-c", "user.name=Tests", "-c",
                  "user.email=tests@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm",
                  "\u521d\u59cb\u5316")):
        subprocess.run(["git", *args], cwd=source, check=True, capture_output=True)
    implementation = "    return 42\n# " + "complete candidate " * 1000 + "\n"
    counts = {"analyst": 0, "coder": 0, "tester": 0}
    recovery = []

    def handler(request):
        body = json.loads(request.content)
        messages = body["messages"]
        system = str(messages[0]["content"])
        role = (
            "analyst" if "OpenCollab's Analyst" in system
            else "coder" if "OpenCollab's Coder" in system else "tester"
        )
        counts[role] += 1
        step = counts[role]
        latest = next((str(m["content"]) for m in reversed(messages) if m["role"] == "tool"), "")
        name = None
        args = {}
        if role == "analyst":
            if step in (1, 3):
                if step == 3:
                    location = re.search(r"full candidate retained at ([^\n]+)", latest)
                    assert location, latest
                    patch_file = Path(location.group(1))
                    assert patch_file.is_file() and patch_file.stat().st_size > WORKTREE_DIFF_MAX_CHARS
                    recovery.append(patch_file)
                    assert (source / "second.py").read_text() == "value = 0\n"
                name, args = "spawn_agent", {"role": "coder", "task": "Implement first candidate." if step == 1
                                              else "Revise the current conflicted implementation."}
            elif step in (2, 4):
                found = re.search(r"Completed coder aid=(\d+)", latest)
                assert found, latest
                if step == 2:
                    assert "chars truncated" in latest
                    (source / "answer.py").write_text("def answer():\n    return 99\n")
                name, args = "adopt_agent", {"aid": int(found.group(1))}
            elif step == 5:
                assert "Adopted coder" in latest
                name, args = "spawn_agent", {"role": "tester", "task": "Verify answer 43 and second 3."}
            else:
                assert "PASS" in latest
                return completion_http_response(request, text="Completed the fresh revision and verification.")
        elif role == "coder":
            within = (step - 1) % 4 + 1
            second = step > 4
            if within == 1:
                name, args = "file_read", {"path": "answer.py"}
            elif within == 2:
                before = 99 if second else 1
                assert f"return {before}" in latest
                name, args = "apply_patch", {
                    "path": "answer.py", "mode": "line_replace", "start_line": 2, "end_line": 2,
                    "expected_str": f"    return {before}\n",
                    "new_str": "    return 43\n" if second else implementation,
                }
            elif within == 3:
                name, args = "apply_patch", {"path": "second.py", "mode": "line_replace", "start_line": 1,
                                            "end_line": 1, "expected_str": "value = 0\n",
                                            "new_str": "value = 3\n" if second else "value = 2\n"}
            else:
                assert "Error" not in latest
                return completion_http_response(request, text="Completed the candidate.")
        elif step == 1:
            name, args = "file_read", {"path": "answer.py"}
        elif step == 2:
            assert "return 43" in latest
            name, args = "file_read", {"path": "second.py"}
        else:
            assert "value = 3" in latest
            return completion_http_response(request, text="PASS\nRead both revised implementation files.")
        return http_response(request, 200, json={
            "id": f"response-{role}-{step}", "object": "chat.completion", "created": 1, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{"id": f"{role}-{step}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}],
            }}], "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        })

    install_sdk_transport(monkeypatch, "openai", handler)
    artifacts = tmp_path / "artifacts" if save_artifacts else None
    result = await OpenCollab(source, model="local-fixture", api_key="controlled-test",  # pragma: allowlist secret
                             base_url="https://controlled.invalid/v1", config={"llm_max_retries": 0}).team(
        "Implement, recover from an adoption conflict, and verify.", artifacts=artifacts, trace=False,
        max_steps=15, timeout=30, budget=100_000,
    )
    assert result.ok, (result.reason, result.error)
    subprocess.run([sys.executable, "-B", "-c", "from answer import answer; from second import value; "
                    "assert (answer(), value) == (43, 3)"], cwd=source, check=True)
    assert len(recovery) == 1
    patch = recovery[0].read_text()
    assert "full candidate " not in patch
    assert "complete candidate " * 1000 in patch and "value = 2" in patch
    if save_artifacts:
        assert recovery[0].parent == artifacts
    else:
        assert recovery[0].parent.name.startswith("opencollab-team-recovery-")
        assert recovery[0].parent.stat().st_mode & 0o777 == 0o700
    restored = tmp_path / "restored"
    restored.mkdir()
    (restored / "answer.py").write_text("def answer():\n    return 1\n")
    (restored / "second.py").write_text("value = 0\n")
    subprocess.run(["git", "apply", str(recovery[0])], cwd=restored, check=True, capture_output=True)
    subprocess.run([sys.executable, "-B", "-c", "from answer import answer; from second import value; "
                    "assert (answer(), value) == (42, 2); print('FULL_CANDIDATE_RECOVERED')"],
                   cwd=restored, check=True, capture_output=True)
    listed = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=source,
                            check=True, capture_output=True, text=True)
    assert listed.stdout.count("worktree ") == 1
