"""CLI option placement must select the requested workflow workspace."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from click import unstyle
from typer.testing import CliRunner

import opencollab.adapters.cli.main as cli_main
from tests.support.paths import PACKAGE_ROOT


def _project(path: Path, name: str) -> Path:
    workflows = path / "workflows"
    workflows.mkdir(parents=True)
    configs = path / "configs"
    configs.mkdir()
    (configs / ".env").write_text("OPENCOLLAB_API_KEY=fixture-key\n", encoding="utf-8")  # pragma: allowlist secret
    (workflows / "probe.py").write_text(
        "from pathlib import Path\n"
        "from opencollab.workflows import workflow\n"
        "root = Path(__file__).parent.parent\n"
        "(root / 'discovered').write_text('loaded', encoding='utf-8')\n"
        f"@workflow(name={name!r})\n"
        "async def probe(ctx, args):\n"
        "    (root / 'executed').write_text('ran', encoding='utf-8')\n"
        "    return {'workspace': str(root), 'budget': ctx.budget.total}\n",
        encoding="utf-8",
    )
    return path

def _cli(cwd: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "opencollab", *arguments],
        cwd=cwd,
        env={
            "PATH": os.environ["PATH"],
            "PYTHONPATH": str(PACKAGE_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "NO_COLOR": "1",
            "TERM": "dumb",
            "COLUMNS": "240",
            "TERMINAL_WIDTH": "240",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )

def _assert_untouched(*projects: Path) -> None:
    for project in projects:
        assert not (project / "discovered").exists()
        assert not (project / "executed").exists()
        assert not (project / ".opencollab").exists()

@pytest.mark.parametrize("command", [["run", "scope-probe"], ["list"]], ids=["run", "list"])
def test_cli_rejects_root_workflow_options_before_project_discovery(tmp_path, command):
    requested = _project(tmp_path / "requested", "scope-probe")
    current = _project(tmp_path / "current", "scope-probe")
    result = _cli(
        current,
        ["--workspace", str(requested), "--budget", "15000", "workflow", *command],
    )

    assert result.returncode == 2, result.stdout + result.stderr
    diagnostic = unstyle(result.stdout + result.stderr)
    assert "--workspace" in diagnostic and "--budget" in diagnostic
    assert "workflow run NAME" in diagnostic and "workflow list" in diagnostic
    _assert_untouched(requested, current)

@pytest.mark.parametrize(
    "options",
    [
        ["--workspace", "."],
        ["--trace"],
        ["--api-key", "fixture-secret-never-print"],  # pragma: allowlist secret
    ],
    ids=["explicit-default", "flag", "secret"],
)
def test_cli_rejects_explicit_root_options_and_keeps_values_private(tmp_path, options):
    current = _project(tmp_path / "current", "scope-probe")
    result = _cli(current, [*options, "workflow", "run", "scope-probe"])

    assert result.returncode == 2, result.stdout + result.stderr
    diagnostic = unstyle(result.stdout + result.stderr)
    assert options[0] in diagnostic
    assert "fixture-secret-never-print" not in diagnostic
    _assert_untouched(current)

def test_cli_workflow_child_options_select_requested_workspace_and_budget(tmp_path):
    requested = _project(tmp_path / "requested", "scope-probe")
    current = _project(tmp_path / "current", "scope-probe")
    result = _cli(
        current,
        ["workflow", "run", "scope-probe", "--workspace", str(requested), "--budget", "15000", "--no-save"],
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"workspace": str(requested), "budget": 15000}
    assert (requested / "executed").read_text(encoding="utf-8") == "ran"
    _assert_untouched(current)

def test_cli_workflow_defaults_keep_current_workspace(tmp_path):
    current = _project(tmp_path / "current", "scope-probe")
    result = _cli(current, ["workflow", "run", "scope-probe", "--no-save"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"workspace": str(current), "budget": 1_000_000}
    assert (current / "executed").read_text(encoding="utf-8") == "ran"

@pytest.mark.parametrize("explicit_workspace", [False, True], ids=["default", "child-option"])
def test_cli_workflow_list_workspace_remains_valid(tmp_path, explicit_workspace):
    requested = _project(tmp_path / "requested", "requested-only")
    current = _project(tmp_path / "current", "current-only")
    options = ["--workspace", str(requested)] if explicit_workspace else []
    selected, untouched = (requested, current) if explicit_workspace else (current, requested)
    result = _cli(current, ["workflow", "list", *options])

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{selected.name}-only" in result.stdout and f"{untouched.name}-only" not in result.stdout
    assert (selected / "discovered").exists()
    assert not (selected / "executed").exists()
    assert not (selected / ".opencollab").exists()
    _assert_untouched(untouched)

@pytest.mark.parametrize(
    "command",
    [[], ["workflow"], ["workflow", "run"], ["workflow", "list"]],
    ids=["root", "workflow", "run", "list"],
)
def test_cli_help_keeps_project_untouched(tmp_path, command):
    current = _project(tmp_path / "current", "scope-probe")
    result = _cli(current, [*command, "--help"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Usage" in unstyle(result.stdout)
    _assert_untouched(current)

def test_cli_interactive_root_options_keep_their_configuration(tmp_path, monkeypatch):
    captured = {}
    config_file = tmp_path / "config.env"
    monkeypatch.setenv("OPENCOLLAB_CONFIG_FILE", str(config_file))
    config_file.write_text("OPENCOLLAB_API_KEY=fixture-key\n", encoding="utf-8")  # pragma: allowlist secret

    async def run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli_main, "_run", run)
    result = CliRunner().invoke(
        cli_main.app,
        ["--workspace", str(tmp_path), "--budget", "15000", "--model", "chosen-model", "--trace", "--prompt", "work"],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["workspace"] == str(tmp_path)
    assert captured["cfg"]["budget"] == 15000
    assert captured["cfg"]["model"] == "chosen-model"
    assert captured["trace"] is True
    assert captured["one_shot_prompt"] == "work"
