"""Native test options retain their values instead of creating extra targets."""

from __future__ import annotations

import re
import shlex
import shutil
import sys

import pytest
from _pytest.config import get_config

from opencollab import OpenCollab
from opencollab.adapters.tools.evidence.command_evidence import parse_test_command
from opencollab.builtin_workflows import WeaveAdapter
from opencollab.environments import local_environment


@pytest.mark.parametrize(("command", "targets"), [
    ("python -m pytest -rA tests/test_example.py", ("tests/test_example.py",)),
    ("python -m pytest -r A tests/test_example.py", ("tests/test_example.py",)),
    ("python -m pytest --report-chars A tests/test_example.py", ("tests/test_example.py",)),
    ("python -m pytest --max-warnings 5 tests/test_example.py", ("tests/test_example.py",)),
    ("python -m pytest -r A -k example tests", ("tests",)),
    ("go test -json -coverprofile=coverage.out .", (".",)),
    ("go test -json -coverprofile coverage.out .", (".",)),
    ("go test -json -covermode count -coverpkg ./... -coverprofile coverage.out ./pkg", ("./pkg",)),
])
def test_native_test_options_do_not_become_targets(command, targets):
    parsed = parse_test_command(command, "/workspace")
    assert parsed is not None and parsed.provable
    assert parsed.targets == targets

@pytest.mark.parametrize("command", ["python -m pytest -r", "go test -json -coverprofile"])
def test_missing_native_option_value_cannot_prove_execution(command):
    parsed = parse_test_command(command, "/workspace")
    assert parsed is None or not parsed.provable

def test_pytest_core_required_values_follow_installed_option_metadata():
    for group in get_config()._parser._groups:
        for option in group.options:
            attrs = option.attrs()
            if attrs.get("action", "store") not in {"store", "append"} or attrs.get("nargs") is not None:
                continue
            for name in option.names():
                parsed = parse_test_command(f"python -m pytest {name} value test_example.py", "/workspace")
                assert parsed is not None and parsed.provable, name
                assert parsed.targets == ("test_example.py",), name
                missing = parse_test_command(f"python -m pytest {name}", "/workspace")
                assert missing is not None and not missing.provable, name

@pytest.mark.parametrize(("arguments", "targets"), [
    ("--durations 5 -v --disable-warnings 2026.py 5", ("2026.py", "5")),
    ("--durations 5 -- --durations 5", ("--durations", "5")),
    ("--plugin-option 5 test_example.py", ("5", "test_example.py")),
])
def test_pytest_option_values_preserve_positional_targets(arguments, targets):
    parsed = parse_test_command(f"python -m pytest {arguments}", "/workspace")
    assert parsed is not None and parsed.provable
    assert parsed.targets == targets

async def verify(workspace, command, name):
    environment = local_environment(workspace)
    await environment.setup()
    adapter = WeaveAdapter(check_commands=[command])

    async def flow(ctx, inputs):
        check = await adapter.verify(ctx, None, None)
        return check.report

    try:
        result = await OpenCollab(
            workspace, model="unused-verification-fixture",
            api_key="fixture", environment=environment,  # pragma: allowlist secret
        ).workflow(flow, {}, artifacts=workspace / name, trace=False)
        assert result.ok, (result.reason, result.error)
        return result.output
    finally:
        await environment.cleanup()

@pytest.mark.parametrize("report_option", ["-rA", "-r A"])
async def test_real_pytest_report_option_checks_actual_target(tmp_path, report_option):
    target = tmp_path / "test_example.py"
    target.write_text("def test_example():\n    assert 1 + 1 == 2\n")
    command = (f"{shlex.quote(sys.executable)} -B -m pytest -p no:cacheprovider --color=no "
               f"{report_option} test_example.py")
    passed = await verify(tmp_path, command, "passed-artifacts")
    assert passed["executed"] and passed["ok"]
    records = passed["checks"][0]["test_records"]
    assert [row["target"] for row in records] == ["test_example.py"]
    assert all(row["verified"] for row in records)
    target.write_text("def test_example():\n    assert 1 + 1 == 3\n")
    failed = await verify(tmp_path, command, "failed-artifacts")
    assert failed["executed"] and not failed["ok"]
    assert failed["checks"][0]["exit_code"] == 1

@pytest.mark.parametrize("target", ["test_example.py", "5"])
async def test_real_pytest_core_option_values_check_the_same_nodes(tmp_path, target):
    test_path = "test_example.py" if target.endswith(".py") else f"{target}/test_example.py"
    source = tmp_path / test_path
    source.parent.mkdir(exist_ok=True)
    source.write_text(
        "import pytest\n@pytest.mark.parametrize('value', range(4))\n"
        "def test_example(value):\n    assert value >= 0\n"
    )
    value_options = [
        ("--durations", "5"), ("--durations-min", "0"), ("--verbosity", "1"),
        ("--show-capture", "all"), ("--log-level", "INFO"), ("--log-cli-level", "INFO"),
        ("--log-format", "%(message)s"), ("--pythonwarnings", "default"),
    ]
    nodes = []
    for form in ("separated", "equals"):
        options = [
            token for name, value in value_options
            for token in ([name, value] if form == "separated" else [f"{name}={value}"])
        ]
        command = "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 " + shlex.join([
            sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "--color=no", "-rA",
            *options, "-W", "default", "--", target,
        ])
        passed = await verify(tmp_path, command, f"{form}-artifacts")
        check = passed["checks"][0]
        assert passed["executed"] and passed["ok"], check
        assert check["exit_code"] == 0
        assert [(row["target"], row["verified"]) for row in check["test_records"]] == [(target, True)]
        nodes.append(re.findall(r"^PASSED (.+)$", check["output"], re.MULTILINE))
    expected = [f"{test_path}::test_example[{value}]" for value in range(4)]
    assert nodes == [expected, expected]

@pytest.mark.skipif(shutil.which("go") is None, reason="Go is required for native coverage execution")
@pytest.mark.parametrize("coverage_options", [
    "-coverprofile=coverage.out",
    "-coverprofile coverage.out -covermode count -coverpkg ./...",
])
async def test_real_go_coverage_options_check_actual_package(tmp_path, coverage_options):
    (tmp_path / "go.mod").write_text("module example.org/nativeproof\n\ngo 1.22\n")
    source = tmp_path / "math.go"
    source.write_text("package nativeproof\nfunc Add(a, b int) int { return a + b }\n")
    (tmp_path / "math_test.go").write_text(
        'package nativeproof\nimport "testing"\n'
        'func TestAdd(t *testing.T) { if Add(1, 1) != 2 { t.Fatal("addition failed") } }\n'
    )
    command = f"GOTOOLCHAIN=local GOPROXY=off go test -json -count=1 {coverage_options} ."
    passed = await verify(tmp_path, command, "passed-artifacts")
    assert passed["executed"] and passed["ok"]
    records = passed["checks"][0]["test_records"]
    assert [row["target"] for row in records] == ["."]
    assert all(row["verified"] for row in records)
    assert (tmp_path / "coverage.out").read_text().startswith(("mode: set", "mode: count"))
    source.write_text("package nativeproof\nfunc Add(a, b int) int { return 0 }\n")
    failed = await verify(tmp_path, command, "failed-artifacts")
    assert failed["executed"] and not failed["ok"]
    assert failed["checks"][0]["exit_code"] != 0
