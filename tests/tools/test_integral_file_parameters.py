"""Schema-valid integral JSON numbers work at native file-tool boundaries."""

import json

import pytest

from opencollab.adapters.env import LocalEnvironment
from opencollab.adapters.tools.apply_patch import ApplyPatchTool
from opencollab.adapters.tools.apply_patch_engine import _apply_line_replace
from opencollab.adapters.tools.fs import FileReadTool
from opencollab.application.schema_validate import validate
from opencollab.application.tool_execution import ToolRuntime
from opencollab.bootstrap import build_session
from opencollab.domain.agent import Agent
from tests.support.session_run_loop_test_support import FakeLLM, llm_response, tool_call


@pytest.mark.parametrize("tool_type", [FileReadTool, ApplyPatchTool])
@pytest.mark.parametrize("number", [1, 1.0])
async def test_integral_parameters_run_through_session(tmp_path, tool_type, number):
    source = tmp_path / "sample.txt"
    source.write_text("alpha\nbeta\ngamma\n")
    tool = tool_type()
    args = {"path": "sample.txt", "offset": number, "limit": number + 1}
    if tool.name == "apply_patch":
        args = {"path": "sample.txt", "mode": "line_replace", "start_line": number,
                "end_line": number, "expected_str": "alpha", "new_str": "updated"}
    assert validate(args, tool.parameters) == []
    model = FakeLLM([
        llm_response(tool_calls=[tool_call(name=tool.name, arguments=json.dumps(args))]),
        llm_response(content="done"),
    ])
    environment = LocalEnvironment(str(tmp_path))
    session = build_session(
        agent=Agent(name="test", system_prompt="Use the requested tool.", tools=[tool]),
        env=environment, llm=model,
    )
    try:
        await session.add_user_message("Read or update the first line.")
        assert await session.run_loop() == "done"
        results = [m["content"] for m in model.calls[1]["messages"] if m.get("role") == "tool"]
        assert len(results) == 1
        assert "error" not in results[0].lower()
        if tool.name == "file_read":
            assert "1\talpha\n2\tbeta" in results[0]
            assert "3\tgamma" not in results[0]
        else:
            assert source.read_text() == "updated\nbeta\ngamma\n"
    finally:
        await environment.cleanup()

@pytest.mark.parametrize("value", [True, False, 1.5, float("inf"), float("nan"), "1", None])
@pytest.mark.parametrize("key", ["start_line", "end_line"])
def test_line_replace_rejects_non_integer_arguments_without_editing(value, key):
    params = {"start_line": 1, "end_line": 1, "expected_str": "alpha", "new_str": "updated", key: value}
    updated, error = _apply_line_replace("alpha\nbeta\n", params)
    assert updated is None
    assert "integer" in error

def test_integral_insert_sentinel_preserves_zero_end_line():
    params = {"start_line": 1.0, "end_line": 0.0, "expected_str": "", "new_str": "new"}
    updated, error = _apply_line_replace("alpha\n", params)
    assert error == ""
    assert updated == "new\nalpha\n"

@pytest.mark.parametrize("params", [{"start_line": -1.0, "end_line": 1.0},
                                   {"start_line": 2.0, "end_line": 0.0}])
def test_integral_ranges_keep_existing_bounds(params):
    updated, error = _apply_line_replace("alpha\n", {**params, "new_str": "new"})
    assert updated is None
    assert "must be" in error

async def test_integral_read_range_works_with_read_file_only_backend():
    class Environment:
        workspace = "/unused"

        async def read_file(self, path):
            return "alpha\nbeta\ngamma\n"

    result = await FileReadTool().execute_with_runtime(
        {"path": "sample.txt", "offset": 2.0, "limit": 1.0},
        ToolRuntime(environment=Environment(), safety_policy=None, permission_policy=None),
    )
    assert "2\tbeta" in result
    assert "\talpha" not in result and "\tgamma" not in result

@pytest.mark.parametrize("value", [True, False, 1.5, float("inf"), float("nan"), "1", None, 0, -1])
@pytest.mark.parametrize("key", ["offset", "limit"])
async def test_read_range_rejects_invalid_numbers(tmp_path, value, key):
    environment = LocalEnvironment(str(tmp_path))
    (tmp_path / "sample.txt").write_text("alpha\n")
    try:
        with pytest.raises(ValueError):
            await FileReadTool().execute_with_runtime(
                {"path": "sample.txt", key: value},
                ToolRuntime(environment=environment, safety_policy=None, permission_policy=None),
            )
    finally:
        await environment.cleanup()
