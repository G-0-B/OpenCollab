"""State and read-tool preparation for scout evidence tests."""

import asyncio

from opencollab.domain.session import TurnEnforcementState
from tests.support.tool_execution_test_support import build_sensor_use_case


class _ReadStub:
    name = "file_read"

    def to_openai_schema(self):
        return {"type": "function", "function": {"name": self.name, "parameters": {}}}

class _FakeState:
    def __init__(self, used_tokens=0, scout_ledger=None, messages=None):
        self.used_tokens = used_tokens
        self.wind_down_done = False
        self.wind_down_token_mark = 0
        self.messages = messages if messages is not None else []
        self.turn = TurnEnforcementState(
            scout_ledger=scout_ledger if scout_ledger is not None else []
        )

class ScriptedTool:
    def __init__(self, name, outputs):
        self.name = name
        self._outputs = list(outputs)
        self.calls = []

    async def execute_with_runtime(self, args, runtime):
        self.calls.append(args)
        return self._outputs.pop(0) if self._outputs else ""

def _call(name, args, cid="c1"):
    return {"id": cid, "function": {"name": name, "arguments": args}}

def _run_one(state, tool, name, args):
    result = asyncio.run(build_sensor_use_case(state, tool).process([_call(name, args)]))
    result.apply_to(state)
    return result
