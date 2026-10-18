# OpenCollab Python Package

OpenCollab is installed as a Python package and exposes a small public API over
the framework runtime. The [project homepage](../README.md) gives the quick
start. Run repository commands from the repository root unless a command says
otherwise.

| Task | Read |
| --- | --- |
| Install or start the terminal | [Install](#install) · [Commands](#commands) |
| Embed an agent, team, or workflow | [Python SDK](#python-sdk) |
| Add a tool or workflow | [Custom tools](#custom-tools) · [Workflow authoring](#workflow-authoring) |
| Understand team delivery | [Cooperative teams](#cooperative-teams) |
| Set providers, models, or teams | [Configuration](../configs/README.md) |
| Control budgets, cancellation, and cleanup | [Runtime controls](../docs/evaluation-runtime.md) |

## Install

This guide describes the current source checkout. Weave, `RunControl`, and
`WorkflowContext.agent_run` were added after the `v0.9.3` release. Use a current
`main` checkout and the source-installation commands below for those features.
Release tags and their attached wheels retain the APIs published with each tag.

OpenCollab supports Linux and macOS hosts. Its local file adapters require
descriptor-relative operations, no-follow `stat`, `O_NOFOLLOW`, fd-based
directory listing, callable `fcntl.flock`, and an atomic no-clobber rename. The
rename primitive is `renameat2(RENAME_NOREPLACE)` on Linux and
`renameatx_np(RENAME_EXCL)` on macOS 10.12 and newer. OpenCollab reports a
capability error when the host lacks a required primitive.

Install the project and development dependencies with `uv`.

```bash
uv sync --locked --extra dev
```

Or create a conventional virtual environment with `pip`.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Commands

During development, run the registered CLI from the project environment.

```bash
uv run opencollab --workspace .
```

It runs the current checkout in the project environment, resolves
`configs/.env`, and starts agent 0 with the built-in Self-Collaboration team.
Team runs use the root command with `--team-config PATH`. To work with team files,
use `team init` to create an editable team file and `team show` to inspect its
roles and topology. `scripts/start_opencollab.sh` remains available for
environments that need its physical-path handling.

After installation, invoke the CLI directly from the active environment.

```bash
opencollab --workspace .
opencollab team init team.yaml
opencollab team show --team-config team.yaml
opencollab --team-config team.yaml --workspace .
opencollab workflow list --workspace .
OPENCOLLAB_WORKFLOWS_DIR=path/to/workflows \
  opencollab workflow run NAME --args '{"goal": "..."}'
```

A workflow directory contains caller-authored Python modules tagged with
`@workflow`. Both `workflow list` and `workflow run` combine them with installed
workflows, including [Duo](../docs/duo.md). `OPENCOLLAB_WORKFLOWS_DIR` selects the
caller directory. Relative paths resolve from the workspace, and duplicate
names raise the registry's existing error.

In an interactive session, `/help` shows the local controls. Tab and Shift+Tab
select an agent, `/save` saves the lead session, and `/exit` leaves the terminal.
A reply to a waiting agent question is delivered before local commands.

For workflows that accept a `goal`, `--task TEXT` and `--task-file PATH` provide
that argument directly. Other arguments still use `--args`; an explicit `goal`
in that JSON conflicts with either shortcut.

For the root team command, `--trace` enables trajectory recording,
`--no-worktrees` disables per-child git-worktree isolation, and
`--team-config PATH` selects a team YAML file. `--yolo` auto-approves risky
commands. `--prompt TEXT` or `--prompt-file PATH` runs one turn and exits.
Add `--hold` to inspect agents at the prompt after that turn.

Workflow options follow `workflow run NAME` or `workflow list`. For example,
use `opencollab workflow run NAME --workspace . --no-trace`. Workflow runs
save sessions and record orchestration by default. `--no-trace` keeps saved
sessions and the workflow manifest, while `--no-save` disables run artifacts.

## Python SDK

The package root exposes `OpenCollab`, `RunResult`, `RunError`, the
`workflow` decorator, and `RunControl`, `BudgetSnapshot`, `BudgetDecision`,
and `RunEvent` for controlled agent runs. Compatibility follows package SemVer.

```python
import asyncio

from opencollab import OpenCollab


async def main() -> None:
    oc = OpenCollab(".")  # resolves configs/.env and environment variables once
    result = await oc.agent(
        "Inspect this repository and report its release readiness.",
        budget=100_000,
        artifacts="artifacts/release-readiness",
    )
    print(result.raise_for_status().output)


asyncio.run(main())
```

The same client exposes `agent(...)`, `team(...)`, and `workflow(...)`. All
three return `RunResult`. Import optional authoring contracts from
`opencollab.tools`, `opencollab.environments`, and `opencollab.workflows`.
Installed collaboration protocols are public through
`opencollab.builtin_workflows`, and Git patch parsing is public through
`opencollab.patches`. Named agent profile resolution is public through
`opencollab.profiles`.
Treat other package paths as internal. An `artifacts` directory, when supplied,
must be new or empty because each run claims it for executable evidence.
`team(...)` uses the built-in Self-Collaboration team unless its `config=`
argument names a team YAML file. Its `cleanup_timeout` bounds scheduler
shutdown and must be a finite positive number.

### Select a run and handle its result

| Call | Selection and limits |
| --- | --- |
| `agent(...)` / `agent2(...)` | Base currently maps to [Single2](../docs/single2.md); `agent2` selects it explicitly. `max_steps` limits steps; `steps` is its older alias. |
| `team(...)` | `config=` selects a team file; otherwise the built-in team is used. See [team settings and hooks](../configs/README.md#team) and [delivery](#cooperative-teams). |
| `workflow(...)` | Accepts an installed or discovered name, a decorated function, or a workflow spec. `concurrency` limits agent sessions; `task_concurrency` independently limits active parallel/pipeline units and defaults to `concurrency`. Mixed work can peak at their sum. |

Workflow runs also accept `max_steps`, `system_prompt`, and `cleanup_timeout`.
Named calls use the same workspace discovery as the CLI; functions and specs
run directly.

`RunResult.status` is `completed`, `stopped`, or `failed`. Only `completed` has
`ok=True`; a stop retains its `reason` and partial work. `raise_for_status()`
returns completed results or raises `RunError`, whose `result` retains the
original outcome. Invalid arguments raise before execution; lifecycle or
persistence failures can raise `RunError` directly. Execution completion still
requires task-specific verification before claiming the application is correct.

`RunControl` adds per-request budget decisions and session events to `agent`
and `agent2`; workflow roles use `ctx.agent_run`. The
[runtime control reference](../docs/evaluation-runtime.md#budget-decisions-and-session-events)
contains the callback contract and an example. It also explains
[limits and deadlines](../docs/evaluation-runtime.md#limits-and-deadlines),
[cancellation and resource ownership](../docs/evaluation-runtime.md#cancellation-usage-and-resource-ownership),
run identity, late usage, and failure recovery. `timeout` bounds execution and
`cleanup_timeout` separately bounds owned shutdown.

`OpenCollab.configuration` is a read-only snapshot of effective provider, model,
protocol, context, token, timeout, sampling, and thinking settings.
`thinking_params` is deep-copied. Keys and base URLs are excluded;
`base_url_sha256` identifies the endpoint without exposing embedded credentials.
See [configuration](../configs/README.md) for resolution and transport settings.

### Tools and environments

Use `builtin_tools` to compose tools through the public package.

```python
from opencollab.tools import builtin_tools

tools = builtin_tools(
    "file_read",
    "file_write",
    "bash",
    allow_file_creation=False,
)
```

The helper returns fresh tools in caller order. Headless shell tools require
process-isolated environments, and limits for unselected tools are rejected.
Use `bash` to run the repository's native test command and inspect its exit code
and output. The `VerificationTool` protocol remains available for tools that
provide parser-backed test evidence. `evidence_tools` composes native tools
with a Bash observer for workflows such as Duo. See
[native test evidence](../docs/test-evidence.md) for record collection and
supported runners.
See [test-runner migration](../docs/migrations/0.7.0.md#native-bash-testing-replaces-run_tests) for existing
team files, saved runs, and verifier integrations.

Environment composition uses the same narrow public module.

```python
from opencollab.environments import (
    docker_environment,
    local_environment,
    worktree_environment,
)

host = local_environment(".")
isolated_source = worktree_environment(".")
container = docker_environment("python:3.11-slim", isolated_source)
```

These factories return caller-owned environments whose setup has not started.
Every public environment supports `await environment.setup()`. Docker
environments also accept `mount_dir`. Host and worktree environments raise an
argument error for `mount_dir`. The caller controls setup, mount, and cleanup
timing. Concrete adapter classes live in `adapters`, and Bootstrap wires them.

A caller-supplied environment remains caller-owned. The
[runtime ownership reference](../docs/evaluation-runtime.md#cancellation-usage-and-resource-ownership)
explains which cleanup and quiescence fields the SDK can establish.

### Custom tools

Application-specific tools can implement the public `opencollab.tools.Tool`
protocol without inheriting an adapter or importing framework internals. This
read-only example exposes the application's test policy to an agent.

```python
class ProjectPolicy:
    name = "project_policy"
    description = "Look up the project's test command for a language."
    parameters = {
        "type": "object",
        "properties": {"language": {"type": "string", "enum": ["python", "javascript"]}},
        "required": ["language"],
        "additionalProperties": False,
    }
    default_timeout = 30.0
    disable_outer_timeout = False

    def to_openai_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute_with_runtime(self, params, runtime):
        commands = {"python": "python -m pytest -q", "javascript": "npm test"}
        return commands[params["language"]]


policy = ProjectPolicy()
```

Pass the instance in `await oc.agent("Find this Python project's test command.",
tools=[policy])`, or in a workflow role's `tools=[policy]`. The schema declares
accepted JSON arguments; execution returns text for the model. Framework runtime
access is supplied to `execute_with_runtime`; this in-memory lookup needs none.
Environment-backed tools should use that supplied runtime rather than assuming
the caller's host directory. Keep tool names unique within a call. To contribute
a built-in tool, see the [contributor extension guide](../CONTRIBUTING.md#extend-a-tool-or-workflow).

### Workflow authoring

A workflow is a plain async Python function tagged with `@workflow`. Create
`workflows/implement_and_review.py` with this implementation.

```python
from typing import Any

from opencollab import workflow
from opencollab.tools import builtin_tools
from opencollab.workflows import WorkflowContext


@workflow(name="implement-and-review")
async def implement_and_review(
    ctx: WorkflowContext,
    inputs: dict[str, Any],
) -> str | dict[str, Any] | None:
    tools = inputs.get("tools")
    if tools is None:
        tools = builtin_tools(
            "bash", "file_read", "file_write", "apply_patch", "git_diff", "grep",
            headless=inputs.get("allow_unisolated_shell") is not True,
        )
    draft = await ctx.agent(
        f"Implement and verify: {inputs['task']}",
        tools=tools,
        timeout=900,
    )
    await ctx.log(f"Tokens spent: {ctx.tokens_spent()}")
    diff = await ctx.diff()
    return await ctx.agent(
        f"Review the implementation and fix gaps:\n{draft}\n\nDiff:\n{diff or ''}",
        tools=tools,
    )
```

OpenCollab discovers top-level `*.py` modules in `workflows/` by default and
combines them with installed workflows in the CLI and SDK name lookup. Run
the decorated function through the CLI.

```bash
uv run opencollab workflow run implement-and-review \
  --args '{"task": "Add a regression test for the target bug.", "allow_unisolated_shell": true}'
```

This CLI example explicitly authorizes shell commands on the host. For an
environment that provides OS process isolation, keep `allow_unisolated_shell`
false or omit it. Both rounds receive the same coding tools.

Set `OPENCOLLAB_WORKFLOWS_DIR` to use another directory. The same decorated
function can be passed directly to `await OpenCollab(".").workflow(...)` when
embedding OpenCollab in Python. The [Duo guide](../docs/duo.md) shows built-in
workflow calls and the [Chinese guide](../docs/duo/README.zh-CN.md) gives the same
usage examples.

Evidence-preserving workflows can call `ctx.draft_findings(...)` to capture a
structured cite-or-abstain draft before exploration. OpenCollab owns this
generic capture primitive. OpenCollab-Eval owns benchmark prompts, policies,
datasets, and runners.

For current internals, see [runtime design](../docs/runtime-design.md) and the
[contributor architecture rule](../CONTRIBUTING.md#the-architecture-rule-enforced-by-lint-imports).
OpenCollab owns generic sessions and collaboration protocols; OpenCollab-Eval
owns benchmark prompts, policies, datasets, and runners.

## Architecture

Dependencies point inward: `bootstrap → adapters → application → domain`.
The [contributor guide](../CONTRIBUTING.md#project-layout) describes each layer
and the enforced exceptions. The [runtime guide](../docs/runtime-design.md#session-progress-and-restoration)
explains session transitions, persistence, and restoration.

## How it works

### Cooperative teams

A team stores active agents in a session table and schedules them cooperatively.
The `spawn` operation creates child sessions. Dynamic child grants are reserved
before the first `await`; prebuilt teams instead apply a cumulative cap per seat
within the shared pool. A team file may give each role its own allowance.
[Runtime allocation rules](../docs/runtime-design.md#token-accounting-and-allocation)
explain these modes and their in-flight overshoot boundary.
Each child can work in an isolated git worktree and return its diff with the
result.
The built-in Analyst adopts a completed direct Coder's complete changes with
`adopt_agent(aid)` before starting a fresh Tester. The aid is included in the
Coder's completion result. Adoption preserves the source index, combines
independent increments, and rejects conflicts atomically. Teams that declare
this tool create each child from the current source contents, including user
edits and earlier adopted changes. Plain directories retain independent copies.
Within the active entry turn, a later completed revision from the same Coder
applies only the changes since its last successful adoption. Repeating adoption
of the same completion leaves the delivery workspace unchanged.
`opencollab team init` exports this protocol. Adoption failures report a complete
recovery patch in the run's artifacts, or a private temporary recovery directory
when artifacts are disabled. Export that patch before releasing the run.
Custom teams without the tool retain their configured diff-return protocol.

The runtime exposes ports for model clients, execution environments, tools,
context shaping, and skills. See [runtime design](../docs/runtime-design.md)
for the constraints a replacement must preserve and
[benchmark results](../docs/results.md) and [adherence](../docs/adherence.md)
for recorded comparisons.

## Development

Follow [CONTRIBUTING.md](../CONTRIBUTING.md) for the development environment,
architecture checks, tests, and contribution rules. The [documentation index](../docs/README.md)
routes feature guides, migration records, and historical evidence by task.

## License

OpenCollab is licensed under the
[Mulan Permissive Software License v2](../LICENSE) (`MulanPSL-2.0`).
