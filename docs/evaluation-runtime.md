# Configured evaluation runtime

The public SDK exposes the runtime controls used by evaluation integrations
through `OpenCollab.agent()`, `team()`, and `workflow()`. Resolve model and
transport settings when constructing `OpenCollab(...)`, then select run budgets,
deadlines, profiles, and artifact paths on each call. See
[configuration](../configs/README.md) for environment variables and
[the package guide](../opencollab/README.md) for SDK and workflow examples.

| Task | Section |
| --- | --- |
| Set token, step, and time limits | [Limits and deadlines](#limits-and-deadlines) |
| Adjust allowances or observe a session | [Budget decisions and session events](#budget-decisions-and-session-events) |
| Interpret cancellation, cost, and cleanup | [Cancellation, usage, and resource ownership](#cancellation-usage-and-resource-ownership) |
| Recover partial changes | [Failed work and workspace recovery](#failed-work-and-workspace-recovery) |
| Run incremental workflow sessions | [Controlled workflow agent calls](#controlled-workflow-agent-calls) |

## Limits and deadlines

Explicit `OPENCOLLAB_UNBOUNDED_LIMITS=true` carries unbounded token and step
settings through standalone agents, workflow roles, and candidate sub-workflows.
The default standalone agent uses Base, currently mapped to Single2. With the
switch enabled, omitted token and step limits carry `None` into that session.
Explicit positive `budget`, `max_steps`, or `steps` values retain their limits
for Single2. A public `workflow()` call using the default
`limit_mode="environment"` with the switch enabled sets its token and role-step
limits to `None`, including when `budget` or `max_steps` is passed.
`limit_mode="explicit"` retains the supplied workflow and per-call limits,
including inside candidate child workflows. See
[controlled workflow agent calls](#controlled-workflow-agent-calls) for an example.
SDK `team()` continues to use its shared or declared per-role token allowances
and its `max_steps` setting. Finite settings use normal budget reservations.

SDK `timeout=` bounds a complete run, and workflow-role `timeout=` bounds that
role. Model request, connection, first-event, and stream-idle timeouts are
separate transport controls. `cleanup_timeout=` bounds owned shutdown after
execution ends. The SDK configuration validates transport timeouts as finite
positive seconds. Lower-level native Responses adapter calls also accept
`None` for stream waits, retaining transport timeout causes in that mode.
Unbounded token and step limits preserve the configured transport and shutdown
settings.

## Budget decisions and session events

`RunControl` (0.9.4 and newer) configures `agent` / `agent2` calls and
`ctx.agent_run` sessions. Import it, `BudgetSnapshot`, `BudgetDecision`, and
`RunEvent` from `opencollab`. The effective `budget` is the hard authorization;
`initial_soft_budget_tokens` starts a smaller allowance clipped to that grant.

A sync or async `decide_budget(snapshot)` returns
`BudgetDecision(soft_budget_tokens, final_prompt=None)` to keep or increase the
soft allowance. It runs at PRECHECK after cancellation and loop
checks, before budget wind-down, and after input estimation for each session
request. A response that exceeds the soft allowance can also trigger
`after_response` before the ordinary budget stop; hard exhaustion does not call
the policy. Context-overflow and tool-choice retries create session requests;
transport retries stay inside the same request. The snapshot carries run and
session IDs, agent ID, used tokens, steps, both allowances, input reservation,
minimum output requirement, and trigger reason.

A decrease, an increase below actual spend, or an increase past the hard grant
fails the run with the policy error. Keeping an already-crossed soft allowance
produces the ordinary soft-budget stop; hard exhaustion keeps the hard-budget
stop. A `final_prompt` enters the current request and recomputes its input
reservation. Extend or close while enough allowance remains for that request.

This example grants one additional tranche after half of the initial allowance
has been spent. Applications should replace that condition with their own
executable progress evidence.

```python
import asyncio

from opencollab import BudgetDecision, OpenCollab, RunControl


def decide(snapshot):
    allowance = snapshot.soft_budget_tokens
    if snapshot.used_tokens >= 50_000:
        allowance = snapshot.hard_budget_tokens
    return BudgetDecision(allowance)


async def main():
    events = []
    result = await OpenCollab(".").agent2(
        "Inspect this repository and report its release readiness.",
        budget=200_000,
        run_control=RunControl(
            initial_soft_budget_tokens=100_000,
            decide_budget=decide,
            on_event=events.append,
        ),
    )
    print(result.status, result.reason)


asyncio.run(main())
```

`on_event(event)` is synchronous and independent of `trace`. Each event carries
`type`, `run_id`, `session_id`, `aid`, `used_tokens`, `steps`, and detached `data`.
`usage` includes completed calls, failed calls with reported usage, summaries,
and late responses; `purpose` and `late` identify those paths. Transport retries
can combine reported usage in one enclosing call's event. Cache-read,
cache-creation, reasoning, and estimated-usage fields preserve provider metadata
without adding cached input twice. `context_shaping` reports normal and emergency
compaction; `budget_decision`, `error`, and `session_stopped` expose decisions and
lifecycle state. A stopped event records outstanding provider requests.

The receiver stays attached to its original run through late accounting and
cleanup, ending with `cleanup_completed`. Receiver errors remain in
`result.metrics["observation_errors"]` (or the controlled call's
`observation_errors`) after accounting and cleanup.
`history_trigger_tokens` accepts an integer of at least two and caps a known
model's compaction trigger; the target stays 75% of that trigger. Unknown windows
keep fixed defaults. `tool_cancellation_cleanup_timeout` accepts finite positive
seconds. Omitted controls retain runtime defaults; trajectories record effective
history thresholds and their source.

## Cancellation, usage, and resource ownership

Request lifecycle traces record `llm_call_started`, `llm_call`, and
`llm_call_cancelled` with the same response session and step identifiers.
Started and cancelled records carry zero known tokens. Completed records retain
reported usage. Late provider success or error usage is charged once during
owned draining, including after caller cancellation. Cancelled calls therefore
need their final persisted usage and cleanup outcome when interpreting cost.
Caller cancellation propagates as `asyncio.CancelledError` after owned shutdown.
A run deadline produces a stopped result with `reason="timeout"`.

Run token totals add input and output. Cache read and cache creation counts are
already included in input tokens, and reasoning counts follow the provider's
output accounting. Provider usage records preserve raw counters and identify
estimated usage. `OPENCOLLAB_API_USAGE_LOG` selects an optional append-only
provider ledger, while enabled trajectories retain per-call usage. Existing
lightweight tracers continue to work when they omit an optional flush method.

### Native clients and run identity

`OpenCollab.create_model_client()` constructs a caller-owned native model
transport from the resolved configuration. Close it with `await model.close()`
after use. A protocol-compatible wrapper can be passed to
`OpenCollab.agent(llm=...)`. `OpenCollab.configuration` includes protocol,
context size, retry count, and provider error time allowance alongside the
other non-secret fields. `OpenCollab.read_session_snapshot(path)` reads the
native snapshot and completed journal entries through the storage adapter.

Every public SDK run generates a `run_id` or accepts one from the caller.
The result's `metrics["run_id"]` joins the agent snapshot, team manifest, or
workflow manifest and any recorded trajectory. Saved workflow manifests retain
the same identity with tracing disabled and on failed outcomes. Artifact
directories must be new or empty. A caller-supplied environment remains owned
by the caller, whose cleanup determines environment quiescence. The result's
`session_quiesced` separately describes session and persistence completion.

For caller-owned environments, `environment_quiesced`, `cleanup_quiesced`, and
`execution_quiesced` stay `None` until the caller performs and verifies cleanup.
Workflow results aggregate session, step, token, and markup-recovery metrics.
`RunResult.agent_failures` preserves sanitized child-provider failures. Nested
and OS failures include a bounded `exception_chain` with types, modules, integer
`errno`, and HTTP status when available; messages, bodies, and filesystem paths
are excluded.

## Failed work and workspace recovery

Failed Team children deliver available worktree patches with their failure
results. Prebuilt teammates receive the same partial work in their stop notices.
Their ERROR or STOPPED status and original failure reason remain available.
When capture fails, or a bounded result or notice cannot carry the complete
partial patch, native worktrees retain the full files and report a recovery
location. The SDK reports the retained cleanup state. Export the full changes
from that location before releasing the worktree. A later successful
`get_diff()` clears the native retention state and permits cleanup.

A revoked execution environment stops its current session before another model
call. Workflow calls also check the shared environment before acquiring work and
after waiting for an agent slot, so queued roles terminate with the revocation
error while existing transcript, usage, and candidate recovery records survive.
An isolated candidate's environment remains distinct from its parent environment.

## Provider and tool boundary evidence

Structured exploration, structured correction, and final synthesis inherit the
configured reasoning policy. Correction uses the caller's remaining role time.
Provider adapters retain typed output-limit responses, including Chat
Completions `length` and Responses `max_tokens` finish reasons, with their
available content and usage. They retry classified transient failures and
check streamed output against the terminal response. On the Responses path,
`OPENCOLLAB_EXTERNAL_PROVIDER_ISOLATION=1` keeps retryable failures within one
logical call and removes its retry-count and provider-error-time allowance.
The caller retains cancellation and the configured request deadline.

Rejected, oversized malformed tool arguments are shortened in the model's
read-time view, with both chat tool calls and Responses items kept paired. The
original transcript retains the rejected arguments. Successful and pending tool
calls remain exact. Small context windows retain the two most recent evidence
groups while using the existing pressure-triggered history processing.

If a role already requires a write or structured submission and the model returns
only prose, the session retries that requirement once with the same tool set. A
second prose-only response stops the session instead of declaring the requirement
complete. Both responses remain in its usage and transcript.

Responses errors with `string_above_max_length` trigger context compaction when
the provider identifies input content, tool output, or history arguments.
Errors for instruction, metadata, tool description, and name fields retain their
ordinary request-error classification. Structured roles whose input still
exceeds the provider limit after compaction finish without issuing the same
oversized corrective request.

A caller may set `OPENCOLLAB_REQUIRE_INSTRUCTIONS_ECHO=1` for endpoints that echo
submitted Responses instructions. Each completed response is compared with the
submitted instruction text before its tool calls are delivered. Requests with
no system instructions explicitly send an empty string in this mode. Setting
`OPENCOLLAB_INSTRUCTIONS_AUDIT_DIR` stores each response's requested and returned
instructions, model controls, and usage under its response identifier. This
opt-in check detects a gateway replacing the submitted role instructions during
a remote model call. Repository revisions, database identifiers, transactions,
uniqueness, type checks, and local request tests cannot observe that remote
response rewrite. The check remains at the provider delivery boundary.

## Candidate workspaces

Workflows can supply `candidate_workspace` to `OpenCollab.workflow()`. The
backend creates isolated candidate leases and reports source evidence through
`source_diff(exclude_paths)`. Source change detection uses this same evidence,
including for environments outside Git repositories. A backend with `adopt_run`
receives the complete selected `CandidateRun`, so equal file patches can still
identify different candidate environments. Existing patch backends keep their
`adopt(diff, preserve_paths)` path. Candidate execution continues to compare the
source before and after each lease, restore unintended source edits, and retain
the workflow's executable verification tools.

`git_diff` can display observed filesystem changes when a non-Git environment
provides `get_filesystem_diff(path, stat_only)`. The displayed evidence is limited
to the requested path and the tool's normal output allowance.

Terminal traces retain null token and step limits in unbounded mode while still
recording final usage and stop reasons.

## Controlled workflow agent calls

`RunControl` and `WorkflowContext.agent_run` are available in OpenCollab 0.9.4
and newer. See the [package installation guide](../opencollab/README.md#install).

`WorkflowContext.agent_run` runs a fresh session and returns its execution
result, including its stop reason, tokens, steps, session identity and cleanup
state. Existing `ctx.agent` calls keep their text or structured-output result.

```python
from opencollab import OpenCollab, RunControl, workflow
from opencollab.tools import builtin_tools

@workflow(name="incremental-edit")
async def incremental_edit(ctx, inputs):
    tools = builtin_tools("bash", "file_read", "apply_patch", "git_diff", headless=False)
    first = await ctx.agent_run(
        inputs["first_task"], label="first", budget=100_000,
        tools=tools,
        max_steps=40, system_prompt="Implement the assigned change and verify it.",
        run_control=RunControl(initial_soft_budget_tokens=80_000),
    )
    if not first.workspace_ready:
        return {"status": "cleanup_failed", "first": first.status}
    second = await ctx.agent_run(inputs["second_task"], label="second", budget=100_000, tools=tools)
    return {"first": first.status, "second": second.status}

result = await OpenCollab("/path/to/application").workflow(
    incremental_edit, {"first_task": "...", "second_task": "..."},
    agent_profile="single2", budget=200_000, concurrency=1, limit_mode="explicit",
)
```

Each call has a new conversation. Calls using `isolation=False` continue over
the same application files. Pass explicit tools to each role as with
`ctx.agent`. A per-call system prompt replaces the default profile/role prompt
exactly; the selected profile still supplies tool configuration, shaping and
safety. Omitting the prompt retains the existing profile composition.

The workflow reserves a hard budget before the session starts. The actual
grant can be lower than the requested budget because other calls hold shares
of the pool. `RunControl` receives this actual grant and can expand the soft
allowance within it. Unused reservations become available after call cleanup.
Budget refusal and waiting for a concurrency slot remain distinct operations.

Native clients returned by `OpenCollab.create_model_client()` receive a fresh
provider retry time allowance for each SDK agent or workflow run. Sessions,
summaries, and candidates within one workflow share that allowance, including
when other SDK runs use the same connection concurrently. Direct
`model.complete(...)` calls retain the model's existing retry allowance.
OC-created agent and auxiliary summary transports share the agent's allowance
as well.

A timed-out call may have provider or tool work still settling. `agent_run`
waits up to `cleanup_timeout` for the call's owned tasks. With incomplete
cleanup it returns a failed result and prevents further work in the shared
context. The workflow still owns the pending tasks and their budget until
they settle. `cleanup_complete` distinguishes a stable result from a usage
snapshot. External task cancellation propagates to the caller.

Execution completion is separate from application correctness. Workflows
should return their executable verification result alongside these session
records. `ctx.run_id` identifies the current workflow run; child contexts
retain the parent run identity while sessions have separate identities.
