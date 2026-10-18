# Runtime design: context, limits, and steering

Use the [package guide](../opencollab/README.md) for API examples and the
[configuration guide](../configs/README.md) for settings. This guide explains
runtime invariants and the reasons behind them. Source links name modules and
symbols rather than revision-dependent line numbers.

## Context assembly and the model's view

A session retains its conversation in `SessionState.messages`. Before a model
call, shaping builds a bounded view; it does not discard the stored tool results.
Steering may deliberately update the conversation as described below. A shaped
request and the persisted transcript therefore serve different purposes.

[ContextBuilder](../opencollab/bootstrap/context_builder.py) constructs an
ordered `ContextPlan`. A profiled role starts with the profile prompt, followed
by its role card. The team section, skill catalog, and project map appear only
when applicable and non-empty. A delegated task becomes a startup user message.
Tool schemas reach the provider separately; they are not copied into the prompt.

[ContextPlan](../opencollab/domain/context.py) assembles by `ContextPosition`,
not by layer name. `Agent.system_prompt` exposes joined system text, while
`startup_system_messages()` preserves the individual fragments in a live
session. Both system and user seeds carry `_ctx` name, layer, and priority.
This preserves source identity after construction; the model request adapters
project internal messages into their provider's format.

| Source | Default priority | Position |
| --- | ---: | --- |
| Profile / role identity | 100 | System |
| Declared team | 90 | System |
| Available skills | 85 | System |
| Delegated task | 80 | User context |
| Project map | 30 | System |

`PIN_FLOOR = 70` protects high-priority sources from history compaction.
Project context is lower priority. An ordinary user turn is not automatically
a tagged startup source: the lead receives its task through `add_user_message`,
whereas a spawned worker receives a tagged task from the session factory.
Do not assume the same goal string has the same shaping treatment on both paths.
External evaluation packages assemble benchmark fields into that string;
OpenCollab owns the session, not the benchmark's task construction.

There is no general deferred context loader or memory/RAG source in the current
plan. Add a source when there is a real producer and a defined retention need;
the old loader proposal's empty placeholders and timing axes are not API contracts.
Skills use a catalog plus a generic tool result instead; see
[skill loading and its rationale](../skills/README.md#why-a-catalog-and-one-tool).

### Shaping under pressure

The default pipeline in
[`_build_default_shaper`](../opencollab/bootstrap/container.py) applies:

1. A cap on each tool result, including oversized rejected tool arguments.
2. Pressure-triggered clearing of old compactable tool output.
3. Removal of eligible old tool-exchange groups.
4. Summarization of the remaining eligible span through the supplied summarizer.

Call/result pairing and recent evidence constrain what can be removed. The
pipeline uses the session's provider-aware, usage-calibrated token estimate;
standalone shapers retain their local estimate. Each later layer sees the view
produced by earlier layers, so it can do nothing if pressure has already eased.

For a known window, `history_trigger_target` subtracts output headroom and a
safety buffer, then targets 75% of that trigger. The gap avoids immediate repeated
compaction. Session wiring scales the reserves and recent-group count for small
windows and honors a lower `RunControl.history_trigger_tokens`. Unknown windows
use the fixed 120,000 / 90,000 trigger and target. The effective settings are
recorded by `session.history_compaction`; use those records for comparisons.

`no_history_compaction` keeps the per-result cap and removes the history layers.
`EagerToolOutputClearShaper` remains available for explicit composition but is
**absent from the default pipeline**. Always clearing old results can change a
previously reusable prompt prefix and provoke rereads even at low context usage.
These costs motivated pressure-triggered history changes.

A classified provider context overflow forces compaction and one retry. A
second overflow stops the session with its reason retained. Unrelated request
errors keep their own classification. A protected seed that is itself too large
can still prevent progress; compaction does not make arbitrary inputs fit.

The summarizer uses the structured handoff prompt in
[compaction_prompt.py](../opencollab/application/compaction_prompt.py). It keeps
intent, decisions, errors, user directions, pending work, and a next action.
The wrapper excludes tool execution and strips the analysis wrapper from the
returned summary. Its third-party provenance remains in
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Token accounting and allocation

`used_tokens` measures cumulative input plus output across calls;
`context_tokens` reports input size for a call. Neither is a dollar budget.
Repeatedly sending history spends tokens again even if its stored text is
unchanged. Provider cache counters are retained without double-counting input;
missing usage may be estimated and marked as such. Reasoning follows the
provider's output accounting. See [evaluation runtime](evaluation-runtime.md)
for late usage, cancellation, and cleanup ownership.

Allocation depends on how the run is constructed:

| Mode | Allocation rule |
| --- | --- |
| Prebuilt declared team, shared pool | Each seat has a cumulative cap `min(total, int(c * total / N))`, currently `c = 1`. Seating reserves nothing; turn grants are bounded by both remaining pool and remaining seat allowance. |
| Dynamic roster | Child creation synchronously reserves at most `min(total, max(10_000, total // 4))`, bounded by the unallocated remainder. Agent 0's running turn can lease the remaining pool; it has no declared-roster equal-share cap. |
| Team with declared role budgets | Each role has its own allowance. All roles must have one; a separate SDK pool is rejected. |
| Workflow | Agent calls reserve from the workflow pool before execution and release unused reservations after cleanup. Explicit and environment-controlled limits are separate modes. |

The declared-team rule makes a seat comparable to a solo agent when the team
pool is `N` times the solo allowance. It also distinguishes an unused role from
a role whose allowance ran out; cap-refusal records preserve that distinction.
It does not establish equal realized work or collaboration. The
[Adherence guide](adherence.md) separates configured conditions from observations.

Reservation bookkeeping is in
[SchedulerPersistenceMixin](../opencollab/application/_scheduler_persistence.py)
and the pure allocation functions are in
[domain/scheduler.py](../opencollab/domain/scheduler.py). Dynamic reservations
remain owned while late provider work settles, so returning a stopped result
must not make unaccounted tokens available to another child.

Session prechecks enforce finite token and step ceilings between calls. A
shared-pool team also checks aggregate usage. Calls already in flight can exceed
the remaining allowance; these are graceful execution limits, not a real-time
billing cap. `STOPPED` retains a reason and available partial results rather than
being treated as successful task verification. Workflow and profile handling of
unbounded limits is documented in
[evaluation runtime](evaluation-runtime.md) and [controlled calls](controlled-workflow-agents.md).

## Steering and structured completion

[Steering](../opencollab/application/steering.py) takes counters and tool
capabilities as values. It returns a message, optional tool choice, and escalation
level. Budget/progress reporting and reads-without-write nudges have independent
switches. Budget cadence can be every step, threshold crossings, or off; threshold
crossings need a finite budget. Turning write nudges off also disables their
structured-submission escalation, without changing budget status reporting.

With write nudges enabled, 8 reads can prompt an edit and 16 can require one.
Structured-output sessions can instead be directed to submit their gathered
result. Landed edits and unknown write effects affect which instruction is
appropriate. The run loop also restricts tools for a hard escalation; forcing
some tool call alone does not ensure the required action.

A trailing user message receives the steering text in the stored conversation.
Ordinary continuations add it only to the shaped request. Models requiring a
stable thinking prefix retain sent steering in the transcript as well. This
provider constraint supersedes the early proposal's unconditional claim that
all continuation steering is ephemeral.

Workflow sessions are fresh calls, with profile/role composition or a supplied
system prompt. A structured call first explores with its tools and capture tool.
Only a missing valid capture triggers a corrective session restricted to the
capture tool. That session keeps its own system instructions and copies prior
exploration, so it can submit from actual evidence. A valid capture cancels
further exploration. A provider may reject forced tool choice and fall back to
`auto`; the call can still return no valid payload. See
[workflow_structured.py](../opencollab/application/workflow_structured.py).

### Why these choices were made

The June 23 steering proposal recorded a run (`analyst-p6749`, code `dae2edf`,
`django-11564`, kimi-k2.6) with 55 file reads and 52 searches, no edits over the
first 118 steps, a prose fix at step 235, and an empty patch at the 1,800-second
limit. That observation motivated testing reads-without-write intervention.
It is a historical failure case, not a controlled estimate of the intervention's
benefit. Keep model, task, prompt, limits, and actual tool behavior in any comparison.
Benchmark-specific final-write policies belong in the external evaluation package.

Deferred choices remain conditional: objective reinjection needs evidence that
its extra prompt content helps; a general context retention policy needs a real
additional source. Neither the old proposal nor an available port demonstrates
that those additions improve outcomes.

## Where to check a change

| Invariant | Executable evidence |
| --- | --- |
| Context sources, provenance, startup injection | `tests/runtime/test_context_source.py`, `test_context_builder.py`, `test_context_assembly_golden.py` |
| Pressure shaping, recent evidence, overflow recovery | `tests/runtime/` shaping, compaction, and context-overflow topics |
| Seat caps, reservations, prebuilt roster | `tests/workflows/test_scheduler_budget.py`, `test_prebuilt_team_topology.py` |
| Steering and structured submission | `tests/agents/test_session_run_loop_steering.py`, `tests/workflows/test_workflow_structured_output.py` |
| Resume and late-usage accounting | `tests/runtime/` persistence, provider, and usage topics |

Use the [test directory map](../tests/README.md) to locate related regressions
and the [testing guide](testing.md) for commands. These tests establish runtime
behavior; model-effect claims require separately recorded experiments.
