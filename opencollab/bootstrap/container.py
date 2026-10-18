"""Composition root for OpenCollab.

Wires concrete adapters into the application use cases. This module owns the
session-construction *core* — the only place ``LLMClient`` is instantiated, so
``build_session_runtime`` and its model-client wiring live here together (tests
monkeypatch ``container.LLMClient`` and rely on that). The rest of the
composition root is split into focused siblings and re-exported here so
``from opencollab.bootstrap.container import X`` keeps resolving every name:

- ``tool_registry``      — tool-name -> Tool resolution + curated name sets.
- ``context_builder``    — ``SpawnConfig`` + ``ContextBuilder`` (role -> Agent).
- ``runtime_context``    — ``RuntimeContext`` + workspace safety policy.
- ``session_factory``    — ``build_spawn_session`` / ``DefaultSessionFactory`` +
                           run-folder transcript helpers.
- ``scheduler_factory``  — ``build_scheduler`` (the CLI and SDK entry point).

``session_factory`` and ``scheduler_factory`` import back from this module, so
they are re-exported lazily via module ``__getattr__`` (PEP 562) to keep import
order acyclic regardless of which module is imported first.
"""

from __future__ import annotations

import copy
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from opencollab.adapters.env import Environment, LocalEnvironment
from opencollab.adapters.llm import LLMClient, is_context_overflow_error
from opencollab.adapters.llm.anthropic_provider import _anthropic_thinking_is_always_on
from opencollab.adapters.llm.providers import is_anthropic
from opencollab.adapters.llm.retry import RetryTimeBudget
from opencollab.adapters.skills.file_skill_store import FileSkillStore
from opencollab.adapters.skills.null_skill_store import NullSkillStore
from opencollab.adapters.storage import SessionStore
from opencollab.application.autosave import AutoSaveSubscriber
from opencollab.application.compaction_summary import ReadTimeSummarizer
from opencollab.application.event_bus import EventBus
from opencollab.application.exception_notes import add_exception_note
from opencollab.application.ports import (
    AskUserPort,
    EventPublisherPort,
    LLMPort,
    PermissionPort,
    SafetyPolicyPort,
    SessionStorePort,
    ShaperPort,
    SkillStorePort,
    TracePort,
)
from opencollab.application.run_control import RunControl
from opencollab.application.session import SessionRuntime
from opencollab.application.session_run import SessionRunUseCase
from opencollab.application.shaping import (
    DEFAULT_COMPACT_BUFFER_TOKENS,
    DEFAULT_HISTORY_KEEP_RECENT_GROUPS,
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    DEFAULT_TOOL_CLEAR_KEEP_RECENT,
    DEFAULT_TOOL_RESULT_BUDGET,
    AutoCompactShaper,
    OldHistorySnipShaper,
    PerToolResultBudgetShaper,
    ShaperPipeline,
    ToolOutputClearShaper,
    approx_messages_tokens,
    history_trigger_target,
)
from opencollab.application.tool_execution import ToolExecutionUseCase
from opencollab.bootstrap.context_builder import ContextBuilder, SpawnConfig
from opencollab.bootstrap.context_policy import ContextPolicy
from opencollab.bootstrap.runtime_context import (
    RuntimeContext,
    build_runtime_context,
    build_workspace_safety_policy,
)
from opencollab.bootstrap.tool_registry import (
    COMPACTABLE_TOOL_NAMES,
    build_tools_for_role,
)
from opencollab.domain.agent import DEFAULT_MAX_TOKENS_PER_STEP, Agent
from opencollab.domain.session import SessionState
from opencollab.domain.token_estimation import estimate_request_tokens

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    # Re-exported at runtime via ``__getattr__`` (see bottom of module); declared
    # here so static tooling sees them as defined names in ``__all__``.
    from opencollab.bootstrap.scheduler_factory import build_scheduler
    from opencollab.bootstrap.session_factory import (
        DefaultSessionFactory,
        SnapshotSessionError,
        agent_save_path,
        build_session,
        build_spawn_session,
        load_session,
        make_run_dir,
        snapshot_session,
    )

def _build_initial_state(
    agent: Agent,
    seed_user_messages: list[dict[str, Any]] | None = None,
    seed_system_messages: list[dict[str, Any]] | None = None,
) -> SessionState:
    messages: list[dict[str, Any]] = (
        list(seed_system_messages)
        if seed_system_messages
        else [{"role": "system", "content": agent.system_prompt}]
    )
    if seed_user_messages:
        messages.extend(seed_user_messages)
    return SessionState(messages=messages)

def _resolve_llm(
    agent: Agent,
    llm: LLMPort | None,
    llm_timeout: float | None,
    provider_retry_budget: Any | None = None,
) -> LLMPort:
    """The injected ``llm`` if given, else a fresh ``LLMClient`` for the agent."""
    if llm is not None:
        return llm
    return LLMClient(
        model=agent.model,
        api_key=agent.api_key,
        base_url=agent.base_url,
        provider=agent.provider,
        wire_protocol=getattr(agent, "wire_protocol", "chat_completions"),
        max_retries=getattr(agent, "llm_max_retries", 3),
        request_timeout=llm_timeout,
        connect_timeout=getattr(agent, "llm_connect_timeout", 30.0),
        first_event_timeout=getattr(agent, "llm_first_event_timeout", 180.0),
        stream_idle_timeout=getattr(agent, "llm_stream_idle_timeout", 180.0),
        stream_chat=getattr(agent, "llm_stream_chat", False),
        context_window=getattr(agent, "context_window", None),
        provider_error_time_budget=getattr(agent, "provider_error_time_budget", 0.0),
        provider_retry_budget=provider_retry_budget,
    )

def _build_summarizer(
    agent: Agent,
    llm: LLMPort | None,
    resolved_llm: LLMPort,
    llm_timeout: float | None,
    auto_save_path: str | None,
    provider_retry_budget: Any | None = None,
    completion_handler: Callable[..., Awaitable[Any]] | None = None,
    output_limit: Callable[..., int] | None = None,
) -> ReadTimeSummarizer:
    """Build the read-time summarizer that powers ``AutoCompactShaper``.

    The session awaits the structured handoff prompt through its completion
    handler. Standalone synchronous callers retain a bridge, so each owned
    summary client is created and closed inside the completion coroutine.
    """
    reasoning_effort = getattr(agent, "reasoning_effort", None)
    summary_extra = (
        {"reasoning_effort": reasoning_effort}
        if reasoning_effort is not None
        else {}
    )
    # Preserve native settings for models whose summary request requires thinking.
    if is_anthropic(agent.provider) and _anthropic_thinking_is_always_on(agent.model):
        summary_extra.update(
            thinking=True,
            thinking_params=getattr(agent, "thinking_params", None)
            or {"thinking": {"type": "adaptive"}},
        )
    if llm is not None:

        async def _summary_complete(
            request: list[dict[str, Any]], *, on_response: Callable[[Any], None] | None = None, **kwargs: Any
        ) -> Any:
            response = await resolved_llm.complete(
                request,
                temperature=0.0,
                **(summary_extra | kwargs),
            )
            if on_response is not None:
                on_response(response)
            return response
    else:

        async def _summary_complete(
            request: list[dict[str, Any]], *, on_response: Callable[[Any], None] | None = None, **kwargs: Any
        ) -> Any:
            client = LLMClient(
                model=agent.model,
                api_key=agent.api_key,
                base_url=agent.base_url,
                provider=agent.provider,
                wire_protocol=getattr(agent, "wire_protocol", "chat_completions"),
                max_retries=getattr(agent, "llm_max_retries", 3),
                request_timeout=llm_timeout,
                connect_timeout=getattr(agent, "llm_connect_timeout", 30.0),
                first_event_timeout=getattr(agent, "llm_first_event_timeout", 180.0),
                stream_idle_timeout=getattr(agent, "llm_stream_idle_timeout", 180.0),
                stream_chat=getattr(agent, "llm_stream_chat", False),
                context_window=getattr(agent, "context_window", None),
                provider_error_time_budget=getattr(agent, "provider_error_time_budget", 0.0),
                provider_retry_budget=provider_retry_budget,
            )
            primary_failure: BaseException | None = None
            try:
                response = await client.complete(
                    request,
                    temperature=0.0,
                    **(summary_extra | kwargs),
                )
                if on_response is not None:
                    on_response(response)
                return response
            except BaseException as exc:
                primary_failure = exc
                raise
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        outcome = close()
                        if inspect.isawaitable(outcome):
                            await outcome
                    except BaseException as close_failure:
                        if primary_failure is None:
                            raise
                        add_exception_note(
                            primary_failure,
                            f"summary client close also failed: {type(close_failure).__name__}: {close_failure}",
                        )

    async def _bounded_complete(request: list[dict[str, Any]]) -> Any:
        kwargs = dict(summary_extra)
        if context_window and context_window > 0:
            if completion_handler is not None:
                kwargs["_summary_output_limit"] = _window_output_limit
            else:
                kwargs["max_output_tokens"] = _window_output_limit(request)
        if completion_handler is None:
            return await _summary_complete(request, **kwargs)
        return await completion_handler(_summary_complete, request, **kwargs)

    def _estimate_summary_tokens(request: list[dict[str, Any]]) -> int:
        estimator = getattr(resolved_llm, "estimate_request_tokens", None)
        if callable(estimator):
            return estimator(
                request,
                thinking=summary_extra.get("thinking", False),
                thinking_params=summary_extra.get("thinking_params"),
            )
        return estimate_request_tokens(request)

    def _window_output_limit(request: list[dict[str, Any]]) -> int:
        return min(max_output_tokens, context_window - _estimate_summary_tokens(request))

    def _summary_output_limit(request: list[dict[str, Any]]) -> int:
        assert output_limit is not None
        return output_limit(
            request, None, thinking=summary_extra.get("thinking", False),
            thinking_params=summary_extra.get("thinking_params"),
        )

    context_window = getattr(resolved_llm, "context_window", lambda: None)()
    max_output_tokens = getattr(agent, "max_tokens_per_step", DEFAULT_MAX_TOKENS_PER_STEP)
    return ReadTimeSummarizer(
        _bounded_complete,
        transcript_path=auto_save_path,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        estimate_tokens=_estimate_summary_tokens,
        output_limit=_summary_output_limit if output_limit is not None else None,
    )

def _history_compaction_settings(
    resolved_llm: LLMPort, history_trigger_tokens: int | None = None,
    *, max_output_tokens: int = DEFAULT_MAX_TOKENS_PER_STEP,
) -> dict[str, Any]:
    """The history-compaction thresholds a session will actually run under.

    One resolution point for two consumers: the shaper wiring below, which hands
    the numbers to the reactive layers, and the trajectory record, which writes
    them down. Deriving them twice would let the recorded value drift from the
    enforced one, which is the whole failure this record exists to close.
    """
    context_window = getattr(resolved_llm, "context_window", lambda: None)()
    allowances: dict[str, int] = {}
    if context_window and context_window > 0:
        # Fixed large-window headroom can consume a small window entirely.
        # Start with its configured output, restoring the standard reserve as
        # the window grows. Cap the input safety buffer at a third of the window.
        allowances = {
            "output_reserve": min(
                DEFAULT_OUTPUT_RESERVE_TOKENS,
                max_output_tokens + max(0, context_window - DEFAULT_OUTPUT_RESERVE_TOKENS),
            ),
            "buffer": min(DEFAULT_COMPACT_BUFFER_TOKENS, context_window // 3),
        }
    history_trigger, history_target = history_trigger_target(context_window, **allowances)
    if history_trigger_tokens is not None and context_window and context_window > 0:
        history_trigger = min(history_trigger, history_trigger_tokens)
        history_target = int(history_trigger * 0.75)
    return {
        "context_window_tokens": context_window,
        "history_trigger_tokens": history_trigger,
        "history_target_tokens": history_target,
        # Record whether thresholds came from a known window or fixed defaults.
        "history_thresholds_from": (
            ("context_window_with_run_control" if history_trigger_tokens is not None else "context_window")
            if context_window and context_window > 0
            else "fixed_default"
        ),
    }

def _trace_history_compaction(
    tracer: TracePort | None,
    *,
    aid: int,
    agent: Agent,
    resolved_llm: LLMPort,
    shaper_injected: bool,
    context_policy: ContextPolicy,
    history_trigger_tokens: int | None = None,
) -> None:
    """Record, once per session, the compaction thresholds it runs under.

    The reactive history layers scale to the active model's window, so two arms
    of one experiment can compact an order of magnitude apart -- a 1,015,576-token
    window triggers at 982,576, a model the capability table does not recognise
    falls back to the fixed 120,000 -- and nothing on disk said which. Recovering
    it meant guessing the window and redoing the arithmetic. This writes both
    thresholds, the ``context_window`` they came from, and the branch that
    produced them, keyed by the same ``aid`` every other record carries.

    Observation only, and guarded: a record that cannot be built must not change
    how the session runs.
    """
    if tracer is None:
        return
    try:
        payload: dict[str, Any] = {
            "aid": aid,
            "model": getattr(agent, "model", None),
            "context_policy": None if shaper_injected else context_policy.name,
        }
        if not shaper_injected and not context_policy.history_compaction:
            payload.update(
                {
                    "context_window_tokens": None,
                    "history_trigger_tokens": None,
                    "history_target_tokens": None,
                    "history_thresholds_from": "disabled_by_context_policy",
                }
            )
        elif shaper_injected:
            # Injected shapers own their thresholds, so these settings are unknown.
            payload.update(
                {
                    "context_window_tokens": None,
                    "history_trigger_tokens": None,
                    "history_target_tokens": None,
                    "history_thresholds_from": "injected_shaper",
                }
            )
        else:
            payload.update(_history_compaction_settings(
                resolved_llm, history_trigger_tokens,
                max_output_tokens=getattr(agent, "max_tokens_per_step", DEFAULT_MAX_TOKENS_PER_STEP),
            ))
        tracer.log_step(step_type="session.history_compaction", payload=payload)
    except Exception as exc:  # noqa: BLE001 - observability is non-authoritative
        logger.error("history compaction trace failed: %s", exc)

def _build_default_shaper(
    resolved_llm: LLMPort,
    summarizer: ReadTimeSummarizer,
    *,
    preserve_tool_result_tail: bool = False,
    history_compaction: bool = True,
    tool_result_budget: int = DEFAULT_TOOL_RESULT_BUDGET,
    history_trigger_tokens: int | None = None,
    max_output_tokens: int = DEFAULT_MAX_TOKENS_PER_STEP,
) -> ShaperPort:
    """Assemble the default lazy-degradation shaper pipeline.

    ``history_compaction=False`` keeps only the per-tool-result budget, and
    ``tool_result_budget`` sets that budget; both are how a team file's
    context policy reaches this wiring (see ``bootstrap.context_policy``).

    Cheapest/lowest-loss first: per-tool-result budget bounds any one result;
    the reactive history layers then bound the *total* view once it crosses the
    trigger — clear old tool *content* in place (lowest loss) → snip whole old
    tool turns → auto-compact (summarize the remaining old span via the
    Claude Code-derived compaction prompt). All read-time over a copy; transcript stays full for lossless
    resume.

    Every layer here is pressure-triggered: each no-ops until an estimated
    context size crosses its trigger. ``EagerToolOutputClearShaper`` is
    deliberately not among them, although it was. It ran unconditionally on
    every call, and the two costs it carries are both paid at low context,
    where there is no pressure to relieve:

    * It rewrites history. Each call stubs the one result that has just aged
      past its keep-recent window, so the message at that depth differs from
      the message sent last call — the provider's prompt prefix stops matching
      there and the whole tail after it is re-read at full input price, every
      call. Append-only history matches to the end. The layers below rewrite
      too, but only once pressure is real, and against a window that is
      genuinely too large to send.
    * The model re-reads what was cleared. Runs on this repository's own tasks
      showed the same file re-read eight and nine times while the context peak
      sat at about two percent of the window, so the tokens the clearing saved
      were spent again on fetching the same bytes back.

    Turning it off hands its job to the layers below, which do the same work on
    the same messages when the context is actually large. It is not deleted:
    ``opencollab.application.shaping`` still exports it, and a caller that
    wants an always-on front rung can compose one.
    """
    # Resolve thresholds once for both runtime wiring and trajectory records.
    per_result = PerToolResultBudgetShaper(
        tool_result_budget,
        preserve_tail=preserve_tool_result_tail,
    )
    if not history_compaction:
        return ShaperPipeline((per_result,))
    settings = _history_compaction_settings(
        resolved_llm, history_trigger_tokens, max_output_tokens=max_output_tokens,
    )
    history_trigger = settings["history_trigger_tokens"]
    history_target = settings["history_target_tokens"]
    # A small input allowance cannot retain the same number of maximum-size
    # tool exchanges as a large window. Keep the latest location/evidence pair.
    affordable_groups = max(2, history_target // (DEFAULT_TOOL_RESULT_BUDGET // 4))

    # The runner scopes the shared estimator to this session's provider
    # projection and measured usage. Reactive layers and reports use the same
    # additive weights; standalone callers retain the local fallback.
    history_kwargs = {
        "estimate_tokens": approx_messages_tokens,
        "trigger_tokens": history_trigger,
        "target_tokens": history_target,
        "keep_recent_groups": min(DEFAULT_HISTORY_KEEP_RECENT_GROUPS, affordable_groups),
    }
    return ShaperPipeline(
        (
            per_result,
            ToolOutputClearShaper(
                compactable_tools=COMPACTABLE_TOOL_NAMES,
                keep_recent=min(DEFAULT_TOOL_CLEAR_KEEP_RECENT, affordable_groups),
                **history_kwargs,
            ),
            OldHistorySnipShaper(**history_kwargs),
            AutoCompactShaper(summarizer=summarizer, **history_kwargs),
        )
    )

def agent_trace_view(tracer: TracePort | None, aid: int, agent: Agent) -> TracePort | None:
    """Bind ``tracer`` to one agent so every record it writes names that agent.

    A tracer without ``for_agent`` (a test double, a caller's own recorder) is
    returned unchanged.
    """
    for_agent = getattr(tracer, "for_agent", None)
    return for_agent(aid, agent) if callable(for_agent) else tracer

def build_skill_store(workspace: str | None) -> SkillStorePort:
    """Resolve the workspace's ``skills/`` directory into a skill store.

    A repo opts into skills by adding a ``skills/`` directory at its root. When
    that directory exists we scan it with ``FileSkillStore``; otherwise (no
    workspace, or no ``skills/`` dir) we return a ``NullSkillStore`` so behavior
    is byte-identical to today — no catalog appears and no body resolves. This
    is the only place that picks the concrete store type.
    """
    if workspace:
        skills_root = Path(workspace) / "skills"
        if skills_root.is_dir():
            return FileSkillStore(skills_root)
    return NullSkillStore()

def build_session_runtime(
    *,
    agent: Agent,
    env: Environment | None = None,
    tracer: TracePort | None = None,
    max_budget_tokens: int | None = 1_000_000,
    max_steps: int | None = 100,
    auto_save_path: str | None = None,
    event_sink: EventPublisherPort | None = None,
    permission_policy: PermissionPort | None = None,
    ask_policy: AskUserPort | None = None,
    safety_policy: SafetyPolicyPort | None = None,
    llm: LLMPort | None = None,
    llm_timeout: float | None = 600.0,
    provider_retry_budget: Any | None = None,
    store: SessionStorePort | None = None,
    auto_save_callback: Callable[[], None] | None = None,
    auto_save_prepare_callback: Callable[
        [], Callable[[], None] | None
    ] | None = None,
    aid: int = -1,
    seed_user_messages: list[dict[str, Any]] | None = None,
    seed_system_messages: list[dict[str, Any]] | None = None,
    shaper: ShaperPort | None = None,
    team_budget_exhausted: Callable[[], bool] | None = None,
    agent_profile: Any | None = None,
    context_policy: ContextPolicy | None = None,
    run_id: str | None = None,
    run_control: RunControl | None = None,
) -> SessionRuntime:
    """Build a ``SessionRuntime`` with the same construction order
    ``Session.__init__`` used to perform inline.

    ``auto_save_callback`` is the bound method the facade exposes for direct
    autosave. ``auto_save_prepare_callback`` freezes a payload on the event
    loop and returns the file-I/O operation run by the subscriber worker.
    ``seed_system_messages`` retain source-level provenance for layered system
    context; ``seed_user_messages`` are startup user-context messages appended
    after them (e.g. a spawned agent's task);
    ``shaper`` reshapes the message list before each model call; without
    one, ``context_policy`` (default: the default policy) picks the layers.
    """
    # A session may narrow these controls during wind-down. Keep the reusable
    # template intact while retaining the tool objects and their resources.
    agent = copy.copy(agent)
    agent.tools = list(getattr(agent, "tools", ()) or ())
    agent.tool_choice = copy.deepcopy(getattr(agent, "tool_choice", None))
    resolved_context = context_policy if context_policy is not None else ContextPolicy()
    tracer = agent_trace_view(tracer, aid, agent)
    resolved_env = env if env is not None else LocalEnvironment()
    resolved_store: SessionStorePort = store if store is not None else SessionStore()

    event_bus = EventBus(event_sink)
    auto_save_subscriber: AutoSaveSubscriber | None = None
    if auto_save_path and auto_save_callback is not None:
        auto_save_subscriber = AutoSaveSubscriber(
            auto_save_callback,
            prepare_fn=auto_save_prepare_callback,
        )
        event_bus.subscribe(auto_save_subscriber)

    state = _build_initial_state(
        agent,
        seed_user_messages,
        seed_system_messages,
    )
    state.aid = aid

    # The main transport and each auxiliary summary belong to one session.
    if llm is None and provider_retry_budget is None:
        error_allowance = getattr(agent, "provider_error_time_budget", 0.0)
        if error_allowance > 0:
            provider_retry_budget = RetryTimeBudget(error_allowance)
    resolved_llm = _resolve_llm(agent, llm, llm_timeout, provider_retry_budget)

    if agent_profile is not None:
        safety_policy = agent_profile.wrap_safety(
            safety_policy,
            resolved_env.workspace,
        )
    tool_options: dict[str, Any] = {}
    if run_control is not None and run_control.tool_cancellation_cleanup_timeout is not None:
        tool_options["cancellation_cleanup_timeout"] = run_control.tool_cancellation_cleanup_timeout
    tool_execution = ToolExecutionUseCase(
        agent=agent,
        environment=resolved_env,
        state=state,
        event_publisher=event_bus,
        tracer=tracer,
        permission_policy=permission_policy,
        ask_policy=ask_policy,
        safety_policy=safety_policy,
        **tool_options,
    )
    runner = SessionRunUseCase(
        agent=agent,
        state=state,
        llm=resolved_llm,
        event_publisher=event_bus,
        tool_execution=tool_execution,
        tracer=tracer,
        max_budget_tokens=max_budget_tokens,
        max_steps=max_steps,
        shaper=shaper,
        team_budget_exhausted=team_budget_exhausted,
        run_control=run_control,
        run_id=run_id,
        # Inject the adapter's context-overflow classifier to preserve the
        # application layer's dependency direction.
        is_context_overflow=is_context_overflow_error,
        # Bound each generation by the provider timeout to preserve the run budget.
        per_call_timeout=llm_timeout,
    )
    if run_control is not None:
        event_bus.subscribe(runner._observe_runtime_event)
    summarizer = _build_summarizer(
        agent,
        llm,
        resolved_llm,
        llm_timeout,
        auto_save_path,
        provider_retry_budget,
        completion_handler=runner._invoke_summary,
        output_limit=runner._request_output_limit,
    )
    shaper_options = resolved_context.shaper_options()
    shaper_options["max_output_tokens"] = getattr(agent, "max_tokens_per_step", DEFAULT_MAX_TOKENS_PER_STEP)
    if run_control is not None and run_control.history_trigger_tokens is not None:
        shaper_options["history_trigger_tokens"] = run_control.history_trigger_tokens
    resolved_shaper: ShaperPort
    if shaper is not None:
        resolved_shaper = shaper
    elif agent_profile is not None:
        resolved_shaper = agent_profile.build_shaper(
            resolved_llm, summarizer, **shaper_options
        )
    else:
        resolved_shaper = _build_default_shaper(
            resolved_llm, summarizer, **shaper_options
        )
    # Profile shapers use ``_build_default_shaper`` and its derived thresholds.
    # Only caller-supplied shapers have independently chosen thresholds.
    _trace_history_compaction(
        tracer,
        aid=aid,
        agent=agent,
        resolved_llm=resolved_llm,
        shaper_injected=shaper is not None,
        context_policy=resolved_context,
        history_trigger_tokens=None if run_control is None else run_control.history_trigger_tokens,
    )
    runner.shaper = resolved_shaper

    return SessionRuntime(
        state=state,
        event_bus=event_bus,
        llm=resolved_llm,
        store=resolved_store,
        tool_execution=tool_execution,
        runner=runner,
        auto_save_path=auto_save_path,
        run_id=run_id,
        auto_save_subscriber=auto_save_subscriber,
        owns_llm=llm is None,
        context_summarizer=summarizer,
    )

# Factory modules import this module. Defer their re-exports through
# PEP 562 ``__getattr__`` to avoid an import cycle.

_SESSION_FACTORY_EXPORTS = frozenset(
    {
        "DefaultSessionFactory",
        "SnapshotSessionError",
        "agent_save_path",
        "build_session",
        "build_spawn_session",
        "load_session",
        "make_run_dir",
        "snapshot_session",
    }
)
_SCHEDULER_FACTORY_EXPORTS = frozenset({"build_scheduler"})

def __getattr__(name: str) -> Any:
    if name in _SESSION_FACTORY_EXPORTS:
        from opencollab.bootstrap import session_factory

        return getattr(session_factory, name)
    if name in _SCHEDULER_FACTORY_EXPORTS:
        from opencollab.bootstrap import scheduler_factory

        return getattr(scheduler_factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ContextBuilder",
    "DefaultSessionFactory",
    "RuntimeContext",
    "SessionRuntime",
    "SnapshotSessionError",
    "SpawnConfig",
    "agent_save_path",
    "make_run_dir",
    "build_runtime_context",
    "build_session",
    "build_session_runtime",
    "build_skill_store",
    "build_scheduler",
    "build_spawn_session",
    "build_tools_for_role",
    "build_workspace_safety_policy",
    "load_session",
    "snapshot_session",
]
