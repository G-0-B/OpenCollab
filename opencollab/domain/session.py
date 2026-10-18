from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from opencollab.domain.pending import PendingEventTable
from opencollab.domain.tool_loops import LoopState

MAX_SCOUT_LEDGER_CARDS = 256

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class SessionPhase(Enum):
    IDLE = "idle"
    PRECHECK = "precheck"
    CALLING_LLM = "calling_llm"
    HANDLING_RESPONSE = "handling_response"
    EXECUTING_TOOLS = "executing_tools"
    AWAITING_EVENTS = "awaiting_events"
    AUTOSAVING = "autosaving"
    DONE = "done"
    STOPPED = "stopped"
    ERROR = "error"

    def is_terminal(self) -> bool:
        return self in TERMINAL_PHASES

TERMINAL_PHASES = frozenset(
    {
        SessionPhase.DONE,
        SessionPhase.STOPPED,
        SessionPhase.ERROR,
    }
)

# transition_to validates these edges. set_phase is reserved for scheduler
# admission and snapshot restore. fail and cancel can terminate any phase.
# STOPPED carries the controlled-halt cause in terminal_reason and delivers
# an explicit failed result to the parent. ERROR represents an unhandled fault.
# AWAITING_EVENTS suspends without completing the turn. The scheduler resumes
# it once every pending row is filled.
PHASE_TRANSITIONS: dict[SessionPhase, frozenset[SessionPhase]] = {
    SessionPhase.IDLE: frozenset({SessionPhase.PRECHECK}),
    SessionPhase.PRECHECK: frozenset(
        {SessionPhase.CALLING_LLM, SessionPhase.STOPPED}
    ),
    SessionPhase.CALLING_LLM: frozenset(
        {SessionPhase.HANDLING_RESPONSE, SessionPhase.STOPPED}
    ),
    # AUTOSAVING is the empty-stop retry route: the turn produced nothing to
    # execute, so it autosaves like any step and loops back to PRECHECK.
    SessionPhase.HANDLING_RESPONSE: frozenset(
        {
            SessionPhase.EXECUTING_TOOLS,
            SessionPhase.DONE,
            SessionPhase.STOPPED,
            SessionPhase.AUTOSAVING,
        }
    ),
    SessionPhase.EXECUTING_TOOLS: frozenset(
        {SessionPhase.AUTOSAVING, SessionPhase.AWAITING_EVENTS}
    ),
    SessionPhase.AWAITING_EVENTS: frozenset({SessionPhase.AUTOSAVING}),
    # Save the step before either continuing or completing a submitted turn.
    SessionPhase.AUTOSAVING: frozenset({SessionPhase.PRECHECK, SessionPhase.DONE}),
    SessionPhase.DONE: frozenset({SessionPhase.IDLE}),
    SessionPhase.STOPPED: frozenset({SessionPhase.IDLE}),
    SessionPhase.ERROR: frozenset({SessionPhase.IDLE}),
}

class InvalidPhaseTransition(Exception):
    """Raised when the run loop attempts an edge absent from PHASE_TRANSITIONS."""

    def __init__(self, src: SessionPhase, dst: SessionPhase):
        self.src = src
        self.dst = dst
        super().__init__(f"Illegal session phase transition: {src.value} -> {dst.value}")

@dataclass
class TurnEnforcementState:
    """Per-turn progress and enforcement state, checkpointed and reset together.

    Counters are maintained regardless of ``enforcement_strength``. Wind-down
    latches live on ``SessionState`` to survive within-turn persistence and
    rollback, and reset alongside this record on a fresh user turn.
    """

    # Path-normalized call hashes drive repeated-call blocking.
    recent_call_hashes: list[str] = field(default_factory=list)
    loop_state: LoopState = field(default_factory=LoopState)
    # Unknown write outcomes avoid steering that assumes a known unchanged workspace.
    write_effect_unknown: bool = False
    last_progress_unknown: bool = False
    # Successful edits reset the read count used by write steering.
    reads_since_last_edit: int = 0
    # After an edit, steering requires structured submission instead of another
    # write. A new user turn resets this obligation.
    has_landed_write: bool = False
    # Repeated content or normalized calls are low-yield, as are
    # empty and no-match results.
    low_yield_since_progress: int = 0
    distinct_evidence_count: int = 0
    # Novelty memory for the sensor: holds both content and (tool, args) call
    # hashes, so a re-read at a shifted range still scores zero gain.
    seen_result_hashes: set[str] = field(default_factory=set)
    # Evidence cards let salvage recover a scout's gathered findings.
    scout_ledger: list[dict[str, Any]] = field(default_factory=list)
    # Count tool batches without a landed write or novel informative result.
    steps_since_progress: int = 0
    # Count loop-blocked calls toward the repeated-warning stop.
    loop_blocked_since_progress: int = 0

@dataclass(frozen=True)
class _UserTurnCheckpoint:
    message_count: int
    timestamp_count: int
    phase: SessionPhase
    terminal_reason: str | None
    pending_external_user_turn: dict[str, Any] | None
    pending_step_latency: float | None
    submitted_summary: str | None
    turn: TurnEnforcementState
    wind_down_done: bool
    wind_down_attempts: int
    wind_down_token_mark: int

@dataclass
class SessionState:
    messages: list[dict[str, Any]]
    # Native Anthropic requests retain their compacted view beside the complete
    # transcript. Later thinking is bound to this view, including after restore.
    anthropic_history: dict[str, Any] | None = field(default=None, kw_only=True)
    used_tokens: int = 0
    # Real size (provider ``input_tokens``) of the current context as of the
    # last LLM call. 0 means "no real measurement yet — fall back to estimate".
    context_tokens: int = 0
    # max_steps is a session-lifetime cap, like used_tokens.
    # reset_for_user_turn deliberately retains this count.
    step_count: int = 0
    # Session-lifetime tally of completions with recovered Kimi tool-call markup.
    markup_recovered: int = 0
    # Checkpoint, restore, and reset the per-turn enforcement state as one unit.
    turn: TurnEnforcementState = field(default_factory=TurnEnforcementState)
    # Durable within a turn so restore cannot grant extra protected submit calls.
    # The token mark measures submit-turn cost. A new user turn resets these latches.
    wind_down_done: bool = False
    # Counts protected provider calls allocated by the wind-down gate. It is
    # durable so a restore cannot re-grant the one compatibility retry.
    wind_down_attempts: int = 0
    wind_down_token_mark: int = 0
    # Session-lifetime latch for the protected call carved from the hard token
    # budget. Per-turn watchdog/low-yield wind-down state resets between user
    # turns, but the same lifetime reserve must never be allocated twice.
    budget_reserve_consumed: bool = field(default=False, kw_only=True)
    phase: SessionPhase = SessionPhase.IDLE
    # Terminal cause, cleared on resume to IDLE.
    terminal_reason: str | None = None
    aid: int = -1
    pending_events: PendingEventTable = field(default_factory=PendingEventTable)
    # Keep timestamps index-aligned in a sidecar so messages retain their API shape.
    message_timestamps: list[str] = field(default_factory=list)
    # Persist queued messages while the session is mid-turn,
    # then append them to provider history when delivery is safe.
    pending_user_messages: list[dict[str, Any]] = field(default_factory=list)
    # An external user turn has already been appended but no runner has begun
    # it yet. Unlike scheduler-owned ``pending_user_messages``, this records
    # the crash window between accepting a public turn and starting its driver.
    pending_external_user_turn: dict[str, Any] | None = None
    # First message position *after* the history visible when the active turn
    # began. It is retained only while a deferred turn is suspended so a
    # restored runner can keep answer lookup within that turn.
    active_turn_start_message_index: int | None = None
    # Elapsed provider/tool time for a deferred step whose process-local
    # PendingStep response was released while waiting for child results.
    pending_step_latency: float | None = None
    # Accepted completion for a turn still waiting on earlier deferred tools.
    submitted_summary: str | None = None

    def __post_init__(self) -> None:
        self._align_timestamps()

    def _align_timestamps(self) -> None:
        if len(self.message_timestamps) < len(self.messages):
            now = _now_iso()
            self.message_timestamps += [now] * (
                len(self.messages) - len(self.message_timestamps)
            )

    def enriched_messages(self, start: int = 0) -> list[dict[str, Any]]:
        """Messages with their creation ``timestamp`` merged in, for persistence.

        Re-aligns first so messages appended by bypassing ``append_message``
        (e.g. direct list mutation) still get a timestamp.
        """
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("message start must be a non-negative integer")
        self._align_timestamps()
        return [
            {**msg, "timestamp": ts}
            for msg, ts in zip(
                self.messages[start:],
                self.message_timestamps[start:],
            )
        ]

    def queue_pending_user_message(self, message: dict[str, Any]) -> None:
        self.pending_user_messages.append({**message, "timestamp": _now_iso()})

    def discard_pending_user_message(self, content: str) -> None:
        for index, message in enumerate(self.pending_user_messages):
            if message.get("content") == content:
                del self.pending_user_messages[index]
                return

    def discard_pending_user_message_id(self, message_id: str) -> None:
        for index, message in enumerate(self.pending_user_messages):
            if message.get("message_id") == message_id:
                del self.pending_user_messages[index]
                return

    def enriched_pending_user_messages(self) -> list[dict[str, Any]]:
        return [dict(message) for message in self.pending_user_messages]

    def append_queued_external_user_turn(self, content: str, *, defer: bool = False) -> None:
        """Atomically mark and append a public user turn awaiting its driver."""
        pending = {
            "turn_id": uuid.uuid4().hex,
            "status": "queued",
            "content": content,
        }
        self.pending_external_user_turn = pending
        if defer:
            return
        self.append_message({"role": "user", "content": content})
        pending["message_index"] = len(self.messages) - 1

    def consume_queued_external_user_turn(self) -> None:
        """Mark a queued external turn as claimed by the current runner."""
        if self.pending_external_user_turn is not None:
            if "message_index" not in self.pending_external_user_turn:
                self.append_message({"role": "user", "content": self.pending_external_user_turn["content"]})
            self.pending_external_user_turn = None

    def start_active_turn(self, message_index: int) -> None:
        """Persist the answer-lookup boundary for an active user turn."""
        self.active_turn_start_message_index = message_index

    def clear_active_turn(self) -> None:
        """Discard a turn boundary once that turn reaches a terminal phase."""
        self.active_turn_start_message_index = None
        self.pending_step_latency = None
        self.submitted_summary = None

    @property
    def is_done(self) -> bool:
        return self.phase == SessionPhase.DONE

    def append_message(self, message: dict[str, Any]) -> None:
        self.messages.append(message)
        self.message_timestamps.append(_now_iso())

    def replace_messages(self, messages: list[dict[str, Any]]) -> None:
        """Swap the conversation, re-deriving aligned timestamps.

        Precedence per message: an embedded ``timestamp`` (e.g. a resumed
        transcript) wins; else the prior timestamp of the same dict object
        (preserved across a slice-and-rebuild); else now.
        """
        prior = {id(m): ts for m, ts in zip(self.messages, self.message_timestamps)}
        rebuilt = [
            (
                ({k: v for k, v in m.items() if k != "timestamp"}, m["timestamp"])
                if isinstance(m, dict) and "timestamp" in m
                else (m, prior.get(id(m)) or _now_iso())
            )
            for m in messages
        ]
        self.messages = [m for m, _ in rebuilt]
        self.message_timestamps = [ts for _, ts in rebuilt]
        self.anthropic_history = None

    def add_used_tokens(self, tokens: int) -> None:
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("used token increment must be a non-negative integer")
        self.used_tokens += tokens

    def add_markup_recovered(self, count: int) -> None:
        self.markup_recovered += count

    def set_used_tokens(self, tokens: int) -> None:
        self.used_tokens = tokens

    def set_context_tokens(self, tokens: int) -> None:
        self.context_tokens = tokens

    def advance_step(self) -> int:
        self.step_count += 1
        return self.step_count

    def set_step_count(self, step_count: int) -> None:
        self.step_count = step_count

    def mark_done(self) -> None:
        self.phase = SessionPhase.DONE
        self.terminal_reason = "completed"

    def clear_done(self) -> None:
        self.resume_to_idle()

    def set_phase(self, phase: SessionPhase) -> None:
        """Unchecked phase assignment reserved for out-of-band sets: scheduler
        process birth/enqueue, snapshot/restore, and tests. Run-loop edges go
        through ``transition_to``; the abnormal terminal escapes go through
        ``fail``/``cancel``. Prefer those over poking ``set_phase`` directly.
        """
        self.phase = phase

    def transition_to(self, phase: SessionPhase, *, reason: str | None = None) -> None:
        """Validated run-loop transition. Raises ``InvalidPhaseTransition`` if
        the edge is absent from ``PHASE_TRANSITIONS``. ``reason`` records the
        ``terminal_reason`` detail when crossing into a terminal phase.
        """
        if phase not in PHASE_TRANSITIONS.get(self.phase, frozenset()):
            raise InvalidPhaseTransition(self.phase, phase)
        self.phase = phase
        if phase.is_terminal() and reason is not None:
            self.terminal_reason = reason

    def fail(self, reason: str | None = None) -> None:
        """Abnormal escape to ERROR from any phase. ERROR has no inbound edge in
        ``PHASE_TRANSITIONS`` by design — it is reachable only through this
        escape (and snapshot/restore). Callers pair it with a raised exception;
        ``reason`` typically carries that exception's description.
        """
        self.phase = SessionPhase.ERROR
        self.terminal_reason = reason or "error"

    def cancel(self, reason: str | None = None) -> None:
        """Out-of-band escape to STOPPED from any phase, used by the scheduler
        when an agent task is killed mid-loop. The in-loop cancel uses the
        validated PRECHECK -> STOPPED edge via ``transition_to`` instead.
        """
        self.phase = SessionPhase.STOPPED
        self.terminal_reason = reason or "cancelled"

    def resume_to_idle(self) -> None:
        """The single validated reset from a terminal phase back to IDLE for a
        fresh turn or re-run. No-op when the phase is not terminal. Clears the
        terminal_reason so it never leaks into the next turn.
        """
        if self.phase.is_terminal():
            self.transition_to(SessionPhase.IDLE)
            self.terminal_reason = None

    def checkpoint_user_turn(self) -> _UserTurnCheckpoint:
        """Capture only state changed while accepting a new user message."""
        return _UserTurnCheckpoint(
            message_count=len(self.messages),
            timestamp_count=len(self.message_timestamps),
            phase=self.phase,
            terminal_reason=self.terminal_reason,
            pending_external_user_turn=copy.deepcopy(self.pending_external_user_turn),
            pending_step_latency=self.pending_step_latency,
            submitted_summary=self.submitted_summary,
            # Deep-copy the whole per-turn window as one pristine, reusable unit.
            turn=copy.deepcopy(self.turn),
            wind_down_done=self.wind_down_done,
            wind_down_attempts=self.wind_down_attempts,
            wind_down_token_mark=self.wind_down_token_mark,
        )

    def restore_user_turn(self, checkpoint: _UserTurnCheckpoint) -> None:
        """Roll back an interrupted user-message append to its commit boundary."""
        del self.messages[checkpoint.message_count :]
        del self.message_timestamps[checkpoint.timestamp_count :]
        self.phase = checkpoint.phase
        self.terminal_reason = checkpoint.terminal_reason
        self.pending_external_user_turn = copy.deepcopy(
            checkpoint.pending_external_user_turn
        )
        self.pending_step_latency = checkpoint.pending_step_latency
        self.submitted_summary = checkpoint.submitted_summary
        # A fresh copy each restore, so the checkpoint stays reusable.
        self.turn = copy.deepcopy(checkpoint.turn)
        self.wind_down_done = checkpoint.wind_down_done
        self.wind_down_attempts = checkpoint.wind_down_attempts
        self.wind_down_token_mark = checkpoint.wind_down_token_mark

    def reset_for_user_turn(self) -> None:
        """Prepare an existing session to accept a new user turn.

        Resets only *per-turn* state: the terminal phase (back to IDLE) and the
        whole per-turn enforcement window. ``step_count`` and ``used_tokens`` are
        intentionally preserved — both ``max_steps`` and ``max_budget_tokens``
        are session-lifetime caps, so a long-lived interactive/messaged session
        keeps accumulating across turns rather than getting a fresh allowance.
        Durable wind-down latches are current-turn state and reset here. The
        session-lifetime budget-reserve latch is intentionally preserved.
        """
        self.resume_to_idle()
        self.turn = TurnEnforcementState()
        self.wind_down_done = False
        self.wind_down_attempts = 0
        self.wind_down_token_mark = 0
        self.pending_step_latency = None
        self.submitted_summary = None

    def record_evidence_signal(
        self,
        content_hash: str,
        call_hash: str,
        intrinsic_low_yield: bool,
        *,
        card: dict[str, Any] | None = None,
    ) -> None:
        """Record result novelty and optionally append a classified evidence card.

        A hit requires unseen content and call hashes plus a false
        ``intrinsic_low_yield`` flag. Empty/no-match results remain low-yield even
        when new. Remember both hashes and classify cards as ``hit``, ``NO-MATCH``
        or ``duplicate`` using that same decision. ``card=None`` updates counters only.
        """
        already_seen = (
            content_hash in self.turn.seen_result_hashes
            or call_hash in self.turn.seen_result_hashes
        )
        self.turn.seen_result_hashes.add(content_hash)
        self.turn.seen_result_hashes.add(call_hash)
        if not intrinsic_low_yield and not already_seen:
            self.turn.low_yield_since_progress = 0
            self.turn.distinct_evidence_count += 1
            outcome = "hit"
        else:
            self.turn.low_yield_since_progress += 1
            outcome = "NO-MATCH" if intrinsic_low_yield else "duplicate"
        if card is not None:
            self.turn.scout_ledger.append(
                {
                    "tool": card.get("tool", ""),
                    "target": card.get("target", ""),
                    "outcome": outcome,
                    "snippet": card.get("snippet", ""),
                }
            )
            if len(self.turn.scout_ledger) > MAX_SCOUT_LEDGER_CARDS:
                del self.turn.scout_ledger[:-MAX_SCOUT_LEDGER_CARDS]

    def remember_tool_call_hash(self, call_hash: str, max_window: int | None = None) -> None:
        self.turn.recent_call_hashes.append(call_hash)
        if max_window is not None and len(self.turn.recent_call_hashes) > max_window:
            expired = self.turn.recent_call_hashes[:-max_window]
            self.turn.recent_call_hashes = self.turn.recent_call_hashes[-max_window:]
            for key in expired:
                if key in self.turn.loop_state.operations and key not in self.turn.recent_call_hashes:
                    operation = self.turn.loop_state.operations.get(key)
                    if operation is not None and operation.inflight_tool_call_id is None:
                        self.turn.loop_state.operations.pop(key, None)

    def replace_recent_tool_hashes(self, call_hashes: list[str]) -> None:
        self.turn.recent_call_hashes = call_hashes
