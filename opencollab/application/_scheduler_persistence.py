"""Scheduler persistence and token-budget accounting."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable

from opencollab.application.autosave import AutoSaveSubscriber
from opencollab.application.events import SchedulerEventFactory
from opencollab.domain.scheduler import (
    PER_AGENT_BUDGET_SHARE,
    dynamic_roster_share,
    per_agent_cap,
    split_budget,
)

logger = logging.getLogger(__name__)


class SchedulerPersistenceMixin:
    def _track_review_parent_lease_release(self, parent_aid: int, delta: int) -> None:
        tracker = self._review_parent_lease_tracker.get()
        if tracker is None or tracker[0] != parent_aid:
            return
        tracker[1]["outstanding"] = max(0, tracker[1].get("outstanding", 0) + delta)

    def set_manifest_writer(
        self,
        fn: Callable[[], None],
        *,
        prepare_fn: Callable[[], Callable[[], None] | None] | None = None,
    ) -> None:
        """Inject the team-manifest persister (called on every roster change)."""
        self._manifest_writer = fn
        self._manifest_subscriber = AutoSaveSubscriber(fn, prepare_fn=prepare_fn)

    def _write_manifest(self) -> Exception | None:
        if self._manifest_writer is None or self._manifest_subscriber is None:
            return None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                self._manifest_writer()
            except Exception as exc:
                self._scheduler_persistence_errors.append(exc)
                logger.warning("manifest write failed: %s", exc)
                return exc
            return None
        try:
            self._manifest_subscriber.enqueue()
        except Exception as exc:
            self._scheduler_persistence_errors.append(exc)
            logger.warning("manifest write enqueue failed: %s", exc)
            return exc
        return None

    def _autosave_session(self, aid: int) -> asyncio.Task[None] | None:
        session = self._sessions.get(aid)
        if session is None:
            return None
        save_path = getattr(session, "auto_save_path", None)
        if not save_path:
            return None
        enqueue = getattr(session, "enqueue_auto_save", None)
        try:
            if callable(enqueue):
                return enqueue()
            save = getattr(session, "save", None)
            if not callable(save):
                return None
            subscriber = self._fallback_autosavers.get(aid)
            if subscriber is None:
                subscriber = AutoSaveSubscriber(lambda: save(save_path))
                self._fallback_autosavers[aid] = subscriber
            return subscriber.enqueue()
        except Exception as exc:
            self._scheduler_persistence_errors.append(exc)
            logger.warning("session auto-save enqueue failed for aid %s: %s", aid, exc)
            return None

    def _autosave_all_sessions(self) -> tuple[asyncio.Task[None], ...]:
        owners: list[asyncio.Task[None]] = []
        for aid in list(self._sessions):
            owner = self._autosave_session(aid)
            if owner is not None:
                owners.append(owner)
        return tuple(owners)

    @property
    def events(self) -> SchedulerEventFactory:
        """The scheduler-event builders (the orchestration event vocabulary)."""
        return self._events

    @property
    def used_tokens(self) -> int:
        """Total tokens across all agents."""
        return self.table.total_used_tokens

    @property
    def allocated_tokens(self) -> int:
        """Tokens spent plus the unspent part of every active turn lease."""
        return self._budget_committed()

    @property
    def budget_exhausted(self) -> bool:
        """Whether aggregate team spend reached the shared cap.

        This also catches in-flight overshoot of individual grants.
        """
        if self._role_budgets:
            # Independent allowances: the team is out of tokens only when every
            # seated agent has spent its own. One agent's overshoot is that
            # agent's, and must not stop a teammate with allowance left.
            entries = self.table.entries
            return bool(entries) and all(
                self._session_used_tokens(aid) >= self._agent_cap(aid)
                for aid in entries
            )
        return self.used_tokens >= self._max_budget_tokens

    def _role_of(self, aid: int) -> str | None:
        if aid == 0 and self._entry_role is not None:
            return self._entry_role
        scb = self.table.get(aid)
        return scb.agent.name if scb is not None else None

    def _agent_cap(self, aid: int, role: str | None = None) -> int | None:
        """The most ``aid`` may spend: its role's allowance, or the shared rule's cap."""
        if self._role_budgets:
            return self._role_budgets[role if role is not None else self._role_of(aid)]
        return self._per_agent_cap()

    def _declared_team_size(self) -> int:
        """N — how many roles the team config declares, agent 0's role included."""
        return max(1, len(self._roles))

    def _per_agent_cap(self) -> int | None:
        """Return the declared roster's cumulative seat cap, else None.

        Prebuilt rosters use ``c * total / N``; dynamic rosters have no fixed N
        and reserve ``dynamic_roster_share`` at creation instead.
        """
        if not self._prebuild_team:
            return None
        return per_agent_cap(self._max_budget_tokens, self._declared_team_size())

    def _entry_agent_takes_the_pool(self, aid: int) -> bool:
        """Whether dynamic agent 0 may lease the unallocated remainder.

        Declared seats, including agent 0, instead obey the same per-agent cap.
        """
        return aid == 0 and self._per_agent_cap() is None

    def _entry_start_budget(self) -> int:
        """Return agent 0's actual spend ceiling, also shown in its budget prompt.

        Passing the whole pool for a declared seat would overstate its allowance.
        """
        cap = self._agent_cap(0)
        return self._max_budget_tokens if cap is None else cap

    def _seed_entry_lease(self) -> None:
        """Idempotently reserve agent 0's dynamic share before children can book.

        Declared rosters reserve nothing at seating; an idle seat holds no tokens.
        """
        if self._per_agent_cap() is not None:
            return
        self._turn_lease[0] = dynamic_roster_share(self._max_budget_tokens)
        self._lease_baseline[0] = self._session_used_tokens(0)

    def _session_used_tokens(self, aid: int) -> int:
        scb = self.table.get(aid)
        if scb is not None:
            return max(0, int(getattr(scb.state, "used_tokens", 0) or 0))
        session = self._sessions.get(aid)
        return max(0, int(getattr(session, "used_tokens", 0) or 0))

    def _lease_remaining(self, aid: int, grant: int, baseline: int) -> int:
        used = self._session_used_tokens(aid)
        return max(0, grant - max(0, used - baseline))

    def _is_entry_session(self, aid: int) -> bool:
        """True when ``aid`` names the registered agent 0 (the entry session)."""
        return (
            aid == 0
            and self._lead_session is not None
            and self._sessions.get(0) is self._lead_session
        )

    def _budget_committed(self) -> int:
        """Return spent tokens plus outstanding dynamic reservations.

        Declared seat caps reserve nothing; a dynamic live lease still owns its
        unspent remainder to prevent a subsequent spawn from oversubscribing.
        """
        committed = max(0, self.used_tokens)
        if self._per_agent_cap() is not None:
            return committed
        for aid, grant in self._turn_lease.items():
            committed += self._lease_remaining(aid, grant, self._lease_baseline.get(aid, 0))
        for aid, grant, baseline, _tasks, _finished in self._pending_provider_budget_leases.values():
            committed += self._lease_remaining(aid, grant, baseline)
        return committed

    def _defer_budget_lease_for_provider_cleanup(self, aid: int) -> bool:
        """Hold a dynamic grant until late provider usage settles.

        The provider callback updates usage before this scheduler callback releases
        the grant. Group tasks under one reservation to avoid duplicate booking.
        """
        if self._per_agent_cap() is not None:
            return False
        session = self._sessions.get(aid)
        runner = getattr(session, "runner", None)
        pending = getattr(runner, "pending_cleanup_tasks", ())
        tasks = frozenset(task for task in pending if isinstance(task, asyncio.Task))
        lease = self._current_turn_lease(aid)
        if not tasks or lease is None:
            return False

        grant, baseline = lease
        self._release_turn_lease(aid)
        reservation = object()
        self._pending_provider_budget_leases[reservation] = (
            aid,
            grant,
            baseline,
            tasks,
            set(),
        )
        for task in tasks:
            task.add_done_callback(
                lambda finished, key=reservation: self._finish_provider_budget_lease(
                    key, finished
                )
            )
        return True

    def _finish_provider_budget_lease(
        self, reservation: object, task: asyncio.Task[object]
    ) -> None:
        """Drop one deferred grant after every provider cleanup owner settles."""
        lease = self._pending_provider_budget_leases.get(reservation)
        if lease is None:
            return
        lease[4].add(task)
        if not lease[3].issubset(lease[4]):
            return
        self._pending_provider_budget_leases.pop(reservation, None)

    def _reserve_child_budget(self, aid: int, role: str | None = None) -> int:
        """Return a declared seat cap or synchronously reserve a dynamic child grant.

        Declared seating books nothing. Dynamic booking happens before any await
        so batched spawns see updated allocations.
        """
        cap = self._agent_cap(aid, role)
        if cap is not None:
            return cap
        grant = split_budget(self._max_budget_tokens, self._budget_committed())
        self._turn_lease[aid] = grant
        self._lease_baseline[aid] = self._session_used_tokens(aid)
        return grant

    def _grant_under_cap(self, aid: int, cap: int, used: int) -> int:
        """Grant the smaller of unspent team tokens and the agent's remaining allowance.

        The agent cap applies to cumulative spend, not just this turn.
        """
        agent_remaining = cap - used
        pool_remaining = self._max_budget_tokens - self.used_tokens
        if self._role_budgets:
            # Independent allowances: the pool is only their sum, so what is
            # left of it says nothing about what this agent may still spend.
            return max(0, agent_remaining)
        if agent_remaining <= 0 < pool_remaining:
            self._trace_agent_cap_reached(
                aid,
                cap=cap,
                used=used,
                pool_remaining=pool_remaining,
            )
        return max(0, min(agent_remaining, pool_remaining))

    def _trace_agent_cap_reached(
        self,
        aid: int,
        *,
        cap: int,
        used: int,
        pool_remaining: int,
    ) -> None:
        """Record a seat stopped by its cap while the shared pool still has tokens.

        This distinguishes unused roles from exhausted allowances. ``requested`` is
        the pool-only grant; ``remaining`` retains negative overshoot, and
        ``would_exceed_by`` is measured against it. Tracing cannot change allocation.
        """
        tracer = self._tracer
        if tracer is None:
            return
        try:
            scb = self.table.get(aid)
            state = getattr(scb, "state", None)
            payload = {
                "agent_id": aid,
                "role": scb.agent.name if scb is not None else None,
                "step_count": getattr(state, "step_count", None),
                "requested": max(0, pool_remaining),
                "per_agent_cap": cap,
                "used": used,
                "remaining": cap - used,
                "pool_remaining": pool_remaining,
                "spent": self.used_tokens,
                "total": self._max_budget_tokens,
                "team_size": self._declared_team_size(),
                "share": PER_AGENT_BUDGET_SHARE,
                "would_exceed_by": max(0, pool_remaining) - (cap - used),
            }
            tracer.log_step(step_type="agent_cap_reached", payload=payload)
        except Exception as exc:  # noqa: BLE001 — observability is non-authoritative
            logger.error("agent budget cap trace failed: %s", exc)

    def _reserve_turn_lease(self, aid: int) -> int:
        """Lease the remaining pool within a declared seat's cap, or the dynamic
        remainder not reserved by another live lease.
        """
        self._release_turn_lease(aid)
        baseline = self._session_used_tokens(aid)
        cap = self._agent_cap(aid)
        if cap is None:
            grant = max(0, self._max_budget_tokens - self._budget_committed())
        else:
            grant = self._grant_under_cap(aid, cap, baseline)
        if grant > 0:
            self._turn_lease[aid] = grant
            self._lease_baseline[aid] = baseline
        self._set_session_budget_limit(aid, baseline + grant)
        return grant

    def _set_session_budget_limit(self, aid: int, limit: int) -> None:
        session = self._sessions.get(aid)
        if session is None:
            return
        if hasattr(session, "max_budget_tokens"):
            session.max_budget_tokens = limit
        runner = getattr(session, "runner", None)
        if runner is not None and hasattr(runner, "max_budget_tokens"):
            runner.max_budget_tokens = limit

    def _reserve_message_budget(self, aid: int) -> bool:
        """Acquire a fresh lease before a terminal teammate starts another turn."""
        session = self._sessions.get(aid)
        if session is None:
            return False
        if self._per_agent_cap() is not None:
            # Declared roster: one path for every agent, agent 0 included. A zero
            # grant here means this agent has spent its own cap (or the pool is
            # dry), so let the turn start and reach an explicit budget terminal
            # rather than leave a durable inbox entry that can never become
            # runnable.
            self._reserve_turn_lease(aid)
            return True
        if self._is_entry_session(aid):
            grant = self._reserve_turn_lease(aid)
        elif aid in self._turn_lease:
            return True
        else:
            grant = self._reserve_child_budget(aid)
            baseline = self._session_used_tokens(aid)
            self._set_session_budget_limit(aid, baseline + grant)
            if grant <= 0:
                self._release_turn_lease(aid)
        # When spend already reached the ceiling, run one zero-budget precheck so
        # the queued turn reaches an explicit budget terminal instead of leaving
        # a durable inbox entry that can never become runnable.
        return grant > 0 or self.budget_exhausted

    def _release_turn_lease(self, aid: int) -> tuple[int, int] | None:
        """Reclaim ``aid``'s reservation so a later turn can reuse the headroom.

        Idempotent, and the same for every agent: agent 0 is released out of the
        one lease table like any other.
        """
        grant = self._turn_lease.pop(aid, None)
        baseline = self._lease_baseline.pop(aid, None)
        if grant is None:
            return None
        return grant, baseline or 0

    def _current_turn_lease(self, aid: int) -> tuple[int, int] | None:
        """Snapshot ``aid``'s lease without mutating the shared allocation."""
        grant = self._turn_lease.get(aid)
        if grant is None:
            return None
        return grant, self._lease_baseline.get(aid, 0)

    def _restore_turn_lease(self, aid: int, lease: tuple[int, int] | None) -> None:
        if lease is None:
            return
        old_grant, old_baseline = lease
        used = self._session_used_tokens(aid)
        old_remaining = max(0, old_grant - max(0, used - old_baseline))
        available = max(0, self._max_budget_tokens - self._budget_committed())
        grant = min(old_remaining, available)
        baseline = used
        if grant > 0:
            self._turn_lease[aid] = grant
            self._lease_baseline[aid] = baseline
        self._set_session_budget_limit(aid, baseline + grant)
