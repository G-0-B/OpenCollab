"""Structured-output forcing for :class:`WorkflowContext`.

The "make the agent commit a schema-valid payload" concern peeled out of
``workflow.py``: a first free-exploration pass, then — only when the capture is
genuinely empty — one corrective pass restricted to the capture tool with a
named-function ``tool_choice``. ``WorkflowStructuredMixin`` is mixed into
``WorkflowContext`` (it reaches the engine — ``_factory`` / ``_run_session_turn``
/ ``_capped_session_budget`` / ``log`` — via ``self``); ``_named_tool_choice``
also serves the evidence-preserving workflow mixin, which imports it from here.

Pure application layer: domain + stdlib imports only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from opencollab.application._anthropic_history import _strip_anthropic_thinking
from opencollab.application.async_timeout import CallerTimeoutError
from opencollab.application.structured_output import StructuredOutputTool

_STRUCTURED_INSTRUCTION = (
    "\n\nFinish by calling the `structured_output` tool — do not answer in "
    "free text."
)

# An endpoint may reject forced tool_choice and fall back to auto.
# Explicit submit-only wording still directs that retry toward the capture tool.
_STRUCTURED_RETRY = (
    "You MUST call the `structured_output` tool now, exactly once, with your "
    "final result based on what you have already gathered, conforming to the "
    "required schema. Do not explore further or answer in prose."
)

def _named_tool_choice(tool_name: str) -> dict[str, Any]:
    """Build an OpenAI-style choice forcing ``tool_name``.

    ``SessionRunUseCase._complete`` can retry with ``auto`` after a 400 rejection.
    """
    return {"type": "function", "function": {"name": tool_name}}

def _schema_satisfied(captured: Any, schema: dict[str, Any]) -> bool:
    """Require a captured dict with every required top-level key.

    ``StructuredOutputTool`` validates the full schema before capture.
    """
    if not isinstance(captured, dict):
        return False
    required = schema.get("required") if isinstance(schema, dict) else None
    if not required:
        # No required keys: the tool already validated the payload, so any
        # captured dict (even ``{}``) is an accepted commit, not a miss.
        return True
    return all(key in captured for key in required)

def _structured_retry_timeout(remaining: float | None) -> float | None:
    return remaining

class WorkflowStructuredMixin:
    """``schema=`` structured-output primitives mixed into ``WorkflowContext``."""

    async def _run_structured_agent(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        label: str | None,
        tools: Sequence[Any] | None,
        isolation: bool,
        timeout: float | None = None,
        budget: int | None = None,
    ) -> dict | None:
        """Run a schema-bound session and return a validated payload or None.

        The first pass offers the capture tool and exploration tools with the
        endpoint's default tool choice. An unsatisfied capture triggers one retry
        carrying the first conversation, restricted to the named capture tool.
        A parsed markup tool call that already satisfied the schema needs no retry.

        Endpoints may reject forced tool choice and fall back to auto. A missing
        capture after the retry returns None. Successful capture sets a cancellation
        event checked before each model call to prevent repeated submissions.
        """
        deadline = self._timeout_deadline(timeout)
        capture_done = asyncio.Event()
        capture_tool = StructuredOutputTool(schema, on_capture=capture_done.set)
        seeded_prompt = prompt + _STRUCTURED_INSTRUCTION
        combined_tools = [capture_tool, *(tools or [])]
        session_budget = self._capped_session_budget(budget)
        try:
            session = await self._build_workflow_session(
                prompt=seeded_prompt,
                budget=session_budget,
                tools=combined_tools,
                isolation=isolation,
                label=label,
                # Schema-bound agents inherit run-wide reasoning so their configuration
                # remains comparable with other agents in the same run.
                thinking=None,
            )
        except Exception as exc:  # noqa: BLE001 — factory failure must not abort the fleet
            self._record_agent_failure(label, exc)
            await self.log(f"structured agent build failed ({label or 'agent'}): {exc}")
            return None

        # Track immediately so tokens count even if a run_loop raises midway.
        self._track_session(session)
        try:
            await self._run_session_turn(
                session,
                seeded_prompt,
                deadline=deadline,
                cancel_event=capture_done,
            )
            if _schema_satisfied(capture_tool.captured, schema):
                return capture_tool.captured
            terminal_reason = str(getattr(getattr(session, "state", None), "terminal_reason", "") or "")
            if terminal_reason.startswith("context overflow:"):
                await self.log(f"structured agent input exceeded provider limits ({label or 'agent'})")
                return None
        except CallerTimeoutError:
            await self.log(f"structured agent timed out ({label or 'agent'}) after {timeout}s")
            return None
        except Exception as exc:  # noqa: BLE001 — one dead agent never kills the fleet
            self._record_agent_failure(label, exc)
            await self.log(f"structured agent failed ({label or 'agent'}): {exc}")
            if _schema_satisfied(capture_tool.captured, schema):
                return capture_tool.captured

        if self._active_call_has_pending_cleanup():
            await self.log(f"structured retry skipped while prior execution drains ({label or 'agent'})")
            return None

        # Reuse the capture tool so its payload and cancellation event span both
        # passes. Carry exploration history into a submit-only corrective session.
        # Endpoints may reject the forced choice and fall back to auto.
        try:
            retry_timeout = _structured_retry_timeout(
                self._remaining_timeout(deadline)
            )
        except CallerTimeoutError:
            await self.log(
                f"structured agent timed out ({label or 'agent'}) after {timeout}s"
            )
            return None
        return await self._forced_structured_commit(
            prompt,
            session,
            capture_tool,
            capture_done,
            schema=schema,
            label=label,
            isolation=isolation,
            timeout=retry_timeout,
            budget=budget,
        )

    async def _forced_structured_commit(
        self,
        prompt: str,
        prior_session: Any,
        capture_tool: StructuredOutputTool,
        capture_done: asyncio.Event,
        *,
        schema: dict[str, Any],
        label: str | None,
        isolation: bool,
        timeout: float | None,
        budget: int | None,
    ) -> dict | None:
        """Run a corrective session restricted to the named capture tool.

        Carry the first pass's conversation before adding the retry instruction,
        so submission can use its exploration results. Provider compatibility
        fallback can change a rejected forced choice to ``auto``.
        """
        deadline = self._timeout_deadline(timeout)
        retry_prompt = prompt + "\n\n" + _STRUCTURED_RETRY
        session_budget = self._capped_session_budget(budget)
        try:
            session = await self._build_workflow_session(
                prompt=retry_prompt,
                budget=session_budget,
                tools=[capture_tool],
                isolation=isolation,
                label=label,
                tool_choice=_named_tool_choice(capture_tool.name),
                thinking=None,
            )
        except Exception as exc:  # noqa: BLE001 — factory failure must not abort the fleet
            self._record_agent_failure(label, exc)
            await self.log(f"structured retry build failed ({label or 'agent'}): {exc}")
            return None

        self._track_session(session)
        try:
            if not self._carry_exploration(prior_session, session):
                await self.log(
                    f"structured retry could not carry exploration "
                    f"({label or 'agent'}); continuing from the retry prompt"
                )
            await self._run_session_turn(
                session,
                retry_prompt,
                deadline=deadline,
                cancel_event=capture_done,
            )
        except CallerTimeoutError:
            await self.log(f"structured retry timed out ({label or 'agent'}) after {timeout}s")
            return None
        except Exception as exc:  # noqa: BLE001 — one dead agent never kills the fleet
            self._record_agent_failure(label, exc)
            await self.log(f"structured retry failed ({label or 'agent'}): {exc}")
            return None
        return capture_tool.captured if _schema_satisfied(capture_tool.captured, schema) else None

    @staticmethod
    def _carry_exploration(prior_session: Any, session: Any) -> bool:
        """Carry prior exploration while preserving the corrective system messages.

        Use an independent list with shared message dicts, which callers leave intact.
        Read ``state.messages``, falling back to ``messages`` for older factories.
        Return False on failure so the caller can report degraded recovery and proceed.
        """
        prior_state = getattr(prior_session, "state", None)
        prior = getattr(prior_state, "messages", None)
        if prior is None:
            prior = getattr(prior_session, "messages", None)
        if prior is None:
            return False

        session_state = getattr(session, "state", None)
        try:
            current = getattr(session_state, "messages", None)
            if current is None:
                current = getattr(session, "messages", None)
            system = [message for message in current or [] if message.get("role") == "system"]
            carried = (
                system + [message for message in prior if message.get("role") != "system"]
                if system else list(prior)
            )
            # The corrective session changes system/tools. Its exploration
            # evidence remains useful; prior native thinking binds to the old
            # prefix and must stay out of this new conversation.
            if getattr(prior_state, "anthropic_history", None) is not None:
                carried = _strip_anthropic_thinking(carried)
            if session_state is not None and hasattr(session_state, "messages"):
                session_state.messages = carried
            else:
                session.messages = carried
        except Exception:  # noqa: BLE001 — carry-over is best-effort, never fatal
            return False
        return True
