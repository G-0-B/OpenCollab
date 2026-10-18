"""Render session and scheduler events for the terminal UI.

Settled output goes to scrollback. The in-flight HUD becomes ANSI rows for
``adapters.cli.live_prompt``, which exclusively owns the bottom screen region.

``renderer_events`` updates state and ``renderer_display`` builds renderables.
Their mixins share the same ``TUI`` instance.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.control import Control
from rich.panel import Panel
from rich.text import Text

from opencollab.adapters.tui.brand_motion import MARK_HEX, PulseDot
from opencollab.adapters.tui.renderer_display import _RendererDisplayMixin
from opencollab.adapters.tui.renderer_events import _RendererEventsMixin
from opencollab.domain.session import TERMINAL_PHASES

# Throttle HUD rendering independently of the prompt's faster refresh.
HUD_FRAME_INTERVAL = 0.05
MAX_HISTORY_BLOCKS_PER_AGENT = 400
MAX_TERMINAL_AGENT_STATES = 128
MAX_TERMINAL_AGENT_SUMMARIES = 256
_TERMINAL_RENDER_STATES = frozenset(
    {
        "idle",
        "failed",
        "cancelled",
        *(phase.value for phase in TERMINAL_PHASES),
    }
)

@dataclass
class _AgentRenderState:
    """Mutable live-render state owned by one scheduler agent.

    ``history_blocks`` is the agent's bounded settled transcript; it is retained
    after printing because switching focus reprints an agent in full.
    ``printed_blocks`` is how much of it has already reached scrollback, so an
    agent that holds focus streams incrementally instead of reprinting.
    """

    current_text: str = ""
    turn_has_text: bool = False
    active_tools: dict[str, dict] = field(default_factory=dict)
    status_lines: list[Text] = field(default_factory=list)
    history_blocks: list[Any] = field(default_factory=list)
    history_omitted_blocks: int = 0
    printed_blocks: int = 0
    step: int = 0
    thinking: PulseDot | None = None

@dataclass(frozen=True)
class _AgentFocusTarget:
    """One keyboard-selectable live agent."""

    aid: int
    role: str

class TUI(_RendererEventsMixin, _RendererDisplayMixin):
    """Render streaming text, tools, and status from runtime and scheduler events.

    ``event_handler`` dispatches both event families to their dedicated handlers.
    """

    def __init__(self, console: Console | None = None, *, filter_messages: bool = False):
        self.console = console or Console()
        self._agent_states: dict[int, _AgentRenderState] = {}
        # aid -> {"role": str, "state": str} for the live team roster panel.
        self._roster: dict[int, dict] = {}
        self._terminal_agent_order: list[int] = []
        self._terminal_agent_summaries: dict[int, dict[str, Any]] = {}
        self._terminal_summary_order: list[int] = []
        self._terminal_summaries_omitted = 0
        # Brand motion (single pulsing dot). ``_motion`` is the shared,
        # seconds-less tool-execution spinner. Each agent owns its own LLM-wait
        # indicator so hidden streams remain intact while another agent is selected.
        self._motion = PulseDot(muted_style=self._STYLE_MUTED, show_seconds=False)
        # The prompt that owns the bottom region asks for a redraw through this
        # callback; ``None`` means nothing is painting a HUD (a non-TTY run).
        self._redraw: Callable[[], None] | None = None
        self._queued_turns = 0
        self._hud_cache: tuple[Any, str] | None = None
        self._status_cache: tuple[Any, str] | None = None
        # Compatibility input that must not control event collection.
        self._filter_messages = filter_messages
        self._selected_aid = 0
        self._state_for(0)
        # Optional callable returning the full team roster (live agents +
        # configured "available" roles). When set, the team panel renders from
        # it so the roster stays visible during a turn, not only after a spawn.
        self._team_provider: Any | None = None
        # Agents whose streamed text reached scrollback during the last turn,
        # including text settled early by a tool or preserved status event.
        self._drained_partial_aids: frozenset[int] = frozenset()

    def _state_for(self, aid: int) -> _AgentRenderState:
        """Return one agent's render state, creating it on first observation."""
        return self._agent_states.setdefault(aid, _AgentRenderState())

    def _append_history_block(self, state: _AgentRenderState, block: Any) -> None:
        state.history_blocks.append(block)
        overflow = len(state.history_blocks) - MAX_HISTORY_BLOCKS_PER_AGENT
        if overflow > 0:
            del state.history_blocks[:overflow]
            state.history_omitted_blocks += overflow
            state.printed_blocks = max(0, state.printed_blocks - overflow)

    def set_redraw(self, redraw: Callable[[], None] | None) -> None:
        """Register the bottom-region redraw callback, or ``None`` to detach.

        CLI ``patch_stdout`` handles scrollback writes. This callback repaints changed
        HUD state.
        """
        self._redraw = redraw
        self._hud_cache = None
        self._status_cache = None

    def _drain_pending(self, aid: int | None = None) -> None:
        """Print the focused agent's not-yet-printed settled blocks."""
        target_aid = self._selected_aid if aid is None else aid
        if target_aid != self._selected_aid:
            return
        self._drain_agent_tail(target_aid)

    def _drain_agent_tail(self, aid: int) -> None:
        """Print an agent's unprinted tail, adding a band when it lacks focus.

        Already printed blocks must be skipped here to avoid duplicating the
        transcript.
        """
        state = self._state_for(aid)
        pending = state.history_blocks[state.printed_blocks:]
        if not pending:
            return
        state.printed_blocks = len(state.history_blocks)
        blocks = pending if aid == self._selected_aid else [self._focus_band(aid), *pending]
        self._print_blocks(blocks)

    def _fully_printed(self, aid: int) -> bool:
        """Has everything this agent has settled reached scrollback?"""
        state = self._state_for(aid)
        return state.printed_blocks == len(state.history_blocks)

    def _print_blocks(self, blocks: list[Any]) -> None:
        for block in blocks:
            self.console.print(block)

    def _screen_top_prelude(self) -> list[Any]:
        """Return the scroll-and-home prelude, or nothing without a screen.

        Scroll the visible screen into scrollback before a focus switch so the
        previous agent's transcript survives. Return the prelude for one combined
        write because a prompt repaint between scrolling and homing would add rows.
        """
        height = self.console.height
        if not self.console.is_terminal or height < 1:
            return []
        return [Text("\n" * (height - 1)), Control.home()]

    def _track_agent_render_lifecycle(self, aid: int, state: str) -> None:
        if aid in self._terminal_summary_order:
            self._terminal_summary_order.remove(aid)
        self._terminal_agent_summaries.pop(aid, None)
        if aid in self._terminal_agent_order:
            self._terminal_agent_order.remove(aid)
        if aid != 0 and state in _TERMINAL_RENDER_STATES:
            self._terminal_agent_order.append(aid)
        while len(self._terminal_agent_order) > MAX_TERMINAL_AGENT_STATES:
            victim = next(
                (
                    candidate
                    for candidate in self._terminal_agent_order
                    if candidate != self._selected_aid
                ),
                None,
            )
            if victim is None:
                break
            self._terminal_agent_order.remove(victim)
            render_state = self._agent_states.pop(victim, None)
            roster = self._roster.pop(victim, {})
            self._terminal_agent_summaries[victim] = {
                "aid": victim,
                "role": str(roster.get("role", "agent")),
                "state": str(roster.get("state", "idle")),
                "step": render_state.step if render_state is not None else 0,
                "retained_history_blocks": (
                    len(render_state.history_blocks) if render_state is not None else 0
                ),
                "omitted_history_blocks": (
                    render_state.history_omitted_blocks
                    if render_state is not None
                    else 0
                ),
            }
            self._terminal_summary_order.append(victim)
            while len(self._terminal_summary_order) > MAX_TERMINAL_AGENT_SUMMARIES:
                forgotten = self._terminal_summary_order.pop(0)
                self._terminal_agent_summaries.pop(forgotten, None)
                self._terminal_summaries_omitted += 1

    def _event_aid(self, aid: Any) -> int:
        """Normalize legacy events without an aid to agent 0, independent of focus."""
        return aid if isinstance(aid, int) and aid >= 0 else 0

    def _agent_label(self, aid: int) -> str:
        """Return the stable display label for an agent.

        Child agents keep compact ``A<n>`` labels. Agent 0 uses its configured
        entry role, while the built-in team naturally remains ``Lead``.
        """
        if aid != 0:
            return f"A{aid}"
        role = next(
            (role for entry_aid, role, _state in self._team_entries() if entry_aid == 0),
            "lead",
        )
        label = str(role).strip().replace("_", " ").replace("-", " ")
        return label.title() or "Lead"

    @property
    def _selected_state(self) -> _AgentRenderState:
        return self._state_for(self._selected_aid)

    # Compatibility accessors keep the display mixin and narrow external tests
    # focused on the selected agent while all event data remains separately held.
    @property
    def _current_text(self) -> str:
        return self._selected_state.current_text

    @_current_text.setter
    def _current_text(self, value: str) -> None:
        self._selected_state.current_text = value

    @property
    def _active_tools(self) -> dict[str, dict]:
        return self._selected_state.active_tools

    @_active_tools.setter
    def _active_tools(self, value: dict[str, dict]) -> None:
        self._selected_state.active_tools = value

    @property
    def _status_lines(self) -> list[Text]:
        return self._selected_state.status_lines

    @_status_lines.setter
    def _status_lines(self, value: list[Text]) -> None:
        self._selected_state.status_lines = value

    @property
    def _step(self) -> int:
        return self._selected_state.step

    @_step.setter
    def _step(self, value: int) -> None:
        self._selected_state.step = value

    @property
    def _thinking(self) -> PulseDot | None:
        return self._selected_state.thinking

    @_thinking.setter
    def _thinking(self, value: PulseDot | None) -> None:
        self._selected_state.thinking = value

    def set_team_provider(self, provider: Any) -> None:
        """Supply a callable returning the full team roster so the live display
        shows the team continuously (matching the prompt's bottom toolbar)."""
        self._team_provider = provider

    @property
    def selected_aid(self) -> int:
        return self._selected_aid

    @property
    def terminal_agent_summaries(self) -> tuple[dict[str, Any], ...]:
        """Bounded summaries for terminal agents evicted from detailed TUI state."""
        return tuple(
            dict(self._terminal_agent_summaries[aid])
            for aid in self._terminal_summary_order
        )

    @property
    def terminal_agent_summaries_omitted(self) -> int:
        """Number of oldest summaries omitted to retain a stable memory bound."""
        return self._terminal_summaries_omitted

    @property
    def selected_role(self) -> str | None:
        """Compatibility view: configured-only roles are not input targets."""
        return None

    def _focus_targets(self) -> list[_AgentFocusTarget]:
        """Return live agents in stable aid order.

        Configured-but-unspawned roles remain visible in the roster, but without
        an aid they cannot own history or receive user input.
        """
        live_roles: dict[int, str] = {}

        for aid, role, state in self._team_entries():
            role = str(role)
            if not isinstance(aid, int) or aid < 0:
                continue
            if (
                aid == 0
                or aid in self._agent_states
                or state not in _TERMINAL_RENDER_STATES
            ):
                live_roles.setdefault(aid, role)

        for aid in self._agent_states:
            fallback_role = "lead" if aid == 0 else "agent"
            role = str(self._roster.get(aid, {}).get("role", fallback_role))
            live_roles.setdefault(aid, role)

        return [
            _AgentFocusTarget(aid=aid, role=live_roles[aid])
            for aid in sorted(live_roles)
        ]

    def _focus_is_selected(self, target: _AgentFocusTarget) -> bool:
        return self._selected_aid == target.aid

    def _select_focus(self, target: _AgentFocusTarget) -> int:
        changed = not self._focus_is_selected(target)
        self._selected_aid = target.aid
        if changed:
            self._reprint_focused_agent()
            self._refresh()
        return self._selected_aid

    def _reprint_focused_agent(self) -> None:
        """Redraw the focused agent's retained trajectory under a labelled band.

        Name blocks evicted by the history bound. Start on the first screen row via
        ``_screen_top_prelude`` and let trajectories taller than the terminal scroll.
        """
        state = self._selected_state
        band = self._focus_band(self._selected_aid)
        blocks = list(state.history_blocks)
        omitted = state.history_omitted_blocks
        state.printed_blocks = len(state.history_blocks)
        if omitted:
            blocks.insert(
                0,
                Text(
                    f"... {omitted} older history blocks omitted; "
                    "full history remains in the run trace.",
                    style=self._STYLE_MUTED,
                ),
            )
        self.console.print(Group(*self._screen_top_prelude(), band, *blocks))

    def select_agent(self, aid: int) -> int:
        """Select an existing agent and return the resulting aid."""
        target = next(
            (candidate for candidate in self._focus_targets() if candidate.aid == aid),
            None,
        )
        return self._selected_aid if target is None else self._select_focus(target)

    def select_relative_agent(self, offset: int) -> int:
        """Move through live team members with wraparound."""
        targets = self._focus_targets()
        if not targets:
            return self._selected_aid
        index = next(
            (
                position
                for position, target in enumerate(targets)
                if self._focus_is_selected(target)
            ),
            0,
        )
        return self._select_focus(targets[(index + offset) % len(targets)])

    def select_previous_agent(self) -> int:
        return self.select_relative_agent(-1)

    def select_next_agent(self) -> int:
        return self.select_relative_agent(1)

    def record_user_message(self, aid: int, content: str) -> None:
        """Add one user turn to the target's transcript and settle it."""
        state = self._state_for(aid)
        self._flush_current_text_to_timeline(state)
        self._append_history_block(state, self._user_block(content))
        self._drain_pending(aid)

    def _refresh(self) -> None:
        """Ask the bottom region's owner to repaint the HUD."""
        self._hud_cache = None
        self._status_cache = None
        if self._redraw is not None:
            self._redraw()

    def status_ansi(self, width: int | None = None) -> str | None:
        """Return ANSI status below the input, or ``None`` when empty.

        Cache until state or pulse changes because prompts request it per keystroke.
        """
        width = self.console.width if width is None else width
        key = (width, int(time.monotonic() / HUD_FRAME_INTERVAL))
        cached = self._status_cache
        if cached is not None and cached[0] == key:
            return cached[1] or None
        row = self._build_status_row()
        rendered = "" if row is None else self._render_ansi(row, width)
        self._status_cache = (key, rendered)
        return rendered or None

    def hud_ansi(self, width: int | None = None) -> str | None:
        """Return in-flight ANSI rows, or ``None`` when idle.

        The prompt owns their redraws. Cache between state or pulse changes because
        prompt repaints request the frame repeatedly.
        """
        width = self.console.width if width is None else width
        key = (width, int(time.monotonic() / HUD_FRAME_INTERVAL))
        cached = self._hud_cache
        if cached is not None and cached[0] == key:
            return cached[1] or None
        frame = self._build_hud()
        rendered = "" if frame is None else self._render_ansi(frame, width)
        self._hud_cache = (key, rendered)
        return rendered or None

    def _render_ansi(self, renderable: Any, width: int) -> str:
        """Render one renderable to the ANSI rows the prompt will paint."""
        with self.console.capture() as capture:
            self.console.print(renderable, width=width, end="", crop=True)
        return capture.get().rstrip("\n")

    def set_queued_turns(self, count: int) -> None:
        """Show queued-message count so typing during a turn has visible confirmation."""
        if count == self._queued_turns:
            return
        self._queued_turns = count
        self._refresh()

    def start_turn(self, aid: int) -> None:
        """Open a turn on ``aid``: show its wait indicator and repaint."""
        # Animate model wait until text or tool progress arrives. ``step_start``
        # adds the step counter.
        label = self._agent_label(aid)
        self._state_for(aid).thinking = self._new_thinking_bar(f"{label} thinking…")
        self._refresh()

    def settle_turn(self, final_aid: int | None = None) -> None:
        """Settle trailing streamed text and clear live-only chrome at turn end.

        Flush ``final_aid`` even when another agent has focus, since a one-shot run
        has no later opportunity to display that agent's tail.
        """
        for state in self._agent_states.values():
            self._flush_current_text_to_timeline(state)
            state.status_lines.clear()
            state.thinking = None
        self._drain_pending()
        if final_aid is not None:
            self._drain_agent_tail(final_aid)
        self._drained_partial_aids = frozenset(
            aid for aid, state in self._agent_states.items()
            if state.turn_has_text and self._fully_printed(aid)
        )
        for state in self._agent_states.values():
            state.turn_has_text = False
        self._refresh()

    def drained_partial_answer(self, aid: int) -> bool:
        """Report whether this agent's text reached scrollback at ``settle_turn``.

        The CLI checks this before printing ``SchedulerTurnError.partial_answer``
        to avoid displaying the same partial answer twice.
        """
        return aid in self._drained_partial_aids

    def print_welcome(
        self,
        *,
        model: str | None = None,
        provider: str | None = None,
        workspace: str | None = None,
        budget: int | None = None,
        interactive: bool = True,
    ) -> None:
        """Compact 'Calm HUD' banner: wordmark + tagline, then an optional
        aligned key/value metadata line. All fields are optional so a bare
        ``print_welcome()`` still renders a clean two-line banner."""
        import os

        from rich.console import Group
        from rich.rule import Rule

        title = Text.assemble(
            ("◆ ", MARK_HEX),
            ("OpenCollab", self._STYLE_HEADING),
            ("  multi-agent dev", self._STYLE_MUTED),
        )
        tagline_text = (
            "Type a message · /help for controls · Ctrl+C interrupts · 'exit' quits"
            if interactive
            else "Running issue · Ctrl+C interrupts"
        )
        tagline = Text(tagline_text, style=self._STYLE_MUTED)

        fields: list[tuple[str, str]] = []
        if provider or model:
            fields.append(("model", f"{provider + ':' if provider else ''}{model or '?'}"))
        if workspace:
            home = os.path.expanduser("~")
            cwd = os.path.abspath(str(workspace))
            if home != "~" and cwd.startswith(home):
                cwd = "~" + cwd[len(home):]
            fields.append(("cwd", cwd))
        if budget is not None:
            fields.append(("budget", f"{budget:,} tok"))

        body: list[Any] = [title, tagline]
        if fields:
            meta = Text(no_wrap=True, overflow="ellipsis")
            for i, (key, val) in enumerate(fields):
                if i:
                    meta.append("   ", style=self._STYLE_MUTED)
                meta.append(f"{key} ", style=self._STYLE_KEY)
                meta.append(val, style=self._STYLE_SUBTLE)
            body += [Rule(style=self._STYLE_MUTED), meta]

        self.console.print(Panel.fit(
            Group(*body),
            border_style=self._STYLE_MUTED,
            padding=(0, 2),
        ))

    def print_stats(self, tokens: int, steps: int) -> None:
        """Quiet hairline-led receipt under a completed turn."""
        self.console.print()
        self.console.print(Text.assemble(
            ("─ ", self._STYLE_MUTED),
            (f"{tokens:,}", self._STYLE_SUBTLE), (" tokens", self._STYLE_MUTED),
            ("  ·  ", self._STYLE_MUTED),
            (f"{steps}", self._STYLE_SUBTLE), (" steps", self._STYLE_MUTED),
        ))

    def print_turn_divider(self) -> None:
        """Hairline rule between turns (Rule is exempt from the glyph walk)."""
        from rich.rule import Rule

        self.console.print(Rule(style=self._STYLE_MUTED))

    def reset(self) -> None:
        """Reset live chrome for the next turn while retaining agent history."""
        for state in self._agent_states.values():
            self._flush_current_text_to_timeline(state)
            state.active_tools.clear()
            state.status_lines.clear()
            state.step = 0
            state.thinking = None
        self._state_for(0)
