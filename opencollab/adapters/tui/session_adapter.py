from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from opencollab.application.ports import AskUserPort, EventPublisherPort, PermissionPort


class TuiEventSink(EventPublisherPort):
    """Bridge the async event bus to the TUI handler for runtime and team events."""

    def __init__(self, tui):
        self.tui = tui

    async def emit(self, event: Any) -> None:
        result = self.tui.event_handler(event)
        if asyncio.iscoroutine(result):
            await result

class TuiPermissionPolicy(PermissionPort):
    """Ask y/N permission questions through the terminal's shared input line."""

    def __init__(self, read_line: Callable[[str], Awaitable[str]]):
        self._read_line = read_line

    async def confirm(self, prompt: str) -> bool:
        try:
            answer = await self._read_line(f"{prompt} [y/N] ")
        except (EOFError, KeyboardInterrupt):
            return False
        return answer.strip().lower() in ("y", "yes")

class TuiAskUserPolicy(AskUserPort):
    """Ask policy that reads a free-text answer from the same input line.

    On EOF/interrupt it raises ``EOFError`` so the tool reports the user
    declined.
    """

    def __init__(self, read_line: Callable[[str], Awaitable[str]]):
        self._read_line = read_line

    async def ask(self, question: str) -> str:
        return await self._read_line(f"[Agent asks] {question}\n> ")
