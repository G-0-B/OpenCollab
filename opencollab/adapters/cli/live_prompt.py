"""Own the terminal's persistent input line and surrounding HUD.

prompt_toolkit paints the HUD, input, and status row. Top alignment and a
non-growing input keep the status row flush beneath the input.

A session-wide ``patch_stdout`` routes writes through ``run_in_terminal``
above the prompt, preserving scrollback and exclusive ownership of its rows.
Concurrent permission and ``ask_user`` questions share a reader queue and
take precedence over the user's unfinished message.
"""

from __future__ import annotations

import asyncio
import codecs
import contextlib
import io
import os
import stat
import sys
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.filters import to_filter
from prompt_toolkit.formatted_text import ANSI, to_formatted_text
from prompt_toolkit.layout.containers import VerticalAlign, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

from opencollab.adapters.cli.prompt_view import build_agent_navigation_bindings
from opencollab.adapters.tui.theme import BRAND_VIOLET

REFRESH_INTERVAL = 0.1

_PROMPT_STYLE = Style.from_dict({
    "prompt.chevron": f"{BRAND_VIOLET} bold",
    # The row supplies Rich colors instead of the default reverse-video bar.
    "bottom-toolbar": "noreverse bg:default",
    "bottom-toolbar.text": "noreverse bg:default",
})
_CHEVRON = [("class:prompt.chevron", "❯"), ("", " ")]
_FALLBACK_PROMPT = "> "

@dataclass
class _Question:
    """One agent waiting for the user to answer on the shared input line."""

    text: str
    future: asyncio.Future

class _ConsoleInput:
    """One buffered, serialized reader for a redirected input stream."""

    def __init__(self, stream: Any):
        self.lock = asyncio.Lock()
        decoder = codecs.getincrementaldecoder(getattr(stream, "encoding", None) or "utf-8")(
            errors=getattr(stream, "errors", None) or "strict"
        )
        self.decoder = io.IncrementalNewlineDecoder(decoder, translate=True)
        self.buffer = ""
        self.eof = False

    def _take_line(self) -> str | None:
        end = self.buffer.find("\n")
        if end >= 0:
            line, self.buffer = self.buffer[:end], self.buffer[end + 1:]
            return line
        if self.eof and self.buffer:
            line, self.buffer = self.buffer, ""
            return line
        return None

    async def read(self, fd: int) -> str:
        line = self._take_line()
        if line is not None:
            return line
        if self.eof:
            raise EOFError
        loop = asyncio.get_running_loop()
        result: asyncio.Future[str] = loop.create_future()

        def on_readable() -> None:
            if result.done():
                return
            try:
                chunk = os.read(fd, 4096)
                self.eof = not chunk
                self.buffer += self.decoder.decode(chunk, final=self.eof)
                line = self._take_line()
                if line is not None:
                    result.set_result(line)
                elif self.eof:
                    result.set_exception(EOFError())
            except BaseException as exc:
                result.set_exception(exc)

        try:
            loop.add_reader(fd, on_readable)
        except (AttributeError, NotImplementedError) as exc:
            raise RuntimeError("console input fallback is unavailable on this event loop") from exc
        try:
            return await result
        finally:
            loop.remove_reader(fd)

_CONSOLE_INPUTS: weakref.WeakKeyDictionary[Any, _ConsoleInput] = weakref.WeakKeyDictionary()

async def read_console_line(console: Any, prompt_text: str) -> str:
    """Read a line while preserving input order, buffering, and cancellation."""
    stream = sys.stdin
    try:
        fd = stream.fileno()
    except (AttributeError, OSError) as exc:
        raise RuntimeError("console input fallback requires a selectable stdin") from exc
    reader = _CONSOLE_INPUTS.get(stream)
    if reader is None:
        reader = _ConsoleInput(stream)
        _CONSOLE_INPUTS[stream] = reader
    async with reader.lock:
        console.print(prompt_text, end="")
        # Regular files are immediately readable but cannot be registered with
        # every event-loop selector (notably Linux epoll).
        metadata = os.fstat(fd)
        if stat.S_ISCHR(metadata.st_mode) and metadata.st_rdev == os.stat(os.devnull).st_rdev:
            raise EOFError
        if stat.S_ISREG(metadata.st_mode):
            line = stream.readline()
            if line == "":
                raise EOFError
            return line.rstrip("\r\n")
        return await reader.read(fd)

def _pin_status_row_under_input(session: Any) -> None:
    """Keep the toolbar directly beneath the input.

    The renderer claims the remaining screen height. Top alignment and a
    non-growing input window put that slack below the toolbar.
    Layout access is best-effort so prompt_toolkit changes leave the run usable.
    """
    try:
        session.layout.container.align = VerticalAlign.TOP
        for container in session.layout.walk():
            if (
                isinstance(container, Window)
                and isinstance(container.content, BufferControl)
                and container.content.buffer is session.default_buffer
            ):
                container.dont_extend_height = to_filter(True)
    except Exception:
        pass

class LivePrompt:
    """Own the input line and the HUD above it.

    With ``interactive=False``, read stdin directly and leave terminal output
    unpatched.
    """

    def __init__(self, tui: Any, *, console: Any, interactive: bool = True):
        self._tui = tui
        self._console = console
        self._interactive = interactive
        self._session: Any | None = None
        self._questions: deque[_Question] = deque()
        self._stashed_input = ""

    @property
    def interactive(self) -> bool:
        return self._interactive

    @contextlib.contextmanager
    def attached(self):
        """Own the bottom region for the duration of the block.

        Keep ``patch_stdout`` active throughout and repaint when the TUI requests it.
        """
        if not self._interactive:
            yield self
            return
        with patch_stdout(raw=True):
            self._tui.set_redraw(self.invalidate)
            try:
                yield self
            finally:
                self._tui.set_redraw(None)

    def invalidate(self) -> None:
        """Repaint the bottom region, if it is on screen."""
        session = self._session
        app = getattr(session, "app", None)
        if app is not None and getattr(app, "is_running", False):
            app.invalidate()

    async def read(self) -> str:
        """Read one line: a user turn, or an answer an agent is waiting for."""
        if not self._interactive:
            return await read_console_line(self._console, _FALLBACK_PROMPT)
        session = self._ensure_session()
        try:
            line = await session.prompt_async(
                self._message,
                # Each prompt resets the buffer from its default. Keep the
                # user's draft until a read accepts it after the questions.
                default=self._stashed_input if not self._questions else "",
                bottom_toolbar=self._status,
                refresh_interval=REFRESH_INTERVAL,
                key_bindings=build_agent_navigation_bindings(self._tui),
                style=_PROMPT_STYLE,
            )
        except KeyboardInterrupt:
            if not self._questions:
                self._stashed_input = ""
            raise
        except (EOFError, asyncio.CancelledError):
            raise
        except Exception:
            line = await read_console_line(self._console, _FALLBACK_PROMPT)
        if not self._questions:
            self._stashed_input = ""
        return line

    async def ask(self, text: str) -> str:
        """Ask the user something on an agent's behalf and wait for the answer.

        The question takes the input line ahead of whatever the user was
        typing; their half-written message is put back when the last pending
        question has been answered.
        """
        if not self._interactive:
            return await read_console_line(self._console, text)
        question = _Question(text, asyncio.get_running_loop().create_future())
        self._questions.append(question)
        if len(self._questions) == 1:
            self._stash_input()
        self.invalidate()
        try:
            return await question.future
        finally:
            self._retire(question)

    def deliver(self, line: str) -> bool:
        """Hand ``line`` to the agent that asked first, if one is waiting."""
        while self._questions:
            question = self._questions.popleft()
            if question.future.done():
                continue
            question.future.set_result(line)
            self._settle_questions()
            return True
        return False

    def decline_pending(self) -> bool:
        """Decline waiting questions on interrupt.

        Cooperative turn cancellation cannot reach tools waiting for an answer.
        ``EOFError`` makes permission prompts deny and ``ask_user`` report a decline.
        """
        pending = list(self._questions)
        self._questions.clear()
        for question in pending:
            if not question.future.done():
                question.future.set_exception(EOFError())
        if pending:
            self._settle_questions()
        return bool(pending)

    @property
    def pending_question(self) -> str | None:
        """The question currently holding the input line, if any."""
        return self._questions[0].text if self._questions else None

    def _ensure_session(self) -> Any:
        if self._session is None:
            from prompt_toolkit import PromptSession

            session = PromptSession(erase_when_done=True)
            _pin_status_row_under_input(session)
            self._session = session
        return self._session

    def _message(self) -> Any:
        """Render the HUD and question or chevron above the cursor.

        Include status here when the terminal cannot render the bottom toolbar.
        """
        rows: list[Any] = []
        hud = self._tui.hud_ansi()
        if hud:
            rows.extend(to_formatted_text(ANSI(hud)))
            rows.append(("", "\n"))
        if not self._toolbar_renders():
            status = self._tui.status_ansi()
            if status:
                rows.extend(to_formatted_text(ANSI(status)))
                rows.append(("", "\n"))
        question = self.pending_question
        if question is not None:
            rows.extend(to_formatted_text(question))
            return rows
        rows.extend(_CHEVRON)
        return rows

    def _status(self) -> Any:
        """The row under the input line."""
        status = self._tui.status_ansi()
        return to_formatted_text(ANSI(status)) if status else ""

    def _toolbar_renders(self) -> bool:
        """Whether prompt_toolkit will actually paint the toolbar right now.

        It holds a ``bottom_toolbar`` back until the renderer knows how much
        room is left below the cursor, which on a vt100 terminal means a
        cursor-position report. One that never answers gets no toolbar, ever.
        """
        app = getattr(self._session, "app", None)
        renderer = getattr(app, "renderer", None)
        if renderer is None:
            return False
        try:
            return bool(renderer.height_is_known)
        except Exception:
            return False

    def _buffer(self) -> Any | None:
        return getattr(self._session, "default_buffer", None)

    def _stash_input(self) -> None:
        buffer = self._buffer()
        if buffer is None:
            return
        self._stashed_input = buffer.text
        buffer.text = ""

    def _settle_questions(self) -> None:
        """Give the input line back to the user once no agent is waiting."""
        if self._questions:
            self.invalidate()
            return
        buffer = self._buffer()
        stashed = self._stashed_input
        if buffer is not None and stashed:
            buffer.text = stashed
            buffer.cursor_position = len(stashed)
        self.invalidate()

    def _retire(self, question: _Question) -> None:
        """Drop a question that was cancelled rather than answered."""
        try:
            self._questions.remove(question)
        except ValueError:
            return
        self._settle_questions()

__all__ = ["LivePrompt", "REFRESH_INTERVAL", "read_console_line"]
