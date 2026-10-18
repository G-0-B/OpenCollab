"""Merge CLI arguments with .env defaults and render missing-key hints.

Provider-specific key knowledge lives in ``bootstrap.config``.
"""

from __future__ import annotations

import typer
from rich.console import Console

from opencollab.bootstrap.config import accepted_api_key_envs, missing_api_key


def missing_api_key_for(provider: str | None, api_key: str | None, base_url: str | None = None) -> bool:
    return missing_api_key(provider, api_key, base_url)

def print_missing_key_hint(
    console: Console, provider: str | None, base_url: str | None = None
) -> None:
    from opencollab.adapters.tui import TUI

    tui = TUI(console)
    tui.print_welcome()
    accepted = " or ".join(accepted_api_key_envs(provider, base_url))
    console.print(
        f"[red]Missing API key[/red]: pass [bold]--api-key[/bold] or set [bold]{accepted}[/bold]."
    )

def resolve_config(workspace: str, model: str | None, provider: str | None,
                   api_key: str | None, base_url: str | None, budget: int | None) -> dict:
    """Merge CLI args with .env defaults. CLI args take precedence."""
    from opencollab.bootstrap.config import build_config

    # ``filter_messages`` remains a compatibility input.
    try:
        cfg = build_config(
            workspace,
            overrides={
                "model": model,
                "provider": provider,
                "api_key": api_key,
                "base_url": base_url,
                "budget": budget,
            },
        ).model_dump()
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"Cannot load configuration: {exc}") from exc
    return cfg
