"""Run one provider completion using the normal OpenCollab configuration."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from opencollab.adapters.llm import LLMClient
from opencollab.bootstrap.config import build_config
from opencollab.bootstrap.inspection import configured_model_client

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = "Reply with one short sentence confirming provider connectivity."

class CompletionClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...

    async def close(self) -> None: ...

async def request_completion(
    prompt: str,
    *,
    workspace: Path = WORKSPACE,
    client_type: type[CompletionClient] = LLMClient,
) -> str:
    """Return one completion using the same configuration as the framework."""
    config = build_config(str(workspace))
    if not config.api_key:
        raise ValueError("provider API key is missing from the OpenCollab configuration")
    client = configured_model_client(config.model_dump(), client_type=client_type)
    try:
        thinking_params = config.thinking_params if config.thinking or config.thinking_params_explicit else None
        if config.wire_protocol == "responses" and not config.thinking_params_explicit:
            thinking_params = None
        max_output_tokens = min(config.max_output_tokens, 256)
        minimum_output = getattr(client, "minimum_output_tokens", None)
        if callable(minimum_output):
            max_output_tokens = max(max_output_tokens, minimum_output(
                max_output_tokens=config.max_output_tokens,
                thinking=config.thinking,
                thinking_params=thinking_params,
            ))
        response = await client.complete(
            [
                {"role": "system", "content": "You are a concise connectivity probe."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_output_tokens=max_output_tokens,
            thinking=config.thinking,
            thinking_params=thinking_params,
            reasoning_effort=config.reasoning_effort,
        )
        return str(response.content or "")
    finally:
        await client.close()

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(asyncio.run(request_completion(args.prompt, workspace=args.workspace)))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
