"""Published DeepSeek windows retain file evidence through public agents."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from opencollab import OpenCollab
from opencollab.adapters.tools.fs import FileReadTool
from opencollab.domain.token_estimation import estimate_request_tokens
from tests.runtime.test_llm_chat_reasoning_replay_http import _completion_response


@contextmanager
def _seven_file_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            index = len(requests) - 1
            calls = None
            if index < 7:
                calls = [{"id": f"read_{index}", "type": "function", "function": {
                    "name": "file_read", "arguments": json.dumps({"path": f"evidence_{index}.txt"}),
                }}]
            response = _completion_response(
                body["model"], False, calls=calls,
                content="Continue reading." if calls else "done",
                reasoning="Read the next source file." if calls else None,
            )
            payload = response.json()
            prompt_tokens = estimate_request_tokens(body["messages"], body.get("tools"))
            payload["usage"].update(
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens + payload["usage"]["completion_tokens"],
            )
            content = json.dumps(payload).encode()
            self.send_response(response.status_code)
            self.send_header("Content-Type", response.headers["Content-Type"])
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

@pytest.mark.parametrize("model", [
    "deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash-vision-exp",
    "gateway/deepseek-flash-2026-10-14", "gateway/deepseek-v4-pro-2026-10-14",
    "gateway/deepseek-v4-flash-vision-exp-2026-10-14",
])
@pytest.mark.parametrize("window", [None, 64_000, 1_000_000])
async def test_public_agent_retains_files_with_published_window(monkeypatch, tmp_path, model, window):
    monkeypatch.delenv("OPENCOLLAB_CONTEXT_WINDOW", raising=False)
    monkeypatch.setenv("OPENCOLLAB_API_USAGE_LOG", str(tmp_path / "usage.jsonl"))
    for index in range(7):
        (tmp_path / f"evidence_{index}.txt").write_text(
            f"ORIGINAL_EVIDENCE_{index}\n" + "".join(
                f"{line} documented source text for file {index} carries independent details "
                "and ordinary project data\n"
                for line in range(500)
            ), encoding="utf-8",
        )
    config = {"provider": "openai", "wire_protocol": "chat_completions", "llm_max_retries": 0}
    if window is not None:
        config["context_window"] = window

    with _seven_file_server() as (base_url, requests):
        app = OpenCollab(
            tmp_path, model=model, api_key="controlled-test",  # pragma: allowlist secret
            base_url=base_url, config=config,
        )
        result = await app.agent(
            "Compare the seven source files and retain their original details for the final response.",
            tools=(FileReadTool(),), artifacts=tmp_path / "artifacts", trace=True,
            max_steps=9, budget=200_000,
        )

    assert result.ok and result.output == "done"
    assert len(requests) == 8
    assert all(request["model"] == model for request in requests)
    original_results = {}
    for index, request in enumerate(requests[1:]):
        fresh = next(message for message in request["messages"] if message.get("tool_call_id") == f"read_{index}")
        assert f"ORIGINAL_EVIDENCE_{index}" in fresh["content"]
        original_results[fresh["tool_call_id"]] = fresh["content"]
    final_results = {
        message["tool_call_id"]: message["content"]
        for message in requests[-1]["messages"] if message["role"] == "tool"
    }
    assert len(final_results) == 7
    if window == 64_000:
        assert final_results["read_0"] == "[Old tool result content cleared]"
        assert final_results["read_1"] == "[Old tool result content cleared]"
        for index in range(2, 7):
            assert final_results[f"read_{index}"] == original_results[f"read_{index}"]
    else:
        assert final_results == original_results
