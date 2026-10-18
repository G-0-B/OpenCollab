"""Explicit team submit tools deliver their answers through the real CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from opencollab import OpenCollab
from tests.support.paths import REPO_ROOT

ANSWER = "The verified total is 1499."
PREFACE = "I have checked the supplied totals."

@pytest.mark.parametrize("mode", ["submit", "same_text", "same_text_whitespace", "preface", "plain"])
def test_explicit_submit_team_prints_its_answer_once(tmp_path, mode):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            content = {"same_text": ANSWER, "same_text_whitespace": ANSWER + "\n",
                       "preface": PREFACE, "plain": ANSWER}.get(mode)
            message = {"role": "assistant", "content": content}
            if mode != "plain":
                summary = f" {ANSWER}\n" if mode == "same_text_whitespace" else ANSWER
                message["tool_calls"] = [{
                    "id": "submit-answer", "type": "function", "function": {
                        "name": "submit", "arguments": json.dumps({"summary": summary}),
                    },
                }]
            response = json.dumps({
                "id": "chat-local", "object": "chat.completion", "created": 1, "model": body["model"],
                "choices": [{"index": 0, "message": message,
                             "finish_reason": "stop" if mode == "plain" else "tool_calls"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    team = tmp_path / "team.yaml"
    team.write_text(
        "entry: lead\nroles:\n  lead:\n    prompt: Return the verified total.\n"
        "    tools: [submit]\ntopology:\n  lead: []\n",
        encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("OPENCOLLAB_", "OPENAI_", "ANTHROPIC_", "DASHSCOPE_"))}
    env.update(PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1", NO_PROXY="127.0.0.1,localhost",
               OPENCOLLAB_API_USAGE_LOG=str(tmp_path / "usage.jsonl"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [sys.executable, "-B", "-m", "opencollab", "--workspace", str(tmp_path),
             "--team-config", str(team), "--no-worktrees", "--provider", "openai",
             "--model", "local-submit", "--api-key", "local-test",
             "--base-url", f"http://127.0.0.1:{server.server_port}/v1", "--budget", "100000",
             "--prompt", "Return the verified total."],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(requests) == 1
    assert [tool["function"]["name"] for tool in requests[0]["tools"]] == ["submit"]
    snapshots = list((tmp_path / ".opencollab" / "sessions").rglob("agent_0_*.json"))
    assert len(snapshots) == 1
    snapshot = OpenCollab.read_session_snapshot(snapshots[0])
    assert snapshot["messages"][-1]["content"] == ANSWER
    assert result.stdout.count(ANSWER) == 1, result.stdout
    if mode == "preface":
        assert result.stdout.count(PREFACE) == 1
