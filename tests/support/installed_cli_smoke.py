"""Exercise an installed wheel's CLI from a workspace outside the checkout."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import opencollab


def main() -> None:
    package = Path(opencollab.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    cli = Path(sys.executable).with_name("opencollab")
    results = []
    with tempfile.TemporaryDirectory(prefix="opencollab-installed-cli-") as directory:
        workspace = Path(directory)
        workflows = workspace / "workflows"
        workflows.mkdir()
        (workflows / "smoke.py").write_text(
            "from opencollab import workflow\n"
            "@workflow(name='offline-cli-smoke')\n"
            "async def smoke(ctx, args):\n"
            "    return {'fixture': args['fixture'], 'workspace': ctx.workspace_root}\n",
            encoding="utf-8",
        )
        env = {key: value for key, value in os.environ.items() if not key.startswith("OPENCOLLAB_")}
        env.pop("PYTHONPATH", None)
        env["OPENCOLLAB_WORKFLOWS_DIR"] = str(workflows)
        env["OPENCOLLAB_API_KEY"] = "offline-unused"  # pragma: allowlist secret
        commands = (
            ("--help",),
            ("workflow", "--help"),
            ("workflow", "run", "--help"),
            ("workflow", "list"),
            ("workflow", "run", "offline-cli-smoke", "--workspace", str(workspace),
             "--args", '{"fixture":"parsed-value"}', "--budget", "123", "--no-save", "--no-trace"),
        )
        for arguments in commands:
            result = subprocess.run(
                [str(cli), *arguments], cwd=workspace, env=env, capture_output=True, text=True, timeout=30,
            )
            assert result.returncode == 0, (arguments, result.stdout, result.stderr)
            if "offline-cli-smoke" in arguments:
                assert "parsed-value" in result.stdout, result.stdout
                assert str(workspace) in result.stdout, result.stdout
            results.append({"arguments": arguments, "exit_code": result.returncode})
    versions = {name: metadata.version(name) for name in ("opencollab", "typer")}
    try:
        versions["click"] = metadata.version("click")
    except metadata.PackageNotFoundError:
        versions["click"] = "vendored by Typer"
    print(json.dumps({
        "versions": versions,
        "package": str(package), "commands": results,
    }, indent=2))

if __name__ == "__main__":
    main()
