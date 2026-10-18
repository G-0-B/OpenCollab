"""Workflow packages expose adjacent data through importlib.resources."""

from __future__ import annotations

import asyncio
import importlib.machinery
import importlib.util
import sys
from importlib.resources import files

import pytest

from opencollab import OpenCollab
from opencollab.bootstrap import _workflow_runtime_discovery as workflow_discovery


def _import_state():
    return (
        {name for name in sys.modules if name.startswith("_opencollab_workflow_")},
        dict(workflow_discovery._WORKFLOW_IMPORT_FINDER._package_roots),
    )

def _bytecode_contents(directory):
    return {path.relative_to(directory): path.read_bytes() for path in directory.rglob("*.pyc")}

def _write_resource_workflow(workspace, owner, value):
    directory = workspace / "workflows"
    directory.mkdir(parents=True)
    assets = directory if owner == "root" else directory / "_assets"
    assets.mkdir(exist_ok=True)
    if owner == "package":
        (assets / "__init__.py").write_text("", encoding="utf-8")
    payload = assets / "nested"
    payload.mkdir()
    (payload / "answer.txt").write_text(value, encoding="utf-8")
    (directory / "_helper.py").write_text(f"VALUE = {value!r}\n", encoding="utf-8")
    owner_import = (
        "    owner = import_module(__package__)\n" if owner == "root"
        else "    from . import _assets as owner\n"
    )
    (directory / "entry.py").write_text(
        "import asyncio\n"
        "from importlib import import_module\n"
        "from importlib.resources import files\n"
        "from opencollab import workflow\n"
        "from ._helper import VALUE\n"
        '@workflow(name="package-resource")\n'
        "async def run(ctx, args):\n"
        + owner_import
        + "    await asyncio.sleep(0)\n"
        '    return [files(owner).joinpath("nested").joinpath("answer.txt").read_text(encoding="utf-8"), VALUE]\n',
        encoding="utf-8",
    )
    return directory, assets

@pytest.mark.parametrize("owner", ["package", "namespace", "root"])
async def test_sdk_resources_keep_workspace_isolation_and_reload(tmp_path, monkeypatch, owner):
    monkeypatch.delenv("OPENCOLLAB_WORKFLOWS_DIR", raising=False)
    first = tmp_path / "first"
    second = tmp_path / "second"
    directory, assets = _write_resource_workflow(first, owner, "ALPHA")
    _write_resource_workflow(second, owner, "BRAVO")

    name, parent = ("workflows", first) if owner == "root" else ("_assets", directory)
    control_spec = importlib.machinery.PathFinder.find_spec(name, [str(parent)])
    control = importlib.util.module_from_spec(control_spec)
    control_spec.loader.exec_module(control)
    assert files(control).joinpath("nested").joinpath("answer.txt").read_text(encoding="utf-8") == "ALPHA"

    before = _import_state()
    bytecode_before = _bytecode_contents(tmp_path)
    clients = [OpenCollab(path, config={"model": "test-model"}) for path in (first, second)]
    results = await asyncio.gather(*(client.workflow("package-resource", trace=False) for client in clients))

    assert [result.status for result in results] == ["completed", "completed"], [result.reason for result in results]
    assert [result.output for result in results] == [["ALPHA", "ALPHA"], ["BRAVO", "BRAVO"]]
    assert _import_state() == before
    assert _bytecode_contents(tmp_path) == bytecode_before

    (assets / "nested" / "answer.txt").write_text("NEWER", encoding="utf-8")
    (directory / "_helper.py").write_text('VALUE = "NEWER"\n', encoding="utf-8")
    refreshed = await clients[0].workflow("package-resource", trace=False)

    assert refreshed.status == "completed", refreshed.reason
    assert refreshed.output == ["NEWER", "NEWER"]
    assert _import_state() == before
    assert _bytecode_contents(tmp_path) == bytecode_before
