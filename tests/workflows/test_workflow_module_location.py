"""File-backed workflows and their siblings retain ordinary module locations."""

from pathlib import Path

import pytest

from opencollab import OpenCollab
from opencollab.bootstrap.workflow_runtime import load_workflow_specs


@pytest.mark.parametrize("resource_owner", ["workflow", "sibling", "package"])
async def test_named_workflow_can_read_adjacent_resources(tmp_path, resource_owner):
    directory = tmp_path / "workflows"
    directory.mkdir()
    reader = '''from pathlib import Path
import json
def read():
    return {"data": json.loads(Path(__file__).with_name("settings.json").read_text()),
            "file": __file__, "origin": __spec__.origin, "located": __spec__.has_location}
'''
    if resource_owner == "workflow":
        dependency = reader
        location = directory / "entry.py"
    elif resource_owner == "sibling":
        dependency = "from .helper import read\n"
        location = directory / "helper.py"
        location.write_text(reader)
    else:
        package = directory / "resources"
        package.mkdir()
        dependency = "from .resources import read\n"
        location = package / "__init__.py"
        location.write_text(reader)
    location.with_name("settings.json").write_text('{"answer": 42}')
    (directory / "entry.py").write_text(dependency + '''from opencollab.workflows import workflow
@workflow(name="adjacent-resource")
async def run(ctx, args):
    return read()
''')

    client = OpenCollab(tmp_path, config={"model": "test-model"})
    result = await client.workflow("adjacent-resource", trace=False)

    assert result.status == "completed", result.reason
    assert result.output == {"data": {"answer": 42}, "file": str(location),
                             "origin": str(location), "located": True}
    assert not list(directory.rglob("*.pyc"))

async def test_relative_workflow_path_keeps_location_after_directory_change(tmp_path, monkeypatch):
    directory = tmp_path / "workflows"
    directory.mkdir()
    source = directory / "entry.py"
    source.write_text('''from pathlib import Path
from opencollab.workflows import workflow
@workflow(name="location")
async def run(ctx, args):
    return Path(__file__).with_name("value.txt").read_text()
''')
    source.with_name("value.txt").write_text("local resource")
    monkeypatch.chdir(tmp_path)
    spec = load_workflow_specs("workflows/entry.py")[0]
    monkeypatch.chdir(tmp_path.parent)
    assert await spec.fn(None, {}) == "local resource"
    assert Path(spec.fn.__wrapped__.__globals__["__file__"]) == source
