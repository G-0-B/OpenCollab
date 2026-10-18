"""Caller workflow files follow Python source encoding rules."""

from __future__ import annotations

import importlib.util
import sys

import pytest
from typer.testing import CliRunner

from opencollab import OpenCollab
from opencollab.adapters.cli.main import app
from opencollab.bootstrap.workflow_runtime import load_workflow_specs


def _encoded(text, encoding):
    cookie = "# coding: cp1252\n" if encoding == "cp1252" else ""
    return (cookie + text).encode(encoding)

@pytest.mark.parametrize("entry", ["cli", "sdk"])
@pytest.mark.parametrize("location", ["entry", "sibling", "package"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252"])
async def test_public_workflow_uses_python_source_encoding(tmp_path, monkeypatch, entry, location, encoding):
    monkeypatch.delenv("OPENCOLLAB_WORKFLOWS_DIR", raising=False)
    directory = tmp_path / "workflows"
    directory.mkdir()
    file = directory / "entry.py"
    value = 'value = "caf\u00e9"\n'
    imports = ""
    if location == "entry":
        declarations = value
        target = file
    else:
        declarations = ""
        if location == "sibling":
            target = directory / "_values.py"
            imports = "from ._values import value\n"
        else:
            target = directory / "_values" / "__init__.py"
            target.parent.mkdir()
            imports = "from ._values import value\n"
        target.write_bytes(_encoded(value, encoding))
    source = (
        "from opencollab import workflow\n" + imports + declarations
        + '@workflow(name="source-encoding")\n'
        + "async def run(ctx, args):\n"
        + "    return value\n"
    )
    file.write_bytes(_encoded(source, encoding if location == "entry" else "utf-8"))
    expected = target.read_bytes()
    control = importlib.util.spec_from_file_location("python_encoding_control", target)
    module = importlib.util.module_from_spec(control)
    control.loader.exec_module(module)
    assert module.value == "caf\u00e9"
    before = {name for name in sys.modules if name.startswith("_opencollab_workflow_")}

    if entry == "cli":
        result = CliRunner().invoke(app, ["workflow", "list", "--workspace", str(tmp_path)])
        assert result.exit_code == 0, result.exception
        assert "source-encoding" in result.stdout
    else:
        result = await OpenCollab(tmp_path, config={"model": "test-model"}).workflow("source-encoding", trace=False)
        assert result.status == "completed", (result.reason, result.error)
        assert result.output == "caf\u00e9"
    assert target.read_bytes() == expected
    assert {name for name in sys.modules if name.startswith("_opencollab_workflow_")} == before

@pytest.mark.parametrize("payload", [b"value = '\xff'\n", b"# coding: missing-codec\nvalue = 1\n"])
def test_invalid_python_encoding_preserves_file_and_import_state(tmp_path, payload):
    source = tmp_path / "entry.py"
    source.write_bytes(payload)
    before = {name for name in sys.modules if name.startswith("_opencollab_workflow_")}
    with pytest.raises(SyntaxError):
        compile(payload, str(source), "exec")
    with pytest.raises(SyntaxError):
        load_workflow_specs(str(source))
    assert source.read_bytes() == payload
    assert {name for name in sys.modules if name.startswith("_opencollab_workflow_")} == before
