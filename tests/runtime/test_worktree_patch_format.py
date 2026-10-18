"""Exported worktree patches remain applicable under Git display settings."""

import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from opencollab.adapters.env import WorktreeEnvironment


@pytest.mark.parametrize("git_repo", [False, True])
@pytest.mark.parametrize("settings", [
    {}, {"diff.noprefix": "true"}, {"color.ui": "always"}, {"diff.context": "0"},
    {"diff.srcPrefix": "before/custom/", "diff.dstPrefix": "after/custom/", "color.diff": "always"},
])
async def test_exported_patch_applies_with_display_settings(tmp_path, monkeypatch, git_repo, settings):
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.py").write_text("alpha\nbeta\ngamma\ndelta\n")
    (source / "binary.bin").write_bytes(b"\x00\xfforiginal\n")

    def git(*args, **kwargs):
        return subprocess.run(["git", "-C", str(source), *args], text=True, capture_output=True, check=True, **kwargs)

    global_config = tmp_path / "gitconfig"
    global_config.touch()
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    if git_repo:
        git("init", "-q")
        git("add", ".")
        git("-c", "user.name=Test User", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", "fixture")
    for key, value in settings.items():
        git("config", "--local" if git_repo else "--global", key, value)

    env = WorktreeEnvironment(str(source))
    try:
        await env.setup()
        await env.write_file("source.py", "alpha\nBETA\ngamma\ndelta\n")
        (Path(env.workspace) / "binary.bin").write_bytes(b"\x00\xffupdated\n")
        await env.write_file("new.txt", "new contents\n")
        patch = await env.get_diff()
        assert "\x1b[" not in patch
        assert env.workspace not in patch
        assert "GIT binary patch" in patch
        git("apply", "--check", "-", input=patch)
        git("apply", "-", input=patch)
        assert (source / "source.py").read_text() == "alpha\nBETA\ngamma\ndelta\n"
        assert (source / "binary.bin").read_bytes() == b"\x00\xffupdated\n"
        assert (source / "new.txt").read_text() == "new contents\n"
        for key, value in settings.items():
            assert git("config", "--get", key).stdout.strip() == value
    finally:
        await env.cleanup()

@pytest.mark.parametrize("binary", [False, True])
async def test_directory_copy_rename_exports_an_applicable_patch(tmp_path, monkeypatch, binary):
    source = tmp_path / "source"
    source.mkdir()
    before_name, after_name = "before name.txt", "after name.txt"
    payload = b"\x00\xffrename payload\n" if binary else b"ordinary rename contents\n"
    (source / before_name).write_bytes(payload)
    config = tmp_path / "gitconfig"
    config.write_text("[diff]\n\trenames = true\n")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    env = WorktreeEnvironment(str(source))
    try:
        await env.setup()
        workspace = Path(env.workspace)
        (workspace / before_name).rename(workspace / after_name)
        patch = await env.get_diff()
        assert env.workspace not in patch
        assert "opencollab-cp-baseline-" not in patch
        for arguments in [("apply", "--check", "-"), ("apply", "-")]:
            subprocess.run(
                ["git", "-C", str(source), *arguments], input=patch,
                text=True, capture_output=True, check=True,
            )
        assert not (source / before_name).exists()
        assert (source / after_name).read_bytes() == payload
    finally:
        await env.cleanup()

@pytest.mark.parametrize("temporary_name", [
    "plain temp", "\u4e2d\u6587\u4e34\u65f6", 'quoted " directory', "tab\tdirectory",
])
@pytest.mark.parametrize("quote_path", [True, False])
async def test_directory_copy_patch_applies_after_cleanup_with_quoted_temp_paths(
    tmp_path, monkeypatch, temporary_name, quote_path,
):
    source = tmp_path / "source"
    source.mkdir()
    before = {
        "source.py": b"value = 1\n",
        '\u5b50\u76ee\u5f55/quoted " name\t.txt': "old caf\u00e9\n".encode(),
        "\u6570\u636e\nblob.bin": b"\x00\xfforiginal\n",
        "legacy\\records.csv": b"name\ncaf\xe9\n",
        "old \u8bb0\u5f55.txt": b"pi\xf1ata\n",
    }
    after = {
        "source.py": b"value = 2\n",
        '\u5b50\u76ee\u5f55/quoted " name\t.txt': "new caf\u00e9\n".encode(),
        "\u6570\u636e\nblob.bin": b"\x00\xffupdated\n",
        "legacy\\records.csv": b"name\npi\xf1ata\n",
        'new\t"\\ file.txt': b"new contents\n",
    }
    for name, contents in before.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    temporary = tmp_path / temporary_name
    temporary.mkdir()
    monkeypatch.setenv("TMPDIR", str(temporary))
    monkeypatch.setattr(tempfile, "tempdir", None)
    config = tmp_path / "gitconfig"
    config.write_text(f"[core]\n\tquotePath = {str(quote_path).lower()}\n")
    configured = config.read_bytes()
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(config))
    env = WorktreeEnvironment(str(source))
    try:
        workspace = Path(await env.setup())
        assert workspace.parent == temporary.resolve()
        for name in before.keys() - after.keys():
            (workspace / name).unlink()
        for name, contents in after.items():
            (workspace / name).write_bytes(contents)
        patch = await env.get_diff()
        assert "+value = 2\n" in patch
        assert "GIT binary patch" in patch
        assert "\ufffd" not in patch
        assert "opencollab-cp-" not in patch
        assert {name: (source / name).read_bytes() for name in before} == before
        await env.cleanup()
        assert not workspace.exists()
        assert not list(temporary.iterdir())
        for arguments in [("apply", "--check", "-"), ("apply", "-")]:
            subprocess.run(
                ["git", "-C", str(source), *arguments], input=patch.encode(),
                capture_output=True, check=True,
            )
        delivered = {
            str(path.relative_to(source)): path.read_bytes()
            for path in source.rglob("*") if path.is_file()
        }
        assert delivered == after
        assert config.read_bytes() == configured
    finally:
        await env.cleanup()

@pytest.mark.parametrize("git_repo", [False, True])
@pytest.mark.parametrize("change", ["value", "format", "minify_and_value"])
async def test_exported_patch_ignores_global_json_textconv(tmp_path, monkeypatch, git_repo, change):
    source = tmp_path / "source workspace"
    source.mkdir()
    relative = Path("project data") / "settings.json"
    original = {"name": "demo", "score": 1}
    changed = {"name": "demo", "score": 1 if change == "format" else 2}

    def encode(data, pretty=False):
        options = {"indent": 2, "sort_keys": True} if pretty else {"separators": (",", ":")}
        return (json.dumps(data, **options) + "\n").encode()

    before = encode(original, pretty=change == "minify_and_value")
    after = encode(changed, pretty=change == "format")
    (source / relative).parent.mkdir()
    (source / relative).write_bytes(before)
    global_config = tmp_path / "gitconfig"
    global_config.touch()
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))

    def git(*args, **kwargs):
        return subprocess.run(["git", "-C", str(source), *args], capture_output=True, check=True, **kwargs)

    formatter = tmp_path / "pretty_json.py"
    formatter.write_text(
        "import json, sys\n"
        "with open(sys.argv[1]) as source:\n"
        "    print(json.dumps(json.load(source), indent=2, sort_keys=True))\n"
    )
    attributes = tmp_path / "attributes"
    attributes.write_text("*.json diff=pretty-json\n")
    converter = shlex.quote(sys.executable) + " " + shlex.quote(str(formatter))
    git("config", "--global", "core.attributesFile", str(attributes))
    git("config", "--global", "diff.pretty-json.textconv", converter)
    configured = global_config.read_bytes()
    if git_repo:
        git("init", "-q")
        git("add", ".")
        git("-c", "user.name=Test User", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "-qm", "\u51c6\u5907JSON\u8865\u4e01\u5bf9\u7167")

    env = WorktreeEnvironment(str(source))
    try:
        await env.setup()
        await env.write_file(str(relative), after.decode())
        patch = await env.get_diff()
        assert patch
        assert (source / relative).read_bytes() == before
        git("apply", "--check", "-", input=patch.encode())
        git("apply", "-", input=patch.encode())
        assert (source / relative).read_bytes() == after
        assert global_config.read_bytes() == configured
    finally:
        await env.cleanup()
