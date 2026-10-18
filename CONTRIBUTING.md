# Contributing to OpenCollab

This guide explains how to prepare a development environment and submit changes
that satisfy OpenCollab's checks and dependency rule.

## Project layout

OpenCollab follows strict clean architecture. Dependencies point inward.

```
adapters  →  application  →  domain
```

- `opencollab/domain/` contains pure value objects and the session FSM. It uses the standard library and performs no I/O.
- `opencollab/application/` contains use cases, the scheduler, and ports in `application/ports.py`. It imports `domain` and the standard library.
- `opencollab/adapters/` contains the CLI, TUI, LLM providers, tools, environments, tracing, and session store.
- `opencollab/bootstrap/` is the composition root and the only layer that knows concrete types.
- `opencollab/sdk/` is the versioned boundary for external workflow and evaluation packages.
- `scripts/` contains framework launchers and provider diagnostics.

## Development setup

OpenCollab uses [uv](https://docs.astral.sh/uv/). Run the following command from
the repository root.

```bash
uv sync --locked --extra dev   # create .venv with runtime + dev dependencies
```

Copy the example config and point it at an OpenAI-compatible or Anthropic
endpoint. **Never commit real API keys.**

```bash
cp configs/.env.example configs/.env   # then set OPENCOLLAB_API_KEY
```

## Checks (must pass before a PR)

```bash
uv run ruff check .   # lint the whole repository (root ruff.toml)
uv run lint-imports   # architecture contracts
uv run deptry .       # dependency hygiene
uv run pytest -q      # Python test suite
```

New behavior needs tests, and the suite must stay green. Provider credentials
are needed for live runs, not for the regular test suite.

### Find and run the relevant tests

Tests are grouped by behavior. Topic filenames and function names identify
individual regressions; example-specific tests stay beside their example.

| Directory | Behavior |
| --- | --- |
| `tests/agents/` | Profiles, single-agent sessions, turns and completion |
| `tests/workflows/` | Teams, topology, messaging, scheduling and workflows |
| `tests/tools/` | Tools, files, shell commands, Git and executable evidence |
| `tests/runtime/` | Providers, environments, cancellation, context, storage and tracing |
| `tests/interfaces/` | CLI, TUI and public SDK entry points |
| `tests/packaging/` | Architecture, repository ownership, metadata and distributions |
| `tests/support/` | Reusable doubles, preparation and test path resolution |

Use a directory, topic file, or pytest node ID for a focused run.

```bash
uv run pytest -q tests/agents
uv run pytest -q tests/workflows/test_duo_runtime.py
uv run pytest -q tests/tools/test_safe_files.py::test_read_rejects_fifo_without_blocking
```

The default suite also includes Mini Edict, ARC-Bench r6, and IO2Code tests, as
listed in `pyproject.toml`. Workspaces use pytest's system temporary directory.
Keep logs and reports outside the repository, for example:

```bash
uv run pytest -q -p no:cacheprovider --junitxml=/tmp/opencollab-tests.xml
```

Import shared preparation explicitly from `tests.support`. Keep a helper beside
its topic until another topic needs it. `tests.support.paths.REPO_ROOT` anchors
to the test tree in a checkout or source archive; `PACKAGE_ROOT` identifies the
imported package for subprocess tests. Session envelopes and recording doubles
live in `session_runtime_test_support`; tool builders live in
`tool_execution_test_support`. Construct each scenario with explicit inputs.
Fixtures own resources whose setup and cleanup follow the test lifetime.

### DOM and browser checks

With Node 20 or newer and the Python environment prepared:

```bash
npm ci --prefix tests/workflows/blueprint_dom --no-audit --no-fund
npm test --prefix tests/workflows/blueprint_dom
```

The DOM tests use `.venv/bin/python` by default. Set `OPENCOLLAB_TEST_PYTHON` to
an absolute interpreter path for another environment. ARC-Bench r6's opt-in
browser tests are skipped by the regular Python run. Its separate job uses
Node 22.12 or newer, locked npm dependencies, and Chromium; follow the
[example checker guide](examples/arcbench-evolution-r6/CHECKS.md) to run it.

### Test an unpacked source archive

The source distribution includes the tests and repository-check inputs.
Initialize local Git metadata for checks that enumerate source files, then use
the same development commands. A checkout already has this metadata.

```bash
git init --quiet
uv sync --locked --extra dev
uv run pytest -q
```

See [RELEASING.md](RELEASING.md) for clean builds and installed-artifact probes.

### CI and local hooks

CI runs the commands above on Python 3.10 through 3.14. The Python 3.12 job also
runs the team blueprint DOM regressions with Node 20 or newer. Distribution and
macOS jobs exercise installed artifacts and platform-specific integrity
behavior. A separate ARC-Bench r6 job runs the example's browser checks with
Node and Chromium. The local setups are described
[above](#dom-and-browser-checks).

The title workflow checks Conventional Commit syntax and rejects Chinese prose
in PR titles and non-merge commits pushed to `main`. It accepts a merge commit after the PR-title
check. Use a merge commit to retain the reviewed feature-branch history.

File hygiene rejects a newly added file larger than 512,000 bytes or a change
that grows a file past that limit. Python modules crossing 800 lines receive an
advisory warning. The complete-tree check runs on pushes to `main`. Consider
splitting modules by responsibility and public interface width when a module
becomes difficult to navigate or maintain. Commit `.tex` and `.md` sources and
keep compiled PDFs out of Git history.

The local pre-commit hooks cover Ruff with autofix, added large files,
whitespace, merge conflicts, YAML, TOML, and the existing secret baseline.
Install them with the following commands. Run the development commands above
as well because CI also exercises architecture, dependencies, and tests.

```bash
uv tool install pre-commit
pre-commit install
```

## The architecture rule (enforced by `lint-imports`)

`lint-imports` fails the build on any inward → outward import. The contracts live in
`.importlinter`, including the explicitly retained CLI and sibling-cycle
exceptions. Run it locally with `uv run lint-imports`.

- Never import an outer layer from an inner one (e.g. `domain` importing `adapters`).
- Need an outer capability inside? Add a **port** in `application/ports.py`, then
  wire the concrete implementation in `bootstrap/`.
- When splitting a module, keep its public names re-exported so import paths stay stable.

## Extend a tool or workflow

For application-specific tools, implement the public structural
[`Tool` protocol](opencollab/tools.py) and pass instances through
`OpenCollab.agent(tools=[...])` or each workflow role's `tools=` argument.
`Tool` describes the interface; it is not a base class supplying an
implementation. Provide `name`, `description`, the JSON Schema `parameters`,
`default_timeout`, `disable_outer_timeout`, `to_openai_schema()`, and async
`execute_with_runtime(params, runtime)`. Return a string result. Keep execution
state scoped to the call and use the supplied runtime for environment access.
The [SDK tool example](opencollab/README.md#custom-tools) shows the complete contract.

For a framework-owned tool, implement it under `opencollab/adapters/tools/`
using the existing adapter base, and register its factory in
`opencollab/bootstrap/tool_registry.py`. Expose it in `opencollab/tools.py` only
if it belongs in the public built-in set. Give roles explicit tool lists rather
than expanding everyone's permissions. Add a regression in `tests/tools/`
covering the tool's actual result and relevant permission or failure boundary.

For authored workflows, follow the executable
[workflow example](opencollab/README.md#workflow-authoring). Put reusable
framework behavior behind the public API; keep benchmark scoring and remote
execution in [OpenCollab-Eval](https://github.com/RISE-X-Lab/OpenCollab-Eval).
When diagnosing a run, distinguish execution status from application correctness
and use [run identity, usage and cleanup evidence](docs/evaluation-runtime.md)
before attributing a failure to a model or a workflow.

## Commits & pull requests

When changing an English document with an existing Chinese translation, update
both in the same PR. The Hygiene workflow checks the PR's net changes from its
merge base and lists any missing Chinese updates. Documents without a Chinese
version are skipped, and Chinese-only corrections can be submitted independently.
Pairs named `NAME.md` and `NAME.zh-CN.md` are discovered automatically. The Duo
guide also pairs `docs/duo.md` with `docs/duo/README.zh-CN.md`.

Run the same check locally after committing the edits.

```bash
python3 scripts/check_doc_translations.py origin/main HEAD
```

Use [Conventional Commits](https://www.conventionalcommits.org/) format with an
English type prefix and English descriptions for commits, PR titles, and merge
subjects. Commit bodies and PR descriptions also use English. Supported types
include `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`, `build`,
`style`, and `revert`. A `refactor` commit
preserves behavior.

Keep each PR focused and describe its changed behavior and verification in
English. Code, comments, and canonical tracked documentation use English;
translations retain their declared language. The title check detects Chinese
prose, not English grammar; authors and reviewers own the wording.
Commit authorship and `Co-authored-by` trailers are reserved for human
contributors. Merge with a merge commit to preserve feature-branch history and
human authorship.

## Documentation and comments

Keep setup, usage, and testing at their existing entry points and link to them. Move useful decisions and constraints
from superseded plans into maintained guides before removing the plans. Preserve release migrations and research evidence.

Use clear names and types. Comments and docstrings explain reasons, behavior, failures, ownership, assumptions, or constraints
the code leaves implicit, including caller requirements, edge cases, and tradeoffs. Self-explanatory functions need no
docstring. Avoid repeating parameter and return signatures, line-by-line narration, section banners, and past fixes.
Keep brief explanations near code, longer design rationale in maintained guides, and change history in Git or PRs.
Link to existing documentation instead of duplicating it.

Before editing text, inspect CLI help, workflow discovery and descriptions, tool/schema metadata, prompts, documentation, and
static source analysis consumers, then verify affected behavior. Preserve executable behavior, test inputs, safety and tool
instructions, lint directives, and copyright.
Markdown under `configs/`, packaged prompts, and skill directories can be runtime input. Experiment cards and generation
belong to [OpenCollab-Eval](https://github.com/RISE-X-Lab/OpenCollab-Eval/tree/main/experiment/framework). Edit their source templates
and run assembly and configuration checks there. Their exact text is an experiment condition.

## Contribution license

By submitting a contribution, you represent that you have the legal right to
provide it and that it does not knowingly include material you cannot license.
You license the contribution to the project under the
[Mulan Permissive Software License v2](LICENSE), the same license that applies
to the repository. OpenCollab currently uses this inbound-equals-outbound rule
without requiring a Developer Certificate of Origin sign-off.

Secret-baseline updates use a dedicated, baseline-only pull request. A
maintainer applies the `security-baseline-update` label after reviewing every
entry. The security workflow rejects a baseline change mixed with source or
documentation changes.

## Reporting issues

- Functional bugs and feature requests: open a GitHub issue using the templates.
- Security vulnerabilities must not be reported in a public issue. Use the
  private process in [SECURITY.md](SECURITY.md).
