# IO2Code reconstruction

[Chinese guide](README.zh-CN.md)

Reconstruct a hidden program from its input/output behaviour alone. The
benchmark is IO2CodeBench from the FromIO paper ([arXiv 2605.15334](https://arxiv.org/abs/2605.15334),
code at [JiaruQian/IO2Code](https://github.com/JiaruQian/IO2Code)): 125 small
Python functions, each with 8 visible examples and 15 held-out tests. A task
counts as solved only when all 15 held-out tests pass.

Everything specific to the benchmark lives in this directory. The exploration
protocol is OpenCollab's built-in [`reconstruct` workflow](../../docs/reconstruct.md);
OpenCollab also provides the agent sessions, tools, Docker environment, budgets
and traces.

```text
main.py        binds an IO2Code task to the reconstruct workflow, scores held-out tests, writes result.json
tasks.py       loads a task from the IO2Code checkout (oracle + held-out stay in this process)
tools.py       query_oracle, check_candidate, candidate execution inside the container
workflow.py    a free "probe, code, check" workflow and the benchmark's fixed mode (the non-protocol modes)
tests/         offline unit tests (no model, no Docker, no checkout)
```

## Prerequisites

- Linux or macOS with Docker. The agent works inside a `python:3.12-slim`
  container with no network; candidate programs are executed in that same
  container, never on the host.
- An IO2Code checkout at the revision this example was validated against:

  ```bash
  git clone https://github.com/JiaruQian/IO2Code.git
  git -C IO2Code checkout ef15827
  export IO2CODE_ROOT=$PWD/IO2Code
  ```

  The checkout is read only by this process (oracle functions, held-out
  tests, input generators). It is never mounted into the container.
- A model configured for OpenCollab in `configs/.env` (see
  [configs/README.md](../../configs/README.md)): `OPENCOLLAB_MODEL`,
  `OPENCOLLAB_API_KEY`, `OPENCOLLAB_BASE_URL` and, for Responses-only models,
  `OPENCOLLAB_WIRE_PROTOCOL=responses`. Set `OPENCOLLAB_MAX_OUTPUT_TOKENS` high
  enough for reasoning models (65536 was used for the results below). Models
  that re-encrypt reasoning in the terminal Responses event (gpt-6-luna) need
  `OPENCOLLAB_TRUST_STREAMED_OUTPUT_ON_TERMINAL_MISMATCH=1` in the process
  environment.

## Run

From the repository root:

```bash
uv run python examples/io2code_oc/main.py --list
uv run python examples/io2code_oc/main.py Abs_Current Extra_PolishRPN \
  --random-share 0.5 --coverage-gate --docker python:3.12-slim
```

Each task gets a fresh git workspace and container under
`examples/io2code_oc/runs/<run-name>/<task>/`, with `result.json`
(status, held-out score, queries, tokens, the final program), `oracle_queries.json`
(every input/output pair the agent was given, with its source) and OpenCollab's
`artifacts/` trace. A task that crashes or is skipped still gets a
`result.json` (status `crashed`/`skipped`), so `summary.json`, rebuilt from
every `result.json` in the run directory, keeps the full task count.
`--concurrency N` runs tasks side by side; `--resume` skips finished tasks and
retries crashed or skipped ones. Failed provider calls that produce neither a candidate nor
successful output are recorded as `infrastructure_error` with `agent_failures` and retried by
`--resume`, including older empty initial-proposal failures saved as completed.

## Modes

| `--mode` | evidence the agent gets | who queries the oracle |
|---|---|---|
| `paper-active` (default) | 2 examples to start, a new batch (+2 each time) only after the candidate passes every known example; 5 consecutive promotions stop early; at most 50 iterations | the built-in `reconstruct` workflow, following the paper's Section 4.6 protocol; the agent proposes inputs |
| `fixed` | the benchmark's 8 visible examples | nobody |
| `active` | nothing up front | the agent, through `query_oracle`, up to `--max-queries` |

`paper-active` is the built-in `reconstruct` workflow, which mirrors
`run_active_dio_agent.py` from the paper's repository (batch sizes, promotion
rule, early stop, retry of rejected proposals, the stage score that picks the
final program). The program-writing step is an OpenCollab agent session instead
of the paper's evolutionary DIO-Agent. `main.py` only supplies the binding: the
oracle, the benchmark's input generator, candidate execution in the container,
the tools, and the held-out scoring afterwards.

## Optional disciplines (the reconstruct workflow's arguments)

| flag | rule |
|---|---|
| `--random-share 0.5` | half of every batch comes from the benchmark's own input generator instead of the agent, so part of the evidence is independent of the agent's hypothesis |
| `--coverage-gate` | a promotion counts towards early stop only once the known inputs cover the workflow's structural case categories (empty, singleton, boundary values, duplicates, ...); missing categories are requested from the agent |
| `--memorisation-gate` | a candidate that embeds known inputs verbatim is not a verified pass |
| `--complexity-gate` | a pass by an over-complex candidate needs more evidence before it counts |
| `--domain-check` | agent-proposed inputs outside the task's numeric range are rejected |
| `--rea` | all of the above with `--random-share 0.5` |

## Results

These results cover the 115 non-multimodal tasks. The 10 multimodal tasks
need a vision model at runtime and are not covered.

Model `gpt-6-luna` (OpenAI Responses API, `OPENCOLLAB_MAX_OUTPUT_TOKENS=65536`), OpenCollab at
`1212d526`, IO2Code at `ef15827`, Docker image `python:3.12-slim`, one run per task and setting.
Solved = all 15 held-out tests pass.

| setting | Base (62) | Algorithm (35) | Geometry (18) | total | tokens |
|---|---|---|---|---|---|
| `--mode fixed` (benchmark protocol, 8 visible examples) | 49 | 28 | 13 | 90 / 115 (78.3%) | 2.3M |
| `--mode paper-active` (paper protocol) | 53 | 33 | 18 | 104 / 115 (90.4%) | 12.7M |
| `--random-share 0.5 --coverage-gate` | 60 | 35 | 18 | 113 / 115 (98.3%) | 18.4M |
| `--rea` (all five disciplines) | 60 | 35 | 18 | 113 / 115 (98.3%) | 30.4M |

Reading the three paper-protocol rows:

- Every failure of the plain paper protocol ended by early stop with all self-chosen inputs passing
  (Prev4/5, Sum_Last7, Alternating_Last3, Div_7, Base_4_Addition, Bit_Palindrome, both parentheses
  tasks, Extra_IsPalindrome): the agent over-fits inputs it picked itself. Those are also the tasks the
  fixed protocol solved, because the benchmark's random examples exposed the rule.
- Drawing half of every batch from the benchmark's generator plus the coverage gate fixed all of them;
  two new failures appeared (Base_5_Addition: a lookup table after the agent's own out-of-range inputs
  polluted the evidence; Majority_0_3 9/15).
- The remaining three disciplines fixed those two but lost two others (Bit_Palindrome 14/15,
  Majority_0_2 11/15) at +65% tokens: the complexity gate makes the agent patch a long spurious rule
  instead of abandoning it. The two-discipline setting is the cost-effective default.

Per-task query counts in the paper protocol are 42 at early stop (2 + 4 + 6 + 8 + 10 + 12).
A rerun of 12 tasks on `deepseek-flash` (DeepSeek-V4.1-Flash, low reasoning effort, 1.5M tokens per
task) with the two-discipline setting solved 5/12; the failures ran out of budget after 1–4 iterations.

## Trust model

The oracle and the held-out tests never leave this process. The agent's
container has no network and sees only its workspace. Candidate programs run
inside that container with a 30 s timeout, and their outputs are compared with
the same rules as IO2Code's evaluator. Task names are never shown to the agent
(many of them spell out the rule).
