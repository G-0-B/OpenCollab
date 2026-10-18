# OpenCollab documentation

Start with the [project README](../README.md) for a first run. Use this index
when extending a runtime or reproducing a result.

## Use and extend OpenCollab

| Task | Guide |
| --- | --- |
| Install; use the CLI, SDK, or authored workflows | [Package guide](../opencollab/README.md) |
| Set providers, models, teams, and context policy | [Configuration](../configs/README.md) |
| Choose the default or a concrete agent profile | [Base and Single2](single2.md) |
| Compare and adopt two isolated candidates | [Duo](duo.md) · [Chinese translation](duo/README.zh-CN.md) |
| Run ordered work with checks, repair, and continuation | [Weave](weave.md) |
| Reconstruct a hidden program through active exploration | [Reconstruct](reconstruct.md) |
| Control workflow calls; integrate evaluation usage, cancellation and cleanup | [Evaluation runtime](evaluation-runtime.md) |
| Inspect model capabilities and effective profiles offline | [Model inspection](model-inspection.md) |
| Record native test observations and parse Git patches | [Test evidence](test-evidence.md) |
| Add on-demand instructions | [Skills](../skills/README.md) |
| Understand context, budget allocation, and steering | [Runtime design](runtime-design.md) |

## Contribute and maintain

| Task | Guide |
| --- | --- |
| Prepare a contribution and follow collaboration rules | [Contributing](../CONTRIBUTING.md) |
| Find tests; run Python, DOM, and browser checks | [Contributor checks and test map](../CONTRIBUTING.md#checks-must-pass-before-a-pr) |
| Use launchers, diagnostics, and experiment-card generation | [Scripts](../scripts/README.md) |
| Measure tool-loop processing and progress recovery | [Tool-loop measurements](development/tool-loop-progress.md) |
| Publish a tagged GitHub release | [Releasing](../RELEASING.md) |
| Report a vulnerability privately | [Security](../SECURITY.md) |

## Research evidence and history

[Benchmark results](results.md) reports the recorded cross-harness comparison
and paired Duo–Base test. [Adherence](adherence.md) separates declared team
conditions from measured collaboration. The
[collaborating-team study](https://github.com/RISE-X-Lab/OpenCollab-Eval/tree/main/experiment/framework/collaborating-team.md)
and experiment cards live with their generators and checks in OpenCollab-Eval. Benchmark datasets, runs, and remote execution belong to
[OpenCollab-Eval](https://github.com/RISE-X-Lab/OpenCollab-Eval#readme).

The [ICLR integration record](iclr-integration.md) and
[source coverage](iclr-source-coverage.json) preserve the October 1 integration
sequence and ancestry. The [June architecture survey](archive/2026-06-10-architecture-survey.md)
retains historical findings and rejected alternatives, not a current defect list
or refactoring plan. Its unresolved recommendations need fresh verification.

## Upgrade an older integration

Migration guides describe the named release, not current signatures or defaults.

| From / to | Guide |
| --- | --- |
| 0.7.x → 0.8.0 | [Base profile and Single2 defaults](migrations/0.8.0.md) |
| 0.6.x → 0.7.0 | [Duo, evaluation runtime, and native Bash testing](migrations/0.7.0.md) |
| 0.5.x → 0.6.0 | [Public API, teams, workflows, and lifecycle](migrations/0.6.0.md) |
| Earlier versions → 0.5.0 | [Runtime failures, isolation, configuration, and budgeting](migrations/0.5.0.md) |
