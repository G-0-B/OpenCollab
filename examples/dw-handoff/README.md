# Scripted three-agent handoff

This example sequences an analyst, coder, and tester over one repository.
Each agent uses a separate Git worktree, and the coder's commit is handed
to the tester for verification.

The related [handoff study](https://github.com/RISE-X-Lab/OpenCollab-Eval/blob/main/experiment/framework/collaborating-team.md)
and [model-directed team configuration](https://github.com/RISE-X-Lab/OpenCollab-Eval/blob/main/experiment/framework/configs/team.handoff.experiment.yaml)
are maintained in OpenCollab-Eval.

## Why every agent gets a worktree, including the ones that mostly read

Linked worktrees share `.git/objects`, so a sha committed in one is reachable
from every other the moment it exists. Isolation does not restrict what an agent
can receive — it is what makes receiving **observable**.

A tester that shares the coder's directory holds the coder's edits whether or
not it was ever told about them, so "the tester worked from the coder's output"
is not a question the run can answer. With separate trees, the tester holds that
work only if it checked out the sha.

## Reading the handoff record

Each round records the offered commit and the commit the tester reports adopting.

```
"offered_commit": "a1b2c3…",     # what the script handed over
"adopted_commit": "a1b2c3…",     # what the tester says it checked out
"handoff_taken": true            # whether those are the same
```

The script offers the commit and the tester chooses whether to check it out.
The returned `handoff_taken` field compares the tester's reported
`adopted_commit` with the offered string. It records the tester's claim.
Use the tester's tool trajectory, including `git rev-parse HEAD` and the
repository's native verification command, to verify checkout and test execution.

## Running it

From the root of your OpenCollab source checkout, set the example's discovery
directory and run it against the target workspace. The absolute directory
keeps discovery tied to this example when the target repository is elsewhere.

```bash
export OPENCOLLAB_WORKFLOWS_DIR="$(pwd)/examples/dw-handoff/workflows"
opencollab workflow run dw-handoff \
  --workspace /path/to/a/git/repo \
  --args '{"goal": "the task the agents are given", "allow_unisolated_shell": true}'
```

`allow_unisolated_shell` grants all three roles permission to run commands on
the host. Set it to JSON `true` for this local invocation so the coder can
commit and the tester can check out the offered commit. The default is `false`,
which requires an OS process sandbox for Bash. Each agent still receives its
own Git worktree when host shell execution is enabled. A worktree isolates
files and shares Git objects while commands run with the host's privileges.
With an injected environment that provides an OS process sandbox, keep the
default and pass only the task's `goal`.

The workspace must be a git repository with at least one commit. Worktrees need
something to branch from. Each agent's tree is created detached at the
workspace's base commit and removed when the run ends.

Isolation initialization excludes the current run's untracked transcript
records. For subsequent runs, add generated session records under
`.opencollab/sessions/` to the project's Git ignore rules to keep the source tree
clean. Tracked changes remain part of the source check.

**A worktree is a fresh checkout, so uncommitted files in the workspace are not
in it.** If your task depends on files that were written into the workspace
without being committed — a harness that applies test files and leaves them
dirty, for example — those files will be missing from every agent's tree. Commit
them first, or copy them in at creation.
