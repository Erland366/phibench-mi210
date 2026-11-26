# Tasks come from any agent-assisted system work, in Φ-Bench's own format

Status: accepted (2026-11-19)

The group (meeting of 2026-11-19) asked for a procedure that turns the system work each person already does with a
coding agent into benchmark tasks. A task's source is therefore any finished piece of such work: a repository, the
commit before and after, and the trace. A GEAK run is one kind of source, not the only one. Every task is an exact
Φ-Bench task package with its row in the 8 unchanged columns, both speed-scored (`performance`) and pass/fail
(`implementation`, which upstream already uses for 8 of its 85 tasks).

## Considered options

- GEAK runs only: the existing tooling, but no teammate has GEAK or its Round check, so it cannot scale to them.
- A lighter format of our own (repository, commit and test command, no image), converted later: cheaper per task,
  but it leaves "can the task be run the same way everywhere" unanswered and makes the merge into Φ-Bench a second
  project.

## Consequences

- Every task needs a Dockerfile that builds and a verifier; making that cheap is the procedure's main job.
- The GEAK-specific input contract (`docs/inputs.md`) covers one source only; other sources need their own.
