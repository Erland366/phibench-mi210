# Contributors build their own tasks with shared tooling

Status: accepted (2026-11-19)

About five people, each aiming at about 50 tasks, cannot all go through one person: Φ-Bench ended with 85 tasks out
of about 1,800 repositories largely because people built and tested every task by hand. So each contributor fills
in a task form, and one shared tool builds the package and runs the checks; the first few tasks from each
contributor are reviewed before their tasks count.

## Considered options

- Contributors send their work to one person, who builds every package: one consistent hand, but that person
  becomes the bottleneck.

## Consequences

- `row.py` has to be generalized so that it no longer depends on GEAK's output files.
- Contributors write their task's tests themselves (an agent may draft them; the contributor must read them).
