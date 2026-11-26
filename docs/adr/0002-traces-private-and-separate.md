# Traces are kept private and separate from the published tasks

Status: accepted (2026-11-19); "successful only" amended by ADR 0008, which keeps failed attempts too

The ML team wants the traces as training data, and only successful ones, so that each has a known answer (meeting
of 2026-11-19). A trace describes the solution, and the solver must never see the solution (AGENTS.md rule 5), so
traces are stored apart from the public task packages, private to the group, each linked to its task. Nothing from
a trace is copied into a task's `instruction.md` or `environment/`. Traces go to a private Hugging Face dataset
(for example `Erland/rsi-traces`) with the group's members added, because contributors work on other machines;
each upload is still the dataset owner's call.

## Considered options

- Publish traces next to each task on Hugging Face: simpler, but it hands every solver the solution and settles,
  by default, whether Claude outputs may be used to train other models, which the group has not decided.
