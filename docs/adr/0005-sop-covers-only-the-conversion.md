# The procedure covers only turning a trace into a task, and tasks go to a new dataset

Status: accepted (2026-11-19)

The procedure (SOP) the group asked for is about one thing: turning a contributor's agent conversation into a task
and a stored trace. It ends when the task's image builds, the task passes its checks (a speed task with its
measured anchor), and the trace slice is scrubbed and stored: the image is part of the benchmark, so building and
checking it is part of the conversion. It is written for any kind of system work; our fast FP8 MoE kernel is only
a worked example in an appendix, not the shape of the procedure.
Evaluating models on the tasks (such as a Qwen 27B baseline) and training on the traces are separate work. The
group's tasks go to a new Hugging Face dataset (for example `Erland/rsi-system-tasks`), private until each task is
reviewed and made public by the owner's decision, because `Erland/phibench-mi210-rows` is public and named after
one GPU.

## Consequences

- Every task must run with `run_task.py`, so that evaluating a model later needs no change to the task.
- `Erland/phibench-mi210-rows` keeps its three MI210 rows; whether they move to the new dataset is decided later.
