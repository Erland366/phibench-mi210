# A task is built from public code only

Status: accepted (2026-11-19)

The task dataset is meant to become public, so a task's starting code and every file in its image must come from
public sources. Work done in a private repository or on a private branch cannot become a task, though its trace may
still go to the private trace store (ADR 0002).

## Considered options

- Allow private-code tasks in the private dataset and decide about publishing later: more tasks now, but every such
  task would have to be separated out again before the dataset goes public.
