# A task may target any hardware, stated in the task and anchored on it

Status: accepted (2026-11-19)

The group's strength is the range of hardware it works on (AMD data-center GPUs, NVIDIA, edge devices, a coming
mixed-hardware data center), so tasks are not limited to MI210. Φ-Bench has no hardware column and we keep its 8
columns unchanged, so each task states its hardware in its summary, `instruction.md` and `task.toml`, and a speed
task's anchor is measured on that hardware only.

## Consequences

- A speed task can only be scored on matching hardware, and its row says so; anchors from different hardware are
  not comparable.
