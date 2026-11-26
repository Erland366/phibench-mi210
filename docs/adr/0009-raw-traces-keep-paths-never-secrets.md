# Raw traces keep internal paths but never secrets

Status: accepted (2026-11-26)

A trace's raw files are kept byte for byte so that they show exactly what the model saw (ADR 0008), and they are
full of internal paths and node names. The trace dataset is private to the group, so those stay. Secrets (tokens,
passwords, API keys, a proxy address with credentials) are never stored: a raw file that holds one has only that
secret blanked out, and the record's `scrub` field says which file and what was changed. If traces are ever made
public, they get a full scrub of paths, nodes and accounts first, as a separate step.

## Considered options

- Scrub paths and node names from every trace before storing it: safer to share, but the raw model calls would no
  longer be what the model saw, which is why they are kept.

## Consequences

- The task package is held to the stricter rule as before: `instruction.md` and `environment/` name nothing internal
  (checked by `row.py` and `premerge.py`), because the solver sees them and the task dataset may become public.
