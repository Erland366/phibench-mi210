# Inputs: what this repository reads from a GEAK run

This repository does not run GEAK or the Round check. Both live in the GEAK repository (`evaluation/geak/`), which needs
the `./k2` tooling and the node setup. Here we only read the folders they write. This page is the contract: if
the GEAK repository changes these files, update this page and the code together.

## A GEAK run folder (`--run-dir`)

`outcome.json`, from the GEAK runner:

| Field | Used for |
|---|---|
| `models_outside_cap` | must be empty, or the row is refused |
| `status`, `missing_workflow_outputs`, `seconds`, `transcript_models` | copied into the row's provenance |

`chat/` is recorded by path only. Nothing from it is copied: it describes the solution.

## A Round check folder (`--round-check`)

The independent Round check of one candidate (`round_check_jais3.py prepare` then `measure`).

`prepare.json`:

| Field | Used for |
|---|---|
| `editable` | must equal the spec's editable files |
| `source.kind`, `source.candidate_dir` | where the candidate files are; only `directory` is supported |
| `candidate.<file>.sha256` | the candidate files must be exactly the bytes measured |
| `frozen_task.path`, `frozen_task.sha256` | the frozen task files in `task/` must be exactly these |
| `stacked_start.path` (optional) | the later starting files; become `baseline2.patch` (not for pieces) |

`verdict.json`:

| Field | Used for |
|---|---|
| `summary.verdict` = `pass`, `summary.correctness` = `PASS`, `summary.control_identical` = true, no `summary.cases_failed` | otherwise the row is refused |
| `summary.score_speedup` | the provisional anchor; must be at least 1.1 |
| `rows.<row>`: `weight`, `start_ms`, `candidate_ms`, `verdict`, `rounds_ms.A`, `rounds_ms.A2`, `vs_stacked_start` | per-piece scoring (`pieces.py`) |
| `runtime` | copied into provenance |

`task/`: the frozen task files (the served kernel's `baseline_*` files, the runner `scripts/task_runner.py`, the
routing recording). They become the package's `environment/repo/` and grader.

## Pieces

A piece is one change from a larger GEAK patch, measured alone (see the README). The GEAK repository builds it and runs its
Round check (`python -m evaluation.geak.phibench_pieces build --plan PLAN --results RESULTS --out OUT`, then
`round_check_jais3.py measure --work-dir OUT/<piece>/work`, one at a time on an idle node). This repository scores
`OUT/<piece>/work/verdict.json` with the plan's frozen rule (`python -m phibench_mi210.pieces --plan PLAN --out OUT`),
and a spec with `piece: {plan, id}` turns a qualifying piece into a row (`--round-check OUT/<piece>/work`).

The plans live here in `specs/`; the GEAK repository's build step reads the same plan file.
