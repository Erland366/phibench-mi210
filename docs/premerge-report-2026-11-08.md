# Report: Φ-Bench MI210 rows, state before merging into the dataset

**Date:** 2026-11-08 (updated 2026-11-09)
**Author:** Erland (with Claude Code)
**Status:** In Progress (not ready to merge)

## Objective

Three draft rows for the [Φ-Bench dataset](https://huggingface.co/datasets/faibench-Frontier-Infra-Bench/faibench_Frontier_Infra_Bench)
(arXiv 2609.10226) were built from GEAK runs on AMD MI210, kept separate with the dataset's 8 columns unchanged so they
can be merged later. This report lists every check a merge needs, its state per row, and where the evidence is. It is
the reference for that later merge and for any paper that uses these rows.

## How to check again

```bash
# every automatic check plus the review items, on the local rows or on a download of the Hugging Face copy
uv run python -m phibench_mi210.premerge --rows /data/erland/results/phibench-rows \
    --provenance /data/erland/results/phibench-rows/provenance
```

The review items (owner approval, license, research checks) are in
`specs/premerge-review.json`; update their status there. A row is READY only when no automatic
check fails and no blocking item is open. The last full output is
`/data/erland/results/phibench-rows-premerge-check-20261109.md`.

## The rows

| Row | Scored on | `ref_speedup` | Anchor | Reference patch | Ready |
|---|---|---|---|---|---|
| `mi210-jais3-routed-moe-w8a16` | whole layer, 1-8192 tokens | 1.132514 | calibrated | kernel code and tables | no |
| `mi210-jais3-moe-m256` | 256 tokens only | 1.201437 | calibrated | one new decode-table entry (21 lines) | no |
| `mi210-jais3-moe-m1024` | 1024 tokens only | 1.249117 | calibrated | one table entry (14 lines) | no |

Local packages: `/data/erland/results/phibench-rows`. Public draft copy: `Erland/phibench-mi210-rows` on Hugging Face
(commit `323e515`, 2026-11-09, pushed with git so executable bits are kept; public since 2026-11-07 by the user's
choice). Fetch it with `git clone`: an HTTP download never sets executable bits, and `verify_package.py` then fails.

## Automatic checks (premerge.py)

All three rows pass: the 8 columns, order and types of the published catalog, and no task_dir clash; the three catalog
files agree; `ref_speedup.txt`, `task.toml` and the catalog hold the same anchor; the dataset's own `verify_package.py`;
no internal path, node or account name; MI210 named in the summary and instruction, and nothing solver-visible names
the reference's origin; `*.patch` and `solution/` kept out of the image; scripts executable as upstream stores them.

All three rows pass every automatic check (2026-11-09); each warns that the anchor is copied into the image.

The Hugging Face copy, fetched with `git clone`, matches the local rows file for file and passes the same checks
(2026-11-09). Its first uploads (through the upload API) had stored every file 100644; it was re-pushed with git.

## Calibration (package verifier, Enroot image from each package's Dockerfile)

Rules frozen in `specs/calibration-2026-11-08.json` and `calibration-combined-2026-11-08.json`.
Node 303, GPU 0, one run at a time; one run takes about 4 min 47 s.

| Row | No-change | Negative | Oracle runs (speedup) | Anchor |
|---|---|---|---|---|
| m256 | 1.0075, reward 0 | fails correctness | 1.1745, 1.1956, 1.2014, 1.2045, 1.2051 | 1.201437 |
| m1024 | 1.0002, reward 0 | fails correctness | 1.2504, 1.2491, 1.2464, 1.2521, 1.2441 | 1.249117 |
| combined | 0.9988, reward 0 | fails correctness | 1.1334, 1.1329, 1.1320, 1.1325, 1.1322 (2026-11-09) | 1.132514 |

Oracle-agent check (`solution/solve.sh` lands the reference, candidate scoring), 2026-11-09: combined 1.1322, m256
1.2136, m1024 1.2403, each within about 1% of its anchor (run-to-run and day-to-day spread). The combined row's
`baseline2` (the later GEAK start) scored 1.1104.

**Invalid first attempt on the combined row (2026-11-08).** All its runs (12:21-13:01) and the three oracle-agent runs (12:56-13:11) shared
node 303, including GPU 0, with Jais-3 Screen servers that another session started at 10:22-10:24. The frozen rule
requires the node to itself, so none of them is used; they are kept in
`/data/erland/results/phibench-calibration/invalid-shared-node-20261108`. The m256 and m1024 runs ended by 10:08, before
the servers started. The cause on our side: the queue checked the GPUs only once, at the start. `enroot.py
score` now refuses to start unless every GPU on the node is idle, and discards a run that finds the node busy afterwards.
Screen numbers measured on 303 between 12:21 and 13:11 may have been slowed by these runs. The second attempt
(2026-11-09 07:49-08:36) waited for 15 idle minutes and held the node's GPU lock (`K2_GPU_LOCK`) throughout.

## Research checks

- **No-insight table search** (`phibench_table_search.py`, plans frozen in `table-search-m*.json`; 96 tries, 60 minutes,
  both tables): **m1024 is too easy.** Its reference entry is an exact copy of the existing 8192 entry; the finalist
  scored 1.2516 with the verifier, above the anchor. The user kept the row for now. **m256 survived**: nothing above
  1.1x on its row, finalist 0.998. Not applicable to the combined row (whole layer; the table-only piece of the same
  GEAK result reached only 1.02x, 2026-11-07).
- **Real solver run:** none on any row.
- **One-size tasks:** correctness covers every token count, but only the named size is timed, so a solution can make
  other sizes slower without losing points.

## Open before a merge

Blocking, none of them doable on our nodes:
1. Owners' approval (Omar and the other Φ-Bench owners) and the license label: deferred by the user (2026-11-09).
2. One `docker buildx` build of each image, by whoever merges: our nodes have no Docker access, only Apptainer (which
   cannot build from a Dockerfile). The images were built with Enroot from the same Dockerfiles.

Decided 2026-11-09: keep m1024 (README marks it too easy); support AMD in the upstream runner. `scripts/run_task.py`
now passes `/dev/kfd` and `/dev/dri` (ROCm options, GPU subset through `HIP_VISIBLE_DEVICES`) when a task's
`gpu_types` are AMD, with NVIDIA behaviour unchanged; `scripts/run_task-amd.patch` is the change for upstream (applies
to their revision 3bcefac). Checked with `--dry-run` and unit tests only, since Docker is not available here; the
owners must accept it.

Done since 2026-11-08: the combined row's calibration, the oracle-agent check on all three rows, and a `git push`
upload that keeps executable bits.

Not blocking, for the owners: whether the image may carry the anchor (upstream's README says no, its own example lh
task does), and whether MI210 stated in the summary, instruction, `task.toml` and caveat is enough without a hardware
column (upstream anchors are on NVIDIA H20).

## Evidence

- Code: `src/phibench_mi210/` (`row.py`, `pieces.py`, `enroot.py`, `table_search.py`, `premerge.py`,
  `templates/run_task.py`, the AMD runner); tests in `tests/`.
- Calibration runs: `/data/erland/results/phibench-calibration/<task>/` (each run's `/logs/verifier`, `<mode>.json`).
- Search: `/data/erland/results/phibench-search/` (trials, summaries, finalists, finalist verifier runs).
- Earlier packages: `/data/erland/results/phibench-rows-provisional-20261108`, `phibench-rows-before-*`.
- Chronology: `docs/experiment-log.md`.
