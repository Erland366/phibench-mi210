# Report: Φ-Bench MI210 rows, state before merging into the dataset

**Date:** 2026-11-08
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
python -m evaluation.geak.phibench_premerge --rows /data/erland/results/phibench-rows \
    --provenance /data/erland/results/phibench-rows/provenance
```

The review items (owner approval, license, research checks) are in
`evaluation/geak/phibench_specs/premerge-review.json`; update their status there. A row is READY only when no automatic
check fails and no blocking item is open. The last full output is
`/data/erland/results/phibench-rows-premerge-check-20261108.md`.

## The rows

| Row | Scored on | `ref_speedup` | Anchor | Reference patch | Ready |
|---|---|---|---|---|---|
| `mi210-jais3-routed-moe-w8a16` | whole layer, 1-8192 tokens | 1.130592 | provisional | kernel code and tables | no |
| `mi210-jais3-moe-m256` | 256 tokens only | 1.201437 | calibrated | one new decode-table entry (21 lines) | no |
| `mi210-jais3-moe-m1024` | 1024 tokens only | 1.249117 | calibrated | one table entry (14 lines) | no |

Local packages: `/data/erland/results/phibench-rows`. Public draft copy: `Erland/phibench-mi210-rows` on Hugging Face
(commit `1149a15a`, public since 2026-11-07 by the user's choice).

## Automatic checks (phibench_premerge.py)

All three rows pass: the 8 columns, order and types of the published catalog, and no task_dir clash; the three catalog
files agree; `ref_speedup.txt`, `task.toml` and the catalog hold the same anchor; the dataset's own `verify_package.py`;
no internal path, node or account name; MI210 named in the summary and instruction, and nothing solver-visible names
the reference's origin; `*.patch` and `solution/` kept out of the image; scripts executable as upstream stores them.

| Row | Fails | Warns |
|---|---|---|
| combined | `calibrated` (anchor provisional) | anchor copied into the image |
| m256 | none | anchor copied into the image |
| m1024 | none | anchor copied into the image |

The **Hugging Face copy fails** `verify_package.py` and `executable` for all three rows: the upload API stored every file
as 100644, so `solve.sh`, `pre_artifacts.sh`, `tests/test.sh` and the runtime scripts lose their executable bit (upstream
stores them 100755). The next upload must go through `git push`, which keeps the mode. The local packages are correct
(the runtime scripts were fixed 2026-11-08; contents unchanged).

## Calibration (package verifier, Enroot image from each package's Dockerfile)

Rules frozen in `evaluation/geak/phibench_specs/calibration-2026-11-08.json` and `calibration-combined-2026-11-08.json`.
Node 303, GPU 0, one run at a time; one run takes about 4 min 47 s.

| Row | No-change | Negative | Oracle runs (speedup) | Anchor |
|---|---|---|---|---|
| m256 | 1.0075, reward 0 | fails correctness | 1.1745, 1.1956, 1.2014, 1.2045, 1.2051 | 1.201437 |
| m1024 | 1.0002, reward 0 | fails correctness | 1.2504, 1.2491, 1.2464, 1.2521, 1.2441 | 1.249117 |
| combined | not valid | not valid | not valid (see below) | 1.130592, provisional |

**Invalid attempt on the combined row.** All its runs (12:21-13:01) and the three oracle-agent runs (12:56-13:11) shared
node 303, including GPU 0, with Jais-3 Screen servers that another session started at 10:22-10:24. The frozen rule
requires the node to itself, so none of them is used; they are kept in
`/data/erland/results/phibench-calibration/invalid-shared-node-20261108`. The m256 and m1024 runs ended by 10:08, before
the servers started. The cause on our side: the queue checked the GPUs only once, at the start. `phibench_enroot.py
score` now refuses to start unless every GPU on the node is idle, and discards a run that finds the node busy afterwards.
Screen numbers measured on 303 between 12:21 and 13:11 may have been slowed by these runs.

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

Blocking:
1. Calibrate the combined row on an idle node (noop, negative, 5 oracle runs), then rebuild it with `--calibration`.
2. Run the oracle-agent (`solution/solve.sh`) check on all three rows on an idle node.
3. Decide m1024 (too easy).
4. Owners' approval (Omar and the other Φ-Bench owners) and the license label.
5. Build each image once with `docker buildx` (only Enroot has been used; no docker group on the nodes).
6. The upstream runner (`scripts/run_task.py`) passes only NVIDIA GPUs; the owners must accept a change or a manual path.
7. Re-upload with `git push` so the Hugging Face copy keeps executable bits.

Not blocking, for the owners: whether the image may carry the anchor (upstream's README says no, its own example lh
task does), and whether MI210 stated in the summary, instruction, `task.toml` and caveat is enough without a hardware
column (upstream anchors are on NVIDIA H20).

## Evidence

- Code: `evaluation/geak/phibench_row.py`, `phibench_pieces.py`, `phibench_enroot.py`, `phibench_table_search.py`,
  `phibench_premerge.py`; tests in `evaluation/tests/test_geak_phibench_*.py`.
- Calibration runs: `/data/erland/results/phibench-calibration/<task>/` (each run's `/logs/verifier`, `<mode>.json`).
- Search: `/data/erland/results/phibench-search/` (trials, summaries, finalists, finalist verifier runs).
- Earlier packages: `/data/erland/results/phibench-rows-provisional-20261108`, `phibench-rows-before-*`.
- Chronology: `references/experiment-log.md`, entries 2026-11-06 to 2026-11-08.
