# Φ-Bench MI210 rows

Turn GEAK kernel-agent results on AMD Instinct MI210 into task rows for the
[Φ-Bench dataset](https://huggingface.co/datasets/faibench-Frontier-Infra-Bench/faibench_Frontier_Infra_Bench)
(arXiv 2609.10226), calibrate them on the GPU, and check them before they are merged into the dataset.

The rows keep the dataset's 8 columns unchanged (`bench, task_dir, infra_name, reward_class, gpus, max_submissions,
ref_speedup, summary`). There is no hardware column, so MI210 is stated in each summary, instruction, `task.toml` and
anchor caveat. The draft rows are published at
[`Erland/phibench-mi210-rows`](https://huggingface.co/datasets/Erland/phibench-mi210-rows) (fetch with `git clone`;
an HTTP download drops the executable bits the dataset's checker requires).

## Rows

| Row | Scored on | `ref_speedup` | Reference patch | Note |
|---|---|---|---|---|
| `mi210-jais3-routed-moe-w8a16` | whole MoE layer, 1-8192 tokens | 1.132514 | kernel code and tile tables | |
| `mi210-jais3-moe-m256` | 256 tokens only | 1.201437 | one new decode-table entry | survived a no-insight search |
| `mi210-jais3-moe-m1024` | 1024 tokens only | 1.249117 | one table entry | too easy: kept by decision |

All three are calibrated with their own verifier (median of 5 runs, alone on an MI210 node) and pass every
automatic pre-merge check. Not merged yet: see [the pre-merge report](docs/premerge-report-2026-11-08.md).

## Layout

```
src/phibench_mi210/
  row.py           a passed GEAK run (or a qualifying piece) -> a task package and a catalog row
  pieces.py        score the pieces of a GEAK result, each on the rows it changes
  enroot.py        build a task image with Enroot from its Dockerfile; run its verifier in a fresh copy
  table_search.py  a solver with no kernel insight: search one tile-table entry (is a one-entry task too easy?)
  premerge.py      every automatic check before a merge, plus the review items, in one report
  templates/       package templates; vendored/ holds byte copies of upstream files (NOTICE.md);
                   run_task.py is upstream's runner with AMD GPU support, run_task-amd.patch the change
specs/             per task: spec.json and instruction.md (the solver's prompt); frozen plans; premerge-review.json
tests/             CPU-only tests
docs/              inputs.md (what we read from GEAK), the pre-merge report, experiment log, troubleshooting
```

## Setup

```bash
uv sync
uv run pytest
```

Python 3.11 or newer, standard library only (plus `pytest` for the tests). GPU steps need an MI210 node with Enroot
and the pinned `vllm/vllm-openai-rocm:v0.32.0` image file.

## From a GEAK run to a merged row

1. **In `deploy_bbq`:** run GEAK and its independent Round check. For a large result, split it into pieces, build
   and measure each one there (see [docs/inputs.md](docs/inputs.md)).
2. **Score pieces** (if any): `uv run python -m phibench_mi210.pieces --plan specs/<task>/<plan>.json --out OUT`.
3. **Write the task:** `specs/<task>/spec.json` (names, summary, scrub list, a deliberately wrong edit) and
   `specs/<task>/instruction.md`, the solver's whole prompt. Nothing is copied from GEAK's chat: it describes the
   solution.
4. **Build the row:** `uv run python -m phibench_mi210.row --spec specs/<task> --run-dir RUN --round-check RC --out ROWS`.
   It refuses, and writes nothing, on any failed check.
5. **Build the image:** `uv run python -m phibench_mi210.enroot build --task-dir ROWS/tasks/lh/<task> --base BASE.sqsh
   --base-sha256 HEX --data DATA`.
6. **Calibrate** on one MI210, under a frozen plan written first (`specs/calibration-*.json`): `score --mode noop`,
   `--mode negative`, `--mode oracle --repeat 5`, `--mode solve`. Each run needs every GPU on the node idle, before
   and after.
7. **Set the anchor:** step 4 again with `--force --calibration DIR` (the median of the oracle runs; it refuses
   unless the self-checks pass and the package is the one calibrated).
8. **Is it too easy?** For a one-entry task, `table_search.py` under a frozen plan (`specs/table-search-*.json`),
   then the finalist through the verifier.
9. **Check:** `uv run python -m phibench_mi210.premerge --rows ROWS --provenance ROWS/provenance`. Update
   `specs/premerge-review.json` as decisions are made.
10. **Publish** the draft rows with `git push` to the Hugging Face dataset repository (never the upload API, which
    drops executable bits), then run step 9 on a fresh clone.

## Relation to deploy_bbq

This work began in `deploy_bbq` (branch `enroot`, `evaluation/geak/phibench_*`); the files here carry their history.
GEAK, the Round check and the piece build step stay there. The old copies in `deploy_bbq` are removed once this
repository is accepted.
