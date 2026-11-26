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
  roots.py         the named roots ($RESULTS, $WORK) that specs and run sheets write instead of machine paths
  traces.py        agent runs on a task -> trace records in conversation/chats/, labeled by the task's verifier
                   or the independent check; session-requests, convert, upload (see below)
  templates/       package templates; vendored/ holds byte copies of upstream files (NOTICE.md);
                   run_task.py is upstream's runner with AMD GPU support, run_task-amd.patch the change
specs/             per task: spec.json and instruction.md (the solver's prompt); frozen plans; premerge-review.json;
                   run sheets for traces (traces-*.json, and examples/run-sheet-example.json)
tests/             CPU-only tests
docs/              sop.md (turning any agent work into a task), inputs.md (what we read from GEAK), adr/ (decisions),
                   the pre-merge report, experiment log, troubleshooting
CONTEXT.md         the words we use (task, row, trace, anchor, ...)
```

To turn your own work with a coding agent into a task, follow [docs/sop.md](docs/sop.md). The route below is the one
for GEAK runs.

## Setup

```bash
uv sync
uv run pytest
cp specs/local-roots.example.json specs/local-roots.json   # then set this machine's folders in it
```

Specs and run sheets never write a machine path: they write `$RESULTS/...` (the results folder on the MI210 node
that runs GEAK and holds the rows) and `$WORK/...` (the work folder on the machine that serves the open models and
runs the CPU tasks). Each machine maps the names to its own folders in `specs/local-roots.json`, which git ignores
(or in the file named by `PHIBENCH_ROOTS`). A name that is not set on the machine is refused.

Python 3.11 or newer, standard library only (plus `pytest` for the tests). GPU steps need an MI210 node with Enroot
and the pinned `vllm/vllm-openai-rocm:v0.32.0` image file.

## From a GEAK run to a merged row

1. **In the GEAK repository** (separate from this one): run GEAK and its independent Round check. For a large result, split it into pieces, build
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

## From an agent run to a trace record

A trace is the recorded conversation of one run of an agent on a task, kept with its label for training. The
records live in the private Hugging Face dataset
[`Erland/rsi-traces`](https://huggingface.co/datasets/Erland/rsi-traces); SOP step 6 ([docs/sop.md](docs/sop.md))
says when to make one, and [docs/trace-records.md](docs/trace-records.md) describes the format.

1. **Find the slice** (your own Claude Code session only): list the requests you typed, with the uuid to start at.

   ```bash
   uv run python -m phibench_mi210.traces session-requests --session ~/.claude/projects/<folder>/<session>.jsonl
   ```

2. **Write the run sheet:** a JSON file in `specs/` with one entry per run. Copy the matching entry from
   [specs/examples/run-sheet-example.json](specs/examples/run-sheet-example.json): your own Claude Code session, an
   open model's run (Qwen Code, mini-swe-agent) or a GEAK run. The sheet names the run folder and the task package
   (`task_dir`); the task's id, size, kind, anchor and scoring hardware are read from the package, never typed again.
3. **Convert:** one record per run. Every conversation goes to `conversation/chats/` of its record, one file per
   chat, in the same message shape for every source; the label comes from the task's verifier or the independent
   check, never from the agent.

   ```bash
   uv run python -m phibench_mi210.traces convert --sheet specs/<sheet>.json --out <dataset folder>
   ```

   It refuses a run, and writes nothing for it, when the sheet is incomplete, disagrees with the run or the task
   package, or a file holds a secret; it never overwrites a stored record.
4. **Upload:** checks every file on Hugging Face against the local one, and that the dataset is still private.

   ```bash
   uv run --with huggingface_hub python -m phibench_mi210.traces upload --out <dataset folder> --message "<what changed>"
   ```

`remessage --out <dataset folder>` rebuilds every record's chats from its raw files after the chat format changes,
and `index --out <dataset folder>` rebuilds `index.jsonl`. A Codex session cannot be read yet, and Claude Code saves
its thinking without text, so a Claude chat has no readable reasoning.

## Where GEAK runs

GEAK, its Round check and the piece build step run in the GEAK repository, separate from this one; this
repository only reads the folders they write ([docs/inputs.md](docs/inputs.md)). This work began there, and the files
here carry their history.
