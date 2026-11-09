# Experiment log: Φ-Bench MI210 rows

Entries 2026-11-06 to 2026-11-09, copied from `deploy_bbq` `references/experiment-log.md`, where this work started.
They use the file names of that time: `evaluation/geak/phibench_<x>.py` is now `src/phibench_mi210/<x>.py`,
`phibench_specs/` is `specs/`, `phibench_templates/` is `src/phibench_mi210/templates/`, and
`training_reports/phibench-mi210-rows-premerge-2026-11-08.md` is `docs/premerge-report-2026-11-08.md`. Commit hashes
in the entries are `deploy_bbq` commits. Add new entries at the end.

### 2026-11-06 — Converter from a finished GEAK run to a Φ-Bench row (CPU only)

- **Type:** tooling, no GPU work, nothing published.
- **Added:** `evaluation/geak/phibench_row.py`, its templates and the MoE v2 spec; output in
  `/data/erland/results/phibench-rows` (dataset's 8 columns unchanged, MI210 stated in the summary and package).
- **Result:** MoE v2 (`jais3-moe-v2-v032-01`) gives row `mi210-jais3-routed-moe-w8a16`, `ref_speedup` 1.130592; the
  dataset's `verify_package.py` passes. MoE round 2 (1.094x), MoVA v1 and v2 and attention round 2 are refused.
- **Open:** the anchor is provisional (from the Round check, a different harness). Not done: image build, calibration
  with the package's verifier, its self-checks. `scripts/run_task.py` only passes NVIDIA devices.
- **Details:** `evaluation/geak/README.md`, "From a finished run to a Φ-Bench row".

### 2026-11-07 — The five pieces of the MoE v2 result, each scored on its own rows

- **Type:** Round check on one MI210 of node303 (GPU 0), one piece at a time, nothing else on the node. Plan and rule
  frozen before timing: `evaluation/geak/phibench_specs/jais3-moe-w8a16-v032-v2/pieces.json`. Tool:
  `evaluation/geak/phibench_pieces.py`. Results: `/data/erland/results/phibench-pieces/jais3-moe-v2` (`scores.json`).
- **Result: 0 of 5 qualify.** Target-row scores: SiLU fusion 1.035x, `COL_LOOP` 1.080x, decode down 1.091x, large-batch
  tiles 1.020x, 512-token decode route 1.029x, all under the 1.1x minimum. Whole-layer: 1.028x, 1.061x, 1.021x, 1.013x,
  1.008x. Correctness passed on all five. Slower rows outside the targets: SiLU m2, `COL_LOOP` m4, large-batch tiles m1.
- **Learned:** two single rows clear 1.1x on the frozen pieces: m1024 in `COL_LOOP` (1.248x) and m256 in decode down
  (1.187x). Both tile entries set no `COL_LOOP` or `DOWN_T`, so they may work on the served code alone; not yet tested.
  m1 is not trustworthy: its served time ranged 0.049-0.055 ms across the five runs, and identical m1 code differed by
  over 5% inside one run, so SiLU's m1 1.196x means little.
- **Not established:** kernel-only timing was not measured; these rows were picked after seeing the results, so they
  need a fresh frozen measurement before any row is built.

### 2026-11-07 — Two single table entries of MoE v2, alone on the served code

- **Type:** Round check, one MI210 of node303, one at a time. Plan frozen before timing:
  `evaluation/geak/phibench_specs/jais3-moe-w8a16-v032-v2/pieces_single_rows.json`. Results:
  `/data/erland/results/phibench-pieces/jais3-moe-v2-single-rows`.
- **Result: neither qualifies under the frozen rule, only because of small rows.** The 1024 tile entry (14 lines
  changed) gives m1024 1.258x; the new 256 decode entry (21 lines added) gives m256 1.175x. Both repeat the five-piece
  run, and correctness passed. Flagged slower: m2 and m4 (0.976x, 0.963x) with the 1024 entry, m4 (0.978x) with the 256
  entry.
- **Open question for the user:** those rows run byte-identical code and tables, and their outputs did not change. The
  v1 MoE identical-files run (`aa-round-check-20261029-rule2`) also flagged m1 slower (0.965x) and m8 faster (1.022x), so
  this band can flag unchanged small rows. The 0.963x on m4 is a little beyond that run's largest gap. Not changing the
  rule without the user's decision.

### 2026-11-07 — Two one-row Φ-Bench tasks from MoE v2 table entries

- **Rule change (user decision):** the "no slower row" test now covers only the rows a patch changes; other slower rows
  are reported. Written in `pieces_single_rows_rerun.json` before the rerun. The earlier plans keep their old rule.
- **Rerun result: both qualify.** m1024 entry 1.241x, m256 entry 1.216x, correctness passed. The slower flags moved
  between runs (m1 and m8; m2), which fits false flags on unchanged code.
- **Rows:** `mi210-jais3-moe-m1024` (`ref_speedup` 1.241187) and `mi210-jais3-moe-m256` (1.215727), in
  `/data/erland/results/phibench-rows`; `verify_package.py` passes and the 8 columns are unchanged. Not uploaded.
- **Package check on one MI210 (not calibration):** the shipped runner with the new row-only metric, one ABBA block:
  m1024 1.253x, m256 1.188x; correctness passed with the reference applied.
- **Still open:** image build, the packaged verifier's self-checks, and anchor calibration (at least 5 runs). The m256
  check landed 2.3% under its anchor, and a reference that only ties its anchor scores 0.

### 2026-11-07 — Two one-row Φ-Bench tasks published to Hugging Face

- `Erland/phibench-mi210-rows` commit `0907f81f` adds `mi210-jais3-moe-m1024` and `mi210-jais3-moe-m256`, the catalog
  files and a new README status section. The combined row is unchanged. A logged-out download matched all 138 files.
- **Visibility:** the dataset was private at upload on 2026-11-06 and was found public on 2026-11-07, with no commit
  recording the change. The user chose to keep it public and upload. `provenance/` is not uploaded.
- The rows are still drafts: anchors provisional, image and verifier self-checks not done, license label unconfirmed.

### 2026-11-08 — Φ-Bench one-entry rows: Enroot image, calibration, and a no-insight search

- **Image:** `phibench_enroot.py` builds each task image with Enroot from the package's own Dockerfile on the
  pinned v0.32.0 base (SHA256 checked); all Dockerfile checks passed (no reference patch in the image, editable-file
  hashes, torch/vLLM versions). Each verifier run uses a hard-link copy of the image and checks the image unchanged.
  Two bugs found and fixed on the way: Enroot passes only `--env`-named variables (a first run silently ran mode
  `candidate`; discarded), and deleting a copy must not chmod its files (it changed 4 file modes of the m256 image,
  since restored; no effect on results, the container runs as root).
- **Calibration (rules frozen in `phibench_specs/calibration-2026-11-08.json`), 303 GPU 0, one run at a time:**
  m256 no-change 1.0075x reward 0, negative fails correctness, oracle 1.1745/1.1956/1.2014/1.2045/1.2051 →
  **anchor 1.201437** (was 1.215727 provisional). m1024 no-change 1.0002x, negative fails correctness, oracle
  1.2504/1.2491/1.2464/1.2521/1.2441 → **anchor 1.249117** (was 1.241187). One run takes about 4 min 47 s.
- **Search (rules frozen in `phibench_specs/table-search-m*.json`), 305, 96 trials each, both tables:** m256 best
  1.010x on its row (the unchanged code), nothing above 1.1x; verifier on the finalist 0.998x, `no_speedup`.
  m1024: copying the existing 8192 entry to 1024 gives exactly the reference entry; 10 trials beat 1.1x; verifier
  on the finalist **1.2516x, above the anchor (reward about 0.009): too easy by the frozen rule.**
- Packages rebuilt with `phibench_row.py --calibration` (only the anchor files changed); the provisional ones are
  kept in `/data/erland/results/phibench-rows-provisional-20261108`. Hugging Face not updated.
- **Published (user decision, 2026-11-08):** `Erland/phibench-mi210-rows` commit `1149a15a` carries the calibrated
  m256 and m1024 packages, the catalog and a README marking m1024 as too easy (kept for now, user's call). A
  logged-out download matched all 138 files.

### 2026-11-08 — Combined row calibration attempt (invalid) and the pre-merge check

- **Invalid:** every run of `mi210-jais3-routed-moe-w8a16` (12:21-13:01) and the three `solve.sh` runs (12:56-13:11)
  shared node 303, GPU 0 included, with Jais-3 Screen servers started at 10:22-10:24 by another session. Not used;
  kept in `phibench-calibration/invalid-shared-node-20261108`. Our queue had checked the GPUs only at its start.
  `phibench_enroot.py score` now requires every GPU idle before and after each run. Screen numbers on 303 between
  12:21 and 13:11 may have been slowed by these runs.
- **Pre-merge check:** `phibench_premerge.py` (automatic checks) plus `phibench_specs/premerge-review.json` (decisions
  and research checks); report in `training_reports/phibench-mi210-rows-premerge-2026-11-08.md`. Found: the Hugging
  Face copy lost every executable bit (upload API), so `verify_package.py` fails on a download; and our
  `environment/runtime/*.sh` were not stored executable (fixed, contents unchanged). No row is ready to merge.

### 2026-11-09 — Combined row calibrated on an idle node

- Waited for 15 idle minutes on 303, then held `K2_GPU_LOCK` for the whole queue (07:49-08:36); no run found the node
  busy. `mi210-jais3-routed-moe-w8a16`: no-change 0.9988 reward 0, negative fails correctness, oracle
  1.1334/1.1329/1.1320/1.1325/1.1322 → **anchor 1.132514** (provisional was 1.130592; the invalid shared-node attempt
  gave 1.137155). `baseline2` 1.1104.
- Oracle-agent (`solve.sh`) check: combined 1.1322, m256 1.2136 (anchor 1.201437), m1024 1.2403 (anchor 1.249117):
  within about 1%, the spread between runs and days. The reference alone can land above or below its own anchor.
- Package rebuilt with `--calibration` (anchor files only). All automatic pre-merge checks pass on all three rows;
  open: m1024 decision, owners, license, a docker buildx build, the upstream runner on AMD, and a `git push` upload.
- **Published (user decision, 2026-11-09):** `Erland/phibench-mi210-rows` commit `323e515`, pushed with git: the
  combined row's calibrated package, 21 scripts stored 100755 as upstream stores them, and a README with all three
  calibrations. A fresh clone matches the local rows and passes every automatic pre-merge check; an HTTP download
  never sets executable bits, so fetch with `git clone`.

### 2026-11-09 — User decisions for the merge, and AMD support in the Φ-Bench runner

- Keep m1024 (marked too easy). Owners and license deferred. No Docker build on our nodes (no docker group; only
  Apptainer, which cannot build from a Dockerfile): left to whoever merges.
- Φ-Bench's `scripts/run_task.py` (vendored at revision 3bcefac) passed only NVIDIA GPUs. Our version adds an `amd`
  mode: `--device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --group-add video`, GPU subset through
  `HIP_VISIBLE_DEVICES`, chosen by `auto` when task.toml `gpu_types` are AMD (or none and the host has only /dev/kfd).
  NVIDIA hosts and tasks unchanged. Checked by unit tests and `--dry-run` on the three tasks (no Docker here); the
  patch applies to upstream with `git apply`. Shipped in the rows as `scripts/run_task.py` and `run_task-amd.patch`.
