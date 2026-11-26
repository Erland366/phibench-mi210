# Experiment log: Φ-Bench MI210 rows

Entries 2026-11-06 to 2026-11-09, copied from the GEAK repository's `references/experiment-log.md`, where this work started.
They use the file names of that time: `evaluation/geak/phibench_<x>.py` is now `src/phibench_mi210/<x>.py`,
`phibench_specs/` is `specs/`, `phibench_templates/` is `src/phibench_mi210/templates/`, and
`training_reports/phibench-mi210-rows-premerge-2026-11-08.md` is `docs/premerge-report-2026-11-08.md`. Commit hashes
in the entries are commits of the GEAK repository. Add new entries at the end.

### 2026-11-06 — Converter from a finished GEAK run to a Φ-Bench row (CPU only)

- **Type:** tooling, no GPU work, nothing published.
- **Added:** `evaluation/geak/phibench_row.py`, its templates and the MoE v2 spec; output in
  `$RESULTS/phibench-rows` (dataset's 8 columns unchanged, MI210 stated in the summary and package).
- **Result:** MoE v2 (`jais3-moe-v2-v032-01`) gives row `mi210-jais3-routed-moe-w8a16`, `ref_speedup` 1.130592; the
  dataset's `verify_package.py` passes. MoE round 2 (1.094x), MoVA v1 and v2 and attention round 2 are refused.
- **Open:** the anchor is provisional (from the Round check, a different harness). Not done: image build, calibration
  with the package's verifier, its self-checks. `scripts/run_task.py` only passes NVIDIA devices.
- **Details:** `evaluation/geak/README.md`, "From a finished run to a Φ-Bench row".

### 2026-11-07 — The five pieces of the MoE v2 result, each scored on its own rows

- **Type:** Round check on one MI210 of node303 (GPU 0), one piece at a time, nothing else on the node. Plan and rule
  frozen before timing: `evaluation/geak/phibench_specs/jais3-moe-w8a16-v032-v2/pieces.json`. Tool:
  `evaluation/geak/phibench_pieces.py`. Results: `$RESULTS/phibench-pieces/jais3-moe-v2` (`scores.json`).
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
  `$RESULTS/phibench-pieces/jais3-moe-v2-single-rows`.
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
  `$RESULTS/phibench-rows`; `verify_package.py` passes and the 8 columns are unchanged. Not uploaded.
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
  kept in `$RESULTS/phibench-rows-provisional-20261108`. Hugging Face not updated.
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

### 2026-11-23 — Apptainer instead of Enroot or Docker: the m256 verifier gives the calibrated result

- **Why:** auh7-1b-gpu-240 (Slurm job 519331, all 8 MI210s) has Docker but no access to its daemon, and the pinned
  Enroot base file stayed on node 303. Apptainer 1.5.3 builds without root (`--fakeroot`).
- **Image:** `vllm/vllm-openai-rocm:v0.32.0` pulled from Docker Hub as a SIF (sha256 `45c95432…186948c`; torch
  2.13.0, vLLM 0.32.0+rocm723, Triton 3.8.0, ROCm 7.2.3). The m256 task image is a hand translation of its
  Dockerfile (`$WORK/phibench-apptainer/defs/`); every Dockerfile check passed. Not yet a tool.
- **Run (plan frozen first in `specs/apptainer-check-2026-11-23.json`), GPU 0, node idle before and after:** noop
  1.0056 reward 0, no hard fail (Enroot: 1.0075); oracle **1.2115**, no hard fail, reward 0.045 (Enroot calibration
  runs 1.1745-1.2051, anchor 1.201437). Both inside their pass marks. About 4 min 50 s per run, as with Enroot. One
  oracle run, so no spread; the anchor is unchanged.
- Evidence: `$WORK/phibench-apptainer/runs/2026-11-23-check/` (console logs, `/logs/verifier`).
- **Open:** a build tool that reads each task's Dockerfile, the other two tasks, and the agent step for a local model.

### 2026-11-23 — A local model as the solver: Qwen3.8-27B served next to the m256 task (first solver run invalid)

- **Plan** (frozen first): `specs/qwen-agent-pilot-2026-11-23.json`. Scripts in
  `$WORK/phibench-apptainer/scripts/` (`serve_qwen.sh`, `score_once.sh`, `run_agent.py`,
  `final_score.sh`).
- **Model server:** Qwen/Qwen3.8-27B (BF16, 27.8B parameters, 55.6 GB) on vLLM 0.32.0 from the pinned base SIF,
  tensor parallel 4 on GPUs 4-7, text only, context 131072. Ready in about 4.5 min; about 56 GB held per GPU; KV cache
  2.27M tokens; one request about 51 tokens/s. The linear-attention layers run (Triton kernels on gfx90a).
- **routed-moe image** built the same way as m256 (`defs/mi210-jais3-routed-moe-w8a16.def`); every Dockerfile check
  passed. Not run yet.
- **Interference, GPU 0, server up and idle on GPUs 4-7, GPUs 0-3 idle before and after:** noop 0.9945 reward 0
  (idle node 1.0056), oracle **1.2155** reward 0.063, no hard fail (idle node 1.2115). Both shifts are inside the
  spread of the Enroot calibration runs (1.1745-1.2051); the baseline was not slower (0.0461 ms against 0.0469 ms).
  One run each. Evidence: `$WORK/phibench-apptainer/runs/2026-11-23-qwen-interference/`.
- **Solver run 1** on m256 (mini-swe-agent 2.4.6, GPU 0, 4 h limit) started 18:03 UTC:
  `$WORK/phibench-apptainer/runs/2026-11-23-qwen-m256-1/`. **Invalid as a score; kept.**
  - The solver ended at 20:36 UTC after 2 h 33 min and 26 model calls: one model request ran past litellm's default
    600 s client timeout. No edit to the working tree, no submission, no read of `/opt/loop/private`, `/tests` or
    the patches. It read the code, measured the baseline with its own script, and was profiling memory bandwidth.
  - Cause of the slowness: generation fell with context length, from about 30 tokens/s at a few thousand tokens to
    about 7 at 35k and 5 at 50k (server log, one request at a time). A reply of a few thousand tokens (thinking
    included) then takes 5 to 16 min.
  - The final score did not run: `apptainer exec --writable` cannot bind `/tests` because the image has no `/tests`
    directory (the Dockerfile checks it is absent). Nothing was measured. `final_score.sh` needs to create `/tests`
    in the sandbox first, as the upstream harness creates it only after the agent ends.
  - The server was stopped by `final_score.sh`; every GPU idle after.

### 2026-11-24 — Qwen solver: decode fixed, run 2 started

- **Decode speed by attention backend** (vLLM 0.32.0, Qwen3.8-27B, TP 4 on GPUs 4-7, one request, 256 tokens):
  `ROCM_ATTN` (the default, used in run 1) 24.1 tok/s at 6k context and 5.1 at 50k; `TRITON_ATTN` 49.8 and **40.8**;
  `ROCM_AITER_UNIFIED_ATTN` does not start on gfx90a. A third request (about 150k tokens) was over the 131k limit and
  refused, my mistake in sizing the prompt. Evidence:
  `$WORK/phibench-apptainer/runs/2026-11-24-decode-backends/`.
- **Amended plan** `specs/qwen-agent-pilot-2026-11-24.json` (approved): `TRITON_ATTN`, 3600 s per model reply,
  and `/tests` created in the sandbox before the final score (checked: the bind works and is read-only).
- **Solver run 2** on m256 started 2026-11-24 08:16 UTC:
  `$WORK/phibench-apptainer/runs/2026-11-24-qwen-m256-2/`.
  - **Final score (node idle): speedup 1.1428, reward 0** (correct, every gate passed, no hard fail; anchor 1.201437,
    so below the reference). Baseline 0.0468 ms, candidate 0.0410 ms. One run.
  - The change: one tile-table entry in `configs.json` (the 256-token gate/up entry `N768`: `num_warps` 4 to 8,
    `num_stages` 2 to 1, new `K_UNROLL` 2). No kernel change.
  - The solver made 1 submission (in-session speedup 1.1138 with the server on GPUs 4-7) and did not finalize; the
    final-score script finalized it, as planned.
  - It ended after 60 min and 56 model calls with 3 format errors in a row, as the prompt reached 128.8k of the
    131k context: replies were cut off. The cause: the chat template sends every earlier thinking block back to the
    model (`preserve_thinking` on by default); thinking was about half of the context.
  - No read of `/opt/loop/private`, `/tests` or the patches.
- **Next:** mini-swe-agent has no context management. The user chose Qwen Code 0.28.4 (auto-compaction built in)
  as the harness for the next runs.

### 2026-11-24 — Qwen Code as the harness: setup and compaction test (no timing)

- **Install:** Qwen Code 0.28.4 (npm) in `$WORK/agents/qwen-code` and Node.js 26.12.1 (official
  build, SHA256 checked) in `$WORK/agents/node`, mounted read-only at `/opt/agent` in the
  task sandbox. Script: `$WORK/phibench-apptainer/scripts/run_qwen_code.sh`. Settings: context
  window 131072, request timeout 1 h, usage statistics, telemetry and update checks off, web tools removed, raw API
  logging on. The server now runs with tool calls on (`--enable-auto-tool-choice --tool-call-parser qwen3_coder`).
- **Setup failures on the way (kept in `runs/2026-11-24-qwen-code-compaction-test{,-2,-3}`):** `--writable` cannot
  create the `/opt/agent` bind target (made in the sandbox first); Apptainer refuses `--env HOME` (Qwen Code's
  `QWEN_HOME` used instead); every API call failed with `Value of "this" must be of DOMException`. Cause: Apptainer
  passes the host's proxy variables through even with `--cleanenv`, and Qwen Code sent the local request through the
  proxy. Fix: the proxy variables are unset inside the container. Without them the container has no route to the
  internet (checked), as the task requires. The mini-swe-agent runs 1 and 2 did have the proxy; neither trajectory
  contains a network command.
- **Compaction, context window lowered on purpose, a prompt that reads large files:** at 32k the summary was made
  (status COMPRESSED) but a single turn of six parallel file reads was larger than the whole window, so the session
  stopped (test 4, an artifact of the small window). At 65k, one file per turn (test 5): compaction ran 5 times, the
  prompt fell from about 33-35k to 20-27k each time, the session finished (19 turns, `success`). Each summary took
  3-6 min (8-14k tokens of thinking). In 0.28.4 the trigger is min(85% of the window, window minus a reserve for the
  summary minus 13k), about 33k for 65k and an estimated 98k for 131k. Qwen Code also sends past thinking back to the
  model; compaction is what bounds it. Evidence: `runs/2026-11-24-qwen-code-compaction-test-5/` (stream, raw API logs).
- **A leftover process held the GPU lock.** `/app/entrypoint.sh` starts `/app/timer.sh` (an endless loop) in the
  background. `final_score.sh` ran with `--contain` (no PID namespace), so after run 2's final score the timer kept
  running on the host with the lock's file handle, and the first start of run 3 exited at once on `flock -n`. Fix:
  `--pid` in `final_score.sh` and `run_qwen_code.sh`, so every process of the container ends with it (checked).
  `score_once.sh` already used `--containall`, which includes `--pid`. Run 2's score is unaffected: the timer only
  writes files under `/app/.timer`.
- **Solver run 3** (Qwen Code, plan `specs/qwen-code-run-2026-11-24.json`) started 2026-11-24 10:14 UTC:
  `$WORK/phibench-apptainer/runs/2026-11-24-qwen-code-m256-3/`.
  - The session used the whole 4 h (stopped by the wall-time limit, exit 124): 484 model messages, 76 shell commands,
    60 file reads, 46 edits or writes. Auto-compaction ran 6 times, at 90k-99k prompt tokens, and the session went on
    after each one.
  - **One submission, in session (server on GPUs 4-7): speedup 1.4382, reward 0.980**, correct, every gate passed.
    The solver ran `submit.sh` as a background shell. It did not finalize; the final-score script did.
  - The change: a new 256-token entry in the decode table `dec_configs.json` (BLOCK_SIZE_M 32, FUSE_SILU 2, g1/g2
    tiles with K_UNROLL 2 and waves_per_eu 2) and a `precompile` function in `moe_w8a16.py` that runs `moe_forward`
    once per token count the tables can reach, so nothing compiles after warm-up. No kernel change.
  - **Final score not counted.** It ran 14:16-14:21 UTC, and GPUs 0, 2 and 3 were busy after it: the user's
    `wm_vendor_mismatch` Wan2.2 generation jobs started on the node at 14:16:51 (they do not take this lock). The
    disturbed run read baseline 0.134 ms (normally 0.047) and speedup 1.705; kept in `final-busy/`, not a result. The
    solver's sandbox is kept, finalized, for a final score on an idle node.
  - **The solver read the in-session grader.** It listed `/opt/loop/private/tests`, read `score_engine.sh`,
    `grade_config.json`, `test.sh` (with the reward formula) and `grade.py`, and searched for the reference speedup
    (found in `/opt/verifier-correctness-manifest.json`, which the Dockerfile puts in the image on purpose). Under
    Apptainer the solver is root in its sandbox, as upstream's solver is root in Docker, so `0700` hides nothing. The
    held-out seeds are only in `tests/`, which is not in the sandbox during the session. No network command.
- **Correction (2026-11-25): run 3 broke the user's GPU split.** The user had split this node: wm_vendor_mismatch on
  GPUs 0-3 (its plan E29, its queue ran 2026-11-24 13:04-15:51 UTC on GPUs 0-3), this project on GPUs 4-7. Run 3 put
  the solver on GPU 0. So from 13:11 UTC the solver's session shared GPU 0 with Wan2.2 generation, and the in-session
  speedup 1.4382 (14:04 UTC) was measured on a shared GPU: not a clean number either. Runs 1 and 2 and the
  interference check ran before that queue started. Rescore of the kept, finalized candidate on GPU 4 once every GPU
  is idle: `specs/qwen-code-run-2026-11-24-rescore.json`, log
  `runs/2026-11-24-qwen-code-m256-3.rescore.log`.
  - **Rescore, 2026-11-25 06:02-06:06 UTC, GPU 4, every GPU idle for 3 min before and idle after: speedup 1.1374,
    reward 0** (correct, every gate passed; baseline 0.0466 ms, candidate 0.0410 ms; anchor 1.201437). This is run 3's
    final score. The in-session 1.4382 was inflated by the shared GPU. Run 2 (mini-swe-agent) scored 1.1428 the same way.

### 2026-11-25 — Upstream Phi-Bench task wro-vllm-v1-priority-request-queue: image, calibration, solver run

- **Task:** upstream Phi-Bench revision `3bcefac`, `tasks/lh/wro-vllm-v1-priority-request-queue` (CPU only, Python;
  the vLLM V1 scheduler's priority queue, an unsorted list with an O(n) scan per pop; reference patch: `heapq`).
  Downloaded unchanged (24 files) to `$WORK/phibench-apptainer/upstream/3bcefac/`.
- **Upstream issues noticed, not changed:** `instruction.md` says matching the reference scores about 1.0 and a correct
  unoptimized submission about 0.5, but `compute_reward.py` gives 0 unless the speedup exceeds `ref_speedup` (1.0 at
  its square). `test.sh` takes one timing sample each of baseline and candidate. The timed workload is 4000 adds and a
  full drain (no removes). The instruction's "red line" says any read of a verifier or scoring file scores the task 0.
- **Image:** `images/wro-vllm-v1-priority-request-queue.sif`, a hand translation of the package's Dockerfile
  (`defs/wro-vllm-v1-priority-request-queue.def`; the overlay, anti-leak and smoke steps copied verbatim). Every
  Dockerfile check passed (torch 2.7.1+cu126, vLLM 0.10.1.1, scope file wired into site-packages, one clean baseline
  commit, anti-leak asserts, `import vllm`). Two failed builds kept (`images/*.failed-apt.log`, `*.failed-tmp.log`):
  apt does not read `HTTPS_PROXY` (given a proxy config file for the build only, written with command echo off and
  deleted at the end; checked absent from the image), and fakeroot cannot write the host's `/tmp` in a build (apt's
  temporary files moved inside the image for the build).
- **Calibration** (plan `specs/calibration-wro-vllm-pq-2026-11-25.json`, approved): noop, negative, oracle x5, solve,
  each with the node quiet (no process above 10% CPU over 5 s) before and after. Waiting for the user's
  triton_ai_compiler jobs to finish. Script `scripts/calibrate_cpu.sh`, log `runs/2026-11-25-wro-pq-calibration.log`.
  - **Result (15:09-15:29 UTC, node quiet before and after every run):** noop speedup 0.9948, reward 0; negative fails
    correctness, reward 0; oracle 323.37, 322.07, 328.05, 316.27, 319.51 (baseline about 946-959 ms, oracle about
    2.92-3.02 ms); solve 321.37, inside the oracle range. Every pass mark met. **Local anchor: 322.071786** (median;
    upstream 312.0 on Intel Sapphire Rapids). Evidence: `runs/2026-11-25-wro-pq-calibration/`.
  - The anchor is in a local copy of the package (`local/wro-vllm-v1-priority-request-queue-epyc9334`, only the
    manifest differs) and in `images/wro-vllm-v1-priority-request-queue-epyc9334.sif` (the calibrated image with
    `/opt/verifier-correctness-manifest.json` replaced; test.sh reads the anchor only there). A first version of the
    manifest named the node and internal paths, visible to the solver; replaced before any run.
- **Solver run** (Qwen Code, plan `specs/qwen-code-wro-pq-2026-11-25.json`): `runs/2026-11-25-qwen-code-wro-pq-1/`.
  - **Session (15:37-16:47 UTC, 70 of 90 min):** the solver stopped by itself after 3 of 16 submissions and
    finalized on submission 2. In-session speedups: 114.12 (heap with an eager removal map), 327.17 (heap with lazy
    removal), 325.99 (submission 2 plus `__slots__`). It tested further ideas locally (sorting the tail on drain,
    lazy heapify) and found them flat or slower. 66 turns, largest prompt 98.7k tokens, no compaction [wrong: 1 compaction, see the correction of 2026-11-26].
  - **Final score (16:49-16:50 UTC, server stopped, node quiet 3 checks before and after):** speedup 317.51 against
    the anchor 322.07 (vs_oracle 0.986), every gate passed, **reward 0**. The finalized code matches the reference
    (a heapq heap); 317.51 and the in-session 327.17 both sit inside the oracle's own range (316.27-328.05), so the
    solver tied the reference and did not beat it.
  - **Audit:** no read of `/opt/loop/private`, `score_engine.sh`, `sanitize_feedback.py`, `/tests` or the patches;
    its only `/opt` commands were `submit.sh` calls; no network command; it ran its own banned-token scan (clean).
  - Evidence: `final/verifier/reward.json`, `logs/loop/` (state.jsonl, finalize.json), `stream.jsonl`,
    `logs/agent/openai/`.
- **Verdict and time taken (wro-pq):** the setup works end to end; the solver tied the reference, reward 0. Time per
  step: download about 1 min (12:18 UTC, 24 files); image build 13 min 54 s (after two failed attempts, 12:24 and
  12:25, fixed within minutes); 12:40-15:09 between build and calibration, mostly waiting for another user's CPU jobs to finish; calibration
  8 runs in 19 min (15:09-15:29); anchored image rebuild about 2 min; solver session 70 min (15:37-16:47); final
  score 3 min including the quiet checks (16:47-16:50). About 4 h 30 min from download to result, about
  half of it waiting for a quiet node.

### 2026-11-25: Phi-Bench upstream row wro-llamacpp-simd-q5k (away mode, CONTRACT.md revision 1)

- **Task:** upstream 3bcefac `tasks/lh/wro-llamacpp-simd-q5k`: make llama.cpp's Q5_K x Q8_K dot product
  (`ggml/src/ggml-cpu/arch/x86/quants.c`) fast; the start is a scalar version, the reference is llama.cpp's own AVX2
  code (public, so likely known to the model). CPU only, 16 submissions, 90 min. Plans
  `specs/calibration-wro-llamacpp-q5k-2026-11-25.json`, `specs/qwen-code-wro-llamacpp-q5k-2026-11-25.json`.
- **Download:** 18:23-18:33 UTC (10 min, 1125 files, the vendored ggml tree), unchanged.
- **Image:** `defs/wro-llamacpp-simd-q5k.def`, checks verbatim; the cmake pre-warm's `/tmp` paths moved to
  `/var/tmp`. Build 1 (18:37) failed in the upstream anti-leak check: `find / -xdev` saw the build host's `/tmp`,
  which held another user's pytest output with a file named `negative.patch` (nothing of this task). Build 2 with
  an empty directory bound at `/tmp`: passed every check in 1 min 36 s (18:38-18:39). gcc 12.2.0, cmake 3.25.1.
- **Calibration** (18:42-19:05 UTC, 23 min, node quiet before and after every run; `runs/2026-11-25-wro-q5k-calibration/`):
  noop 1.0019 reward 0; negative fails the checksum (1152 != 2304), reward 0; oracle 2.588867, 2.602713, 2.595194,
  2.604461, 2.600214 (oracle about 234 ms, scalar baseline about 607 ms, cv under 1%); solve 2.591161. Every pass mark
  met. **Local anchor 2.600214**, against upstream's 11.53 on Intel Sapphire Rapids: a factor 4.4 lower here with
  the same image and compiler, so on this CPU the scalar start is already close to the AVX2 reference. Reward 0
  below 2.60x, 1.0 at 6.76x. Local copy `local/wro-llamacpp-simd-q5k-epyc9334` (anchor line of tests/test.sh and the
  manifest changed), image `images/wro-llamacpp-simd-q5k-epyc9334.sif` (manifest only).
- **Solver run:** `runs/2026-11-25-qwen-code-wro-q5k-1/`, job log `runs/2026-11-25-row-q5k-solver.job.log`.
  - **Session (19:19-20:49 UTC, the full 90 min, stopped by the time limit):** **no submission at all**. It read the
    ggml headers, wrote its own scratch harness in `/tmp` (link stubs for `ggml_cpu_init`, its own compile line) and spent
    the session on compile errors in that harness and in its AVX2 code; at the end it was still fixing a wrong
    intrinsic name (`_mm256_extracti128_si128` for `_mm256_extracti128_si256`). 95 model calls, 107 tool calls, largest
    prompt 98.8k tokens, no compaction [wrong: 2 compactions, see the correction of 2026-11-26]; five single calls of 5-10 min each (long thinking at about 90k context).
  - **Final score (20:49-20:52 UTC, the working tree as left, no finalize since nothing was submitted):** correct,
    speedup 1.3808 against the anchor 2.6002, **reward 0**.
  - **Audit:** no read of `/opt/loop` (not even `submit.sh`), `/tests` or the patches; no network command.
- **Verdict and time taken (q5k):** the setup is sound (calibration passed, final score ran); the solver failed the
  task by never submitting: it used the whole budget on a private test harness instead of the submit loop. Times:
  download 10 min, image 1 failed build + 1 min 36 s, calibration 23 min, anchored image and the row 2-4 images
  10 min, server start 2 min, solver 90 min, final score 3.5 min: 2 h 29 min from download to result (18:23-20:52).

### 2026-11-25: overnight queue, rows 2-4 (wro-causal-delivery-vclock, wro-memory-accounting-sim, wro-secure-agg-shamir-committee)

- **Images** (built 19:07-19:17 UTC, before the q5k solver session): defs generated by `scripts/gen_def.py` (RUN
  bodies verbatim, each run by `/bin/sh`; TMPDIR inside the image and a deleted apt proxy config added; an empty
  directory bound at `/tmp`). Every Dockerfile check passed; builds 5 min 14 s, 3 min 59 s, 25 s; no proxy text in any log.
  Plans `specs/calibration-<row>-2026-11-25.json`, `specs/qwen-code-<row>-2026-11-25.json`.
- **Invalid start (20:52-20:59 UTC):** two copies of the queue ran: at 19:20 I replaced the first queue, but the kill
  reached only its nohup wrapper. Both started row 2's calibration at 20:52; the verifier runs stayed one at a time
  under the lock, but the second copy stopped on `noop-1 exists` and went on to row 3. Both copies stopped at 20:59; the
  outputs are kept as `runs/2026-11-25-causal-delivery-vclock-calibration.collided/` (noop 0.9946 and negative 0, not
  counted) and `*.collided.job.log`. Restarted 21:00 as one queue, `scripts/overnight_queue.sh`, which now holds a lock
  so a second copy cannot start.
- **Row 2 calibration (wro-causal-delivery-vclock, 21:00-21:22 UTC, 22 min, node quiet before and after every run;
  `runs/2026-11-25-causal-delivery-vclock-calibration/`):** noop 1.0107 reward 0; negative reward 0 (correctness
  incomplete); oracle 177.068082, 175.908397, 177.094904, 175.955891, 175.443802; solve 171.072 (within 10% of the
  median). **Local anchor 175.955891** (upstream 187.500717 on Sapphire Rapids, 6% higher). Local copy and image
  `local/wro-causal-delivery-vclock-coupled-epyc9334`, `images/wro-causal-delivery-vclock-coupled-epyc9334.sif`.
- **Row 2 solver, first attempt (21:24 UTC): did not start.** Node.js exited 127 within a second: `libatomic.so.1`
  is missing from the `python:3.11-slim` image (the earlier images had it). Kept as
  `runs/2026-11-25-qwen-code-causal-delivery-vclock-1.agent-not-started/`; the queue was stopped before row 3 (same
  base image). Fix: `agents/node-v26.12.1-rpath`, a copy of the same Node binary with rpath `/opt/agent/lib-bookworm`,
  holding Debian bookworm's `libatomic.so.1` (from the q5k image, libatomic1 12.2.0-14+deb12u1); only Node loads it, no
  `LD_LIBRARY_PATH`. Checked in the row 2 sandbox: `node --version` v26.12.1, Qwen Code 0.28.4. `scripts/run_qwen_code.sh`
  now uses it for every row. Restarted 21:29: row 2 solver stage, then rows 3-4.
- **Row 2 solver (`runs/2026-11-25-qwen-code-causal-delivery-vclock-1/`, 21:29-22:59 UTC, the full 90 min, stopped by the
  time limit, 5 of 16 submissions, auto-finalized on submission 4):** in-session speedups 376.27 (fails the in-session
  `implausible_vs_oracle` gate), 331.55, 327.73, 347.70, 430.74 (`implausible_vs_oracle` again). 77 model calls, 82 tool
  calls, largest prompt 98.4k tokens, no compaction [wrong: 2 compactions, see the correction of 2026-11-26].
  - **Final score (22:59-23:02 UTC):** correct, every gate passed, speedup **360.87** against the anchor 175.96
    (vs_oracle 2.05), **reward 0.139**: the first positive reward of these runs. The change: a heap of ready messages
    per sender with an exact slow path for messages outside the fast case (`causal/buffer.py` +282 lines,
    `causal/channel.py` +57/-36).
  - **Audit:** no read of `/opt/loop` beyond `submit.sh`, `/tests` or the patches; no network command. An independent
    review (fresh agent, raw evidence only, 23:05 UTC) found the score legitimate: a general algorithm with no
    workload constants; 120k fuzz cases matched the baseline (outputs, pending count, clock), on inputs the verifier
    never uses too; the only mismatch is duplicate `msg_id`s, where the baseline itself behaves oddly and the
    instruction does not say ids are unique. The final tree is submission 4's code (same tree hash); 360.87 against its
    in-session 347.70 is timing noise on the same code.
  - **Upstream defects the review found (recorded, not changed):** (1) the in-session grader
    (`opt/loop/private/tests/compute_reward.py:49`) fails any speedup above 2.0x the reference (here above 351.9x),
    while the final verifier (`tests/compute_reward.py:66`) fails only above 10x; submissions 1 (376.3x) and 5 (430.7x)
    were correct and were rejected only for being fast. (2) `sanitize_feedback.py:65-66` reports that rejection to the
    solver as an early crash, so the solver went looking for crashes that did not exist. The kept score is lower than
    the solver's best correct code. Under the in-session rule even the final 360.87 would have been rejected. (3) The
    verifier's correctness set (`workload.py:128-147`) has 26 well-formed cases and never exercises the slow path.
- **Verdict and time taken (row 2):** setup sound after the libatomic fix; the solver beat the reference about 2x.
  Times: download 15 s, image 5 min 14 s, calibration 22 min (after a 7 min collided start), first solver attempt lost
  to libatomic (about 5 min to fix), server 2 min, solver 90 min, final score 3 min: about 2 h 10 min of row time
  (20:52-23:02) plus the build at 19:07.
- **Row 3 calibration (wro-memory-accounting-sim, 23:02-23:24 UTC, 22 min, node quiet before and after every run;
  `runs/2026-11-25-memory-accounting-sim-calibration/`):** noop 0.9985 reward 0; negative reward 0 (correctness 0);
  oracle 1081.68, 1122.23, 1130.41, 1161.85, 1107.20; solve 1133.05. **Local anchor 1122.234676** (upstream 1344.398352,
  20% higher). Solver `runs/2026-11-25-qwen-code-memory-accounting-sim-1/`.
- **Row 3 solver (23:26-00:56 UTC, the full 90 min, stopped by the time limit, 2 of 16 submissions,
  auto-finalized on submission 2):** in-session 41.77 and 42.34, both correct. 52 model calls, 53 tool calls,
  largest prompt 97.8k tokens, no compaction [wrong: 2 compactions, see the correction of 2026-11-26]. Change: `memsim/accountant.py` +199 lines, `memsim/scheduler.py` +17 lines changed.
  - **Final score (00:56-00:59 UTC):** correct, speedup **40.62** against the anchor 1122.23 (vs_oracle 0.036),
    **reward 0**.
  - **Audit:** no grader, `/tests`, patch or `/logs` reads; no network command.
- **Verdict and time taken (row 3):** setup sound; the solver made a correct 41x gain but stayed 27x short of the
  reference. Times: download 15 s, image 3 min 59 s, calibration 22 min, server 2 min, solver 90 min, final score
  3 min: 1 h 57 min of row time (23:02-00:59).
- **Row 4 calibration (wro-secure-agg-shamir-committee, 00:59-01:19 UTC, 20 min, node quiet before and after every run;
  `runs/2026-11-25-secure-agg-shamir-committee-calibration/`):** noop 0.9960 reward 0; negative reward 0 (correctness
  failed); oracle 50.271535, 50.246781, 50.590443, 49.749164, 51.067651; solve 50.387. **Local anchor 50.271535**
  (upstream 57.658178, 15% higher).
- **Row 4 solver (`runs/2026-11-25-qwen-code-secure-agg-shamir-committee-1/`, 01:22-02:52 UTC, the full 90 min, stopped by
  the time limit, 2 of 16 submissions):** in-session 132.79 and 130.59, about 2.6x the reference. Both passed every
  correctness gate (`submissions/{1,2}/verifier_state.json`: scope, import origin, correctness, benchmark all true) and
  both were rejected only by the in-session `implausible_vs_oracle` gate (above 2x the reference), which the solver is
  told is a crash. It spent the rest of the session making its code robust against a crash that did not exist. 49
  model calls, 51 tool calls, largest prompt 92.5k tokens, no compaction [wrong: 1 compaction, see the correction of 2026-11-26].
  - **Final score (02:52-02:55 UTC):** the auto-finalize found no accepted submission; `submit.sh --finalize` resets
    the tree to the baseline and plants only an accepted winner, so it planted nothing and the final score measured
    the unchanged baseline: speedup 1.0105, **reward 0**. The solver's code is kept only in
    `logs/loop/submissions/{1,2}/candidate.diff` (and its git trees).
  - **Audit:** no grader, `/tests`, patch or `/logs` reads; no network command; no git reset by the solver.
- **Verdict and time taken (row 4):** setup sound; the solver very likely solved the task (correct, about 2.6x the
  reference in-session), but the upstream harness threw the result away: the 2x gate (found in row 2) here rejected
  every submission, and finalize then restored the baseline. Times: download 15 s, image 25 s, calibration 20 min,
  server 2 min, solver 90 min, final score 3 min: 1 h 56 min of row time (00:59-02:55).

### 2026-11-26: Correction, the five CPU-task sessions did compact

- **What was wrong:** the entries above for wro-pq, q5k, causal-delivery, memory-accounting and secure-agg said
  "no compaction". I had checked the largest prompt (below the expected trigger of about 98k) and searched the stream
  for "compress"; neither finds a compaction. The trace converter (`src/phibench_mi210/traces.py`, which detects the
  summarizer call by its instruction) counted compactions in every run, and the raw model calls confirm it.
- **Measured** (`logs/agent/openai/`: the summarizer call's prompt tokens, then the next agent call's):
  - wro-pq: 1, 98,684 → 22,905 (call 36)
  - q5k: 2, 98,830 → 17,263 and 96,891 → 32,193
  - causal-delivery: 2, 85,314 → 38,318 and 98,383 → 27,786
  - memory-accounting: 2, 97,776 → 30,525 and 95,880 → 23,754
  - secure-agg: 1 compaction made of 2 summarizer calls (02:14 and 02:18 UTC): the first, in the usual form, left
    81,219 tokens; Qwen Code then called the summarizer again with the instruction as the system message and no
    tools; the next agent call had 28,769 tokens.
  - m256 (Qwen Code): 6, as recorded.
  So the "largest prompt" figures above are the prompts at which compaction started.
- **Also seen:** wro-pq's last call (16:47 UTC) and the 65k compaction test's last call are a "managed memory extraction
  subagent" side call of Qwen Code, with its own system prompt; it is not part of the solver's conversation.
- **Fixed:** the two slides of the 26 November deck that repeated the claim. The scores and labels are unchanged.

