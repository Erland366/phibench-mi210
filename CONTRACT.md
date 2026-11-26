# Contract (away mode) · revision 1, agreed 2026-11-25 18:45 UTC by the user ("Yes, go")

- Objective while away: run upstream Phi-Bench CPU rows one after another with the Qwen Code solver; for each row
  record whether it is good (setup sound, calibration passes, solver score and audit) and how long each step took.
- Level: 2 (follow the plan). The per-row protocol and the row order below are the plan; picking the next row from
  the list is not a new experiment.
- Per-row protocol (the same as wro-pq, `specs/calibration-wro-vllm-pq-2026-11-25.json` and
  `specs/qwen-code-wro-pq-2026-11-25.json`): download the package unchanged from revision 3bcefac; translate
  `environment/Dockerfile` to an Apptainer def (checks verbatim, deviations written in the def and the log); build;
  freeze the two plan files for the row; calibrate on the quiet node (noop, negative, oracle x5, solve); local anchor =
  oracle median in a local package copy and a derived image, with no node name or internal path in solver-visible
  files; Qwen Code run (instruction.md verbatim, the task's own wall limit and submissions, model server on GPUs 4-7);
  final score with `QUIET=cpu scripts/final_score.sh`; audit (grader reads, banned tokens, network); log entry with
  verdict and step times.
- Row order: 1 wro-llamacpp-simd-q5k (in progress); 2 wro-causal-delivery-vclock-coupled;
  3 wro-memory-accounting-sim-coupled; 4 wro-secure-agg-shamir-committee-coupled; then kfc CPU rows in catalog order.
  Only rows with gpus = 0: the GPU rows' images are CUDA-only and do not run on MI210.
- GPUs: auh7-1b-gpu-240 GPUs 4-7 for the model server only, under `/tmp/phibench-auh7-240-gpus.lock`. Never GPUs 0-3
  (wm_vendor_mismatch), never another user's processes.
- Stop when: 2026-11-26 05:00 UTC; or two unexplained failures across rows; or a must-ask item. A row whose setup
  fails twice for the same reason is recorded as not runnable here and skipped. A busy node means wait, not stop;
  no timing run starts on a busy node.
- Must ask first: changing the model, harness, prompt, sampling or limits; editing upstream files other than the
  anchor; any GPU row; commit, push or upload; deleting any run output.
- At stop: leave no server running unless a solver session is in flight; the hand-back goes at the top of STATE.md.
- Amendments: none.
