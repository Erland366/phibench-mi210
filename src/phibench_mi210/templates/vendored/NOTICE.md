Files in this directory are byte copies from the Φ-Bench dataset,
`faibench-Frontier-Infra-Bench/faibench_Frontier_Infra_Bench` at revision
`3bcefac184b72ebf4d9cbf12fd543fd0d9aa232d`, Apache License 2.0 (`LICENSE`):

- `submit.sh`, `score_engine.sh`, `sanitize_feedback.py`, `entrypoint.sh`, `timer.sh`, `pre_artifacts.sh`,
  `solve.sh`: from `tasks/lh/wro-w8a16-groupdequant-matmul-sol`. `phibench_row.py` substitutes the task names in
  `pre_artifacts.sh` and `solve.sh` when it generates a package.
- `verify_package.py`: `scripts/verify_package.py`.
- `run_task.py`: `scripts/run_task.py`. Our AMD version is `../run_task.py`; `../run_task-amd.patch` is the
  difference, for upstream.
- `upstream_task_catalog.jsonl`: `task_catalog.jsonl`, the reference for the 8 columns and the existing task names.

Do not edit them here. To move to a newer dataset revision, copy the files again and update this note.
