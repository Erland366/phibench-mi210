# Troubleshooting

Symptom, cause and fix of problems met in this project. The first section is copied from `deploy_bbq`
`references/troubleshooting.md` (2026-11-08); file names there are as in `docs/experiment-log.md`.
## Enroot task images for Φ-Bench (2026-11-08)

- **Symptom:** the verifier ran mode `candidate` although `KERNELBENCH_VERIFY_MODE` was set. **Cause:** `enroot start`
  passes into the container only the variables named with `--env`. **Fix:** name every variable (`phibench_enroot.py`).
- **Symptom:** "the built image changed during the run" (`/.lock`, `/etc/passwd`, git objects, file modes). **Cause:**
  a hard-link copy shares inodes, so Enroot's start-up writes, git's timestamp refresh and a `chmod -R` before deleting
  the copy all reach the built image. **Fix:** real copies of the written paths (`WRITABLE`), and chmod directories
  only when deleting.
- **Symptom:** `verify_package.py` fails on a Hugging Face download ("solution/solve.sh 没有可执行位"). **Cause:**
  `HfApi.upload_folder` stores every file 100644. **Fix:** upload with `git push`, which keeps 100755.
- **Symptom:** timings far off (baseline 40% slower). **Cause:** another job started on the node mid-queue; the
  queue checked the GPUs only once. **Fix:** `score` checks every GPU idle before and after each run.
