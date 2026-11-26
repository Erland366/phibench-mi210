# Troubleshooting

Symptom, cause and fix of problems met in this project. The first section is copied from the GEAK
repository's `references/troubleshooting.md` (2026-11-08); file names there are as in `docs/experiment-log.md`.
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

## Apptainer task images (2026-11-23)

- **Symptom:** the Dockerfile's "no reference patch" check fails in `%post`, listing `oracle.patch` files under
  `/tmp/pytest-of-*`. **Cause:** Apptainer binds the host's `/tmp` into the build, and with `APPTAINER_TMPDIR` on the
  same filesystem `find / -xdev` does not skip it. **Fix:** `APPTAINER_TMPDIR` on another filesystem (`/dev/shm`).
- **Do not use `--rocm`:** it binds the host's ROCm libraries (7.0.0 on auh7-1b-gpu-240) into an image built on
  7.2.3. Bind `/dev/kfd` and `/dev/dri` only, as the Dockerfile's `docker run` line passes them.
- `torch.cuda.get_device_name(0)` is empty in the v0.32.0 image (no AMD name table in it); the verifier checks the
  architecture (`gfx90a`), which is correct.
- Image files show as owned by the running user, so the solver can read `/opt/loop/private` (0700). Upstream's agent
  runs as root in Docker, so it can read it there too.
