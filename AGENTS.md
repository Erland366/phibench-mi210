# AGENTS.md: working rules for this repository

Read [README.md](README.md) first, then [docs/inputs.md](docs/inputs.md). The global agent contract still applies;
these rules are specific to this project.

1. **Research records are the point.** Every GPU result goes into `docs/experiment-log.md` (date, what, numbers,
   where the evidence is). Failed and invalid runs are kept and recorded, never deleted or silently redone.
2. **Freeze before running.** Write the plan (`specs/calibration-*.json`, `specs/table-search-*.json`,
   `specs/<task>/pieces*.json`) before the first GPU run it governs. A rule changed after seeing results needs the
   user's approval and a fresh run under the new rule.
3. **A node to itself.** Timing runs need every GPU on the node idle before and after each run (`enroot.py score`
   checks this). Take the node's GPU lock (`K2_GPU_LOCK`, `/tmp/deploy-bbq-node303-gpus.lock` on 303) with
   `flock -x -n` for a queue, and never start GPU work on another job's GPUs. Ask before using GPUs.
4. **The 8 columns never change.** Hardware goes in the summary, instruction, `task.toml` and caveat.
5. **The solver sees `instruction.md` and `environment/`.** Nothing there may name the reference's origin (GEAK,
   rounds) or any internal path, node or account; `row.py` and `premerge.py` check this. Do not copy from GEAK chat.
6. **Publishing is outward-facing.** The Hugging Face dataset is public. Upload only when the user asks, with
   `git push`, and check a fresh clone with `premerge.py` afterwards.
7. **Upstream files stay byte copies.** `templates/vendored/` is not edited; changes to an upstream file live next
   to it with a patch (as `templates/run_task.py` and `run_task-amd.patch`), pinned by a test.
8. **Plain language** in docs and messages; no unexplained terms.
