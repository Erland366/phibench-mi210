# How to turn your agent work into a benchmark task

This procedure turns systems work that you finished with a coding agent (Claude Code, Codex or a similar agent) into a Φ-Bench task and a set of traces. The **task** is a package with its image, which another model can be given and scored on. A **trace** is the recorded conversation of one run on the task, starting with your own, and it is kept privately as training data.

Any systems work that a program can check qualifies, such as a kernel made faster, a serving bug fixed or a feature added. Each step ends with an example from the task `mi210-jais3-moe-m256`, a speed task on AMD Instinct MI210 in the Hugging Face dataset `Erland/phibench-mi210-rows`, whose source is a GEAK agent run.

## What you need

- Finished work in a public repository, with the commit before your change and the commit after it.
- A check that tells a correct change from a wrong one, such as tests that fail before the change and pass after it, or a timing that shows the speedup.
- The stored conversation in which you and the agent did the work.
- Docker with `buildx`, or on a machine without Docker our Enroot builder (`enroot.py build`) or an Apptainer image translated from the Dockerfile.
- For a speed task, the hardware that the task will state, with no other jobs on it while you measure.
- This repository set up with `uv sync`, and `specs/local-roots.json` naming this machine's folders (README, Setup), for storing traces in step 6.

## Step 1. Pick the work and decide its kind

Pick work that was accepted (merged, deployed or kept), that a program can check in minutes, and that would make sense as a request to a capable engineer who is not shown the solution. Skip work that only changes numbers someone could find by trying values, and work that needs more machines than a task can request.

| **Kind** | **Scored on** | **Example** |
|---|---|---|
| **Speed task** (`performance`) | the speedup over the starting code, relative to the anchor | a kernel, a scheduler setting, a serving path |
| **Pass/fail task** (`implementation`) | whether the tests pass | a bug fix, a missing feature, a port to new hardware |

Also choose its size for the `bench` column, which is `kfc` for one function, `lh` for a change across several files and `e2e` for a whole system.

**Example.** The work was one new entry in the decode tile table of the Mixture-of-Experts (MoE) kernel that a vLLM v0.32 server runs on MI210. Measured alone, it made batches of 256 tokens about 1.2× faster with correct results. It is a speed task of size `lh`, because the solver may change the kernel and both of its tile tables.

**Files.** [`task_catalog.jsonl`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/task_catalog.jsonl), the line for `mi210-jais3-moe-m256`.

## Step 2. Fill in the task form

| **Field** | **What to write** |
|---|---|
| `id` | a short name, with the hardware first if the task depends on it |
| `kind`, `bench` | the kind and size from step 1 |
| `repo`, `base_commit`, `solution_commit` | the public repository, the commit before the change and the commit after it |
| `editable` | the files the solver may change |
| `check` | the command that tests or times the work |
| `hardware` | the device and the number of GPUs |
| `trace` | the session files, and where the slice starts and ends |
| `summary` | two or three sentences on what the code is, what must be done, how it is scored and on what hardware |

**Example.** The editable files are `moe_w8a16.py`, `configs.json` and `dec_configs.json`, the hardware is one MI210, and the check is the runner `scripts/task_runner.py`. Because the source is a GEAK run, the run's frozen starting files took the place of the two commits.

**Files.** [`task.toml`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/task.toml).

## Step 3. Write the instruction and the tests yourself

- `instruction.md` is everything the solver is told. It states the problem, the editable files, the rules, the hardware and the scoring. It must not hint at the solution or name where the task came from, or any internal path, node, account or URL. Do not paste from your conversation with the agent, because that text describes the solution.
- The tests in `tests/` decide the score. For a pass/fail task they fail on the starting code and pass on the reference solution. For a speed task they check correctness first and then time the solver's code against the starting code.
- An agent may draft both, but you read every line, because you know what a correct result means for this work.

**Example.** The instruction describes the layer, the three editable files, the math that must not change and the runner's rules, and says that only 256 tokens are scored. It does not mention the table entry. The tests check correctness over all batch sizes (error within 1.02× of the starting code's) before timing 256 tokens.

**Files.** [`instruction.md`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/instruction.md), [`tests/test.sh`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/tests/test.sh), [`tests/grade.py`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/tests/grade.py) and [`tests/grade_config.json`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/tests/grade_config.json).

## Step 4. Build the package and its image

The package follows the Φ-Bench layout.

```
<id>/
  task.toml          id, kind, image, hardware, editable files, scoring
  instruction.md     the solver's whole prompt
  environment/       Dockerfile, starting code (repo/), runtime and submit scripts
  tests/             the verifier, and for a speed task ref_speedup.txt
  solution/solve.sh  applies the reference solution
  oracle.patch       the reference solution
  negative.patch     a wrong edit made on purpose
```

Build the image from the package folder with `docker buildx build --load -f environment/Dockerfile -t <image> .` The image holds the starting code and never the reference solution.

**Example.** The image starts from `vllm/vllm-openai-rocm:v0.32.0` and adds the starting code and the scripts. The reference solution adds a 21-line table entry and is not in the image. It was built with Enroot, because our MI210 nodes have no Docker.

**Files.** [`environment/Dockerfile`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/environment/Dockerfile), [`environment/repo/moe_w8a16.py`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/environment/repo/moe_w8a16.py) and [`environment/repo/dec_configs.json`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/environment/repo/dec_configs.json).

## Step 5. Run the checks

A task counts only when all of these pass. Write down what you will run and which result passes before the first run, and keep every result, including runs you discard.

- **Every task.** The image builds and holds no part of the reference solution. The tests reject the wrong edit and every failed attempt from your trace that ran but gave wrong results. The reference solution passes when `solution/solve.sh` applies it. `instruction.md` and `environment/` name nothing internal.
- **Pass/fail task.** The tests fail on the starting code and pass with the reference solution.
- **Speed task.** The starting code passes the correctness check and scores 0. The reference solution is at least **1.1×** faster, as the median of **5 runs** made one at a time with nothing else on the hardware. That median is the anchor (`ref_speedup`), and it holds only for the hardware the task states. Write it in the anchor file (`tests/ref_speedup.txt`) and in `task.toml`'s `[primary_metric]`, where `ref_speedup` is the same value, `ref_speedup_source` names the anchor file and `calibrated_on` names the device it was measured on. When you measure it again on other hardware, update both, because a package whose two anchors differ is refused when its traces are stored.
- **Table or config solutions (optional).** Run the too-easy search (`table_search.py`). If a search that only tries values finds the solution, the task is too easy.

**Example.** The unchanged code scored 0 (1.0075×), and the wrong edit, which drops the router weight, failed correctness. Five reference runs measured 1.1745 to 1.2051×, so the anchor is their median, 1.201437. The too-easy search tried 96 table entries and its best reached 1.010×, so the task is not too easy.

**Files.** [`oracle.patch`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/oracle.patch), [`negative.patch`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/negative.patch), [`solution/solve.sh`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/solution/solve.sh), [`tests/ref_speedup.txt`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/tests/ref_speedup.txt) and [`tests/ref_speedup.caveat.md`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/tests/ref_speedup.caveat.md).

## Step 6. Store the trace of every run

Every run on the task becomes one record, starting with your own session and including every later run of another agent or model.

- **Find your session.** Claude Code keeps each session in `~/.claude/projects/<folder>/<session>.jsonl`, with its helper agents under `<session>/subagents/`. Codex keeps each session in `~/.codex/sessions/<year>/<month>/<day>/rollout-*.jsonl`. GEAK keeps one Claude Code conversation per agent inside its run folder.
- **Cut the slice.** A session usually holds several pieces of work, so keep only the part from your request for this work to its end, with the failed attempts inside it. `python -m phibench_mi210.traces session-requests --session <file>` lists every request you typed with its uuid. Put the uuid of your request for this work and the uuid of your next request in the run sheet, and the converter keeps the lines from the first up to the second, without it, with the helper agents that started inside them. Your first request in the slice is stored as the record's prompt, and the record notes that it is not the task's instruction.
- **Label the run.** Your own run is successful, because its result became the reference solution. A later run is labeled by the task's verifier, successful when its final reward is above 0 and unsuccessful when it is 0. A GEAK run is labeled by the independent check of its result. A run that was not a fair attempt, such as a setup error or a busy machine during timing, is invalid. The agent's own report never sets the label.
- **Leave the raw files as they are.** The converter keeps every model call of an open model as sent, with its reasoning, and the harness's own log; the session files of Claude and Codex as they are, thinking included; and the final patch and each submission's diff and score. Do not edit or strip them first.
- **Secrets are refused.** No tokens, passwords, API keys or proxy addresses with credentials may be stored. The converter refuses a run whose files hold one; blank the secret in the original and note it under `scrub` in the run sheet. Internal paths and node names may stay, because the trace dataset is private.
- **Write the run sheet, then convert.** Describe every run in a run sheet (below), then run `python -m phibench_mi210.traces convert --sheet SHEET.json --out DATASET`. The converter writes one record per run, and puts every conversation in `conversation/chats/` of the record, one file per conversation, in the same message shape, whatever agent produced it. A Claude Code session or a Qwen run has one chat, and a GEAK run has one chat per agent, so the chat is in the same place and reads the same way for every source. The records go to the private dataset `Erland/rsi-traces`, never into the task package.
- **Upload.** `uv run --with huggingface_hub python -m phibench_mi210.traces upload --out DATASET --message "<what changed>"` uploads the dataset folder, then checks that every file on Hugging Face has the same content as the local one and that the dataset is still private.

**The run sheet.** A run sheet is a JSON file that lists the runs to convert and states, for each run, the facts that its folder does not hold. The converter reads everything else from the run folder, such as the conversation, the submissions and the final score. Write the sheet before converting and commit it in `specs/`, so that every record can be traced to the sheet that produced it. The sheet names the machine that holds the run folders once (`machine`), and then gives one entry per run with these fields.

| Field | What to write | Where to find it |
|---|---|---|
| `record_id` | the record's name, usually the run folder's name | the run folder |
| `source` | `claude_code_session`, `codex_session`, `geak_run` or `open_model_run` | how the run was made |
| `run_dir` | the path of the run folder, starting with a named root such as `$WORK` (README, Setup); for your own session, the Claude Code project folder that holds the session file | the run folder |
| `session`, `slice` | for your own Claude Code session, the session file, and in `slice` the uuid of your request for this work (`from_uuid`) and of your next request (`until_uuid`, not included; leave it out to keep the session to its end) | `session-requests` |
| `task_dir` | the path of the Φ-Bench task package the run was on, with a named root; the converter reads the task's id, size, kind and scoring hardware from its `task.toml`, and the anchor from the file `task.toml` names (`tests/ref_speedup.txt` for our rows), so the sheet does not repeat them | the task package |
| `task` | only `origin`, the dataset and its revision | the task's dataset |
| `agent` | the `harness` and its `version` | the harness's own log |
| `model` | `name`, one of `Qwen3.8-27B` or `claude-opus-5-5`; for a model we served, also `precision` and `served_by` with `server` (`vLLM`), `version` and `settings` | the model server's log |
| `hardware` | empty when Anthropic or OpenAI served the model; for a model we served, `served_on`, such as `{"device": "AMD Instinct MI210", "count": 4}`, with a device that is one of `AMD Instinct MI210` or `AMD EPYC 9334` | the run's job log |
| `serving_log`, `job_log` | the model server's log and the run's job log, for an open-model run | the run's machine |
| `prompt_vs_task` | how the prompt the agent saw relates to the task: `instruction` (the task's `instruction.md`), `same problem, other prompt` or `different problem`; required for your own session and a GEAK run, and left out for an open-model run, which is always given the instruction | how the run was started |
| `invalid_reason` | why the run was not a fair attempt; leave it out for a fair one | the run's logs |
| `invalid_scores` | a final score that was thrown away, with the reason | the run folder |
| `submission_timing`, `unclean_submissions` | `checked` once someone has checked every in-session timing for another job on the same machine, and the submissions that shared it | the machine's job logs |
| `notes` | anything a reader of the record must know | |

`specs/examples/run-sheet-example.json` has one filled entry for each kind of run: your own Claude Code session, an open model's run and a GEAK run. Copy the entry that matches your run and change every value. For your own session, the task package's `oracle.patch` must be the session's result.

A GEAK run also names the measurement that sets its label and, when it is successful, its row (`docs/trace-records.md`). A GEAK run with no Φ-Bench task has no package, so it gives `task` (`id`, `bench`, `kind`, `origin`), `anchor` and `hardware.scored_on` itself instead of `task_dir`.

The converter refuses a run, and writes nothing for it, when a field is missing, a field repeats what the task package says, a device, model or server is not in its lists (add a new one to the converter on purpose), a named file does not exist, the anchor differs from the one the final score used, the harness version differs from the one in the run's log, an open-model run claims another prompt than the task's instruction, or a file holds a secret. It tries every run in the sheet, lists each refusal with its reason, and never overwrites a record that is already stored.

**Example.** The source run of this task is the first GEAK MoE run, stored as one successful record with 38 chats, 30 of them from a model. It is marked as having worked on a larger problem, the whole layer, because this task is one table entry of its result. Two runs of Qwen3.8-27B reached 1.143× and 1.137× against the anchor of 1.201437, so both are unsuccessful, and a third had a setup error and is invalid. The Qwen runs are described in the run sheet `specs/traces-qwen-runs-2026-11.json` and the GEAK runs in `specs/traces-geak-runs-2026-11.json`.

**Files.** [`Erland/rsi-traces`](https://huggingface.co/datasets/Erland/rsi-traces) (private), and the record format in `docs/trace-records.md`.

## Step 7. Submit for review

Send the package, the check results and the record ids for review. The first few tasks from each contributor are reviewed before that contributor's tasks count. Reviewed tasks go to the group's task dataset, which stays private until the owner decides to make it public. Publish with `git push` to the Hugging Face dataset repository, because the web upload drops the executable bits that the checker needs.

**Example.** The package was published with `git push` to `Erland/phibench-mi210-rows`, and a fresh clone passed every automatic check. It is not merged into Φ-Bench yet, because the owners, the license and a Docker build are still open.

**Files.** [the dataset page](https://huggingface.co/datasets/Erland/phibench-mi210-rows).

## Not automated yet

- The task form has no file format yet, and `row.py` builds packages only from GEAK runs. Until both exist, write the package by hand.
- The converter reads your own Claude Code session, GEAK runs, and open-model runs through Qwen Code and mini-swe-agent. It cannot read a Codex session yet, because no Codex session file has been shared to build the reader from.
- Claude Code saves its thinking blocks without their text, in GEAK runs and in a person's own session alike, so a Claude chat has no readable reasoning.
