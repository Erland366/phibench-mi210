# How to turn your agent work into a benchmark task

This procedure turns a piece of systems work that you finished with a coding agent (Claude Code, Codex or a similar agent) into a Φ-Bench task and a trace. The **task** is a package, with its image, that another model can be given and scored on. The **trace** is the part of your conversation with the agent in which the work was done, and it is kept privately as training data.

The procedure works for any systems work that a program can check, such as a kernel made faster, a serving bug fixed or a feature added. It ends when your task passes its checks and your trace is stored. Evaluating models on the tasks and training on the traces are separate work.

Each step below ends with an example from a finished task, `mi210-jais3-moe-m256`, a speed task on AMD Instinct MI210 that is published in the Hugging Face dataset `Erland/phibench-mi210-rows`. Its source is a GEAK agent run rather than a conversation with Claude Code, and the examples say where that changed a step.

## What you need

- Finished work in a public repository, with the commit before your change and the commit after it.
- A check that separates a correct change from a wrong one, such as tests that fail before the change and pass after it, or a timing that shows the speedup.
- The stored conversation in which you and the agent did the work. Claude Code stores each session as a JSONL file under `~/.claude/projects/<folder>/`.
- Docker with `buildx`. On a machine without Docker, use our Enroot builder (`enroot.py build`).
- For a speed task, the hardware that the task will state, with no other jobs running on it while you measure.

## Step 1. Pick the work and decide its kind

Work is suitable for a task when it meets these conditions.

- It was accepted (merged, deployed or kept) and has one clear before and after.
- A program can check it in minutes, without anyone judging the output by eye.
- It would make sense as a request to a capable engineer who is not shown the solution.

Then decide its kind, because the kind sets how the task is scored.

| **Kind** | **Scored on** | **Example** |
|---|---|---|
| **Speed task** (`reward_class = performance`) | the speedup over the starting code, relative to the anchor | a kernel, a scheduler setting, a serving path |
| **Pass/fail task** (`reward_class = implementation`) | whether the tests pass | a bug fix, a missing feature, a port to new hardware |

Also decide its size, which Φ-Bench records in the `bench` column. Use `kfc` for completing one function, `lh` for a longer change across several files, and `e2e` for setting up or optimizing a whole system.

Skip work that only changes numbers, because someone could find those values by trying them without reading the code. The optional too-easy search in step 5 checks for this. Also skip work that needs more machines than a task can request.

**Example.** The work was one new entry in the decode tile table of the Mixture-of-Experts (MoE) kernel that a vLLM v0.32 server runs on MI210, for batches of 256 tokens. A GEAK agent found it inside a larger change, which was split into pieces, and this entry was measured on its own: about 1.2× faster at 256 tokens, with correct results. It is a speed task, and its bench is `lh`, because the solver may change the kernel and both of its tile tables.

**Files to look at:** [`task_catalog.jsonl`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/task_catalog.jsonl), the line for `mi210-jais3-moe-m256`.

## Step 2. Fill in the task form

The task form is one short file per task, with the fields below.

| **Field** | **What to write** |
|---|---|
| `id` | a short name, for example `mi210-vllm-moe-fp8-convert`, with the hardware first if the task depends on it |
| `kind`, `bench` | the kind and size chosen in step 1 |
| `repo`, `base_commit`, `solution_commit` | the public repository, the commit before the change and the commit after it |
| `editable` | the files the solver may change |
| `check` | the command that tests or times the work, and the output it prints |
| `hardware` | the device and the number of GPUs, for example one AMD Instinct MI210 |
| `trace` | the session file or files, and where the slice starts and ends (step 6) |
| `summary` | two or three sentences that state what the code is, what must be done, how it is scored and on what hardware |

**Tool not ready.** The file format of the task form and the builder that reads it do not exist yet. Today a task is described in `specs/<task>/spec.json`, which is written for GEAK runs and documented in `inputs.md`.

**Example.** The id is `mi210-jais3-moe-m256`, the kind `performance` and the bench `lh`. The editable files are `moe_w8a16.py`, `configs.json` and `dec_configs.json`, the hardware is one AMD Instinct MI210, and the check is the runner `scripts/task_runner.py` (compile, correctness, performance). Because the source is a GEAK run, the frozen starting files of the run took the place of the two commits, and the form was `specs/jais3-moe-m256/spec.json`.

**Files to look at:** [`task.toml`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/task.toml).

## Step 3. Write the instruction and the tests yourself

- `instruction.md` is everything the solver is told. It states the problem, the files the solver may change, the rules, the hardware and how the work is scored. It must not hint at the solution. It must also not name where the task came from (an agent, a project or a person), or any internal path, node, account or URL. Do not paste text from your conversation with the agent, because that text describes the solution.
- The tests in `tests/` decide the score. For a pass/fail task, they fail on the starting code and pass on the reference solution. For a speed task, they first check that the solver's code is correct and then time it against the starting code.
- An agent may draft the instruction and the tests. You then read every line, because you are the person who knows what a correct result means for this work.

**Example.** The instruction, about 3,900 characters, describes the layer (100 experts, top-8, hidden size 2560, FP8 weights with one scale per 128×128 block), the three editable files, the math that must not change and the rules the runner enforces, and says that only 256 tokens are scored. It does not mention the table entry. Three internal phrases, among them a data path, were replaced before the text was used. The tests first check correctness over all batch sizes (error within 1.02× of the starting code's, deterministic, no compilation during inference) and then time 256 tokens against the starting code.

**Files to look at:** [`instruction.md`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/instruction.md), [`tests/test.sh`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/tests/test.sh), [`tests/grade.py`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/tests/grade.py) and [`tests/grade_config.json`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/tests/grade_config.json).

## Step 4. Build the package and its image

The package follows the Φ-Bench layout exactly.

```
<id>/
  task.toml          id, kind, image, hardware, editable files, scoring
  instruction.md     the solver's whole prompt
  environment/       Dockerfile, the starting code (repo/), the runtime and submission scripts
  tests/             the verifier; for a speed task also ref_speedup.txt and its caveat
  solution/solve.sh  applies the reference solution (oracle.patch)
  oracle.patch       the reference solution
  negative.patch     the wrong edit made on purpose
```

Build the image from the package folder with `docker buildx build --load -f environment/Dockerfile -t <image> .` The image must contain the starting code and must never contain the reference solution.

**Tool not ready.** The builder that turns a task form into this package does not exist yet. Today `row.py` builds packages for GEAK runs only.

**Example.** The package was built by `row.py` from the GEAK run and its independent measurement. The image starts from `vllm/vllm-openai-rocm:v0.32.0` and adds the starting code and the scripts. The reference solution, `oracle.patch`, adds the 21-line table entry and is not in the image. The image was built with Enroot, following the Dockerfile step by step, because our MI210 nodes have no Docker; a Docker build of it is still to be done.

**Files to look at:** [`environment/Dockerfile`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/environment/Dockerfile), [`environment/repo/moe_w8a16.py`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/environment/repo/moe_w8a16.py) and [`environment/repo/dec_configs.json`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/environment/repo/dec_configs.json).

## Step 5. Run the checks

A task counts only when every check below passes. Before you run anything, write the plan for your measurement, stating what you will run, how many times, and which result passes. Keep every result, including failed runs and runs you discarded.

### Every task

- The image builds from its Dockerfile and contains no part of the reference solution.
- The tests reject the wrong edit. This includes the wrong edit made on purpose (`negative.patch`) and every failed attempt from your trace that ran but gave wrong results.
- The reference solution passes when it is applied the way a solver's change would be, with `solution/solve.sh`.
- `instruction.md` and `environment/` name nothing internal. Run the scrub check of step 6 on the package to confirm this.

### Pass/fail task

- The tests fail on the starting code and pass with the reference solution.

### Speed task

- The starting code passes the correctness check and scores 0.
- The reference solution is at least **1.1×** faster than the starting code. Take the median of **5 runs**, made one at a time, with nothing else running on the hardware before or after each run. This median is the anchor (`ref_speedup`).
- The anchor holds only for the hardware that the task states.

### Tasks whose solution is a table or config entry (optional)

Run the too-easy search (`table_search.py`). If a search that only tries values, without reading the code, finds the solution, the task is too easy.

**Example.** The plan, `specs/calibration-2026-11-08.json`, was written before the first run. On one MI210, one run at a time, the unchanged code scored 0 (1.0075×), and the wrong edit, which drops the router weight on the down matmul, failed correctness. The reference solution measured 1.1745, 1.1956, 1.2014, 1.2045 and 1.2051×, so the anchor is their median, 1.201437. The too-easy search tried 96 table entries: the best reached 1.010×, and the verifier gave its finalist 0.998×, so the task is not too easy. A calibration of another task was thrown away, and kept on record, because another job was using the same machine; since then every run checks that all GPUs are idle before and after it.

**Files to look at:** [`oracle.patch`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/oracle.patch), [`negative.patch`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/negative.patch), [`solution/solve.sh`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/solution/solve.sh), [`tests/ref_speedup.txt`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/tests/ref_speedup.txt) and [`tests/ref_speedup.caveat.md`](https://huggingface.co/datasets/Erland/phibench-mi210-rows/blob/main/tasks/lh/mi210-jais3-moe-m256/tests/ref_speedup.caveat.md).

## Step 6. Cut, clean and store the trace

- **Cut the slice.** One session usually contains several pieces of work. Mark the slice from your request for this work to the accepted solution, and keep the failed attempts inside it, because the ML team wants the failed attempts that led to the solution and not only the final answer. Keep only slices that end in an accepted solution.
- **Clean the slice.** Remove the agent's thinking blocks. Keep the messages, tool calls and tool outputs in their original JSONL form.
- **Run the scrub check.** The slice must contain no secrets, tokens, passwords, proxy settings, private paths, node names, account names or other people's data. Check the tool outputs most carefully, because these items appear there most often.
- **Store the slice** in the group's private trace dataset, with a small metadata file that records the task `id`, the agent and model, the date and the session file or files. Never put the trace in the task package.

**Tool not ready.** The slice cutter and the scrub check do not exist yet. Until they do, cut the slice by hand, search it for the items listed in the scrub check, and ask before uploading.

**Example.** Not done yet for this task. The GEAK run's conversation is stored with the run, but it has not been cut, cleaned or uploaded, because the trace dataset does not exist yet.

## Step 7. Submit for review

Send the package, the check results and the link to the trace for review. The first few tasks from each contributor are reviewed before that contributor's tasks count. Reviewed tasks go to the group's task dataset, which stays private until the owner decides to make it public. Publish with `git push` to the Hugging Face dataset repository and never with the web upload, because the web upload drops the executable bits that the checker needs.

**Example.** The package was published with `git push` to `Erland/phibench-mi210-rows` (commit `323e515`), and a fresh clone passed every automatic check. An earlier web upload had dropped every executable bit, which is why this step says `git push`. The task is not merged into Φ-Bench yet: the owners, the license and a Docker build are still open.

**Files to look at:** [the dataset page](https://huggingface.co/datasets/Erland/phibench-mi210-rows), whose README holds the calibration notes and the advice to fetch with `git clone`.
