# Trace records: one run of an agent on a task

> **Status (2026-11-26): agreed with the agent on the second machine. The converter (`src/phibench_mi210/traces.py`)
> reads Qwen Code and mini-swe-agent runs; it is tested on small folders shaped like the real runs, not yet on the
> real runs.**

The trace dataset holds one record per **run** (see [CONTEXT.md](../CONTEXT.md)): one session of an agent on one
task, from its prompt to its end. The decisions behind it are ADR 0002 (private, apart from the tasks) and ADR 0008
(every run kept, labeled by the task's verifier).

## Sources

| Source (`source`) | What one run is | Main conversation file |
|---|---|---|
| `claude_code_session` | the slice of a person's Claude Code session that did the work (SOP step 6) | the session JSONL slice, thinking blocks removed |
| `geak_run` | one whole GEAK run, all its agents (ADR 0010) | every agent's Claude Code JSONL, thinking blocks removed |
| `codex_session` | the slice of a person's Codex session (not read yet) | the session JSONL slice, reasoning removed |
| `open_model_run` | one run of an open model through a harness (Qwen Code, mini-swe-agent) | the raw model calls (`logs/agent/openai/*.json`) |

Not runs, and not stored: harness tests on dummy prompts, verifier calibration runs, collided runs. A run that was
meant as an attempt but whose agent never started is stored as invalid.

A run is stored only once its task exists as a Φ-Bench row (user, 2026-11-26). GEAK runs whose result has not been
turned into a row yet (the attention and MoVA runs, for example) keep their conversations where they are and wait.

## Converting: the run sheet

`python -m phibench_mi210.traces convert --sheet SHEET.json --out DATASET` converts the runs a sheet lists; `index
--out DATASET` rebuilds `index.jsonl`. The sheet holds what a run folder does not say itself:

```json
{"machine": "auh7-1b-gpu-240",
 "runs": [{"record_id": "2026-11-25-qwen-code-causal-delivery-vclock-1", "source": "open_model_run",
           "run_dir": ".../runs/2026-11-25-qwen-code-causal-delivery-vclock-1",
           "task": {"id": "wro-causal-delivery-vclock-coupled-loop16", "bench": "lh", "kind": "performance",
                    "origin": "upstream Phi-Bench 3bcefac"},
           "agent": {"harness": "qwen-code"},
           "model": {"name": "Qwen3.8-27B", "precision": "bf16", "served_by": "vLLM 0.32.0, TP4, 131072 context"},
           "hardware": {"served_on": "4 x MI210", "scored_on": "AMD EPYC 9334"},
           "anchor": {"ref_speedup": 175.96, "evidence": ".../runs/2026-11-25-...-calibration"},
           "serving_log": ".../logs/serve-qwen-...-20261125T2127.log",
           "job_log": ".../runs/2026-11-25-qwen-code-causal-delivery-vclock-1.job.log",
           "submission_timing": "checked", "unclean_submissions": {},
           "invalid_reason": null, "invalid_scores": [], "notes": ""}]}
```

The converter checks the sheet against the run: the harness version and the anchor the final score used must match
what the sheet says, or it refuses. `submission_timing: "checked"` means someone checked each in-session timing for a
shared machine; the ones that were shared go in `unclean_submissions` (`{"<n>": "<reason>"}`). Without it, every
submission's `timing_clean` is unknown (`null`). A thrown-away final score goes in `invalid_scores`
(`[{"dir": "final-busy", "reason": "..."}]`). Readers exist for `open_model_run` with Qwen Code and mini-swe-agent, and for `geak_run`;
others (a person's Claude Code or Codex session) are refused by name until they are written.

A GEAK run's sheet entry has `"agent": {"harness": "geak"}` and names its evidence and, for a successful run, its
row (ADR 0010):

```json
{"source": "geak_run", "run_dir": ".../geak-agent/jais3-moe-v2-v032-01",
 "evidence": {"kind": "round_check", "dir": ".../geak-agent/jais3-moe-v2-v032-01.round-check-current"},
 "row": {"provenance": ".../phibench-rows/provenance/mi210-jais3-routed-moe-w8a16.json"}}
```

For a row made from one piece of a result, the evidence is the piece:
`{"kind": "piece", "scores": ".../scores.json", "piece": "e2-m256-decode", "work_dir": ".../e2-m256-decode/work",
"plan": "specs/..."}`. The converter labels the run successful only when the evidence passed (the Round check: pass,
correctness PASS, control identical, score at least 1.1; a piece: its `qualifies` flag) and the row's provenance
names this run, this evidence and this task; it refuses a passed run without its row. A run that did not pass is
unsuccessful, with no row.

## Layout

```
index.jsonl                      one line per record: record_id, task id, source, model, label, reward, speedup
records/<record_id>/
  record.json                    everything below the files: metadata, outcome, submissions, provenance
  prompt.txt                     the exact prompt the agent got
  conversation/raw/...           the original files, copied byte for byte, never edited
  conversation/messages.jsonl    the run as one list of messages, generated from raw/ (see below)
  code/final.patch               the final working tree against the starting code (absent if it was not saved)
  code/final.status.txt          git status of that tree, next to the patch
  code/submissions/<n>/          each submission's diff and its verifier result, if the harness submitted
  harness/                       what the harness itself wrote: exit code and sandbox (agent_result.json),
                                 stderr, sandbox build log, start and finish times, the job log with step times
  harness/loop/                  the submission loop's own files: finalize.json (score trajectory, winning
                                 submission, the solver's stated reason for stopping), best.json,
                                 finalize_apply.log, count, dev/
  serving/                       the model server's log for this run (flags, version, cache statistics)
  verifier/                      the counted final score: reward.json, the full verifier output, finalize.txt,
                                 console log, exit code
  verifier/invalid-<n>/          a final score that was thrown away (for example one taken on a busy GPU),
                                 kept with its reason in outcome.invalid_scores
```

For an open-model run, `conversation/raw/` holds the raw model calls (`logs/agent/openai/*.json`), the harness
stream (`stream.jsonl`), the harness's own chat recording and its home folder files (settings, usage record), or
`trajectory.json` for mini-swe-agent.

For a GEAK run the record holds, in place of the open-model files:

```
  conversation/raw/claude/...        every agent's Claude Code JSONL, thinking blocks removed, all else as recorded
  conversation/readable/             GEAK's readable copy of each agent's chat (chat/*.md and its index)
  conversation/agents/<nn>-<role>.jsonl   each agent's messages, in the order the run started them
  conversation/messages.jsonl        all agents' messages, each line naming its agent
  harness/                           outcome.json, receipt.json, the driver's spec and result, transcript.jsonl
  harness/geak/                      GEAK's own working files: plan, roadmap, insight log, metrics, each engineer's
                                     report, self-reported result and patch, the integration patches and logs
  code/final.patch                   GEAK's final patch, when it wrote one
  code/final/                        the candidate files the Round check measured
  code/reference.patch               for a row made from one piece of the result: that piece
  verifier/round-check/              the independent measurement that sets the label (prepare.json, verdict.json)
  verifier/piece-scores.json         for a piece: the scores the row was accepted on
```

Workspaces, baselines and profiler output stay in the run folder, which `provenance.run_dir` names. Each engineer
attempt is listed under `submissions` with GEAK's own reported speedup, marked as self-reported; it never sets the
label.

`finalize_apply.log` matters for reading the outcome: it says whether the loop planted the winning submission into
the tree before the final score, or left the starting code (as when every submission was rejected), so it decides
what the final score measured.

`record_id` is the run's own folder name where it has one (for example
`2026-11-25-qwen-code-causal-delivery-vclock-1`), and `<date>-geak-<run>-<attempt>` for a GEAK attempt.

## `record.json`

| Field | Holds |
|---|---|
| `record_id`, `source` | as above |
| `task` | `id`, `bench`, `kind` (`performance` or `implementation`), `origin` (which dataset and revision, for example upstream Φ-Bench `3bcefac` or `Erland/phibench-mi210-rows` `323e515`) |
| `agent` | harness and version (for example Qwen Code 0.28.4), its settings file |
| `model` | name (for example Qwen3.8-27B), precision, how it was served (vLLM version, GPUs, context length) |
| `hardware` | where the run and its score happened (for example 4 × MI210 serving, scored on AMD EPYC 9334) |
| `anchor` | the `ref_speedup` the score used, where it was measured and the path of its evidence (upstream's; ours from this repo's calibration; or a local recalibration on the scoring machine, which keeps upstream's value as `upstream_ref_speedup`) |
| `time` | started, finished, wall-clock limit |
| `outcome` | `label` (`successful`, `unsuccessful`, `invalid` or `unscored`), `reward`, `speedup`, `correctness_ok`, `hard_fail_reasons`, `invalid_reason`, the harness exit code, whether the final tree was the winning submission or the starting code, the path of the counted verifier output, and `invalid_scores` (thrown-away scores with their reasons) |
| `submissions` | per submission: number, speedup, reward, correctness, hard-fail reasons, diff path, and `timing_clean` (false when the machine was shared during that timing, with the reason) |
| `conversation` | which raw files are the main copy, whether reasoning is included, whether thinking was dropped, number of compactions |
| `provenance` | machine, original paths, date copied, SHA-256 of every copied file |
| `scrub` | whether the scrub check passed, by whom, when, and anything it changed |

The label follows ADR 0008: `successful` when the final reward is above 0, `unsuccessful` when it is 0, `invalid`
when the run was not a fair attempt, `unscored` until the task's verifier has scored it. The agent's own report
never sets it.

## `messages.jsonl`

One line per message, in the OpenAI chat shape so that every source looks the same:
`{"role": "system" | "user" | "assistant" | "tool", "content": ..., "reasoning": ..., "tool_calls": [...],
"tool_call_id": ..., "call": <index of the model call that produced or first sent it>}`.
Where the harness compacted its history, a line `{"role": "event", "event": "compaction", "summary": ...}` marks
the point, and the raw calls in `raw/` show exactly what the model saw after it. `reasoning` is filled for open
models and left out for Claude.

How a compaction looks in Qwen Code's raw calls (Qwen Code 0.28.4): the request's last message is a user message
that starts "You are the component that summarizes a conversation when its context window is about to overflow",
sent with `max_tokens` 20000. The response's `content` is the summary and `reasoning_content` the model's reasoning.
The next call restarts from a short history: the system message, the summary fed back as a user `<state_snapshot>`
message, a short assistant acknowledgement, and a user message listing the files read before. The compaction call
itself is an event in `messages.jsonl`, not a turn of the task.

## Secrets and paths

Raw files keep internal paths and node names, because the dataset is private (ADR 0009). Secrets are never stored:
a raw file that holds a token, password, API key or a proxy address with credentials has only that secret blanked
out, and `record.json` `scrub` lists the file and the change. Making traces public would need a full scrub first.

## Known gaps

- Size: each raw call repeats the conversation so far. The six Qwen Code runs of 2026-11-24 to 25 hold 19 to 53 MB
  of raw calls each (49 to 190 files, the largest 660 KB), about 167 MB in all, which compresses well. Where the
  dataset is stored is still to be decided.
- Readers for a person's own Claude Code or Codex session, with the slice cutter, are not written yet.
