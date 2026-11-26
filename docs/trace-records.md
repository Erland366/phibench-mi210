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
| `claude_code_session` | the slice of a person's Claude Code session that did the work (SOP step 6) | the session JSONL slice, thinking included |
| `geak_run` | one whole GEAK run, all its agents (ADR 0010) | every agent's Claude Code JSONL as recorded, thinking included |
| `codex_session` | the slice of a person's Codex session (not read yet) | the session JSONL slice, reasoning removed |
| `open_model_run` | one run of an open model through a harness (Qwen Code, mini-swe-agent) | the raw model calls (`logs/agent/openai/*.json`) |

Not runs, and not stored: harness tests on dummy prompts, verifier calibration runs, collided runs. A run that was
meant as an attempt but whose agent never started is stored as invalid.

A run is stored only once its task exists as a Φ-Bench row (user, 2026-11-26). GEAK runs whose result has not been
turned into a row yet (the attention and MoVA runs, for example) keep their conversations where they are and wait.

## Converting: the run sheet

`python -m phibench_mi210.traces convert --sheet SHEET.json --out DATASET` converts the runs a sheet lists; `index
--out DATASET` rebuilds `index.jsonl`. The sheet holds what a run folder does not say itself, and nothing the task
package says (sheet format 2, 2026-11-26; `specs/examples/run-sheet-example.json` has one entry of each kind):

```json
{"machine": "auh7-1b-gpu-240",
 "runs": [{"record_id": "2026-11-25-qwen-code-causal-delivery-vclock-1", "source": "open_model_run",
           "run_dir": ".../runs/2026-11-25-qwen-code-causal-delivery-vclock-1",
           "task_dir": ".../local/wro-causal-delivery-vclock-coupled-epyc9334",
           "task": {"origin": "upstream Phi-Bench 3bcefac, recalibrated on the AMD EPYC 9334"},
           "agent": {"harness": "qwen-code"},
           "model": {"name": "Qwen3.8-27B", "precision": "bf16",
                     "served_by": {"server": "vLLM", "version": "0.32.0", "settings": "tensor parallel 4, ..."}},
           "hardware": {"served_on": {"device": "AMD Instinct MI210", "count": 4}},
           "serving_log": ".../logs/serve-qwen-...-20261125T2127.log",
           "job_log": ".../runs/2026-11-25-qwen-code-causal-delivery-vclock-1.job.log",
           "submission_timing": "checked", "unclean_submissions": {},
           "invalid_reason": null, "invalid_scores": [], "notes": ""}]}
```

From `task_dir` the converter reads the task's `id`, `bench` and `kind` (`task.toml` `[task]`), the anchor
(the file `task.toml` names in `ref_speedup_source`: `tests/ref_speedup.txt` for our rows, the verifier manifest for the upstream CPU tasks; it must agree with `task.toml`'s `ref_speedup`) and `hardware.scored_on`
(`[primary_metric] calibrated_on`, with `[environment] gpus` as the count for a GPU), and refuses a sheet entry that
repeats any of them. `model.name`, the devices and `served_by.server` are fixed choices (`MODEL_NAMES`, `DEVICES` and
`SERVERS` in `traces.py`); anything else is refused, and details such as which GPUs go in `notes`. `served_on`,
`precision` and `served_by` are for a model we served ourselves (`open_model_run`) and are refused for a Claude or
Codex session. Only a GEAK run with no Φ-Bench task leaves out `task_dir` and gives `task` (`id`, `bench`, `kind`,
`origin`), `anchor` and `hardware.scored_on` itself.

The converter checks the sheet against the run: the harness version and the anchor the final score used must match
what the sheet says, or it refuses. `submission_timing: "checked"` means someone checked each in-session timing for a
shared machine; the ones that were shared go in `unclean_submissions` (`{"<n>": "<reason>"}`). Without it, every
submission's `timing_clean` is unknown (`null`). A thrown-away final score goes in `invalid_scores`
(`[{"dir": "final-busy", "reason": "..."}]`). Readers exist for `open_model_run` with Qwen Code and mini-swe-agent,
for `geak_run`, and for `claude_code_session`; a Codex session is refused by name until its reader is written.

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
index.jsonl                      one line per record: record_id, task id, source, model, label, reward, speedup, chats
viewer/chats.jsonl               one row per chat with its messages and its run's label, for the Hugging Face viewer
records/<record_id>/
  record.json                    everything below the files: metadata, outcome, submissions, provenance
  prompt.txt                     the exact prompt the agent got
  conversation/raw/...           the original files, copied byte for byte, never edited
  conversation/chats/<nn>-<name>.jsonl   the chats: one file per conversation, generated from raw/ (see below)
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

**Where the chat is, for every source (layout `chats-v1`):** `conversation/chats/`, one file per conversation the
model had, in the order the conversations started, each listed in `record.json` `conversation.chats` with its
number, name, file, number of messages and whether it is `synthetic`. A run with one agent (an open model's run, a
person's session) has one chat, `01-solver.jsonl`. A GEAK run has one chat per agent, for example
`10-verify-r1-d0.jsonl`, because each agent is a separate conversation that never sees the others. One chat file is
the unit to read and to train on. There is no file that runs several chats together.

For an open-model run, `conversation/raw/` holds the raw model calls (`logs/agent/openai/*.json`), the harness
stream (`stream.jsonl`), the harness's own chat recording and its home folder files (settings, usage record), or
`trajectory.json` for mini-swe-agent.

For a GEAK run the record holds, in place of the open-model files:

```
  conversation/raw/claude/...        every agent's Claude Code JSONL, byte copies, thinking included
  conversation/readable/             GEAK's readable copy of each agent's chat (chat/*.md and its index)
  harness/                           outcome.json, receipt.json, the driver's spec and result, transcript.jsonl
  harness/geak/                      GEAK's own working files: plan, roadmap, insight log, metrics, each engineer's
                                     report, self-reported result and patch, the integration patches and logs
  code/final.patch                   GEAK's final patch, when it wrote one
  code/final/                        the candidate files the Round check measured
  code/reference.patch               for a row made from one piece of the result: that piece
  verifier/round-check/              the independent measurement that sets the label (prepare.json, verdict.json)
  verifier/piece-scores.json         for a piece: the scores the row was accepted on
```

`prompt.txt` is the top-level agent's first message (GEAK's own prompt). A GEAK chat's entry in
`conversation.chats` also gives when the agent started, its models, its readable copy and its raw file. A chat whose
every reply came from the harness rather than a model (Claude Code names the model `<synthetic>`, for example when
GEAK stops and closes its remaining steps) is kept and marked `synthetic: true`; leave it out before training.

For a person's own Claude Code session (source `claude_code_session`, harness `claude-code`), the sheet entry also
gives `session` (the session file), `slice` (`from_uuid`, the person's request for the work, and `to_uuid`, the last
line of the work, or left out for the end of the file; `python -m phibench_mi210.traces session-requests --session
<file>` lists the requests with their uuids); its `task_dir` is the task package whose `oracle.patch` is the session's
result. The record holds:

```
  conversation/raw/claude/<session>.slice.jsonl   the slice: the session's original lines, byte for byte
  conversation/raw/claude/subagents/              the helper agents that started inside the slice, with their meta files
  conversation/chats/01-solver.jsonl              the session; 02-... one chat per helper agent
  code/reference.patch                            the task's oracle.patch, the session's result
```

Its label is `successful`, because its result is the task's reference solution; the converter refuses the entry when
the slice does not start at a request the person typed or the task's `oracle.patch` does not exist.
`conversation.slice` records the line numbers, the two uuids, the first and last time and the Claude Code versions,
and provenance keeps the whole session file's hash next to the slice's. A Codex session is refused until a Codex
reader exists.

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
| `task` | `task_dir` (the task package the facts were read from, when there is one), `id`, `bench`, `kind` (`performance` or `implementation`), `origin` (which dataset and revision, for example upstream Φ-Bench `3bcefac` or `Erland/phibench-mi210-rows` `323e515`), and `prompt_vs_task`: `instruction` when the agent was given the task's own `instruction.md` (every open-model run), `same problem, other prompt` (a GEAK run on its own task, with GEAK's prompt), or `different problem` (a row made from one piece of a GEAK run's result, such as m256 and m1024, whose conversation worked on the whole layer); filter on it before training |
| `agent` | harness and version (for example Qwen Code 0.28.4), its settings file |
| `model` | `name` (one of `MODEL_NAMES`), and for a model we served `precision` and `served_by` (`server`, `version`, `settings`); both `null` for a Claude or Codex session |
| `hardware` | `served_on` (`{"device", "count"}` for a model we served, `null` otherwise) and `scored_on` (`{"device", "count"}`, from the task package); `device` is one of `DEVICES` |
| `anchor` | the `ref_speedup` the score used and `evidence`, the task package's `tests/ref_speedup.txt`; `null` for a GEAK run with no Φ-Bench task |
| `time` | started, finished, wall-clock limit |
| `outcome` | `label` (`successful`, `unsuccessful`, `invalid` or `unscored`), `reward`, `speedup`, `correctness_ok`, `hard_fail_reasons`, `invalid_reason`, the harness exit code, whether the final tree was the winning submission or the starting code, the path of the counted verifier output, and `invalid_scores` (thrown-away scores with their reasons) |
| `submissions` | per submission: number, speedup, reward, correctness, hard-fail reasons, diff path, and `timing_clean` (false when the machine was shared during that timing, with the reason) |
| `conversation` | `layout` (`chats-v1`) and `chats` (every chat: `n`, `name`, `file`, `messages`, `synthetic`), which raw files are the main copy, whether reasoning is included, whether thinking was dropped (never, since 2026-11-26), for Claude traces the number of thinking blocks and how many hold text, number of compactions |
| `provenance` | machine, original paths, date copied, SHA-256 of every copied file |
| `scrub` | whether the scrub check passed, by whom, when, and anything it changed |

The label follows ADR 0008: `successful` when the final reward is above 0, `unsuccessful` when it is 0, `invalid`
when the run was not a fair attempt, `unscored` until the task's verifier has scored it. The agent's own report
never sets it.

## A chat file

Every chat file has the same message shape (`messages-v1`), whatever the source: a person's Claude Code or Codex
session, a GEAK run, or an open model's run. The raw files next to it stay in each tool's own format. Each line is one
message with exactly these fields, in this order:

| Field | Holds |
|---|---|
| `turn` | its position in the file, from 0 |
| `role` | `system`, `user`, `assistant`, `tool`, or `event` for something the harness did (see below) |
| `content` | plain text (a message sent as several text parts is joined with newlines; `extra.content_parts` says how many) |
| `reasoning` | the model's own reasoning: an open model's reasoning, or the text of Claude's thinking blocks; `null` when there is none. Claude Code can store a thinking block with an empty text and only a signature, as in every block of our GEAK runs, so their `reasoning` is `null` although the blocks are kept in the raw files |
| `tool_calls` | for an assistant message, the tools it called: `[{"id", "type": "function", "function": {"name", "arguments": <JSON text>}}]` |
| `tool_call_id` | for a tool message, which call it answers |
| `agent` | the name of the chat it belongs to: `solver` for a run with one agent, the GEAK agent (for example `eng r1_d0:compute`) otherwise |
| `source_index` | where it is in the raw files: the model call for Qwen Code, the step for mini-swe-agent; `null` for Claude Code |
| `event` | for `role: event`, what happened |
| `extra` | anything only one source has: mini-swe-agent's parsed `actions` and `format_error`, a tool message's `is_error`, Claude Code's `timestamp` and `model`, a compaction's `part`, a side call's `system` |

Events: `compaction` (the harness summarized the history; `content` is the summary, `reasoning` the model's
reasoning; consecutive summarizer calls are parts of one compaction), `message_updated` (the harness added text parts
in front of an earlier message; `content` is what it added), `side_call` (a harness call that is not the solver, such
as Qwen Code's memory extraction; it does not enter the conversation), `failed_call` (`content` is the error),
`history_rewritten` (the next request did not continue the previous one in any known way), and `exit`
(mini-swe-agent's end; `content` is the exit status). Where the harness compacted its history, the raw calls in
`raw/` show exactly what the model saw after it.

`python -m phibench_mi210.traces remessage --out DATASET` rebuilds every stored record's chats from the raw files
inside it in the current shape and layout, for example after either changes; it never touches a raw file, removes
the generated files of the earlier layout (`conversation/messages.jsonl` and `conversation/agents/`, before
2026-11-26), and `record.json` notes the format, the layout and when the chats were rebuilt.

How a compaction looks in Qwen Code's raw calls (Qwen Code 0.28.4): the request's last message is a user message
that starts "You are the component that summarizes a conversation when its context window is about to overflow",
sent with `max_tokens` 20000. The response's `content` is the summary and `reasoning_content` the model's reasoning.
The next call restarts from a short history: the system message, the summary fed back as a user `<state_snapshot>`
message, a short assistant acknowledgement, and a user message listing the files read before. The compaction call
itself is an event in the chat, not a turn of the task.

## Secrets and paths

Raw files keep internal paths and node names, because the dataset is private (ADR 0009). Secrets are never stored:
a raw file that holds a token, password, API key or a proxy address with credentials has only that secret blanked
out, and `record.json` `scrub` lists the file and the change. Making traces public would need a full scrub first.

## Known gaps

- Size: each raw call repeats the conversation so far. The six Qwen Code runs of 2026-11-24 to 25 hold 19 to 53 MB
  of raw calls each (49 to 190 files, the largest 660 KB), about 167 MB in all, which compresses well. Where the
  dataset is stored is still to be decided.
- Readers for a person's own Claude Code or Codex session, with the slice cutter, are not written yet.
