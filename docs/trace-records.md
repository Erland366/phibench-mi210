# Trace records: one run of an agent on a task

> **Status (2026-11-26): draft for agreement between the two machines; no converter exists yet.**

The trace dataset holds one record per **run** (see [CONTEXT.md](../CONTEXT.md)): one session of an agent on one
task, from its prompt to its end. The decisions behind it are ADR 0002 (private, apart from the tasks) and ADR 0008
(every run kept, labeled by the task's verifier).

## Sources

| Source (`source`) | What one run is | Main conversation file |
|---|---|---|
| `claude_code_session` | the slice of a person's Claude Code session that did the work (SOP step 6) | the session JSONL slice, thinking blocks removed |
| `geak_attempt` | one GEAK engineer attempt (for example round 1, engineer 0) | that engineer's subagent JSONL |
| `open_model_run` | one run of an open model through a harness (Qwen Code, mini-swe-agent) | the raw model calls (`logs/agent/openai/*.json`) |

Not runs, and not stored: harness tests on dummy prompts, verifier calibration runs, collided runs. A run that was
meant as an attempt but whose agent never started is stored as invalid.

## Layout

```
index.jsonl                      one line per record: record_id, task id, source, model, label, reward, speedup
records/<record_id>/
  record.json                    everything below the files: metadata, outcome, submissions, provenance
  prompt.txt                     the exact prompt the agent got
  conversation/raw/...           the original files, copied byte for byte, never edited
  conversation/messages.jsonl    the run as one list of messages, generated from raw/ (see below)
  code/final.patch               the final working tree against the starting code (absent if it was not saved)
  code/submissions/<n>/          each submission's diff and its verifier result, if the harness submitted
  verifier/                      the final verifier output (reward.json and its logs)
```

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
| `anchor` | the `ref_speedup` the score used and where it was measured (upstream's, or a local recalibration and its path) |
| `time` | started, finished, wall-clock limit |
| `outcome` | `label` (`successful`, `unsuccessful`, `invalid` or `unscored`), `reward`, `speedup`, `correctness_ok`, `hard_fail_reasons`, `invalid_reason`, and the path of the verifier output it came from |
| `submissions` | per submission: number, speedup, reward, correctness, hard-fail reasons, diff path |
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

## Secrets and paths

Raw files keep internal paths and node names, because the dataset is private (ADR 0009). Secrets are never stored:
a raw file that holds a token, password, API key or a proxy address with credentials has only that secret blanked
out, and `record.json` `scrub` lists the file and the change. Making traces public would need a full scrub first.

## Known gaps

- The size of `raw/` for open-model runs grows with every call (each request repeats the conversation so far); it
  has to be measured before choosing where the dataset is stored.
- GEAK attempts are `unscored` until each attempt's patch is run through the verifier of the task built from that
  GEAK run.
