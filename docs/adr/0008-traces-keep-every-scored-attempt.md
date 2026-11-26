# The trace dataset keeps every run, labeled successful, unsuccessful or invalid by Φ-Bench's reward

Status: accepted (2026-11-26); amends ADR 0002, which kept successful traces only

Upstream Φ-Bench stores no agent conversations, only each submission's speedups. We store them, for training, and
we keep unsuccessful runs as well as successful ones: on a task with a reference solution, an unsuccessful run still
has a known answer. Each record is one run of one agent on one task (a contributor's trace slice, one GEAK engineer
attempt, or one open-model run such as Qwen with Qwen Code), with its submissions kept inside it. Its label comes
from the task's own verifier with Φ-Bench's definition, never from the agent's own report: **successful** when the
final reward is above 0 (a speed task beats the anchor; a pass/fail task passes its tests), **unsuccessful** when it
is 0, and **invalid** when the run was not a fair attempt (a setup error, a busy machine during timing, an agent that
never started). A run not scored yet is unscored until it is. Every record also keeps the raw speedup, reward and
hard-fail reasons. Choosing which tasks are for training and which for evaluation is the user's, later; for now
every run is stored with its task id.

For open models the model's own reasoning is kept; for Claude traces thinking blocks are still dropped. The main
copy of a conversation is the raw model calls (each request and response as sent), because a harness that compacts
its history rewrites what its own log shows; the harness's log is kept as is, and a flat message list is generated
from the raw calls. Raw files are never edited.

A run is stored only once its task exists as a Φ-Bench row (user, 2026-11-26): the row comes first, then the traces
of the runs on it. For GEAK runs ADR 0010 narrows this to successful ones: unsuccessful GEAK runs are stored now.

## Consequences

- A GEAK attempt is scored on the task built from that GEAK run; since the anchor is the run's final result, most of
  its earlier attempts will be unsuccessful by this definition even when they were correct and somewhat faster,
  which the raw speedup keeps visible.
- A run whose final code was not saved has no patch in its record; it is never reconstructed.
- Not runs: harness tests on dummy prompts, calibration runs of the verifier, runs whose agent never started are
  stored as invalid only if they were meant as attempts.
