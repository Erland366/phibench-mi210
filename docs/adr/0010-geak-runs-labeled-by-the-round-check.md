# A GEAK run is one record, labeled by the independent Round check against the row rule

Status: accepted (2026-11-26); refines ADR 0008 for GEAK runs

A task's reference solution comes out of a whole GEAK run (planning, several engineers, verifying, integrating), so
one GEAK run is one record: every agent's conversation kept as its own file and message list, and the engineer
attempts with GEAK's own reported results inside it. Its label comes from the independent Round check with the
rule used to accept a row (correct, and at least 1.1x faster, the paper's minimum): **successful** when the run's
result passed and became a row's reference solution, **unsuccessful** when it was measured and was not fast enough
or not correct. GEAK's own reported speedups never set the label. Each record says which rule labeled it
(`outcome.label_rule`), so GEAK records and model runs on a task (labeled by the task's verifier, ADR 0008) are
never mixed up.

The row-first rule of ADR 0008 applies to successful records: a GEAK run's successful record is stored once its row
exists. Unsuccessful GEAK runs (MoVA v1 and v2, both attention runs; the second attention run committed nothing,
which counts as unsuccessful, not invalid) are stored now, tied to the GEAK task they worked on and marked as having
no Φ-Bench row. Pilot, probe and smoke runs tested the setup and are not stored.

When one GEAK run is the source of several rows (the first MoE run gave both `mi210-jais3-moe-m256` and
`mi210-jais3-moe-m1024`), each row gets its own record holding the same run, with the task defined as that row and a
note that its reference solution is one piece of the run's result. Such a run gets no record of its own for the task
it was given, even when its whole result failed it (the first MoE run's whole result was 1.094x): the same
conversation would otherwise be both a successful and an unsuccessful example. The whole result is kept in each
record as context (`outcome.run_result`), never as the label. Because that conversation worked on the whole layer
and never saw the row's instruction, each such record says so (`task.prompt_vs_task` = `different problem`, user,
2026-11-26), so that whoever trains can leave it out. Claude's thinking blocks are dropped, as for every
Claude trace.

## Considered options

- Label the source run "reference", outside the verifier labels: by Φ-Bench's reward a reference only ties its own
  anchor, but the user chose successful and unsuccessful by the row rule instead.
- One record per engineer attempt: the reference came from the whole run, not one engineer.
- One record linked to several tasks: breaks "one record, one task".
