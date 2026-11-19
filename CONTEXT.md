# Φ-Bench task creation

How the group turns finished system work, done with a coding agent, into Φ-Bench tasks that a model can be scored
on, and keeps the conversations as training data for the RSI work.

## Language

**Task**:
One package a solver works on: an image built from its Dockerfile, `instruction.md`, tests, a reference solution
and `task.toml`, in Φ-Bench's format.
_Avoid_: sample, problem, example

**Row**:
A task's single line in the catalog, with Φ-Bench's 8 columns unchanged.
_Avoid_: entry, record

**Trace**:
The recorded conversation in which a person and a coding agent solved the work a task is built from.
_Avoid_: trajectory, transcript, chat, sample

**Reference solution**:
The patch that solves a task; for a speed-scored task its measured speedup is the anchor.
_Avoid_: oracle patch (upstream's file name `oracle.patch` stays), ground truth, answer

**Anchor**:
The speedup of the reference solution, measured on the task's own verifier and stated hardware; it is the row's
`ref_speedup`.
_Avoid_: target, baseline speedup

**Contributor**:
A person who turns one of their own traces into a task.
_Avoid_: annotator, author

**Source**:
Where a task's solved work comes from: a contributor's agent conversation, or a GEAK agent run (one kind of source
among several).

**Task form**:
The short file a contributor fills in for one task (repository and base commit, solution commit, how it is checked,
hardware, the trace slice), from which the task package is built.
_Avoid_: spec (our older `spec.json` is the GEAK-source version of it), intake

**Trace slice**:
The part of a stored agent session, marked by the contributor, from the request to the accepted solution; one
session usually holds many pieces of work, so a trace is a slice, not a whole session.

**Attempt**:
One try within a trace at a solution, ending when it is checked (it fails, or it is accepted).
_Avoid_: round (taken: the Round check is GEAK's independent measurement), iteration, try

**Scrub check**:
The automatic check that a trace slice holds no secrets, proxy settings, private paths or node names before it
leaves the contributor's machine.

**Wrong edit**:
An incorrect change shipped with a task, which the verifier must reject; it proves the checks can fail. Each
task has one made on purpose, plus any failed attempt from its trace that ran but gave wrong results.
_Avoid_: negative (upstream's file name `negative.patch` stays)

## Kinds of task

**Speed task**:
A task scored on how much faster the solver's code is than the starting code, relative to the anchor
(`reward_class: performance`).
_Avoid_: performance task, optimization task

**Pass/fail task**:
A task scored only on whether its tests pass, such as adding a feature (`reward_class: implementation`).
_Avoid_: implementation task, feature task
