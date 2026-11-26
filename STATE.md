# State
Project phibench-mi210 (Phi-Bench upstream rows on Apptainer, Qwen Code solver) · Mode attended (away period ended 2026-11-26 02:55 UTC) · Updated 2026-11-26 03:00 UTC

## Hand-back / plan for this period
Hand-back for the away period 2026-11-25 18:45 - 2026-11-26 02:55 UTC (stopped early: no contract row left that fits before 05:00).
- Ran 4 upstream CPU rows end to end (download, Apptainer image, calibration x5, Qwen Code 90 min, final score, audit).
  All audits clean (no grader/test/patch reads, no network).
- q5k (llama.cpp SIMD): 0 submissions in 90 min (lost in its own scratch harness); final 1.38x vs anchor 2.60, reward 0.
- causal-delivery: **reward 0.139** (360.9x vs anchor 176.0x); independent review: legitimate.
- memory-accounting: correct 40.6x vs anchor 1122x, reward 0.
- secure-agg: both submissions correct at ~131x (2.6x the reference) but rejected by the upstream in-session 2x
  "implausible" gate, reported to the solver as a crash; finalize then restored the baseline: final 1.01x, reward 0.
- Failures and fixes: q5k build saw host /tmp (bound an empty dir); two queue copies collided on row 2 (my kill missed;
  outputs kept .collided, queue now single-instance); slim images lack libatomic for Node (rpath copy of node; row 2
  solver rerun). Time per row about 2 h; anchors on this EPYC 9334 are 6-20% below upstream, q5k 4.4x below.
- GPU use: model server on GPUs 4-7 about 6 h 10 min (4 sessions); GPUs 0-3 untouched. Nothing committed.
- Decisions for you are below; the main one: the in-session 2x gate cost rows 2 and 4 their best correct code.

## Now
- Question: can Qwen3.8-27B + Qwen Code beat Phi-Bench references on CPU rows here, and what does each row cost in time?
- Best current answer: 1 positive of 5 rows: causal-delivery reward 0.139 (360.9x vs 176.0x, independent review: legitimate); wro-pq and q5k reward 0
- Running: nothing (model server stopped, all GPUs idle)

## Waiting for the user
- Rescore the rejected secure-agg submissions (logs/loop/submissions/{1,2}/candidate.diff) and the rejected causal-delivery 430.7x submission with the final verifier, as an exploratory number (about 3 min CPU each). Recommendation: yes, labelled exploratory; the official scores stay as recorded.
- Next contract row, kfc arch-tiling-dram-traffic-min (downloaded 01:03 UTC, def generated, not built): its agent limit is 6 h (21600 s), 1 submission, and it scores DRAM traffic (a count, so the anchor should not depend on the CPU). It cannot finish inside this period; anchor files are tests/ref_speedup.txt plus a baked dev copy, so scripts/anchor_local.py needs a small extension. Recommendation: run it next with its full 6 h limit.
- Upstream in-session grader rejects correct code above 2x the reference and reports it as a crash (row 2; final verifier allows 10x). Recommendation: keep runs as upstream (recorded), report it upstream; do not patch the in-session grader without a new plan.
- q5k anchor on this CPU is 2.60 vs upstream 11.53 (4.4x lower, same image): recommend noting it as a hardware caveat for that row.

## Transitions
- 2026-11-25 18:45 away mode entered (CONTRACT.md revision 1)
- 2026-11-25 18:37 q5k build 1 failed: upstream anti-leak find saw the host /tmp (another user's pytest negative.patch); rebuilding with an empty dir bound at /tmp
- 2026-11-25 19:10 q5k calibration done: anchor 2.600214 (upstream 11.53); row 2-4 downloaded, defs generated (scripts/gen_def.py)
- 2026-11-25 19:30 rows 2-4 images built, plans frozen, overnight queue started
- 2026-11-25 20:52 row 1 q5k done: 0 submissions, final 1.3808 vs anchor 2.6002, reward 0, audit clean; row 2 started
- 2026-11-25 20:59 two queue copies collided on row 2 (my kill at 19:20 missed the inner bash); both stopped, outputs kept as .collided, one locked queue restarted 21:00
- 2026-11-25 21:25 row 2 calibration done (anchor 175.956); solver did not start (libatomic missing in slim image); fixed with an rpath node copy; restarted 21:29
- 2026-11-25 23:03 row 2 done: final 360.87 vs anchor 175.96, reward 0.139 (first positive); independent review running; row 3 started
- 2026-11-25 23:08 row 2 independent review: score legitimate; found upstream in-session 2x gate + 'crash' feedback defect (rejected correct 430.7x code)
- 2026-11-25 23:28 row 3 calibration done (anchor 1122.23, upstream 1344.40); solver started 23:26
- 2026-11-26 00:59 row 3 done: final 40.62 vs anchor 1122.23, reward 0, audit clean; row 4 started
- 2026-11-26 01:05 kfc row 1 prepared but not queued: 6 h agent limit does not fit before 05:00
- 2026-11-26 02:55 row 4 done: submissions correct ~131x but rejected by the 2x gate; final = baseline 1.01x, reward 0. Queue done; away period ended
