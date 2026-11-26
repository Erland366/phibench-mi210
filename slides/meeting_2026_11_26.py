"""Slides API requests for a new deck for the 26 November 2026 meeting (RSI Project, Phi-Bench CPU tasks):
https://docs.google.com/presentation/d/1Yi-ySoCePOsV7Ygh1URc3YVmD6avtt_V4EcMFzNIv0I
Style copied from the 19 and 25 November slides of 1ViuLjCdFt1i5K9QAviP53aK7_tDCfDIR5NAlH5BI5t4 (title 24 pt, Arial 15 pt
table with bold header, marked cells FFF27A, a 14 pt line under the table, "Extra slides" divider).
Layouts of the new deck: p2 title slide (the existing slide "p"), p3 section header, p6 title only.
Every number is from docs/experiment-log.md (25-26 November) and the run files in
/vast/users/qirong.ho/erland/phibench-apptainer/runs/ (sources in each slide's comment and notes).
Usage: python3 meeting_2026_11_26.py slides > requests.json ; python3 meeting_2026_11_26.py notes NOTES_IDS.json > notes.json
"""
import json
import sys

P = "m26"
X, W = 320040, 8503920
TITLE_Y, TABLE_Y, LINE_Y = 228600, 1216151, 4308348
YELLOW = {"red": 1, "green": 0.9490196, "blue": 0.47843137}
SRC = "Source: docs/experiment-log.md, entries of 25-26 November; ~/erland/phibench-apptainer/runs/"

DECK_TITLE = "Qwen3.8-27B as the solver on five Phi-Bench CPU tasks"
DATE = "Meeting, 26 November 2026"

MAIN = [
    dict(id="sum",
         # final/verifier/reward.json of runs/2026-11-25-qwen-code-{wro-pq,wro-q5k,causal-delivery-vclock,
         # memory-accounting-sim,secure-agg-shamir-committee}-1; reference here = median of 5 oracle runs in
         # runs/2026-11-25-*-calibration; upstream = tests/verifier-correctness-manifest.json of each package
         title="Qwen3.8-27B beat the reference on 1 of 5 CPU tasks, causal delivery at 2.05× the reference",
         cols=["Task", "Reference speedup here (upstream)", "Submissions", "Final speedup", "Reward"],
         rows=[["vLLM priority queue (Python)", "322.07 (312.0)", "3", "317.51", "0"],
               ["llama.cpp Q5_K dot product (C)", "2.60 (11.53)", "0", "1.38", "0"],
               ["Causal delivery (Python)", "175.96 (187.50)", "5", "360.87", "0.139"],
               ["Memory accounting (Python)", "1122.23 (1344.40)", "2", "40.62", "0"],
               ["Secure aggregation (Python)", "50.27 (57.66)", "2", "1.01", "0"]],
         widths=[2.95, 2.0, 1.4, 1.2, 0.95], mark=[(3, 3), (3, 4)],
         line="Speedup is the time of the unchanged code divided by the time of the solver's code. Reward is 0 up to "
              "the reference speedup. One run per task.",
         notes=["Be careful. One run per task, so we do not know how much a run varies.",
                "Be careful. The upstream references were measured on Intel Sapphire Rapids. We measured ours on this "
                "node's AMD EPYC 9334, 5 runs each, and use the median.",
                "If somebody asks why reward is only 0.139 at 2.05x: reward is ln(speedup/ref)/ln(ref), so 1.0 needs "
                "the square of the reference speedup (about 31,000x here).",
                "Source: final/verifier/reward.json of each runs/2026-11-25-qwen-code-*-1; calibration in "
                "runs/2026-11-25-*-calibration."]),
    dict(id="harn",
         # m256 runs: runs/2026-11-24-qwen-m256-2 (mini-swe-agent), runs/2026-11-24-qwen-code-m256-3 (Qwen Code, final =
         # rescore on an idle GPU 4); 25 November deck slides m25_sum and m25_mini
         title="On m256, Qwen Code ran its full 4 h and mini-swe-agent stopped at 60 min, at the same speedup",
         cols=["Harness", "Session", "Ended by", "Submissions", "Final speedup", "Reward"],
         rows=[["mini-swe-agent 2.4.6", "60 min", "Context full (128,825 of 131,072 tokens)", "1", "1.1428", "0"],
               ["Qwen Code 0.28.4", "4 h", "The 4 h time limit", "1", "1.1374", "0"]],
         widths=[1.8, 1.0, 2.15, 1.45, 1.15, 0.95], mark=[(1, 1), (2, 1)],
         line="One run each on mi210-jais3-moe-m256 with Qwen3.8-27B, final speedup on an idle node. The reference patch "
              "gives 1.2014, so both score 0. The five CPU tasks ran with Qwen Code.",
         notes=["Be careful. One run each, and the runs differ in sampling too (temperature 0.6 against the model's "
                "default 1.0), so the equal score says nothing about which harness solves better.",
                "Be careful. The Qwen Code run read the in-session grader files; under Apptainer the solver is root in "
                "its sandbox, so their 0700 mode hides nothing (25 November slides).",
                "Source: docs/experiment-log.md, 24-25 November; runs/2026-11-24-qwen-m256-2, "
                "runs/2026-11-24-qwen-code-m256-3 (final score = the rescore of 25 November)."]),
    dict(id="mini",
         # runs/2026-11-24-qwen-m256-2/trajectory.json: assistant replies, minutes after start, prompt tokens (25 Nov slide m25_mini)
         title="mini-swe-agent stopped after 60 min because the earlier thinking filled the 131k context",
         cols=["Model reply", "Minutes after start", "Prompt tokens"],
         rows=[["1", "0.0", "1,461"], ["10", "0.4", "26,655"], ["20", "8.2", "45,234"], ["30", "23.4", "77,841"],
               ["40", "44.9", "119,053"], ["48 (last)", "59.0", "128,825"]],
         widths=[3.1, 3.1, 3.1], mark=[(6, 2)],
         line="The chat template sends every earlier thinking block back to the model. At the end, thinking was "
              "184,580 of 384,805 characters of the conversation.",
         notes=["Be careful. mini-swe-agent has no context management at all: it sends the whole history every turn, "
                "so a session of this model ends when the history reaches 131k tokens.",
                "If somebody asks why Qwen Code: we wanted automatic compaction of the context, which mini-swe-agent "
                "does not have; Qwen Code 0.28.4 has it built in.",
                "Source: runs/2026-11-24-qwen-m256-2/trajectory.json."]),
    dict(id="comp",
         # runs/2026-11-24-qwen-code-compaction-test-4, -5 and runs/2026-11-24-qwen-code-m256-3: settings.json
         # (contextWindowSize) and logs/agent/openai/ (prompt_tokens of the summary request and of the request after it)
         title="Qwen Code summarizes the conversation near the limit and goes on",
         cols=["Run", "Context window (tokens)", "Compactions", "Prompt before (tokens)", "Prompt after (tokens)", "How it ended"],
         rows=[["Test, large file reads", "32,768", "1", "–", "–", "One turn was larger than the window"],
               ["Test, one file per turn", "65,536", "5", "31,679–35,605", "about 20,000–27,000", "Finished, 19 turns"],
               ["m256 solver run", "131,072", "6", "90,397–98,876", "17,060–31,419", "The 4 h time limit"]],
         widths=[1.55, 1.15, 1.3, 1.45, 1.55, 1.5], mark=[(3, 3), (3, 4)], size=14,
         line="Each summary took 3–6 min in the 65k test. The five CPU task sessions compacted too, 8 times in all "
              "(1 or 2 each).",  # 26 Nov: corrected; first version said none (wrong check, see experiment log)
         notes=["Be careful. The 32k test failed because one turn of six parallel file reads was larger than the whole "
                "window; it is an effect of the small window, not of compaction.",
                "Be careful. The model writes the summary itself, so details of earlier turns can be lost.",
                "Source: runs/2026-11-24-qwen-code-compaction-test-4, -5 and runs/2026-11-24-qwen-code-m256-3, "
                "settings.json and logs/agent/openai/."]),
    dict(id="gate",
         # logs/loop/state.jsonl and logs/loop/submissions/N/verifier_state.json of the causal-delivery and secure-agg runs;
         # environment/loop/private/tests/compute_reward.py:49 (vs_oracle > 2.0), sanitize_feedback.py:48-49 and 65
         title="The in-session grader rejected 4 correct submissions for being over 2× the reference",
         cols=["Task", "Submission", "In-session speedup", "Rejected above", "Correctness"],
         rows=[["Causal delivery", "1", "376.27", "351.91", "passed"],
               ["Causal delivery", "5", "430.74", "351.91", "passed"],
               ["Secure aggregation", "1", "132.79", "100.54", "passed"],
               ["Secure aggregation", "2", "130.59", "100.54", "passed"]],
         widths=[2.6, 1.5, 2.1, 1.8, 1.6], mark=[(1, 2), (2, 2), (3, 2), (4, 2)],
         line="The solver is told the check \"did not complete\" and to look for an early crash. Upstream removed this "
              "limit from the final verifier, not from the in-session one.",
         notes=["Be careful. With no accepted submission, secure aggregation was scored on the unchanged code (1.01), "
                "because finalize puts back only an accepted submission.",
                "Be careful. The rejected code was not scored by the final verifier, so 430.74 and 132.79 are "
                "in-session numbers only.",
                "If somebody asks whether the 2.05x is real: an independent review found a general algorithm with no "
                "special cases, and 120k random cases gave the same output as the unchanged code.",
                "Source: tests/compute_reward.py of the secure-agg package says the 2x guard was removed because it "
                "only ever gave reward 0 to correct large speedups; the in-session copy in "
                "environment/loop/private/tests/compute_reward.py:49 still has it."]),
    dict(id="end",
         # stream.jsonl and logs/agent/openai/ (largest prompt_tokens) of each run; logs/loop/finalize.json
         title="Four of the five sessions ran to the 90-min limit",
         cols=["Task", "Submissions", "How the session ended", "Largest prompt (tokens)"],
         rows=[["Priority queue", "3", "Stopped itself at 70 min, tied with the reference", "98,684"],
               ["llama.cpp Q5_K", "0", "Time limit, still fixing its own test program", "98,830"],
               ["Causal delivery", "5", "Time limit", "98,383"],
               ["Memory accounting", "2", "Time limit", "97,776"],
               ["Secure aggregation", "2", "Time limit, after looking for the reported crash", "92,549"]],
         widths=[1.9, 1.4, 3.6, 1.6], mark=[(2, 1)], size=14, line_y=4560000,
         line="The context holds 131,072 tokens; each session compacted 1 or 2 times, at prompts of "
              "85,314–98,830 tokens. No session read grader files, tests or reference patches, or used the network.",
         notes=["Be careful. On llama.cpp the reference is llama.cpp's own public AVX2 code, which the model may know.",
                "If somebody asks why llama.cpp got no submission: the solver wrote its own test program in /tmp and "
                "spent the session on its link and compile errors; its unsubmitted code scored 1.38.",
                "Source: runs/2026-11-25-qwen-code-*-1/stream.jsonl (tool calls) and logs/agent/openai/ (prompt tokens)."]),
    dict(id="time",
         # docs/experiment-log.md "Verdict and time taken" lines; job logs runs/2026-11-25-row-*.job.log
         title="Each task took about 2 h, 90 min of it the solver session",
         cols=["Task", "Download", "Image build", "Calibration (8 runs)", "Solver", "Final score", "Whole task"],
         rows=[["Priority queue", "1 min", "14 min", "19 min", "70 min", "3 min", "4 h 30 min"],
               ["llama.cpp Q5_K", "10 min", "2 min", "23 min", "90 min", "4 min", "2 h 29 min"],
               ["Causal delivery", "15 s", "5 min", "22 min", "90 min", "3 min", "2 h 10 min"],
               ["Memory accounting", "15 s", "4 min", "22 min", "90 min", "3 min", "1 h 57 min"],
               ["Secure aggregation", "15 s", "25 s", "20 min", "90 min", "3 min", "1 h 56 min"]],
         widths=[1.9, 1.15, 1.05, 1.3, 0.95, 1.0, 1.15], mark=[],
         line="The priority queue waited about 2 h 30 min for a quiet node. The last three images were built before "
              "their tasks started and are not in the whole-task time.",
         notes=["Be careful. Causal delivery's 2 h 10 min includes a 7 min false start and about 5 min to fix the "
                "Node.js start (extra slide \"Three setup problems overnight\").",
                "Be careful. Calibration and final scoring wait for a quiet node; on a busy node they take longer.",
                SRC]),
    dict(id="open",
         title="What is still open",
         cols=["Item", "Status or question"],
         rows=[["2× limit in session", "Report it upstream? Score the 4 rejected submissions with the final verifier, as extra numbers?"],
               ["Repeat runs", "Every number is one run."],
               ["llama.cpp reference", "2.60 here against 11.53 on Intel, with the same image and compiler."],
               ["Next task", "kfc arch-tiling-dram-traffic-min is downloaded. Its solver limit is 6 h."],
               ["Time per task", "About 2 h, 90 min of it the solver session."]],
         widths=[2.6, 6.7], mark=[], line=None,
         notes=["Be careful. Scoring the rejected submissions again is about 3 min of CPU each; those numbers would be "
                "extra, not replace the recorded scores.",
                "If somebody asks about kfc arch-tiling: 1 submission, 6 h solver limit, and it scores DRAM traffic, a "
                "count, so its reference should not depend on the CPU.",
                SRC]),
]

EXTRA = [
    dict(id="calib",
         # runs/2026-11-25-*-calibration/*/verifier/reward.json
         title="On every task the 5 reference runs lie within 4% of their median",
         cols=["Task", "Unchanged code", "Reference, 5 runs", "Median (used)", "Upstream"],
         rows=[["Priority queue", "0.99", "316.27–328.05", "322.07", "312.0"],
               ["llama.cpp Q5_K", "1.00", "2.589–2.604", "2.60", "11.53"],
               ["Causal delivery", "1.01", "175.44–177.09", "175.96", "187.50"],
               ["Memory accounting", "1.00", "1081.68–1161.85", "1122.23", "1344.40"],
               ["Secure aggregation", "1.00", "49.75–51.07", "50.27", "57.66"]],
         widths=[2.4, 1.6, 2.2, 1.6, 1.5], mark=[],
         line="Upstream measured on Intel Sapphire Rapids, we on AMD EPYC 9334. The negative patch scores 0 on every task.",
         notes=["Be careful. On llama.cpp the same image and compiler (gcc 12.2, AVX2 fixed) give 2.60 here, so the "
                "scalar start is much closer to the AVX2 code on this CPU.",
                "Be careful. The upstream solution script (solve.sh) lands within 3% of the median on every task (causal delivery 171.07).",
                "Source: runs/2026-11-25-*-calibration/{noop,negative,oracle-1..5,solve}-1/verifier/reward.json."]),
    dict(id="fix",
         # docs/experiment-log.md, entries of 25 November; runs/*.collided, runs/*.agent-not-started
         title="Three setup problems overnight, each fixed before a counted run",
         cols=["Problem", "Fix"],
         rows=[["The anti-leak check of the image build saw the host's /tmp", "An empty directory bound at /tmp during the build"],
               ["Two copies of the queue started the same calibration", "Both stopped, outputs kept apart, the queue now runs only once"],
               ["Node.js did not start in the slim Python images (no libatomic)", "A copy of Node that loads libatomic from its own folder"]],
         widths=[4.8, 4.5], mark=[], line=None,
         notes=["Be careful. The collided calibration runs (noop and negative of causal delivery) are kept but not counted.",
                "Source: docs/experiment-log.md; runs/2026-11-25-causal-delivery-vclock-calibration.collided/, "
                "runs/2026-11-25-qwen-code-causal-delivery-vclock-1.agent-not-started/."]),
    dict(id="decode",
         # runs/2026-11-24-decode-backends/ (docs/experiment-log.md, 24 November)
         title="TRITON_ATTN decodes 8× faster than the default attention at 50k context on MI210",
         cols=["Attention backend in vLLM 0.32.0", "Decode at 6k context (tokens/s)", "Decode at 50k context (tokens/s)"],
         rows=[["ROCM_ATTN (the default)", "24.1", "5.1"],
               ["TRITON_ATTN (used in every run after the first)", "49.8", "40.8"],
               ["ROCM_AITER_UNIFIED_ATTN", "does not start on gfx90a", "–"]],
         widths=[4.1, 2.6, 2.6], mark=[(2, 2)],
         line="Qwen3.8-27B, TP4 on GPUs 4–7, one request, 256 tokens. The first mini-swe-agent run used the default "
              "and stopped after 2 h 33 min, when one reply ran past the 600 s client timeout.",
         notes=["Be careful. One request at a time; with several requests the numbers would differ.",
                "Source: runs/2026-11-24-decode-backends/; the first run is runs/2026-11-23-qwen-m256-1 (invalid, kept)."]),
    dict(id="qcfix",
         # docs/experiment-log.md, 24 November; runs/2026-11-24-qwen-code-compaction-test{,-2,-3}
         title="Three fixes made Qwen Code run inside the task container",
         cols=["Problem", "Fix"],
         rows=[["Apptainer cannot make the /opt/agent mount point in a writable sandbox", "The directory is made in the sandbox first"],
               ["Apptainer refuses to set HOME", "QWEN_HOME instead"],
               ["Every model call failed: Apptainer passes the host's proxy, and Qwen Code sent local calls through it",
                "Proxy variables unset in the container, which also leaves it no route to the internet, as the task requires"]],
         widths=[4.8, 4.5], mark=[], line=None,
         notes=["Be careful. The two mini-swe-agent runs had the proxy; neither trajectory contains a network command.",
                "Source: docs/experiment-log.md, 24 November; scripts/run_qwen_code.sh."]),
]

ALL = MAIN + EXTRA
req = []


def table_slide(d, at=None):
    sid, tid, tab, lid = f"{P}_{d['id']}", f"{P}_{d['id']}_t", f"{P}_{d['id']}_tab", f"{P}_{d['id']}_line"
    cs = {"objectId": sid, "slideLayoutReference": {"layoutId": "p6"}}
    if at is not None:
        cs["insertionIndex"] = at
    req.extend([{"createSlide": {**cs,
                 "placeholderIdMappings": [{"layoutPlaceholder": {"type": "TITLE", "index": 0}, "objectId": tid}]}},
                {"insertText": {"objectId": tid, "text": d["title"]}},
                {"updatePageElementTransform": {"objectId": tid, "applyMode": "ABSOLUTE", "transform": {
                    "scaleX": 2.83464, "scaleY": 0.283464, "shearX": 0, "shearY": 0, "translateX": X, "translateY": TITLE_Y, "unit": "EMU"}}},
                {"updateTextStyle": {"objectId": tid, "style": {"fontSize": {"magnitude": 24, "unit": "PT"}}, "fields": "fontSize"}}])
    nr, nc = len(d["rows"]) + 1, len(d["cols"])
    req.append({"createTable": {"objectId": tab, "rows": nr, "columns": nc, "elementProperties": {
        "pageObjectId": sid, "size": {"width": {"magnitude": W, "unit": "EMU"}, "height": {"magnitude": 320800 * nr, "unit": "EMU"}},
        "transform": {"scaleX": 1, "scaleY": 1, "translateX": X, "translateY": TABLE_Y, "unit": "EMU"}}}})
    for j, w in enumerate(d["widths"]):
        req.append({"updateTableColumnProperties": {"objectId": tab, "columnIndices": [j],
                    "tableColumnProperties": {"columnWidth": {"magnitude": round(w / sum(d["widths"]) * W), "unit": "EMU"}},
                    "fields": "columnWidth"}})
    for i, row in enumerate([d["cols"]] + d["rows"]):
        for j, v in enumerate(row):
            cell = {"rowIndex": i, "columnIndex": j}
            req.append({"insertText": {"objectId": tab, "cellLocation": cell, "text": v}})
            req.append({"updateTextStyle": {"objectId": tab, "cellLocation": cell, "textRange": {"type": "ALL"},
                        "style": {"fontFamily": "Arial", "fontSize": {"magnitude": d.get("size", 15), "unit": "PT"}, "bold": i == 0},
                        "fields": "fontFamily,fontSize,bold"}})
    for (i, j) in d["mark"]:
        req.append({"updateTableCellProperties": {"objectId": tab,
                    "tableRange": {"location": {"rowIndex": i, "columnIndex": j}, "rowSpan": 1, "columnSpan": 1},
                    "tableCellProperties": {"tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": YELLOW}}}},
                    "fields": "tableCellBackgroundFill"}})
    if d["line"]:
        req.extend([{"createShape": {"objectId": lid, "shapeType": "TEXT_BOX", "elementProperties": {
                        "pageObjectId": sid, "size": {"width": {"magnitude": W, "unit": "EMU"}, "height": {"magnitude": 538000, "unit": "EMU"}},
                        "transform": {"scaleX": 1, "scaleY": 1, "translateX": X, "translateY": d.get("line_y", LINE_Y), "unit": "EMU"}}}},
                    {"insertText": {"objectId": lid, "text": d["line"]}},
                    {"updateTextStyle": {"objectId": lid, "textRange": {"type": "ALL"},
                     "style": {"fontFamily": "Arial", "fontSize": {"magnitude": 14, "unit": "PT"}}, "fields": "fontFamily,fontSize"}}])


def slides():
    # the deck's own first slide "p" (layout p2) becomes the title slide
    req.extend([{"insertText": {"objectId": "i0", "text": DECK_TITLE}},
                {"insertText": {"objectId": "i1", "text": DATE}},
                {"updateTextStyle": {"objectId": "i0", "textRange": {"type": "ALL"}, "style": {"fontSize": {"magnitude": 36, "unit": "PT"}}, "fields": "fontSize"}}])
    for d in MAIN:
        table_slide(d)
    req.extend([{"createSlide": {"objectId": f"{P}_div_extra", "slideLayoutReference": {"layoutId": "p3"},
                 "placeholderIdMappings": [{"layoutPlaceholder": {"type": "TITLE", "index": 0}, "objectId": f"{P}_div_extra_t"}]}},
                {"insertText": {"objectId": f"{P}_div_extra_t", "text": "Extra slides"}}])
    for d in EXTRA:
        table_slide(d)
    return req


def notes(ids):  # ids: {slide objectId: speakerNotesObjectId}, read from the deck after the slides exist
    for d in ALL:
        nid = ids[f"{P}_{d['id']}"]
        req.append({"insertText": {"objectId": nid, "text": "Only if needed:\n" + "\n".join("- " + n for n in d["notes"])}})
    return req


def add(ids):  # 26 Nov, 09:40: the harness slides, added to the built deck at their place in the deck order
    order = ["title"] + [d["id"] for d in MAIN] + ["div_extra"] + [d["id"] for d in EXTRA]
    for d in ALL:
        if d["id"] in ids:
            table_slide(d, at=order.index(d["id"]))
    return req


if __name__ == "__main__":
    if sys.argv[1] == "add":
        print(json.dumps(add(sys.argv[2].split(",")), separators=(",", ":"), ensure_ascii=False)); sys.exit()
    out = slides() if sys.argv[1] == "slides" else notes(json.load(open(sys.argv[2])))
    print(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
