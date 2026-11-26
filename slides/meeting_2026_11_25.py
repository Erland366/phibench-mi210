"""Slides API requests for the 25 November 2026 meeting, inserted at the front of the deck
https://docs.google.com/presentation/d/1ViuLjCdFt1i5K9QAviP53aK7_tDCfDIR5NAlH5BI5t4 (the 19 November slides stay after them).
Style copied from the 19 November slides: title 24 pt at (320040, 228600) EMU, table at y 1216151, Arial 15 pt, bold
header, marked cells FFF27A, a 14 pt line under the table. Every number is from docs/experiment-log.md and the run files
in /vast/users/qirong.ho/erland/phibench-apptainer/runs/ (sources in each slide's comment).
Usage: python3 meeting_2026_11_25.py > requests.json
"""
import json

P = "m25"          # prefix of every object id, so nothing collides with the 19 November slides
X, W = 320040, 8503920
TITLE_Y, TABLE_Y = 228600, 1216151
YELLOW = {"red": 1, "green": 0.9490196, "blue": 0.47843137}

SLIDES = [
    dict(id="sum",
         # runs/2026-11-23-qwen-m256-1/agent_result.json, runs/2026-11-24-qwen-m256-2/final, runs/2026-11-24-qwen-code-m256-3/final
         title="Qwen3.8-27B as the solver scored 1.14× on m256 with both harnesses, below the reference 1.20×",
         cols=["Run", "Harness", "Ended by", "Submissions", "Final speedup", "Reward"],
         rows=[["1", "mini-swe-agent", "a model reply over 600 s, after 2 h 33 min", "0", "not scored (setup error)", "–"],
               ["2", "mini-swe-agent", "context full, after 60 min", "1", "1.1428", "0"],
               ["3", "Qwen Code", "the 4-hour limit", "1", "1.1374", "0"]],
         widths=[0.6, 1.5, 2.6, 1.2, 1.9, 0.9], mark=[(2, 4), (3, 4)],
         line="One run each, on mi210-jais3-moe-m256. Final speedup on an idle node. The reference patch gives 1.2014."),
    dict(id="setup",
         # runs/2026-11-23-check, runs/2026-11-23-qwen-interference, runs/2026-11-24-decode-backends, README calibration
         title="The task scores as calibrated in Apptainer, also next to the model server on GPUs 4–7",
         cols=["Setup", "Unchanged repository", "Reference patch"],
         rows=[["Enroot, calibration (5 runs of the reference)", "1.0075", "1.1745–1.2051"],
               ["Apptainer, node idle", "1.0056", "1.2115"],
               ["Apptainer, model server on GPUs 4–7", "0.9945", "1.2155"]],
         widths=[4.3, 2.3, 2.7], mark=[(2, 2), (3, 2)],
         line="Qwen3.8-27B (BF16) on vLLM 0.32.0, TP4 on GPUs 4–7, context 131,072 tokens, TRITON_ATTN "
              "(41 tokens/s at 50k context, the default ROCM_ATTN 5). Apptainer, because we have no Docker access on this node."),
    dict(id="mini",
         # runs/2026-11-24-qwen-m256-2/trajectory.json: assistant replies, minutes after start, prompt tokens
         title="mini-swe-agent stopped after 60 min because the earlier thinking filled the 131k context",
         cols=["Model reply", "Minutes after start", "Prompt tokens"],
         rows=[["1", "0.0", "1,461"], ["10", "0.4", "26,655"], ["20", "8.2", "45,234"], ["30", "23.4", "77,841"],
               ["40", "44.9", "119,053"], ["48 (last)", "59.0", "128,825"]],
         widths=[3.1, 3.1, 3.1], mark=[(6, 2)],
         line="The chat template sends every earlier thinking block back to the model. At the end, thinking was "
              "184,580 of 384,805 characters of the conversation."),
    dict(id="qc",
         # runs/2026-11-24-qwen-code-m256-3: logs/agent/openai (190 requests), stream.jsonl, final/verifier/reward.json
         title="Qwen Code compacted 6 times and kept the prompt under 99k tokens for the full 4 hours",
         cols=["Run 3", "Value"],
         rows=[["Session", "4 h, stopped by the time limit"],
               ["Model requests", "190, 6 of them compactions"],
               ["Prompt at each compaction (tokens)", "90,397–98,876"],
               ["Submissions", "1"],
               ["Change", "precompile() in moe_w8a16.py (56 lines), a new 256-token entry in dec_configs.json (22 lines)"],
               ["Final speedup, idle node", "1.1374, reward 0"]],
         widths=[3.4, 5.9], mark=[(3, 1)],
         line=None),
    dict(id="shared",
         # runs/2026-11-24-qwen-code-m256-3: logs/loop/state.jsonl, final-busy/, final/; wm_vendor_mismatch .pool/runners/main_240_g0-3.log
         title="Run 3's in-session 1.44× was measured on a GPU shared with another project's jobs",
         cols=["Score", "Time (UTC)", "GPU", "Other jobs", "Baseline (ms)", "Speedup"],
         rows=[["In session (submit.sh)", "24 Nov 14:04", "0", "Wan2.2 queue on GPUs 0–3", "–", "1.4382"],
               ["Final score, first try", "24 Nov 14:17", "0", "Wan2.2 jobs started at 14:16", "0.134", "1.7050, not counted"],
               ["Final score, again", "25 Nov 06:02", "4", "none, node idle", "0.0466", "1.1374"]],
         widths=[2.1, 1.4, 0.6, 2.4, 1.1, 1.7], mark=[(3, 5)],
         line="The node is split, GPUs 0–3 for another project and 4–7 for this one. Run 3 put the solver on GPU 0 by "
              "mistake. Runs 1 and 2 ran before those jobs."),
    dict(id="grader",
         # runs/2026-11-24-qwen-code-m256-3/stream.jsonl, tool calls
         title="In Apptainer the solver can read the hidden grader, and run 3 did",
         cols=["What it read", "How"],
         rows=[["/opt/loop/private/tests/ (file list, line counts)", "shell, ls and wc"],
               ["test.sh, with the reward formula", "read_file"],
               ["grade.py", "read_file"],
               ["grade_config.json", "read_file"],
               ["/opt/loop/score_engine.sh (first 100 lines)", "shell, head"],
               ["/opt/verifier-correctness-manifest.json (holds the reference speedup)", "shell, cat"]],
         widths=[6.6, 2.7], mark=[],
         line="The solver is root in its sandbox, as upstream's solver is root in Docker, so 0700 hides nothing. No "
              "reference patch is in the image. The held-out seeds are only in tests/, outside the sandbox."),
    dict(id="open",
         title="What is still open",
         cols=["Item", "Status or question"],
         rows=[["Repeat runs", "Every number is one run. How much does a run vary?"],
               ["Harness", "Qwen Code, with compaction. mini-swe-agent stops at 60 min."],
               ["Hidden grader", "Readable in Apptainer. Hide it, or accept it as upstream does?"],
               ["Node for timing", "Timing needs every GPU idle, and the node is shared with other projects."],
               ["Other tasks", "routed-moe-w8a16 image built, not run. m1024 is too easy."]],
         widths=[2.4, 6.9], mark=[],
         line=None),
]

EMU_IN = 914400
req = []
# date slide, as on 19 November: the centred title only
req += [{"createSlide": {"objectId": f"{P}_date", "insertionIndex": 0, "slideLayoutReference": {"layoutId": "p17"},
         "placeholderIdMappings": [{"layoutPlaceholder": {"type": "CENTERED_TITLE", "index": 0}, "objectId": f"{P}_date_t"},
                                   {"layoutPlaceholder": {"type": "SUBTITLE", "index": 0}, "objectId": f"{P}_date_s"}]}},
        {"deleteObject": {"objectId": f"{P}_date_s"}},
        {"insertText": {"objectId": f"{P}_date_t", "text": "Meeting, 25 November 2026"}}]

for k, d in enumerate(SLIDES, start=1):
    sid, tid, tab, lid = f"{P}_{d['id']}", f"{P}_{d['id']}_t", f"{P}_{d['id']}_tab", f"{P}_{d['id']}_line"
    req += [{"createSlide": {"objectId": sid, "insertionIndex": k, "slideLayoutReference": {"layoutId": "p18"},
             "placeholderIdMappings": [{"layoutPlaceholder": {"type": "TITLE", "index": 0}, "objectId": tid}]}},
            {"insertText": {"objectId": tid, "text": d["title"]}},
            {"updatePageElementTransform": {"objectId": tid, "applyMode": "ABSOLUTE", "transform": {
                "scaleX": 2.83464, "scaleY": 0.283464, "shearX": 0, "shearY": 0, "translateX": X, "translateY": TITLE_Y, "unit": "EMU"}}},
            {"updateTextStyle": {"objectId": tid, "style": {"fontSize": {"magnitude": 24, "unit": "PT"}}, "fields": "fontSize"}}]
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
                        "style": {"fontFamily": "Arial", "fontSize": {"magnitude": 15, "unit": "PT"}, "bold": i == 0},
                        "fields": "fontFamily,fontSize,bold"}})
    for (i, j) in d["mark"]:
        req.append({"updateTableCellProperties": {"objectId": tab,
                    "tableRange": {"location": {"rowIndex": i, "columnIndex": j}, "rowSpan": 1, "columnSpan": 1},
                    "tableCellProperties": {"tableCellBackgroundFill": {"solidFill": {"color": {"rgbColor": YELLOW}}}},
                    "fields": "tableCellBackgroundFill"}})
    if d["line"]:
        req += [{"createShape": {"objectId": lid, "shapeType": "TEXT_BOX", "elementProperties": {
                    "pageObjectId": sid, "size": {"width": {"magnitude": W, "unit": "EMU"}, "height": {"magnitude": 538000, "unit": "EMU"}},
                    "transform": {"scaleX": 1, "scaleY": 1, "translateX": X, "translateY": 4308348, "unit": "EMU"}}}},
                {"insertText": {"objectId": lid, "text": d["line"]}},
                {"updateTextStyle": {"objectId": lid, "textRange": {"type": "ALL"},
                 "style": {"fontFamily": "Arial", "fontSize": {"magnitude": 14, "unit": "PT"}}, "fields": "fontFamily,fontSize"}}]

print(json.dumps(req))
