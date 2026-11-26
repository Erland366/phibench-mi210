"""Tables for the three trace slides of the 26 November meeting deck, in the style of the deck's other table slides
(table at 0.35 in, 1.33 in; Arial 15 pt; bold header; yellow FFF27A on the cells the title is about).

Sources:
  m26_sop    docs/sop.md (the SOP, same text as the Google Doc), commit f31c5fd
  m26_trfmt  docs/trace-records.md, docs/adr/0008-0010, src/phibench_mi210/traces.py
  m26_trsheet specs/traces-qwen-runs-2026-11.json, the causal-delivery entry (slide duplicated from m26_trfmt)
  m26_trset  Erland/rsi-traces index.jsonl at HF commit 0adb60d9 (15 records)
"""
import json, sys

EMU = 914400
TABLES = {
    'm26_setup': dict(
        cols=['Part', 'What we used'], widths=[2.2, 7.1],
        rows=[['Model', 'Qwen3.8-27B, bf16'],
              ['Serving', 'vLLM 0.32.0 on 4 MI210 (tensor parallel 4), 131k-token context'],
              ['Agent harness', 'mini-swe-agent 2.4.6 at first, then Qwen Code 0.28.4'],
              ['Tasks', 'm256 (our MI210 row) and 5 upstream Φ-Bench CPU tasks'],
              ['Sandbox', "Apptainer container from the task's image, no network"],
              ['Session', '90 min and up to 16 submissions per CPU task; 4 h on m256'],
              ['Scoring', "The task's own verifier; CPU tasks on an AMD EPYC 9334"]],
        mark=set()),
    'm26_sop': dict(
        cols=['Step', 'What the contributor does'], widths=[3.3, 6.0],
        rows=[['1. Pick the work', 'Accepted work that a program can check in minutes'],
              ['2. Fill in the task form', 'Two commits, editable files, the check, the hardware'],
              ['3. Write instruction and tests', 'Written by hand, with no hint of the solution'],
              ['4. Build package and image', 'Φ-Bench layout, no reference solution in the image'],
              ['5. Run the checks', 'Reference at least 1.1× faster (median of 5 runs)'],
              ["6. Store every run's trace", 'Own session first, then every later run on the task'],
              ['7. Submit for review', 'First tasks of each contributor are reviewed']],
        mark=set()),
    'm26_trfmt': dict(
        cols=['Source', 'Conversation kept', 'Label from'], widths=[2.6, 3.6, 3.1],
        rows=[['Own Claude Code or Codex session', 'Session files, thinking removed', 'Its result is the checked reference, so successful'],
              ['GEAK run (Claude agents)', "Every agent's session, thinking removed", 'Round check, correct and at least 1.1×'],
              ['Qwen Code run', 'Every model call as sent, with reasoning', 'Task verifier, reward above 0'],
              ['mini-swe-agent run', 'Its trajectory, with reasoning', 'Task verifier, reward above 0']],
        mark={(r, 2) for r in range(4)}),
    'm26_trsheet': dict(
        cols=['The sheet states', 'Causal-delivery Qwen run', 'Checked against the run'], widths=[2.9, 3.5, 2.9],
        rows=[['Run folder and source', 'open_model_run', 'The folder exists'],
              ['Task and its kind', 'causal delivery, performance', '—'],
              ['Agent and model', 'Qwen Code 0.28.4, Qwen3.8-27B', "Version in the run's log"],
              ['Hardware used', '4 MI210; AMD EPYC 9334', 'The serving log exists'],
              ['Anchor and its evidence', '175.955891', 'Anchor of the final score'],
              ['Prompt against the task', 'instruction', 'Instruction, for a model run'],
              ['Invalid reason, timing check', 'none; not checked yet', '—']],
        mark=set()),
    'm26_trset': dict(
        cols=['Source', 'Runs', 'Successful', 'Unsuccessful', 'Invalid'], widths=[3.3, 1.5, 1.5, 1.5, 1.5],
        rows=[['GEAK runs (Claude agents)', '7', '3', '4', '0'],
              ['Qwen Code runs', '6', '1', '5', '0'],
              ['mini-swe-agent runs', '2', '0', '1', '1'],
              ['All', '15', '4', '10', '1']],
        mark={(r, 2) for r in range(4)}),
}
LINE_Y = {'m26_setup': 5.0, 'm26_sop': 4.45, 'm26_trfmt': 4.55, 'm26_trsheet': 5.0, 'm26_trset': 3.35}
YELLOW = {'red': 1.0, 'green': 0xF2 / 255, 'blue': 0x7A / 255}

reqs = []
for slide, t in TABLES.items():
    tab = slide + '_tab'
    data = [t['cols']] + t['rows']
    R, C = len(data), len(t['cols'])
    assert abs(sum(t['widths']) - 9.3) < 1e-6
    reqs.append({'createTable': {'objectId': tab, 'rows': R, 'columns': C, 'elementProperties': {
        'pageObjectId': slide,
        'size': {'width': {'magnitude': int(9.3 * EMU), 'unit': 'EMU'}, 'height': {'magnitude': int(R * 0.35 * EMU), 'unit': 'EMU'}},
        'transform': {'scaleX': 1, 'scaleY': 1, 'translateX': 320040, 'translateY': 1216151, 'unit': 'EMU'}}}})
    for c, w in enumerate(t['widths']):
        reqs.append({'updateTableColumnProperties': {'objectId': tab, 'columnIndices': [c],
                     'tableColumnProperties': {'columnWidth': {'magnitude': int(w * EMU), 'unit': 'EMU'}},
                     'fields': 'columnWidth'}})
    for r, row in enumerate(data):
        for c, v in enumerate(row):
            loc = {'rowIndex': r, 'columnIndex': c}
            reqs.append({'insertText': {'objectId': tab, 'cellLocation': loc, 'text': v}})
            reqs.append({'updateTextStyle': {'objectId': tab, 'cellLocation': loc, 'textRange': {'type': 'ALL'},
                         'style': {'fontFamily': 'Arial', 'fontSize': {'magnitude': 15, 'unit': 'PT'}, 'bold': r == 0},
                         'fields': 'fontFamily,fontSize,bold'}})
    for r, c in sorted(t['mark']):
        reqs.append({'updateTableCellProperties': {'objectId': tab,
                     'tableRange': {'location': {'rowIndex': r + 1, 'columnIndex': c}, 'rowSpan': 1, 'columnSpan': 1},
                     'tableCellProperties': {'tableCellBackgroundFill': {'solidFill': {'color': {'rgbColor': YELLOW}}}},
                     'fields': 'tableCellBackgroundFill.solidFill.color'}})
    reqs.append({'updatePageElementTransform': {'objectId': slide + '_line', 'applyMode': 'ABSOLUTE', 'transform': {
        'scaleX': 1, 'scaleY': 1, 'translateX': 320040, 'translateY': int(LINE_Y[slide] * EMU), 'unit': 'EMU'}}})
json.dump(reqs, open(sys.argv[1], 'w'), ensure_ascii=False)
print(len(reqs), 'requests')
