"""The Claude Code and GEAK readers, on a small GEAK run folder shaped like jais3-moe-v2-v032-01."""
import json

import pytest

from phibench_mi210 import traces


def jl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj))


AGENT_LINES = [
    {'type': 'user', 'message': {'role': 'user', 'content': 'You are the engineer.'}, 'timestamp': 't0'},
    {'type': 'assistant', 'message': {'id': 'm1', 'role': 'assistant', 'model': 'claude-opus-5-5',
                                      'content': [{'type': 'thinking', 'thinking': '', 'signature': 'sig'}]}},
    {'type': 'assistant', 'message': {'id': 'm1', 'role': 'assistant', 'model': 'claude-opus-5-5',
                                      'content': [{'type': 'text', 'text': 'Reading the kernel.'}]}},
    {'type': 'assistant', 'message': {'id': 'm1', 'role': 'assistant', 'model': 'claude-opus-5-5',
                                      'content': [{'type': 'tool_use', 'id': 'tu1', 'name': 'Read',
                                                   'input': {'file_path': 'k.py'}}]}},
    {'type': 'attachment', 'attachment': {'kind': 'reminder'}},
    {'type': 'user', 'message': {'role': 'user', 'content': [
        {'type': 'tool_result', 'tool_use_id': 'tu1', 'content': 'def k(): ...'}]}},
    {'type': 'assistant', 'message': {'id': 'm2', 'role': 'assistant', 'model': 'claude-opus-5-5',
                                      'content': [{'type': 'thinking', 'thinking': 'It looks finished.',
                                                   'signature': 'sig2'},
                                                  {'type': 'text', 'text': 'Done.'}]}},
]


def test_claude_code_messages_join_split_replies_and_keep_thinking(tmp_path):
    f = tmp_path / 'a.jsonl'
    jl(f, AGENT_LINES)
    rows = traces.messages_from_claude_code(f)
    assert [r['role'] for r in rows] == ['user', 'assistant', 'tool', 'assistant']
    assert rows[1]['content'] == 'Reading the kernel.' and rows[1]['tool_calls'][0]['id'] == 'tu1'
    assert json.loads(rows[1]['tool_calls'][0]['function']['arguments']) == {'file_path': 'k.py'}
    assert rows[2] == {'role': 'tool', 'tool_call_id': 'tu1', 'content': 'def k(): ...', 'is_error': False,
                       'timestamp': None}
    assert 'reasoning' not in rows[1]  # its thinking block was stored with an empty text
    assert rows[3]['reasoning'] == 'It looks finished.' and rows[3]['content'] == 'Done.'


def test_count_thinking_counts_blocks_and_those_with_text(tmp_path):
    f = tmp_path / 'a.jsonl'
    jl(f, AGENT_LINES)
    assert traces.count_thinking(f) == {'thinking_blocks': 2, 'thinking_blocks_with_text': 1}


@pytest.fixture
def geak(tmp_path):
    run = tmp_path / 'geak-agent' / 'jais3-moe-v2-v032-01'
    proj = run / 'home/.claude/projects/-proj'
    top, eng = proj / 'sess.jsonl', proj / 'sess/subagents/workflows/wf_1/agent-a1.jsonl'
    jl(top, AGENT_LINES[:1] + AGENT_LINES[-1:])
    jl(eng, AGENT_LINES)
    write(run / 'chat/01-top-level-agent.md', f'# top-level agent\n\nSource: `{top}`\n')
    write(run / 'chat/02-eng-r1_d0-compute.md', f'# eng r1_d0\n\nSource: `{eng}`\n')
    write(run / 'chat/README.md',
          '| # | Started (UTC) | Role | Messages | Tool calls | Models | File |\n|---|---|---|---|---|---|---|\n'
          '| 1 | 08:42:03 | top-level agent | 2 | 0 | claude-opus-5-5 | [01-top-level-agent.md](01-top-level-agent.md) |\n'
          '| 2 | 08:53:19 | eng r1_d0:compute | 4 | 1 | claude-opus-5-5 | [02-eng-r1_d0-compute.md](02-eng-r1_d0-compute.md) |\n')
    write(run / 'outcome.json', {'status': 'incomplete', 'seconds': 5470.3, 'transcript_models': {'claude-opus-5-5': 540}})
    write(run / 'transcript.jsonl', '{"type": "SystemMessage"}\n')
    task = run / 'exp/team_task_1/task'
    write(task / 'roadmap.md', 'plan')
    write(task / 'round_1/engineer_0/worker_result.json',
          {'engineer_id': 'r1_d0', 'specialty': 'compute', 'strategy': 'retune', 'status': 'partial',
           'speedup_weighted': 1.0149, 'measurement_valid': True})
    write(task / 'round_1/engineer_0/best_patch.diff', 'diff\n')
    write(task / 'round_1/engineer_0/workspace/big.bin', 'x' * 100)
    cand = run.parent / 'jais3-moe-v2-v032-01.current-best'
    write(cand / 'moe_w8a16.py', 'fast')
    rc = run.parent / 'jais3-moe-v2-v032-01.round-check-current'
    write(rc / 'prepare.json', {'source': {'kind': 'directory', 'candidate_dir': str(cand)}})
    write(rc / 'verdict.json', {'summary': {'verdict': 'pass', 'correctness': 'PASS', 'control_identical': True,
                                            'score_speedup': 1.1306, 'cases_failed': []}})
    write(run.parent / 'jais3-moe-v2-v032-01.round-check-current.measure.log', 'measured\n')
    prov = tmp_path / 'provenance.json'
    write(prov, {'run': {'dir': str(run)}, 'round_check': {'dir': str(rc)},
                 'row': {'task_dir': 'mi210-jais3-routed-moe-w8a16'}})
    return run, rc, prov


def geak_entry(run, rc, prov, **over):
    e = {'record_id': 'geak-moe-v2', 'source': 'geak_run', 'run_dir': str(run),
         'task': {'id': 'mi210-jais3-routed-moe-w8a16', 'bench': 'lh', 'kind': 'performance',
                  'origin': 'Erland/phibench-mi210-rows 323e515'},
         'agent': {'harness': 'geak'}, 'model': {'name': 'claude-opus-5-5'}, 'hardware': {'scored_on': 'MI210'},
         'anchor': {'ref_speedup': None}, 'prompt_vs_task': 'same problem, other prompt', 'evidence': {'kind': 'round_check', 'dir': str(rc)},
         'row': {'provenance': str(prov)}}
    e.update(over)
    return e


def load(out, rid='geak-moe-v2'):
    return json.loads((out / 'records' / rid / 'record.json').read_text())


def test_geak_run_with_its_row_is_successful(tmp_path, geak):
    out = tmp_path / 'ds'
    traces.convert({'machine': 'm', 'runs': [geak_entry(*geak)]}, out)
    d, rec = out / 'records/geak-moe-v2', load(out)
    assert rec['outcome']['label'] == 'successful' and rec['outcome']['speedup'] == 1.1306
    assert rec['outcome']['label_rule'].startswith('Round check') and rec['outcome']['reward'] is None
    chats = rec['conversation']['chats']
    assert [c['name'] for c in chats] == ['top-level agent', 'eng r1_d0:compute']
    assert [c['file'] for c in chats] == ['conversation/chats/01-top-level-agent.jsonl',
                                          'conversation/chats/02-eng-r1-d0-compute.jsonl']
    assert rec['conversation']['layout'] == 'chats-v1'
    assert rec['conversation']['thinking_dropped'] is False and rec['conversation']['reasoning_included'] is True
    assert rec['conversation']['thinking_blocks'] == 3  # 1 in the top-level agent, 2 in the engineer
    assert rec['conversation']['thinking_blocks_with_text'] == 2
    assert rec['submissions'][0]['geak_reported_speedup'] == 1.0149
    assert (d / 'harness/geak/round_1/engineer_0/best_patch.diff').is_file()
    assert not (d / 'harness/geak/round_1/engineer_0/workspace').exists()
    assert (d / 'code/final/moe_w8a16.py').read_text() == 'fast'
    assert (d / 'verifier/round-check/verdict.json').is_file()
    assert (d / 'verifier/jais3-moe-v2-v032-01.round-check-current.measure.log').is_file()
    raw = list((d / 'conversation/raw/claude').rglob('*.jsonl'))
    assert len(raw) == 2  # byte copies of the originals, thinking included
    originals = {p.name: p for p in run_files(geak[0])}
    assert all(f.read_bytes() == originals[f.name].read_bytes() for f in raw)
    assert not any(f.get('changed') for f in rec['provenance']['files'])
    assert not (d / 'conversation/messages.jsonl').exists() and not (d / 'conversation/agents').exists()
    eng = [json.loads(l) for l in (d / chats[1]['file']).read_text().splitlines()]
    assert {m['agent'] for m in eng} == {'eng r1_d0:compute'} and len(eng) == chats[1]['messages']
    assert eng[-1]['reasoning'] == 'It looks finished.'


def run_files(run):
    return list((run / 'home/.claude/projects').rglob('*.jsonl'))


def test_passed_run_without_its_row_is_refused(tmp_path, geak):
    e = geak_entry(*geak)
    del e['row']
    with pytest.raises(traces.RecordError, match='only with its row'):
        traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')


def test_row_must_name_this_run(tmp_path, geak):
    run, rc, prov = geak
    write(prov, {'run': {'dir': '/elsewhere'}, 'round_check': {'dir': str(rc)},
                 'row': {'task_dir': 'mi210-jais3-routed-moe-w8a16'}})
    with pytest.raises(traces.RecordError, match='names run'):
        traces.convert({'machine': 'm', 'runs': [geak_entry(run, rc, prov)]}, tmp_path / 'ds')


def test_slow_run_is_unsuccessful_without_a_row(tmp_path, geak):
    run, rc, prov = geak
    write(rc / 'verdict.json', {'summary': {'verdict': 'pass', 'correctness': 'PASS', 'control_identical': True,
                                            'score_speedup': 1.0938, 'cases_failed': []}})
    e = geak_entry(run, rc, prov, task={'id': 'geak:jais3-moe-w8a16-v032', 'bench': 'lh', 'kind': 'performance',
                                        'origin': 'GEAK task, no Phi-Bench row'})
    del e['row']
    traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')
    rec = load(tmp_path / 'ds')
    assert rec['outcome']['label'] == 'unsuccessful' and rec['outcome']['speedup'] == 1.0938


def test_no_candidate_is_unsuccessful(tmp_path, geak):
    run, rc, prov = geak
    write(rc / 'verdict.json', {'verdict': 'no_candidate', 'reason': 'final_patch.diff is empty'})
    e = geak_entry(run, rc, prov)
    del e['row']
    traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')
    rec = load(tmp_path / 'ds')
    assert rec['outcome']['label'] == 'unsuccessful'
    assert rec['outcome']['evidence']['verdict'] == 'no_candidate'


def test_piece_evidence_labels_by_its_qualifies_flag(tmp_path, geak):
    run, rc, prov = geak
    work = tmp_path / 'pieces/e2-m256-decode/work'
    write(work / 'verdict.json', {'summary': {'verdict': 'not_faster'}})
    write(work.parent / 'piece.patch', '+ 256 entry\n')
    scores = tmp_path / 'pieces/scores.json'
    write(scores, [{'id': 'e2-m256-decode', 'qualifies': True, 'target_speedup': 1.2157, 'target_rows': ['m256'],
                    'whole_layer_speedup': 1.0122, 'correctness': 'PASS', 'reasons': []}])
    write(prov, {'run': {'dir': str(run)}, 'round_check': {'dir': str(work)}, 'row': {'task_dir': 'mi210-jais3-moe-m256'},
                 'piece': {'id': 'e2-m256-decode'}})
    e = geak_entry(run, rc, prov, task={'id': 'mi210-jais3-moe-m256', 'bench': 'lh', 'kind': 'performance',
                                        'origin': 'Erland/phibench-mi210-rows 323e515'},
                   evidence={'kind': 'piece', 'scores': str(scores), 'piece': 'e2-m256-decode', 'work_dir': str(work)})
    whole = tmp_path / 'geak-agent/jais3-moe-v032-01.round-check-final'
    write(whole / 'verdict.json', {'summary': {'verdict': 'pass', 'correctness': 'PASS', 'score_speedup': 1.0938}})
    e['run_result'] = {'dir': str(whole), 'note': 'whole result not fast enough; this entry alone was'}
    traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')
    rec = load(tmp_path / 'ds')
    assert rec['outcome']['label'] == 'successful' and rec['outcome']['speedup'] == 1.2157
    assert rec['outcome']['run_result']['speedup'] == 1.0938
    assert (tmp_path / 'ds/records/geak-moe-v2/verifier/run-round-check/verdict.json').is_file()
    assert (tmp_path / 'ds/records/geak-moe-v2/code/reference.patch').read_text() == '+ 256 entry\n'


def test_invalid_reason_still_wins_for_geak(tmp_path, geak):
    traces.convert({'machine': 'm', 'runs': [geak_entry(*geak, invalid_reason='driver crashed')]}, tmp_path / 'ds')
    assert load(tmp_path / 'ds')['outcome']['label'] == 'invalid'


def test_prompt_is_the_top_level_agents_first_message(tmp_path, geak):
    traces.convert({'machine': 'm', 'runs': [geak_entry(*geak)]}, tmp_path / 'ds')
    d = tmp_path / 'ds/records/geak-moe-v2'
    assert (d / 'prompt.txt').read_text() == 'You are the engineer.'
    rec = load(tmp_path / 'ds')
    assert rec['prompt_source'] == "generated from the top-level agent's first user message"
    assert rec['task']['prompt_vs_task'] == 'same problem, other prompt'


def test_synthetic_chat_is_kept_and_marked(tmp_path, geak):
    run, rc, prov = geak
    stub = run / 'home/.claude/projects/-proj/sess/subagents/workflows/wf_1/agent-a2.jsonl'
    jl(stub, [{'type': 'user', 'message': {'role': 'user', 'content': 'validate'}},
              {'type': 'assistant', 'message': {'id': 's1', 'role': 'assistant', 'model': '<synthetic>',
                                                'content': [{'type': 'text', 'text': 'No response requested.'}]}}])
    write(run / 'chat/03-director-validate.md', f'# director:validate\n\nSource: `{stub}`\n')
    readme = run / 'chat/README.md'
    readme.write_text(readme.read_text() + '| 3 | 10:13:08 | director:validate | 2 | 0 | <synthetic> | '
                                           '[03-director-validate.md](03-director-validate.md) |\n')
    traces.convert({'machine': 'm', 'runs': [geak_entry(run, rc, prov)]}, tmp_path / 'ds')
    rec = load(tmp_path / 'ds')
    assert [c['synthetic'] for c in rec['conversation']['chats']] == [False, False, True]
    assert rec['conversation']['synthetic_chats'] == 1
    d = tmp_path / 'ds/records/geak-moe-v2'
    assert (d / 'conversation/chats/03-director-validate.jsonl').is_file()
    rows = [json.loads(l) for l in (tmp_path / 'ds/viewer/chats.jsonl').read_text().splitlines()]
    assert [(r['chat'], r['synthetic']) for r in rows][-1] == ('director:validate', True)


def test_geak_entry_must_say_how_its_prompt_relates_to_the_task(tmp_path, geak):
    e = geak_entry(*geak)
    del e['prompt_vs_task']
    with pytest.raises(traces.RecordError, match='prompt_vs_task'):
        traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')


def test_open_model_run_cannot_claim_another_prompt(tmp_path):
    from test_traces import entry
    with pytest.raises(traces.RecordError, match='instruction'):
        traces.convert({'machine': 'm', 'runs': [entry(prompt_vs_task='different problem')]}, tmp_path / 'ds')
