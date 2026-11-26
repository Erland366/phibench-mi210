"""The open-model readers, on small run folders shaped like the Qwen Code and mini-swe-agent runs of 2026-11-24/25."""
import json

import pytest

from phibench_mi210 import traces

SYS = {'role': 'system', 'content': 'You are Qwen Code.'}
TASK = {'role': 'user', 'content': [{'type': 'text', 'text': 'reminder'}, {'type': 'text', 'text': 'Make it faster.'}]}


def asst(text, call_id=None, reasoning='think'):
    m = {'role': 'assistant', 'content': text, 'reasoning_content': reasoning}
    if call_id:
        m['tool_calls'] = [{'id': call_id, 'type': 'function',
                            'function': {'name': 'run_shell_command', 'arguments': '{"command": "ls"}'}}]
    return m


def tool(call_id, text='out'):
    return {'role': 'tool', 'tool_call_id': call_id, 'content': [{'type': 'text', 'text': text}]}


def resent(m):
    """How Qwen Code resends an earlier assistant message: reasoning in two fields, no refusal."""
    r = dict(m)
    r['reasoning'] = r.get('reasoning_content')
    return r


def call(messages, response_message, error=None):
    resp = None if error else {'choices': [{'message': dict(response_message, refusal=None)}]}
    return {'timestamp': 't', 'request': {'messages': messages, 'model': 'qwen3.8-27b'}, 'response': resp,
            'error': {'message': error} if error else None}


def test_messages_emit_each_message_once():
    a1 = asst('', 'c1')
    a2 = asst('done')
    calls = [call([SYS, TASK], a1), call([SYS, TASK, resent(a1), tool('c1')], a2)]
    rows, counts = traces.messages_from_openai_calls(calls)
    assert [r['role'] for r in rows] == ['system', 'user', 'assistant', 'tool', 'assistant']
    assert rows[2]['tool_calls'][0]['id'] == 'c1' and rows[2]['reasoning'] == 'think'
    assert [r['call'] for r in rows] == [0, 0, 0, 1, 1]
    assert counts == {'calls': 2, 'failed_calls': 0, 'compactions': 0, 'history_rewrites': 0, 'side_calls': 0}


def test_compaction_becomes_one_event_and_the_new_history_follows():
    a1 = asst('', 'c1')
    summarize = {'role': 'user', 'content': [{'type': 'text', 'text': traces.COMPACTION_PROMPT + ' ...'}]}
    summary = {'role': 'assistant', 'content': '<analysis>so far</analysis>', 'reasoning_content': 'r'}
    snapshot = {'role': 'user', 'content': [{'type': 'text', 'text': '<state_snapshot>so far'}]}
    ack = {'role': 'assistant', 'content': 'Got it. Thanks for the additional context!'}
    files = {'role': 'user', 'content': [{'type': 'text', 'text': 'The following files were recently accessed'}]}
    a2 = asst('next', 'c2')
    calls = [
        call([SYS, TASK], a1),
        call([SYS, TASK, resent(a1), tool('c1'), summarize], summary),
        call([SYS, snapshot, ack, files], a2),
    ]
    rows, counts = traces.messages_from_openai_calls(calls)
    roles = [(r['role'], r.get('event')) for r in rows]
    assert roles == [('system', None), ('user', None), ('assistant', None),
                     ('event', 'compaction'),
                     ('user', None), ('assistant', None), ('user', None), ('assistant', None)]
    assert rows[3]['summary'] == '<analysis>so far</analysis>' and rows[3]['reasoning'] == 'r'
    assert counts['compactions'] == 1 and counts['history_rewrites'] == 0


def test_failed_call_is_an_event_and_the_retry_continues():
    a1 = asst('', 'c1')
    calls = [call([SYS, TASK], a1),
             call([SYS, TASK, resent(a1), tool('c1')], None, error='Stream exceeded its 900000ms cap'),
             call([SYS, TASK, resent(a1), tool('c1')], asst('ok'))]
    rows, counts = traces.messages_from_openai_calls(calls)
    assert [(r['role'], r.get('event')) for r in rows] == [
        ('system', None), ('user', None), ('assistant', None), ('event', 'failed_call'), ('tool', None),
        ('assistant', None)]
    assert counts['failed_calls'] == 1


def test_system_form_summarizer_is_part_of_the_same_compaction():
    a1 = asst('', 'c1')
    summarize = {'role': 'user', 'content': [{'type': 'text', 'text': traces.COMPACTION_PROMPT}]}
    sys_summarize = {'role': 'system', 'content': traces.COMPACTION_PROMPT + ' (system form)'}
    snapshot = {'role': 'user', 'content': [{'type': 'text', 'text': '<state_snapshot>x'}]}
    calls = [call([SYS, TASK], a1),
             call([SYS, TASK, resent(a1), tool('c1'), summarize], {'role': 'assistant', 'content': 's1'}),
             call([sys_summarize, {'role': 'user', 'content': 'summary again'}], {'role': 'assistant', 'content': 's2'}),
             call([SYS, snapshot], asst('go'))]
    rows, counts = traces.messages_from_openai_calls(calls)
    events = [(r['event'], r.get('part')) for r in rows if r['role'] == 'event']
    assert events == [('compaction', 1), ('compaction', 2)]
    assert counts['compactions'] == 1 and counts['history_rewrites'] == 0 and counts['side_calls'] == 0
    assert not any(r['role'] == 'system' and traces.COMPACTION_PROMPT in str(r['content']) for r in rows)


def test_side_call_does_not_enter_the_conversation():
    a1 = asst('', 'c1')
    side = {'role': 'system', 'content': 'You are now acting as the managed memory extraction subagent.'}
    calls = [call([SYS, TASK], a1),
             call([side, {'role': 'user', 'content': 'extract'}], {'role': 'assistant', 'content': 'memories'}),
             call([SYS, TASK, resent(a1), tool('c1')], asst('done'))]
    rows, counts = traces.messages_from_openai_calls(calls)
    assert [(r['role'], r.get('event')) for r in rows] == [
        ('system', None), ('user', None), ('assistant', None), ('event', 'side_call'), ('tool', None),
        ('assistant', None)]
    assert counts['side_calls'] == 1 and counts['history_rewrites'] == 0
    assert rows[3]['system'].startswith('You are now acting')


def test_reminders_prepended_after_compaction_are_an_update_not_a_rewrite():
    a1 = asst('', 'c1')
    summarize = {'role': 'user', 'content': [{'type': 'text', 'text': traces.COMPACTION_PROMPT}]}
    snap_part = {'type': 'text', 'text': '<state_snapshot>so far'}
    snapshot = {'role': 'user', 'content': [snap_part]}
    reminders = [{'type': 'text', 'text': '<system-reminder>skills'}, {'type': 'text', 'text': '<system-reminder>git'}]
    snapshot_later = {'role': 'user', 'content': reminders + [snap_part]}
    a2 = asst('', 'c2')
    calls = [call([SYS, TASK], a1),
             call([SYS, TASK, resent(a1), tool('c1'), summarize], {'role': 'assistant', 'content': 's'}),
             call([SYS, snapshot], a2),
             call([SYS, snapshot_later, resent(a2), tool('c2')], asst('done'))]
    rows, counts = traces.messages_from_openai_calls(calls)
    assert counts['history_rewrites'] == 0
    upd = [r for r in rows if r.get('event') == 'message_updated']
    assert upd == [{'role': 'event', 'event': 'message_updated', 'index': 1, 'added_parts': reminders, 'call': 3}]
    assert [r['role'] for r in rows[-2:]] == ['tool', 'assistant']


def test_unexpected_rewrite_is_marked():
    calls = [call([SYS, TASK], asst('a')), call([SYS, {'role': 'user', 'content': 'other'}], asst('b'))]
    rows, counts = traces.messages_from_openai_calls(calls)
    assert counts['history_rewrites'] == 1
    assert ('event', 'history_rewritten') in [(r['role'], r.get('event')) for r in rows]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj))


@pytest.fixture
def qwen_run(tmp_path):
    run = tmp_path / '2026-11-25-qwen-code-x-1'
    write(run / 'prompt.txt', 'Make it faster.')
    write(run / 'agent_result.json', {'sandbox': '/tmp/s', 'exit_code': 124, 'wall_limit_s': 5400})
    write(run / 'started', '2026-11-25T21:29:34Z\n')
    write(run / 'finished', '2026-11-25T23:00:00Z\n')
    write(run / 'settings.json', {'model': {'generationConfig': {'contextWindowSize': 131072}}})
    write(run / 'stream.jsonl', json.dumps({'type': 'system', 'subtype': 'init', 'model': 'qwen3.8-27b',
                                            'qwen_code_version': '0.28.4'}) + '\n')
    a1 = asst('', 'c1')
    write(run / 'logs/agent/openai/openai-2026-11-25T21-41-40.992Z-aaaa.json', call([SYS, TASK], a1))
    write(run / 'logs/agent/openai/openai-2026-11-25T21-42-00.000Z-bbbb.json',
          call([SYS, TASK, resent(a1), tool('c1')], asst('done')))
    write(run / 'logs/loop/state.jsonl',
          json.dumps({'submission': 1, 'ts': 1, 'correctness_ok': False, 'reward': 0.0, 'speedup': 376.2,
                      'tree': 't1', 'hard_fail_reasons': ['implausible_vs_oracle']}) + '\n' +
          json.dumps({'submission': 2, 'ts': 2, 'correctness_ok': True, 'reward': 0.1, 'speedup': 200.0,
                      'tree': 't2', 'hard_fail_reasons': []}) + '\n')
    for n in (1, 2):
        write(run / f'logs/loop/submissions/{n}/candidate.diff', f'diff {n}\n')
        write(run / f'logs/loop/submissions/{n}/reward.json', {'reward': 0, 'speedup': 1})
    write(run / 'logs/loop/finalize.json', {'plant_method': 'read-tree', 'winning_submission': 2,
                                            'termination_reason': 'time'})
    write(run / 'logs/loop/count', '2')
    write(run / 'final/verifier/reward.json', {'reward': 0.138925, 'speedup': 360.87, 'ref_speedup': 175.96,
                                               'gates': {'correctness_ok': True}, 'hard_fail_reasons': []})
    write(run / 'final/exit', '0\n')
    write(run / 'final/finalize.txt', 'submissions 2; finalize record: /logs/loop/finalize.json\n')
    write(run / 'final/model.patch', 'diff final\n')
    write(run / 'final/model_status.txt', ' M a.py\n')
    write(tmp_path / 'logs/serve-qwen-x-20261125T2127.log', 'version 0.32.0\n')
    return run


def sheet_entry(run, **over):
    e = {'record_id': run.name, 'source': 'open_model_run', 'run_dir': str(run),
         'task': {'id': 'wro-causal-delivery-vclock', 'bench': 'lh', 'kind': 'performance',
                  'origin': 'upstream Phi-Bench 3bcefac'},
         'agent': {'harness': 'qwen-code'}, 'model': {'name': 'Qwen3.8-27B', 'precision': 'bf16'},
         'hardware': {'scored_on': 'AMD EPYC 9334'},
         'anchor': {'ref_speedup': 175.96, 'evidence': 'runs/2026-11-25-x-calibration'},
         'serving_log': str(run.parent / 'logs/serve-qwen-x-20261125T2127.log'),
         'submission_timing': 'checked'}
    e.update(over)
    return e


def test_qwen_code_run_becomes_a_record(tmp_path, qwen_run):
    out = tmp_path / 'ds'
    traces.convert({'machine': 'auh7-1b-gpu-240', 'runs': [sheet_entry(qwen_run)]}, out)
    d = out / 'records' / qwen_run.name
    rec = json.loads((d / 'record.json').read_text())
    assert rec['outcome']['label'] == 'successful'
    assert rec['outcome']['reward'] == 0.138925 and rec['outcome']['harness_exit_code'] == 124
    assert rec['outcome']['final_tree'] == 'winning submission 2'
    assert rec['agent']['version'] == '0.28.4' and rec['model']['served_name'] == 'qwen3.8-27b'
    assert [s['n'] for s in rec['submissions']] == [1, 2] and rec['submissions'][0]['timing_clean'] is True
    assert rec['conversation']['calls'] == 2
    for rel in ('prompt.txt', 'code/final.patch', 'code/final.status.txt', 'code/submissions/2/candidate.diff',
                'harness/loop/finalize.json', 'harness/agent_result.json', 'verifier/final/reward.json',
                'serving/serve-qwen-x-20261125T2127.log', 'conversation/raw/stream.jsonl'):
        assert (d / rel).is_file(), rel
    assert len(list((d / 'conversation/raw/openai').iterdir())) == 2
    assert rec['conversation']['chats'] == [{'n': 1, 'name': 'solver', 'file': 'conversation/chats/01-solver.jsonl',
                                             'messages': 5, 'synthetic': False}]
    rows = [json.loads(l) for l in (d / 'conversation/chats/01-solver.jsonl').read_text().splitlines()]
    assert [r['role'] for r in rows] == ['system', 'user', 'assistant', 'tool', 'assistant']
    assert {r['agent'] for r in rows} == {'solver'}
    assert not (d / 'conversation/messages.jsonl').exists()


def test_viewer_rows_have_one_shape(tmp_path, qwen_run):
    out = tmp_path / 'ds'
    traces.convert({'machine': 'm', 'runs': [sheet_entry(qwen_run)]}, out)
    runs = [json.loads(l) for l in (out / 'viewer/chats.jsonl').read_text().splitlines()]
    assert len(runs) == 1 and runs[0]['label'] == 'successful' and runs[0]['chat'] == 'solver'
    msgs = runs[0]['messages']
    assert {tuple(m) for m in msgs} == {tuple(k for k in traces.MESSAGE_FIELDS if k != 'extra')}
    assert all(m['content'] is None or isinstance(m['content'], str) for m in msgs)
    assert isinstance(msgs[2]['tool_calls'], str)
    assert (out / 'README.md').read_text().startswith('---\nconfigs:')


def test_anchor_mismatch_is_refused(tmp_path, qwen_run):
    e = sheet_entry(qwen_run, anchor={'ref_speedup': 322.07})
    with pytest.raises(traces.RecordError, match='ref_speedup'):
        traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')


def test_version_mismatch_is_refused(tmp_path, qwen_run):
    e = sheet_entry(qwen_run, agent={'harness': 'qwen-code', 'version': '0.27.0'})
    with pytest.raises(traces.RecordError, match='0.28.4'):
        traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')


def test_unclean_submission_timing_is_kept(tmp_path, qwen_run):
    e = sheet_entry(qwen_run, unclean_submissions={'2': 'GPU 0 shared with another queue'})
    traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')
    rec = json.loads((tmp_path / 'ds/records' / qwen_run.name / 'record.json').read_text())
    assert [s['timing_clean'] for s in rec['submissions']] == [True, False]
    assert rec['submissions'][1]['timing_note'] == 'GPU 0 shared with another queue'


def test_unchecked_submission_timing_is_unknown(tmp_path, qwen_run):
    e = sheet_entry(qwen_run)
    del e['submission_timing']
    traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')
    rec = json.loads((tmp_path / 'ds/records' / qwen_run.name / 'record.json').read_text())
    assert {s['timing_clean'] for s in rec['submissions']} == {None}


def test_thrown_away_score_is_kept_as_invalid_score(tmp_path, qwen_run):
    write(qwen_run / 'final-busy/verifier/reward.json', {'reward': 0.0, 'ref_speedup': 175.96})
    e = sheet_entry(qwen_run, invalid_scores=[{'dir': 'final-busy', 'reason': 'GPU shared with another job'}])
    traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')
    d = tmp_path / 'ds/records' / qwen_run.name
    rec = json.loads((d / 'record.json').read_text())
    assert rec['outcome']['invalid_scores'] == [{'dir': 'verifier/invalid-1', 'reason': 'GPU shared with another job',
                                                 'reward': 0.0}]
    assert (d / 'verifier/invalid-1/verifier/reward.json').is_file()
    assert rec['outcome']['label'] == 'successful'


def test_valid_run_without_final_score_is_refused(tmp_path, qwen_run):
    (qwen_run / 'final/verifier/reward.json').unlink()
    with pytest.raises(traces.RecordError, match='no final score'):
        traces.convert({'machine': 'm', 'runs': [sheet_entry(qwen_run)]}, tmp_path / 'ds')


def test_invalid_run_without_final_score_is_stored(tmp_path, qwen_run):
    (qwen_run / 'final/verifier/reward.json').unlink()
    e = sheet_entry(qwen_run, invalid_reason='setup error before the agent edited anything')
    traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')
    rec = json.loads((tmp_path / 'ds/records' / qwen_run.name / 'record.json').read_text())
    assert rec['outcome']['label'] == 'invalid' and 'reward' not in rec['outcome']


def test_serving_log_is_required(tmp_path, qwen_run):
    e = sheet_entry(qwen_run)
    del e['serving_log']
    with pytest.raises(traces.RecordError, match='serving_log'):
        traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')


def test_never_submitted_run_measured_its_working_tree(tmp_path, qwen_run):
    import shutil
    shutil.rmtree(qwen_run / 'logs/loop')
    traces.convert({'machine': 'm', 'runs': [sheet_entry(qwen_run)]}, tmp_path / 'ds')
    rec = json.loads((tmp_path / 'ds/records' / qwen_run.name / 'record.json').read_text())
    assert rec['outcome']['final_tree'] == 'working tree as left (never submitted)' and rec['submissions'] == []


def test_keep_going_converts_the_rest_and_lists_refusals(tmp_path, qwen_run):
    good = sheet_entry(qwen_run)
    bad = sheet_entry(qwen_run, record_id='bad-run', anchor={'ref_speedup': 1.0})
    out = tmp_path / 'ds'
    r = traces.convert({'machine': 'm', 'runs': [bad, good]}, out, keep_going=True)
    assert r['written'] == [qwen_run.name] and r['refused'][0][0] == 'bad-run'
    assert 'ref_speedup' in r['refused'][0][1]
    assert not (out / 'records/bad-run').exists() and not list((out / 'records').glob('.*staging'))
    again = traces.convert({'machine': 'm', 'runs': [good]}, out, keep_going=True)
    assert again == {'written': [], 'already_stored': [qwen_run.name], 'refused': []}


def test_cli_exits_2_when_a_run_is_refused(tmp_path, qwen_run):
    sheet = tmp_path / 'sheet.json'
    sheet.write_text(json.dumps({'machine': 'm', 'runs': [sheet_entry(qwen_run, anchor={'ref_speedup': 1.0})]}))
    assert traces.main(['convert', '--sheet', str(sheet), '--out', str(tmp_path / 'ds')]) == 2


def test_mini_swe_agent_format_error_turn_keeps_the_rejected_reply():
    traj = {'messages': [
        {'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'task'},
        {'role': 'user', 'content': 'Format error:\n\n<error>\nExpected exactly 1 action, found 3.\n</error>',
         'extra': {'interrupt_type': 'FormatError', 'n_actions': 3, 'model_response': 'three commands',
                   'response': {'choices': [{'message': {'content': 'three commands', 'reasoning_content': 'hmm'}}]}}},
        {'role': 'user', 'content': 'Format error: again',
         'extra': {'interrupt_type': 'FormatError', 'model_response': 'only text', 'response': None}}]}
    rows = traces.messages_from_mini_swe_agent(traj)
    assert [(r['role'], r.get('format_error')) for r in rows] == [
        ('system', None), ('user', None), ('assistant', True), ('user', None), ('assistant', True), ('user', None)]
    assert rows[2]['content'] == 'three commands' and rows[2]['reasoning'] == 'hmm'
    assert rows[4]['content'] == 'only text' and 'reasoning' not in rows[4]


def test_mini_swe_agent_run_becomes_a_record(tmp_path, qwen_run):
    run = qwen_run
    (run / 'stream.jsonl').unlink()
    (run / 'prompt.txt').unlink()
    write(run / 'agent_result.json', {'exit_status': 'RepeatedFormatError', 'model_calls': 56, 'wall_s': 3587})
    write(run / 'trajectory.json', {
        'info': {'mini_version': '2.4.6', 'config': {'model': {'model_name': 'hosted_vllm/qwen3.8-27b'}}},
        'messages': [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'task'},
                     {'role': 'assistant', 'content': 'THOUGHT: x', 'reasoning_content': 'r',
                      'extra': {'actions': [{'command': 'ls'}]}},
                     {'role': 'user', 'content': '<returncode>0</returncode>'},
                     {'role': 'exit', 'content': 'RepeatedFormatError'}]})
    e = sheet_entry(run, agent={'harness': 'mini-swe-agent', 'version': '2.4.6'})
    traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')
    d = tmp_path / 'ds/records' / run.name
    rec = json.loads((d / 'record.json').read_text())
    assert rec['outcome']['harness_exit_status'] == 'RepeatedFormatError'
    assert (d / 'prompt.txt').read_text() == 'task'
    assert rec['prompt_source'] == 'generated from trajectory.json messages[1]'
    assert rec['model']['served_name'] == 'hosted_vllm/qwen3.8-27b'
    rows = [json.loads(l) for l in (d / 'conversation/chats/01-solver.jsonl').read_text().splitlines()]
    assert rows[2]['reasoning'] == 'r' and rows[2]['extra']['actions'] == [{'command': 'ls'}]
    assert rows[-1]['role'] == 'event' and rows[-1]['event'] == 'exit'
    assert rows[-1]['content'] == 'RepeatedFormatError' and rows[-1]['source_index'] == 4
