"""A contributor's own Claude Code session: cutting the slice, its helper agents, and the record it becomes."""
import json

import pytest

from phibench_mi210 import traces
from task_package import make_task


def line(obj):
    return json.dumps(obj) + '\n'


def user(uuid, ts, text):
    return {'type': 'user', 'uuid': uuid, 'timestamp': ts, 'version': '2.9.1',
            'message': {'role': 'user', 'content': text}}


def reply(uuid, ts, mid, blocks):
    return {'type': 'assistant', 'uuid': uuid, 'timestamp': ts, 'version': '2.9.1',
            'message': {'id': mid, 'role': 'assistant', 'model': 'claude-opus-5-5', 'content': blocks}}


SESSION = [
    user('u0', '2026-11-20T09:00:00Z', 'Rename the variable in utils.py'),  # earlier, unrelated work
    reply('a0', '2026-11-20T09:00:05Z', 'm0', [{'type': 'text', 'text': 'Renamed.'}]),
    {'type': 'file-history-snapshot', 'messageId': 'x'},
    user('u1', '2026-11-20T10:00:00Z', 'Make the MoE kernel faster for 256 tokens'),  # the slice starts here
    reply('a1', '2026-11-20T10:00:05Z', 'm1', [{'type': 'thinking', 'thinking': 'Look at the tile table.',
                                                 'signature': 's'}]),
    reply('a2', '2026-11-20T10:00:06Z', 'm1', [{'type': 'tool_use', 'id': 't1', 'name': 'Read',
                                                 'input': {'file_path': 'dec_configs.json'}}]),
    {'type': 'user', 'uuid': 'u2', 'timestamp': '2026-11-20T10:00:07Z',
     'message': {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': '{}'}]}},
    reply('a3', '2026-11-20T10:30:00Z', 'm2', [{'type': 'text', 'text': 'Added the 256 entry; 1.2x faster.'}]),
    user('u3', '2026-11-20T11:00:00Z', 'Now write the meeting notes'),  # later, unrelated work
]


@pytest.fixture
def session(tmp_path):
    proj = tmp_path / '.claude/projects/-home-me-repo'
    s = proj / 'sess.jsonl'
    s.parent.mkdir(parents=True)
    s.write_text(''.join(line(o) for o in SESSION))
    subs = proj / 'sess/subagents'
    subs.mkdir(parents=True)
    (subs / 'agent-bench-1.jsonl').write_text(
        line({'type': 'user', 'timestamp': '2026-11-20T10:10:00Z', 'isSidechain': True,
              'message': {'role': 'user', 'content': 'Time the kernel'}}) +
        line(reply('b1', '2026-11-20T10:11:00Z', 'mb', [{'type': 'text', 'text': '1.21x'}])))
    (subs / 'agent-bench-1.meta.json').write_text(json.dumps({'name': 'bench'}))
    (subs / 'agent-notes-2.jsonl').write_text(  # started after the slice ended: not part of this work
        line({'type': 'user', 'timestamp': '2026-11-20T11:05:00Z', 'message': {'role': 'user', 'content': 'notes'}}))
    task = make_task(tmp_path, 'mi210-jais3-moe-m256', 1.201437, 'AMD Instinct MI210', gpus=1)
    (task / 'oracle.patch').write_text('+ 256 entry\n')
    return s, task


def entry(s, task, **over):
    e = {'record_id': '2026-11-20-claude-code-m256', 'source': 'claude_code_session', 'run_dir': str(s.parent),
         'session': str(s), 'slice': {'from_uuid': 'u1', 'until_uuid': 'u3'},
         'task_dir': str(task), 'task': {'origin': 'Erland/phibench-mi210-rows'},
         'agent': {'harness': 'claude-code', 'version': '2.9.1'}, 'model': {'name': 'claude-opus-5-5'},
         'hardware': {}, 'prompt_vs_task': 'same problem, other prompt'}
    e.update(over)
    return e


def test_session_requests_lists_only_what_a_person_typed(session):
    reqs = traces.session_requests(session[0])
    assert [(r['uuid'], r['line']) for r in reqs] == [('u0', 1), ('u1', 4), ('u3', 9)]
    assert reqs[1]['text'] == 'Make the MoE kernel faster for 256 tokens'


def test_session_slice_becomes_a_successful_record(tmp_path, session):
    s, task = session
    out = tmp_path / 'ds'
    traces.convert({'machine': 'm', 'runs': [entry(s, task)]}, out)
    d = out / 'records/2026-11-20-claude-code-m256'
    rec = json.loads((d / 'record.json').read_text())
    assert rec['outcome']['label'] == 'successful' and 'reference solution' in rec['outcome']['label_rule']
    # the raw slice is exactly the original lines 4-8, nothing before or after
    original = s.read_bytes().splitlines(keepends=True)
    assert (d / 'conversation/raw/claude/sess.slice.jsonl').read_bytes() == b''.join(original[3:8])
    assert rec['conversation']['slice']['lines'] == [4, 8]
    cut = [f for f in rec['provenance']['files'] if f['path'].endswith('sess.slice.jsonl')][0]
    assert cut['cut'] == 'lines 4-8' and cut['source_sha256'] == traces.sha256(s)
    # the same layout as every other source: the session is 01-solver, the helper agent that started inside it is 02
    chats = rec['conversation']['chats']
    assert [(c['n'], c['name'], c['file']) for c in chats] == [
        (1, 'solver', 'conversation/chats/01-solver.jsonl'), (2, 'bench', 'conversation/chats/02-bench.jsonl')]
    assert sorted(p.name for p in (d / 'conversation/chats').iterdir()) == ['01-solver.jsonl', '02-bench.jsonl']
    assert not (d / 'conversation/raw/claude/subagents/agent-notes-2.jsonl').exists()
    msgs = [json.loads(l) for l in (d / 'conversation/chats/01-solver.jsonl').read_text().splitlines()]
    assert [m['role'] for m in msgs] == ['user', 'assistant', 'tool', 'assistant']
    assert msgs[1]['reasoning'] == 'Look at the tile table.' and {m['agent'] for m in msgs} == {'solver'}
    assert 'Rename' not in json.dumps(msgs) and 'meeting notes' not in json.dumps(msgs)
    assert (d / 'prompt.txt').read_text() == 'Make the MoE kernel faster for 256 tokens'
    assert (d / 'code/reference.patch').read_text() == '+ 256 entry\n'
    assert rec['conversation']['thinking_blocks'] == 1 and rec['agent']['version'] == '2.9.1'
    assert rec['task']['prompt_vs_task'] == 'same problem, other prompt'


def test_remessage_rebuilds_a_session_record_unchanged(tmp_path, session):
    out = tmp_path / 'ds'
    traces.convert({'machine': 'm', 'runs': [entry(*session)]}, out)
    before = {p: p.read_text() for p in out.glob('records/*/conversation/chats/*.jsonl')}
    traces.remessage(out)
    assert {p: p.read_text() for p in before} == before


def test_slice_must_start_at_a_request(tmp_path, session):
    s, task = session
    with pytest.raises(traces.RecordError, match='not a request a person typed'):
        traces.convert({'machine': 'm', 'runs': [entry(s, task, slice={'from_uuid': 'a1'})]}, tmp_path / 'ds')


def test_slice_ends_before_the_next_request(tmp_path, session):
    s, task = session
    # until_uuid must be a request too, so both ends come from session-requests
    with pytest.raises(traces.RecordError, match='not a request a person typed'):
        traces.convert({'machine': 'm', 'runs': [entry(s, task, slice={'from_uuid': 'u1', 'until_uuid': 'a3'})]},
                       tmp_path / 'ds')
    with pytest.raises(traces.RecordError, match='not after from_uuid'):
        traces.convert({'machine': 'm', 'runs': [entry(s, task, slice={'from_uuid': 'u1', 'until_uuid': 'u0'})]},
                       tmp_path / 'ds')
    with pytest.raises(traces.RecordError, match='until_uuid'):  # the earlier key is refused, not ignored
        traces.convert({'machine': 'm', 'runs': [entry(s, task, slice={'from_uuid': 'u1', 'to_uuid': 'a3'})]},
                       tmp_path / 'ds')
    # without until_uuid the slice runs to the end of the session, including the later request
    traces.convert({'machine': 'm', 'runs': [entry(s, task, slice={'from_uuid': 'u1'})]}, tmp_path / 'ds')
    rec = json.loads((tmp_path / 'ds/records/2026-11-20-claude-code-m256/record.json').read_text())
    assert rec['conversation']['slice']['lines'] == [4, 9]


def test_session_needs_its_task_package(tmp_path, session):
    s, task = session
    with pytest.raises(traces.RecordError, match='task.toml'):
        traces.convert({'machine': 'm', 'runs': [entry(s, tmp_path / 'nowhere')]}, tmp_path / 'ds')
    (task / 'oracle.patch').unlink()
    with pytest.raises(traces.RecordError, match='oracle.patch'):
        traces.convert({'machine': 'm', 'runs': [entry(s, task)]}, tmp_path / 'ds')


def test_secret_in_the_slice_is_refused(tmp_path, session):
    s, task = session
    s.write_text(s.read_text().replace('1.2x faster', 'token hf_' + 'a' * 34))
    with pytest.raises(traces.RecordError, match='matches'):
        traces.convert({'machine': 'm', 'runs': [entry(s, task)]}, tmp_path / 'ds')
    assert not (tmp_path / 'ds/records/2026-11-20-claude-code-m256').exists()


def test_example_run_sheet_passes_the_sheet_check(tmp_path):
    from pathlib import Path
    sheet = json.loads((Path(__file__).parent.parent / 'specs/examples/run-sheet-example.json').read_text())
    assert {e['source'] for e in sheet['runs']} == {'claude_code_session', 'open_model_run', 'geak_run'}
    for e in sheet['runs']:
        # the example's task packages live on the run machines; check every other field against a stand-in package
        stand_in = make_task(tmp_path / e['record_id'], 'stand-in', 1.5, 'AMD Instinct MI210', gpus=1)
        traces.check_sheet_entry({**e, 'task_dir': str(stand_in)})
        assert (e['source'], e['agent']['harness']) in traces.ADAPTERS
    session = [e for e in sheet['runs'] if e['source'] == 'claude_code_session'][0]
    assert {'session', 'slice', 'task_dir', 'prompt_vs_task'} <= set(session)


def test_codex_session_is_refused_with_its_reason(tmp_path, session):
    s, task = session
    e = entry(s, task, source='codex_session', agent={'harness': 'codex'})
    with pytest.raises(traces.RecordError, match='no Codex reader yet'):
        traces.convert({'machine': 'm', 'runs': [e]}, tmp_path / 'ds')
