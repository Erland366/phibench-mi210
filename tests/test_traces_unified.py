"""One message shape for every source, and rebuilding stored records' messages from their raw files."""
import json

from phibench_mi210 import traces

from test_traces_geak import geak, geak_entry  # noqa: F401  (fixture)
from test_traces_open_models import qwen_run, sheet_entry  # noqa: F401  (fixture)


def test_unify_gives_every_source_the_same_fields():
    rows = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'a'}, {'type': 'text', 'text': 'b'}], 'call': 0},
        {'role': 'assistant', 'content': None, 'reasoning': 'r', 'call': 0,
         'tool_calls': [{'id': 'c', 'type': 'function', 'function': {'name': 'x', 'arguments': '{}'}}]},
        {'role': 'event', 'event': 'compaction', 'part': 1, 'summary': 's', 'reasoning': 'r2', 'call': 1},
        {'role': 'assistant', 'content': 'THOUGHT', 'step': 3, 'actions': [{'command': 'ls'}]},
        {'role': 'tool', 'content': 'out', 'tool_call_id': 'c', 'is_error': False, 'agent': 'eng', 'timestamp': 't'},
    ]
    out = traces.unify(rows)
    assert all(tuple(m) == traces.MESSAGE_FIELDS for m in out)
    assert [m['turn'] for m in out] == [0, 1, 2, 3, 4]
    assert out[0]['content'] == 'a\nb' and out[0]['extra'] == {'content_parts': 2} and out[0]['source_index'] == 0
    assert out[2]['content'] == 's' and out[2]['event'] == 'compaction' and out[2]['extra'] == {'part': 1}
    assert out[3]['source_index'] == 3 and out[3]['extra'] == {'actions': [{'command': 'ls'}]}
    assert out[4]['agent'] == 'eng' and out[4]['extra'] == {'is_error': False, 'timestamp': 't'}
    assert traces.unify(out) == out  # already unified: unchanged


CHAT_KEYS = {'n', 'name', 'file', 'messages', 'synthetic'}


def test_qwen_and_geak_records_share_one_layout_and_shape(tmp_path, qwen_run, geak):
    out = tmp_path / 'ds'
    traces.convert({'machine': 'm', 'runs': [sheet_entry(qwen_run), geak_entry(*geak)]}, out)
    shapes = set()
    for rid in (qwen_run.name, 'geak-moe-v2'):
        d = out / 'records' / rid
        rec = json.loads((d / 'record.json').read_text())
        conv = rec['conversation']
        assert conv['messages_format'] == traces.MESSAGES_FORMAT and conv['layout'] == traces.CHATS_LAYOUT
        assert all(CHAT_KEYS <= set(c) for c in conv['chats'])
        # the chat is in the same place for every source: one file per chat in conversation/chats/
        assert sorted(p.relative_to(d).as_posix() for p in (d / 'conversation/chats').iterdir()) == \
            sorted(c['file'] for c in conv['chats'])
        assert not (d / 'conversation/messages.jsonl').exists() and not (d / 'conversation/agents').exists()
        for c in conv['chats']:
            for line in (d / c['file']).read_text().splitlines():
                shapes.add(tuple(json.loads(line)))
    assert shapes == {traces.MESSAGE_FIELDS}


def test_remessage_moves_old_records_to_the_chats_layout(tmp_path, qwen_run, geak):
    out = tmp_path / 'ds'
    traces.convert({'machine': 'm', 'runs': [sheet_entry(qwen_run), geak_entry(*geak)]}, out)
    before = {p: p.read_text() for p in out.glob('records/*/conversation/chats/*.jsonl')}
    raw_before = {p: p.read_bytes() for p in out.glob('records/*/conversation/raw/**/*') if p.is_file()}
    # turn both records back into the earlier layout, as stored before chats-v1
    q, g = out / 'records' / qwen_run.name, out / 'records/geak-moe-v2'
    (q / 'conversation/chats/01-solver.jsonl').rename(q / 'conversation/messages.jsonl')
    (q / 'conversation/chats').rmdir()
    rq = json.loads((q / 'record.json').read_text())
    del rq['conversation']['chats'], rq['conversation']['layout']
    (q / 'record.json').write_text(json.dumps(rq))
    (g / 'conversation/chats').rename(g / 'conversation/agents')
    (g / 'conversation/messages.jsonl').write_text('{"role": "user", "content": "all agents run together"}\n')
    rg = json.loads((g / 'record.json').read_text())
    rg['conversation']['agents'] = [{**{k: v for k, v in c.items() if k != 'name'}, 'role': c['name'],
                                     'messages_file': c['file'].replace('chats', 'agents')}
                                    for c in rg['conversation'].pop('chats')]
    rg['conversation']['training_unit'] = 'old'
    (g / 'record.json').write_text(json.dumps(rg))
    assert sorted(traces.remessage(out)) == sorted([qwen_run.name, 'geak-moe-v2'])
    assert {p: p.read_text() for p in before} == before  # same chats as a fresh conversion
    assert {p: p.read_bytes() for p in raw_before} == raw_before  # raw files untouched
    for d in (q, g):
        assert not (d / 'conversation/messages.jsonl').exists() and not (d / 'conversation/agents').exists()
        conv = json.loads((d / 'record.json').read_text())['conversation']
        assert conv['layout'] == traces.CHATS_LAYOUT and 'agents' not in conv and 'training_unit' not in conv
        assert 'messages_rebuilt_at' in conv
