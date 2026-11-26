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


def test_qwen_and_geak_records_share_one_shape(tmp_path, qwen_run, geak):
    out = tmp_path / 'ds'
    traces.convert({'machine': 'm', 'runs': [sheet_entry(qwen_run), geak_entry(*geak)]}, out)
    shapes = set()
    for rid in (qwen_run.name, 'geak-moe-v2'):
        rec = json.loads((out / 'records' / rid / 'record.json').read_text())
        assert rec['conversation']['messages_format'] == traces.MESSAGES_FORMAT
        for line in (out / 'records' / rid / 'conversation/messages.jsonl').read_text().splitlines():
            shapes.add(tuple(json.loads(line)))
    assert shapes == {traces.MESSAGE_FIELDS}


def test_remessage_rebuilds_from_the_raw_files_and_matches_a_fresh_conversion(tmp_path, qwen_run, geak):
    out = tmp_path / 'ds'
    traces.convert({'machine': 'm', 'runs': [sheet_entry(qwen_run), geak_entry(*geak)]}, out)
    before = {p: p.read_text() for p in out.glob('records/*/conversation/**/*.jsonl') if 'raw' not in p.parts}
    raw_before = {p: p.read_bytes() for p in out.glob('records/*/conversation/raw/**/*') if p.is_file()}
    # an old-format messages file, as the first upload had
    old = out / 'records' / qwen_run.name / 'conversation/messages.jsonl'
    old.write_text('{"role": "system", "content": "old", "call": 0}\n')
    assert sorted(traces.remessage(out)) == sorted([qwen_run.name, 'geak-moe-v2'])
    after = {p: p.read_text() for p in before}
    assert after == before  # same as a fresh conversion
    assert {p: p.read_bytes() for p in raw_before} == raw_before  # raw files untouched
    rec = json.loads((out / 'records' / qwen_run.name / 'record.json').read_text())
    assert 'messages_rebuilt_at' in rec['conversation']
