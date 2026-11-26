import json
import tempfile
from pathlib import Path

import pytest

from phibench_mi210 import traces
from task_package import make_task

TASK_DIR = make_task(Path(tempfile.mkdtemp()), 'x', 2.0, 'AMD EPYC 9334')
SERVED = {'device': 'AMD Instinct MI210', 'count': 4}
QWEN = {'name': 'Qwen3.8-27B', 'precision': 'bf16',
        'served_by': {'server': 'vLLM', 'version': '0.32.0', 'settings': 'tensor parallel 4'}}


def entry(record_id='run-1', **over):
    e = {
        'record_id': record_id, 'source': 'open_model_run', 'run_dir': 'RUN', 'task_dir': str(TASK_DIR),
        'task': {'origin': 'upstream 3bcefac'},
        'agent': {'harness': 'fake', 'version': '0'}, 'model': QWEN, 'hardware': {'served_on': SERVED},
    }
    e.update(over)
    return e


def test_task_facts_come_from_the_task_package():
    e = traces.check_sheet_entry(entry())
    assert e['task'] == {'id': 'x', 'bench': 'lh', 'kind': 'performance', 'origin': 'upstream 3bcefac',
                         'task_dir': str(TASK_DIR)}
    assert e['anchor']['ref_speedup'] == 2.0 and e['anchor']['evidence'].endswith('tests/ref_speedup.txt')
    assert e['hardware'] == {'served_on': SERVED, 'scored_on': {'device': 'AMD EPYC 9334', 'count': 1}}


@pytest.mark.parametrize('over,match', [
    ({'anchor': {'ref_speedup': 2.0}}, 'come from the task package'),
    ({'task': {'id': 'x', 'origin': 'o'}}, 'come from the task package'),
    ({'hardware': {'served_on': SERVED, 'scored_on': {'device': 'AMD EPYC 9334', 'count': 1}}},
     'come from the task package'),
    ({'task': {}}, 'task.origin'),
    ({'task_dir': None}, 'task_dir'),
    ({'hardware': {'served_on': {'device': 'MI210', 'count': 4}}}, 'is not one of'),
    ({'hardware': {'served_on': {'device': 'AMD Instinct MI210', 'count': '4'}}}, 'whole number'),
    ({'hardware': {'served_on': SERVED, 'gpus': '4-7'}}, 'put details in notes'),
    ({'hardware': {}}, 'served_on'),
    ({'model': {**QWEN, 'name': 'qwen'}}, 'is not one of'),
    ({'model': {**QWEN, 'served_by': 'vLLM 0.32.0'}}, 'served_by'),
    ({'model': {**QWEN, 'served_by': {**QWEN['served_by'], 'server': 'sglang'}}}, 'is not one of'),
])
def test_sheet_entry_takes_only_fixed_choices(over, match):
    with pytest.raises(traces.RecordError, match=match):
        traces.check_sheet_entry(entry(**over))


def test_a_session_has_no_served_on_or_served_by():
    e = entry(source='claude_code_session', model={'name': 'claude-opus-5-5'}, hardware={})
    checked = traces.check_sheet_entry(e)
    assert checked['hardware']['served_on'] is None and checked['model']['served_by'] is None
    with pytest.raises(traces.RecordError, match='only for a model we served'):
        traces.check_sheet_entry({**e, 'hardware': {'served_on': SERVED}})


@pytest.mark.parametrize('reward,invalid,expected', [
    (0.14, None, 'successful'),
    (0.0, None, 'unsuccessful'),
    (None, None, 'unscored'),
    (0.5, 'setup error', 'invalid'),
    (None, 'agent never started', 'invalid'),
])
def test_label_follows_phibench_reward(reward, invalid, expected):
    assert traces.label(reward, invalid) == expected


@pytest.mark.parametrize('text', [
    b'proxy http://user:s3cret@proxy.example:3128',
    b'Authorization: Bearer abcdefghijklmnopqrstuvwxyz0123',
    b'key sk-abcdefghijklmnopqrstuvwxyz',
    b'hf_' + b'a' * 34,
    b'ghp_' + b'b' * 36,
    b'AKIAABCDEFGHIJKLMNOP',
    b'-----BEGIN OPENSSH PRIVATE KEY-----',
])
def test_secrets_are_found(tmp_path, text):
    f = tmp_path / 'f'
    f.write_bytes(b'before ' + text + b' after')
    assert traces.find_secrets(f)


def test_paths_and_local_key_are_not_secrets(tmp_path):
    f = tmp_path / 'f'
    f.write_bytes(b'/vast/users/someone/run cd /data/x; api key "local"; git@github.com:o/r.git')
    assert traces.find_secrets(f) == []


def test_anchor_comes_from_the_file_task_toml_names(tmp_path):
    d = make_task(tmp_path, 'cpu', 175.955891, 'AMD EPYC 9334')
    (d / 'tests/ref_speedup.txt').unlink()
    manifest = d / 'tests/verifier-correctness-manifest.json'
    manifest.write_text(json.dumps({'ref_speedup': 175.955891, 'upstream_ref_speedup': 187.500717}))
    toml = d / 'task.toml'
    toml.write_text(toml.read_text().replace('tests/ref_speedup.txt', 'tests/verifier-correctness-manifest.json'))
    facts = traces.task_facts(d)
    assert facts['anchor'] == {'ref_speedup': 175.955891, 'upstream_ref_speedup': 187.500717,
                               'evidence': str(manifest)}
    # a package whose task.toml still holds another anchor (upstream's) says two things, and is refused
    toml.write_text(toml.read_text().replace('ref_speedup = 175.955891', 'ref_speedup = 187.500717'))
    with pytest.raises(traces.RecordError, match='must say one anchor'):
        traces.task_facts(d)
    # and so is one that is calibrated on a device not in the list
    toml.write_text(toml.read_text().replace('ref_speedup = 187.500717', 'ref_speedup = 175.955891')
                    .replace('AMD EPYC 9334', 'Intel Sapphire Rapids CPU lane'))
    with pytest.raises(traces.RecordError, match='is not one of'):
        traces.task_facts(d)


def test_sheet_entry_must_be_complete():
    e = entry()
    del e['model']
    with pytest.raises(traces.RecordError, match='model'):
        traces.check_sheet_entry(e)
    with pytest.raises(traces.RecordError, match='record_id'):
        traces.check_sheet_entry(entry(record_id='a/b'))


@pytest.fixture
def fake_adapter(monkeypatch, tmp_path):
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'prompt.txt').write_text('do it')
    (run / 'reward.json').write_text(json.dumps({'reward': 0.25, 'speedup': 3.0}))

    def adapter(e, builder, record):
        builder.copy(run / 'prompt.txt', 'prompt.txt')
        builder.copy(run / 'reward.json', 'verifier/reward.json')
        r = json.loads((run / 'reward.json').read_text())
        record['outcome'].update(reward=r['reward'], speedup=r['speedup'])
        chat, msgs = traces.solver_chat([{'role': 'user', 'content': 'do it'}])
        builder.write_jsonl(chat['file'], msgs)
        record['conversation'] = {'layout': traces.CHATS_LAYOUT, 'chats': [chat]}

    monkeypatch.setitem(traces.ADAPTERS, ('open_model_run', 'fake'), adapter)
    return run


def test_reader_without_chats_is_refused(tmp_path, monkeypatch):
    monkeypatch.setitem(traces.ADAPTERS, ('open_model_run', 'fake'), lambda e, b, r: None)
    with pytest.raises(traces.RecordError, match='wrote no chats'):
        traces.convert({'machine': 'm', 'runs': [entry()]}, tmp_path / 'ds')
    assert not (tmp_path / 'ds/records/run-1').exists()


def test_convert_writes_record_and_index(tmp_path, fake_adapter):
    out = tmp_path / 'ds'
    done = traces.convert({'machine': 'm1', 'runs': [entry()]}, out)
    assert done == {'written': ['run-1'], 'already_stored': [], 'refused': []}
    rec = json.loads((out / 'records/run-1/record.json').read_text())
    assert rec['outcome']['label'] == 'successful'
    assert {f['path'] for f in rec['provenance']['files']} == {'prompt.txt', 'verifier/reward.json'}
    assert all(len(f['sha256']) == 64 for f in rec['provenance']['files'])
    assert (out / 'records/run-1/prompt.txt').read_text() == 'do it'
    index = [json.loads(l) for l in (out / 'index.jsonl').read_text().splitlines()]
    assert index == [{'record_id': 'run-1', 'task': 'x', 'source': 'open_model_run', 'model': 'Qwen3.8-27B',
                      'label': 'successful', 'reward': 0.25, 'speedup': 3.0, 'prompt_vs_task': 'instruction', 'chats': 1}]


def test_invalid_reason_wins_over_reward(tmp_path, fake_adapter):
    out = tmp_path / 'ds'
    traces.convert({'machine': 'm1', 'runs': [entry(invalid_reason='busy GPU')]}, out)
    rec = json.loads((out / 'records/run-1/record.json').read_text())
    assert rec['outcome']['label'] == 'invalid' and rec['outcome']['invalid_reason'] == 'busy GPU'


def test_existing_record_is_never_overwritten(tmp_path, fake_adapter):
    out = tmp_path / 'ds'
    traces.convert({'machine': 'm1', 'runs': [entry()]}, out)
    with pytest.raises(traces.RecordError, match='never overwritten'):
        traces.convert({'machine': 'm1', 'runs': [entry()]}, out)


def test_secret_refuses_the_whole_record(tmp_path, fake_adapter):
    (fake_adapter / 'prompt.txt').write_text('use http://u:p4ss@proxy:80 please')
    out = tmp_path / 'ds'
    with pytest.raises(traces.RecordError, match='url_with_credentials'):
        traces.convert({'machine': 'm1', 'runs': [entry()]}, out)
    assert not list((out / 'records').iterdir())


def test_unknown_harness_is_refused(tmp_path):
    with pytest.raises(traces.RecordError, match='no converter yet'):
        traces.convert({'machine': 'm1', 'runs': [entry(agent={'harness': 'nope'})]}, tmp_path / 'ds')


def test_machine_is_required(tmp_path):
    with pytest.raises(traces.RecordError, match='machine'):
        traces.convert({'runs': [entry()]}, tmp_path / 'ds')
