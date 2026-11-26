import json

import pytest

from phibench_mi210 import traces


def entry(record_id='run-1', **over):
    e = {
        'record_id': record_id, 'source': 'open_model_run', 'run_dir': 'RUN',
        'task': {'id': 'lh/x', 'bench': 'lh', 'kind': 'performance', 'origin': 'upstream 3bcefac'},
        'agent': {'harness': 'fake', 'version': '0'}, 'model': {'name': 'm'}, 'hardware': {'scored_on': 'cpu'},
        'anchor': {'ref_speedup': 2.0},
    }
    e.update(over)
    return e


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


def test_sheet_entry_must_be_complete():
    e = entry()
    del e['anchor']
    with pytest.raises(traces.RecordError, match='anchor'):
        traces.check_sheet_entry(e)
    with pytest.raises(traces.RecordError, match='kind'):
        traces.check_sheet_entry(entry(task={'id': 'x', 'bench': 'lh', 'kind': 'speed', 'origin': 'o'}))
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

    monkeypatch.setitem(traces.ADAPTERS, ('open_model_run', 'fake'), adapter)
    return run


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
    assert index == [{'record_id': 'run-1', 'task': 'lh/x', 'source': 'open_model_run', 'model': 'm',
                      'label': 'successful', 'reward': 0.25, 'speedup': 3.0}]


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
