"""Named roots: specs and run sheets write $RESULTS/... and $WORK/..., and each machine maps them to its folders."""
import json
import subprocess
from pathlib import Path

import pytest

from phibench_mi210 import roots, traces

REPO = Path(__file__).resolve().parents[1]


def test_expand_replaces_names_in_nested_values():
    table = {'RESULTS': '/r/', 'WORK': '/w'}
    got = roots.expand({'a': '$RESULTS/x', 'b': ['$WORK/y', 'plain'], 'c': 1, 'd': 'see $WORK/z and $RESULTS'}, table)
    assert got == {'a': '/r/x', 'b': ['/w/y', 'plain'], 'c': 1, 'd': 'see /w/z and /r'}


def test_unknown_or_unset_names_are_refused():
    with pytest.raises(roots.RootError, match='not a root name'):
        roots.expand('$HOME/x', {'RESULTS': '/r'})
    with pytest.raises(roots.RootError, match='not set on this machine'):
        roots.expand('$WORK/x', {'RESULTS': '/r'})


def test_roots_file_is_named_by_the_environment(tmp_path, monkeypatch):
    f = tmp_path / 'roots.json'
    f.write_text(json.dumps({'RESULTS': '/somewhere'}))
    monkeypatch.setenv('PHIBENCH_ROOTS', str(f))
    assert roots.expand('$RESULTS/a') == '/somewhere/a'
    f.write_text(json.dumps({'DATA': '/x'}))
    with pytest.raises(roots.RootError, match='not root names'):
        roots.expand('$RESULTS/a')
    monkeypatch.setenv('PHIBENCH_ROOTS', str(tmp_path / 'missing.json'))
    with pytest.raises(roots.RootError, match='does not exist'):
        roots.expand('$RESULTS/a')


def test_a_sheet_with_an_unset_root_is_refused_before_anything_is_read(tmp_path, monkeypatch):
    f = tmp_path / 'roots.json'
    f.write_text(json.dumps({'RESULTS': '/r'}))
    monkeypatch.setenv('PHIBENCH_ROOTS', str(f))
    with pytest.raises(traces.RecordError, match='not set on this machine'):
        traces.convert({'machine': 'm', 'runs': [{'record_id': 'x', 'run_dir': '$WORK/run'}]}, tmp_path / 'ds')


def test_no_machine_path_is_written_in_the_repository():
    # Only the leak check and its test data name the real patterns, on purpose.
    out = subprocess.run(['git', '-C', str(REPO), 'grep', '-l', '-E', '/data/erland|/vast/users/qirong|/home/erlan'],
                         capture_output=True, text=True).stdout.split()
    assert set(out) <= {'src/phibench_mi210/row.py', 'tests/test_row.py', 'tests/test_roots.py'}


def test_example_roots_file_lists_every_name():
    example = json.loads((REPO / 'specs/local-roots.example.json').read_text())
    assert set(example) == set(roots.ROOT_NAMES)
