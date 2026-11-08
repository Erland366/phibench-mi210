"""CPU-only tests of the GEAK run to Phi-Bench row converter, on a small fake run."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from evaluation.geak import phibench_row as pr

SUMMARY = 'A routed MoE layer on MI210. ' + 'x' * 700
RUNNER = '"""Runner naming GEAK, to be replaced."""\nPIN = "@@PIN@@"\nprint("runner")\n'
START = 'def moe_forward(x):\n    # the served kernel\n    return x * 2\n'
CANDIDATE = 'def fast_secret_kernel(x):\n    return x * 2\n\ndef moe_forward(x):\n    return fast_secret_kernel(x)\n'


def make_case(tmp: Path, score=1.2, verdict='pass', correctness='PASS') -> dict:
    task, cand, run, check, spec_dir = (tmp / n for n in ('rc/task', 'cand', 'run', 'rc', 'spec'))
    (task / 'scripts').mkdir(parents=True)
    cand.mkdir()
    run.mkdir()
    spec_dir.mkdir()
    (task / 'base.py').write_text(START)
    (task / 'routing.json').write_text('{"source": "/data/erland/secret/x"}\n')
    routing_sha = pr.sha256((task / 'routing.json').read_bytes())
    (task / 'scripts/runner.py').write_text(RUNNER.replace('@@PIN@@', routing_sha))
    (cand / 'kernel.py').write_text(CANDIDATE)
    (run / 'outcome.json').write_text(json.dumps({'status': 'incomplete', 'models_outside_cap': []}))
    (check / 'verdict.json').write_text(json.dumps({'summary': {
        'verdict': verdict, 'correctness': correctness, 'control_identical': True, 'cases_failed': [],
        'score_speedup': score}}))
    (check / 'prepare.json').write_text(json.dumps({
        'editable': ['kernel.py'], 'source': {'kind': 'directory', 'candidate_dir': str(cand)},
        'candidate': {'kernel.py': {'sha256': pr.sha256(CANDIDATE.encode())}},
        'frozen_task': {'sha256': {'base.py': pr.sha256(START.encode()), 'routing.json': routing_sha}}}))
    (spec_dir / 'instruction.md').write_text('# Make moe_forward fast on MI210.\n')
    spec = {'task_dir': 'mi210-test-task', 'bench': 'lh', 'infra_name': 'test_infra', 'title': 'Test task',
            'summary': SUMMARY, 'agent_timeout_sec': 60, 'cpus': 1, 'memory_mb': 1024, 'gpus': 1,
            'max_submissions': 16, 'runner': 'scripts/runner.py', 'runner_docstring': 'Frozen runner.\n',
            'baseline_of': {'kernel.py': 'base.py'}, 'frozen_files': ['base.py', 'routing.json'],
            'scrub': [['/data/erland/secret/', 'run/']], 'banned_tokens': ['torch.matmul'],
            'correctness_seeds': [11], 'dev_correctness_seeds': [0], 'abba_blocks': 3, 'dev_abba_blocks': 1,
            'runtime': {'base_image': 'public/image:1', 'torch_prefix': '2', 'vllm_prefix': '0', 'arch': 'gfx90a',
                        'description': 'test'},
            'negative': [{'file': 'kernel.py', 'find': 'return x * 2\n', 'replace': 'return x * 3\n'}]}
    (spec_dir / 'spec.json').write_text(json.dumps(spec))
    return {'spec_dir': spec_dir, 'run_dir': run, 'round_check': check, 'out': tmp / 'out'}


def test_builds_a_row_with_exactly_the_dataset_columns(tmp_path):
    case = make_case(tmp_path)
    result = pr.build(**case)
    upstream = json.loads((pr.VENDORED / 'upstream_task_catalog.jsonl').read_text().splitlines()[0])
    rows = (case['out'] / 'task_catalog.jsonl').read_text().splitlines()
    row = json.loads(rows[0])
    assert len(rows) == 1 and list(row) == list(upstream)
    assert {k: type(v) for k, v in row.items()} == {k: type(v) for k, v in upstream.items()}
    assert row['ref_speedup'] == 1.2 and row['task_dir'] == 'mi210-test-task'
    assert 'verify_package' in result


def test_start_is_the_baseline_and_the_reference_stays_out_of_the_image(tmp_path):
    case = make_case(tmp_path)
    pr.build(**case)
    package = case['out'] / 'tasks/lh/mi210-test-task'
    assert (package / 'environment/repo/kernel.py').read_text() == START
    assert 'fast_secret_kernel' in (package / 'oracle.patch').read_text()
    image_files = [package / 'instruction.md', *(package / 'environment').rglob('*')]
    assert not any('fast_secret_kernel' in p.read_text() for p in image_files if p.is_file())
    assert '*.patch' in (package / '.dockerignore').read_text()
    subprocess.run(['git', 'apply', '--check', '-p1', str(package / 'oracle.patch')],
                   cwd=package / 'environment/repo', check=True)


def test_internal_paths_are_scrubbed_and_the_runner_pin_follows(tmp_path):
    case = make_case(tmp_path)
    pr.build(**case)
    repo = case['out'] / 'tasks/lh/mi210-test-task/environment/repo'
    routing = (repo / 'routing.json').read_bytes()
    assert b'/data/erland' not in routing
    runner = (repo / 'scripts/runner.py').read_text()
    assert pr.sha256(routing) in runner and 'GEAK' not in runner


def test_row_values_only_live_in_columns_and_hardware_lives_in_the_package(tmp_path):
    case = make_case(tmp_path)
    pr.build(**case)
    toml = (case['out'] / 'tasks/lh/mi210-test-task/task.toml').read_text()
    assert 'gpu_types = ["MI210"]' in toml and 'calibrated_on = "AMD Instinct MI210"' in toml
    provenance = json.loads((case['out'] / 'provenance/mi210-test-task.json').read_text())
    assert provenance['calibration_status'] == 'provisional' and provenance['run']['status'] == 'incomplete'


@pytest.mark.parametrize('kwargs, message', [
    ({'score': 1.094}, 'under the benchmark minimum'),
    ({'verdict': 'mixed'}, "verdict is 'mixed'"),
    ({'correctness': 'FAIL'}, "correctness is 'FAIL'"),
])
def test_refuses_weak_evidence_and_writes_nothing(tmp_path, kwargs, message):
    case = make_case(tmp_path, **kwargs)
    with pytest.raises(pr.Refused, match=message):
        pr.build(**case)
    assert not case['out'].exists()


def test_refuses_a_candidate_that_is_not_what_the_round_check_measured(tmp_path):
    case = make_case(tmp_path)
    (tmp_path / 'cand/kernel.py').write_text(CANDIDATE + '# changed\n')
    with pytest.raises(pr.Refused, match='not the file the Round check measured'):
        pr.build(**case)


def test_refuses_an_instruction_naming_a_reference_only_identifier(tmp_path):
    case = make_case(tmp_path)
    (case['spec_dir'] / 'instruction.md').write_text('# On MI210, write fast_secret_kernel.\n')
    with pytest.raises(pr.Refused, match='gives away'):
        pr.build(**case)


def test_refuses_internal_paths_left_in_a_package(tmp_path):
    case = make_case(tmp_path)
    spec = json.loads((case['spec_dir'] / 'spec.json').read_text())
    spec['scrub'] = []
    (case['spec_dir'] / 'spec.json').write_text(json.dumps(spec))
    with pytest.raises(pr.Refused, match='internal names'):
        pr.build(**case)


@pytest.mark.parametrize('change, message', [
    ({'extra': 1}, 'differ from the dataset'),
    ({'reward_class': 'implementation'}, 'only performance rows'),
    ({'task_dir': 'wro-w8a16-groupdequant-matmul-sol'}, 'already exists'),
    ({'summary': 'too short on MI210'}, 'characters'),
    ({'summary': 'y' * 700}, 'must say the task runs on MI210'),
    ({'bench': 'kfc', 'max_submissions': 16}, 'not allowed'),
])
def test_row_checks(change, message):
    upstream = [json.loads(line) for line in (pr.VENDORED / 'upstream_task_catalog.jsonl').read_text().splitlines()]
    row = dict(zip(pr.COLUMNS, ('lh', 'mi210-x', 'infra', 'performance', 1, 16, 1.2, SUMMARY)))
    row.update(change)
    with pytest.raises(pr.Refused, match=message):
        pr.check_row(row, upstream)


def make_piece_case(tmp: Path, target=1.2, correctness='PASS') -> dict:
    case = make_case(tmp)
    rows = {'m1024': {'weight': 0.05, 'start_ms': target, 'candidate_ms': 1.0, 'verdict': 'faster',
                      'rounds_ms': {'A': [target] * 3, 'A2': [target] * 3}},
            'm2': {'weight': 0.95, 'start_ms': 1.0, 'candidate_ms': 1.05, 'verdict': 'slower',
                   'rounds_ms': {'A': [1.0] * 3, 'A2': [1.0] * 3}}}
    (case['round_check'] / 'verdict.json').write_text(json.dumps({'rows': rows, 'summary': {
        'verdict': 'not_faster', 'correctness': correctness, 'control_identical': True, 'cases_failed': [],
        'score_speedup': 1.0}}))
    (case['spec_dir'] / 'plan.json').write_text(json.dumps({
        'min_target_speedup': 1.1, 'slower_rows_scope': 'target',
        'pieces': [{'id': 'e1', 'start': 'served', 'patch': 'x', 'target_rows': ['m1024']}]}))
    spec = json.loads((case['spec_dir'] / 'spec.json').read_text())
    spec['piece'] = {'plan': 'plan.json', 'id': 'e1'}
    (case['spec_dir'] / 'spec.json').write_text(json.dumps(spec))
    return case


def test_a_piece_row_is_anchored_and_graded_on_its_target_rows(tmp_path):
    case = make_piece_case(tmp_path)
    result = pr.build(**case)
    assert result['row']['ref_speedup'] == 1.2
    package = case['out'] / 'tasks/lh/mi210-test-task'
    for tests in (package / 'tests', package / 'environment/loop/private/tests'):
        assert json.loads((tests / 'grade_config.json').read_text())['target_rows'] == ['m1024']
    provenance = json.loads((case['out'] / 'provenance/mi210-test-task.json').read_text())
    assert provenance['piece']['rows_slower'] == ['m2']


@pytest.mark.parametrize('kwargs, message', [
    ({'target': 1.05}, 'does not qualify'),
    ({'correctness': 'FAIL'}, 'does not qualify'),
])
def test_a_piece_that_fails_the_plan_writes_nothing(tmp_path, kwargs, message):
    case = make_piece_case(tmp_path, **kwargs)
    with pytest.raises(pr.Refused, match=message):
        pr.build(**case)
    assert not case['out'].exists()


def load_grade(tmp: Path, target_rows):
    (tmp / 'grade.py').write_bytes((pr.TEMPLATES / 'grade.py').read_bytes())
    (tmp / 'grade_config.json').write_text(json.dumps({'runner': 'r.py', 'target_rows': target_rows}))
    spec = importlib.util.spec_from_file_location('grade_under_test', tmp / 'grade.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OUTPUT = 'Perf: 2.0000 ms (m2)  weight=0.50\nPerf: 3.0000 ms (m1024)  weight=0.10\nWeighted sum: 1.3000 ms (sum)\n'


def test_grader_times_only_the_target_rows(tmp_path):
    assert load_grade(tmp_path, ['m1024']).metric_ms(OUTPUT) == pytest.approx(0.3)


def test_grader_keeps_the_weighted_sum_without_target_rows(tmp_path):
    assert load_grade(tmp_path, None).metric_ms(OUTPUT) == pytest.approx(1.3)


def test_grader_refuses_a_missing_target_row(tmp_path):
    with pytest.raises(SystemExit, match='no time for'):
        load_grade(tmp_path, ['m8192']).metric_ms(OUTPUT)


def test_rebuild_needs_force_and_replaces_the_row(tmp_path):
    case = make_case(tmp_path)
    pr.build(**case)
    with pytest.raises(pr.Refused, match='pass --force'):
        pr.build(**case)
    pr.build(**case, force=True)
    assert len((case['out'] / 'task_catalog.jsonl').read_text().splitlines()) == 1


def _reward(mode, speedup, hard=(), corr=1.0):
    reward = 0.0 if hard or speedup <= 1.2 else 0.5
    return {'mode': mode, 'speedup': speedup, 'reward': reward, 'hard_fail_reasons': list(hard),
            'correctness_frac': corr}


def _calibration(tmp_path, oracle, noop=None, negative=None):
    import json as _json
    runs = {'oracle': oracle, 'noop': noop or [_reward('noop', 1.001)],
            'negative': negative or [_reward('negative', 0, ['correctness_failed'], 0.5)]}
    for mode, rewards in runs.items():
        (tmp_path / f'{mode}.json').write_text(_json.dumps({'runs': [{'reward': r} for r in rewards]}))
    return tmp_path


def test_calibration_anchor_is_the_median_of_the_oracle_runs(tmp_path):
    cal = pr.check_calibration(_calibration(tmp_path, [_reward('oracle', x) for x in (1.20, 1.25, 1.22, 1.30, 1.21)]))
    assert cal['anchor'] == 1.22


@pytest.mark.parametrize('case', ['four_runs', 'oracle_hard_fail', 'wrong_mode', 'noop_scores', 'negative_passes',
                                  'under_minimum'])
def test_calibration_refuses_what_the_frozen_rules_refuse(tmp_path, case):
    oracle = [_reward('oracle', 1.2) for _ in range(5)]
    noop = negative = None
    if case == 'four_runs':
        oracle = oracle[:4]
    elif case == 'oracle_hard_fail':
        oracle[2] = _reward('oracle', 1.2, ['timing_failed'])
    elif case == 'wrong_mode':
        oracle[0] = _reward('candidate', 1.2)
    elif case == 'noop_scores':
        noop = [_reward('noop', 1.3)]
    elif case == 'negative_passes':
        negative = [_reward('negative', 1.0)]
    elif case == 'under_minimum':
        oracle = [_reward('oracle', 1.05) for _ in range(5)]
    with pytest.raises(pr.Refused):
        pr.check_calibration(_calibration(tmp_path, oracle, noop, negative))


def test_calibration_holds_only_for_the_package_it_ran_on(tmp_path):
    new, old = tmp_path / 'new', tmp_path / 'old'
    for root, grade in ((new, 'a'), (old, 'a')):
        (root / 'tests').mkdir(parents=True)
        (root / 'tests/grade.py').write_text(grade)
        (root / 'tests/ref_speedup.txt').write_text(str(len(str(root))))
    pr.same_package(new, old)  # only the anchor file differs
    (old / 'tests/grade.py').write_text('b')
    with pytest.raises(pr.Refused):
        pr.same_package(new, old)
