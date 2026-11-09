"""CPU-only tests of the per-piece scoring rule."""
import pytest

from phibench_mi210 import pieces as pp


def row(weight, start, candidate, stacked=None, state='faster', stacked_state='faster'):
    record = {'weight': weight, 'start_ms': start, 'candidate_ms': candidate, 'verdict': state,
              'rounds_ms': {'A': [start] * 3, 'A2': [start] * 3}}
    if stacked is not None:
        record['vs_stacked_start'] = {'stacked_ms': stacked, 'verdict': stacked_state}
    return record


def verdict(rows, state='pass', correctness='PASS', control=True):
    return {'rows': rows, 'summary': {'verdict': state, 'correctness': correctness, 'control_identical': control}}


SERVED = {'id': 'p', 'start': 'served', 'target_rows': ['m1']}
STACKED = {'id': 'p', 'start': 'stacked', 'target_rows': ['m1']}


def test_target_score_uses_only_the_target_rows():
    rows = {'m1': row(0.5, 1.2, 1.0), 'm2': row(0.5, 1.0, 1.0, state='within noise')}
    score = pp.score_piece(SERVED, verdict(rows), 1.1)
    assert score['target_speedup'] == pytest.approx(1.2)
    assert score['whole_layer_speedup'] == pytest.approx(2.2 / 2.0)
    assert score['qualifies']


def test_a_stacked_piece_is_compared_with_the_stacked_files_not_the_served_kernel():
    rows = {'m1': row(1.0, 2.0, 1.0, stacked=1.05)}
    score = pp.score_piece(STACKED, verdict(rows), 1.1)
    assert score['target_speedup'] == pytest.approx(1.05)
    assert not score['qualifies'] and 'under 1.1x' in score['reasons'][0]


def test_a_slower_row_outside_the_target_disqualifies():
    rows = {'m1': row(0.5, 1.2, 1.0), 'm2': row(0.5, 1.0, 1.1, state='slower')}
    score = pp.score_piece(SERVED, verdict(rows), 1.1)
    assert not score['qualifies'] and score['rows_slower'] == ['m2']


def test_target_scope_reports_but_ignores_slower_rows_the_piece_does_not_change():
    rows = {'m1': row(0.5, 1.2, 1.0), 'm2': row(0.5, 1.0, 1.1, state='slower')}
    score = pp.score_piece(SERVED, verdict(rows), 1.1, 'target')
    assert score['qualifies'] and score['rows_slower'] == ['m2']


def test_target_scope_still_fails_a_slower_target_row():
    piece = {**SERVED, 'target_rows': ['m1', 'm2']}
    rows = {'m1': row(0.5, 2.0, 1.0), 'm2': row(0.5, 1.0, 1.1, state='slower')}
    assert not pp.score_piece(piece, verdict(rows), 1.1, 'target')['qualifies']


@pytest.mark.parametrize('kwargs', [{'state': 'fail_rules'}, {'correctness': 'FAIL'}, {'control': False}])
def test_a_failed_round_check_disqualifies_however_fast(kwargs):
    assert not pp.score_piece(SERVED, verdict({'m1': row(1.0, 2.0, 1.0)}, **kwargs), 1.1)['qualifies']


def test_a_gain_inside_the_noise_band_disqualifies():
    record = row(1.0, 1.11, 1.0)
    record['rounds_ms'] = {'A': [1.0] * 3, 'A2': [1.2] * 3}
    score = pp.score_piece(SERVED, verdict({'m1': record}), 1.1)
    assert not score['qualifies'] and 'noise band' in score['reasons'][0]


def test_the_noise_band_is_the_round_checks_rule():
    assert pp.noise_band([0.001] * 10) == 0.02  # the 2% floor
    assert pp.noise_band([0.01] * 9 + [0.5]) == 0.02  # one outlier round of 10 does not set the band
    assert pp.noise_band([0.03] * 8 + [0.04, 0.5]) == 0.04
