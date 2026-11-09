"""Score the separate pieces of a GEAK result, each on the rows it changes, with a plan's frozen rule.

    python -m phibench_mi210.pieces --plan PLAN.json --out OUT

Building and measuring a piece needs the GEAK task files and the Round check, so it stays in deploy_bbq
(`python -m evaluation.geak.phibench_pieces build`, then `round_check_jais3.py measure` on each OUT/<piece>/work, one
at a time on an idle MI210 node). This step only reads OUT/<piece>/work/verdict.json; see docs/inputs.md.

A piece starts from the served kernel (the task's baseline_* files, the Round check's arm A) or from the task's
stacked starting files (arm S). Its target score is sum(weight x start time) / sum(weight x candidate time) over its
target rows. Its noise band comes from the two served arms A and A2 on the same rows, as the Round check's own score.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

FAILED_CHECKS = ('fail_correctness', 'fail_control', 'fail_rules')

# The Round check's noise rule (deploy_bbq evaluation/geak/round_check_jais3.py, rule of 2026-10-29): the band is
# max(2%, the 90th percentile, nearest rank, of the per-round |A2/A - 1| gaps). Copied so that this repository needs
# no GEAK code; tests/test_pieces.py pins it.
MIN_NOISE = 0.02
NOISE_PERCENTILE = 0.90


def nearest_rank(values: list[float], q: float) -> float:
    """The ceil(q x n)-th smallest value (nearest-rank percentile)."""
    if not values:
        raise ValueError('no values')
    ordered = sorted(values)
    return ordered[min(len(ordered), max(1, math.ceil(q * len(ordered)))) - 1]


def noise_band(gaps: list[float]) -> float:
    return max(MIN_NOISE, nearest_rank(gaps, NOISE_PERCENTILE))


def score_piece(piece: dict, verdict: dict, min_speedup: float, slower_scope: str = 'all') -> dict:
    """The plan's frozen rule on one piece's verdict.json. slower_scope: which rows fail the piece when slower, 'all'
    14 or only its 'target' rows (the rows it changes); slower rows elsewhere are still reported."""
    rows, summary = verdict['rows'], verdict['summary']
    served = piece['start'] == 'served'

    def start_ms(row):
        return row['start_ms'] if served else row['vs_stacked_start']['stacked_ms']

    def row_state(row):
        return row['verdict'] if served else row['vs_stacked_start']['verdict']

    target = {k: rows[k] for k in piece['target_rows']}
    target_start = sum(r['weight'] * start_ms(r) for r in target.values())
    target_candidate = sum(r['weight'] * r['candidate_ms'] for r in target.values())
    speedup = target_start / target_candidate
    gaps = []
    for i in range(len(next(iter(target.values()))['rounds_ms']['A'])):
        a = sum(r['weight'] * r['rounds_ms']['A'][i] for r in target.values())
        a2 = sum(r['weight'] * r['rounds_ms']['A2'][i] for r in target.values())
        gaps.append(abs(a2 / a - 1))
    noise = noise_band(gaps)
    whole_start = sum(r['weight'] * start_ms(r) for r in rows.values())
    whole_candidate = sum(r['weight'] * r['candidate_ms'] for r in rows.values())
    slower = [k for k, r in rows.items() if row_state(r) == 'slower']
    failing = slower if slower_scope == 'all' else [k for k in slower if k in target]
    checks_ok = (summary['correctness'] == 'PASS' and summary['control_identical']
                 and summary['verdict'] not in FAILED_CHECKS)
    reasons = []
    if not checks_ok:
        reasons.append(f"Round check: {summary['verdict']}, correctness {summary['correctness']}")
    if speedup < min_speedup:
        reasons.append(f'target {speedup:.3f}x is under {min_speedup}x')
    if speedup <= 1 + noise:
        reasons.append(f'target {speedup:.3f}x is inside its noise band {noise:.3f}')
    if failing:
        reasons.append('slower rows: ' + ', '.join(failing))
    return {'id': piece['id'], 'start': piece['start'], 'target_rows': piece['target_rows'], 'target_speedup': speedup,
            'target_noise': noise, 'whole_layer_speedup': whole_start / whole_candidate, 'rows_slower': slower,
            'row_speedups': {k: start_ms(r) / r['candidate_ms'] for k, r in rows.items()},
            'round_check_verdict': summary['verdict'], 'correctness': summary['correctness'],
            'qualifies': not reasons, 'reasons': reasons}


def cmd_score(plan: dict, out: Path) -> int:
    scores = []
    for piece in plan['pieces']:
        path = out / piece['id'] / 'work' / 'verdict.json'
        if not path.exists():
            print(f"{piece['id']}: no verdict yet")
            continue
        scores.append(score_piece(piece, json.loads(path.read_text()), plan['min_target_speedup'],
                                  plan.get('slower_rows_scope', 'all')))
    (out / 'scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    for s in scores:
        state = 'QUALIFIES' if s['qualifies'] else 'no: ' + '; '.join(s['reasons'])
        print(f"{s['id']:24} target {s['target_speedup']:.3f}x (noise {s['target_noise']:.3f}) "
              f"whole layer {s['whole_layer_speedup']:.3f}x  slower rows {s['rows_slower'] or 'none'}  {state}")
    if scores:
        print(f"median target {statistics.median(s['target_speedup'] for s in scores):.3f}x")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='the directory deploy_bbq built the pieces in')
    args = parser.parse_args(argv)
    return cmd_score(json.loads(args.plan.read_text()), args.out)


if __name__ == '__main__':
    sys.exit(main())
