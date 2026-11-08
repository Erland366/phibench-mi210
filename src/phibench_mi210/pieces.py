"""Measure the separate pieces of a GEAK result, each scored only on the rows it changes.

    # host, CPU: one candidate directory and one Round check work directory per piece
    python -m evaluation.geak.phibench_pieces build --plan PLAN.json --results RESULTS --out OUT

    # one MI210, one piece at a time, nothing else on the node: the normal Round check on OUT/<piece>/work
    ./k2 py --gpus N --image v0.32.0 --dev --mount WORKTREE --mount OUT -- \
        WORKTREE/evaluation/geak/round_check_jais3.py measure --work-dir OUT/<piece>/work

    # host, CPU: apply the plan's frozen rule to the verdicts
    python -m evaluation.geak.phibench_pieces score --plan PLAN.json --out OUT

A piece starts from the served kernel (the task's baseline_* files, the Round check's arm A) or from the task's
stacked starting files (arm S). Its target score is sum(weight x start time) / sum(weight x candidate time) over its
target rows. Its noise band comes from the two served arms A and A2 on the same rows, as the Round check's own score.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from evaluation.geak import round_check_jais3 as rc

FAILED_CHECKS = ('fail_correctness', 'fail_control', 'fail_rules')


def build_piece(piece: dict, task: str, results: Path, out: Path, tasks_root: Path = rc.TASKS_DIR) -> Path:
    """Candidate directory: the piece's start files with its patch applied. Refuses when the patch does not apply."""
    candidate = out / piece['id'] / 'candidate'
    candidate.mkdir(parents=True)
    for editable, baseline in rc.TASKS[task].items():
        source = baseline if piece['start'] == 'served' else editable
        shutil.copy(tasks_root / task / source, candidate / editable)
    patch = results / piece['patch']
    shutil.copy(patch, out / piece['id'] / 'piece.patch')
    subprocess.run(['git', 'apply', str(patch.resolve())], cwd=candidate, check=True)
    return candidate


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
    noise = rc.noise_band(gaps)
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


def cmd_build(plan: dict, results: Path, out: Path) -> int:
    for piece in plan['pieces']:
        candidate = build_piece(piece, plan['task'], results, out)
        code = rc.main(['prepare', '--task', plan['task'], '--candidate', str(candidate),
                        '--out-dir', str(out / piece['id'] / 'work')])
        if code != 0:
            print(f"{piece['id']}: prepare exited {code}, not measurable", file=sys.stderr)
    return 0


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
    parser.add_argument('command', choices=('build', 'score'))
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--results', type=Path, help='build: the directory the plan\'s patch paths start from')
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    if args.command == 'build':
        if args.results is None:
            parser.error('build needs --results')
        return cmd_build(plan, args.results, args.out)
    return cmd_score(plan, args.out)


if __name__ == '__main__':
    sys.exit(main())
