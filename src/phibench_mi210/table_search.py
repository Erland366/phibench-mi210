"""A solver with no kernel insight: search one tile-table entry, to see whether a one-entry task is too easy.

Runs inside the task image (python3, no third-party imports), from a copy of /app/repo:

    python3 phibench_table_search.py --plan PLAN.json --repo /app/repo --out OUT

The plan names the target token count, the arms (which table the entry goes into: "served" = configs.json,
"dec" = dec_configs.json), the trial budget (count and minutes) and the seed. Candidates, in order: every existing
entry of an arm's table copied to the target key, nearest key first, the arms interleaved; then random entries, each
field drawn from the values that field takes across that table's entries, the arms alternating. Each trial is one
`scripts/task_runner.py performance` run in a fresh copy of the repository; the target row's time is its score, and
a run that fails or prints no target row is a failed trial. The unchanged repository is timed at the start and after
every 16 trials. The 3 fastest trials are timed twice more; the one with the lowest median is the finalist, written
to OUT/finalist/ (the three editable files) for the package's own verifier.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TABLES = {'served': 'configs.json', 'dec': 'dec_configs.json'}
EDITABLE = ('moe_w8a16.py', 'configs.json', 'dec_configs.json')


def leaves(entry: dict, prefix=()):
    for key, value in entry.items():
        if isinstance(value, dict):
            yield from leaves(value, prefix + (key,))
        else:
            yield prefix + (key,), value


def set_leaf(entry: dict, path: tuple, value) -> None:
    for key in path[:-1]:
        entry = entry[key]
    entry[path[-1]] = value


def candidates(tables: dict, target: int, arms: list, rng: random.Random, count: int) -> list[dict]:
    """[{'arm', 'how', 'entry'}] in trial order."""
    copies = {}
    for arm in arms:
        table = tables[arm]
        keys = sorted((k for k in table if k != target), key=lambda k: (abs(k - target), k))
        copies[arm] = [{'arm': arm, 'how': f'copy {k}', 'entry': copy.deepcopy(table[k])} for k in keys]
    out = []
    while any(copies.values()):
        for arm in arms:
            if copies[arm]:
                out.append(copies[arm].pop(0))
    values = {}
    for arm in arms:
        pool = {}
        for entry in tables[arm].values():
            for path, value in leaves(entry):
                pool.setdefault(path, [])
                if value not in pool[path]:
                    pool[path].append(value)
        values[arm] = pool
    i = 0
    while len(out) < count:
        arm = arms[i % len(arms)]
        nearest = min(tables[arm], key=lambda k: (abs(k - target), k))
        entry = copy.deepcopy(tables[arm][nearest])
        for path, _ in leaves(tables[arm][nearest]):
            set_leaf(entry, path, rng.choice(values[arm].get(path, [None])))
        out.append({'arm': arm, 'how': f'random {i}', 'entry': entry})
        i += 1
    return out[:count]


def work_copy(repo: Path, tmp: Path, change: dict | None, target: int) -> Path:
    work = Path(tempfile.mkdtemp(dir=tmp))
    shutil.copytree(repo, work, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.git'))
    if change:
        path = work / TABLES[change['arm']]
        table = json.loads(path.read_text())
        table[str(target)] = change['entry']
        table = dict(sorted(table.items(), key=lambda kv: int(kv[0])))  # vLLM reads keys in file order
        path.write_text(json.dumps(table, indent=4))
    return work


def time_row(repo: Path, tmp: Path, change: dict | None, target: int, row: str) -> dict:
    work = work_copy(repo, tmp, change, target)
    start = time.time()
    run = subprocess.run([sys.executable, 'scripts/task_runner.py', 'performance'], cwd=work, capture_output=True,
                         text=True)
    found = re.findall(rf'^Perf: ([0-9.]+) ms \({row}\)', run.stdout, re.M)
    result = {'seconds': round(time.time() - start, 1), 'exit': run.returncode,
              'ms': float(found[-1]) if run.returncode == 0 and found else None}
    if result['ms'] is None:
        result['error'] = (run.stderr or run.stdout)[-600:]
    shutil.rmtree(work)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path('/app/repo'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text())
    target, row, arms = plan['target_tokens'], f'm{plan["target_tokens"]}', plan['arms']
    tables = {arm: {int(k): v for k, v in json.loads((args.repo / TABLES[arm]).read_text()).items()} for arm in arms}
    trials = candidates(tables, target, arms, random.Random(plan['seed']), plan['max_trials'])
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / 'trials.jsonl'
    tmp = Path(tempfile.mkdtemp(prefix='table-search-'))
    deadline = time.time() + 60 * plan['max_minutes']
    baseline, done = [time_row(args.repo, tmp, None, target, row)], []
    print(f"baseline {baseline[-1]}", flush=True)
    for i, trial in enumerate(trials):
        if time.time() > deadline:
            break
        result = {'trial': i, 'arm': trial['arm'], 'how': trial['how'], 'entry': trial['entry'],
                  **time_row(args.repo, tmp, trial, target, row)}
        done.append(result)
        with log.open('a') as fh:
            fh.write(json.dumps(result) + '\n')
        print(f"{i:3} {trial['arm']:6} {trial['how']:12} {result['ms']}", flush=True)
        if (i + 1) % 16 == 0:
            baseline.append(time_row(args.repo, tmp, None, target, row))
    timed = sorted((r for r in done if r['ms'] is not None), key=lambda r: r['ms'])[:3]
    for r in timed:
        r['retimes'] = [r['ms']] + [time_row(args.repo, tmp, r, target, row)['ms'] for _ in range(2)]
        r['median_ms'] = statistics.median(x for x in r['retimes'] if x is not None)
    baseline.append(time_row(args.repo, tmp, None, target, row))
    base_ms = statistics.median(b['ms'] for b in baseline if b['ms'] is not None)
    summary = {'plan': plan, 'trials_run': len(done), 'failed': sum(r['ms'] is None for r in done),
               'baseline_ms': base_ms, 'baseline_runs': baseline, 'top3': timed}
    if timed:
        best = min(timed, key=lambda r: r['median_ms'])
        summary['finalist'] = {k: best[k] for k in ('trial', 'arm', 'how', 'entry', 'median_ms')}
        summary['finalist']['row_speedup'] = base_ms / best['median_ms']
        final = work_copy(args.repo, tmp, best, target)
        (args.out / 'finalist').mkdir(exist_ok=True)
        for name in EDITABLE:
            shutil.copy(final / name, args.out / 'finalist' / name)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({k: summary[k] for k in ('trials_run', 'failed', 'baseline_ms')} |
                     {'finalist': summary.get('finalist', {}).get('how'),
                      'row_speedup': summary.get('finalist', {}).get('row_speedup')}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
