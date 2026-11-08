#!/usr/bin/env python3
"""Grade /app/repo with the frozen task runner in grader/ (never the runner in the repo).

    python3 grade.py identity | correctness | timing

Prints one `GRADE_RESULT {json}` line. grade_config.json holds the expected runtime, the correctness seeds and the
number of ABBA blocks: tests/ uses the held-out seeds, the in-session copy in /opt/loop/private/tests the public ones.
With target_rows set, the timed metric is the sum of weight x time over those rows instead of the runner's weighted sum.
"""
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/app/repo')
CONFIG = json.loads((HERE / 'grade_config.json').read_text())
RUNNER = CONFIG['runner']


def result(payload: dict) -> None:
    print('GRADE_RESULT ' + json.dumps(payload), flush=True)


def identity() -> int:
    import torch
    from importlib.metadata import version
    want, problems = CONFIG['runtime'], []
    if not torch.__version__.startswith(want['torch_prefix']):
        problems.append(f'torch {torch.__version__}, expected {want["torch_prefix"]}*')
    vllm_version = version('vllm')
    if not vllm_version.startswith(want['vllm_prefix']):
        problems.append(f'vllm {vllm_version}, expected {want["vllm_prefix"]}*')
    if not torch.cuda.is_available():
        problems.append('no GPU visible: pass /dev/kfd and /dev/dri to the container')
    else:
        arch = getattr(torch.cuda.get_device_properties(0), 'gcnArchName', '')
        if want['arch'] not in arch:
            problems.append(f'GPU arch {arch!r}, expected {want["arch"]} (the anchor is calibrated on MI210)')
    result({'ok': not problems, 'problems': problems})
    return 1 if problems else 0


def work_tree(root: Path, source: str) -> Path:
    """The frozen grader files plus the scope files: from the working tree or from the baseline at HEAD."""
    work = root / source
    shutil.copytree(HERE / 'grader', work)
    for name in CONFIG['scope']:
        if source == 'candidate':
            shutil.copyfile(REPO / name, work / name)
        else:
            shown = subprocess.run(['git', '-C', str(REPO), 'show', f'HEAD:{name}'], check=True, capture_output=True)
            (work / name).write_bytes(shown.stdout)
    return work


def correctness() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        work = work_tree(Path(tmp), 'candidate')
        spec = importlib.util.spec_from_file_location('task_runner', work / RUNNER)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        runner.SEEDS = tuple(CONFIG['correctness_seeds'])
        try:
            rc = runner.cmd_correctness()
        except SystemExit as stop:  # the runner's static check and frozen-file check exit non-zero
            print(stop, flush=True)
            rc = 1
    result({'correctness_frac': 1.0 if rc == 0 else 0.0})
    return 0


def metric_ms(stdout: str) -> float:
    targets = CONFIG.get('target_rows')
    if not targets:
        found = re.findall(r'^Weighted sum: ([0-9.]+) ms', stdout, re.M)
        if not found:
            raise SystemExit('performance printed no weighted sum')
        return float(found[-1])
    rows = {row: (float(ms), float(weight)) for ms, row, weight in
            re.findall(r'^Perf: ([0-9.]+) ms \((m[0-9]+)\)\s+weight=([0-9.]+)', stdout, re.M)}
    missing = [row for row in targets if row not in rows]
    if missing:
        raise SystemExit(f'performance printed no time for {missing}')
    return sum(rows[row][0] * rows[row][1] for row in targets)


def weighted_ms(work: Path) -> float:
    run = subprocess.run([sys.executable, RUNNER, 'performance'], cwd=work, capture_output=True, text=True,
                         env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    print(run.stdout, flush=True)
    if run.returncode:
        raise SystemExit(f'performance failed (exit {run.returncode}): {run.stderr[-800:]}')
    return metric_ms(run.stdout)


def timing() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        base, cand = work_tree(Path(tmp), 'baseline'), work_tree(Path(tmp), 'candidate')
        blocks = []
        for _ in range(CONFIG['abba_blocks']):
            a1, b1, b2, a2 = weighted_ms(base), weighted_ms(cand), weighted_ms(cand), weighted_ms(base)
            blocks.append({'baseline_ms': math.sqrt(a1 * a2), 'candidate_ms': math.sqrt(b1 * b2)})
    ratios = [b['baseline_ms'] / b['candidate_ms'] for b in blocks]
    if len(ratios) > 1 and len(set(ratios)) == 1:
        raise SystemExit('every paired ratio is identical: the measurement is not real')
    result({'baseline_ms': statistics.median(b['baseline_ms'] for b in blocks),
            'candidate_ms': statistics.median(b['candidate_ms'] for b in blocks),
            'speedup': statistics.median(ratios), 'blocks': blocks})
    return 0


if __name__ == '__main__':
    commands = {'identity': identity, 'correctness': correctness, 'timing': timing}
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        raise SystemExit(f'usage: grade.py {"|".join(commands)}')
    sys.exit(commands[sys.argv[1]]())
