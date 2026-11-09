"""Turn a finished GEAK run into a Phi-Bench task package and one catalog row (CPU only).

    python3 -m evaluation.geak.phibench_row \
        --spec evaluation/geak/phibench_specs/<task> --run-dir RUN --round-check RUN.round-check --out OUT

The independent Round check decides: it names the exact candidate files, and its measured speedup becomes the anchor
(GEAK's own claim is never used). The row keeps the dataset's 8 columns; hardware lives in the package. The spec holds
what cannot be generated: names, the summary and instruction (no solution hints) and a deliberately wrong edit.

A spec with `piece` {plan, id} makes a row from one piece of a result (phibench_pieces.py): --round-check is that
piece's work directory, the plan's frozen rule replaces the whole-layer test, the anchor is the score on the piece's
target rows, and the grader times only those rows.

Any failed check raises Refused and writes nothing. A package made here is `provisional` until its own verifier has
calibrated the anchor on the GPU (at least 5 runs) and its no-change and negative self-checks have run.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path

from evaluation.geak.phibench_pieces import score_piece

HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / 'phibench_templates'
VENDORED = TEMPLATES / 'vendored'

COLUMNS = ('bench', 'task_dir', 'infra_name', 'reward_class', 'gpus', 'max_submissions', 'ref_speedup', 'summary')
MIN_REF_SPEEDUP = 1.1  # arXiv 2609.10226: "a stable performance ratio of at least 1.1"
SUMMARY_CHARS = (629, 1101)  # the range of the 85 published summaries (dataset revision in vendored/NOTICE.md)
VENDOR_REFERENCE_TASK = 'wro-w8a16-groupdequant-matmul-sol'
# never in a package: internal locations and names
DENY_EVERYWHERE = re.compile(r'/data/erland|/home/erlan|auh7-|k2-amd-project|erland\.pg')
# never where the solver can read it, besides the identifiers only the reference has (hidden_identifiers)
DENY_FOR_SOLVER = re.compile(r'\bGEAK\b|\bround[ -]?2\b|current_best|final_patch|\bcandidate of\b', re.I)
TEXT_SUFFIXES = {'.py', '.md', '.json', '.sh', '.toml', '.tmpl', '.txt', '.patch', '.jsonl', ''}


class Refused(Exception):
    """A check failed: nothing is written."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise Refused(f'missing {path}')
    return json.loads(path.read_text())


def scrub(data: bytes, pairs: list[list[str]]) -> bytes:
    text = data.decode()
    for old, new in pairs:
        text = text.replace(old, new)
    return text.encode()


def check_evidence(verdict: dict, outcome: dict) -> float:
    """The Round check must have passed in full; returns its measured speedup."""
    if verdict.get('verdict') == 'no_candidate':
        raise Refused(f'the run produced no candidate: {verdict.get("reason", "")}')
    summary = verdict.get('summary')
    if not isinstance(summary, dict):
        raise Refused('the Round check verdict has no summary')
    for key, want in (('verdict', 'pass'), ('correctness', 'PASS'), ('control_identical', True)):
        if summary.get(key) != want:
            raise Refused(f'Round check {key} is {summary.get(key)!r}, needed {want!r}')
    if summary.get('cases_failed'):
        raise Refused(f'Round check failed cases: {summary["cases_failed"]}')
    speedup = summary.get('score_speedup')
    if not isinstance(speedup, (int, float)) or speedup < MIN_REF_SPEEDUP:
        raise Refused(f'measured speedup {speedup} is under the benchmark minimum of {MIN_REF_SPEEDUP}x')
    if outcome.get('models_outside_cap'):
        raise Refused(f'the run used models outside the cap: {outcome["models_outside_cap"]}')
    return float(speedup)


def check_piece_evidence(piece_ref: dict, spec_dir: Path, verdict: dict, outcome: dict) -> tuple[float, dict]:
    """The plan's frozen rule, computed again from the Round check's verdict; returns the anchor and the score."""
    plan = load_json(spec_dir / piece_ref['plan'])
    pieces = [p for p in plan['pieces'] if p['id'] == piece_ref['id']]
    if len(pieces) != 1:
        raise Refused(f'the plan has {len(pieces)} pieces named {piece_ref["id"]!r}')
    if pieces[0]['start'] != 'served':
        raise Refused('a piece row needs a piece measured from the served kernel, the start of the task')
    score = score_piece(pieces[0], verdict, plan['min_target_speedup'], plan.get('slower_rows_scope', 'all'))
    if not score['qualifies']:
        raise Refused('the piece does not qualify: ' + '; '.join(score['reasons']))
    if verdict['summary'].get('cases_failed'):
        raise Refused(f'Round check failed cases: {verdict["summary"]["cases_failed"]}')
    if score['target_speedup'] < MIN_REF_SPEEDUP:
        raise Refused(f'target speedup {score["target_speedup"]} is under the benchmark minimum of {MIN_REF_SPEEDUP}x')
    if outcome.get('models_outside_cap'):
        raise Refused(f'the run used models outside the cap: {outcome["models_outside_cap"]}')
    return float(score['target_speedup']), score


def check_row(row: dict, upstream_rows: list[dict]) -> None:
    """The row must match the published dataset exactly in columns, order and types."""
    if not upstream_rows or tuple(upstream_rows[0]) != COLUMNS:
        raise Refused(f'the vendored catalog no longer has the expected columns: {list(upstream_rows[0])}')
    if tuple(row) != COLUMNS:
        raise Refused(f'row columns {list(row)} differ from the dataset {list(COLUMNS)}')
    if row['bench'] not in {r['bench'] for r in upstream_rows}:
        raise Refused(f'unknown bench {row["bench"]!r}')
    if row['reward_class'] != 'performance':
        raise Refused(f'only performance rows can come from a GEAK run, not {row["reward_class"]!r}')
    if row['gpus'] not in (0, 1) or isinstance(row['gpus'], bool):
        raise Refused(f'gpus must be 0 or 1, not {row["gpus"]!r}')
    budget = {'lh': (1, 16), 'e2e': (1, 16), 'kfc': (1,)}[row['bench']]
    if row['max_submissions'] not in budget:
        raise Refused(f'max_submissions {row["max_submissions"]} not allowed for {row["bench"]}: {budget}')
    if not isinstance(row['ref_speedup'], float) or row['ref_speedup'] <= 1:
        raise Refused(f'ref_speedup must be a number above 1, not {row["ref_speedup"]!r}')
    if row['task_dir'] in {r['task_dir'] for r in upstream_rows}:
        raise Refused(f'task_dir {row["task_dir"]!r} already exists in the dataset')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]*', row['task_dir']):
        raise Refused(f'task_dir {row["task_dir"]!r} must be lowercase with hyphens')
    if not SUMMARY_CHARS[0] <= len(row['summary']) <= SUMMARY_CHARS[1]:
        raise Refused(f'summary has {len(row["summary"])} characters, published range is {SUMMARY_CHARS}')
    if 'MI210' not in row['summary']:
        raise Refused('the summary must say the task runs on MI210: the row has no hardware column')


def hidden_identifiers(reference: dict[str, bytes], start: dict[str, bytes]) -> set[str]:
    """Code names only the reference has: a solver-visible file naming one gives the solution away."""
    found: set[str] = set()
    for name, data in reference.items():
        if name.endswith('.py'):
            for tok in tokenize.generate_tokens(io.StringIO(data.decode()).readline):
                if tok.type == tokenize.NAME:
                    found.add(tok.string)
        else:
            found.update(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', data.decode()))
    # a word the starting files already contain, even in a comment, gives nothing away
    known = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', '\n'.join(d.decode() for d in start.values())))
    return {n for n in found - known if '_' in n.strip('_') or any(c.isupper() for c in n)}


def scan_solver_visible(files: dict[str, str], hidden: set[str]) -> None:
    pattern = re.compile(r'(?<![A-Za-z0-9_])(' + '|'.join(map(re.escape, sorted(hidden))) + r')(?![A-Za-z0-9_])') \
        if hidden else None
    problems = []
    for name, text in files.items():
        for number, line in enumerate(text.splitlines(), 1):
            hit = DENY_FOR_SOLVER.search(line) or (pattern.search(line) if pattern else None)
            if hit:
                problems.append(f'{name}:{number}: {hit.group(0)!r}')
    if problems:
        raise Refused('solver-visible text gives away the solution or its origin:\n  ' + '\n  '.join(problems[:20]))


def scan_package(package: Path) -> None:
    problems = []
    for path in sorted(p for p in package.rglob('*') if p.is_file()):
        if path.suffix not in TEXT_SUFFIXES:
            continue
        hit = DENY_EVERYWHERE.search(path.read_text(errors='ignore'))
        if hit:
            problems.append(f'{path.relative_to(package)}: {hit.group(0)!r}')
    if problems:
        raise Refused('internal names in the package:\n  ' + '\n  '.join(problems[:20]))


def replace_docstring(source: str, docstring: str) -> str:
    node = ast.parse(source).body[0]
    if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
        raise Refused('the task runner has no module docstring to replace')
    lines = source.splitlines(keepends=True)
    return '"""' + docstring + '"""\n' + ''.join(lines[node.end_lineno:])


def make_patch(start: dict[str, bytes], end: dict[str, bytes]) -> str:
    """A -p1 patch from start to end, checked by applying it forward."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        env = ['-c', 'user.email=row@local', '-c', 'user.name=row']
        def git(*args: str) -> str:
            return subprocess.run(['git', *env, '-C', tmp, *args], check=True, capture_output=True, text=True).stdout
        git('init', '-q')
        for name, data in start.items():
            (repo / name).write_bytes(data)
        git('add', '-A')
        git('commit', '-qm', 'start')
        for name, data in end.items():
            (repo / name).write_bytes(data)
        patch = git('diff', '--no-color')
        if not patch.strip():
            raise Refused('the patch is empty: the end state equals the start')
        git('checkout', '-q', '--', '.')
        (repo / '.p').write_text(patch)
        git('apply', '-p1', '.p')
        for name, data in end.items():
            if (repo / name).read_bytes() != data:
                raise Refused(f'the patch does not reproduce {name}')
    return patch


def negative_edit(start: dict[str, bytes], edits: list[dict]) -> dict[str, bytes]:
    if not edits:
        raise Refused('the spec has no negative edit: each target needs a deliberately wrong version')
    end = dict(start)
    for edit in edits:
        text = end[edit['file']].decode()
        if text.count(edit['find']) != 1:
            raise Refused(f'negative edit must match exactly once in {edit["file"]}, found {text.count(edit["find"])}')
        end[edit['file']] = text.replace(edit['find'], edit['replace']).encode()
    return end


def bash_words(words: list[str]) -> str:
    return ' '.join('"' + w.replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$') + '"' for w in words)


def fill(template: str, values: dict[str, str]) -> str:
    for key, value in values.items():
        template = template.replace(f'@@{key}@@', value)
    left = re.findall(r'@@[A-Z_]+@@', template)
    if left:
        raise Refused(f'template placeholders not filled: {left}')
    return template


def write(path: Path, data: str | bytes, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    if mode is not None:
        path.chmod(mode)


def task_toml(spec: dict, image: str, scope: list[str], ref: float) -> str:
    paths = ', '.join(json.dumps(s) for s in scope)
    return f'''schema_version = "2.0"

artifacts = ["/logs/artifacts/model.patch", "/logs/artifacts/model_files.txt"]

[task]
id = {json.dumps(spec['task_dir'])}
bench = {json.dumps(spec['bench'])}
kind = "performance"
language = "python"
title = {json.dumps(spec['title'])}
instruction = "instruction.md"

[agent]
timeout_sec = {spec['agent_timeout_sec']}

[environment]
docker_image = {json.dumps(image)}
dockerfile = "environment/Dockerfile"
build_timeout_sec = 3600
cpus = {spec['cpus']}
memory_mb = {spec['memory_mb']}
gpus = {spec['gpus']}
gpu_types = ["MI210"]
allow_internet = false

[verifier]
entrypoint = "tests/test.sh"
mode = "shared"
task_kind = "acceleration"
contract_pattern = "kernelbench_generic_v1"
primary_edit_paths = [{paths}]
forbidden_edit_paths = ["*"]
result_paths = ["/logs/verifier/reward.json"]

[primary_metric]
name = "speedup"
goal = "maximize"
ref_speedup = {ref}
ref_speedup_source = "tests/ref_speedup.txt"
calibrated_on = "AMD Instinct MI210"

[loop]
min_submissions = 1
max_submissions = {spec['max_submissions']}
submit = "bash /opt/loop/submit.sh"
'''


def write_tests(root: Path, spec: dict, scope: list[str], grader: dict[str, bytes], seeds: list[int], blocks: int,
                ref_text: str | None, caveat: str | None, manifest: str | None,
                target_rows: list[str] | None = None) -> None:
    values = {'TASK_ID': spec['task_dir'], 'SCOPE_TEXT': ', '.join(scope), 'SCOPE_BASH': bash_words(scope),
              'BANNED_BASH': bash_words(spec['banned_tokens'])}
    write(root / 'test.sh', fill((TEMPLATES / 'test.sh.tmpl').read_text(), values), 0o755)
    shutil.copyfile(TEMPLATES / 'compute_reward.py', root / 'compute_reward.py')
    shutil.copyfile(TEMPLATES / 'grade.py', root / 'grade.py')
    write(root / 'grade_config.json', json.dumps(
        {'runner': spec['runner'], 'scope': scope, 'correctness_seeds': seeds, 'abba_blocks': blocks,
         'target_rows': target_rows,
         'runtime': {k: spec['runtime'][k] for k in ('torch_prefix', 'vllm_prefix', 'arch')}}, indent=2) + '\n')
    for name, data in grader.items():
        write(root / 'grader' / name, data)
    if ref_text is not None:
        write(root / 'ref_speedup.txt', ref_text + '\n')
        write(root / 'ref_speedup.caveat.md', caveat)
        write(root / 'verifier-correctness-manifest.json', manifest)


MIN_CALIBRATION_RUNS = 5  # arXiv 2609.10226: the anchor is the median of at least 5 runs
ANCHOR_FILES = ('task.toml', 'tests/ref_speedup.txt', 'tests/ref_speedup.caveat.md',
                'tests/verifier-correctness-manifest.json')  # the only files the anchor changes


def check_calibration(cal_dir: Path) -> dict:
    """The package verifier's own runs (phibench_enroot.py score: oracle.json, noop.json, negative.json): at least 5
    oracle runs with no hard fail, a no-change run scoring 0 without failing correctness, and a negative run failing
    correctness. Returns the anchor (median oracle speedup) and the evidence."""
    runs = {}
    for mode in ('oracle', 'noop', 'negative'):
        path = cal_dir / f'{mode}.json'
        if not path.is_file():
            raise Refused(f'calibration: {path} missing')
        runs[mode] = [r['reward'] for r in load_json(path)['runs']]
        if not runs[mode] or any(r is None or r['mode'] != mode for r in runs[mode]):
            raise Refused(f'calibration: a {mode} run has no reward, or the verifier ran another mode')
    oracle = runs['oracle']
    if len(oracle) < MIN_CALIBRATION_RUNS:
        raise Refused(f'calibration: {len(oracle)} oracle runs, at least {MIN_CALIBRATION_RUNS} needed')
    failed = [r['hard_fail_reasons'] for r in oracle if r['hard_fail_reasons'] or r['correctness_frac'] < 1]
    if failed:
        raise Refused(f'calibration: oracle runs failed: {failed}')
    if any(r['reward'] != 0 or 'correctness_failed' in r['hard_fail_reasons'] for r in runs['noop']):
        raise Refused('calibration: the no-change self-check scored above 0 or failed correctness')
    if any(r['reward'] != 0 or 'correctness_failed' not in r['hard_fail_reasons'] for r in runs['negative']):
        raise Refused('calibration: the negative self-check did not fail correctness')
    speedups = [r['speedup'] for r in oracle]
    anchor = round(statistics.median(speedups), 6)
    if anchor < MIN_REF_SPEEDUP:
        raise Refused(f'calibration: the calibrated anchor {anchor} is under the paper\'s {MIN_REF_SPEEDUP}x minimum')
    return {'dir': str(cal_dir), 'anchor': anchor, 'oracle_speedups': speedups,
            'noop_speedups': [r['speedup'] for r in runs['noop']],
            'negative_hard_fails': [r['hard_fail_reasons'] for r in runs['negative']]}


def same_package(new: Path, calibrated: Path) -> None:
    """Calibration holds only for the package it ran on: every file but the anchor's must be unchanged."""
    if not calibrated.is_dir():
        raise Refused(f'calibration: {calibrated} does not exist, so nothing was calibrated')
    files = lambda root: {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()
                          and str(p.relative_to(root)) not in ANCHOR_FILES}
    a, b = files(new), files(calibrated)
    differ = sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    if differ:
        raise Refused(f'calibration: the package differs from the calibrated one in {differ[:5]}')


def build(spec_dir: Path, run_dir: Path, round_check: Path, out: Path, force: bool = False,
          calibration: Path | None = None) -> dict:
    spec = load_json(spec_dir / 'spec.json')
    verdict = load_json(round_check / 'verdict.json')
    outcome = load_json(run_dir / 'outcome.json')
    piece_score = None
    if 'piece' in spec:
        speedup, piece_score = check_piece_evidence(spec['piece'], spec_dir, verdict, outcome)
        ref = round(speedup, 6)
    else:
        ref = round(check_evidence(verdict, outcome), 6)
    target_rows = piece_score['target_rows'] if piece_score else None
    cal = check_calibration(calibration) if calibration else None
    if cal:
        ref = cal['anchor']

    prepare = load_json(round_check / 'prepare.json')
    scope = sorted(spec['baseline_of'])
    if sorted(prepare['editable']) != scope:
        raise Refused(f'spec scope {scope} differs from the task editable files {sorted(prepare["editable"])}')
    if prepare['source'].get('kind') != 'directory':
        raise Refused(f'candidate source kind {prepare["source"].get("kind")!r} is not supported yet')
    candidate_dir = Path(prepare['source']['candidate_dir'])
    task_dir = round_check / 'task'

    # the candidate and the frozen files must be the very bytes the Round check measured
    raw_candidate = {}
    for name in scope:
        raw_candidate[name] = (candidate_dir / name).read_bytes()
        if sha256(raw_candidate[name]) != prepare['candidate'][name]['sha256']:
            raise Refused(f'{candidate_dir / name} is not the file the Round check measured')
    frozen_shas = prepare['frozen_task']['sha256']
    raw_frozen = {}
    for name in spec['frozen_files']:
        raw_frozen[name] = (task_dir / name).read_bytes()
        if sha256(raw_frozen[name]) != frozen_shas[name]:
            raise Refused(f'{task_dir / name} is not the frozen file of the task')

    # the start is the served baseline, so a solver does not begin with earlier rounds' gain already in hand
    frozen = {name: scrub(data, spec['scrub']) for name, data in raw_frozen.items()}
    start = {name: frozen[spec['baseline_of'][name]] for name in scope}
    candidate = {name: scrub(data, spec['scrub']) for name, data in raw_candidate.items()}
    if start == candidate:
        raise Refused('the candidate equals the baseline')

    runner_text = replace_docstring((task_dir / spec['runner']).read_text(), spec['runner_docstring'])
    for name in spec['frozen_files']:
        old, new = sha256(raw_frozen[name]), sha256(frozen[name])
        if old != new:
            if runner_text.count(old) != 1:
                raise Refused(f'the runner must pin {name} exactly once, found {runner_text.count(old)}')
            runner_text = runner_text.replace(old, new)
    runner = runner_text.encode()

    row = dict(zip(COLUMNS, (spec['bench'], spec['task_dir'], spec['infra_name'], 'performance', spec['gpus'],
                             spec['max_submissions'], ref, spec['summary'])))
    upstream_rows = [json.loads(line) for line in (VENDORED / 'upstream_task_catalog.jsonl').read_text().splitlines()
                     if line.strip()]
    check_row(row, upstream_rows)

    oracle = make_patch(start, candidate)
    for token in spec['banned_tokens']:
        added = [line for line in oracle.splitlines() if line.startswith('+') and token in line]
        if added:
            raise Refused(f'the reference itself uses the banned token {token!r}: {added[0][:100]}')
    negative = make_patch(start, negative_edit(start, spec['negative']))
    baseline2 = None
    stacked = prepare.get('stacked_start') or {}
    if stacked.get('path') and not piece_score:  # a piece is only compared with the served kernel
        arm = {name: scrub((Path(stacked['path']) / name).read_bytes(), spec['scrub']) for name in scope}
        baseline2 = make_patch(start, arm)

    instruction = (spec_dir / 'instruction.md').read_text()
    if 'MI210' not in instruction:
        raise Refused('the instruction must say the task runs on MI210')
    visible = {'instruction.md': instruction, 'summary': spec['summary'], spec['runner']: runner.decode()}
    visible.update({f'repo/{n}': d.decode() for n, d in {**start, **frozen}.items()})
    scan_solver_visible(visible, hidden_identifiers(candidate, start))

    image = f'fai/{spec["bench"]}-{spec["task_dir"]}:oss'
    calibration_note = ('The value is the Round check speedup of the reference, measured with graph replay on MI210 '
                        'by a different harness. It must be measured again with this package\'s own verifier before '
                        'any scored use.')
    if cal:
        calibration_note = (f'The value is the median of {len(cal["oracle_speedups"])} oracle-mode runs of this '
                            f'package\'s own verifier on one MI210 ({", ".join(f"{x:.4f}" for x in cal["oracle_speedups"])}), '
                            'in an image built from environment/Dockerfile with Enroot. Self-checks: no-change speedup '
                            f'{", ".join(f"{x:.4f}" for x in cal["noop_speedups"])} with reward 0; the negative patch '
                            'fails correctness with reward 0.')
    status = 'calibrated' if cal else 'provisional'
    if target_rows:
        calibration_note += f' It covers only the scored rows: {", ".join(target_rows)}.'
    caveat = fill((TEMPLATES / 'ref_speedup.caveat.md.tmpl').read_text(),
                  {'REF': str(ref), 'RUNTIME': spec['runtime']['description'], 'IMAGE': image,
                   'CALIBRATION_STATUS': status, 'CALIBRATION_NOTE': calibration_note})
    manifest = json.dumps({'ref_speedup': ref, 'perf_metric': 'speedup', 'reward_kind': 'gated_oracle',
                           'calibrated_on': 'AMD Instinct MI210', 'calibration_status': status}, indent=2) + '\n'
    grader = {**frozen, spec['runner']: runner}

    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / spec['task_dir']
        write(package / 'instruction.md', instruction)
        write(package / 'task.toml', task_toml(spec, image, scope, ref))
        write(package / '.dockerignore', '*.patch\nsolution/\n**/__pycache__/\n**/*.pyc\n')
        write(package / 'oracle.patch', oracle)
        write(package / 'negative.patch', negative)
        if baseline2:
            write(package / 'baseline2.patch', baseline2)

        repo = package / 'environment' / 'repo'
        for name, data in {**start, **frozen, spec['runner']: runner}.items():
            write(repo / name, data)
        sha_lines = ' '.join(f"'{sha256(start[n])}  {n}'" for n in scope)
        write(package / 'environment' / 'Dockerfile', fill((TEMPLATES / 'Dockerfile.tmpl').read_text(), {
            'TASK_ID': spec['task_dir'], 'IMAGE': image, 'BASE_IMAGE': spec['runtime']['base_image'],
            'SCOPE_SHA_LINES': sha_lines, 'TORCH_PREFIX': spec['runtime']['torch_prefix'],
            'VLLM_PREFIX': spec['runtime']['vllm_prefix']}))
        runtime = package / 'environment' / 'runtime'
        runtime.mkdir(parents=True)
        for name in ('entrypoint.sh', 'timer.sh'):
            shutil.copyfile(VENDORED / name, runtime / name)
            (runtime / name).chmod(0o755)  # stored executable, as upstream stores environment/runtime/*.sh
        write(runtime / 'run_dev_bench.sh', '#!/usr/bin/env bash\nset -euo pipefail\n'
              'KERNELBENCH_VERIFY_MODE="${KERNELBENCH_VERIFY_MODE:-candidate}" bash /tests/test.sh\n', 0o755)

        loop = package / 'environment' / 'loop'
        loop.mkdir(parents=True)
        for name in ('submit.sh', 'score_engine.sh', 'sanitize_feedback.py'):
            shutil.copyfile(VENDORED / name, loop / name)
        if f'MAX_SUBMISSIONS={spec["max_submissions"]}\n' not in (loop / 'submit.sh').read_text():
            raise Refused(f'the vendored submit.sh does not allow {spec["max_submissions"]} submissions')
        write(loop / 'private' / 'manifest.json', json.dumps(
            {'max_submissions': spec['max_submissions'], 'min_submissions': 1,
             'note': 'dev reward is an in-session proxy; the runner end-of-session /tests score is authoritative.',
             'perf_metric': 'speedup', 'reward_form': 'raw_speedup(base_ms/cand_ms); no-op ~= 1.0',
             'schema_version': 'kernelbench_loop_manifest_v1', 'scope': scope,
             'scoring': 'task tests/test.sh in candidate mode on the live /app/repo', 'task_id': spec['task_dir']},
            indent=2, sort_keys=True) + '\n')
        (loop / 'private' / 'tests').mkdir()
        write_tests(loop / 'private' / 'tests', spec, scope, grader, spec['dev_correctness_seeds'],
                    spec['dev_abba_blocks'], None, None, None, target_rows)
        (package / 'tests').mkdir()
        write_tests(package / 'tests', spec, scope, grader, spec['correctness_seeds'], spec['abba_blocks'],
                    str(ref), caveat, manifest, target_rows)

        scope_text = ' '.join(scope)
        pre = (VENDORED / 'pre_artifacts.sh').read_text().replace(VENDOR_REFERENCE_TASK, spec['task_dir'])
        pre = re.sub(r'^#\s+evidence .*\n', '', pre, flags=re.M)
        write(package / 'pre_artifacts.sh', pre, 0o755)
        solve = (VENDORED / 'solve.sh').read_text()
        solve = solve.replace('SCOPE=("w8a16/matmul.py")', f'SCOPE=({bash_words(scope)})')
        for old, new in ((VENDOR_REFERENCE_TASK, spec['task_dir']), ('fai/lh-w8a16-groupdequant-matmul:oss', image),
                         ('W8A16 group-dequant matmul (w8a16/matmul.py), H20', f'{spec["title"]}, MI210'),
                         ('docker run --rm --gpus all', 'docker run --rm --device /dev/kfd --device /dev/dri'),
                         ('w8a16/matmul.py', scope_text)):
            solve = solve.replace(old, new)
        write(package / 'solution' / 'solve.sh', solve, 0o755)

        scan_package(package)
        report = verify_package(package, spec['bench'], spec['task_dir'])

        provenance = {
            'row': row, 'calibration_status': status, **({'calibration': cal} if cal else {}),
            'run': {'dir': str(run_dir), 'status': outcome.get('status'),
                    'missing_workflow_outputs': outcome.get('missing_workflow_outputs', []),
                    'seconds': outcome.get('seconds'), 'transcript_models': outcome.get('transcript_models'),
                    'chat': str(run_dir / 'chat')},
            'round_check': {'dir': str(round_check), 'verdict': verdict['summary']['verdict'],
                            'score_speedup': verdict['summary']['score_speedup'],
                            'runtime': verdict.get('runtime')},
            'candidate': {'dir': str(candidate_dir), 'sha256': {n: sha256(raw_candidate[n]) for n in scope}},
            'task_source': prepare['frozen_task'].get('path'),
            **({'piece': {'plan': spec['piece']['plan'], **piece_score}} if piece_score else {}),
            'not_done': (['docker build (the image was built with Enroot from the same Dockerfile)'] if cal else
                         ['docker build', 'anchor calibration with this verifier (at least 5 runs on MI210)',
                          'no-change self-check', 'negative self-check', 'oracle self-check'])
                        + ['run_task.py GPU passthrough (it only passes NVIDIA devices)'],
            'verify_package': report,
        }

        target = out / 'tasks' / spec['bench'] / spec['task_dir']
        if cal:
            same_package(package, target)
        if target.exists():
            if not force:
                raise Refused(f'{target} exists; pass --force to rebuild it')
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(package, target, symlinks=True)

    write(out / 'provenance' / f'{spec["task_dir"]}.json', json.dumps(provenance, indent=2) + '\n')
    write_catalog(out, row, spec, scope, bool(baseline2))
    (out / 'scripts').mkdir(exist_ok=True)
    shutil.copyfile(VENDORED / 'verify_package.py', out / 'scripts' / 'verify_package.py')
    # the dataset's runner with AMD passthrough (upstream's passes only NVIDIA GPUs), and the change for upstream
    shutil.copyfile(TEMPLATES / 'run_task.py', out / 'scripts' / 'run_task.py')
    (out / 'scripts' / 'run_task.py').chmod(0o755)
    shutil.copyfile(TEMPLATES / 'run_task-amd.patch', out / 'scripts' / 'run_task-amd.patch')
    return {'row': row, 'package': str(target), 'verify_package': report, 'run_status': outcome.get('status')}


def verify_package(package: Path, bench: str, task_dir: str) -> str:
    """Their own checker, on a scratch copy laid out as the dataset."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'scripts').mkdir()
        shutil.copyfile(VENDORED / 'verify_package.py', root / 'scripts' / 'verify_package.py')
        shutil.copytree(package, root / 'tasks' / bench / task_dir, symlinks=True)
        run = subprocess.run([sys.executable, str(root / 'scripts' / 'verify_package.py'), bench],
                             capture_output=True, text=True, env={'PATH': '/usr/bin:/bin', 'PYTHONUTF8': '1'})
        output = (run.stdout + run.stderr).replace(tmp, '<root>')
    if run.returncode:
        raise Refused('verify_package.py failed:\n' + output)
    return output.strip()


def write_catalog(out: Path, row: dict, spec: dict, scope: list[str], has_baseline2: bool) -> None:
    """The rows made so far, in the dataset's formats; merging later means appending them to its files."""
    path = out / 'task_catalog.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []
    rows = [r for r in rows if r['task_dir'] != row['task_dir']] + [row]
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    by_bench = {b: sum(r['bench'] == b for r in rows) for b in sorted({r['bench'] for r in rows})}
    write(out / 'task_catalog.json', json.dumps({'schema': 'fai_bench_task_catalog_v2', 'task_count': len(rows),
                                                 'by_bench': by_bench, 'tasks': rows}, indent=1, ensure_ascii=False) + '\n')
    index_path = out / 'tasks_index.json'
    index = load_json(index_path) if index_path.is_file() else {'schema': 'fai_bench_task_index_v3', 'tasks': []}
    entry = {'bench': row['bench'], 'task': row['task_dir'],
             'package_root': f'tasks/{row["bench"]}/{row["task_dir"]}', 'layout': 'flat', 'gpus': row['gpus'],
             'dockerfiles': ['Dockerfile'], 'primary_edit_paths': scope, 'task_kind': 'acceleration',
             'perf_metric': None, 'ref_speedup': row['ref_speedup'], 'has_oracle': True,
             'oracle_artifacts': ['oracle.patch'], 'vendored_repo': True, 'has_loop16': True,
             'loop_harness': 'inline', 'min_submissions': 1, 'max_submissions': row['max_submissions'],
             'reward_class': row['reward_class']}
    index['tasks'] = [t for t in index['tasks'] if t['task'] != row['task_dir']] + [entry]
    index['task_count'] = len(index['tasks'])
    write(index_path, json.dumps(index, indent=1) + '\n')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--spec', type=Path, required=True, help='directory with spec.json and instruction.md')
    parser.add_argument('--run-dir', type=Path, required=True, help='the GEAK run directory')
    parser.add_argument('--round-check', type=Path, required=True, help='the independent Round check directory')
    parser.add_argument('--out', type=Path, required=True, help='output root, laid out like the dataset')
    parser.add_argument('--force', action='store_true', help='rebuild an existing package in --out')
    parser.add_argument('--calibration', type=Path,
                        help='directory of the package verifier runs (phibench_enroot.py score): sets the anchor')
    args = parser.parse_args()
    try:
        result = build(args.spec, args.run_dir, args.round_check, args.out, args.force, args.calibration)
    except Refused as refusal:
        print(f'REFUSED: {refusal}', file=sys.stderr)
        return 2
    print(json.dumps(result['row'], indent=2, ensure_ascii=False))
    anchor = 'calibrated' if args.calibration else 'provisional (not yet calibrated)'
    print(f'package: {result["package"]}\nrun status: {result["run_status"]}\nanchor: {anchor}')
    print(result['verify_package'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
