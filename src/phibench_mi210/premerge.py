"""Everything to check before our MI210 rows are merged into the Φ-Bench dataset, in one report.

    python -m phibench_mi210.premerge --rows ROWS [--provenance DIR] [--review REVIEW.json]

ROWS is laid out like the dataset (task_catalog.jsonl, task_catalog.json, tasks_index.json, tasks/<bench>/<task>/):
the local output of row.py, or a download of the Hugging Face copy. Automatic checks, per row:

  columns      the 8 columns, order and types of the published catalog; task_dir not already in the dataset
  catalogs     task_catalog.jsonl, task_catalog.json and tasks_index.json list the same rows and anchors
  anchor       tests/ref_speedup.txt, task.toml and the catalog hold the same ref_speedup
  calibrated   the caveat says calibrated; with --provenance, at least 5 oracle runs whose median is the anchor
  verifier     the dataset's own scripts/verify_package.py passes
  internal     no internal path, node or account name in any file
  solver text  instruction.md and the summary say MI210; nothing solver-visible names the reference's origin
  executable   every *.sh outside environment/loop/ is executable, as upstream stores them (100755); the Hugging
               Face upload API drops the bit, so check a download too
  hidden       .dockerignore keeps *.patch and solution/ out of the image; tests/ is not copied in, except the anchor
               manifest (a WARN: upstream's README says the in-image copy omits the anchor, its own example
               task includes it)

REVIEW (default specs/premerge-review.json) holds what no script can decide: owner approval,
license, research checks such as a no-insight search or a real solver run, each with a status, a note and its
evidence. A row is READY when no automatic check fails and no blocking review item is open.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tomllib
from pathlib import Path

from phibench_mi210 import row as pr

DEFAULT_REVIEW = Path(__file__).resolve().parents[2] / 'specs' / 'premerge-review.json'


def upstream_rows() -> list[dict]:
    return [json.loads(line) for line in (pr.VENDORED / 'upstream_task_catalog.jsonl').read_text().splitlines()
            if line.strip()]


def check_one(rows: Path, row: dict, others: dict, provenance: Path | None) -> list[tuple[str, str, str]]:
    """[(check, PASS|WARN|FAIL, detail)] for one catalog row."""
    out = []

    def add(name, ok, detail='', warn=False):
        out.append((name, 'PASS' if ok else ('WARN' if warn else 'FAIL'), detail))

    try:
        pr.check_row(row, upstream_rows())
        add('columns', True)
    except pr.Refused as exc:
        add('columns', False, str(exc))
    task, ref = row['task_dir'], row['ref_speedup']
    package = rows / 'tasks' / row['bench'] / task
    in_json = others['json'].get(task)
    in_index = others['index'].get(task)
    add('catalogs', in_json == row and in_index is not None and in_index.get('ref_speedup') == ref,
        '' if in_json == row else 'task_catalog.json differs or lacks the row')
    if not package.is_dir():
        add('package', False, f'{package} missing')
        return out
    text = (package / 'tests/ref_speedup.txt').read_text().strip()
    toml = tomllib.loads((package / 'task.toml').read_text())
    anchors = {'ref_speedup.txt': float(text), 'task.toml': toml['primary_metric']['ref_speedup'], 'catalog': ref}
    add('anchor', len(set(anchors.values())) == 1, json.dumps(anchors))
    caveat = (package / 'tests/ref_speedup.caveat.md').read_text()
    status = re.search(r'Status: \*\*(\w+)\*\*', caveat)
    calibrated = bool(status) and status.group(1) == 'calibrated'
    detail = f'caveat status {status.group(1) if status else "missing"}'
    if provenance is not None:
        record = json.loads((provenance / f'{task}.json').read_text()) if (provenance / f'{task}.json').is_file() else {}
        cal = record.get('calibration') or {}
        runs = cal.get('oracle_speedups', [])
        calibrated = calibrated and len(runs) >= pr.MIN_CALIBRATION_RUNS and \
            round(statistics.median(runs), 6) == ref
        detail += f'; {len(runs)} oracle runs in provenance'
    add('calibrated', calibrated, detail)
    try:
        pr.verify_package(package, row['bench'], task)
        add('verifier', True)
    except pr.Refused as exc:
        add('verifier', False, str(exc)[-300:])
    hits = [str(p.relative_to(package)) for p in package.rglob('*')
            if p.is_file() and pr.DENY_EVERYWHERE.search(p.read_text(errors='ignore'))]
    add('internal', not hits, ', '.join(hits[:5]))
    instruction = (package / 'instruction.md').read_text()
    visible = [instruction, row['summary']] + [p.read_text(errors='ignore') for p in (package / 'environment').rglob('*')
                                              if p.is_file()]
    named = sorted({m.group(0) for t in visible for m in pr.DENY_FOR_SOLVER.finditer(t)})
    add('solver text', 'MI210' in instruction and 'MI210' in row['summary'] and not named,
        ('names: ' + ', '.join(named)) if named else '')
    plain = [str(p.relative_to(package)) for p in package.rglob('*.sh') if not os.access(p, os.X_OK)
             and p.parent != package / 'environment/loop']  # upstream stores loop/*.sh 100644; the image chmods them
    add('executable', not plain, f'{len(plain)} not executable, e.g. {plain[:3]}' if plain else '')
    ignore = (package / '.dockerignore').read_text().split()
    dockerfile = (package / 'environment/Dockerfile').read_text()
    copies_tests = [line for line in dockerfile.splitlines() if re.match(r'COPY\s+tests/', line)
                    and 'verifier-correctness-manifest.json' not in line]
    add('hidden', '*.patch' in ignore and 'solution/' in ignore and not copies_tests,
        '; '.join(copies_tests))
    manifest = json.loads((package / 'tests/verifier-correctness-manifest.json').read_text())
    if 'COPY tests/verifier-correctness-manifest.json' in dockerfile and 'ref_speedup' in manifest:
        add('anchor in image', False, 'the image carries ref_speedup (upstream README: omit it; its own example task '
            'includes it): ask the dataset owners', warn=True)
    return out


def review_items(review: dict, task: str | None) -> list[dict]:
    return [item for item in review['items'] if item.get('task') == task]


def report(rows: Path, provenance: Path | None, review: dict) -> tuple[str, bool]:
    catalog = [json.loads(line) for line in (rows / 'task_catalog.jsonl').read_text().splitlines() if line.strip()]
    others = {'json': {r['task_dir']: r for r in json.loads((rows / 'task_catalog.json').read_text())['tasks']},
              'index': {t['task']: t for t in json.loads((rows / 'tasks_index.json').read_text())['tasks']}}
    lines, all_ready = [f'# Pre-merge report: {rows}', ''], True
    general = review_items(review, None)
    for row in catalog:
        checks = check_one(rows, row, others, provenance)
        items = review_items(review, row['task_dir'])
        blocking = [i for i in items + general if i['blocking'] and i['status'] != 'done']
        failed = [c for c in checks if c[1] == 'FAIL']
        ready = not failed and not blocking
        all_ready &= ready
        lines += [f'## {row["task_dir"]}: {"READY" if ready else "NOT READY"}  (ref_speedup {row["ref_speedup"]})', '',
                  '| Check | Result | Detail |', '|---|---|---|']
        lines += [f'| {name} | {state} | {detail} |' for name, state, detail in checks]
        lines += [f'| review: {i["check"]} | {i["status"]}{" (blocking)" if i["blocking"] else ""} | '
                  f'{i["note"]} {("[" + i["evidence"] + "]") if i.get("evidence") else ""} |' for i in items]
        lines.append('')
    lines += ['## All rows', '', '| Check | Status | Note |', '|---|---|---|']
    lines += [f'| {i["check"]} | {i["status"]}{" (blocking)" if i["blocking"] else ""} | {i["note"]} '
              f'{("[" + i["evidence"] + "]") if i.get("evidence") else ""} |' for i in general]
    return '\n'.join(lines) + '\n', all_ready


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, help='row.py provenance/ (local output only)')
    parser.add_argument('--review', type=Path, default=DEFAULT_REVIEW)
    parser.add_argument('--write', type=Path, help='also write the report to this Markdown file')
    args = parser.parse_args(argv)
    text, ready = report(args.rows, args.provenance, json.loads(args.review.read_text()))
    print(text)
    if args.write:
        args.write.write_text(text)
    print('ALL ROWS READY' if ready else 'NOT READY: see FAIL rows and open blocking items')
    return 0 if ready else 1


if __name__ == '__main__':
    sys.exit(main())
