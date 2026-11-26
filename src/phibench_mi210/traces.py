"""Turn agent runs on Φ-Bench tasks into trace records (docs/trace-records.md).

    python -m phibench_mi210.traces convert --sheet SHEET.json --out DATASET
    python -m phibench_mi210.traces index --out DATASET

A run sheet (SHEET.json) lists the runs to convert and the facts a run folder does not hold itself: which task it was
on, the model and how it was served, the hardware, the anchor's evidence, and whether the run is invalid and why.
The converter reads everything else from the run folder, copies the raw files byte for byte, and refuses (writing
nothing for that run) when a field is missing, a file it needs is absent, a record with the same id exists, or a
copied file looks like it holds a secret (ADR 0009).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

SOURCES = ('claude_code_session', 'geak_attempt', 'open_model_run')
LABELS = ('successful', 'unsuccessful', 'invalid', 'unscored')

# Patterns that must never be stored (ADR 0009). A hit refuses the record; the person blanks the secret, notes it in
# the sheet's "scrub" entry, and converts again.
SECRET_PATTERNS = {
    'url_with_credentials': re.compile(rb'[a-z][a-z0-9+.-]*://[^\s/:@"\']+:[^\s/@"\']+@'),
    'bearer_token': re.compile(rb'Bearer\s+[A-Za-z0-9._~+/=-]{20,}'),
    'openai_style_key': re.compile(rb'\bsk-[A-Za-z0-9_-]{20,}'),
    'huggingface_token': re.compile(rb'\bhf_[A-Za-z0-9]{30,}'),
    'github_token': re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}'),
    'aws_access_key': re.compile(rb'\bAKIA[0-9A-Z]{16}\b'),
    'private_key_block': re.compile(rb'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
}

REQUIRED_SHEET_FIELDS = ('record_id', 'source', 'run_dir', 'task', 'agent', 'model', 'hardware', 'anchor')
REQUIRED_TASK_FIELDS = ('id', 'bench', 'kind', 'origin')


class RecordError(Exception):
    """A run cannot become a record; the message says why and what to fix."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def find_secrets(path: Path) -> list[str]:
    """Names of the secret patterns found in one file (empty when clean)."""
    data = path.read_bytes()
    return [name for name, pattern in SECRET_PATTERNS.items() if pattern.search(data)]


def label(reward: float | None, invalid_reason: str | None) -> str:
    """ADR 0008 with Φ-Bench's definition: invalid if the run was not a fair attempt, unscored if the verifier has
    not scored it, successful if the final reward is above 0, unsuccessful otherwise."""
    if invalid_reason:
        return 'invalid'
    if reward is None:
        return 'unscored'
    return 'successful' if reward > 0 else 'unsuccessful'


def check_sheet_entry(entry: dict) -> None:
    missing = [k for k in REQUIRED_SHEET_FIELDS if k not in entry]
    if missing:
        raise RecordError(f'sheet entry {entry.get("record_id", "?")}: missing {missing}')
    if entry['source'] not in SOURCES:
        raise RecordError(f'{entry["record_id"]}: source {entry["source"]!r} is not one of {SOURCES}')
    missing = [k for k in REQUIRED_TASK_FIELDS if k not in entry['task']]
    if missing:
        raise RecordError(f'{entry["record_id"]}: task is missing {missing}')
    if entry['task']['kind'] not in ('performance', 'implementation'):
        raise RecordError(f'{entry["record_id"]}: task kind must be performance or implementation')
    if not re.fullmatch(r'[A-Za-z0-9._-]+', entry['record_id']):
        raise RecordError(f'record_id {entry["record_id"]!r}: use letters, digits, ".", "_" and "-" only')


class RecordBuilder:
    """Collects one record's files in a staging folder; nothing reaches the dataset until commit()."""

    def __init__(self, out: Path, record_id: str):
        self.final = out / 'records' / record_id
        if self.final.exists():
            raise RecordError(f'{self.final} already exists; records are never overwritten')
        self.stage = out / 'records' / f'.{record_id}.staging'
        if self.stage.exists():
            shutil.rmtree(self.stage)
        self.stage.mkdir(parents=True)
        self.files: list[dict] = []

    def copy(self, src: Path, dest_rel: str) -> None:
        """Copy one original file byte for byte, refusing it if it looks like it holds a secret."""
        if not src.is_file():
            raise RecordError(f'missing file {src}')
        hits = find_secrets(src)
        if hits:
            raise RecordError(f'{src} matches {hits}; blank the secret, note it under "scrub" in the sheet, convert again')
        dest = self.stage / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        self.files.append({'path': dest_rel, 'source': str(src), 'sha256': sha256(dest), 'bytes': dest.stat().st_size})

    def copy_tree(self, src: Path, dest_rel: str) -> None:
        if not src.is_dir():
            raise RecordError(f'missing folder {src}')
        for f in sorted(p for p in src.rglob('*') if p.is_file()):
            self.copy(f, f'{dest_rel}/{f.relative_to(src).as_posix()}')

    def write_json(self, dest_rel: str, obj) -> None:
        dest = self.stage / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n')

    def write_jsonl(self, dest_rel: str, rows: list[dict]) -> None:
        dest = self.stage / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, 'w') as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + '\n')

    def commit(self, record: dict) -> Path:
        record['provenance']['files'] = self.files
        self.write_json('record.json', record)
        self.stage.rename(self.final)
        return self.final

    def abort(self) -> None:
        if self.stage.exists():
            shutil.rmtree(self.stage)


def base_record(entry: dict, machine: str) -> dict:
    return {
        'record_id': entry['record_id'],
        'source': entry['source'],
        'task': entry['task'],
        'agent': entry['agent'],
        'model': entry['model'],
        'hardware': entry['hardware'],
        'anchor': entry['anchor'],
        'time': {},
        'outcome': {},
        'submissions': [],
        'conversation': {},
        'provenance': {'machine': machine, 'run_dir': str(entry['run_dir']),
                       'converted_at': dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')},
        'scrub': entry.get('scrub', {'secrets_found': [], 'changed': []}),
        'notes': entry.get('notes', ''),
    }


INDEX_FIELDS = ('record_id', 'source', 'label', 'reward', 'speedup')


def rebuild_index(out: Path) -> int:
    """index.jsonl from every record.json; rebuilt whole so that it can never disagree with the records."""
    rows = []
    for rec_path in sorted((out / 'records').glob('*/record.json')):
        r = json.loads(rec_path.read_text())
        rows.append({'record_id': r['record_id'], 'task': r['task']['id'], 'source': r['source'],
                     'model': r['model'].get('name'), 'label': r['outcome'].get('label'),
                     'reward': r['outcome'].get('reward'), 'speedup': r['outcome'].get('speedup')})
    with open(out / 'index.jsonl', 'w') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    return len(rows)


ADAPTERS = {}  # source -> function(entry, builder) -> dict of record parts (filled in per source)


def convert(sheet: dict, out: Path) -> list[str]:
    """Convert every run in the sheet; each run is all or nothing. Returns the record ids written."""
    machine = sheet.get('machine')
    if not machine:
        raise RecordError('sheet: "machine" (where the run folders are) is required')
    written = []
    for entry in sheet['runs']:
        check_sheet_entry(entry)
        adapter = ADAPTERS.get((entry['source'], entry['agent'].get('harness')))
        if adapter is None:
            raise RecordError(f'{entry["record_id"]}: no converter yet for source {entry["source"]} with harness '
                              f'{entry["agent"].get("harness")!r}; available: {sorted(ADAPTERS)}')
        builder = RecordBuilder(out, entry['record_id'])
        try:
            record = base_record(entry, machine)
            adapter(entry, builder, record)
            record['outcome']['label'] = label(record['outcome'].get('reward'), entry.get('invalid_reason'))
            if entry.get('invalid_reason'):
                record['outcome']['invalid_reason'] = entry['invalid_reason']
            builder.commit(record)
        except BaseException:
            builder.abort()
            raise
        written.append(entry['record_id'])
    rebuild_index(out)
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('convert', help='convert the runs listed in a sheet')
    c.add_argument('--sheet', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True, help='the trace dataset folder')
    i = sub.add_parser('index', help='rebuild index.jsonl from the records')
    i.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.cmd == 'convert':
            args.out.mkdir(parents=True, exist_ok=True)
            done = convert(json.loads(args.sheet.read_text()), args.out)
            print(f'wrote {len(done)} records: {", ".join(done)}')
        else:
            print(f'index: {rebuild_index(args.out)} records')
    except RecordError as e:
        print(f'refused: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
