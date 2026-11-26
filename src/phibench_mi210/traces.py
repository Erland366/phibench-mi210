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


def _viewer_message(m: dict) -> dict:
    """One message with the same keys and types in every row, so that the Hugging Face viewer can read it."""
    content = m.get('content')
    if m['role'] == 'event':
        content = m.get('summary') or m.get('error') or m.get('status')
    return {'role': m['role'], 'event': m.get('event'),
            'content': content if isinstance(content, str) or content is None else _text_of(content),
            'reasoning': m.get('reasoning'),
            'tool_calls': json.dumps(m['tool_calls'], ensure_ascii=False) if m.get('tool_calls') else None,
            'tool_call_id': m.get('tool_call_id')}


DATASET_CARD = """---
configs:
- config_name: runs
  data_files: viewer/runs.jsonl
  default: true
- config_name: index
  data_files: index.jsonl
---
# RSI trace records

One record per run of an agent on a Φ-Bench task, labeled by the task's own verifier: successful (final reward
above 0), unsuccessful (reward 0), invalid (not a fair attempt) or unscored. Private to the group.

- `viewer/runs.jsonl`: one row per run with its whole conversation, for reading in the viewer (generated).
- `index.jsonl`: one line per record (generated).
- `records/<record_id>/`: the record itself: `record.json`, the prompt, the raw files byte for byte, the generated
  `messages.jsonl`, the code and the verifier output.

Format and decisions: `docs/trace-records.md`, `CONTEXT.md` and `docs/adr/` in the phibench-mi210 repository.
"""


def rebuild_index(out: Path) -> int:
    """index.jsonl and viewer/runs.jsonl from every record; rebuilt whole so that they can never disagree with the
    records. Writes the dataset card if there is none."""
    rows, runs = [], []
    for rec_path in sorted((out / 'records').glob('*/record.json')):
        r = json.loads(rec_path.read_text())
        row = {'record_id': r['record_id'], 'task': r['task']['id'], 'source': r['source'],
               'model': r['model'].get('name'), 'label': r['outcome'].get('label'),
               'reward': r['outcome'].get('reward'), 'speedup': r['outcome'].get('speedup')}
        rows.append(row)
        msgs_path = rec_path.parent / 'conversation' / 'messages.jsonl'
        msgs = [json.loads(l) for l in msgs_path.read_text().splitlines() if l.strip()] if msgs_path.is_file() else []
        runs.append({**row, 'ref_speedup': r['outcome'].get('ref_speedup'),
                     'harness': r['agent'].get('harness'), 'harness_version': r['agent'].get('version'),
                     'invalid_reason': r['outcome'].get('invalid_reason'),
                     'final_tree': r['outcome'].get('final_tree'), 'submissions': len(r['submissions']),
                     'compactions': r['conversation'].get('compactions'), 'messages': [_viewer_message(m) for m in msgs]})
    with open(out / 'index.jsonl', 'w') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    (out / 'viewer').mkdir(exist_ok=True)
    with open(out / 'viewer' / 'runs.jsonl', 'w') as fh:
        for run in runs:
            fh.write(json.dumps(run, ensure_ascii=False) + '\n')
    if not (out / 'README.md').exists():
        (out / 'README.md').write_text(DATASET_CARD)
    return len(rows)


# ---------------------------------------------------------------------------------------------------- open models

COMPACTION_PROMPT = 'You are the component that summarizes a conversation when its context window is about to overflow.'


def _text_of(content) -> str:
    """The text of a message's content: a string, or a list of text parts."""
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    return '\n'.join(part.get('text', '') for part in content if isinstance(part, dict))


def _key(msg: dict) -> tuple:
    """What identifies a message when a later request resends it (reasoning and refusal fields vary)."""
    calls = tuple((c.get('id'), c['function']['name'], c['function']['arguments']) for c in msg.get('tool_calls') or ())
    return (msg['role'], _text_of(msg.get('content')), calls, msg.get('tool_call_id'))


def _out(msg: dict, call: int) -> dict:
    row = {'role': msg['role'], 'content': msg.get('content')}
    reasoning = msg.get('reasoning_content') or msg.get('reasoning')
    if reasoning:
        row['reasoning'] = reasoning
    if msg.get('tool_calls'):
        row['tool_calls'] = msg['tool_calls']
    if msg.get('tool_call_id'):
        row['tool_call_id'] = msg['tool_call_id']
    row['call'] = call
    return row


def is_compaction(request: dict) -> bool:
    last = request['messages'][-1]
    return last['role'] == 'user' and _text_of(last.get('content')).startswith(COMPACTION_PROMPT)


def messages_from_openai_calls(calls: list[dict]) -> tuple[list[dict], dict]:
    """The run as one list of messages from its raw model calls, in order (docs/trace-records.md).

    A call's request is the conversation so far; only what is new since the previous call is emitted. A compaction
    call becomes one event line (with the summary and the model's reasoning) and the next call's rewritten history is
    emitted whole, after its system message. A failed call becomes an event line. Returns (rows, counts)."""
    rows, ctx = [], None
    counts = {'calls': len(calls), 'failed_calls': 0, 'compactions': 0, 'history_rewrites': 0}
    system_key = None
    for i, call in enumerate(calls):
        req, resp = call['request'], call.get('response')
        msgs = req['messages']
        if resp is None:
            counts['failed_calls'] += 1
            rows.append({'role': 'event', 'event': 'failed_call', 'error': (call.get('error') or {}).get('message'),
                         'call': i})
            continue
        message = resp['choices'][0]['message']
        if is_compaction(req):
            counts['compactions'] += 1
            rows.append({'role': 'event', 'event': 'compaction', 'summary': message.get('content'),
                         'reasoning': message.get('reasoning_content'), 'call': i})
            ctx = None
            continue
        keys = [_key(m) for m in msgs]
        if ctx is not None and keys[:len(ctx)] == ctx:
            new = msgs[len(ctx):]
        else:
            if ctx is not None:
                counts['history_rewrites'] += 1
                rows.append({'role': 'event', 'event': 'history_rewritten', 'call': i})
            new = msgs
            if new and new[0]['role'] == 'system' and _key(new[0]) == system_key:
                new = new[1:]
        for m in new:
            if m['role'] == 'system':
                system_key = _key(m)
            rows.append(_out(m, i))
        rows.append(_out(message, i))
        ctx = keys + [_key(message)]
    return rows, counts


def _read_json(path: Path):
    if not path.is_file():
        raise RecordError(f'missing file {path}')
    return json.loads(path.read_text())


def _read_line(path: Path) -> str:
    if not path.is_file():
        raise RecordError(f'missing file {path}')
    return path.read_text().strip()


def _optional(builder: 'RecordBuilder', src: Path, dest_rel: str) -> bool:
    if src.is_file():
        builder.copy(src, dest_rel)
        return True
    if src.is_dir():
        builder.copy_tree(src, dest_rel)
        return True
    return False


def _final_score(entry: dict, run: Path, builder: 'RecordBuilder', record: dict) -> None:
    """The counted final score (final/verifier/reward.json), thrown-away scores, and the anchor check."""
    final = run / 'final'
    reward_path = final / 'verifier' / 'reward.json'
    if reward_path.is_file():
        r = _read_json(reward_path)
        builder.copy_tree(final / 'verifier', 'verifier/final')
        for name in ('finalize.txt', 'console.log', 'exit', 'started', 'finished'):
            _optional(builder, final / name, f'verifier/{name}')
        _optional(builder, run / 'logs' / 'verifier', 'verifier/session')
        sheet_ref = entry['anchor'].get('ref_speedup')
        if sheet_ref is None or abs(float(r['ref_speedup']) - float(sheet_ref)) > 1e-9:
            raise RecordError(f'{entry["record_id"]}: the final score used ref_speedup {r["ref_speedup"]}, the sheet '
                              f'says {sheet_ref}; fix the sheet or explain the anchor')
        gates = r.get('gates') or {}
        record['outcome'].update(
            reward=float(r['reward']), speedup=r.get('speedup'), ref_speedup=r['ref_speedup'],
            correctness_ok=gates.get('correctness_ok'), gates=gates,
            hard_fail_reasons=r.get('hard_fail_reasons', []), verifier_output='verifier/final/reward.json')
        exit_path = final / 'exit'
        if exit_path.is_file():
            record['outcome']['verifier_exit_code'] = int(_read_line(exit_path))
    elif not entry.get('invalid_reason'):
        raise RecordError(f'{entry["record_id"]}: no final score at {reward_path}; a valid run needs one '
                          '(or give invalid_reason in the sheet)')
    invalid_scores = []
    for n, item in enumerate(entry.get('invalid_scores', []), 1):
        src = run / item['dir']
        builder.copy_tree(src, f'verifier/invalid-{n}')
        invalid_scores.append({'dir': f'verifier/invalid-{n}', 'reason': item['reason'],
                               'reward': _read_json(src / 'verifier' / 'reward.json').get('reward')
                               if (src / 'verifier' / 'reward.json').is_file() else None})
    record['outcome']['invalid_scores'] = invalid_scores


def _loop(entry: dict, run: Path, builder: 'RecordBuilder', record: dict) -> None:
    """Submissions from logs/loop/ (absent when the solver never submitted)."""
    loop = run / 'logs' / 'loop'
    if not loop.is_dir():
        record['outcome']['final_tree'] = 'starting code (no submissions)'
        return
    for f in sorted(loop.iterdir()):
        if f.name != 'submissions':
            _optional(builder, f, f'harness/loop/{f.name}')
    unclean = {int(k): v for k, v in entry.get('unclean_submissions', {}).items()}
    known = entry.get('submission_timing') == 'checked'
    state = loop / 'state.jsonl'
    lines = [json.loads(l) for l in state.read_text().splitlines() if l.strip()] if state.is_file() else []
    for s in lines:
        n = int(s['submission'])
        sub_dir = loop / 'submissions' / str(n)
        builder.copy_tree(sub_dir, f'code/submissions/{n}')
        record['submissions'].append({
            'n': n, 'ts': s.get('ts'), 'speedup': s.get('speedup'), 'reward': s.get('reward'),
            'correctness_ok': s.get('correctness_ok'), 'hard_fail_reasons': s.get('hard_fail_reasons', []),
            'tree': s.get('tree'), 'diff': f'code/submissions/{n}/candidate.diff',
            'timing_clean': (n not in unclean) if known else None, 'timing_note': unclean.get(n)})
    fin = loop / 'finalize.json'
    if fin.is_file():
        f = json.loads(fin.read_text())
        planted = f.get('plant_method') not in (None, 'none') and f.get('winning_submission') is not None
        record['outcome']['final_tree'] = (f'winning submission {f["winning_submission"]}' if planted
                                           else 'starting code (no submission accepted)')
        record['outcome']['termination_reason'] = f.get('termination_reason')


def _common_harness(entry: dict, run: Path, builder: 'RecordBuilder', record: dict) -> dict:
    builder.copy(run / 'prompt.txt', 'prompt.txt')
    result = _read_json(run / 'agent_result.json')
    builder.copy(run / 'agent_result.json', 'harness/agent_result.json')
    for name in ('stderr.log', 'sandbox-build.log', 'started', 'finished'):
        _optional(builder, run / name, f'harness/{name}')
    for key, dest in (('job_log', 'harness/job.log'),):
        if entry.get(key):
            builder.copy(Path(entry[key]), dest)
    if not entry.get('serving_log'):
        raise RecordError(f'{entry["record_id"]}: serving_log (the model server log of this run) is required')
    serving = Path(entry['serving_log'])
    builder.copy(serving, f'serving/{serving.name}')
    record['time'] = {'started': _read_line(run / 'started') if (run / 'started').is_file() else None,
                      'finished': _read_line(run / 'finished') if (run / 'finished').is_file() else None}
    patch = run / 'final' / 'model.patch'
    if patch.is_file():
        builder.copy(patch, 'code/final.patch')
        _optional(builder, run / 'final' / 'model_status.txt', 'code/final.status.txt')
    else:
        record['notes'] = (record['notes'] + ' ' if record['notes'] else '') + 'final patch was not saved'
    return result


def convert_qwen_code(entry: dict, builder: 'RecordBuilder', record: dict) -> None:
    run = Path(entry['run_dir'])
    result = _common_harness(entry, run, builder, record)
    record['time']['wall_limit_s'] = result.get('wall_limit_s')
    record['outcome']['harness_exit_code'] = result.get('exit_code')
    openai_dir = run / 'logs' / 'agent' / 'openai'
    files = sorted(openai_dir.glob('openai-*.json')) if openai_dir.is_dir() else []
    if not files:
        raise RecordError(f'{entry["record_id"]}: no raw model calls in {openai_dir}')
    builder.copy_tree(openai_dir, 'conversation/raw/openai')
    builder.copy(run / 'stream.jsonl', 'conversation/raw/stream.jsonl')
    _optional(builder, run / 'settings.json', 'conversation/raw/settings.json')
    _optional(builder, run / 'logs' / 'agent' / 'home' / '.qwen', 'conversation/raw/qwen-home')
    init = next((json.loads(l) for l in (run / 'stream.jsonl').read_text().splitlines()
                 if l.strip() and json.loads(l).get('subtype') == 'init'), None)
    if init is None:
        raise RecordError(f'{entry["record_id"]}: stream.jsonl has no init line (harness version, model)')
    version = init.get('qwen_code_version')
    if entry['agent'].get('version') not in (None, version):
        raise RecordError(f'{entry["record_id"]}: sheet says Qwen Code {entry["agent"]["version"]}, '
                          f'the stream says {version}')
    record['agent']['version'] = version
    record['model']['served_name'] = init.get('model')
    calls = [json.loads(f.read_text()) for f in files]
    rows, counts = messages_from_openai_calls(calls)
    builder.write_jsonl('conversation/messages.jsonl', rows)
    record['conversation'] = {'main': 'conversation/raw/openai', 'harness_log': 'conversation/raw/stream.jsonl',
                              'reasoning_included': True, 'thinking_dropped': False, **counts}
    _loop(entry, run, builder, record)
    _final_score(entry, run, builder, record)


def messages_from_mini_swe_agent(traj: dict) -> list[dict]:
    rows = []
    for i, m in enumerate(traj['messages']):
        if m['role'] == 'exit':
            rows.append({'role': 'event', 'event': 'exit', 'status': m.get('content'), 'step': i})
            continue
        row = {'role': m['role'], 'content': m.get('content')}
        reasoning = m.get('reasoning_content') or (m.get('provider_specific_fields') or {}).get('reasoning')
        if reasoning:
            row['reasoning'] = reasoning
        actions = (m.get('extra') or {}).get('actions')
        if actions:
            row['actions'] = actions
        row['step'] = i
        rows.append(row)
    return rows


def convert_mini_swe_agent(entry: dict, builder: 'RecordBuilder', record: dict) -> None:
    run = Path(entry['run_dir'])
    result = _common_harness(entry, run, builder, record)
    record['outcome']['harness_exit_status'] = result.get('exit_status')
    traj = _read_json(run / 'trajectory.json')
    builder.copy(run / 'trajectory.json', 'conversation/raw/trajectory.json')
    version = traj.get('info', {}).get('mini_version')
    if entry['agent'].get('version') not in (None, version):
        raise RecordError(f'{entry["record_id"]}: sheet says mini-swe-agent {entry["agent"]["version"]}, '
                          f'the trajectory says {version}')
    record['agent']['version'] = version
    record['model']['served_name'] = traj.get('info', {}).get('config', {}).get('model', {}).get('model_name')
    rows = messages_from_mini_swe_agent(traj)
    builder.write_jsonl('conversation/messages.jsonl', rows)
    record['conversation'] = {'main': 'conversation/raw/trajectory.json', 'reasoning_included': True,
                              'thinking_dropped': False, 'compactions': 0,
                              'steps': sum(1 for r in rows if r['role'] == 'assistant')}
    _loop(entry, run, builder, record)
    _final_score(entry, run, builder, record)


# source, harness -> function(entry, builder, record) that copies the files and fills the record
ADAPTERS = {
    ('open_model_run', 'qwen-code'): convert_qwen_code,
    ('open_model_run', 'mini-swe-agent'): convert_mini_swe_agent,
}


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
