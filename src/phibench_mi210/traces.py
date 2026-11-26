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

from phibench_mi210.roots import RootError, expand

SOURCES = ('claude_code_session', 'codex_session', 'geak_run', 'open_model_run')
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

REQUIRED_SHEET_FIELDS = ('record_id', 'source', 'run_dir', 'agent', 'model', 'hardware')
REQUIRED_TASK_FIELDS = ('id', 'bench', 'kind', 'origin')


class RecordError(Exception):
    """A run cannot become a record; the message says why and what to fix."""


class RecordExists(RecordError):
    """A record with this id is already stored; records are never overwritten."""


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


# Fixed choices in a run sheet (sheet format 2). A new device, model or server is added here on purpose.
DEVICES = ('AMD Instinct MI210', 'AMD EPYC 9334')
MODEL_NAMES = ('Qwen3.8-27B', 'claude-opus-5-5')
SERVERS = ('vLLM',)
SELF_SERVED = ('open_model_run',)  # the sources whose model we served ourselves (served_on, served_by)


def _device(value, where: str) -> dict:
    if not isinstance(value, dict) or set(value) != {'device', 'count'}:
        raise RecordError(f'{where} must be {{"device": one of {DEVICES}, "count": <whole number>}}')
    if value['device'] not in DEVICES:
        raise RecordError(f'{where}: device {value["device"]!r} is not one of {DEVICES}')
    if not isinstance(value['count'], int) or isinstance(value['count'], bool) or value['count'] < 1:
        raise RecordError(f'{where}: count must be a whole number of at least 1')
    return dict(value)


def task_facts(task_dir: Path) -> dict:
    """What a Φ-Bench task package says about itself: the task's id, size and kind (task.toml), its anchor (the file
    task.toml names in ref_speedup_source, which must agree with task.toml's ref_speedup) and the hardware it is
    scored on (calibrated_on)."""
    import tomllib
    toml_path = task_dir / 'task.toml'
    if not toml_path.is_file():
        raise RecordError(f'{task_dir}: no task.toml; task_dir must name a Φ-Bench task package')
    t = tomllib.loads(toml_path.read_text())
    task = {k: t.get('task', {}).get(k) for k in ('id', 'bench', 'kind')}
    if None in task.values():
        raise RecordError(f'{toml_path}: [task] needs id, bench and kind')
    metric = t.get('primary_metric', {})
    anchor = {'ref_speedup': None, 'evidence': None}
    if task['kind'] == 'performance':
        source = metric.get('ref_speedup_source')
        if not source or metric.get('ref_speedup') is None:
            raise RecordError(f'{toml_path}: [primary_metric] needs ref_speedup and ref_speedup_source')
        ref_file = task_dir / source
        if not ref_file.is_file():
            raise RecordError(f'{task_dir}: task.toml names {source} as its anchor file, which does not exist')
        if ref_file.suffix == '.json':  # a verifier manifest: {"ref_speedup": ..., "upstream_ref_speedup": ...}
            manifest = _read_json(ref_file)
            ref = float(manifest['ref_speedup'])
            if manifest.get('upstream_ref_speedup') is not None:
                anchor['upstream_ref_speedup'] = float(manifest['upstream_ref_speedup'])
        else:
            ref = float(ref_file.read_text().split()[0])
        if abs(float(metric['ref_speedup']) - ref) > 1e-9:
            raise RecordError(f'{task_dir}: {source} says {ref}, task.toml says {metric["ref_speedup"]}; the package '
                              'must say one anchor')
        anchor.update(ref_speedup=ref, evidence=str(ref_file))
    device = metric.get('calibrated_on')
    if device is None:
        raise RecordError(f'{toml_path}: [primary_metric] calibrated_on is missing; it names the scoring hardware')
    gpus = t.get('environment', {}).get('gpus') or 0
    scored = _device({'device': device, 'count': gpus if gpus and 'Instinct' in device else 1},
                     f'{toml_path} calibrated_on')
    return {'task': task, 'anchor': anchor, 'scored_on': scored}


def check_sheet_entry(entry: dict) -> dict:
    """Check one run sheet entry and return it with the facts that come from its task package filled in. A run on
    a Φ-Bench task names the package (task_dir), and the task's id, size, kind, anchor and scoring hardware are read
    from it, never typed again; a run with no Φ-Bench task (an unsuccessful GEAK run) states them itself."""
    rid = entry.get('record_id', '?')
    missing = [k for k in REQUIRED_SHEET_FIELDS if k not in entry]
    if missing:
        raise RecordError(f'sheet entry {rid}: missing {missing}')
    if not re.fullmatch(r'[A-Za-z0-9._-]+', entry['record_id']):
        raise RecordError(f'record_id {entry["record_id"]!r}: use letters, digits, ".", "_" and "-" only')
    if entry['source'] not in SOURCES:
        raise RecordError(f'{rid}: source {entry["source"]!r} is not one of {SOURCES}')
    e = json.loads(json.dumps(entry))
    hardware, model = e['hardware'], e['model']
    if not isinstance(hardware, dict) or set(hardware) - {'served_on', 'scored_on'}:
        raise RecordError(f'{rid}: hardware takes only served_on and scored_on; put details in notes')
    if e.get('task_dir'):
        repeated = ([f'task.{k}' for k in ('id', 'bench', 'kind') if k in (e.get('task') or {})]
                    + (['anchor'] if 'anchor' in e else []) + (['hardware.scored_on'] if 'scored_on' in hardware else []))
        if repeated:
            raise RecordError(f'{rid}: {repeated} come from the task package in task_dir; remove them from the sheet')
        facts = task_facts(Path(e['task_dir']))
        origin = (e.get('task') or {}).get('origin')
        if not origin:
            raise RecordError(f'{rid}: task.origin (the dataset and its revision) is required')
        e['task'] = {**facts['task'], 'origin': origin, 'task_dir': e['task_dir']}
        e['anchor'] = facts['anchor']
        hardware['scored_on'] = facts['scored_on']
    elif e['source'] != 'geak_run':
        raise RecordError(f'{rid}: task_dir (the Φ-Bench task package the run was on) is required; only a GEAK run '
                          'with no Φ-Bench task states its task itself')
    else:
        if 'task' not in e or 'anchor' not in e or 'scored_on' not in hardware:
            raise RecordError(f'{rid}: give task_dir (the Φ-Bench task package), or, for a run with no Φ-Bench task, '
                              'task, anchor and hardware.scored_on')
        missing = [k for k in REQUIRED_TASK_FIELDS if k not in e['task']]
        if missing:
            raise RecordError(f'{rid}: task is missing {missing}')
    if e['task']['kind'] not in ('performance', 'implementation'):
        raise RecordError(f'{rid}: task kind must be performance or implementation')
    hardware['scored_on'] = _device(hardware['scored_on'], f'{rid}: hardware.scored_on')
    self_served = e['source'] in SELF_SERVED
    if self_served:
        hardware['served_on'] = _device(hardware.get('served_on'), f'{rid}: hardware.served_on')
    elif hardware.get('served_on') is not None:
        raise RecordError(f'{rid}: hardware.served_on is only for a model we served ourselves ({SELF_SERVED})')
    else:
        hardware['served_on'] = None
    if not isinstance(model, dict) or set(model) - {'name', 'precision', 'served_by'}:
        raise RecordError(f'{rid}: model takes only name, precision and served_by; put details in notes')
    if model.get('name') not in MODEL_NAMES:
        raise RecordError(f'{rid}: model.name {model.get("name")!r} is not one of {MODEL_NAMES}')
    if self_served:
        sb = model.get('served_by')
        if not model.get('precision') or not isinstance(sb, dict) or set(sb) != {'server', 'version', 'settings'}:
            raise RecordError(f'{rid}: a model we served needs precision and served_by {{"server", "version", '
                              '"settings"}')
        if sb['server'] not in SERVERS:
            raise RecordError(f'{rid}: model.served_by.server {sb["server"]!r} is not one of {SERVERS}')
    elif model.get('served_by') is not None or model.get('precision') is not None:
        raise RecordError(f'{rid}: precision and served_by are only for a model we served ourselves')
    else:
        model['precision'], model['served_by'] = None, None
    return e


class RecordBuilder:
    """Collects one record's files in a staging folder; nothing reaches the dataset until commit()."""

    def __init__(self, out: Path, record_id: str):
        self.final = out / 'records' / record_id
        if self.final.exists():
            raise RecordExists(f'{self.final} already exists; records are never overwritten')
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

    def write_cut(self, src: Path, dest_rel: str, data: bytes, cut: str) -> Path:
        """Store whole original lines cut out of a file (a slice of a session), refusing them if they look like they
        hold a secret; provenance keeps the source's hash and which lines were kept."""
        dest = self.stage / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        hits = find_secrets(dest)
        if hits:
            raise RecordError(f'{src} ({cut}) matches {hits}; blank the secret, note it under "scrub" in the sheet, '
                              'convert again')
        self.files.append({'path': dest_rel, 'source': str(src), 'source_sha256': sha256(src), 'cut': cut,
                           'sha256': sha256(dest), 'bytes': dest.stat().st_size})
        return dest

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



MESSAGES_FORMAT = 'messages-v1'
MESSAGE_FIELDS = ('turn', 'role', 'content', 'reasoning', 'tool_calls', 'tool_call_id', 'agent', 'source_index',
                  'event', 'extra')


def unify(rows: list[dict]) -> list[dict]:
    """One message shape for every source (docs/trace-records.md, "A chat file"): the same fields on every line,
    content always plain text, tool calls in the OpenAI function shape, and whatever only one source has under
    extra. Running it on rows already in this shape changes nothing."""
    out = []
    for turn, r in enumerate(rows):
        r = dict(r)
        extra = dict(r.pop('extra', None) or {})
        role = r.pop('role')
        content = r.pop('content', None)
        source_index = r.pop('source_index', None)
        for k in ('call', 'step'):
            if k in r:
                v = r.pop(k)
                source_index = v if source_index is None else source_index
        if role == 'event':
            for k in ('summary', 'error', 'status'):
                if k in r:
                    v = r.pop(k)
                    content = v if content is None else content
            if content is None and r.get('added_parts'):
                content = _text_of(r['added_parts'])
        if content is not None and not isinstance(content, str):
            if isinstance(content, list):
                extra.setdefault('content_parts', len(content))
            content = _text_of(content)
        msg = {'turn': turn, 'role': role, 'content': content, 'reasoning': r.pop('reasoning', None) or None,
               'tool_calls': r.pop('tool_calls', None) or None, 'tool_call_id': r.pop('tool_call_id', None),
               'agent': r.pop('agent', None), 'source_index': source_index, 'event': r.pop('event', None)}
        r.pop('turn', None)
        extra.update(r)
        msg['extra'] = extra
        out.append(msg)
    return out


CHATS_DIR = 'conversation/chats'
CHATS_LAYOUT = 'chats-v1'  # every record: one file per conversation in conversation/chats/, for every source
SOLVER = 'solver'  # the name of the one chat of a run with a single agent (an open-model run, a person's session)


def chat_file(n: int, name: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-').lower()
    return f'{CHATS_DIR}/{n:02d}-{slug}.jsonl'


def solver_chat(rows: list[dict]) -> tuple[dict, list[dict]]:
    """The one chat of a single-agent run: its entry for record.json and its unified messages."""
    msgs = unify([{**r, 'agent': SOLVER} for r in rows])
    return {'n': 1, 'name': SOLVER, 'file': chat_file(1, SOLVER), 'messages': len(msgs), 'synthetic': False}, msgs


def _viewer_message(m: dict) -> dict:
    """A unified message with its tool calls as one JSON string, so that the Hugging Face viewer reads every row."""
    v = {k: m.get(k) for k in MESSAGE_FIELDS if k != 'extra'}
    v['tool_calls'] = json.dumps(m['tool_calls'], ensure_ascii=False) if m.get('tool_calls') else None
    return v


DATASET_CARD = """---
configs:
- config_name: runs
  data_files: viewer/chats.jsonl
  default: true
- config_name: index
  data_files: index.jsonl
---
# RSI trace records

One record per run of an agent on a Φ-Bench task, labeled by the task's own verifier: successful (final reward
above 0), unsuccessful (reward 0), invalid (not a fair attempt) or unscored. Private to the group.

- `viewer/chats.jsonl`: one row per chat (conversation) with its messages and its run's label, for reading in the
  viewer (generated).
- `index.jsonl`: one line per record (generated).
- `records/<record_id>/`: the record itself: `record.json`, the prompt, the raw files byte for byte, the chats in
  `conversation/chats/` (one file per conversation, the same for every source), the code and the verifier output.

Format and decisions: `docs/trace-records.md`, `CONTEXT.md` and `docs/adr/` in the phibench-mi210 repository.
"""


def rebuild_index(out: Path) -> int:
    """index.jsonl and viewer/chats.jsonl from every record; rebuilt whole so that they can never disagree with the
    records. Writes the dataset card."""
    rows, chats = [], []
    for rec_path in sorted((out / 'records').glob('*/record.json')):
        r = json.loads(rec_path.read_text())
        conv = r['conversation']
        if 'chats' not in conv:
            raise RecordError(f'{rec_path}: no conversation.chats; run "remessage" to bring it to {CHATS_LAYOUT}')
        row = {'record_id': r['record_id'], 'task': r['task']['id'], 'source': r['source'],
               'model': r['model'].get('name'), 'label': r['outcome'].get('label'),
               'reward': r['outcome'].get('reward'), 'speedup': r['outcome'].get('speedup'),
               'prompt_vs_task': r['task'].get('prompt_vs_task'), 'chats': len(conv['chats'])}
        rows.append(row)
        for c in conv['chats']:
            p = rec_path.parent / c['file']
            msgs = unify([json.loads(l) for l in p.read_text().splitlines() if l.strip()])
            chats.append({**row, 'chat': c['name'], 'chat_n': c['n'], 'synthetic': c['synthetic'],
                          'label_rule': r['outcome'].get('label_rule'), 'harness': r['agent'].get('harness'),
                          'harness_version': r['agent'].get('version'),
                          'invalid_reason': r['outcome'].get('invalid_reason'),
                          'messages': [_viewer_message(m) for m in msgs]})
    with open(out / 'index.jsonl', 'w') as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    (out / 'viewer').mkdir(exist_ok=True)
    with open(out / 'viewer' / 'chats.jsonl', 'w') as fh:
        for c in chats:
            fh.write(json.dumps(c, ensure_ascii=False) + '\n')
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
    """A summarizer call of Qwen Code: the instruction is the last user message, or (its second form, seen on
    2026-11-26 in a secure-agg run) the system message."""
    msgs = request['messages']
    last, first = msgs[-1], msgs[0]
    return ((last['role'] == 'user' and _text_of(last.get('content')).startswith(COMPACTION_PROMPT))
            or (first['role'] == 'system' and _text_of(first.get('content')).startswith(COMPACTION_PROMPT)))


def _prepended_parts(msgs: list[dict], keys: list[tuple], ctx_msgs: list[dict], ctx: list[tuple]):
    """(index, added parts) when the new request equals the previous context except that one user message got text
    parts added in front of its old parts (Qwen Code re-attaches its session-start reminders to the summary message
    on the second call after a compaction); None otherwise."""
    if len(keys) < len(ctx):
        return None
    diff = [j for j in range(len(ctx)) if keys[j] != ctx[j]]
    if len(diff) != 1:
        return None
    j = diff[0]
    old, new = ctx_msgs[j], msgs[j]
    if old['role'] != 'user' or new['role'] != 'user':
        return None
    old_parts, new_parts = old.get('content'), new.get('content')
    if not (isinstance(old_parts, list) and isinstance(new_parts, list) and len(new_parts) > len(old_parts)):
        return None
    if new_parts[len(new_parts) - len(old_parts):] != old_parts:
        return None
    return j, new_parts[:len(new_parts) - len(old_parts)]


def messages_from_openai_calls(calls: list[dict]) -> tuple[list[dict], dict]:
    """The run as one list of messages from its raw model calls, in order (docs/trace-records.md).

    A call's request is the conversation so far; only what is new since the previous call is emitted. A compaction
    call becomes one event line (with the summary and the model's reasoning); consecutive summarizer calls are one
    compaction in several parts. The next call's rewritten history is emitted whole, after its system message. A call
    whose system message is not the solver's own (a harness side call, such as Qwen Code's memory extraction) becomes
    one side_call event and does not touch the conversation. A failed call becomes an event line. Returns (rows,
    counts)."""
    rows, ctx, ctx_msgs = [], None, None
    counts = {'calls': len(calls), 'failed_calls': 0, 'compactions': 0, 'history_rewrites': 0, 'side_calls': 0}
    system_key = None
    solver_system = None
    in_compaction = False
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
            part = 2 if in_compaction else 1
            if not in_compaction:
                counts['compactions'] += 1
            rows.append({'role': 'event', 'event': 'compaction', 'part': part, 'summary': message.get('content'),
                         'reasoning': message.get('reasoning_content'), 'call': i})
            in_compaction = True
            ctx = None
            continue
        in_compaction = False
        if msgs[0]['role'] == 'system':
            if solver_system is None:
                solver_system = _key(msgs[0])
            elif _key(msgs[0]) != solver_system:
                counts['side_calls'] += 1
                rows.append({'role': 'event', 'event': 'side_call', 'system': _text_of(msgs[0].get('content'))[:300],
                             'content': message.get('content'), 'reasoning': message.get('reasoning_content'),
                             'call': i})
                continue
        keys = [_key(m) for m in msgs]
        updated = _prepended_parts(msgs, keys, ctx_msgs, ctx) if ctx is not None else None
        if updated is not None:
            j, added = updated
            rows.append({'role': 'event', 'event': 'message_updated', 'index': j, 'added_parts': added, 'call': i})
            new = msgs[len(ctx):]
        elif ctx is not None and keys[:len(ctx)] == ctx:
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
        ctx_msgs = msgs + [message]
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
        # No submission loop ran: the final score measured the tree exactly as the agent left it.
        record['outcome']['final_tree'] = 'working tree as left (never submitted)'
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


def _common_harness(entry: dict, run: Path, builder: 'RecordBuilder', record: dict,
                    prompt_from: tuple[str, str] | None = None) -> dict:
    """prompt_from = (text, where it came from), for a harness that keeps the prompt only inside its own log."""
    if (run / 'prompt.txt').is_file():
        builder.copy(run / 'prompt.txt', 'prompt.txt')
        record['prompt_source'] = 'prompt.txt'
    elif prompt_from is not None:
        dest = builder.stage / 'prompt.txt'
        dest.write_text(prompt_from[0])
        record['prompt_source'] = f'generated from {prompt_from[1]}'
    else:
        raise RecordError(f'missing file {run / "prompt.txt"}')
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
    chat, msgs = solver_chat(rows)
    builder.write_jsonl(chat['file'], msgs)
    record['conversation'] = {'layout': CHATS_LAYOUT, 'chats': [chat], 'main': 'conversation/raw/openai',
                              'harness_log': 'conversation/raw/stream.jsonl',
                              'reasoning_included': True, 'thinking_dropped': False, **counts}
    _loop(entry, run, builder, record)
    _final_score(entry, run, builder, record)


def messages_from_mini_swe_agent(traj: dict) -> list[dict]:
    rows = []
    for i, m in enumerate(traj['messages']):
        if m['role'] == 'exit':
            rows.append({'role': 'event', 'event': 'exit', 'status': m.get('content'), 'step': i})
            continue
        extra = m.get('extra') or {}
        if m['role'] == 'user' and extra.get('interrupt_type') == 'FormatError':
            # mini-swe-agent keeps a rejected reply only inside the format-error message: restore it as the
            # assistant turn it was, marked, before the error message.
            resp = extra.get('response') or {}
            reply = ((resp.get('choices') or [{}])[0].get('message') or {}) if resp else {}
            content = reply.get('content') if reply else None
            if content is None:
                content = extra.get('model_response')
            row = {'role': 'assistant', 'content': content, 'format_error': True}
            if reply.get('reasoning_content'):
                row['reasoning'] = reply['reasoning_content']
            row['step'] = i
            rows.append(row)
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
    traj = _read_json(run / 'trajectory.json')
    task_msg = traj['messages'][1] if len(traj.get('messages', [])) > 1 else None
    if task_msg is None or task_msg.get('role') != 'user':
        raise RecordError(f'{entry["record_id"]}: trajectory.json messages[1] is not the task prompt')
    result = _common_harness(entry, run, builder, record,
                             prompt_from=(task_msg['content'], 'trajectory.json messages[1]'))
    record['outcome']['harness_exit_status'] = result.get('exit_status')
    builder.copy(run / 'trajectory.json', 'conversation/raw/trajectory.json')
    version = traj.get('info', {}).get('mini_version')
    if entry['agent'].get('version') not in (None, version):
        raise RecordError(f'{entry["record_id"]}: sheet says mini-swe-agent {entry["agent"]["version"]}, '
                          f'the trajectory says {version}')
    record['agent']['version'] = version
    record['model']['served_name'] = traj.get('info', {}).get('config', {}).get('model', {}).get('model_name')
    rows = messages_from_mini_swe_agent(traj)
    chat, msgs = solver_chat(rows)
    builder.write_jsonl(chat['file'], msgs)
    record['conversation'] = {'layout': CHATS_LAYOUT, 'chats': [chat], 'main': 'conversation/raw/trajectory.json',
                              'reasoning_included': True, 'thinking_dropped': False, 'compactions': 0,
                              'steps': sum(1 for r in rows if r['role'] == 'assistant')}
    _loop(entry, run, builder, record)
    _final_score(entry, run, builder, record)


# ---------------------------------------------------------------------------------------------------- Claude Code

THINKING_BLOCKS = ('thinking', 'redacted_thinking')


def count_thinking(path: Path) -> dict:
    """How many thinking blocks a Claude Code JSONL file holds, and how many of them carry text (Claude Code may
    store a thinking block with an empty text and only a signature)."""
    blocks = with_text = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        msg = json.loads(line).get('message')
        if isinstance(msg, dict) and isinstance(msg.get('content'), list):
            for b in msg['content']:
                if isinstance(b, dict) and b.get('type') in THINKING_BLOCKS:
                    blocks += 1
                    with_text += bool(b.get('thinking'))
    return {'thinking_blocks': blocks, 'thinking_blocks_with_text': with_text}


def _tool_text(content) -> str:
    if isinstance(content, list):
        return '\n'.join(b.get('text', '') if isinstance(b, dict) else str(b) for b in content)
    return '' if content is None else str(content)


def messages_from_claude_code(path: Path) -> list[dict]:
    """One Claude Code conversation as messages: an assistant reply split over several lines (same message id) is
    joined; tool results become tool messages; thinking text becomes the reply's reasoning; the harness's own
    bookkeeping lines are left out."""
    rows, current = [], None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        kind, msg = obj.get('type'), obj.get('message')
        if kind not in ('user', 'assistant') or not isinstance(msg, dict):
            continue
        content = msg.get('content')
        if kind == 'assistant':
            if current is None or current['_id'] != msg.get('id'):
                current = {'role': 'assistant', 'content': '', 'tool_calls': [], '_id': msg.get('id'),
                           'model': msg.get('model'), 'timestamp': obj.get('timestamp')}
                rows.append(current)
            for b in content if isinstance(content, list) else [{'type': 'text', 'text': content or ''}]:
                if b.get('type') == 'text':
                    current['content'] = (current['content'] + '\n' if current['content'] else '') + b.get('text', '')
                elif b.get('type') == 'thinking' and b.get('thinking'):
                    prev = current.get('reasoning')
                    current['reasoning'] = (prev + '\n' if prev else '') + b['thinking']
                elif b.get('type') == 'tool_use':
                    current['tool_calls'].append({'id': b.get('id'), 'type': 'function', 'function': {
                        'name': b.get('name'), 'arguments': json.dumps(b.get('input'), ensure_ascii=False)}})
            continue
        current = None
        if isinstance(content, str):
            rows.append({'role': 'user', 'content': content, 'timestamp': obj.get('timestamp')})
            continue
        for b in content or []:
            if b.get('type') == 'tool_result':
                rows.append({'role': 'tool', 'tool_call_id': b.get('tool_use_id'), 'content': _tool_text(b.get('content')),
                             'is_error': bool(b.get('is_error')), 'timestamp': obj.get('timestamp')})
            elif b.get('type') == 'text':
                rows.append({'role': 'user', 'content': b.get('text', ''), 'timestamp': obj.get('timestamp')})
    for r in rows:
        r.pop('_id', None)
        if r.get('tool_calls') == []:
            r.pop('tool_calls')
    return rows


# ---------------------------------------------------------------------------------------------------- GEAK runs

SYNTHETIC_MODEL = '<synthetic>'  # Claude Code's model name on replies that the harness wrote, not a model
PROMPT_VS_TASK = ('instruction', 'same problem, other prompt', 'different problem')
ROW_RULE_MIN = 1.1  # the paper's minimum; the rule a GEAK result had to pass to become a row (ADR 0010)

# The GEAK working files worth keeping with the record (relative to exp/team_task_*/task/); workspaces, baselines
# and profiler output stay in the run folder, which provenance names.
GEAK_KEEP = ('*.md', '*.json', '*.jsonl', '*.diff', 'round_*/engineer_*/worker_result.json',
             'round_*/engineer_*/report.md', 'round_*/engineer_*/best_patch.diff', 'round_*/engineer_*/verify/*.log',
             'round_*/integrate/*.diff', 'round_*/integrate/*.log', 'round_*/*.md', 'round_*/*.json')


def geak_agents(run: Path) -> list[dict]:
    """The agents of a GEAK run, in order, from chat/README.md and each chat file's Source line."""
    readme = run / 'chat' / 'README.md'
    if not readme.is_file():
        raise RecordError(f'missing {readme}: the per-agent index of the GEAK run')
    agents = []
    for line in readme.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 7 or not cells[0].isdigit():
            continue
        md = re.search(r'\(([^)]+)\)', cells[6])
        md_path = run / 'chat' / (md.group(1) if md else cells[6])
        src = re.search(r'^Source: `([^`]+)`', md_path.read_text(), re.M) if md_path.is_file() else None
        if src is None:
            raise RecordError(f'{md_path}: no "Source:" line naming the agent\'s conversation file')
        agents.append({'n': int(cells[0]), 'started': cells[1], 'role': cells[2], 'models': cells[5],
                       'chat_file': md_path.name, 'source': src.group(1)})
    if not agents:
        raise RecordError(f'{readme}: no agents listed')
    return agents


def geak_evidence(entry: dict) -> dict:
    """The label of a GEAK run from its independent measurement (ADR 0010), never from GEAK's own report."""
    ev = entry.get('evidence')
    if not ev or ev.get('kind') not in ('round_check', 'piece'):
        raise RecordError(f'{entry["record_id"]}: evidence {{"kind": "round_check" | "piece", ...}} is required')
    if ev['kind'] == 'round_check':
        v = _read_json(Path(ev['dir']) / 'verdict.json')
        s = v.get('summary')
        if s is None:  # e.g. "no_candidate": GEAK committed nothing, so there was nothing to measure
            return {'passed': False, 'speedup': None, 'verdict': v.get('verdict'), 'reason': v.get('reason'),
                    'rule': f'Round check: verdict pass, correctness PASS, control identical, score >= {ROW_RULE_MIN}'}
        ok = (s.get('verdict') == 'pass' and s.get('correctness') == 'PASS' and s.get('control_identical') is True
              and not s.get('cases_failed'))
        score = s.get('score_speedup')
        return {'passed': bool(ok and score is not None and score >= ROW_RULE_MIN), 'speedup': score,
                'verdict': s.get('verdict'), 'correctness': s.get('correctness'),
                'control_identical': s.get('control_identical'),
                'rule': f'Round check: verdict pass, correctness PASS, control identical, score >= {ROW_RULE_MIN}'}
    scores = _read_json(Path(ev['scores']))
    hit = [p for p in scores if p.get('id') == ev['piece']]
    if len(hit) != 1:
        raise RecordError(f'{entry["record_id"]}: piece {ev["piece"]!r} not found once in {ev["scores"]}')
    p = hit[0]
    return {'passed': bool(p.get('qualifies')), 'speedup': p.get('target_speedup'), 'target_rows': p.get('target_rows'),
            'whole_layer_speedup': p.get('whole_layer_speedup'), 'correctness': p.get('correctness'),
            'reasons': p.get('reasons', []), 'rule': f'piece rule of the frozen plan {ev.get("plan", "?")}'}


def _check_row(entry: dict, evidence_dir: str) -> None:
    """A successful GEAK record needs its row, and the row's provenance must name this run and this evidence."""
    row = entry.get('row')
    if not row or not row.get('provenance'):
        raise RecordError(f'{entry["record_id"]}: the run passed, so it is a successful record only with its row; '
                          'give "row": {"provenance": <the row\'s provenance json>} (ADR 0008, 0010)')
    prov = _read_json(Path(row['provenance']))
    if Path(prov['run']['dir']).resolve() != Path(entry['run_dir']).resolve():
        raise RecordError(f'{entry["record_id"]}: the row\'s provenance names run {prov["run"]["dir"]}, not this run')
    if Path(prov['round_check']['dir']).resolve() != Path(evidence_dir).resolve():
        raise RecordError(f'{entry["record_id"]}: the row was accepted on {prov["round_check"]["dir"]}, '
                          f'not on the evidence {evidence_dir}')
    if prov.get('row', {}).get('task_dir') != entry['task']['id']:
        raise RecordError(f'{entry["record_id"]}: the provenance is for row {prov.get("row", {}).get("task_dir")!r}, '
                          f'not task {entry["task"]["id"]!r}')
    piece = entry['evidence'].get('piece')
    if piece is not None and (prov.get('piece') or {}).get('id') != piece:
        raise RecordError(f'{entry["record_id"]}: the row came from piece {(prov.get("piece") or {}).get("id")!r}, '
                          f'not {piece!r}')


def _copy_matching(builder: 'RecordBuilder', base: Path, patterns: tuple, dest_rel: str) -> list[str]:
    copied = set()
    for pat in patterns:
        for f in sorted(base.glob(pat)):
            rel = f.relative_to(base).as_posix()
            if f.is_file() and rel not in copied:
                builder.copy(f, f'{dest_rel}/{rel}')
                copied.add(rel)
    return sorted(copied)


def convert_geak_run(entry: dict, builder: 'RecordBuilder', record: dict) -> None:
    run = Path(entry['run_dir'])
    outcome = _read_json(run / 'outcome.json')
    for name in ('outcome.json', 'receipt.json', 'driver-spec.json', 'driver-result.json', 'container.log',
                 'transcript.jsonl'):
        _optional(builder, run / name, f'harness/{name}')
    # the conversations: every agent's Claude Code file as recorded, thinking included, plus the readable copies
    claude_root = run / 'home' / '.claude' / 'projects'
    files = sorted(claude_root.rglob('*.jsonl')) if claude_root.is_dir() else []
    if not files:
        raise RecordError(f'{entry["record_id"]}: no Claude Code conversations under {claude_root}')
    thinking = {'thinking_blocks': 0, 'thinking_blocks_with_text': 0}
    for f in files:
        builder.copy(f, f'conversation/raw/claude/{f.relative_to(claude_root).as_posix()}')
        for k, v in count_thinking(f).items():
            thinking[k] += v
    _optional(builder, run / 'chat', 'conversation/readable')
    chats = []
    for a in geak_agents(run):
        src = Path(a['source'])
        if not src.is_file():
            raise RecordError(f'{entry["record_id"]}: agent {a["n"]} ({a["role"]}) conversation {src} is missing')
        rows = messages_from_claude_code(src)
        rel = chat_file(a['n'], a['role'])
        builder.write_jsonl(rel, unify([{**r, 'agent': a['role']} for r in rows]))
        raw_rel = (f'conversation/raw/claude/{src.relative_to(claude_root).as_posix()}'
                   if claude_root in src.parents else None)
        # A chat whose every reply came from the harness, not a model (Claude Code writes such replies with model
        # "<synthetic>", for example when GEAK stops and closes its remaining steps), is kept and marked.
        models = {r.get('model') for r in rows if r['role'] == 'assistant'}
        synthetic = bool(models) and models <= {SYNTHETIC_MODEL}
        chats.append({'n': a['n'], 'name': a['role'], 'file': rel, 'messages': len(rows), 'synthetic': synthetic,
                      'started': a['started'], 'models': a['models'], 'readable': f'conversation/readable/{a["chat_file"]}',
                      'raw': raw_rel, 'source': a['source']})
    first_user = next((r['content'] for r in messages_from_claude_code(Path(geak_agents(run)[0]['source']))
                       if r['role'] == 'user'), None)
    if not first_user:
        raise RecordError(f'{entry["record_id"]}: the top-level agent has no first user message to use as the prompt')
    (builder.stage / 'prompt.txt').write_text(first_user)
    record['prompt_source'] = "generated from the top-level agent's first user message"
    record['conversation'] = {
        'layout': CHATS_LAYOUT, 'chats': chats, 'main': 'conversation/raw/claude', 'readable': 'conversation/readable',
        'synthetic_chats': sum(c['synthetic'] for c in chats),
        'reasoning_included': True, 'thinking_dropped': False, **thinking}
    record['model']['transcript_models'] = outcome.get('transcript_models')
    record['outcome']['harness_status'] = outcome.get('status')
    record['time'] = {'seconds': outcome.get('seconds')}
    # GEAK's own working files, and each engineer attempt with GEAK's own (self-reported) result
    tasks = sorted((run / 'exp').glob('team_task_*/task'))
    if len(tasks) != 1:
        raise RecordError(f'{entry["record_id"]}: expected one exp/team_task_*/task folder, found {len(tasks)}')
    task_dir = tasks[0]
    _copy_matching(builder, task_dir, GEAK_KEEP, 'harness/geak')
    for wr in sorted(task_dir.glob('round_*/engineer_*/worker_result.json')):
        w = json.loads(wr.read_text())
        rel = wr.parent.relative_to(task_dir).as_posix()
        record['submissions'].append({
            'n': f'{wr.parent.parent.name}/{wr.parent.name}', 'engineer': w.get('engineer_id'),
            'specialty': w.get('specialty'), 'strategy': w.get('strategy'), 'geak_status': w.get('status'),
            'geak_reported_speedup': w.get('speedup_weighted'), 'measurement_valid': w.get('measurement_valid'),
            'diff': f'harness/geak/{rel}/best_patch.diff' if (wr.parent / 'best_patch.diff').is_file() else None,
            'note': 'self-reported by GEAK; not the label'})
    if (task_dir / 'final_patch.diff').is_file():
        builder.copy(task_dir / 'final_patch.diff', 'code/final.patch')
    # the independent measurement that sets the label
    ev = entry['evidence']
    result = geak_evidence(entry)
    ev_dir = Path(ev['dir'] if ev['kind'] == 'round_check' else ev['work_dir'])
    _copy_matching(builder, ev_dir, ('prepare.json', 'verdict.json', '*.log'), 'verifier/round-check')
    for log in sorted(ev_dir.parent.glob(ev_dir.name + '.*.log')):
        builder.copy(log, f'verifier/{log.name}')
    if ev['kind'] == 'piece':
        builder.copy(Path(ev['scores']), 'verifier/piece-scores.json')
        _optional(builder, ev_dir.parent / 'piece.patch', 'code/reference.patch')
    prep = ev_dir / 'prepare.json'
    if prep.is_file():
        cand = (json.loads(prep.read_text()).get('source') or {}).get('candidate_dir')
        if cand and Path(cand).is_dir():
            _copy_matching(builder, Path(cand), ('*.py', '*.json'), 'code/final')
    # Context only, never the label: the whole run's own result when the row was made from one piece of it (the
    # first MoE run: its whole result was 1.094x, two of its table entries each passed alone).
    if entry.get('run_result'):
        rr_dir = Path(entry['run_result']['dir'])
        _copy_matching(builder, rr_dir, ('prepare.json', 'verdict.json'), 'verifier/run-round-check')
        s = _read_json(rr_dir / 'verdict.json').get('summary') or {}
        record['outcome']['run_result'] = {
            'verdict': s.get('verdict'), 'correctness': s.get('correctness'), 'speedup': s.get('score_speedup'),
            'dir': str(rr_dir), 'note': entry['run_result'].get('note', 'the whole run\'s result; not the label')}
    if result['passed']:
        _check_row(entry, str(ev_dir))
        lab = 'successful'
    else:
        lab = 'unsuccessful'
    record['outcome'].update(label=lab, label_rule=result.pop('rule'), speedup=result.pop('speedup'),
                             reward=None, evidence={k: v for k, v in result.items() if k != 'passed'},
                             evidence_dir=str(ev_dir))


# ---------------------------------------------------------------------------------------------- a person's session

def _request_text(obj: dict) -> str | None:
    """The text of a line that is a person's request in a Claude Code session (not a tool result, not a harness note)."""
    if obj.get('type') != 'user' or obj.get('isMeta') or obj.get('isSidechain'):
        return None
    content = (obj.get('message') or {}).get('content')
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content and all(isinstance(b, dict) and b.get('type') == 'text' for b in content):
        return '\n'.join(b.get('text', '') for b in content)
    return None


def session_requests(session: Path) -> list[dict]:
    """Every request a person typed in a Claude Code session, with the uuid to start a slice at (SOP step 6)."""
    out = []
    for n, line in enumerate(session.read_text().splitlines(), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        text = _request_text(obj)
        if text is not None and obj.get('uuid'):
            out.append({'line': n, 'uuid': obj['uuid'], 'timestamp': obj.get('timestamp'), 'text': text})
    return out


def cut_session(session: Path, from_uuid: str, to_uuid: str | None) -> tuple[bytes, dict]:
    """The original lines of a session from the line with from_uuid to the line with to_uuid (or the end), byte for
    byte; and what was cut (line numbers, first and last timestamp, Claude Code versions)."""
    lines = session.read_bytes().splitlines(keepends=True)
    parsed = [json.loads(l) if l.strip() else {} for l in lines]
    start = next((i for i, o in enumerate(parsed) if o.get('uuid') == from_uuid), None)
    if start is None:
        raise RecordError(f'{session}: no line with uuid {from_uuid}; list the requests with "session-requests"')
    if _request_text(parsed[start]) is None:
        raise RecordError(f'{session}: line {start + 1} ({from_uuid}) is not a request a person typed; a slice starts '
                          'at your request for the work')
    end = len(lines) - 1
    if to_uuid:
        end = next((i for i, o in enumerate(parsed) if o.get('uuid') == to_uuid), None)
        if end is None or end < start:
            raise RecordError(f'{session}: no line with uuid {to_uuid} after {from_uuid}')
    kept = parsed[start:end + 1]
    stamps = [o['timestamp'] for o in kept if o.get('timestamp')]
    return b''.join(lines[start:end + 1]), {
        'lines': [start + 1, end + 1], 'from_uuid': from_uuid, 'to_uuid': kept[-1].get('uuid'),
        'started': min(stamps) if stamps else None, 'finished': max(stamps) if stamps else None,
        'versions': sorted({o['version'] for o in kept if o.get('version')})}


def _first_timestamp(path: Path) -> str | None:
    for line in path.read_text().splitlines():
        if line.strip():
            ts = json.loads(line).get('timestamp')
            if ts:
                return ts
    return None


def convert_claude_code_session(entry: dict, builder: 'RecordBuilder', record: dict) -> None:
    """A contributor's own Claude Code session (SOP step 6): the slice from their request for the work to its end, and
    the helper agents started inside it. Its result became the task's reference solution, so it is successful."""
    session = Path(entry.get('session') or '')
    if not session.is_file():
        raise RecordError(f'{entry["record_id"]}: "session" must name the Claude Code session file '
                          '(~/.claude/projects/<folder>/<session>.jsonl)')
    sl = entry.get('slice') or {}
    if not sl.get('from_uuid'):
        raise RecordError(f'{entry["record_id"]}: "slice": {{"from_uuid": ..., "to_uuid": ...}} is required; list the '
                          'requests of the session with "session-requests"')
    ref = Path(entry.get('task_dir') or '')
    if not entry.get('task_dir') or not (ref / 'oracle.patch').is_file():
        raise RecordError(f'{entry["record_id"]}: task_dir must name the task package whose oracle.patch is this '
                          'session\'s result (ADR 0008: the row comes first)')
    data, cut = cut_session(session, sl['from_uuid'], sl.get('to_uuid'))
    version = entry['agent'].get('version')
    if version is not None and version not in cut['versions']:
        raise RecordError(f'{entry["record_id"]}: sheet says Claude Code {version}, the slice says {cut["versions"]}')
    record['agent']['version'] = version or (cut['versions'][-1] if cut['versions'] else None)
    main_rel = f'conversation/raw/claude/{session.stem}.slice.jsonl'
    staged = builder.write_cut(session, main_rel, data, f'lines {cut["lines"][0]}-{cut["lines"][1]}')
    # helper agents (subagents) that started inside the slice, byte for byte, with their meta files
    sub_dir = session.parent / session.stem / 'subagents'
    subs = []
    for f in sorted(sub_dir.glob('*.jsonl')) if sub_dir.is_dir() else []:
        ts = _first_timestamp(f)
        if ts and cut['started'] and cut['finished'] and cut['started'] <= ts <= cut['finished']:
            subs.append((ts, f))
    subs.sort()
    thinking = count_thinking(staged)
    rows = messages_from_claude_code(staged)
    first = next((r['content'] for r in rows if r['role'] == 'user'), None)
    (builder.stage / 'prompt.txt').write_text(first)
    record['prompt_source'] = "generated from the slice's first request"
    main_chat, main_msgs = solver_chat(rows)
    main_chat['raw'] = main_rel
    builder.write_jsonl(main_chat['file'], main_msgs)
    chats = [main_chat]
    for n, (ts, f) in enumerate(subs, 2):
        raw_rel = f'conversation/raw/claude/subagents/{f.name}'
        builder.copy(f, raw_rel)
        meta = f.with_suffix('.meta.json')
        name = f.stem.removeprefix('agent-')
        if meta.is_file():
            builder.copy(meta, f'conversation/raw/claude/subagents/{meta.name}')
            name = _read_json(meta).get('name') or name
        for k, v in count_thinking(f).items():
            thinking[k] += v
        sub_rows = messages_from_claude_code(f)
        rel = chat_file(n, name)
        msgs = unify([{**r, 'agent': name} for r in sub_rows])
        builder.write_jsonl(rel, msgs)
        models = {r.get('model') for r in sub_rows if r['role'] == 'assistant'}
        chats.append({'n': n, 'name': name, 'file': rel, 'messages': len(msgs),
                      'synthetic': bool(models) and models <= {SYNTHETIC_MODEL}, 'started': ts, 'raw': raw_rel})
    builder.copy(ref / 'oracle.patch', 'code/reference.patch')
    record['conversation'] = {'layout': CHATS_LAYOUT, 'chats': chats, 'main': 'conversation/raw/claude',
                              'slice': cut, 'reasoning_included': True, 'thinking_dropped': False, **thinking}
    record['time'] = {'started': cut['started'], 'finished': cut['finished']}
    record['outcome'].update(label='successful', reference_task_dir=str(ref),
                             label_rule="a contributor's own session: its result is the task's reference solution")


# source, harness -> function(entry, builder, record) that copies the files and fills the record
ADAPTERS = {
    ('claude_code_session', 'claude-code'): convert_claude_code_session,
    ('geak_run', 'geak'): convert_geak_run,
    ('open_model_run', 'qwen-code'): convert_qwen_code,
    ('open_model_run', 'mini-swe-agent'): convert_mini_swe_agent,
}


def _convert_one(entry: dict, out: Path, machine: str) -> None:
    entry = check_sheet_entry(entry)
    # How the prompt the agent saw relates to the task the record is for (ADR 0010): an open model was given the
    # task's own instruction; a GEAK run had GEAK's prompt, for the same problem or (a row made from one piece of its
    # result) for a different, larger one.
    pvt = entry.get('prompt_vs_task', 'instruction' if entry['source'] == 'open_model_run' else None)
    if pvt not in PROMPT_VS_TASK:
        raise RecordError(f'{entry["record_id"]}: prompt_vs_task must be one of {PROMPT_VS_TASK} (required for '
                          f'source {entry["source"]})')
    if entry['source'] == 'open_model_run' and pvt != 'instruction':
        raise RecordError(f'{entry["record_id"]}: an open-model run is given the task\'s instruction')
    if entry['source'] == 'codex_session':
        raise RecordError(f'{entry["record_id"]}: no Codex reader yet; it will be written from a real Codex session '
                          'file (~/.codex/sessions/<year>/<month>/<day>/rollout-*.jsonl), which nobody has shared yet')
    adapter = ADAPTERS.get((entry['source'], entry['agent'].get('harness')))
    if adapter is None:
        raise RecordError(f'{entry["record_id"]}: no converter yet for source {entry["source"]} with harness '
                          f'{entry["agent"].get("harness")!r}; available: {sorted(ADAPTERS)}')
    builder = RecordBuilder(out, entry['record_id'])
    try:
        record = base_record(entry, machine)
        record['task']['prompt_vs_task'] = pvt
        adapter(entry, builder, record)
        chats = (record.get('conversation') or {}).get('chats')
        if not chats or not all((builder.stage / c['file']).is_file() for c in chats):
            raise RecordError(f'{entry["record_id"]}: the reader wrote no chats; every record keeps its '
                              f'conversations in {CHATS_DIR}/ ({CHATS_LAYOUT})')
        # A reader that labels by its own rule (GEAK: the Round check) sets the label itself; otherwise the task's
        # verifier reward decides. An invalid reason always wins.
        preset = record['outcome'].get('label')
        record['outcome']['label'] = ('invalid' if entry.get('invalid_reason') else
                                      preset or label(record['outcome'].get('reward'), None))
        record['outcome'].setdefault('label_rule', "task verifier, Φ-Bench reward > 0")
        if entry.get('invalid_reason'):
            record['outcome']['invalid_reason'] = entry['invalid_reason']
        record['conversation']['messages_format'] = MESSAGES_FORMAT
        builder.commit(record)
    except BaseException:
        builder.abort()
        raise


def convert(sheet: dict, out: Path, keep_going: bool = False) -> dict:
    """Convert every run in the sheet; each run is all or nothing. Without keep_going the first refusal is raised;
    with it, every run is tried and the result lists what was written, what was already stored (records are never
    overwritten) and what was refused, with the reason. The index is rebuilt either way. Paths in the sheet are
    written with named roots ($RESULTS/..., $WORK/...; roots.py) and expanded here, before anything is read."""
    try:
        sheet = expand(sheet)
    except RootError as e:
        raise RecordError(f'sheet: {e}') from e
    machine = sheet.get('machine')
    if not machine:
        raise RecordError('sheet: "machine" (where the run folders are) is required')
    result = {'written': [], 'already_stored': [], 'refused': []}
    try:
        for entry in sheet['runs']:
            try:
                _convert_one(entry, out, machine)
            except RecordExists:
                if not keep_going:
                    raise
                result['already_stored'].append(entry['record_id'])
                continue
            except RecordError as e:
                if not keep_going:
                    raise
                result['refused'].append((entry.get('record_id', '?'), str(e)))
                continue
            result['written'].append(entry['record_id'])
    finally:
        if (out / 'records').is_dir():
            rebuild_index(out)
    return result


def chats_of_record(d: Path, rec: dict) -> list[tuple[dict, list[dict]]]:
    """A stored record's chats rebuilt from the raw files inside it (no run folder needed): (entry, messages) for
    each chat. Reads both the current layout and the earlier one (messages.jsonl, and agents for a GEAK run)."""
    raw, conv = d / 'conversation' / 'raw', rec['conversation']
    if rec['source'] == 'open_model_run' and rec['agent'].get('harness') == 'qwen-code':
        files = sorted((raw / 'openai').glob('openai-*.json'))
        if not files:
            raise RecordError(f'{d}: no raw model calls in {raw / "openai"}')
        rows, _ = messages_from_openai_calls([json.loads(f.read_text()) for f in files])
        return [solver_chat(rows)]
    if rec['source'] == 'open_model_run' and rec['agent'].get('harness') == 'mini-swe-agent':
        return [solver_chat(messages_from_mini_swe_agent(_read_json(raw / 'trajectory.json')))]
    if rec['source'] == 'claude_code_session':
        out = []
        for c in conv['chats']:
            rows = messages_from_claude_code(d / c['raw'])
            if c['n'] == 1:
                entry, msgs = solver_chat(rows)
            else:
                msgs = unify([{**r, 'agent': c['name']} for r in rows])
                entry = {**c, 'messages': len(msgs)}
            out.append(({**c, **entry}, msgs))
        return out
    if rec['source'] == 'geak_run':
        out = []
        for c in conv.get('chats') or conv['agents']:
            name = c.get('name') or c['role']
            if not c.get('raw'):
                raise RecordError(f'{d}: chat {c["n"]} has no raw file inside the record')
            rows = messages_from_claude_code(d / c['raw'])
            entry = {'n': c['n'], 'name': name, 'file': chat_file(c['n'], name), 'messages': len(rows),
                     'synthetic': c['synthetic'], 'started': c.get('started'), 'models': c.get('models'),
                     'readable': c.get('readable') or f'conversation/readable/{c["chat_file"]}',
                     'raw': c['raw'], 'source': c.get('source')}
            out.append((entry, unify([{**r, 'agent': name} for r in rows])))
        return out
    raise RecordError(f'{d}: no message reader for source {rec["source"]} with harness {rec["agent"].get("harness")}')


EARLIER_GENERATED = ('conversation/messages.jsonl', 'conversation/agents/')  # generated files of the earlier layout


def remessage(out: Path) -> list[str]:
    """Rebuild every stored record's chats (conversation/chats/) from the raw files inside it, in the current message
    format and layout. Raw files are never touched; generated files of the earlier layout (messages.jsonl, agents/)
    are removed; record.json notes the format, the layout and when the chats were rebuilt."""
    done = []
    for rec_path in sorted((out / 'records').glob('*/record.json')):
        d, rec = rec_path.parent, json.loads(rec_path.read_text())
        chats = chats_of_record(d, rec)
        new_dir = d / 'conversation' / '.chats.new'
        if new_dir.exists():
            shutil.rmtree(new_dir)
        new_dir.mkdir()
        for entry, msgs in chats:
            with open(new_dir / Path(entry['file']).name, 'w') as fh:
                for row in msgs:
                    fh.write(json.dumps(row, ensure_ascii=False) + '\n')
        old = d / CHATS_DIR
        if old.exists():
            shutil.rmtree(old)
        new_dir.rename(old)
        for rel in EARLIER_GENERATED:
            p = d / rel
            if p.is_dir():
                shutil.rmtree(p)
            elif p.is_file():
                p.unlink()
        conv = rec['conversation']
        for k in ('agents', 'training_unit', 'synthetic_agents'):
            conv.pop(k, None)
        conv['layout'], conv['chats'] = CHATS_LAYOUT, [e for e, _ in chats]
        if rec['source'] == 'geak_run':
            conv['synthetic_chats'] = sum(e['synthetic'] for e, _ in chats)
        if rec['source'] == 'open_model_run' and 'prompt_vs_task' not in rec['task']:
            rec['task']['prompt_vs_task'] = 'instruction'  # an open model is given the task's own instruction
        conv['messages_format'] = MESSAGES_FORMAT
        conv['messages_rebuilt_at'] = dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')
        rec_path.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + '\n')
        done.append(rec['record_id'])
    rebuild_index(out)
    return done


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def upload(out: Path, repo: str, message: str) -> dict:
    """Upload the dataset folder to its private Hugging Face repo, then check that every local file is on Hugging Face
    with the same content and that the repo is still private. Generated files of the earlier layout are removed from
    the repo. Needs huggingface_hub (run with: uv run --with huggingface_hub ...) and a logged-in token."""
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        raise RecordError('upload needs huggingface_hub: uv run --with huggingface_hub python -m '
                          'phibench_mi210.traces upload ...') from e
    if not (out / 'index.jsonl').is_file() or not (out / 'records').is_dir():
        raise RecordError(f'{out} is not a trace dataset folder (no index.jsonl or records/)')
    api = HfApi()
    if not api.dataset_info(repo).private:
        raise RecordError(f'refused: {repo} is not private (ADR 0009: traces stay private)')
    commit = api.upload_folder(folder_path=str(out), repo_id=repo, repo_type='dataset', commit_message=message,
                               ignore_patterns=['.cache/**', '**/.*.staging/**', '**/.chats.new/**'],
                               delete_patterns=['records/*/conversation/messages.jsonl',
                                                'records/*/conversation/agents/*', 'viewer/runs.jsonl'])
    local = {p.relative_to(out).as_posix(): p for p in out.rglob('*')
             if p.is_file() and '.cache' not in p.relative_to(out).parts}
    remote = {e.path: e for e in api.list_repo_tree(repo, repo_type='dataset', recursive=True, expand=True)
              if e.__class__.__name__ == 'RepoFile'}
    bad = []
    for rel, p in local.items():
        e = remote.get(rel)
        if e is None:
            continue
        data = p.read_bytes()
        ok = (hashlib.sha256(data).hexdigest() == e.lfs.sha256) if e.lfs else _git_blob_sha1(data) == e.blob_id
        if not ok:
            bad.append(rel)
    info = api.dataset_info(repo)
    return {'commit': getattr(commit, 'oid', str(commit)), 'local': len(local), 'remote': len(remote),
            'missing_on_hf': sorted(set(local) - set(remote)), 'only_on_hf': sorted(set(remote) - set(local)),
            # Hugging Face adds its own LFS line to .gitattributes on every upload, so that file always differs
            'hash_mismatches': [b for b in bad if b != '.gitattributes'], 'private': info.private, 'sha': info.sha}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('session-requests', help='list the requests you typed in a Claude Code session, with the '
                                                'uuid to start a slice at')
    s.add_argument('--session', type=Path, required=True, help='~/.claude/projects/<folder>/<session>.jsonl')
    u = sub.add_parser('upload', help='upload the dataset folder to its private Hugging Face repo and check it')
    u.add_argument('--out', type=Path, required=True, help='the trace dataset folder')
    u.add_argument('--repo', default='Erland/rsi-traces')
    u.add_argument('--message', required=True, help='the Hugging Face commit message')
    c = sub.add_parser('convert', help='convert the runs listed in a sheet')
    c.add_argument('--sheet', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True, help='the trace dataset folder')
    i = sub.add_parser('index', help='rebuild index.jsonl from the records')
    i.add_argument('--out', type=Path, required=True)
    m = sub.add_parser('remessage', help='rebuild every record\'s messages from its raw files, in the current format')
    m.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.cmd == 'session-requests':
            for r in session_requests(args.session):
                text = ' '.join(r['text'].split())
                print(f'line {r["line"]:>6}  {r["timestamp"]}  {r["uuid"]}  {text[:100]}')
            return 0
        if args.cmd == 'upload':
            r = upload(args.out, args.repo, args.message)
            print(f'commit {r["commit"]} | local files {r["local"]} | on Hugging Face {r["remote"]} | missing there '
                  f'{len(r["missing_on_hf"])} | only there {len(r["only_on_hf"])} | different content '
                  f'{len(r["hash_mismatches"])} | private {r["private"]}')
            problems = r['missing_on_hf'] + r['only_on_hf'] + r['hash_mismatches']
            for p in problems[:20]:
                print('  ', p, file=sys.stderr)
            return 2 if problems or not r['private'] else 0
        if args.cmd == 'remessage':
            done = remessage(args.out)
            print(f'rebuilt the chats of {len(done)} records ({CHATS_LAYOUT}, {MESSAGES_FORMAT})')
            return 0
        if args.cmd == 'convert':
            args.out.mkdir(parents=True, exist_ok=True)
            r = convert(json.loads(args.sheet.read_text()), args.out, keep_going=True)
            print(f'wrote {len(r["written"])} records: {", ".join(r["written"])}')
            if r['already_stored']:
                print(f'already stored, not touched: {", ".join(r["already_stored"])}')
            for rid, why in r['refused']:
                print(f'refused {rid}: {why}', file=sys.stderr)
            if r['refused']:
                return 2
        else:
            print(f'index: {rebuild_index(args.out)} records')
    except RecordError as e:
        print(f'refused: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
