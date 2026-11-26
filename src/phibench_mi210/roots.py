"""Named roots for the machine folders that specs and run sheets point into.

A spec or run sheet never writes a machine path. It writes `$RESULTS/...` or `$WORK/...`, and each machine maps
the names to its own folders in a git-ignored file, `specs/local-roots.json` (or the file named by the environment
variable PHIBENCH_ROOTS):

    {"RESULTS": "/data/erland/results", "WORK": "/vast/users/qirong.ho/erland"}

`specs/local-roots.example.json` lists the names and what each one is. A name that is not in the list, or not set on
this machine, is refused with the file to fix; nothing falls back to a guess.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT_NAMES = {
    'RESULTS': 'the results folder on the MI210 node that runs GEAK and holds the rows (node 303)',
    'WORK': 'the work folder on the machine that serves the open models and runs the CPU tasks (amd_gpu)',
}
REPO = Path(__file__).resolve().parents[2]
NAME = re.compile(r'\$([A-Z][A-Z0-9_]*)')


class RootError(ValueError):
    pass


def roots_file() -> Path:
    return Path(os.environ['PHIBENCH_ROOTS']) if os.environ.get('PHIBENCH_ROOTS') else REPO / 'specs' / 'local-roots.json'


def load_roots() -> dict[str, str]:
    path = roots_file()
    if not path.is_file():
        raise RootError(f'{path} does not exist; copy specs/local-roots.example.json there and set the roots of '
                        'this machine')
    roots = json.loads(path.read_text())
    unknown = sorted(set(roots) - set(ROOT_NAMES))
    if unknown:
        raise RootError(f'{path}: {unknown} are not root names; the names are {sorted(ROOT_NAMES)}')
    return roots


def expand(obj, roots: dict[str, str] | None = None):
    """Every `$NAME` in the strings of obj (a string, list or dict, recursively) replaced by its folder. The roots
    file is read only when a name appears."""
    if isinstance(obj, dict):
        return {k: expand(v, roots) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand(v, roots) for v in obj]
    if not isinstance(obj, str) or '$' not in obj or not NAME.search(obj):
        return obj
    table = roots if roots is not None else load_roots()

    def one(m):
        name = m.group(1)
        if name not in ROOT_NAMES:
            raise RootError(f'${name} in {obj!r} is not a root name; the names are {sorted(ROOT_NAMES)}')
        if name not in table:
            raise RootError(f'${name} in {obj!r} is not set on this machine; add it to {roots_file()}')
        return table[name].rstrip('/')
    return NAME.sub(one, obj)
