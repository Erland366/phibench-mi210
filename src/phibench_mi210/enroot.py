"""Build a Φ-Bench task image with Enroot (no Docker access needed) and run its verifier, fresh each time.

    # host, CPU: unpack the pinned base image and follow the task's environment/Dockerfile step by step
    python -m evaluation.geak.phibench_enroot build --task-dir TASK --base BASE.sqsh --base-sha256 HEX --data DATA

    # one MI210: the Dockerfile's "Score" command, in a throwaway copy of the built image per run
    python -m evaluation.geak.phibench_enroot score --task-dir TASK --data DATA --gpu N --mode oracle --repeat 5 \
        --out OUT [--candidate DIR]

    # one MI210: any command in a throwaway copy of the built image, HOST directories mounted read-write
    python -m evaluation.geak.phibench_enroot exec --task-dir TASK --data DATA --gpu N --mount HOST ... -- CMD...

build: FROM must name the pinned base, which is unpacked from BASE.sqsh after its SHA256 is checked. COPY copies from
the task directory into the root filesystem on the host; RUN runs `/bin/sh -c` inside the container as root, with
the image's environment and the ENVs so far; ENV is appended to the image's /etc/environment; WORKDIR, ENTRYPOINT
and CMD are recorded in DATA/<image>.json. Any other instruction, or a failing step, stops the build.

score: each run gets a hard-link copy of the built image (Docker's fresh container), with real copies of the places
a run writes (WRITABLE), with tests/ at /tests read-only
and, in oracle, negative and baseline2 modes, the task directory at /patches read-only. The run executes ENTRYPOINT
`bash /tests/test.sh` with KERNELBENCH_VERIFY_MODE and the mode's patch variable. After the run the copy is deleted
and the built image is checked unchanged (path, type, size, mode, mtime of every file), or the command stops. Timing
needs the node to itself: every GPU must be idle (no use, no memory held) before each run and again after it; a run
that finds the node busy afterwards is renamed <mode>-<k>-busy and not counted, and the command stops.
/logs/verifier of each run is kept in OUT/<mode>-<k>/, and OUT/<mode>.json lists the reward of every run. In
candidate mode, --candidate DIR copies the editable files found in DIR into /app/repo first (the solver's working tree).
Mode solve is the package's own "grade after landing" line and Φ-Bench's `--agent oracle` check: the task directory at
/task read-only, `bash /task/solution/solve.sh && bash /tests/test.sh`, scored in the default candidate mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

BASE_IMAGE = 'vllm/vllm-openai-rocm:v0.32.0'
PATCH_VARS = {'oracle': ('KERNELBENCH_ORACLE_PATCH', 'oracle.patch'),
              'negative': ('KERNELBENCH_NEGATIVE_PATCH', 'negative.patch'),
              'baseline2': ('KERNELBENCH_BASELINE2_PATCH', 'baseline2.patch')}
MODES = ('candidate', 'noop', *PATCH_VARS, 'solve')
WRITABLE = ('.lock', 'etc/passwd', 'etc/group', 'etc/shadow', 'etc/apt/apt.conf.d/enroot', 'app', 'root', 'logs', 'opt/loop', 'opt/verifier-correctness-manifest.json')
RC = 'cd "${PHIBENCH_WORKDIR:-/}" && exec "$@"\n'


class BuildError(RuntimeError):
    pass


def instructions(dockerfile: str) -> list[tuple[str, str]]:
    """(INSTRUCTION, argument) pairs; comments dropped, backslash-newline continuations joined."""
    out, current = [], ''
    for line in dockerfile.splitlines():
        if not current and (not line.strip() or line.lstrip().startswith('#')):
            continue
        if current and line.lstrip().startswith('#'):
            continue  # a comment inside a continued instruction
        if line.endswith('\\'):
            current += line[:-1]  # Docker drops the backslash and the newline
            continue
        current += line
        word, _, arg = current.strip().partition(' ')
        out.append((word.upper(), arg.strip()))
        current = ''
    if current:
        raise BuildError('Dockerfile ends inside a continued instruction')
    return out


def env_pairs(arg: str) -> list[tuple[str, str]]:
    pairs = []
    for token in shlex.split(arg):
        key, eq, value = token.partition('=')
        if not eq:
            raise BuildError(f'ENV {arg!r}: only the KEY=value form is supported')
        pairs.append((key, value))
    return pairs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def image_name(task_dir: Path) -> str:
    return 'phibench-' + task_dir.resolve().name


def enroot_env(data: Path, gpu: str | None = None, extra: dict | None = None) -> dict:
    """The enroot process's environment: nothing from the host but what Enroot itself needs. Enroot passes it into the
    container and adds the image's /etc/environment, as Docker gives a container only the image's ENV."""
    runtime = Path(tempfile.mkdtemp(prefix='phibench-enroot-', dir='/tmp'))
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/root', 'TERM': 'dumb', 'ENROOT_DATA_PATH': str(data),
           'ENROOT_RUNTIME_PATH': str(runtime), 'ENROOT_CACHE_PATH': str(data / '.cache'),
           'ENROOT_TEMP_PATH': str(data / '.tmp'), 'ENROOT_MOUNT_HOME': 'no', 'ENROOT_ROOTFS_WRITABLE': 'yes'}
    if gpu is not None:
        env['HIP_VISIBLE_DEVICES'] = gpu
    env.update(extra or {})
    return env


def start(name: str, data: Path, command: list[str], rc: Path, workdir: str, mounts=(), gpu=None, extra=None,
          log=None, rw_mounts=()) -> int:
    cmd = ['enroot', 'start', '--root', '--rw', '--rc', str(rc)]
    for access, pairs in (('ro', mounts), ('rw', rw_mounts)):
        for source, target in pairs:
            kind = 'dir' if Path(source).is_dir() else 'file'
            cmd += ['--mount', f'{source}:{target}:none:x-create={kind},bind,{access}']
    env = enroot_env(data, gpu, {**(extra or {}), 'PHIBENCH_WORKDIR': workdir})
    for key in ['PHIBENCH_WORKDIR', *(extra or {}), *(['HIP_VISIBLE_DEVICES'] if gpu is not None else [])]:
        cmd += ['--env', key]  # Enroot passes only the variables named here, not its own environment
    cmd += [name, *command]
    try:
        return subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT if log else None).returncode
    finally:
        shutil.rmtree(env['ENROOT_RUNTIME_PATH'], ignore_errors=True)


def write_environment(rootfs: Path, pairs: list[tuple[str, str]]) -> None:
    path = rootfs / 'etc/environment'
    lines = [line for line in path.read_text().splitlines() if line.split('=', 1)[0] not in {k for k, _ in pairs}]
    path.write_text('\n'.join(lines + [f'{k}={v}' for k, v in pairs]) + '\n')


def copy_into(context: Path, rootfs: Path, arg: str, workdir: str) -> None:
    parts = shlex.split(arg)
    if any(p.startswith('--') for p in parts) or len(parts) < 2:
        raise BuildError(f'COPY {arg!r}: only plain COPY SRC... DST is supported')
    *sources, dest = parts
    target = rootfs / (dest if dest.startswith('/') else f'{workdir}/{dest}').lstrip('/')
    for source in sources:
        src = (context / source).resolve()
        if context.resolve() not in src.parents and src != context.resolve():
            raise BuildError(f'COPY {source}: outside the build context')
        if src.is_dir():
            shutil.copytree(src, target, symlinks=True, dirs_exist_ok=True)
        elif dest.endswith('/') or len(sources) > 1:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target / src.name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)


def cmd_build(task_dir: Path, base: Path, base_sha: str, data: Path) -> int:
    dockerfile = task_dir / 'environment/Dockerfile'
    steps = instructions(dockerfile.read_text())
    if not steps or steps[0] != ('FROM', BASE_IMAGE):
        raise BuildError(f'{dockerfile}: FROM must be {BASE_IMAGE}, the image {base} is pinned for')
    name, data = image_name(task_dir), data.resolve()
    rootfs = data / name
    if rootfs.exists():
        raise BuildError(f'{rootfs} exists; remove it to build again (enroot remove -f {name})')
    print(f'checking {base} ...', flush=True)
    if sha256(base) != base_sha:
        raise BuildError(f'{base}: SHA256 is not the pinned {base_sha}')
    data.mkdir(parents=True, exist_ok=True)
    subprocess.run(['enroot', 'create', '--name', name, str(base)], env=enroot_env(data), check=True)
    rc = data / f'{name}.rc'
    rc.write_text(RC)
    meta = {'image': name, 'task_dir': str(task_dir.resolve()), 'dockerfile_sha256': sha256(dockerfile),
            'base': str(base), 'base_sha256': base_sha, 'workdir': '/', 'entrypoint': [], 'cmd': [], 'env': []}
    env: list[tuple[str, str]] = []
    for word, arg in steps[1:]:
        print(f'--> {word} {arg[:100]}', flush=True)
        if word == 'COPY':
            copy_into(task_dir, rootfs, arg, meta['workdir'])
        elif word == 'RUN':
            code = start(name, data, ['/usr/bin/env', *[f'{k}={v}' for k, v in env], '/bin/sh', '-c', arg], rc,
                         meta['workdir'])
            if code:
                raise BuildError(f'RUN failed with exit {code}: {arg[:200]}')
        elif word == 'ENV':
            pairs = env_pairs(arg)
            env = [(k, v) for k, v in env if k not in dict(pairs)] + pairs
            write_environment(rootfs, pairs)
        elif word == 'WORKDIR':
            meta['workdir'] = arg if arg.startswith('/') else f'{meta["workdir"].rstrip("/")}/{arg}'
            (rootfs / meta['workdir'].lstrip('/')).mkdir(parents=True, exist_ok=True)
        elif word in ('ENTRYPOINT', 'CMD'):
            meta[word.lower()] = json.loads(arg)
        else:
            raise BuildError(f'{word}: not supported by this builder')
    meta['env'] = env
    (data / f'{name}.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(f'built {name} in {rootfs}')
    return 0


def busy_gpus(smi: str | None = None) -> list[str]:
    """rocm-smi indices of GPUs that run something or hold memory (over 20 MiB): another job on the node."""
    if smi is None:
        smi = subprocess.run(['rocm-smi', '--showuse', '--showmeminfo', 'vram'], capture_output=True, text=True,
                             check=True, timeout=60).stdout
    use = dict(re.findall(r'GPU\[(\d+)\]\s*: GPU use \(%\): (\d+)', smi))
    used = dict(re.findall(r'GPU\[(\d+)\]\s*: VRAM Total Used Memory \(B\): (\d+)', smi))
    if not use or use.keys() != used.keys():
        raise BuildError('rocm-smi printed no per-GPU use and memory; cannot tell whether the node is idle')
    return sorted((g for g in use if int(use[g]) or int(used[g]) > 20 * 1024 * 1024), key=int)


def listing(rootfs: Path) -> dict:
    """path -> (type, size, mode, mtime_ns) of every entry, without following links; for the unchanged-image check."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(rootfs):
        for entry in dirnames + filenames:
            path = os.path.join(dirpath, entry)
            st = os.lstat(path)
            out[path[len(str(rootfs)):]] = (st.st_mode, st.st_size, st.st_mtime_ns)
    return out


def remove_tree(path: Path) -> None:
    """Directories only: a copy's files are hard links into the built image, and a chmod would change it too."""
    subprocess.run(['find', str(path), '-type', 'd', '!', '-perm', '-u=rwx', '-exec', 'chmod', 'u+rwx', '{}', '+'],
                   check=False, stderr=subprocess.DEVNULL)
    shutil.rmtree(path)


def fresh_copy(data: Path, name: str, gpu: str) -> Path:
    rootfs = data / f'{name}-run-gpu{gpu}'  # one run per GPU at a time
    if rootfs.exists():
        raise BuildError(f'{rootfs} exists: an earlier run did not finish; check it, then remove it')
    subprocess.run(['cp', '-al', str(data / name), str(rootfs)], check=True)
    for rel in WRITABLE:  # real copies where a run writes, so nothing writes through a hard link into the image
        if os.path.lexists(rootfs / rel):
            subprocess.run(['rm', '-rf', str(rootfs / rel)], check=True)
            subprocess.run(['cp', '-a', str(data / name / rel), str(rootfs / rel)], check=True)
    return rootfs


def check_unchanged(data: Path, name: str, before: dict) -> None:
    after = listing(data / name)
    if after != before:
        changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        raise BuildError(f'the built image changed during the run (e.g. {changed[:5]}); it is no longer pristine')


def run_once(task_dir: Path, data: Path, meta: dict, mode: str, gpu: str, out: Path, before: dict,
             candidate: Path | None = None) -> dict:
    name = meta['image']
    rootfs = fresh_copy(data, name, gpu)
    clone = rootfs.name
    if candidate is not None:
        for editable in candidate.iterdir():
            target = rootfs / 'app/repo' / editable.name
            if not target.is_file():
                raise BuildError(f'--candidate {editable.name}: not a file of /app/repo')
            target.unlink()  # a hard link to the built image: replace it, never write through it
            shutil.copy(editable, target)
    extra = {} if mode == 'solve' else {'KERNELBENCH_VERIFY_MODE': mode}
    mounts = [(str((task_dir / 'tests').resolve()), '/tests')]
    verify = ['bash', '/tests/test.sh']
    if mode == 'solve':
        mounts.append((str(task_dir.resolve()), '/task'))
        verify = ['bash', '-lc', 'bash /task/solution/solve.sh && bash /tests/test.sh']
    if mode in PATCH_VARS:
        var, patch = PATCH_VARS[mode]
        extra[var] = f'/patches/{patch}'
        mounts.append((str(task_dir.resolve()), '/patches'))
    out.mkdir(parents=True)
    try:
        with (out / 'console.log').open('w') as log:
            code = start(clone, data, [*meta['entrypoint'], *verify], data / f'{name}.rc',
                         meta['workdir'], mounts, gpu, extra, log)
        verifier = rootfs / 'logs/verifier'
        if verifier.is_dir():
            shutil.copytree(verifier, out / 'verifier')
    finally:
        remove_tree(rootfs)
    check_unchanged(data, name, before)
    reward_path = out / 'verifier/reward.json'
    reward = json.loads(reward_path.read_text()) if reward_path.exists() else None
    return {'exit': code, 'reward': reward}


def cmd_exec(task_dir: Path, data: Path, gpu: str, mounts: list[str], command: list[str]) -> int:
    data = data.resolve()
    meta = json.loads((data / f'{image_name(task_dir)}.json').read_text())
    before = listing(data / meta['image'])
    rootfs = fresh_copy(data, meta['image'], gpu)
    rw = [(str(Path(m).resolve()), str(Path(m).resolve())) for m in mounts]
    try:
        code = start(rootfs.name, data, [*meta['entrypoint'], *command], data / f'{meta["image"]}.rc',
                     meta['workdir'], gpu=gpu, rw_mounts=rw)
    finally:
        remove_tree(rootfs)
    check_unchanged(data, meta['image'], before)
    return code


def cmd_score(task_dir: Path, data: Path, gpu: str, mode: str, repeat: int, out: Path,
              candidate: Path | None = None) -> int:
    if (candidate is None) != (mode != 'candidate'):
        raise BuildError('--candidate DIR goes with --mode candidate, and only with it')
    data = data.resolve()
    meta = json.loads((data / f'{image_name(task_dir)}.json').read_text())
    before = listing(data / meta['image'])
    path = out / f'{mode}.json'
    runs = json.loads(path.read_text())['runs'] if path.exists() else []
    for _ in range(repeat):
        k = len(runs) + 1
        if busy_gpus():
            raise BuildError(f'GPUs {busy_gpus()} are busy: timing needs the node to itself; not starting')
        result = run_once(task_dir, data, meta, mode, gpu, out / f'{mode}-{k}', before, candidate)
        if busy_gpus():
            (out / f'{mode}-{k}').rename(out / f'{mode}-{k}-busy')
            raise BuildError(f'another job started on the node during {mode}-{k}; renamed it {mode}-{k}-busy, '
                             'not counted')
        runs.append(result)
        r = result['reward'] or {}
        print(f"{mode}-{k}: exit {result['exit']} speedup {r.get('speedup')} reward {r.get('reward')} "
              f"hard fails {r.get('hard_fail_reasons')}", flush=True)
        speedups = [x['reward']['speedup'] for x in runs if x['reward']]
        path.write_text(json.dumps({'mode': mode, 'gpu': gpu, 'image': meta['image'], 'runs': runs,
                                    'median_speedup': statistics.median(speedups) if speedups else None},
                                   indent=2) + '\n')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('build', 'score', 'exec'))
    parser.add_argument('--task-dir', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True, help='Enroot data directory for the built images')
    parser.add_argument('--base', type=Path, help='build: the pinned base image .sqsh')
    parser.add_argument('--base-sha256', help='build: its pinned SHA256')
    parser.add_argument('--gpu', help='score: the HIP device to use')
    parser.add_argument('--mode', choices=MODES, default='oracle')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--candidate', type=Path, help='score, candidate mode: directory of editable files')
    parser.add_argument('--mount', action='append', default=[], help='exec: host directory, mounted read-write')
    argv = sys.argv[1:] if argv is None else list(argv)
    command = argv[argv.index('--') + 1:] if '--' in argv else []
    args = parser.parse_args(argv[:argv.index('--')] if '--' in argv else argv)
    try:
        if args.command == 'build':
            if not (args.base and args.base_sha256):
                parser.error('build needs --base and --base-sha256')
            return cmd_build(args.task_dir, args.base, args.base_sha256, args.data)
        if args.command == 'exec':
            if args.gpu is None or not command:
                parser.error('exec needs --gpu and -- CMD')
            return cmd_exec(args.task_dir, args.data, args.gpu, args.mount, command)
        if args.gpu is None or args.out is None:
            parser.error('score needs --gpu and --out')
        return cmd_score(args.task_dir, args.data, args.gpu, args.mode, args.repeat, args.out, args.candidate)
    except BuildError as exc:
        print(f'phibench_enroot: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
