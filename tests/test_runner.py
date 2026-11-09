"""CPU-only tests of the AMD passthrough added to Φ-Bench's scripts/run_task.py (src/phibench_mi210/templates/run_task.py)."""
import importlib.util
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / 'src' / 'phibench_mi210' / 'templates'


def load(path):
    spec = importlib.util.spec_from_file_location(f'runner_{path.parent.name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rt = load(TEMPLATES / 'run_task.py')


def args(**kw):
    base = dict(gpu_passthrough='auto', gpu_devices=None, dry_run=False, nvidia_lib_dir=rt.DEFAULT_NVIDIA_LIB_DIR,
                nvidia_bin_dir=rt.DEFAULT_NVIDIA_BIN_DIR)
    return SimpleNamespace(**{**base, **kw})


def host(monkeypatch, files=(), tools=()):
    monkeypatch.setattr(rt.os.path, 'exists', lambda p: p in files)
    monkeypatch.setattr(rt.shutil, 'which', lambda t: t if t in tools else None)
    monkeypatch.setattr(rt.glob, 'glob', lambda pattern: [])


def task(types):
    return SimpleNamespace(gpus=1, gpu_types=types)


def test_an_amd_task_on_an_amd_host_gets_kfd_and_dri(monkeypatch):
    host(monkeypatch, files={'/dev/kfd'})
    flags, rec = rt.gpu_plan(task(['MI210']), args())
    assert rec['mode'] == 'amd'
    assert flags == ['--device', '/dev/kfd', '--device', '/dev/dri', '--security-opt', 'seccomp=unconfined',
                     '--group-add', 'video']


def test_gpu_devices_choose_amd_gpus_through_hip_visible_devices(monkeypatch):
    host(monkeypatch, files={'/dev/kfd'})
    a = args(gpu_devices='3,4')
    flags, rec = rt.gpu_plan(task(['MI210']), a)
    assert rt.gpu_env_flags(None, task(['MI210']), a, rec, 'img') == ['-e', 'HIP_VISIBLE_DEVICES=3,4']


def test_an_nvidia_task_on_an_amd_host_keeps_the_upstream_refusal(monkeypatch):
    host(monkeypatch, files={'/dev/kfd'})
    flags, rec = rt.gpu_plan(task(['H20']), args())
    assert flags == [] and rec['mode'] == 'none'


def test_an_nvidia_host_is_unchanged(monkeypatch):
    host(monkeypatch, files={'/dev/nvidiactl'}, tools={'nvidia-container-runtime'})
    flags, rec = rt.gpu_plan(task(['H20']), args())
    assert flags == ['--gpus', 'all'] and rec['mode'] == 'toolkit'


def test_an_amd_task_on_a_host_without_kfd_gets_no_gpu_and_says_so(monkeypatch):
    host(monkeypatch)
    flags, rec = rt.gpu_plan(task(['MI210']), args())
    assert flags == [] and rec['mode'] == 'none' and 'no /dev/kfd' in rec['notes'][0]


@pytest.mark.parametrize('name,amd', [('MI210', True), ('MI300X', True), ('AMD Instinct MI210', True),
                                      ('H20', False), ('A100', False), ('MIG-7g', False)])
def test_amd_gpu_names(name, amd):
    assert rt.amd_gpu_type(name) is amd


def test_the_patch_is_the_difference_from_the_vendored_upstream_runner():
    diff = subprocess.run(['diff', '-u', '--label', 'a/scripts/run_task.py', '--label', 'b/scripts/run_task.py',
                           TEMPLATES / 'vendored' / 'run_task.py', TEMPLATES / 'run_task.py'],
                          capture_output=True, text=True).stdout
    assert diff == (TEMPLATES / 'run_task-amd.patch').read_text()
