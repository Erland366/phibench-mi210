"""A minimal Φ-Bench task package for the converter tests: task.toml, tests/ref_speedup.txt, oracle.patch."""
from pathlib import Path


def make_task(root: Path, task_id: str, ref: float, device: str, gpus: int = 0, bench: str = 'lh') -> Path:
    d = root / 'tasks' / bench / task_id
    (d / 'tests').mkdir(parents=True, exist_ok=True)
    (d / 'task.toml').write_text(
        f'[task]\nid = "{task_id}"\nbench = "{bench}"\nkind = "performance"\n\n'
        f'[environment]\ngpus = {gpus}\n\n'
        f'[primary_metric]\nname = "speedup"\nref_speedup = {ref}\nref_speedup_source = "tests/ref_speedup.txt"\n'
        f'calibrated_on = "{device}"\n')
    (d / 'tests' / 'ref_speedup.txt').write_text(f'{ref}\n')
    (d / 'oracle.patch').write_text('+ reference\n')
    return d
