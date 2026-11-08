"""CPU-only tests of the Enroot Dockerfile reader and the table-search candidate order."""
import random

import pytest

from evaluation.geak import phibench_enroot as pe
from evaluation.geak import phibench_table_search as ts


def test_continued_lines_join_as_docker_joins_them():
    steps = pe.instructions('# syntax=x\nFROM a:b\n\nRUN one \\\n && two\nENV A=1 \\\n    B=2\n')
    assert steps == [('FROM', 'a:b'), ('RUN', 'one  && two'), ('ENV', 'A=1     B=2')]
    assert pe.env_pairs(steps[2][1]) == [('A', '1'), ('B', '2')]


def test_a_comment_inside_a_continued_instruction_is_dropped():
    assert pe.instructions('RUN a \\\n# note\n && b\n') == [('RUN', 'a  && b')]


def test_a_dockerfile_ending_mid_instruction_is_refused():
    with pytest.raises(pe.BuildError):
        pe.instructions('RUN a \\\n')


def test_env_without_equals_is_refused():
    with pytest.raises(pe.BuildError):
        pe.env_pairs('A 1')


TABLES = {'served': {64: {'M': 1, 'g': {'N': 8}}, 256: {'M': 2, 'g': {'N': 16}}, 1024: {'M': 4, 'g': {'N': 32}}},
          'dec': {32: {'M': 8, 'g': {'N': 64}}}}


def test_copies_come_first_nearest_key_first_with_the_arms_interleaved():
    out = ts.candidates(TABLES, 256, ['served', 'dec'], random.Random(0), 6)
    assert [(c['arm'], c['how']) for c in out[:3]] == [('served', 'copy 64'), ('dec', 'copy 32'),
                                                        ('served', 'copy 1024')]
    assert [c['how'].split()[0] for c in out[3:]] == ['random'] * 3


def test_random_entries_draw_each_field_from_that_table_only():
    out = ts.candidates(TABLES, 256, ['dec'], random.Random(1), 5)
    assert all(c['entry'] == {'M': 8, 'g': {'N': 64}} for c in out)


SMI = ('GPU[0]          : GPU use (%): 0\nGPU[1]          : GPU use (%): 100\nGPU[2]          : GPU use (%): 0\n'
       'GPU[0]          : VRAM Total Used Memory (B): 13090816\nGPU[1]          : VRAM Total Used Memory (B): 5971019366\n'
       'GPU[2]          : VRAM Total Used Memory (B): 59710193664\n')


def test_a_gpu_in_use_or_holding_memory_is_busy():
    assert pe.busy_gpus(SMI) == ['1', '2']


def test_unreadable_rocm_smi_output_is_refused():
    with pytest.raises(pe.BuildError):
        pe.busy_gpus('no gpus here')
