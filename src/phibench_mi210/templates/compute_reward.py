#!/usr/bin/env python3
"""Reward writer: same curve and output contract as the Phi-Bench performance tasks (SCORING.md)."""
import json
import math
import os


def number(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


LOG = os.environ.get('WRO_LOG', '/logs/verifier')
os.makedirs(LOG, exist_ok=True)
mode = os.environ.get('WRO_MODE', 'candidate')
hard = os.environ.get('WRO_HARD', '').split()
corr_frac = number('WRO_CORR_FRAC', 0)
base_ms = number('WRO_BASE_MS', -1)
cand_ms = number('WRO_CAND_MS', -1)
speedup = number('WRO_SPEEDUP', 0)
ref_speedup = number('WRO_REF', 1)
gates = {'scope_ok': os.environ.get('WRO_SCOPE') == 'true',
         'hardware_ok': os.environ.get('WRO_HARDWARE') == 'true',
         'benchmark_ok': os.environ.get('WRO_BENCH') == 'true',
         'source_scan_ok': os.environ.get('WRO_BAN') == 'true',
         'correctness_ok': corr_frac >= 1.0}

if not hard and corr_frac < 1.0:
    hard.append('correctness_failed')
if not hard and speedup <= 1.0:
    hard.append('no_speedup')
if not hard and ref_speedup <= 1.0:
    hard.append('invalid_ref_speedup')
if hard or speedup <= ref_speedup:
    score = 0.0
else:
    score = min(1.0, math.log(speedup / ref_speedup) / math.log(ref_speedup))

reward = {'schema_version': 'kernelbench_reward_v3_oracle_relative', 'task_type': 'performance',
          'reward': round(score, 6), 'hard_fail_reasons': hard, 'speedup': round(speedup, 6),
          'ref_speedup': ref_speedup, 'baseline_ms': base_ms, 'candidate_ms': cand_ms, 'score': round(score, 6),
          'mode': mode, 'correctness_frac': corr_frac, 'hard_fails': hard, 'gates': gates}
for name, payload in (('verifier_state.json', {'mode': mode, 'hard_fails': hard, 'gates': gates}),
                      ('correctness_results.json', {'correctness_frac': corr_frac, 'passed': corr_frac >= 1.0}),
                      ('benchmark_results.json', {'baseline_ms': base_ms, 'candidate_ms': cand_ms,
                                                  'speedup': round(speedup, 6), 'ref_speedup': ref_speedup}),
                      ('reward.json', reward)):
    with open(os.path.join(LOG, name), 'w') as fh:
        json.dump(payload, fh)
with open(os.path.join(LOG, 'reward.txt'), 'w') as fh:
    fh.write(str(round(score, 6)))
print(json.dumps(reward))  # the line loop16's score_engine.sh parses
