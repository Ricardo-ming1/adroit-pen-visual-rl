from __future__ import annotations

import json
from pathlib import Path

from src.state_distillation.v6_1 import aggregate, paired_binary_statistics
from src.utils import save_json


BASELINE=Path('runs/frozen/vision/full_seed_101/evaluations/awac_test_normal.json')
CANDIDATE=Path('results/v6_1/frozen_e2_test200.json')


def historical_aggregate(rows):
    n=len(rows)
    return {'episode_count':n,'benchmark_success_count':sum(bool(x['benchmark_success']) for x in rows),
        'benchmark_success':sum(bool(x['benchmark_success']) for x in rows)/n,
        'strict_success_count':sum(bool(x['strict_stable_success']) for x in rows),
        'strict_stable_success':sum(bool(x['strict_stable_success']) for x in rows)/n,
        'drop_count':sum(bool(x['dropped']) for x in rows),'nonfinite_count':sum(not bool(x['finite']) for x in rows),
        'goal_exit_after_entry_rate':'NOT_RECORDED_IN_HISTORICAL_FROZEN_BASELINE'}


def main():
    baseline=json.loads(BASELINE.read_text());candidate=json.loads(CANDIDATE.read_text())
    expected=list(range(50000,50200))
    if [int(x['seed']) for x in baseline['episodes']] != expected or [int(x['seed']) for x in candidate['episodes']] != expected:
        raise RuntimeError('frozen test seed identity mismatch')
    b=historical_aggregate(baseline['episodes']);e=aggregate(candidate['episodes']);paired={}
    for key,label in (('benchmark_success','benchmark'),('strict_stable_success','strict')):
        paired[label]=paired_binary_statistics(candidate['episodes'],baseline['episodes'],key,bootstrap_replicates=100000,bootstrap_seed=61200+(label=='strict'))
    output={'bank_seed_start':50000,'episodes':200,'opened_for_v6_1_model_selection':False,
        'historical_e0_reused_without_rerun':True,'historical_e0_exit_metric_available':False,'e0':b,
        'e2':{k:v for k,v in e.items() if k!='episodes'},'paired_e2_vs_e0':paired,
        'checkpoint_selection_or_retraining_after_test':False}
    save_json('results/v6_1/frozen_test.json',output);print(json.dumps(output,indent=2))


if __name__=='__main__':main()
