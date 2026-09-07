from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.confirm_e2 import resumable_policy_eval
from scripts.evaluate_state_distilled_policy import load_estimator
from src.counterfactual.branch_rollout import load_frozen_actor
from src.state_distillation.evaluation import DistilledOraclePolicy
from src.utils import load_yaml, resolve_device, save_json


def parse_checkpoint(value):
    seed,path=value.split(':',1);return int(seed),path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/v6_1_confirmation.yaml');p.add_argument('--checkpoint',action='append',required=True,type=parse_checkpoint);args=p.parse_args()
    c=load_yaml(args.config);spec=c['reproduction'];device=resolve_device(c['device']);seeds=range(int(spec['evaluation_seed_start']),int(spec['evaluation_seed_start'])+int(spec['evaluation_episodes']))
    vision,vs,_=load_frozen_actor(c['vision_checkpoint'],device);oracle,os,_=load_frozen_actor(c['oracle_checkpoint'],device);run=Path(c['run_dir'])/'reproduction';results=Path(c['results_dir'])/'reproduction';results.mkdir(parents=True,exist_ok=True)
    baseline=resumable_policy_eval('Frozen Vision AWAC',run/'e0_eval.partial.json',seeds,10,actor=vision,stats=vs,line='vision')
    rows=[]
    for recipe_seed,path in args.checkpoint:
        estimator,_=load_estimator(path,vision,device);policy=DistilledOraclePolicy(estimator,oracle,os,vs,device)
        result=resumable_policy_eval(f'E2 recipe seed {recipe_seed}',run/f'seed_{recipe_seed}_eval.partial.json',seeds,10,actor=oracle,stats=os,line='oracle',distilled=policy)
        rows.append({'recipe_seed':recipe_seed,'checkpoint':path,'benchmark_success':result['benchmark_success'],'strict_stable_success':result['strict_stable_success'],
            'goal_entry_rate':result['goal_entry_rate'],'goal_exit_after_entry_rate':result['goal_exit_after_entry_rate'],'twenty_step_hold_rate':result['twenty_step_hold_rate']})
    benchmark=np.asarray([r['benchmark_success'] for r in rows]);strict=np.asarray([r['strict_stable_success'] for r in rows])
    bdelta=float(benchmark.mean()-baseline['benchmark_success']);sdelta=float(strict.mean()-baseline['strict_stable_success'])
    passed=bool((bdelta>=0.05 and sdelta>=0.0) or (sdelta>=0.05 and bdelta>=0.0))
    output={'evaluation_seed_start':int(spec['evaluation_seed_start']),'episodes_per_seed':int(spec['evaluation_episodes']),'pooled_episode_count':len(rows)*int(spec['evaluation_episodes']),
        'e0':{k:v for k,v in baseline.items() if k!='episodes'},'seeds':rows,'benchmark_mean':float(benchmark.mean()),'benchmark_std':float(benchmark.std(ddof=1)),
        'strict_mean':float(strict.mean()),'strict_std':float(strict.std(ddof=1)),'benchmark_delta_vs_e0':bdelta,'strict_delta_vs_e0':sdelta,'three_seed_gate_pass':passed}
    save_json(results/'three_seed_reproduction.json',output);print(json.dumps(output,indent=2))


if __name__=='__main__':main()
