from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_fixed_entry_hold import evaluate_root, summarize
from scripts.evaluate_state_distilled_policy import load_estimator
from src.counterfactual.branch_rollout import load_frozen_actor
from src.envs.adroit import VisualAdroitEnv
from src.state_distillation.evaluation import DistilledOraclePolicy
from src.utils import load_yaml, resolve_device, save_json


def parse_checkpoint(value):
    seed,path=value.split(':',1);return int(seed),path


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/v6_1_confirmation.yaml');p.add_argument('--checkpoint',action='append',required=True,type=parse_checkpoint);args=p.parse_args()
    c=load_yaml(args.config);device=resolve_device(c['device']);vision,vs,_=load_frozen_actor(c['vision_checkpoint'],device);oracle,os,_=load_frozen_actor(c['oracle_checkpoint'],device)
    roots=torch.load(Path(c['run_dir'])/'fixed_entry_roots.pt',map_location='cpu',weights_only=False)['roots'];results=[]
    for recipe_seed,path in args.checkpoint:
        estimator,_=load_estimator(path,vision,device);policy=DistilledOraclePolicy(estimator,oracle,os,vs,device);env=VisualAdroitEnv();rows=[]
        try:
            for root in roots:rows.append(evaluate_root(env,root,actor=oracle,line='oracle',stats=os,device=device,steps=c['fixed_entry_hold']['rollout_steps'],distilled=policy))
        finally:env.close()
        result=summarize(rows);results.append({'recipe_seed':recipe_seed,**{k:v for k,v in result.items() if k!='rows'}});print(json.dumps(results[-1]),flush=True)
    output={'common_root_count':len(roots),'same_state_and_history':True,'seeds':results,
        'twenty_step_survival_mean':float(np.mean([x['twenty_step_survival'] for x in results])),
        'fifty_step_survival_mean':float(np.mean([x['fifty_step_survival'] for x in results]))}
    path=Path(c['results_dir'])/'reproduction'/'three_seed_fixed_hold.json';save_json(path,output);print(json.dumps(output,indent=2))


if __name__=='__main__':main()
