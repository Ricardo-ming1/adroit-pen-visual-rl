from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.confirm_e2 import resumable_policy_eval
from scripts.evaluate_state_distilled_policy import load_estimator
from src.counterfactual.branch_rollout import load_frozen_actor
from src.state_distillation.direct_action_head import DirectActionPolicy, FrozenTemporalDirectAction
from src.utils import load_yaml, resolve_device, save_json


class DirectAdapter:
    def __init__(self, policy): self.policy=policy
    def reset(self): self.policy.reset()
    def act(self, observation): return self.policy.act(observation), None


def main():
    p=argparse.ArgumentParser();p.add_argument('--confirmation-config',default='configs/v6_1_confirmation.yaml')
    p.add_argument('--direct-config',default='configs/v6_1_direct_action.yaml');p.add_argument('--checkpoint',required=True)
    p.add_argument('--seed-start',type=int,required=True);p.add_argument('--episodes',type=int,required=True);p.add_argument('--output',required=True);args=p.parse_args()
    cc=load_yaml(args.confirmation_config);dc=load_yaml(args.direct_config);device=resolve_device(cc['device'])
    vision,vs,_=load_frozen_actor(cc['vision_checkpoint'],device);oracle,os,_=load_frozen_actor(cc['oracle_checkpoint'],device)
    estimator,_=load_estimator(dc['e2_checkpoint'],vision,device);state=torch.load(args.checkpoint,map_location=device,weights_only=False)
    model=FrozenTemporalDirectAction(estimator,state['hidden_dims']).to(device);model.action_head.load_state_dict(state['action_head_state']);model.eval()
    policy=DirectAdapter(DirectActionPolicy(model,vs,device));seeds=range(args.seed_start,args.seed_start+args.episodes)
    result=resumable_policy_eval('Phase-balanced Direct Action',Path(cc['run_dir'])/(Path(args.output).stem+'.partial.json'),seeds,10,
        actor=oracle,stats=os,line='oracle',distilled=policy);save_json(args.output,result);print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))


if __name__=='__main__':main()
