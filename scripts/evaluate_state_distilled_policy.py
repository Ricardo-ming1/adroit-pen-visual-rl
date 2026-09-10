from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.counterfactual.branch_rollout import load_frozen_actor
from src.evaluation import evaluate_policy
from src.state_distillation.dataset import StateTargetStats
from src.state_distillation.evaluation import DistilledOraclePolicy, evaluate_distilled_policy, paired_summary
from src.state_distillation.temporal_state_estimator import TemporalStateEstimator
from src.utils import load_yaml, resolve_device, save_json


def load_estimator(path, vision_actor, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TemporalStateEstimator(vision_actor.visual_encoder).to(device)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    return model, StateTargetStats.from_dict(checkpoint["target_stats"])


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/temporal_state_distillation.yaml")
    p.add_argument("--checkpoint"); p.add_argument("--output",required=True); p.add_argument("--episodes",type=int,default=100)
    p.add_argument("--seed-start",type=int,default=30000); p.add_argument("--baseline-output")
    args=p.parse_args(); c=load_yaml(args.config); device=resolve_device(c["device"])
    vision, vision_stats, _=load_frozen_actor(c["vision_checkpoint"],device)
    oracle, oracle_stats, _=load_frozen_actor(c["oracle_checkpoint"],device); seeds=range(args.seed_start,args.seed_start+args.episodes)
    if args.checkpoint:
        model,_=load_estimator(args.checkpoint,vision,device)
        policy=DistilledOraclePolicy(model,oracle,oracle_stats,vision_stats,device); result=evaluate_distilled_policy(policy,seeds)
        if args.baseline_output:
            baseline=json.loads(Path(args.baseline_output).read_text()); result.update(paired_summary(result,baseline))
    else:
        result=evaluate_policy(vision,"vision",vision_stats,seeds,device)
    save_json(args.output,result); print(json.dumps({k:v for k,v in result.items() if k!="episodes"},indent=2))


if __name__=="__main__": main()
