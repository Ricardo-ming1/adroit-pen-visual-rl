from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_state_distilled_policy import load_estimator
from src.counterfactual.branch_rollout import deterministic_action, load_frozen_actor
from src.envs.adroit import VisualAdroitEnv, current_metrics
from src.state_distillation.evaluation import DistilledOraclePolicy
from src.state_distillation.v6_1 import maximum_streak, restore_policy_history
from src.utils import load_yaml, resolve_device, save_json


def evaluate_root(env, root, *, actor, line, stats, device, steps, distilled=None):
    env.reset(seed=int(root["source_seed"])); observation = env.restore_training_state(root["training_state"])
    if distilled is not None: restore_policy_history(distilled, root["history_before_current"])
    official=[]; dropped=[]; finite=True
    for _ in range(int(steps)):
        action = (deterministic_action(actor, line, observation, stats, device) if distilled is None else distilled.act(observation)[0])
        finite = finite and bool(np.isfinite(action).all())
        observation, _, terminated, truncated, info = env.step(np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0))
        official.append(bool(info["official_goal"])); dropped.append(bool(info["dropped"]))
        if terminated or truncated: break
    exit_step = next((i + 1 for i, value in enumerate(official) if not value), None)
    return {"root_id": int(root["root_id"]), "source_seed": int(root["source_seed"]),
        "twenty_step_survival": bool(len(official)>=20 and all(official[:20])),
        "fifty_step_survival": bool(len(official)>=50 and all(official[:50])),
        "steps_to_exit": int(exit_step) if exit_step is not None else int(len(official)+1),
        "max_goal_streak": maximum_streak(official),
        "strict_continuation": bool(len(official)>=20 and all(official[:20]) and not any(dropped) and finite),
        "dropped": bool(any(dropped)), "finite": bool(finite)}


def summarize(rows):
    n=len(rows)
    return {"roots":n, "twenty_step_survival_count":sum(r["twenty_step_survival"] for r in rows),
        "twenty_step_survival":sum(r["twenty_step_survival"] for r in rows)/n,
        "fifty_step_survival_count":sum(r["fifty_step_survival"] for r in rows),
        "fifty_step_survival":sum(r["fifty_step_survival"] for r in rows)/n,
        "mean_steps_to_exit":float(np.mean([r["steps_to_exit"] for r in rows])),
        "mean_max_goal_streak":float(np.mean([r["max_goal_streak"] for r in rows])),
        "strict_continuation_count":sum(r["strict_continuation"] for r in rows),
        "strict_continuation":sum(r["strict_continuation"] for r in rows)/n,
        "drop_count":sum(r["dropped"] for r in rows), "nonfinite_count":sum(not r["finite"] for r in rows),
        "rows":rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/v6_1_confirmation.yaml');args=parser.parse_args()
    config=load_yaml(args.config);device=resolve_device(config['device']);spec=config['fixed_entry_hold']
    vision,vs,_=load_frozen_actor(config['vision_checkpoint'],device);oracle,os,_=load_frozen_actor(config['oracle_checkpoint'],device)
    estimator,_=load_estimator(config['e2_checkpoint'],vision,device);e2=DistilledOraclePolicy(estimator,oracle,os,vs,device)
    bank=torch.load(Path(config['run_dir'])/'fixed_entry_roots.pt',map_location='cpu',weights_only=False);roots=bank['roots']
    policies=(("e0",vision,"vision",vs,None),("e2",oracle,"oracle",os,e2),("oracle",oracle,"oracle",os,None));results={}
    for name,actor,line,stats,distilled in policies:
        env=VisualAdroitEnv();rows=[]
        try:
            for root in roots:
                rows.append(evaluate_root(env,root,actor=actor,line=line,stats=stats,device=device,steps=spec['rollout_steps'],distilled=distilled))
                if len(rows)%20==0:print(json.dumps({'policy':name,'roots':len(rows)}),flush=True)
        finally:env.close()
        results[name]=summarize(rows)
    gap20=results['oracle']['twenty_step_survival']-results['e2']['twenty_step_survival']
    gap50=results['oracle']['fifty_step_survival']-results['e2']['fifty_step_survival']
    output={"bank":str(Path(config['run_dir'])/'fixed_entry_roots.pt'),"common_root_count":len(roots),
        "same_state_and_history":True,"e2_gru_zeroed_at_root":False,"privileged_phase_input":False,
        "policies":results,"e2_vs_oracle_twenty_step_gap":gap20,"e2_vs_oracle_fifty_step_gap":gap50,
        "oracle_hold_gap_triggered":bool(gap20>=float(spec['oracle_gap_trigger']) or gap50>=float(spec['oracle_gap_trigger']))}
    save_json(Path(config['results_dir'])/'fixed_entry_hold200.json',output);print(json.dumps({**output,"policies":{k:{x:y for x,y in v.items() if x!='rows'} for k,v in results.items()}},indent=2))


if __name__=='__main__':main()
