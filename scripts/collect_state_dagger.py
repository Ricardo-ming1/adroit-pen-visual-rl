from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.evaluate_state_distilled_policy import load_estimator
from src.counterfactual.branch_rollout import deterministic_action, load_frozen_actor
from src.envs.adroit import HORIZON, VisualAdroitEnv, current_metrics, privileged_observation
from src.state_distillation.dataset import append_episode, episode_auxiliary_labels
from src.state_distillation.evaluation import DistilledOraclePolicy
from src.state_distillation.oracle_observation import OracleObservationAdapter
from src.utils import load_yaml, resolve_device


@torch.inference_mode()
def collect(env,policy,oracle,oracle_stats,device,seed,targeted):
    observation,_=env.reset(seed=seed);policy.reset();adapter=OracleObservationAdapter();fields={name:[] for name in
        ('rgb','proprio','previous_action','oracle_observation','primitive','oracle_action','executed_action','official')}
    first_entry=None
    for t in range(HORIZON):
        true=privileged_observation(env.env);teacher=deterministic_action(oracle,'oracle',observation,oracle_stats,device)
        action,_=policy.act(observation);metric=current_metrics(env.env)
        fields['rgb'].append(observation['rgb'][-1].transpose(1,2,0).copy());fields['proprio'].append(observation['proprio'].copy())
        fields['previous_action'].append(observation['previous_action'].copy());fields['oracle_observation'].append(true)
        fields['primitive'].append(adapter.primitive_from_runtime(env.env));fields['oracle_action'].append(teacher)
        fields['executed_action'].append(action);fields['official'].append(metric['official_goal'])
        if metric['official_goal'] and first_entry is None:first_entry=t
        observation,_,terminated,truncated,_=env.step(action)
        if terminated or truncated:break
    arrays={k:np.asarray(v) for k,v in fields.items() if k!='official'};phase,e10,e20=episode_auxiliary_labels(np.asarray(fields['official'],bool))
    arrays.update(phase=phase,exit_10=e10,exit_20=e20)
    train_mask=np.ones(len(phase),np.uint8)
    if targeted:
        start=max(0,(first_entry if first_entry is not None else max(0,len(phase)-60))-7);train_mask[:start]=0
    arrays['train_mask']=train_mask
    return arrays,first_entry


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/temporal_state_distillation.yaml');p.add_argument('--checkpoint',required=True)
    p.add_argument('--round',choices=['d1','d2'],required=True);p.add_argument('--output',required=True);args=p.parse_args();c=load_yaml(args.config);device=resolve_device(c['device'])
    vision,vs,_=load_frozen_actor(c['vision_checkpoint'],device);oracle,os,_=load_frozen_actor(c['oracle_checkpoint'],device);model,_=load_estimator(args.checkpoint,vision,device)
    policy=DistilledOraclePolicy(model,oracle,os,vs,device);output=Path(args.output);done=set()
    if output.exists():
        with h5py.File(output,'r') as h:done={int(x.attrs['seed']) for x in h['episodes'].values()};episode_id=len(h['episodes'])
    else:episode_id=0
    count=int(c['dagger']['episodes']);start=int(c['dagger'][f'{args.round}_seed_start']);normal=int(round(count*(1-float(c['dagger']['targeted_fraction']))))
    env=VisualAdroitEnv();entries=0
    try:
        for i,seed in enumerate(range(start,start+count)):
            if seed in done:continue
            targeted=i>=normal;arrays,entry=collect(env,policy,oracle,os,device,seed,targeted)
            append_episode(output,episode_id,seed=seed,source=f'{args.round}_student_visited_{"targeted" if targeted else "normal"}',arrays=arrays)
            episode_id+=1;entries+=entry is not None
            if episode_id%10==0:print(json.dumps({'episodes':episode_id,'goal_entries':entries}),flush=True)
    finally:env.close()
    with h5py.File(output,'r') as h:summary={'episodes':len(h['episodes']),'transitions':sum(len(x['rgb']) for x in h['episodes'].values()),
        'training_rows':sum(int(np.asarray(x['train_mask']).sum()) for x in h['episodes'].values()),'goal_entry_episodes':entries,
        'environment_executed_policy':'student','teacher_action_executed':False}
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
