from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.evaluate_state_distilled_policy import load_estimator
from src.counterfactual.branch_rollout import load_frozen_actor
from src.state_distillation.dataset import StateTargetStats
from src.state_distillation.direct_action_head import (DirectActionSequenceDataset, ExactPhaseBalancedDataset,
    FrozenTemporalDirectAction, phase_balanced_loss, verify_d1_student_source)
from src.utils import load_yaml, save_json, seed_everything


@torch.no_grad()
def evaluate(model, loader, preservation):
    model.eval(); totals={"fine":0.0,"e2_rmse":0.0,"oracle_rmse":0.0};seen=0
    for batch in loader:
        batch={k:v.to(next(model.parameters()).device) for k,v in batch.items()}
        action=model(batch['rgb'],batch['proprio'],batch['previous_action'],batch['mask']);n=len(action);seen+=n
        totals['fine']+=float(phase_balanced_loss(action,batch,warmup=False,preservation_coefficient=preservation))*n
        totals['e2_rmse']+=float(torch.sqrt((action-batch['e2_action']).square().mean()))*n
        totals['oracle_rmse']+=float(torch.sqrt((action-batch['oracle_action']).square().mean()))*n
    return {k:v/max(seen,1) for k,v in totals.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/v6_1_direct_action.yaml');args=p.parse_args();c=load_yaml(args.config)
    seed_everything(int(c['seed']),True);device=torch.device(c['device']);provenance=verify_d1_student_source(c['d1_sequences'])
    vision,vs,_=load_frozen_actor(c['vision_checkpoint'],device)
    estimator,target_stats=load_estimator(c['e2_checkpoint'],vision,device)
    kwargs=dict(target_stats=target_stats,proprio_mean=np.asarray(vs.proprio_mean),proprio_std=np.asarray(vs.proprio_std),
        near_position_error=float(c['near_position_error']),near_orientation_similarity=float(c['near_orientation_similarity']))
    raw_train=DirectActionSequenceDataset([c['d1_sequences']],'train',**kwargs);raw_dev=DirectActionSequenceDataset([c['d1_sequences']],'dev',**kwargs)
    train=ExactPhaseBalancedDataset(raw_train);dev=ExactPhaseBalancedDataset(raw_dev)
    train_loader=DataLoader(train,batch_size=int(c['batch_size']),shuffle=True,num_workers=0,drop_last=True)
    dev_loader=DataLoader(dev,batch_size=int(c['batch_size']),shuffle=False,num_workers=0)
    model=FrozenTemporalDirectAction(estimator,c['hidden_dims']).to(device);optimizer=torch.optim.AdamW(model.action_head.parameters(),lr=float(c['learning_rate']),weight_decay=float(c['weight_decay']))
    curve=[];best=float('inf');best_state=None;updates=0
    stages=[('warmup',int(c['warmup_epochs']),True),('phase_balanced',int(c['finetune_epochs']),False)]
    for stage,epochs,warmup in stages:
        for epoch in range(1,epochs+1):
            model.train();running=seen=0
            for batch in train_loader:
                batch={k:v.to(device) for k,v in batch.items()};action=model(batch['rgb'],batch['proprio'],batch['previous_action'],batch['mask'])
                loss=phase_balanced_loss(action,batch,warmup=warmup,preservation_coefficient=float(c['e2_preservation_coefficient']))
                optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.action_head.parameters(),5.0);optimizer.step()
                running+=float(loss.detach())*len(action);seen+=len(action);updates+=1
            metrics=evaluate(model,dev_loader,float(c['e2_preservation_coefficient']));row={'stage':stage,'epoch':epoch,'train_loss':running/max(seen,1),**{f'dev_{k}':v for k,v in metrics.items()}};curve.append(row)
            if not warmup and metrics['fine']<best:best=metrics['fine'];best_state=copy.deepcopy(model.action_head.state_dict())
            print(json.dumps(row),flush=True)
    if best_state is None:raise RuntimeError('no phase-balanced checkpoint candidate')
    model.action_head.load_state_dict(best_state);run=Path(c['run_dir']);run.mkdir(parents=True,exist_ok=True);checkpoint=run/'phase_balanced_direct_action.pt'
    torch.save({'action_head_state':best_state,'hidden_dims':list(c['hidden_dims']),'e2_checkpoint':c['e2_checkpoint'],
        'frozen_encoder':True,'phase_input':False,'dev_phase_balanced_loss':best},checkpoint)
    result={'checkpoint':str(checkpoint),'updates':updates,'curve':curve,'best_dev_phase_balanced_loss':best,
        'train_rows_balanced':len(train),'dev_rows_balanced':len(dev),'pre_rows':len(raw_train.pre_indices),'post_rows':len(raw_train.post_indices),
        'd1_provenance':provenance,'encoder_mutation_count':0,'state_head_mutation_count':0,'oracle_mutation_count':0,'phase_is_model_input':False}
    save_json(Path(c['results_dir'])/'direct_action_training.json',result);print(json.dumps({k:v for k,v in result.items() if k!='curve'},indent=2))


if __name__=='__main__':main()
