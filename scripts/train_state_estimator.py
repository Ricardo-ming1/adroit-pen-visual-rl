from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.counterfactual.branch_rollout import load_frozen_actor
from src.state_distillation.dataset import EpisodeSequenceDataset, compute_train_target_stats
from src.state_distillation.oracle_observation import OracleObservationAdapter, geodesic_angle, rotation_6d_to_matrix
from src.state_distillation.temporal_state_estimator import TemporalStateEstimator
from src.state_distillation.training import train_estimator
from src.utils import load_yaml, resolve_device, save_json, seed_everything


@torch.inference_mode()
def diagnostics(model, dataset, adapter, oracle, oracle_stats, vision_stats, target_stats, batch_size=128):
    device=next(model.parameters()).device; loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    normalized=[]; geodesics=[]; angular=[]; action_errors=[]
    om=torch.as_tensor(oracle_stats.oracle_mean,device=device);os=torch.as_tensor(oracle_stats.oracle_std,device=device)
    pm=torch.as_tensor(vision_stats.proprio_mean,device=device);ps=torch.as_tensor(vision_stats.proprio_std,device=device)
    tm=torch.as_tensor(target_stats.mean,device=device);ts=torch.as_tensor(target_stats.std,device=device)
    model.eval()
    for batch in loader:
        batch={k:v.to(device) for k,v in batch.items()}; out=model(batch['rgb'],batch['proprio'],batch['previous_action'],batch['mask'])['primitive']
        true=batch['primitive']; normalized.append(((out[:,:9]-true[:,:9])/ts).square().cpu())
        obj=geodesic_angle(rotation_6d_to_matrix(out[:,9:15]),rotation_6d_to_matrix(true[:,9:15]))
        tar=geodesic_angle(rotation_6d_to_matrix(out[:,15:21]),rotation_6d_to_matrix(true[:,15:21]));geodesics.append(torch.stack((obj,tar),-1).cpu())
        angular.append((out[:,6:9]-true[:,6:9]).square().cpu())
        raw_proprio=batch['proprio'][:,-1]*ps+pm; raw=adapter(raw_proprio,out)
        action=oracle.act(actor=(raw-om)/os,deterministic=True); action_errors.append((action-batch['oracle_action']).cpu())
    nr=torch.cat(normalized);geo=torch.cat(geodesics);av=torch.cat(angular);ae=torch.cat(action_errors)
    per=torch.sqrt(ae.square().mean(-1))
    return {'hidden_state_normalized_rmse':float(torch.sqrt(nr.mean())),
            'object_orientation_geodesic_deg':float(torch.rad2deg(geo[:,0]).mean()),
            'target_orientation_geodesic_deg':float(torch.rad2deg(geo[:,1]).mean()),
            'angular_velocity_rmse':float(torch.sqrt(av.mean())),
            'oracle_action_rmse':float(torch.sqrt(ae.square().mean())),
            'oracle_action_rmse_p95':float(torch.quantile(per,.95)),'samples':len(dataset)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/temporal_state_distillation.yaml')
    p.add_argument('--stage',choices=['e1','e2','d1','d2'],required=True);p.add_argument('--data',action='append',required=True)
    p.add_argument('--initialize');p.add_argument('--output');args=p.parse_args();c=load_yaml(args.config)
    stage_seed=int(c['seed'])+{'e1':0,'e2':1,'d1':2,'d2':3}[args.stage];seed_everything(stage_seed,True);device=resolve_device(c['device'])
    vision,vision_stats,_=load_frozen_actor(c['vision_checkpoint'],device);oracle,oracle_stats,_=load_frozen_actor(c['oracle_checkpoint'],device)
    target_stats=compute_train_target_stats(args.data);train=EpisodeSequenceDataset(args.data,'train',target_stats,np.asarray(vision_stats.proprio_mean),np.asarray(vision_stats.proprio_std))
    dev=EpisodeSequenceDataset(args.data,'dev',target_stats,np.asarray(vision_stats.proprio_mean),np.asarray(vision_stats.proprio_std))
    model=TemporalStateEstimator(vision.visual_encoder).to(device)
    if args.initialize:
        init=torch.load(args.initialize,map_location=device,weights_only=False);model.load_state_dict(init['model_state'])
    if args.stage=='e1': model.freeze_cnn()
    else: model.unfreeze_last_block()
    run=Path(c['run_dir']);checkpoint=Path(args.output or run/'checkpoints'/f'{args.stage}.pt')
    epochs=int(c['training']['e1_epochs'] if args.stage=='e1' else c['training']['e2_epochs'] if args.stage=='e2' else c['training']['dagger_epochs'])
    adapter=OracleObservationAdapter().to(device)
    result=train_estimator(model,train,dev,adapter,oracle,oracle_stats,vision_stats,target_stats,epochs=epochs,
        batch_size=int(c['training']['batch_size']),learning_rate=float(c['training']['learning_rate']),
        cnn_learning_rate=float(c['training']['cnn_learning_rate']),action_coef=float(c['training']['action_loss_coefficient']),
        checkpoint=checkpoint,curve_path=run/'curves'/f'{args.stage}.json')
    result['offline_diagnostics']=diagnostics(model,dev,adapter,oracle,oracle_stats,vision_stats,target_stats,int(c['training']['batch_size']))
    result.update(stage=args.stage,checkpoint=str(checkpoint),train_samples=len(train),dev_samples=len(dev),source_files=args.data)
    save_json(Path(c['results_dir'])/f'{args.stage}_training.json',result);print(json.dumps({k:v for k,v in result.items() if k!='curve'},indent=2))


if __name__=='__main__':main()
