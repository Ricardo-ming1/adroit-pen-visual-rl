from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.train_state_estimator import diagnostics
from src.counterfactual.branch_rollout import load_frozen_actor
from src.state_distillation.dataset import EpisodeSequenceDataset, compute_train_target_stats
from src.state_distillation.oracle_observation import OracleObservationAdapter
from src.state_distillation.temporal_state_estimator import TemporalStateEstimator
from src.state_distillation.training import train_estimator
from src.state_distillation.v6_1 import sha256_file
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def datasets(path, stats, vision_stats):
    mean=np.asarray(vision_stats.proprio_mean);std=np.asarray(vision_stats.proprio_std)
    return (EpisodeSequenceDataset([path],'train',stats,mean,std),EpisodeSequenceDataset([path],'dev',stats,mean,std))


def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe-seed',type=int,required=True)
    p.add_argument('--confirmation-config',default='configs/v6_1_confirmation.yaml')
    p.add_argument('--v6-config',default='configs/temporal_state_distillation.yaml');args=p.parse_args()
    cc=load_yaml(args.confirmation_config);v6=load_yaml(args.v6_config);device=resolve_device(cc['device']);source=cc['reproduction']['source_sequences']
    run=Path(cc['run_dir'])/'reproduction'/f'seed_{args.recipe_seed}';result_dir=Path(cc['results_dir'])/'reproduction';run.mkdir(parents=True,exist_ok=True)
    vision,vs,_=load_frozen_actor(cc['vision_checkpoint'],device);oracle,os,_=load_frozen_actor(cc['oracle_checkpoint'],device);adapter=OracleObservationAdapter().to(device)
    stats=compute_train_target_stats([source]);train,dev=datasets(source,stats,vs)
    seed_everything(args.recipe_seed,True);e1=TemporalStateEstimator(vision.visual_encoder).to(device);e1.freeze_cnn();e1_path=run/'e1.pt'
    e1_result=train_estimator(e1,train,dev,adapter,oracle,os,vs,stats,epochs=int(v6['training']['e1_epochs']),batch_size=int(v6['training']['batch_size']),
        learning_rate=float(v6['training']['learning_rate']),cnn_learning_rate=float(v6['training']['cnn_learning_rate']),action_coef=float(v6['training']['action_loss_coefficient']),
        checkpoint=e1_path,curve_path=run/'e1_curve.json');e1_result['offline_diagnostics']=diagnostics(e1,dev,adapter,oracle,os,vs,stats,int(v6['training']['batch_size']))
    train.close();dev.close();train,dev=datasets(source,stats,vs)
    seed_everything(args.recipe_seed+1,True);e2=TemporalStateEstimator(vision.visual_encoder).to(device);e2.load_state_dict(e1.state_dict());e2.unfreeze_last_block();e2_path=run/'e2.pt'
    e2_result=train_estimator(e2,train,dev,adapter,oracle,os,vs,stats,epochs=int(v6['training']['e2_epochs']),batch_size=int(v6['training']['batch_size']),
        learning_rate=float(v6['training']['learning_rate']),cnn_learning_rate=float(v6['training']['cnn_learning_rate']),action_coef=float(v6['training']['action_loss_coefficient']),
        checkpoint=e2_path,curve_path=run/'e2_curve.json');e2_result['offline_diagnostics']=diagnostics(e2,dev,adapter,oracle,os,vs,stats,int(v6['training']['batch_size']))
    train.close();dev.close();summary={'recipe_seed':args.recipe_seed,'source_sequences':source,'source_scale_unchanged':True,
        'e1':e1_result,'e2':e2_result,'e2_checkpoint':str(e2_path),'e2_checkpoint_sha256':sha256_file(e2_path)}
    save_json(result_dir/f'seed_{args.recipe_seed}_training.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in {'e1','e2'}},indent=2))


if __name__=='__main__':main()
