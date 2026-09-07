from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.utils.data import ConcatDataset,Dataset

from scripts.train_state_estimator import diagnostics
from src.counterfactual.branch_rollout import load_frozen_actor
from src.state_distillation.dataset import EpisodeSequenceDataset,compute_train_target_stats
from src.state_distillation.oracle_observation import OracleObservationAdapter
from src.state_distillation.temporal_state_estimator import TemporalStateEstimator
from src.state_distillation.training import train_estimator
from src.utils import load_yaml,resolve_device,save_json,seed_everything


class ExactHalfMixture(Dataset):
    def __init__(self,old,recent):self.old=old;self.recent=recent;self.half=min(len(old),len(recent))
    def __len__(self):return 2*self.half
    def __getitem__(self,index):
        source=self.old if index%2==0 else self.recent;return source[(index//2)%len(source)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/temporal_state_distillation.yaml');p.add_argument('--round',choices=['d1','d2'],required=True)
    p.add_argument('--source',required=True);p.add_argument('--recent',required=True);p.add_argument('--initialize',required=True);args=p.parse_args();c=load_yaml(args.config)
    seed_everything(int(c['seed'])+(2 if args.round=='d1' else 3),True);device=resolve_device(c['device']);vision,vs,_=load_frozen_actor(c['vision_checkpoint'],device);oracle,os,_=load_frozen_actor(c['oracle_checkpoint'],device)
    paths=[args.source,args.recent];stats=compute_train_target_stats(paths);old=EpisodeSequenceDataset([args.source],'train',stats,np.asarray(vs.proprio_mean),np.asarray(vs.proprio_std));recent=EpisodeSequenceDataset([args.recent],'train',stats,np.asarray(vs.proprio_mean),np.asarray(vs.proprio_std));train=ExactHalfMixture(old,recent)
    dev=ConcatDataset([EpisodeSequenceDataset([x],'dev',stats,np.asarray(vs.proprio_mean),np.asarray(vs.proprio_std)) for x in paths])
    model=TemporalStateEstimator(vision.visual_encoder).to(device);model.load_state_dict(torch.load(args.initialize,map_location=device,weights_only=False)['model_state']);model.unfreeze_last_block();adapter=OracleObservationAdapter().to(device)
    checkpoint=Path(c['run_dir'])/'checkpoints'/f'{args.round}.pt';result=train_estimator(model,train,dev,adapter,oracle,os,vs,stats,
        epochs=int(c['training']['dagger_epochs']),batch_size=int(c['training']['batch_size']),learning_rate=float(c['training']['learning_rate']),cnn_learning_rate=float(c['training']['cnn_learning_rate']),action_coef=float(c['training']['action_loss_coefficient']),checkpoint=checkpoint,curve_path=Path(c['run_dir'])/'curves'/f'{args.round}.json')
    result['offline_diagnostics']=diagnostics(model,dev,adapter,oracle,os,vs,stats,int(c['training']['batch_size']));result.update(stage=args.round,checkpoint=str(checkpoint),historical_rows=len(old),recent_rows=len(recent),minibatch_recipe='50% latest DAgger / 50% historical sequence data')
    save_json(Path(c['results_dir'])/f'{args.round}_training.json',result);print(json.dumps({k:v for k,v in result.items() if k!='curve'},indent=2))


if __name__=='__main__':main()
