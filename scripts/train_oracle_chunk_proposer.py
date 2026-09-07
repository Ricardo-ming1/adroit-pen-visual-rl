from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.counterfactual.oracle_chunk import VisualOracleChunkProposer
from src.counterfactual.ranker_dataset import load_candidate_data
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def temporal(feature):
    return np.concatenate([feature["frozen_visual_latent_history"],feature["proprio_history"],
                           feature["previous_base_actions"],feature["previous_executed_actions"]],-1).astype(np.float32)


def main() -> None:
    config=load_yaml('configs/candidate_ranker.yaml'); seed_everything(int(config['seed']),True)
    device=resolve_device(config['device']); run_dir=Path(config['run_dir']); results_dir=Path(config['results_dir'])
    features,_,_=load_candidate_data(run_dir/'expanded_candidate_records.pt')
    chunks=torch.load(run_dir/'oracle_chunk_records.pt',map_location='cpu',weights_only=False)['records']
    feature_by_id={x['root_id']:x for x in features}; chunk_by_id={x['root_id']:x for x in chunks}
    train_features=[x for x in features if x['split']=='train']; all_train=np.concatenate([temporal(x) for x in train_features],0)
    mean=all_train.mean(0); std=np.maximum(all_train.std(0),1e-5)
    def arrays(split):
        xs=[]; ys=[]; positive=[]
        for feature in features:
            if feature['split']!=split: continue
            row=chunk_by_id[feature['root_id']]; safe=bool(row['labels']['strong_gain'] and not row['labels']['harm'])
            xs.append((temporal(feature)-mean)/std); ys.append(row['oracle_z_chunk'] if safe else np.zeros((5,24),np.float32)); positive.append(safe)
        return (torch.as_tensor(np.stack(xs),dtype=torch.float32),torch.as_tensor(np.stack(ys),dtype=torch.float32),torch.as_tensor(positive,dtype=torch.bool))
    train_x,train_y,train_pos=arrays('train'); dev_x,dev_y,dev_pos=arrays('dev')
    model=VisualOracleChunkProposer(temporal_dim=train_x.shape[-1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
    rng=np.random.default_rng(int(config['seed'])+700); curve=[]; best=None; best_loss=float('inf')
    positive_weight=float((~train_pos).sum()/max(int(train_pos.sum()),1))
    for epoch in range(1,161):
        order=rng.permutation(len(train_x)); losses=[]; model.train()
        for start in range(0,len(order),128):
            idx=torch.as_tensor(order[start:start+128]); x=train_x[idx].to(device); y=train_y[idx].to(device); p=train_pos[idx].to(device)
            prediction=model(x); per=F.smooth_l1_loss(prediction,y,reduction='none').mean((1,2))
            weight=torch.where(p,torch.full_like(per,positive_weight),torch.ones_like(per)); loss=(per*weight).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10); optimizer.step(); losses.append(float(loss.detach()))
        if epoch==1 or epoch%5==0 or epoch==160:
            model.eval()
            with torch.no_grad(): pred=model(dev_x.to(device)).cpu()
            pos_rmse=float(torch.sqrt(torch.mean((pred[dev_pos]-dev_y[dev_pos])**2))) if dev_pos.any() else float('nan')
            zero_rms=float(torch.sqrt(torch.mean(pred[~dev_pos]**2))) if (~dev_pos).any() else 0.0
            score=(0 if np.isnan(pos_rmse) else pos_rmse)+zero_rms
            curve.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'dev_positive_rmse':pos_rmse,'dev_zero_rms':zero_rms})
            if score<best_loss: best_loss=score; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best)
    checkpoint={'model_state':best,'model_init':{'temporal_dim':int(train_x.shape[-1]),'hidden_size':128,'horizon':5,'action_dim':24},
                'temporal_mean':mean.tolist(),'temporal_std':std.tolist(),'teacher':'branch-filtered Oracle takeover 5',
                'policy_inputs':'frozen visual latent, proprio, base/executed action history only'}
    torch.save(checkpoint,run_dir/'best_oracle_chunk_proposer.pt')
    summary={'train_roots':len(train_x),'dev_roots':len(dev_x),'train_positive_roots':int(train_pos.sum()),
             'dev_positive_roots':int(dev_pos.sum()),'best_dev_positive_rmse':min(x['dev_positive_rmse'] for x in curve if not np.isnan(x['dev_positive_rmse'])),
             'best_dev_zero_rms':min(x['dev_zero_rms'] for x in curve),'checkpoint':str(run_dir/'best_oracle_chunk_proposer.pt')}
    save_json(results_dir/'oracle_chunk_proposer_summary.json',summary)
    with (results_dir/'oracle_chunk_proposer_curve.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(curve[0]),lineterminator='\n');writer.writeheader();writer.writerows(curve)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
