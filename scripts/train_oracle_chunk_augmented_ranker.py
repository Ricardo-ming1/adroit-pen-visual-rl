from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.counterfactual.candidate_ranker import PrivilegedCandidateRanker, VisualCandidateRanker
from src.counterfactual.ranker_dataset import CandidateDataset, assert_source_split_isolated, training_stats
from src.counterfactual.ranker_training import attach_outcome_metrics, predict_ranker, threshold_search, train_ranker
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def train_line(line, model, train, dev, records, config, device, run_dir, results_dir):
    state,curve=train_ranker(model,train,dev,line=line,config=config['ranker'],device=device);model.load_state_dict(state)
    predictions=predict_ranker(model,DataLoader(dev,batch_size=int(config['ranker']['batch_size'])),line,device)
    summary=attach_outcome_metrics(threshold_search(predictions,dev,config['ranker']),records,'dev')
    torch.save({'line':line,'model_state':state,'selector_threshold':summary['threshold'],
                'branch_dev_summary':summary,'descriptor_dim':120},
               run_dir/f'best_oracle_chunk_augmented_{line}_ranker.pt')
    save_json(results_dir/f'oracle_chunk_augmented_{line}_summary.json',summary)
    flat=[{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in curve]
    with (results_dir/f'oracle_chunk_augmented_{line}_curve.csv').open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=list(flat[0]),lineterminator='\n');w.writeheader();w.writerows(flat)
    return summary


def main() -> None:
    config=load_yaml('configs/candidate_ranker.yaml');seed_everything(int(config['seed']),True)
    device=resolve_device(config['device']);run_dir=Path(config['run_dir']);results_dir=Path(config['results_dir'])
    data=torch.load(run_dir/'expanded_candidate_records.pt',map_location='cpu',weights_only=False)
    proposals=torch.load(run_dir/'proposed_chunk_records.pt',map_location='cpu',weights_only=False)['records']
    proposal_by_id={r['root_id']:r['candidate'] for r in proposals}
    records=[]
    for source in data['records']:
        root={**source,'candidates':[]}
        for candidate in source['candidates']:
            copied={**candidate};descriptor=np.asarray(candidate['descriptor'],np.float32).reshape(-1)
            copied['descriptor']=np.pad(descriptor,(0,120-len(descriptor))).tolist();root['candidates'].append(copied)
        root['candidates'].append(proposal_by_id[source['root_id']]);records.append(root)
    features=data['root_features'];assert_source_split_isolated(features);stats=training_stats(features)
    train=CandidateDataset(features,records,'train',stats);dev=CandidateDataset(features,records,'dev',stats)
    hidden=int(config['ranker']['hidden_size']);privileged=PrivilegedCandidateRanker(
        train.rows[0]['privileged'].shape[-1],18,hidden,descriptor_dim=120)
    privileged_summary=train_line('privileged',privileged,train,dev,records,config,device,run_dir,results_dir)
    visual_summary=None
    if privileged_summary['passes_constraints']:
        visual=VisualCandidateRanker(train.rows[0]['temporal'].shape[-1],18,hidden,descriptor_dim=120)
        visual_summary=train_line('visual',visual,train,dev,records,config,device,run_dir,results_dir)
    torch.save({'root_features':features,'records':records,
                'metadata':{**data['metadata'],'candidates_per_root':18,'descriptor_dim':120,
                            'oracle_chunk_proposal_added':True}},run_dir/'oracle_chunk_augmented_records.pt')
    status={'roots':len(records),'candidate_records':len(train)+len(dev),
            'privileged_pass':privileged_summary['passes_constraints'],
            'visual_trained':visual_summary is not None,
            'visual_pass':visual_summary['passes_constraints'] if visual_summary else None,
            'next_stage':'paired_closed_loop' if visual_summary and visual_summary['passes_constraints'] else 'stop_no_safe_selector'}
    save_json(results_dir/'oracle_chunk_augmented_status.json',status);print(json.dumps(status,indent=2))


if __name__=='__main__':main()
