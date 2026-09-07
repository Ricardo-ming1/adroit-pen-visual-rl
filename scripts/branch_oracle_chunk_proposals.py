from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.counterfactual.branch_dataset import load_root_dataset
from src.counterfactual.branch_rollout import _event_metrics, _rollout_segment, load_frozen_actor
from src.counterfactual.candidate_branches import _candidate_with_chunk, label_candidate
from src.counterfactual.oracle_chunk import VisualOracleChunkProposer
from src.envs.adroit import VisualAdroitEnv
from src.utils import load_yaml, resolve_device, save_json, seed_everything


@torch.inference_mode()
def main() -> None:
    config=load_yaml('configs/candidate_ranker.yaml');seed_everything(int(config['seed']),True)
    device=resolve_device(config['device']);run_dir=Path(config['run_dir']);results_dir=Path(config['results_dir'])
    roots,_=load_root_dataset(run_dir/'expanded_roots.pt')
    data=torch.load(run_dir/'expanded_candidate_records.pt',map_location='cpu',weights_only=False)
    feature_by_id={x['root_id']:x for x in data['root_features']};zero_by_id={x['root_id']:x['candidates'][0] for x in data['records']}
    checkpoint=torch.load(run_dir/'best_oracle_chunk_proposer.pt',map_location=device,weights_only=False)
    model=VisualOracleChunkProposer(**checkpoint['model_init']).to(device);model.load_state_dict(checkpoint['model_state']);model.eval()
    mean=np.asarray(checkpoint['temporal_mean'],np.float32);std=np.asarray(checkpoint['temporal_std'],np.float32)
    vision,vision_stats,_=load_frozen_actor(config['vision_checkpoint'],device)
    oracle,oracle_stats,_=load_frozen_actor(config['oracle_checkpoint'],device)
    progress_path=run_dir/'proposed_chunk_records_progress.pt';records=[]
    if progress_path.exists():records=torch.load(progress_path,map_location='cpu',weights_only=False)['records']
    done={r['root_id'] for r in records};transitions=sum(r['outcome']['steps'] for r in records)
    import src.counterfactual.branch_rollout as rollout_module
    previous=rollout_module._candidate_action;rollout_module._candidate_action=_candidate_with_chunk
    env=VisualAdroitEnv();env.reset(seed=0)
    try:
        for index,root in enumerate(roots):
            if root['root_id'] in done:continue
            feature=feature_by_id[root['root_id']]
            temporal=np.concatenate([feature['frozen_visual_latent_history'],feature['proprio_history'],feature['previous_base_actions'],feature['previous_executed_actions']],-1)
            x=torch.as_tensor((temporal-mean)/std,device=device,dtype=torch.float32).unsqueeze(0)
            z=model(x).squeeze(0).cpu().numpy().astype(np.float32)
            spec={'name':'visual_oracle_chunk_proposal','kind':'fixed_chunk','z_chunk':z,
                  'intervention_steps':5,'local':True,'deployable':True}
            observation=env.restore_training_state(copy.deepcopy(root['state']));official=[];dropped=[];rewards=[];trace=[]
            _,finished=_rollout_segment(env=env,root=root,spec=spec,start_step=0,stop_step=int(root['remaining_steps']),
                observation=observation,official=official,dropped=dropped,rewards=rewards,trace=trace,
                vision_actor=vision,vision_stats=vision_stats,oracle_actor=oracle,oracle_stats=oracle_stats,
                safe_policy=None,device=device,rho=1.0)
            candidate={'candidate_id':17,'candidate':{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in spec.items()},
                       'descriptor':z.reshape(-1).tolist(),'outcome':_event_metrics(root,official,dropped,rewards,float(config['candidates']['gamma']),complete=finished),
                       'completion_mask':bool(finished),'trace':trace,
                       'correction_rms':float(np.mean([x['correction_rms'] for x in trace]))}
            candidate['labels']=label_candidate(candidate,zero_by_id[root['root_id']],config['labels'])
            records.append({'root_id':root['root_id'],'root_type':root['root_type'],'split':root['split'],
                            'source_episode_id':root['source_episode_seed'],'candidate':candidate})
            transitions+=len(rewards)
            if (index+1)%20==0:torch.save({'records':records,'branch_transitions':transitions},progress_path)
    finally:
        rollout_module._candidate_action=previous;env.close()
    torch.save({'records':records,'branch_transitions':transitions},run_dir/'proposed_chunk_records.pt')
    torch.save({'records':records,'branch_transitions':transitions},progress_path)
    summary={'roots':len(records),'branch_transitions':transitions,
             'strong_gain_roots':sum(r['candidate']['labels']['strong_gain'] for r in records),
             'harm_roots':sum(r['candidate']['labels']['harm'] for r in records),
             'mean_correction_rms':float(np.mean([r['candidate']['correction_rms'] for r in records]))}
    save_json(results_dir/'proposed_chunk_branch_summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
