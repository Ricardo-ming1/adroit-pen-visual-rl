from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from src.counterfactual.branch_rollout import deterministic_action, frozen_visual_latent, load_frozen_actor, _candidate_action
from src.counterfactual.candidate_branches import _candidate_with_chunk, candidate_descriptor
from src.counterfactual.candidate_ranker import PrivilegedCandidateRanker, phase_one_hot
from src.counterfactual.oracle_chunk import VisualOracleChunkProposer
from src.counterfactual.ranker_dataset import _privileged, training_stats
from src.envs.adroit import HORIZON, VisualAdroitEnv, current_metrics, full_state
from src.utils import load_yaml, resolve_device, save_json


def load_specs(path):
    data=json.loads(Path(path).read_text()); specs=[]
    for source in data['specs']:
        spec=dict(source)
        for key in ('direction','z_chunk'):
            if key in spec: spec[key]=np.asarray(spec[key],np.float32)
        specs.append(spec)
    return specs


def summarize(official,dropped):
    first=next((i for i,x in enumerate(official) if x),None);streak=maximum=0
    for reached in official: streak=streak+1 if reached else 0;maximum=max(maximum,streak)
    return {'benchmark':bool(any(official)),'strict':bool(len(official)>=20 and all(official[-20:]) and not any(dropped)),
            'exit':bool(first is not None and not all(official[first:])), 'hold20':bool(maximum>=20),'max_streak':maximum}


@torch.inference_mode()
def rollout(seed, intervene, bundle):
    env=VisualAdroitEnv();obs,_=env.reset(seed=seed);vision,vs,oracle,os,ranker,rank_stats,proposer,pmean,pstd,specs,device=bundle
    previous_base=deque([np.zeros(24,np.float32) for _ in range(4)],maxlen=4);previous_executed=deque([np.zeros(24,np.float32) for _ in range(4)],maxlen=4)
    props=deque([obs['proprio'].copy() for _ in range(4)],maxlen=4);latent=frozen_visual_latent(vision,obs,vs,device)
    latents=deque([latent.copy() for _ in range(4)],maxlen=4);selected=0;z_chunk=None;triggered=False;branch_step=0
    official=[];dropped=[];selection=None
    try:
        for _ in range(HORIZON):
            metrics=current_metrics(env.env); streak=0
            for x in reversed(official):
                if not x: break
                streak+=1
            props.append(obs['proprio'].copy());latents.append(frozen_visual_latent(vision,obs,vs,device))
            base=deterministic_action(vision,'vision',obs,vs,device);action=base
            if intervene and not triggered and metrics['official_goal'] and streak<20:
                feature={'simulator_qpos':full_state(env.env)['qpos'],'simulator_qvel':full_state(env.env)['qvel'],
                         'target_orientation':full_state(env.env)['desired_orien']}
                privileged=(_privileged(feature)-rank_stats.privileged_mean)/rank_stats.privileged_std
                temporal=np.concatenate([np.stack(latents),np.stack(props),np.stack(previous_base),np.stack(previous_executed)],-1)
                proposal_input=(temporal-pmean)/pstd
                z_chunk=proposer(torch.as_tensor(proposal_input,device=device).unsqueeze(0)).squeeze(0).cpu().numpy()
                descriptors=[np.pad(candidate_descriptor(s).reshape(-1),(0,120-candidate_descriptor(s).size)) for s in specs]
                descriptors.append(z_chunk.reshape(-1));ids=torch.arange(18,device=device)
                pred=ranker(torch.as_tensor(privileged,device=device).repeat(18,1),
                    torch.as_tensor(phase_one_hot('goal_entry_pending_exit'),device=device).repeat(18,1),
                    torch.as_tensor(np.stack(descriptors),device=device),ids)
                pg=torch.sigmoid(pred['gain_logit']).cpu().numpy();ph=torch.sigmoid(pred['harm_logit']).cpu().numpy()
                eligible=np.flatnonzero((pg>=.6)&(ph<=.3));eligible=eligible[eligible!=0]
                selected=int(eligible[np.argmax(pg[eligible]-2*ph[eligible])]) if len(eligible) else 0
                triggered=True;selection={'step':len(official),'candidate_id':selected,'prob_gain':float(pg[selected]),'prob_harm':float(ph[selected])}
            if triggered and selected>0:
                if selected==17:
                    spec={'kind':'fixed_chunk','z_chunk':z_chunk,'intervention_steps':5};rho=1.0
                else: spec=specs[selected];rho=.05
                action,_,base=_candidate_with_chunk(spec=spec,branch_step=branch_step,observation=obs,vision_actor=vision,vision_stats=vs,
                    oracle_actor=oracle,oracle_stats=os,safe_policy=None,device=device,rho=rho);branch_step+=1
            previous_base.append(base.copy());previous_executed.append(action.copy())
            obs,_,terminated,truncated,info=env.step(action);official.append(bool(info['official_goal']));dropped.append(bool(info['dropped']))
            if terminated or truncated: break
    finally: env.close()
    return {**summarize(official,dropped),'triggered':triggered,'selection':selection}


def main():
    c=load_yaml('configs/temporal_state_distillation.yaml');device=resolve_device(c['device']);base=Path(c['v5_run_dir'])
    vision,vs,_=load_frozen_actor(c['vision_checkpoint'],device);oracle,os,_=load_frozen_actor(c['oracle_checkpoint'],device)
    expanded=torch.load(base/'expanded_candidate_records.pt',map_location='cpu',weights_only=False);rank_stats=training_stats(expanded['root_features'])
    ranker=PrivilegedCandidateRanker(64,18,128,descriptor_dim=120).to(device);rck=torch.load(c['v5_ranker_checkpoint'],map_location=device,weights_only=False)
    ranker.load_state_dict(rck['model_state']);ranker.eval();pck=torch.load(base/'best_oracle_chunk_proposer.pt',map_location=device,weights_only=False)
    proposer=VisualOracleChunkProposer(**pck['model_init']).to(device);proposer.load_state_dict(pck['model_state']);proposer.eval()
    bundle=(vision,vs,oracle,os,ranker,rank_stats,proposer,np.asarray(pck['temporal_mean'],np.float32),np.asarray(pck['temporal_std'],np.float32),load_specs(base/'candidate_bank.json'),device)
    rows=[]
    for index,seed in enumerate(range(6400000,6400200)):
        baseline=rollout(seed,False,bundle);pilot=rollout(seed,True,bundle);rows.append({'seed':seed,'baseline':baseline,'pilot':pilot})
        if (index+1)%20==0:print(json.dumps({'episodes':index+1}),flush=True)
    def rate(side,key):return float(np.mean([r[side][key] for r in rows]))
    gains=harms=neutral=0
    for r in rows:
        b,p=r['baseline'],r['pilot'];gain=(not b['strict'] and p['strict']) or (not b['benchmark'] and p['benchmark']) or (b['exit'] and not p['exit']) or (not b['hold20'] and p['hold20'])
        harm=(b['strict'] and not p['strict']) or (b['benchmark'] and not p['benchmark']) or (not b['exit'] and p['exit']) or (b['hold20'] and not p['hold20'])
        gains+=gain and not harm;harms+=harm;neutral+=not gain and not harm
    result={'episodes':200,'threshold':rck['selector_threshold'],'utility':'strong_gain - 2 * harm',
        'baseline':{k:rate('baseline',k) for k in ('benchmark','strict','exit','hold20')},
        'pilot':{k:rate('pilot',k) for k in ('benchmark','strict','exit','hold20')},
        'interventions':sum(r['pilot']['triggered'] and r['pilot']['selection']['candidate_id']>0 for r in rows),
        'paired_gains':gains,'paired_harms':harms,'paired_neutral':neutral,'episode_level_net_utility':gains-2*harms,
        'candidate_routing_route_closed':True,'deployable_claim':False,'rows':rows}
    save_json('results/v6/v5_privileged_closure_pilot.json',result);print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':main()
