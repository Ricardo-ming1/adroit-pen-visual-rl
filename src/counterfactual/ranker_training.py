from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.counterfactual.candidate_ranker import SelectorThreshold, select_candidate


def _forward(model, batch: dict[str, Any], line: str, device: torch.device):
    descriptor=batch["descriptor"].to(device).float(); candidate=batch["candidate_id"].to(device).long()
    if line == "privileged":
        return model(batch["privileged"].to(device).float(), batch["phase"].to(device).float(), descriptor, candidate)
    return model(batch["temporal"].to(device).float(), descriptor, candidate)


def train_ranker(model, train_dataset, dev_dataset, *, line: str, config: dict[str, Any],
                 device: torch.device) -> tuple[dict[str, torch.Tensor], list[dict[str, Any]]]:
    model.to(device)
    train_loader=DataLoader(train_dataset,batch_size=int(config["batch_size"]),shuffle=True)
    dev_loader=DataLoader(dev_dataset,batch_size=int(config["batch_size"]),shuffle=False)
    gain=np.array([r["strong_gain"] for r in train_dataset.rows]); harm=np.array([r["harm"] for r in train_dataset.rows])
    gain_weight=torch.tensor((len(gain)-gain.sum())/max(gain.sum(),1),device=device,dtype=torch.float32)
    harm_weight=torch.tensor((len(harm)-harm.sum())/max(harm.sum(),1),device=device,dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["learning_rate"]),weight_decay=float(config["weight_decay"]))
    best=None; best_score=-1e9; curve=[]
    for epoch in range(1,int(config["epochs"])+1):
        model.train(); losses=[]
        for batch in train_loader:
            output=_forward(model,batch,line,device); mask=batch["mask"].to(device).float()
            g=batch["strong_gain"].to(device).float(); h=batch["harm"].to(device).float()
            progress=batch["progress"].to(device).float()
            gain_loss=F.binary_cross_entropy_with_logits(output["gain_logit"],g,pos_weight=gain_weight,reduction="none")
            harm_loss=F.binary_cross_entropy_with_logits(output["harm_logit"],h,pos_weight=harm_weight,reduction="none")
            progress_loss=F.smooth_l1_loss(output["progress"],progress,reduction="none").mean(-1)
            # Zero-vs-candidate pairwise conservative ordering: true gains rank above
            # zero score, harms below it. Neutral samples receive no fabricated order.
            zero_id=torch.zeros_like(batch["candidate_id"]).to(device).long()
            zero_descriptor=torch.zeros_like(batch["descriptor"]).to(device).float()
            if line=="privileged": zero=model(batch["privileged"].to(device).float(),batch["phase"].to(device).float(),zero_descriptor,zero_id)
            else: zero=model(batch["temporal"].to(device).float(),zero_descriptor,zero_id)
            score=output["gain_logit"]-output["harm_logit"]; zero_score=zero["gain_logit"]-zero["harm_logit"]
            pair=(g*F.softplus(0.5-score+zero_score)+h*F.softplus(0.5+score-zero_score))/torch.clamp(g+h,min=1.0)
            loss=((gain_loss+harm_loss+float(config["progress_loss_coef"])*progress_loss+
                   float(config["pairwise_loss_coef"])*pair)*mask).sum()/mask.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10); optimizer.step()
            losses.append(float(loss.item()))
        if epoch==1 or epoch%5==0 or epoch==int(config["epochs"]):
            predictions=predict_ranker(model,dev_loader,line,device)
            summary=threshold_search(predictions,dev_dataset,config)
            score=summary["selected_net_strong_gain"]+summary["precision"]-summary["harm_rate"]*2
            curve.append({"epoch":epoch,"train_loss":float(np.mean(losses)),**{k:v for k,v in summary.items() if isinstance(v,(int,float,bool))}})
            if score>best_score:
                best_score=score; best=copy.deepcopy(model.state_dict())
    assert best is not None
    return best,curve


@torch.inference_mode()
def predict_ranker(model, loader, line: str, device: torch.device) -> list[dict[str, Any]]:
    model.eval(); output=[]
    for batch in loader:
        prediction=_forward(model,batch,line,device)
        pg=torch.sigmoid(prediction["gain_logit"]).cpu().numpy(); ph=torch.sigmoid(prediction["harm_logit"]).cpu().numpy()
        progress=prediction["progress"].cpu().numpy()
        for i in range(len(pg)):
            output.append({"root_index":int(batch["root_index"][i]),"root_id":batch["root_id"][i],
                           "root_type":batch["root_type"][i],"candidate_id":int(batch["candidate_id"][i]),
                           "prob_gain":float(pg[i]),"prob_harm":float(ph[i]),
                           "predicted_progress":progress[i].tolist()})
    return output


def selector_metrics(predictions: list[dict[str, Any]], dataset, threshold: SelectorThreshold) -> dict[str, Any]:
    by_root=defaultdict(list)
    for p in predictions: by_root[p["root_id"]].append(p)
    row_by_key={(r["root_id"],int(r["candidate_id"])):r for r in dataset.rows}
    selected=[]
    for root_id, values in by_root.items():
        values=sorted(values,key=lambda x:x["candidate_id"])
        index=select_candidate(np.array([x["prob_gain"] for x in values]),np.array([x["prob_harm"] for x in values]),threshold)
        pred=values[index]; true=row_by_key[(root_id,index)]
        selected.append({**pred,"selected_candidate_id":index,"strong_gain":int(true["strong_gain"]),"harm":int(true["harm"])})
    interventions=[x for x in selected if x["selected_candidate_id"]!=0]
    n=max(len(selected),1); m=len(interventions)
    precision=sum(x["strong_gain"] for x in interventions)/max(m,1)
    harm_rate=sum(x["harm"] for x in interventions)/max(m,1)
    by_phase={}
    for phase in ("goal_entry_pending_exit","stable_hold"):
        rows=[x for x in selected if x["root_type"]==phase]; active=[x for x in rows if x["selected_candidate_id"]!=0]
        by_phase[phase]={"roots":len(rows),"interventions":len(active),"strong_gains":sum(x["strong_gain"] for x in active),"harms":sum(x["harm"] for x in active)}
    return {"coverage":m/n,"precision":precision,"harm_rate":harm_rate,
            "selected_roots":m,"roots":len(selected),
            "selected_strong_gains":sum(x["strong_gain"] for x in interventions),
            "selected_harms":sum(x["harm"] for x in interventions),
            "selected_net_strong_gain":sum(x["strong_gain"]-x["harm"] for x in interventions),
            "by_root_type":by_phase,"selections":selected,
            "threshold":{"gain":threshold.gain,"harm":threshold.harm}}


def threshold_search(predictions: list[dict[str, Any]], dataset, config: dict[str, Any]) -> dict[str, Any]:
    candidates=[]
    for gain in config["thresholds"]["gain"]:
        for harm in config["thresholds"]["harm"]:
            result=selector_metrics(predictions,dataset,SelectorThreshold(float(gain),float(harm)))
            result["passes_constraints"]=bool(result["precision"]>=float(config["precision_min"]) and result["harm_rate"]<=float(config["harm_rate_max"]) and result["coverage"]>=float(config["coverage_min"]) and result["selected_net_strong_gain"]>0)
            candidates.append(result)
    passing=[x for x in candidates if x["passes_constraints"]]
    pool=passing or candidates
    return max(pool,key=lambda x:(x["passes_constraints"],x["selected_net_strong_gain"],x["precision"],-x["harm_rate"],x["coverage"]))


def attach_outcome_metrics(summary: dict[str, Any], records: list[dict[str, Any]], split: str) -> dict[str, Any]:
    root_by_id={r["root_id"]:r for r in records if r["split"]==split}
    zero_benchmark=zero_strict=chosen_benchmark=chosen_strict=0
    zero_exits=chosen_exits=0; oracle_positive=missed_positive=0
    selections=[]
    for selected in summary["selections"]:
        root=root_by_id[selected["root_id"]]; zero=root["candidates"][0]
        chosen=root["candidates"][selected["selected_candidate_id"]]
        z=zero["outcome"]; c=chosen["outcome"]
        zero_benchmark+=int(z["benchmark"]); zero_strict+=int(z["strict"]); zero_exits+=int(z["goal_exit_after_root"])
        chosen_benchmark+=int(c["benchmark"]); chosen_strict+=int(c["strict"]); chosen_exits+=int(c["goal_exit_after_root"])
        available=any(x["labels"]["strong_gain"] and not x["labels"]["harm"] for x in root["candidates"][1:])
        oracle_positive+=int(available); missed_positive+=int(available and not chosen["labels"]["strong_gain"])
        selections.append({**selected,"candidate_name":chosen["candidate"]["name"],
                           "zero_outcome":z,"chosen_outcome":c})
    n=max(len(selections),1)
    return {**summary,"zero_benchmark_rate":zero_benchmark/n,"selected_benchmark_rate":chosen_benchmark/n,
            "zero_strict_rate":zero_strict/n,"selected_strict_rate":chosen_strict/n,
            "zero_goal_exit_rate":zero_exits/n,"selected_goal_exit_rate":chosen_exits/n,
            "oracle_positive_roots":oracle_positive,"top1_regret_roots":missed_positive,
            "top1_regret_rate":missed_positive/max(oracle_positive,1),"selections":selections}
