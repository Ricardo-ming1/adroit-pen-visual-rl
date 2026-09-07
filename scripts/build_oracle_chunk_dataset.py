from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from src.counterfactual.branch_dataset import load_root_dataset
from src.counterfactual.branch_rollout import (
    _event_metrics, _rollout_segment, load_frozen_actor,
)
from src.counterfactual.candidate_branches import label_candidate
from src.envs.adroit import VisualAdroitEnv
from src.utils import load_yaml, resolve_device, save_json, seed_everything


def main() -> None:
    config = load_yaml("configs/candidate_ranker.yaml")
    seed_everything(int(config["seed"]), bool(config["deterministic_torch"]))
    device = resolve_device(config["device"]); run_dir = Path(config["run_dir"])
    roots, _ = load_root_dataset(run_dir / "expanded_roots.pt")
    expanded = torch.load(run_dir / "expanded_candidate_records.pt", map_location="cpu", weights_only=False)
    zero_by_id = {r["root_id"]: r["candidates"][0] for r in expanded["records"]}
    vision, vision_stats, _ = load_frozen_actor(config["vision_checkpoint"], device)
    oracle, oracle_stats, _ = load_frozen_actor(config["oracle_checkpoint"], device)
    progress_path = run_dir / "oracle_chunk_records_progress.pt"; records = []
    if progress_path.exists():
        records = torch.load(progress_path, map_location="cpu", weights_only=False)["records"]
    done = {r["root_id"] for r in records}; transitions = sum(r["outcome"]["steps"] for r in records)
    env = VisualAdroitEnv(); env.reset(seed=0)
    spec = {"name": "oracle_takeover_5", "kind": "oracle_takeover",
            "intervention_steps": 5, "local": False, "deployable": False}
    try:
        for index, root in enumerate(roots):
            if root["root_id"] in done: continue
            observation = env.restore_training_state(copy.deepcopy(root["state"]))
            official=[]; dropped=[]; rewards=[]; trace=[]
            _, finished = _rollout_segment(
                env=env, root=root, spec=spec, start_step=0, stop_step=int(root["remaining_steps"]),
                observation=observation, official=official, dropped=dropped, rewards=rewards, trace=trace,
                vision_actor=vision, vision_stats=vision_stats, oracle_actor=oracle,
                oracle_stats=oracle_stats, safe_policy=None, device=device,
                rho=float(config["candidates"]["rho"]))
            candidate={"outcome":_event_metrics(root,official,dropped,rewards,float(config["candidates"]["gamma"]),complete=finished),
                       "completion_mask":bool(finished)}
            candidate["labels"]=label_candidate(candidate,zero_by_id[root["root_id"]],config["labels"])
            # Invert a rho=1 headroom map; this represents every bounded Oracle action
            # without hard clipping and makes zero the exact AWAC fallback.
            z=[]
            for entry in trace[:5]:
                base=np.asarray(entry["base_action"],np.float32); action=np.asarray(entry["executed_action"],np.float32)
                delta=action-base; room=np.where(delta>=0,1-base,1+base)
                z.append(np.clip(delta/np.maximum(room,1e-8),-1,1).astype(np.float32))
            records.append({"root_id":root["root_id"],"root_type":root["root_type"],"split":root["split"],
                            "source_episode_id":root["source_episode_seed"],"outcome":candidate["outcome"],
                            "labels":candidate["labels"],"oracle_z_chunk":np.stack(z),
                            "oracle_action_chunk":np.asarray([x["executed_action"] for x in trace[:5]],np.float32)})
            transitions += len(rewards)
            if (index+1)%20==0:
                torch.save({"records":records,"branch_transitions":transitions},progress_path)
    finally:
        env.close()
    torch.save({"records":records,"branch_transitions":transitions},run_dir/"oracle_chunk_records.pt")
    torch.save({"records":records,"branch_transitions":transitions},progress_path)
    by_type={}
    for phase in sorted({r["root_type"] for r in records}):
        rows=[r for r in records if r["root_type"]==phase]
        by_type[phase]={"roots":len(rows),"strong_gain":sum(r["labels"]["strong_gain"] for r in rows),
                        "harm":sum(r["labels"]["harm"] for r in rows)}
    summary={"roots":len(records),"branch_transitions":transitions,
             "strong_gain_roots":sum(r["labels"]["strong_gain"] for r in records),
             "harm_roots":sum(r["labels"]["harm"] for r in records),"by_root_type":by_type,
             "teacher_candidate":"Oracle takeover 5 steps (training diagnostic only)",
             "deployment_input":"none; labels only"}
    save_json(Path(config["results_dir"])/"oracle_chunk_branch_summary.json",summary)
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
