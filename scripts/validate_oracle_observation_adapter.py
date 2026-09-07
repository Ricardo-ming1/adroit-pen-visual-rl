from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from src.counterfactual.branch_rollout import load_frozen_actor
from src.envs.adroit import hand_proprio, make_adroit_env, privileged_observation
from src.evaluation import evaluate_policy
from src.state_distillation.oracle_observation import OracleObservationAdapter
from src.utils import load_yaml, resolve_device, save_json


def main():
    config=load_yaml("configs/temporal_state_distillation.yaml"); device=resolve_device(config["device"])
    oracle,stats,_=load_frozen_actor(config["oracle_checkpoint"],device); adapter=OracleObservationAdapter().to(device)
    env=make_adroit_env(); raw_errors=[]; action_errors=[]; desired=[]
    with torch.inference_mode():
        for seed in range(30000,30100):
            env.reset(seed=seed); true=torch.as_tensor(privileged_observation(env),device=device).unsqueeze(0)
            prop=torch.as_tensor(hand_proprio(env),device=device).unsqueeze(0)
            prim=torch.as_tensor(adapter.primitive_from_runtime(env),device=device).unsqueeze(0)
            rebuilt=adapter(prop,prim); raw_errors.append(float((rebuilt-true).abs().max()))
            mean=torch.as_tensor(stats.oracle_mean,device=device);std=torch.as_tensor(stats.oracle_std,device=device)
            a=oracle.act(actor=(true-mean)/std,deterministic=True);b=oracle.act(actor=(rebuilt-mean)/std,deterministic=True)
            action_errors.append(float((a-b).abs().max()));desired.append((true[0,24:27]-true[0,39:42]).cpu().numpy())
    env.close(); metrics=evaluate_policy(oracle,"oracle",stats,range(30000,30100),device)
    result={"schema":adapter.schema(),"max_raw_observation_error":max(raw_errors),"max_action_error":max(action_errors),
            "desired_position_max_deviation":float(np.max(np.abs(np.stack(desired)-np.asarray([0,-.2,.25])))),
            "benchmark_success":metrics["benchmark_success"],"strict_stable_success":metrics["strict_stable_success"],
            "goal_exit_after_entry_rate":metrics["goal_exit_after_entry_rate"],"episodes":metrics["episodes"]}
    save_json("results/v6/oracle_adapter_validation100.json",result);print(json.dumps({k:v for k,v in result.items() if k not in ("episodes","schema")},indent=2))


if __name__=="__main__":main()
