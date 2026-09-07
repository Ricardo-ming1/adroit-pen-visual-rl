from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.algorithms.awac import train_awac, warmup_critic
from src.algorithms.bc import train_bc
from src.algorithms.ppo_anchor import train_ppo_anchor
from src.data.offline import PenOfflineData
from src.evaluation import evaluate_policy
from src.models.actor import SquashedGaussianActor
from src.models.critic import DoubleQCritic, ValueNetwork
from src.utils import load_yaml, package_versions, resolve_device, save_json, seed_everything


def _save_resolved_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _load(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _smoke_config(config: dict[str, Any]) -> None:
    config["bc"].update(epochs=2, eval_interval=1, validation_episodes=2)
    config["critic"].update(updates=20, log_interval=10)
    config["awac"].update(updates=20, eval_interval=10, validation_episodes=2)
    config["ppo"].update(
        rollout_steps=128,
        num_rollouts=1,
        ppo_epochs=2,
        minibatch_size=64,
        validation_episodes=2,
    )


def train_pipeline(
    config_path: str,
    line: str,
    seed: int,
    stage: str = "all",
    no_anchor: bool = False,
    smoke: bool = False,
    resume: bool = False,
) -> Path:
    config = load_yaml(config_path)
    config["seed"] = seed
    config["line"] = line
    config["variant"] = "full_no_anchor" if no_anchor else "full"
    if no_anchor:
        config["ppo"]["reference_kl_coef"] = 0.0
        config["ppo"]["bc_aux_coef"] = 0.0
    if smoke:
        _smoke_config(config)
        config["run_root"] = f"runs/smoke/{line}"
    seed_everything(seed, bool(config.get("deterministic_torch", False)))
    device = resolve_device(config["device"])

    data_config = load_yaml(config["data_config"])
    data = PenOfflineData(
        data_config["output_path"], frame_stack=int(data_config["frame_stack"])
    )
    run_dir = Path(config["run_root"]) / f"{config['variant']}_seed_{seed}"
    (run_dir / "learning_curves").mkdir(parents=True, exist_ok=True)
    _save_resolved_config(run_dir / "config.yaml", config)
    save_json(
        run_dir / "run_meta.json",
        {
            "project": config["project"],
            "line": line,
            "variant": config["variant"],
            "seed": seed,
            "stage_requested": stage,
            "environment_id": data_config["environment_id"],
            "dataset_id": data_config["dataset_id"],
            "environment_and_packages": package_versions(),
            "dataset_path": str(Path(data_config["output_path"]).resolve()),
            "normalization_source": "train episodes only",
            "online_rollout_seed_start": 1_000_000 + seed * 10_000,
        },
    )
    save_json(run_dir / "normalization.json", data.stats.to_dict())

    actor_init = {
        "line": line,
        "action_dim": data.action_dim,
        "frame_stack": int(data_config["frame_stack"]),
        "image_size": int(data_config["image_size"]),
        **config["actor"],
    }
    actor = SquashedGaussianActor(**actor_init).to(device)
    critic = DoubleQCritic().to(device)
    target_critic = critic.target_copy().to(device)
    rng = np.random.default_rng(seed)
    validation_seeds = list(
        range(
            int(config["validation_seeds"]["start"]),
            int(config["validation_seeds"]["start"])
            + int(config["validation_seeds"]["count"]),
        )
    )

    def evaluator(count: int):
        return lambda: evaluate_policy(
            actor,
            line,
            data.stats,
            validation_seeds[:count],
            device,
        )

    stages = ["bc", "critic", "awac", "ppo"]
    requested = stages if stage == "all" else [stage]

    if "bc" in requested:
        metrics = train_bc(
            actor=actor,
            actor_init=actor_init,
            data=data,
            line=line,
            config=config["bc"],
            full_config=config,
            seed=seed,
            device=device,
            run_dir=run_dir,
            rng=rng,
            evaluator=evaluator(int(config["bc"]["validation_episodes"])),
            resume=resume,
        )
        save_json(run_dir / "bc_validation.json", metrics)
    else:
        checkpoint = _load(run_dir / "best_bc_validation.pt", device)
        actor.load_state_dict(checkpoint["actor_state"])

    if "critic" in requested:
        warmup_critic(
            actor=actor,
            actor_init=actor_init,
            critic=critic,
            target_critic=target_critic,
            data=data,
            line=line,
            config=config["critic"],
            full_config=config,
            seed=seed,
            device=device,
            run_dir=run_dir,
            rng=rng,
            resume=resume,
        )
    elif stage in ("awac", "ppo"):
        checkpoint = _load(run_dir / "last_critic_warmup.pt", device)
        critic.load_state_dict(checkpoint["critic_state"])
        target_critic.load_state_dict(checkpoint["target_critic_state"])

    if "awac" in requested:
        metrics = train_awac(
            actor=actor,
            actor_init=actor_init,
            critic=critic,
            target_critic=target_critic,
            data=data,
            line=line,
            config=config["awac"],
            full_config=config,
            seed=seed,
            device=device,
            run_dir=run_dir,
            rng=rng,
            evaluator=evaluator(int(config["awac"]["validation_episodes"])),
            resume=resume,
        )
        save_json(run_dir / "offline_awac_validation.json", metrics)
    elif stage == "ppo":
        checkpoint = _load(run_dir / "best_offline_awac_validation.pt", device)
        actor.load_state_dict(checkpoint["actor_state"])
        critic.load_state_dict(checkpoint["critic_state"])
        target_critic.load_state_dict(checkpoint["target_critic_state"])

    if "ppo" in requested:
        reference_actor = copy.deepcopy(actor).to(device).eval().requires_grad_(False)
        value = ValueNetwork().to(device)
        metrics = train_ppo_anchor(
            actor=actor,
            actor_init=actor_init,
            reference_actor=reference_actor,
            value=value,
            data=data,
            line=line,
            config=config["ppo"],
            full_config=config,
            seed=seed,
            device=device,
            run_dir=run_dir,
            rng=rng,
            evaluator=evaluator(int(config["ppo"]["validation_episodes"])),
            online_seed_start=1_000_000 + seed * 10_000,
            resume=resume,
        )
        save_json(run_dir / "online_ppo_validation.json", metrics)
    return run_dir
