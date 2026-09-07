from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F

from src.algorithms.common import (
    checkpoint_payload,
    restore_rng_state,
    save_checkpoint,
    validation_key,
)
from src.models.actor import actor_inputs
from src.utils import CSVLogger


@torch.no_grad()
def offline_validation(actor, data, line: str, device: torch.device, batch_size: int) -> dict[str, float]:
    actor.eval()
    mse_total = 0.0
    nll_total = 0.0
    count = 0
    for batch in data.sequential_batches(
        batch_size, "validation", transitions=False, line=line, device=device
    ):
        distribution = actor.distribution(**actor_inputs(batch, line))
        deterministic = distribution.deterministic()
        n = batch["action"].shape[0]
        mse_total += float(F.mse_loss(deterministic, batch["action"], reduction="sum"))
        nll_total += float((-distribution.log_prob(batch["action"])).sum())
        count += n
    return {
        "validation_action_mse": mse_total / (count * data.action_dim),
        "validation_action_nll": nll_total / count,
    }


def train_bc(
    *,
    actor,
    actor_init: dict,
    data,
    line: str,
    config: dict,
    full_config: dict,
    seed: int,
    device: torch.device,
    run_dir: Path,
    rng: np.random.Generator,
    evaluator: Callable[[], dict],
    resume: bool = False,
) -> dict:
    actor.train()
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(config["actor_lr"]))
    logger = CSVLogger(run_dir / "learning_curves" / "bc.csv")
    pool = data.indices("train", transitions=False).copy()
    best_key: tuple[float, ...] | None = None
    best_metrics: dict = {}
    start_epoch = 1
    last_path = run_dir / "last_bc.pt"
    best_path = run_dir / "best_bc_validation.pt"
    if resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state"])
        optimizer.load_state_dict(checkpoint["optimizer_states"]["actor"])
        restore_rng_state(checkpoint, rng)
        start_epoch = int(checkpoint["progress"]["epoch"]) + 1
        if best_path.exists():
            best = torch.load(best_path, map_location=device, weights_only=False)
            best_metrics = best["metrics"]
            best_key = validation_key(
                best_metrics, float(best_metrics["validation_action_mse"])
            )

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        rng.shuffle(pool)
        losses: list[float] = []
        for start in range(0, len(pool), int(config["batch_size"])):
            batch = data.batch_from_indices(
                pool[start : start + int(config["batch_size"])],
                line,
                device,
                include_next=False,
            )
            distribution = actor.distribution(**actor_inputs(batch, line))
            loss = F.mse_loss(distribution.deterministic(), batch["action"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            losses.append(float(loss.detach()))

        should_evaluate = epoch == 1 or epoch == int(config["epochs"]) or epoch % int(
            config["eval_interval"]
        ) == 0
        if should_evaluate:
            offline = offline_validation(
                actor, data, line, device, int(config["batch_size"])
            )
            closed_loop = evaluator()
            row = {
                "epoch": epoch,
                "train_action_mse": float(np.mean(losses)),
                **offline,
                **{key: value for key, value in closed_loop.items() if key != "episodes"},
            }
            logger.log(row)
            key = validation_key(closed_loop, offline["validation_action_mse"])
            payload = checkpoint_payload(
                stage="bc",
                seed=seed,
                line=line,
                actor=actor,
                actor_init=actor_init,
                stats=data.stats.to_dict(),
                config=full_config,
                metrics={**offline, **closed_loop},
                optimizer_states={"actor": optimizer.state_dict()},
                progress={"epoch": epoch},
                rng=rng,
            )
            save_checkpoint(run_dir / "last_bc.pt", payload)
            if best_key is None or key > best_key:
                best_key = key
                best_metrics = payload["metrics"]
                save_checkpoint(run_dir / "best_bc_validation.pt", payload)
        elif epoch % 5 == 0:
            payload = checkpoint_payload(
                stage="bc",
                seed=seed,
                line=line,
                actor=actor,
                actor_init=actor_init,
                stats=data.stats.to_dict(),
                config=full_config,
                optimizer_states={"actor": optimizer.state_dict()},
                progress={"epoch": epoch},
                rng=rng,
            )
            save_checkpoint(run_dir / "last_bc.pt", payload)

    best = torch.load(run_dir / "best_bc_validation.pt", map_location=device, weights_only=False)
    actor.load_state_dict(best["actor_state"])
    return best_metrics
