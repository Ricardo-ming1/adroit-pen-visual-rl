from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.state_distillation.dataset import EpisodeSequenceDataset, StateTargetStats
from src.state_distillation.oracle_observation import OracleObservationAdapter, geodesic_angle, rotation_6d_to_matrix


def loss_terms(model, batch, adapter, oracle_actor, oracle_stats, vision_stats,
               target_stats: StateTargetStats, action_coef: float = 1.0) -> dict[str, torch.Tensor]:
    output = model(batch["rgb"], batch["proprio"], batch["previous_action"], batch["mask"])
    predicted = output["primitive"]
    target = batch["primitive"]
    mean = torch.as_tensor(target_stats.mean, device=predicted.device)
    std = torch.as_tensor(target_stats.std, device=predicted.device)
    state_low = F.smooth_l1_loss((predicted[:, :9] - mean) / std, (target[:, :9] - mean) / std)
    pred_obj, true_obj = rotation_6d_to_matrix(predicted[:, 9:15]), rotation_6d_to_matrix(target[:, 9:15])
    pred_tar, true_tar = rotation_6d_to_matrix(predicted[:, 15:21]), rotation_6d_to_matrix(target[:, 15:21])
    rotation_6d = F.smooth_l1_loss(predicted[:, 9:21], target[:, 9:21])
    geodesic = geodesic_angle(pred_obj, true_obj).mean() + geodesic_angle(pred_tar, true_tar).mean()
    state = state_low + 0.25 * rotation_6d + 0.5 * geodesic
    proprio_mean = torch.as_tensor(vision_stats.proprio_mean, device=predicted.device)
    proprio_std = torch.as_tensor(vision_stats.proprio_std, device=predicted.device)
    current_raw_proprio = batch["proprio"][:, -1] * proprio_std + proprio_mean
    predicted_observation = adapter(current_raw_proprio, predicted)
    oracle_mean = torch.as_tensor(oracle_stats.oracle_mean, device=predicted.device)
    oracle_std = torch.as_tensor(oracle_stats.oracle_std, device=predicted.device)
    predicted_dist = oracle_actor.distribution(actor=(predicted_observation - oracle_mean) / oracle_std)
    with torch.no_grad():
        true_dist = oracle_actor.distribution(actor=(batch["oracle_observation"] - oracle_mean) / oracle_std)
        true_action = true_dist.deterministic()
        true_mean = true_dist.mean.detach()
    action = F.mse_loss(predicted_dist.deterministic(), true_action) + 0.25 * F.mse_loss(predicted_dist.mean, true_mean)
    phase = F.cross_entropy(output["phase_logits"], batch["phase"])
    exit_target = torch.stack((batch["exit_10"], batch["exit_20"]), -1)
    auxiliary = phase + F.binary_cross_entropy_with_logits(output["exit_logits"], exit_target)
    total = state + float(action_coef) * action + 0.2 * auxiliary
    return {"total": total, "state": state, "state_low": state_low, "geodesic": geodesic,
            "action": action, "auxiliary": auxiliary}


@torch.no_grad()
def evaluate_loss(model, loader, *args) -> dict[str, float]:
    model.eval(); totals: dict[str, float] = {}; count = 0
    for batch in loader:
        batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        terms = loss_terms(model, batch, *args)
        weight = len(batch["rgb"]); count += weight
        for key, value in terms.items(): totals[key] = totals.get(key, 0.0) + float(value) * weight
    return {key: value / max(count, 1) for key, value in totals.items()}


def train_estimator(model, train_dataset: EpisodeSequenceDataset, dev_dataset: EpisodeSequenceDataset,
                    adapter, oracle_actor, oracle_stats, vision_stats, target_stats: StateTargetStats,
                    *, epochs: int, batch_size: int, learning_rate: float, cnn_learning_rate: float,
                    action_coef: float, checkpoint: str | Path, curve_path: str | Path) -> dict[str, Any]:
    device = next(model.parameters()).device
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    cnn_ids = {id(p) for p in model.visual_encoder.parameters() if p.requires_grad}
    main = [p for p in model.parameters() if p.requires_grad and id(p) not in cnn_ids]
    cnn = [p for p in model.visual_encoder.parameters() if p.requires_grad]
    groups = [{"params": main, "lr": learning_rate}]
    if cnn: groups.append({"params": cnn, "lr": cnn_learning_rate})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    best = float("inf"); best_state = None; curve: list[dict[str, float]] = []
    oracle_actor.requires_grad_(False).eval()
    for epoch in range(1, epochs + 1):
        model.train(); running = 0.0; seen = 0
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            terms = loss_terms(model, batch, adapter, oracle_actor, oracle_stats, vision_stats, target_stats, action_coef)
            optimizer.zero_grad(set_to_none=True); terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            running += float(terms["total"].detach()) * len(batch["rgb"]); seen += len(batch["rgb"])
        dev = evaluate_loss(model, dev_loader, adapter, oracle_actor, oracle_stats, vision_stats, target_stats, action_coef)
        row = {"epoch": epoch, "train_total": running / max(seen, 1), **{f"dev_{k}": v for k, v in dev.items()}}
        curve.append(row)
        if dev["total"] < best:
            best = dev["total"]; best_state = copy.deepcopy(model.state_dict())
    assert best_state is not None
    model.load_state_dict(best_state)
    checkpoint = Path(checkpoint); checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": best_state, "target_stats": target_stats.to_dict(), "dev_loss": best}, checkpoint)
    import json
    path = Path(curve_path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(curve, indent=2) + "\n")
    return {"best_dev_loss": best, "epochs": epochs, "updates": epochs * len(train_loader), "curve": curve}
