from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.data.offline import NormalizationStats
from src.models.gated_residual_student import TemporalGatedVisualResidual
from src.utils import CSVLogger


class BranchSupervisionDataset(Dataset):
    def __init__(
        self,
        roots: list[dict[str, Any]],
        labels: dict[str, dict[str, Any]],
        stats: NormalizationStats,
    ):
        self.root_ids = [root["root_id"] for root in roots]
        mean = np.asarray(stats.proprio_mean, dtype=np.float32)
        std = np.asarray(stats.proprio_std, dtype=np.float32)
        sequences = []
        gate_targets = []
        residual_targets = []
        for root in roots:
            proprio = (np.asarray(root["proprio_history"]) - mean) / std
            sequence = np.concatenate(
                [
                    np.asarray(root["frozen_visual_latent_history"], dtype=np.float32),
                    proprio.astype(np.float32),
                    np.asarray(root["previous_base_actions"], dtype=np.float32),
                    np.asarray(root["previous_executed_actions"], dtype=np.float32),
                ],
                axis=-1,
            )
            label = labels[root["root_id"]]
            sequences.append(sequence)
            gate_targets.append(float(label["gate_target"]))
            residual_targets.append(np.asarray(label["residual_target"], dtype=np.float32))
        self.sequence = torch.as_tensor(np.asarray(sequences), dtype=torch.float32)
        self.gate_target = torch.as_tensor(gate_targets, dtype=torch.float32)
        self.residual_target = torch.as_tensor(
            np.asarray(residual_targets), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.root_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "sequence": self.sequence[index],
            "gate_target": self.gate_target[index],
            "residual_target": self.residual_target[index],
        }


def balanced_sampler(dataset: BranchSupervisionDataset) -> WeightedRandomSampler:
    targets = dataset.gate_target.numpy().astype(np.int64)
    counts = np.bincount(targets, minlength=2)
    if np.any(counts == 0):
        raise ValueError(f"Both gate classes are required, got counts {counts.tolist()}")
    weights = np.where(targets == 1, 0.5 / counts[1], 0.5 / counts[0])
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
    )


def balanced_gate_bce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    positive = targets > 0.5
    negative = ~positive
    terms = []
    if positive.any():
        terms.append(losses[positive].mean())
    if negative.any():
        terms.append(losses[negative].mean())
    return torch.stack(terms).mean()


def supervision_loss(
    z: torch.Tensor,
    gate_logit: torch.Tensor,
    gate_target: torch.Tensor,
    residual_target: torch.Tensor,
    zero_coef: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    gate_loss = balanced_gate_bce(gate_logit, gate_target)
    positive = gate_target > 0.5
    negative = ~positive
    residual_loss = (
        nn.functional.huber_loss(z[positive], residual_target[positive])
        if positive.any()
        else z.sum() * 0.0
    )
    zero_loss = (
        z[negative].square().mean() if negative.any() else z.sum() * 0.0
    )
    total = gate_loss + residual_loss + float(zero_coef) * zero_loss
    return total, {
        "total_loss": float(total.detach()),
        "gate_loss": float(gate_loss.detach()),
        "residual_loss": float(residual_loss.detach()),
        "zero_loss": float(zero_loss.detach()),
    }


@torch.inference_mode()
def dataset_predictions(
    model: TemporalGatedVisualResidual,
    dataset: BranchSupervisionDataset,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    sequence = dataset.sequence.to(device)
    z, gate_logit, _ = model.sequence_outputs(sequence)
    return {
        "probability": torch.sigmoid(gate_logit[:, -1]).cpu().numpy(),
        "z": z[:, -1].cpu().numpy(),
        "gate_target": dataset.gate_target.numpy(),
        "residual_target": dataset.residual_target.numpy(),
    }


def threshold_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = probabilities >= float(threshold)
    positive = targets > 0.5
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    negatives = int(np.sum(~positive))
    precision = tp / max(tp + fp, 1)
    false_positive_rate = fp / max(negatives, 1)
    coverage = float(np.mean(predicted))
    recall = tp / max(int(np.sum(positive)), 1)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "false_positive_rate": float(false_positive_rate),
        "coverage": coverage,
        "recall": float(recall),
        "true_positives": tp,
        "false_positives": fp,
    }


def choose_gate_threshold(
    predictions: dict[str, np.ndarray], config: dict[str, Any]
) -> tuple[dict[str, float] | None, list[dict[str, float]]]:
    rows = [
        threshold_metrics(
            predictions["probability"], predictions["gate_target"], threshold
        )
        for threshold in config["student"]["thresholds"]
    ]
    eligible = [
        row
        for row in rows
        if row["precision"] >= float(config["student"]["gate_precision_min"])
        and row["false_positive_rate"]
        <= float(config["student"]["false_positive_rate_max"])
        and row["coverage"] >= float(config["student"]["coverage_min"])
    ]
    selected = max(
        eligible,
        key=lambda row: (row["recall"], row["coverage"], row["precision"]),
        default=None,
    )
    return selected, rows


def _evaluate_loss(
    model: TemporalGatedVisualResidual,
    loader: DataLoader,
    device: torch.device,
    zero_coef: float,
) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            gate_target = batch["gate_target"].to(device)
            residual_target = batch["residual_target"].to(device)
            z, gate_logit, _ = model.sequence_outputs(sequence)
            _, metrics = supervision_loss(
                z[:, -1],
                gate_logit[:, -1],
                gate_target,
                residual_target,
                zero_coef,
            )
            rows.append(metrics)
    return {
        key: float(np.mean([row[key] for row in rows])) for key in rows[0]
    }


def train_gated_student(
    *,
    base_actor: nn.Module,
    base_checkpoint_path: str,
    stats: NormalizationStats,
    train_roots: list[dict[str, Any]],
    dev_roots: list[dict[str, Any]],
    branch_results: list[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    run_dir: Path,
) -> tuple[TemporalGatedVisualResidual, dict[str, Any]]:
    labels = {result["root_id"]: result for result in branch_results}
    train_data = BranchSupervisionDataset(train_roots, labels, stats)
    dev_data = BranchSupervisionDataset(dev_roots, labels, stats)
    model = TemporalGatedVisualResidual(
        base_actor,
        hidden_size=int(config["student"]["hidden_size"]),
        rho=float(config["branches"]["rho"]),
        initial_gate_probability=float(
            config["student"]["initial_gate_probability"]
        ),
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(
        trainable, lr=float(config["student"]["learning_rate"])
    )
    train_loader = DataLoader(
        train_data,
        batch_size=int(config["student"]["batch_size"]),
        sampler=balanced_sampler(train_data),
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=int(config["student"]["batch_size"]),
        shuffle=False,
    )
    zero_coef = float(config["student"]["zero_loss_coef"])
    logger = CSVLogger(run_dir / "student_learning_curve.csv")
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, int(config["student"]["epochs"]) + 1):
        model.train()
        train_rows = []
        for batch in train_loader:
            sequence = batch["sequence"].to(device)
            gate_target = batch["gate_target"].to(device)
            residual_target = batch["residual_target"].to(device)
            z, gate_logit, _ = model.sequence_outputs(sequence)
            loss, metrics = supervision_loss(
                z[:, -1],
                gate_logit[:, -1],
                gate_target,
                residual_target,
                zero_coef,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            optimizer.step()
            train_rows.append(metrics)
        dev_metrics = _evaluate_loss(model, dev_loader, device, zero_coef)
        row = {"epoch": epoch}
        row.update(
            {
                f"train_{key}": float(np.mean([value[key] for value in train_rows]))
                for key in train_rows[0]
            }
        )
        row.update({f"dev_{key}": value for key, value in dev_metrics.items()})
        logger.log(row)
        if dev_metrics["total_loss"] < best_loss:
            best_loss = dev_metrics["total_loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
                if not key.startswith("base_actor.")
            }
    if best_state is None:
        raise RuntimeError("Student training produced no checkpoint")
    model.load_state_dict(best_state, strict=False)
    train_predictions = dataset_predictions(model, train_data, device)
    predictions = dataset_predictions(model, dev_data, device)
    selected_threshold, threshold_rows = choose_gate_threshold(predictions, config)

    def residual_errors(values: dict[str, np.ndarray]) -> tuple[float, float]:
        positive = values["gate_target"] > 0.5
        if not positive.any():
            return np.nan, np.nan
        model_rmse = np.sqrt(
            np.mean(
                np.square(
                    values["z"][positive]
                    - values["residual_target"][positive]
                )
            )
        )
        zero_rmse = np.sqrt(
            np.mean(np.square(values["residual_target"][positive]))
        )
        return float(model_rmse), float(zero_rmse)

    train_rmse, train_zero_rmse = residual_errors(train_predictions)
    dev_rmse, dev_zero_rmse = residual_errors(predictions)
    residual_fit_pass = bool(
        train_rmse <= 0.8 * train_zero_rmse
        and dev_rmse <= 0.8 * dev_zero_rmse
    )
    summary = {
        "train_roots": len(train_data),
        "dev_roots": len(dev_data),
        "train_positive_roots": int(train_data.gate_target.sum()),
        "dev_positive_roots": int(dev_data.gate_target.sum()),
        "best_dev_total_loss": best_loss,
        "train_positive_residual_rmse": train_rmse,
        "train_zero_predictor_rmse": train_zero_rmse,
        "train_rmse_relative_to_zero": train_rmse / train_zero_rmse,
        "dev_positive_residual_rmse": dev_rmse,
        "dev_zero_predictor_rmse": dev_zero_rmse,
        "dev_rmse_relative_to_zero": dev_rmse / dev_zero_rmse,
        "residual_fit_pass": residual_fit_pass,
        "residual_fit_rule": "train and dev RMSE <= 80% of zero predictor",
        "selected_threshold": selected_threshold,
        "threshold_rows": threshold_rows,
        "gate_dev_pass": selected_threshold is not None,
    }
    checkpoint = {
        "stage": "counterfactual_gated_student_round0",
        "base_checkpoint": base_checkpoint_path,
        "normalization": stats.to_dict(),
        "model_init": {
            "hidden_size": model.hidden_size,
            "rho": model.rho,
            "initial_gate_probability": float(
                config["student"]["initial_gate_probability"]
            ),
        },
        "student_state": best_state,
        "gate_threshold": (
            selected_threshold["threshold"] if selected_threshold is not None else None
        ),
        "metrics": copy.deepcopy(summary),
    }
    torch.save(checkpoint, run_dir / "best_gated_student.pt")
    return model, summary
