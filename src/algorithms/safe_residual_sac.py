from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from src.algorithms.residual_sac import sac_target
from src.models.critic import soft_update
from src.models.safe_residual import headroom_action


def update_safe_residual_sac(
    *,
    residual_actor,
    critic,
    target_critic,
    actor_optimizer,
    critic_optimizer,
    critic_batch: dict[str, torch.Tensor],
    actor_batch: dict[str, torch.Tensor],
    rho_effective: float,
    gamma: float,
    tau: float,
    reward_scale: float,
    temperature: float,
    correction_budget: float,
    lambda_anchor: float,
    q_scale_ema: torch.Tensor,
    q_scale_decay: float,
    update_actor: bool,
    max_grad_norm: float = 10.0,
) -> dict[str, float]:
    with torch.no_grad():
        next_distribution = residual_actor.distribution(critic_batch["next_feature"])
        next_z, next_pre_tanh = next_distribution.rsample()
        next_log_prob = next_distribution.log_prob(next_z, next_pre_tanh)
        next_base = critic_batch["next_feature"][..., -residual_actor.action_dim :]
        next_action, _ = headroom_action(next_base, next_z, rho_effective)
        next_q = target_critic.minimum(critic_batch["next_feature"], next_action)
        target = sac_target(
            critic_batch["reward"],
            critic_batch["terminated"],
            next_q,
            next_log_prob,
            torch.as_tensor(temperature, device=next_q.device),
            gamma,
            reward_scale,
        )

    q1, q2 = critic(critic_batch["feature"], critic_batch["action"])
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    critic_optimizer.step()
    soft_update(target_critic, critic, tau)

    actor_metrics = {
        "actor_loss": 0.0,
        "normalized_q_gain": 0.0,
        "anchor_loss": 0.0,
        "q_scale": float(max(float(q_scale_ema), 1.0)),
        "q_base": 0.0,
        "q_executed": 0.0,
        "predicted_q_gain": 0.0,
        "correction_rms": 0.0,
        "correction_rms_max_dim": 0.0,
        "z_extreme_rate": 0.0,
        "base_boundary_rate": 0.0,
        "executed_boundary_rate": 0.0,
        "residual_added_boundary_rate": 0.0,
        "numerical_clamp_count": 0.0,
        "residual_log_prob": 0.0,
    }
    if update_actor:
        distribution = residual_actor.distribution(actor_batch["feature"])
        z, pre_tanh = distribution.rsample()
        log_prob = distribution.log_prob(z, pre_tanh)
        base_action = actor_batch["feature"][..., -residual_actor.action_dim :]
        action, action_info = headroom_action(base_action, z, rho_effective)

        critic.requires_grad_(False)
        q_base = critic.minimum(actor_batch["feature"], base_action)
        q_executed = critic.minimum(actor_batch["feature"], action)
        with torch.no_grad():
            mean_abs_q = q_base.abs().mean()
            q_scale_ema.mul_(float(q_scale_decay)).add_(
                mean_abs_q, alpha=1.0 - float(q_scale_decay)
            )
            q_scale = q_scale_ema.clamp_min(1.0)
        normalized_q_gain = (q_executed - q_base.detach()) / q_scale.detach()
        correction = action_info["correction"]
        anchor_loss = (correction / float(correction_budget)).square().mean()
        actor_loss = (
            -normalized_q_gain.mean()
            + float(lambda_anchor) * anchor_loss
            + float(temperature) / q_scale.detach() * log_prob.mean()
        )
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(residual_actor.parameters(), max_grad_norm)
        actor_optimizer.step()
        critic.requires_grad_(True)

        correction_rms_by_dim = correction.detach().square().mean(dim=0).sqrt()
        base_boundary = base_action.detach().abs() >= 0.95
        executed_boundary = action.detach().abs() >= 0.95
        actor_metrics = {
            "actor_loss": float(actor_loss.detach()),
            "normalized_q_gain": float(normalized_q_gain.detach().mean()),
            "anchor_loss": float(anchor_loss.detach()),
            "q_scale": float(q_scale.detach()),
            "q_base": float(q_base.detach().mean()),
            "q_executed": float(q_executed.detach().mean()),
            "predicted_q_gain": float((q_executed - q_base).detach().mean()),
            "correction_rms": float(correction.detach().square().mean().sqrt()),
            "correction_rms_max_dim": float(correction_rms_by_dim.max()),
            "z_extreme_rate": float((z.detach().abs() > 0.95).float().mean()),
            "base_boundary_rate": float(base_boundary.float().mean()),
            "executed_boundary_rate": float(executed_boundary.float().mean()),
            "residual_added_boundary_rate": float(
                (executed_boundary & ~base_boundary).float().mean()
            ),
            "numerical_clamp_count": float(
                action_info["numerical_clamp"].detach().sum()
            ),
            "residual_log_prob": float(log_prob.detach().mean()),
        }

    values = torch.cat([q1.detach(), q2.detach()])
    metrics = {
        "critic_loss": float(critic_loss.detach()),
        "q_mean": float(values.mean()),
        "q_std": float(values.std()),
        "q_abs_p99": float(torch.quantile(values.abs(), 0.99)),
        "q_target_mean": float(target.mean()),
        "q_target_std": float(target.std()),
        **actor_metrics,
    }
    if not np.all(np.isfinite(list(metrics.values()))) or not torch.isfinite(target).all():
        raise FloatingPointError(f"Non-finite Safe Residual SAC update: {metrics}")
    return metrics
