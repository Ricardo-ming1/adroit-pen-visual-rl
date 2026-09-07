from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from src.models.critic import soft_update


def bootstrap_mask(terminated: torch.Tensor) -> torch.Tensor:
    """Only true MDP terminations disable bootstrapping; truncations do not."""
    return 1.0 - terminated.to(dtype=torch.float32)


def transition_next_observation(
    next_observation: Any,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> Any:
    """Use the pre-autoreset final observation when a vector env provides it."""
    if (terminated or truncated) and info.get("final_observation") is not None:
        return info["final_observation"]
    return next_observation


def sac_target(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    next_q: torch.Tensor,
    residual_log_prob: torch.Tensor,
    temperature: torch.Tensor,
    gamma: float,
    reward_scale: float,
) -> torch.Tensor:
    return reward_scale * reward + float(gamma) * bootstrap_mask(terminated) * (
        next_q - temperature * residual_log_prob
    )


def combined_action(
    feature: torch.Tensor,
    residual_action: torch.Tensor,
    residual_scale: float,
    action_dim: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_action = feature[..., -action_dim:]
    unclipped = base_action + float(residual_scale) * residual_action
    return unclipped.clamp(-1.0, 1.0), unclipped


def update_residual_sac(
    *,
    residual_actor,
    critic,
    target_critic,
    actor_optimizer,
    critic_optimizer,
    temperature_optimizer,
    log_temperature: torch.Tensor,
    batch: dict[str, torch.Tensor],
    residual_scale: float,
    gamma: float,
    tau: float,
    reward_scale: float,
    target_entropy: float,
    update_actor: bool,
    max_grad_norm: float = 10.0,
) -> dict[str, float]:
    temperature = log_temperature.exp()
    with torch.no_grad():
        next_distribution = residual_actor.distribution(batch["next_feature"])
        next_residual, next_pre_tanh = next_distribution.rsample()
        next_log_prob = next_distribution.log_prob(next_residual, next_pre_tanh)
        next_action, _ = combined_action(
            batch["next_feature"], next_residual, residual_scale
        )
        next_q = target_critic.minimum(batch["next_feature"], next_action)
        target = sac_target(
            batch["reward"],
            batch["terminated"],
            next_q,
            next_log_prob,
            temperature.detach(),
            gamma,
            reward_scale,
        )

    q1, q2 = critic(batch["feature"], batch["action"])
    critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
    critic_optimizer.step()
    soft_update(target_critic, critic, tau)

    actor_loss_value = 0.0
    temperature_loss_value = 0.0
    residual_norm_value = 0.0
    saturation_value = 0.0
    log_prob_value = 0.0
    if update_actor:
        distribution = residual_actor.distribution(batch["feature"])
        residual, pre_tanh = distribution.rsample()
        log_prob = distribution.log_prob(residual, pre_tanh)
        action, unclipped = combined_action(batch["feature"], residual, residual_scale)
        # Critic parameters are fixed for the policy step; gradients still flow
        # through Q with respect to the residual action.
        critic.requires_grad_(False)
        policy_q = critic.minimum(batch["feature"], action)
        actor_loss = (temperature.detach() * log_prob - policy_q).mean()
        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(residual_actor.parameters(), max_grad_norm)
        actor_optimizer.step()
        critic.requires_grad_(True)

        temperature_loss = -(
            log_temperature * (log_prob.detach() + float(target_entropy))
        ).mean()
        temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        temperature_optimizer.step()

        actor_loss_value = float(actor_loss.detach())
        temperature_loss_value = float(temperature_loss.detach())
        residual_norm_value = float(torch.linalg.vector_norm(residual.detach(), dim=-1).mean())
        saturation_value = float((unclipped.detach().abs() > 1.0).float().mean())
        log_prob_value = float(log_prob.mean().detach())

    values = torch.cat([q1.detach(), q2.detach()])
    metrics = {
        "critic_loss": float(critic_loss.detach()),
        "q_mean": float(values.mean()),
        "q_std": float(values.std()),
        "q_target_mean": float(target.mean()),
        "q_target_std": float(target.std()),
        "actor_loss": actor_loss_value,
        "temperature_loss": temperature_loss_value,
        "temperature": float(log_temperature.exp().detach()),
        "residual_l2": residual_norm_value,
        "combined_action_saturation_rate": saturation_value,
        "residual_log_prob": log_prob_value,
    }
    if not np.all(np.isfinite(list(metrics.values()))) or not torch.isfinite(target).all():
        raise FloatingPointError(f"Non-finite Residual SAC update: {metrics}")
    return metrics
