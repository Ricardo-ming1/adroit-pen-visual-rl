from __future__ import annotations

import copy

import numpy as np
import torch

from src.algorithms.residual_sac import (
    combined_action,
    sac_target,
    transition_next_observation,
    update_residual_sac,
)
from src.models.actor import SquashedGaussianActor
from src.models.critic import DoubleQCritic
from src.models.residual import FrozenAWACResidualPolicy, ResidualGaussianActor


def _policy() -> FrozenAWACResidualPolicy:
    base = SquashedGaussianActor(line="vision")
    residual = ResidualGaussianActor(input_dim=256 + 48 + 24)
    return FrozenAWACResidualPolicy(base, residual, residual_scale=0.2)


def _inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "rgb": torch.randint(0, 256, (batch_size, 4, 3, 84, 84), dtype=torch.uint8),
        "proprio": torch.randn(batch_size, 48),
        "previous_action": torch.randn(batch_size, 24).clamp(-1.0, 1.0),
    }


def test_zero_residual_is_exact_frozen_awac_action():
    policy = _policy()
    inputs = _inputs()
    with torch.no_grad():
        base = policy.base_actor.act(**inputs, deterministic=True)
        combined = policy.act(**inputs, deterministic=True)
    assert torch.equal(combined, base)
    assert float((combined - base).abs().max()) < 1e-6


def test_base_encoder_actor_and_normalizer_stay_frozen_after_updates():
    torch.manual_seed(3)
    policy = _policy()
    base_before = {
        key: value.detach().clone() for key, value in policy.base_actor.state_dict().items()
    }
    normalizer = {"mean": [1.0, 2.0], "std": [3.0, 4.0]}
    normalizer_before = copy.deepcopy(normalizer)
    critic = DoubleQCritic(state_dim=328)
    target = critic.target_copy()
    actor_optimizer = torch.optim.Adam(policy.residual_actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    log_temperature = torch.tensor(-2.3, requires_grad=True)
    temperature_optimizer = torch.optim.Adam([log_temperature], lr=1e-4)
    batch = {
        "feature": torch.randn(32, 328),
        "next_feature": torch.randn(32, 328),
        "action": torch.rand(32, 24).mul(2).sub(1),
        "reward": torch.randn(32),
        "terminated": torch.zeros(32, dtype=torch.uint8),
        "truncated": torch.ones(32, dtype=torch.uint8),
    }
    batch["feature"][:, -24:] = batch["feature"][:, -24:].tanh()
    batch["next_feature"][:, -24:] = batch["next_feature"][:, -24:].tanh()
    for _ in range(3):
        metrics = update_residual_sac(
            residual_actor=policy.residual_actor,
            critic=critic,
            target_critic=target,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            temperature_optimizer=temperature_optimizer,
            log_temperature=log_temperature,
            batch=batch,
            residual_scale=0.2,
            gamma=0.99,
            tau=0.005,
            reward_scale=0.02,
            target_entropy=-6.0,
            update_actor=True,
        )
        assert np.isfinite(list(metrics.values())).all()
    assert not policy.base_actor.training
    assert all(not parameter.requires_grad for parameter in policy.base_actor.parameters())
    for key, value in policy.base_actor.state_dict().items():
        assert torch.equal(value, base_before[key])
    assert normalizer == normalizer_before


def test_termination_truncation_bootstrap_and_final_observation():
    target = sac_target(
        reward=torch.zeros(2),
        terminated=torch.tensor([1, 0], dtype=torch.uint8),
        next_q=torch.ones(2),
        residual_log_prob=torch.zeros(2),
        temperature=torch.tensor(0.1),
        gamma=0.99,
        reward_scale=0.02,
    )
    assert torch.allclose(target, torch.tensor([0.0, 0.99]))
    autoreset_observation = {"value": "reset"}
    final_observation = {"value": "final"}
    assert transition_next_observation(
        autoreset_observation, False, True, {"final_observation": final_observation}
    ) is final_observation
    assert transition_next_observation(
        autoreset_observation, False, False, {}
    ) is autoreset_observation


def test_combined_actions_are_bounded_and_residual_log_prob_is_finite():
    actor = ResidualGaussianActor(input_dim=328)
    feature = torch.randn(64, 328)
    feature[:, -24:] = torch.rand(64, 24).mul(2).sub(1)
    distribution = actor.distribution(feature)
    residual, pre_tanh = distribution.rsample()
    action, _ = combined_action(feature, residual, residual_scale=0.4)
    assert torch.isfinite(distribution.log_prob(residual, pre_tanh)).all()
    assert torch.isfinite(action).all()
    assert (action >= -1.0).all() and (action <= 1.0).all()
