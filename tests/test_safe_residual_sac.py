from __future__ import annotations

import copy

import torch

from src.algorithms.residual_sac import sac_target, transition_next_observation
from src.algorithms.safe_residual_sac import update_safe_residual_sac
from src.models.actor import SquashedGaussianActor
from src.models.critic import DoubleQCritic
from src.models.residual import ResidualGaussianActor
from src.models.safe_residual import SafeFrozenAWACResidualPolicy, headroom_action


def _policy() -> SafeFrozenAWACResidualPolicy:
    base = SquashedGaussianActor(line="vision")
    residual = ResidualGaussianActor(
        input_dim=328,
        initial_log_std=-3.0,
        log_std_min=-3.0,
        log_std_max=-3.0,
    )
    return SafeFrozenAWACResidualPolicy(base, residual, rho=0.05)


def _inputs(batch_size: int = 3) -> dict[str, torch.Tensor]:
    return {
        "rgb": torch.randint(0, 256, (batch_size, 4, 3, 84, 84), dtype=torch.uint8),
        "proprio": torch.randn(batch_size, 48),
        "previous_action": torch.rand(batch_size, 24).mul(2).sub(1),
    }


def test_safe_step_zero_exactly_matches_frozen_awac():
    policy = _policy()
    inputs = _inputs()
    with torch.no_grad():
        base = policy.base_actor.act(**inputs, deterministic=True)
        action, info = policy.action_and_info(deterministic=True, **inputs)
    assert torch.equal(info["residual_latent"], torch.zeros_like(info["residual_latent"]))
    assert torch.equal(info["correction"], torch.zeros_like(info["correction"]))
    assert torch.equal(action, base)


def test_safe_headroom_is_naturally_bounded_without_numerical_clamp():
    base = torch.rand(4096, 24).mul(2).sub(1)
    base[0].fill_(1.0)
    base[1].fill_(-1.0)
    z = torch.rand_like(base).mul(2).sub(1)
    action, info = headroom_action(base, z, rho=0.05)
    assert (info["unprotected_action"] <= 1.0).all()
    assert (info["unprotected_action"] >= -1.0).all()
    assert not info["numerical_clamp"].any()
    assert torch.equal(action, info["unprotected_action"])


def test_safe_updates_leave_awac_encoder_actor_and_normalizer_unchanged():
    policy = _policy()
    base_before = {
        key: value.detach().clone() for key, value in policy.base_actor.state_dict().items()
    }
    normalizer = {"mean": [0.0], "std": [1.0]}
    normalizer_before = copy.deepcopy(normalizer)
    critic = DoubleQCritic(state_dim=328)
    target = critic.target_copy()
    actor_optimizer = torch.optim.Adam(policy.residual_actor.parameters(), lr=3e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    q_scale_ema = torch.ones(())
    batch = {
        "feature": torch.randn(64, 328),
        "next_feature": torch.randn(64, 328),
        "action": torch.rand(64, 24).mul(2).sub(1),
        "reward": torch.randn(64),
        "terminated": torch.zeros(64, dtype=torch.uint8),
        "truncated": torch.ones(64, dtype=torch.uint8),
    }
    batch["feature"][:, -24:] = batch["feature"][:, -24:].tanh()
    batch["next_feature"][:, -24:] = batch["next_feature"][:, -24:].tanh()
    for _ in range(3):
        update_safe_residual_sac(
            residual_actor=policy.residual_actor,
            critic=critic,
            target_critic=target,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            critic_batch=batch,
            actor_batch=batch,
            rho_effective=0.05,
            gamma=0.99,
            tau=0.005,
            reward_scale=0.02,
            temperature=1e-4,
            correction_budget=0.03,
            lambda_anchor=0.05,
            q_scale_ema=q_scale_ema,
            q_scale_decay=0.99,
            update_actor=True,
        )
    assert not policy.base_actor.training
    assert all(not parameter.requires_grad for parameter in policy.base_actor.parameters())
    for key, value in policy.base_actor.state_dict().items():
        assert torch.equal(value, base_before[key])
    assert normalizer == normalizer_before


def test_safe_terminated_truncated_bootstrap_and_final_observation():
    target = sac_target(
        reward=torch.zeros(2),
        terminated=torch.tensor([1, 0], dtype=torch.uint8),
        next_q=torch.ones(2),
        residual_log_prob=torch.zeros(2),
        temperature=torch.tensor(1e-4),
        gamma=0.99,
        reward_scale=0.02,
    )
    assert torch.allclose(target, torch.tensor([0.0, 0.99]))
    reset_observation = {"source": "reset"}
    final_observation = {"source": "final"}
    assert transition_next_observation(
        reset_observation,
        False,
        True,
        {"final_observation": final_observation},
    ) is final_observation
