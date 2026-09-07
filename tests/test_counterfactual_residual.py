from __future__ import annotations

import copy

import numpy as np
import torch

from src.counterfactual.branch_dataset import split_roots_by_source_episode
from src.counterfactual.branch_rollout import (
    clone_observation,
    deterministic_action,
    verify_zero_branches,
)
from src.data.offline import NormalizationStats
from src.envs.adroit import VisualAdroitEnv
from src.models.actor import SquashedGaussianActor
from src.models.gated_residual_student import TemporalGatedVisualResidual
from src.models.safe_residual import headroom_action


def _stats() -> NormalizationStats:
    return NormalizationStats(
        oracle_mean=[0.0] * 45,
        oracle_std=[1.0] * 45,
        proprio_mean=[0.0] * 48,
        proprio_std=[1.0] * 48,
    )


def test_zero_branch_restore_reproduces_deterministic_vision_continuation():
    torch.manual_seed(9)
    actor = SquashedGaussianActor(line="vision").eval()
    device = torch.device("cpu")
    stats = _stats()
    env = VisualAdroitEnv()
    observation, _ = env.reset(seed=7_654_321)
    for _ in range(3):
        action = deterministic_action(actor, "vision", observation, stats, device)
        observation, *_ = env.step(action)
    root_state = copy.deepcopy(env.training_state())
    root_observation = clone_observation(observation)
    verification = []
    for _ in range(3):
        action = deterministic_action(actor, "vision", observation, stats, device)
        next_observation, reward, terminated, truncated, info = env.step(action)
        verification.append(
            {
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "next_observation": clone_observation(next_observation),
                "official_goal": bool(info["official_goal"]),
                "dropped": bool(info["dropped"]),
            }
        )
        observation = next_observation
    env.close()
    summary = verify_zero_branches(
        [
            {
                "state": root_state,
                "source_observation": root_observation,
                "source_verification": verification,
            }
        ],
        actor,
        stats,
        device,
    )
    assert summary["pass"]
    assert summary["action_max_abs"] < 1e-6
    assert summary["reward_max_abs"] < 1e-9
    assert summary["next_rgb_max_abs"] == 0.0


def test_closed_gate_is_exact_frozen_awac_action():
    torch.manual_seed(3)
    base = SquashedGaussianActor(line="vision")
    model = TemporalGatedVisualResidual(base, initial_gate_probability=0.05)
    inputs = {
        "rgb": torch.randint(0, 256, (2, 4, 3, 84, 84), dtype=torch.uint8),
        "proprio": torch.randn(2, 48),
        "previous_action": torch.rand(2, 24).mul(2).sub(1),
        "previous_base_action": torch.zeros(2, 24),
        "previous_executed_action": torch.zeros(2, 24),
    }
    action, info = model.action_and_info(**inputs)
    assert not info["gate"].any()
    assert torch.equal(action, info["base_action"])
    assert torch.equal(info["correction"], torch.zeros_like(info["correction"]))


def test_counterfactual_headroom_action_is_naturally_bounded():
    base = torch.rand(2048, 24).mul(2).sub(1)
    z = torch.rand_like(base).mul(2).sub(1)
    action, info = headroom_action(base, z, rho=0.05)
    assert (action >= -1.0).all() and (action <= 1.0).all()
    assert not info["numerical_clamp"].any()


def test_branch_train_dev_split_is_source_episode_isolated():
    roots = [
        {"root_id": f"root_{index}", "source_episode_seed": seed}
        for index, seed in enumerate([10, 10, 11, 12, 12, 13, 14, 15])
    ]
    train, dev = split_roots_by_source_episode(
        roots, dev_fraction=0.25, split_seed=17
    )
    train_sources = {root["source_episode_seed"] for root in train}
    dev_sources = {root["source_episode_seed"] for root in dev}
    assert train_sources
    assert dev_sources
    assert train_sources.isdisjoint(dev_sources)
