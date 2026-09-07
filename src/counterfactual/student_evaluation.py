from __future__ import annotations

from typing import Iterable

import numpy as np
import torch

from src.data.offline import NormalizationStats
from src.envs.adroit import HORIZON, VisualAdroitEnv
from src.evaluation import policy_inputs


@torch.inference_mode()
def evaluate_gated_student(
    model,
    stats: NormalizationStats,
    seeds: Iterable[int],
    device: torch.device,
    gate_threshold: float,
) -> dict:
    model.eval()
    env = VisualAdroitEnv()
    episodes = []
    all_corrections = []
    all_probabilities = []
    all_gate_steps = []
    numerical_clamps = 0
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        previous_base = torch.zeros(1, 24, device=device)
        previous_executed = torch.zeros(1, 24, device=device)
        temporal_history = None
        official = []
        dropped = []
        rewards = []
        gate_values = []
        correction_values = []
        goal_at_gate = []
        finite = True
        for step in range(HORIZON):
            inputs = policy_inputs(observation, "vision", stats, device)
            action_tensor, info = model.action_and_info(
                **inputs,
                previous_base_action=previous_base,
                previous_executed_action=previous_executed,
                temporal_history=temporal_history,
                gate_threshold=gate_threshold,
            )
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            finite = finite and bool(np.isfinite(action).all())
            gate = bool(info["gate"].item())
            probability = float(info["gate_probability"].item())
            correction = info["correction"].squeeze(0).cpu().numpy()
            numerical_clamps += int(info["numerical_clamp"].sum())
            observation, reward, terminated, truncated, step_info = env.step(action)
            official.append(bool(step_info["official_goal"]))
            dropped.append(bool(step_info["dropped"]))
            rewards.append(float(reward))
            gate_values.append(gate)
            correction_values.append(correction)
            if gate:
                goal_at_gate.append(bool(step_info["official_goal"]))
                all_gate_steps.append(step)
            all_probabilities.append(probability)
            all_corrections.append(correction)
            temporal_history = info["temporal_history"]
            previous_base = info["base_action"]
            previous_executed = action_tensor
            if terminated or truncated:
                break
        first = next((index for index, value in enumerate(official) if value), None)
        maximum = 0
        streak = 0
        for value in official:
            streak = streak + 1 if value else 0
            maximum = max(maximum, streak)
        exited = bool(first is not None and not all(official[first:]))
        strict = bool(
            len(official) >= 20
            and all(official[-20:])
            and not any(dropped)
            and finite
        )
        correction_array = np.asarray(correction_values)
        episodes.append(
            {
                "seed": int(seed),
                "benchmark_success": bool(any(official)),
                "strict_stable_success": strict,
                "goal_exit_after_entry": exited,
                "max_consecutive_goal_steps": maximum,
                "dropped": bool(any(dropped)),
                "return": float(sum(rewards)),
                "gate_activation_rate": float(np.mean(gate_values)),
                "gate_activations": int(sum(gate_values)),
                "gate_on_goal_rate": float(
                    np.mean(goal_at_gate) if goal_at_gate else 0.0
                ),
                "correction_rms": float(
                    np.sqrt(np.mean(np.square(correction_array)))
                ),
                "finite": finite,
            }
        )
    env.close()
    numeric = lambda key: np.asarray(
        [episode[key] for episode in episodes], dtype=np.float64
    )
    benchmark = numeric("benchmark_success")
    exits = numeric("goal_exit_after_entry")
    corrections = np.asarray(all_corrections, dtype=np.float64)
    return {
        "benchmark_success": float(benchmark.mean()),
        "strict_stable_success": float(numeric("strict_stable_success").mean()),
        "goal_entry_rate": float(benchmark.mean()),
        "goal_exit_after_entry_rate": float(
            exits[benchmark.astype(bool)].mean() if benchmark.any() else 0.0
        ),
        "drop_rate": float(numeric("dropped").mean()),
        "episode_return": float(numeric("return").mean()),
        "gate_activation_rate": float(numeric("gate_activation_rate").mean()),
        "episodes_with_gate": float(
            np.mean(numeric("gate_activations") > 0)
        ),
        "gate_probability_mean": float(np.mean(all_probabilities)),
        "gate_step_mean": float(np.mean(all_gate_steps)) if all_gate_steps else None,
        "correction_rms": float(np.sqrt(np.mean(np.square(corrections)))),
        "correction_rms_max_dim": float(
            np.sqrt(np.mean(np.square(corrections), axis=0)).max()
        ),
        "numerical_clamp_count": int(numerical_clamps),
        "non_finite_episodes": int(sum(not episode["finite"] for episode in episodes)),
        "episodes": episodes,
    }
