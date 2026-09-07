from __future__ import annotations

import numpy as np

from src.counterfactual.candidate_branches import label_candidate
from src.counterfactual.candidate_ranker import (
    SelectorThreshold,
    VisualCandidateRanker,
    deployable_input_keys,
    select_candidate,
)
from src.counterfactual.ranker_dataset import assert_source_split_isolated


def test_candidate_source_trajectory_split_has_no_leakage():
    features = [
        {"source_episode_id": 10, "split": "train"},
        {"source_episode_id": 10, "split": "train"},
        {"source_episode_id": 11, "split": "dev"},
    ]
    assert_source_split_isolated(features)


def test_visual_ranker_declares_only_deployable_inputs():
    model = VisualCandidateRanker(352, candidate_count=17)
    assert not model.uses_privileged_state
    assert set(deployable_input_keys()) == {
        "frozen_visual_latent_history", "proprio_history",
        "previous_base_actions", "previous_executed_actions",
    }


def test_incomplete_completion_label_is_masked_not_negative():
    outcome = {"strict": False, "benchmark": False, "goal_exit_after_root": False,
               "max_goal_streak": 0, "goal_steps": 0, "steps": 30}
    candidate = {"outcome": outcome, "completion_mask": False}
    zero = {"outcome": outcome, "completion_mask": True}
    label = label_candidate(candidate, zero, {
        "streak_significant_delta": 3, "goal_steps_significant_delta": 3})
    assert label["mask"] is False


def test_selector_returns_exact_zero_when_no_candidate_is_legal():
    selected = select_candidate(
        np.asarray([0.0, 0.79, 0.95]),
        np.asarray([0.0, 0.01, 0.06]),
        SelectorThreshold(gain=0.80, harm=0.05),
    )
    assert selected == 0
