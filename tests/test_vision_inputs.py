import pytest
import torch

from src.models.actor import SquashedGaussianActor, actor_inputs


def test_vision_actor_has_only_deployable_inputs():
    actor = SquashedGaussianActor(line="vision")
    assert actor.deployable_low_dim == 48 + 24
    rgb = torch.zeros(2, 4, 3, 84, 84, dtype=torch.uint8)
    proprio = torch.zeros(2, 48)
    previous_action = torch.zeros(2, 24)
    action = actor.act(
        rgb=rgb,
        proprio=proprio,
        previous_action=previous_action,
        deterministic=True,
    )
    assert action.shape == (2, 24)


def test_privileged_state_is_not_forwarded_to_vision_actor():
    batch = {
        "rgb": torch.zeros(1, 4, 3, 84, 84, dtype=torch.uint8),
        "proprio": torch.zeros(1, 48),
        "previous_action": torch.zeros(1, 24),
        "privileged": torch.randn(1, 45),
    }
    inputs = actor_inputs(batch, "vision")
    assert set(inputs) == {"rgb", "proprio", "previous_action"}
    assert "privileged" not in inputs


def test_vision_actor_rejects_wrong_proprio_shape():
    actor = SquashedGaussianActor(line="vision")
    with pytest.raises(ValueError):
        actor.distribution(
            rgb=torch.zeros(1, 4, 3, 84, 84),
            proprio=torch.zeros(1, 69),
            previous_action=torch.zeros(1, 24),
        )
