import torch

from src.models.actor import SquashedGaussianActor


def test_actor_action_range_and_finite_log_prob():
    actor = SquashedGaussianActor(line="oracle")
    distribution = actor.distribution(actor=torch.randn(32, 45))
    action, pre_tanh = distribution.rsample()
    log_prob = distribution.log_prob(action, pre_tanh)
    assert action.shape == (32, 24)
    assert torch.isfinite(action).all()
    assert (action >= -1.0).all() and (action <= 1.0).all()
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(distribution.log_prob(torch.ones_like(action))).all()


def test_deterministic_action_is_tanh_mean():
    actor = SquashedGaussianActor(line="oracle")
    observation = torch.randn(4, 45)
    distribution = actor.distribution(actor=observation)
    assert torch.allclose(actor.act(actor=observation, deterministic=True), torch.tanh(distribution.mean))


def test_pre_tanh_log_prob_preserves_ratio_for_saturated_actions():
    actor = SquashedGaussianActor(line="oracle")
    with torch.no_grad():
        actor.mean_head.bias.fill_(20.0)
    distribution = actor.distribution(actor=torch.zeros(8, 45))
    action, pre_tanh = distribution.rsample()
    assert (action == 1.0).any()
    old_log_prob = distribution.log_prob(action, pre_tanh)
    new_log_prob = actor.distribution(actor=torch.zeros(8, 45)).log_prob(
        action, pre_tanh
    )
    assert torch.allclose((new_log_prob - old_log_prob).exp(), torch.ones(8))


def test_identical_squashed_gaussians_have_zero_exact_kl():
    actor = SquashedGaussianActor(line="oracle")
    inputs = {"actor": torch.randn(8, 45)}
    first = actor.distribution(**inputs)
    second = actor.distribution(**inputs)
    kl = torch.distributions.kl_divergence(first.base, second.base).sum(dim=-1)
    assert torch.equal(kl, torch.zeros_like(kl))
