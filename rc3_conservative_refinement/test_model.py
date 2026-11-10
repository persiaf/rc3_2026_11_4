"""Synthetic tensors only; backward passes are not optimizer updates."""
import copy
import unittest
import torch
from torch import nn

from .model import (SmallCorrection, RemainingCostCritic, freeze_supervised_base,
                    remaining_cost_loss, full_return_advantage)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.Generator().manual_seed(20261110)

    def test_initial_logits_mask_and_zero_output_are_exact(self):
        head = SmallCorrection(input_dim=4, hidden=8)
        base = torch.tensor([[0.2, -0.1, -torch.inf], [3., -torch.inf, -2.]])
        mask = torch.isfinite(base)
        features = torch.randn(2, 3, 4, generator=self.rng)
        features[~mask] = torch.nan
        self.assertTrue(torch.equal(head(base, features, mask), base))

    def test_positive_amplitude_preserves_initial_learning_signal(self):
        base = freeze_supervised_base(nn.Linear(4, 2))
        before = copy.deepcopy(base.state_dict())
        head = SmallCorrection(input_dim=4, hidden=8)
        features = torch.randn(2, 4, generator=self.rng)
        logits = base(features).mean(0)
        z = head(logits, features, torch.ones(2, dtype=torch.bool))
        loss = -torch.log_softmax(z, 0)[0]
        loss.backward()
        self.assertGreater(float(head.head[-1].weight.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in base.state_dict().items()))

    def test_adjustment_bound_after_synthetic_perturbation(self):
        head = SmallCorrection(input_dim=4, hidden=8)
        with torch.no_grad():
            head.head[-1].bias.fill_(100.)
        base = torch.tensor([0.7, -0.9, -torch.inf])
        mask = torch.isfinite(base)
        z = head(base, torch.zeros(3, 4), mask)
        self.assertLessEqual(float((z[mask] - base[mask]).detach().abs().max()), 0.2500001)
        self.assertTrue(torch.isneginf(z[~mask]).all())

    def test_critic_is_independent_and_has_explicit_units(self):
        head = SmallCorrection(input_dim=4, hidden=8)
        critic = RemainingCostCritic(input_dim=5, hidden=8)
        self.assertFalse(set(map(id, head.parameters())) & set(map(id, critic.parameters())))
        output = critic(torch.zeros(2, 5), task_count=4)
        self.assertTrue((output['raw_remaining_L'] >= 0).all())
        self.assertTrue((output['raw_remaining_K'] >= 0).all())
        expected = -(output['raw_remaining_L'] + .75 * output['raw_remaining_K']
                     - output['terminal_H_credit']) / 4
        self.assertTrue(torch.equal(output['value'], expected))

    def test_critic_backward_cannot_update_actor(self):
        head = SmallCorrection(input_dim=4, hidden=8)
        critic = RemainingCostCritic(input_dim=5, hidden=8)
        output = critic(torch.zeros(2, 5), task_count=4)
        target = dict(raw_remaining_L=[3., 2.], raw_remaining_K=[2., 1.],
                      terminal_H_credit=[.2, .2])
        remaining_cost_loss(output, target, 4).sum().backward()
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        self.assertTrue(any(p.grad is not None for p in critic.parameters()))

    def test_better_complete_outcome_has_positive_advantage(self):
        old_value = torch.tensor([-5., -5.], requires_grad=True)
        actual_returns = torch.tensor([-4., -6.], requires_grad=True)
        advantage = full_return_advantage(actual_returns, old_value)
        self.assertTrue(torch.equal(advantage, torch.tensor([1., -1.])))
        self.assertFalse(advantage.requires_grad)

    def test_actor_only_payload_excludes_critic_and_adam(self):
        head = SmallCorrection(input_dim=4, hidden=8)
        package = head.actor_only_package(dict(path='synthetic-base', sha256='0' * 64))
        self.assertFalse(package['critic_included'])
        self.assertFalse(package['optimizer_included'])
        self.assertNotIn('critic_state', package)
        self.assertNotIn('optimizer_state', package)

    def test_bad_mask_and_zero_amplitude_rejected(self):
        with self.assertRaises(ValueError):
            SmallCorrection(max_adjustment=0.)
        head = SmallCorrection(input_dim=4, hidden=8)
        with self.assertRaises(ValueError):
            head(torch.zeros(2), torch.zeros(2, 4), torch.zeros(2, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
