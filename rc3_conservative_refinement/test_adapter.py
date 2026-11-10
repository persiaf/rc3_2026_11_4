"""Synthetic feature integration without navigation, scene loading or optimizers."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from torch import nn

from .adapter import LookaheadResidualActor, augment_candidate_features, pack_candidate_states
from .costs import remaining_cost_targets
from .model import remaining_cost_loss


class EncoderFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(1, 1)

    def encode(self, problem):
        return {}

    def features(self, env, encoded):
        return dict(x=torch.zeros(3, 202), z0=torch.tensor([.3, .2, -.5]),
                    mask=torch.as_tensor(env.mask()), v=torch.zeros(268), key=('synthetic-problem', (), 'base', 'D'))


class AdapterTests(unittest.TestCase):
    def fixture(self):
        distances = np.array([[0., 1., 2., 3.], [1., 0., 2., 5.],
                              [2., 2., 0., 4.], [3., 5., 4., 0.]])
        p = SimpleNamespace(identity='synthetic-problem', D=distances, C=np.array([[1, 0, 0], [0, 1, 1]], dtype=bool),
                            candidate_ids=np.arange(3))
        return SimpleNamespace(problem=p, remaining=np.array([1, 1], dtype=bool),
                               mask=lambda: np.ones(3, dtype=bool))

    def test_frozen_base_exact_logits_and_new_dimensions(self):
        actor = LookaheadResidualActor(EncoderFixture())
        actor.train()
        self.assertFalse(actor.encoder.training)
        self.assertTrue(all(not p.requires_grad for p in actor.encoder.parameters()))
        env = self.fixture()
        f = actor.features(env, actor.encode(env.problem))
        self.assertEqual(f['x'].shape, (3, 206))
        self.assertTrue(torch.equal(actor.logits(f), f['z0']))
        self.assertAlmostEqual(float(f['x'][0, 202]), .2, places=6)

    def test_paired_conditions_only_remove_two_distance_summaries(self):
        on = LookaheadResidualActor(EncoderFixture(), distances_enabled=True)
        off = LookaheadResidualActor(EncoderFixture(), distances_enabled=False)
        off.load_state_dict(copy.deepcopy(on.state_dict()))
        env = self.fixture()
        a, b = on.features(env, on.encode(env.problem)), off.features(env, off.encode(env.problem))
        self.assertTrue(torch.equal(a['x'][:, :202], b['x'][:, :202]))
        self.assertTrue(torch.equal(a['x'][:, 204:], b['x'][:, 204:]))
        self.assertTrue(torch.equal(b['x'][:, 202:204], torch.zeros(3, 2)))
        self.assertTrue(torch.equal(on.logits(a), off.logits(b)))
        self.assertTrue(torch.equal(a['mask'], b['mask']))

    def test_complete_targets_fit_combined_H_value_without_changing_objective(self):
        target = remaining_cost_targets([2., 3., 1.], final_J=6., task_count=3)
        predicted = dict(raw_remaining_L=torch.tensor(target['raw_remaining_L']),
                         raw_remaining_K=torch.tensor(target['raw_remaining_K']),
                         terminal_H_credit=torch.full((2,), target['H_gain']))
        value = -(predicted['raw_remaining_L'] + .75 * predicted['raw_remaining_K']
                  - predicted['terminal_H_credit']) / 3
        self.assertTrue(torch.allclose(value, torch.tensor(target['returns'])))
        self.assertTrue(torch.equal(remaining_cost_loss(predicted, target, 3), torch.zeros(2)))

    def test_package_retains_condition_and_excludes_critic(self):
        actor = LookaheadResidualActor(EncoderFixture(), distances_enabled=False)
        p = actor.actor_only_package(dict(path='synthetic', sha256='0' * 64))
        self.assertFalse(p['distances_enabled'])
        self.assertFalse(p['critic_included'])
        self.assertEqual(p['input_dim'], 206)

    def test_bad_feature_order_or_dimension_rejected(self):
        with self.assertRaises(ValueError):
            augment_candidate_features(torch.zeros(3, 201), {'finite_features': np.zeros((3, 4))})

    def test_mixed_feature_condition_cache_is_rejected(self):
        on = LookaheadResidualActor(EncoderFixture(), distances_enabled=True)
        off = LookaheadResidualActor(EncoderFixture(), distances_enabled=False)
        env = self.fixture()
        a, b = on.features(env, on.encode(env.problem)), off.features(env, off.encode(env.problem))
        with self.assertRaises(ValueError):
            pack_candidate_states([a, b])
        self.assertEqual(pack_candidate_states([a, a])['x'].shape, (2, 3, 206))

    def test_provider_preparation_once_per_encode_and_foreign_problem_rejected(self):
        from .adapter import prepare_service_distances
        actor = LookaheadResidualActor(EncoderFixture())
        env = self.fixture()
        with patch('rc3_conservative_refinement.adapter.prepare_service_distances', wraps=prepare_service_distances) as prepare:
            encoded = actor.encode(env.problem)
            actor.features(env, encoded)
            env.remaining[0] = False
            actor.features(env, encoded)
            self.assertEqual(prepare.call_count, 1)
        env.problem.identity = 'different-problem'
        with self.assertRaises(ValueError):
            actor.features(env, encoded)


if __name__ == '__main__':
    unittest.main()
