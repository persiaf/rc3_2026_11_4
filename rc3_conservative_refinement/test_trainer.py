"""Loss/admission/checkpoint checks only; no optimizer step or real input read.

Actual accepted/rejected, continuous/resumed optimizer engineering is registered
and run separately by the real-input acceptance entry point.
"""
import copy
import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from .costs import remaining_cost_targets
from .model import RemainingCostCritic, SmallCorrection
from .trainer import (ConservativePPO, PPOConfig, checkpoint_equal,
                      masked_log_softmax, ppo_state_loss, tensor_digest)


class SyntheticActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1).eval().requires_grad_(False)
        self.correction = SmallCorrection(input_dim=206)
        self.distances_enabled = True

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self


class TrainerArithmeticTests(unittest.TestCase):
    def setUp(self):
        # Protect this entire unit suite against accidental optimizer execution.
        self.patches = [patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer in unit suite')),
                        patch.object(torch.optim.SGD, 'step', side_effect=AssertionError('No optimizer in unit suite'))]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def fixture(self):
        actor = SyntheticActor()
        trainer = ConservativePPO(actor, RemainingCostCritic())
        z0 = torch.tensor([[.2, -.3, -torch.inf], [.1, .5, -torch.inf]])
        mask = torch.isfinite(z0)
        batch = dict(x=torch.arange(2 * 3 * 206, dtype=torch.float32).reshape(2, 3, 206) / 1000,
                     v=torch.zeros(2, 268), z0=z0, mask=mask,
                     keys=[('synthetic', (i,), 'base', 'D', 'post-service-v1', True, 'prepared') for i in range(2)])
        target = remaining_cost_targets([2., 1., 3.], final_J=6., task_count=3)
        episode = dict(episode_id='synthetic-opportunity0', batch=batch, task_count=3,
                       actions=torch.tensor([0, 1]), old_logits=z0.clone(), V_old=torch.zeros(2),
                       targets=target, actor_tensor_sha=tensor_digest(actor.correction.state_dict()))
        return trainer, episode

    def test_state_loss_matches_hand_computation_and_clipping(self):
        old = torch.log(torch.tensor([.5, .5]))
        new = torch.log(torch.tensor([[.75, .25], [.25, .75]], requires_grad=True))
        base = torch.log(torch.full((2, 2), .5))
        mask = torch.ones(2, 2, dtype=torch.bool)
        objective, ratios, kl = ppo_state_loss(new, old, torch.tensor([2., -2.]),
            base, mask, torch.tensor([0, 0]), beta=0., ratio_clip=.1)
        self.assertTrue(torch.allclose(ratios, torch.tensor([1.5, .5])))
        self.assertTrue(torch.allclose(objective, torch.tensor([-2.2, 1.8])))
        self.assertTrue(torch.isfinite(kl).all())
        objective.sum().backward()

    def test_illegal_nan_does_not_affect_masked_distribution(self):
        logits = torch.tensor([[0., 1., float('nan')]])
        mask = torch.tensor([[True, True, False]])
        result = masked_log_softmax(logits, mask)
        self.assertTrue(torch.equal(result[:, :2], torch.log_softmax(logits[:, :2], -1)))
        self.assertTrue(torch.isneginf(result[:, 2]).all())

    def test_attached_old_values_and_illegal_actions_rejected(self):
        lp = torch.log_softmax(torch.tensor([[.1, .2]]), -1)
        mask = torch.tensor([[True, False]])
        with self.assertRaises(ValueError):
            ppo_state_loss(lp, torch.tensor([-.4]), torch.ones(1, requires_grad=True),
                           lp, mask, torch.tensor([0]))
        with self.assertRaises(ValueError):
            ppo_state_loss(lp, torch.tensor([-.4]), torch.ones(1), lp, mask, torch.tensor([1]))

    def test_exact_checkpoint_and_rng_restore_without_updates(self):
        trainer, _ = self.fixture()
        metadata = dict(source='synthetic', request_count=0, schedule_cursor=0)
        checkpoint = trainer.state_dict(metadata)
        expected = random.random(), np.random.random(), torch.rand(2)
        trainer.load_state_dict(checkpoint, metadata)
        actual = random.random(), np.random.random(), torch.rand(2)
        self.assertTrue(checkpoint_equal(expected, actual))
        again = copy.deepcopy(checkpoint)
        self.assertTrue(checkpoint_equal(checkpoint, again))
        again['rng']['numpy'][1][0] ^= np.uint32(1)
        self.assertFalse(checkpoint_equal(checkpoint, again))

    def test_source_condition_metadata_or_resource_rewind_rejected(self):
        trainer, _ = self.fixture()
        checkpoint = trainer.state_dict(dict(source='synthetic', request_count=0))
        with self.assertRaises(ValueError):
            trainer.load_state_dict(checkpoint, dict(source='different', request_count=0))
        wrong = copy.deepcopy(checkpoint)
        wrong['distances_enabled'] = False
        with self.assertRaises(ValueError):
            trainer.load_state_dict(wrong, checkpoint['metadata'])
        trainer.counts['actor_optimizer_attempts'] = 1
        with self.assertRaises(ValueError):
            trainer.load_state_dict(checkpoint, checkpoint['metadata'])

    def test_admission_rejects_off_policy_mixed_condition_and_replayed_episodes(self):
        trainer, episode = self.fixture()
        admitted, ids = trainer._admit([episode])
        self.assertEqual(ids, [episode['episode_id']])
        self.assertEqual(len(admitted), 1)
        altered = copy.deepcopy(episode)
        altered['old_logits'][0, 0] += .01
        with self.assertRaises(ValueError):
            trainer._admit([altered])
        altered = copy.deepcopy(episode)
        altered['batch']['keys'][0] = ('synthetic', (), 'base', 'D', 'post-service-v1', False, 'prepared')
        with self.assertRaises(ValueError):
            trainer._admit([altered])
        trainer.used_episode_ids.add(episode['episode_id'])
        with self.assertRaises(ValueError):
            trainer._admit([episode])

    def test_encoder_gradient_or_mode_violation_fails_closed(self):
        trainer, _ = self.fixture()
        trainer.actor.encoder.train()
        with self.assertRaises(RuntimeError):
            trainer.state_dict(dict(source='synthetic'))
        trainer.actor.train()
        next(trainer.actor.encoder.parameters()).requires_grad_(True)
        with self.assertRaises(RuntimeError):
            trainer.state_dict(dict(source='synthetic'))

    def test_numerical_and_fixed_stop_config_validation(self):
        for value in (0., -1., float('nan')):
            with self.assertRaises(ValueError):
                PPOConfig(actor_lr=value)
        with self.assertRaises(ValueError):
            PPOConfig(consecutive_rejection_limit=0)
        with self.assertRaises(ValueError):
            PPOConfig(warm_update_cap=-1)


if __name__ == '__main__':
    unittest.main()
