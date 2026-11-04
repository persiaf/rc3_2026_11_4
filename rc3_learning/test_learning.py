"""Checks for semantics, padding, gradients, reward accounting and checkpoint replay."""
import argparse
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rc3_learning.data import collate, make_instance, plan_from_actions
from rc3_learning.env import ReachabilityCoverRoutingEnv
from rc3_learning.evaluate import load_policy
from rc3_learning.model import AttentionPolicy, rollout
from rc3_learning.train import baseline_update_decision, train_epoch


class LearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.small = make_instance(2, 2, 4)
        cls.large = make_instance(4, 3, 7)
        cls.empty = make_instance(1, 1, 0, counts=[0])

    def setUp(self):
        torch.manual_seed(123)
        self.env = ReachabilityCoverRoutingEnv(seed=123)
        self.policy = AttentionPolicy(32, 1, 4)

    def test_teacher_replay_removes_zero_gain_and_preserves_cost(self):
        original = self.small.teacher['visit_order_candidate_ids']
        redundant = original + original[:1]
        plan = plan_from_actions(self.small, redundant, prune=True)
        self.assertEqual(plan['visit_order_candidate_ids'], original)
        self.assertAlmostEqual(plan['objective_length_plus_stops_m'], self.small.teacher['objective_length_plus_stops_m'])
        with self.assertRaisesRegex(ValueError, 'zero new coverage'):
            plan_from_actions(self.small, redundant)

    def test_mixed_batch_teacher_cost_reward_and_export_agree(self):
        batch = collate([self.small, self.large, self.empty])
        result = rollout(self.policy, self.env, batch, 'teacher')
        expected = torch.tensor([i.teacher['objective_length_plus_stops_m'] for i in (self.small, self.large, self.empty)])
        torch.testing.assert_close(result['cost'], expected, atol=2e-4, rtol=1e-5)
        torch.testing.assert_close(result['reward'], -expected/batch['reference_cost'])
        self.assertTrue(result['state']['completed'].all())
        self.env.check_solution_validity(result['state'], result['actions'])
        self.assertTrue((result['actions'][2] == -1).all())

    def test_single_stop_return_has_no_second_docking_charge(self):
        item = make_instance(1, 1, 2, counts=[1])
        td = self.env.reset(collate([item]))
        action = td['action_mask'][0].nonzero()[0, 0]
        td['action'] = action.reshape(1)
        td = self.env.step(td)['next']
        j = int(action)+1
        expected = item.navigation['costs'][0, j]+item.navigation['costs'][j, 0]+item.penalty
        self.assertAlmostEqual(float(td['total_cost'][0]), expected, places=5)
        self.assertTrue(td['completed'].all())
        td['action'] = torch.tensor([0])
        after = self.env.step(td)['next']
        self.assertAlmostEqual(float(after['total_cost'][0]), expected, places=5)
        self.assertEqual(float(after['reward'][0]), 0.)

    def test_infeasible_batch_is_failed_not_completed(self):
        item = make_instance(1, 1, 0, counts=[1], robot={'arm_max_reach': 0.2}, require_feasible=False)
        result = rollout(self.policy, self.env, collate([item]))
        self.assertTrue(result['state']['failed'].all())
        self.assertFalse(result['state']['completed'].any())
        self.assertTrue(result['state']['remaining'].any())
        with self.assertRaises(ValueError):
            self.env._get_reward(result['state'], result['actions'])

    def test_invalid_action_is_rejected(self):
        td = self.env.reset(collate([self.small]))
        td['action'] = torch.tensor([td['candidate_mask'].shape[1]])
        with self.assertRaises(ValueError):
            self.env.step(td)

    def test_missing_teacher_label_is_not_replaced_by_candidate_zero(self):
        batch = collate([self.small])
        batch['teacher_actions'][0, 0] = -1
        with self.assertRaisesRegex(ValueError, 'missing teacher action'):
            rollout(self.policy, self.env, batch, 'teacher')

    def test_each_active_action_strictly_reduces_remaining_tasks(self):
        td = self.env.reset(collate([self.small, self.large]))
        steps = 0
        while not td['done'].all():
            before = td['remaining'].sum(1)
            active = ~td['done'].squeeze(-1)
            td['action'] = td['action_mask'].long().argmax(1)
            td = self.env.step(td)['next']
            self.assertTrue((td['remaining'].sum(1)[active] < before[active]).all())
            steps += 1
            self.assertLessEqual(steps, len(self.large.scene['tasks']))

    def test_padding_cannot_change_real_candidate_logits(self):
        self.policy.eval()
        single = self.env.reset(collate([self.small]))
        mixed = self.env.reset(collate([self.small, self.large, self.empty]))
        size = single['candidate_mask'].shape[1]
        logits_single = self.policy(single, self.policy.encode(single))[0]
        logits_mixed = self.policy(mixed, self.policy.encode(mixed))[0, :size]
        torch.testing.assert_close(logits_single, logits_mixed, atol=3e-5, rtol=1e-5)
        poisoned = mixed.clone()
        poisoned['task_features'][~poisoned['task_mask']] = 10000.
        poisoned['node_features'][~poisoned['candidate_mask']] = -10000.
        logits_poisoned = self.policy(poisoned, self.policy.encode(poisoned))[0, :size]
        torch.testing.assert_close(logits_single, logits_poisoned, atol=3e-5, rtol=1e-5)
        self.assertTrue(torch.isneginf(self.policy(mixed, self.policy.encode(mixed))[0, size:]).all())

    def test_true_navigation_distance_changes_logits(self):
        td = self.env.reset(collate([self.small]))
        old = self.policy(td, self.policy.encode(td)).detach()
        j = int(td['action_mask'][0].nonzero()[0, 0])
        changed = td.clone()
        changed['costs'][0, 0, j+1] += 10.
        new = self.policy(changed, self.policy.encode(changed)).detach()
        self.assertFalse(torch.allclose(old[:, j], new[:, j], atol=1e-6))

    def test_inference_does_not_require_or_read_teacher(self):
        item = make_instance(2, 2, 4, with_teacher=False)
        with torch.no_grad():
            with_teacher = rollout(self.policy, self.env, collate([self.small]))
            without_teacher = rollout(self.policy, self.env, collate([item]))
        torch.testing.assert_close(with_teacher['actions'], without_teacher['actions'])
        torch.testing.assert_close(with_teacher['cost'], without_teacher['cost'])

    def test_bc_gradient_is_finite_and_step_reduces_same_batch_loss(self):
        batch = collate([self.small, self.large, self.empty])
        before = rollout(self.policy, self.env, batch, 'teacher')
        loss = -before['log_likelihood'].sum()/before['state']['service_count'].sum()
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.policy.parameters() if p.grad is not None))
        optimizer.step()
        after = rollout(self.policy, self.env, batch, 'teacher')
        next_loss = -after['log_likelihood'].sum()/after['state']['service_count'].sum()
        self.assertLess(float(next_loss.detach()), float(loss.detach()))

    def test_reinforce_updates_policy_and_keeps_rollout_baseline_frozen(self):
        frozen = copy.deepcopy(self.policy).eval()
        for p in frozen.parameters():
            p.requires_grad_(False)
        before = {k: v.clone() for k, v in frozen.state_dict().items()}
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        result = train_epoch(self.policy, frozen, self.env, [self.small, self.large], optimizer,
                             argparse.Namespace(batch_size=2), np.random.default_rng(0), 'cpu', 'rl')
        self.assertEqual(result['updates'], 1)
        self.assertTrue(np.isfinite(result['loss']))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in frozen.state_dict().items()))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in self.policy.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in frozen.parameters()))

    def test_baseline_update_uses_one_sided_held_out_comparison(self):
        self.assertFalse(baseline_update_decision([2., 2.], [1., 1.])[0])
        self.assertTrue(baseline_update_decision([1., 1., 1.], [2., 2., 2.])[0])
        self.assertFalse(baseline_update_decision([1.], [2.])[0])

    def test_checkpoint_reload_preserves_actions(self):
        batch = collate([self.small])
        with torch.no_grad():
            before = rollout(self.policy, self.env, batch)
        with tempfile.TemporaryDirectory(prefix='checkpoint_test_', dir=Path(__file__).parent) as folder:
            path = Path(folder)/'model.pt'
            torch.save(dict(model_config=self.policy.config, model_state=self.policy.state_dict()), path)
            restored, _ = load_policy(path, 'cpu')
            with torch.no_grad():
                after = rollout(restored, self.env, batch)
        torch.testing.assert_close(before['actions'], after['actions'])
        torch.testing.assert_close(before['cost'], after['cost'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_teacher_backward(self):
        batch = collate([self.small, self.empty], 'cuda')
        policy = self.policy.to('cuda')
        result = rollout(policy, ReachabilityCoverRoutingEnv('cuda'), batch, 'teacher')
        (-result['log_likelihood'].mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None))
        self.assertTrue(result['state']['completed'].all())


if __name__ == '__main__':
    unittest.main()
