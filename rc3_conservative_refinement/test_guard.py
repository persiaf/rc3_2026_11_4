"""Synthetic-only guard tests: no optimizer.step, weights, datasets or planner."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import torch

from .guard import GuardLimits, check_update, restore_if_rejected


class GuardTests(unittest.TestCase):
    def test_hand_calculated_kl_probability_and_tv(self):
        old = torch.tensor([[math.log(0.75), math.log(0.25)]], dtype=torch.float64)
        new = torch.tensor([[math.log(0.5), math.log(0.5)]], dtype=torch.float64)
        report = check_update(old, old, new, torch.ones_like(old, dtype=torch.bool),
                              GuardLimits(1.0, 1.0, 1.0))
        self.assertTrue(report['accepted'])
        self.assertAlmostEqual(report['maxima']['max_step_kl'],
                               0.75 * math.log(1.5) + 0.25 * math.log(0.5), places=12)
        self.assertAlmostEqual(report['maxima']['max_step_probability_change'], 0.25, places=12)
        self.assertAlmostEqual(report['maxima']['max_base_total_variation'], 0.25, places=12)
        json.dumps(report, allow_nan=False)

    def test_one_legal_action_ignores_illegal_nonfinite_logits(self):
        base = torch.tensor([[4.0, float('nan'), float('-inf')]])
        old = torch.tensor([[-3.0, float('inf'), -9.0]])
        new = torch.tensor([[20.0, 1000.0, float('nan')]])
        report = check_update(base, old, new, torch.tensor([[True, False, False]]))
        self.assertTrue(report['accepted'])
        self.assertEqual(report['maxima'], dict(max_step_kl=0.0,
                         max_step_probability_change=0.0, max_base_total_variation=0.0))
        self.assertEqual(report['greedy_changes_old_to_new'], 0)
        self.assertEqual(report['legal_candidate_count'], 1)

    def test_one_outlier_is_not_hidden_by_pooled_mean(self):
        old = torch.zeros((1001, 2), dtype=torch.float64)
        new = old.clone()
        new[-1] = torch.tensor([1.0, -1.0])
        report = check_update(old, old, new, torch.ones_like(old, dtype=torch.bool))
        self.assertFalse(report['accepted'])
        self.assertLess(report['maxima']['max_step_kl'] / 1001, GuardLimits().max_step_kl)
        self.assertEqual(report['rejected_state_indices']['max_step_kl'], [1000])
        self.assertEqual(report['rejected_state_indices']['max_step_probability_change'], [1000])
        self.assertEqual(report['rejected_state_indices']['max_base_total_variation'], [1000])

    def test_probability_bound_is_checked_independently(self):
        old = torch.zeros((1, 2), dtype=torch.float64)
        new = torch.tensor([[0.1, 0.0]], dtype=torch.float64)
        report = check_update(old, old, new, torch.ones_like(old, dtype=torch.bool),
                              GuardLimits(1.0, 0.01, 1.0))
        self.assertFalse(report['accepted'])
        self.assertEqual(report['rejected_state_indices']['max_step_kl'], [])
        self.assertEqual(report['rejected_state_indices']['max_step_probability_change'], [0])

    def test_base_anchor_is_checked_even_when_step_is_zero(self):
        base = torch.zeros((1, 2), dtype=torch.float64)
        current = torch.tensor([[2.0, -2.0]], dtype=torch.float64)
        report = check_update(base, current, current, torch.ones_like(base, dtype=torch.bool))
        self.assertFalse(report['accepted'])
        self.assertEqual(report['maxima']['max_step_kl'], 0.0)
        self.assertEqual(report['maxima']['max_step_probability_change'], 0.0)
        self.assertEqual(report['rejected_state_indices']['max_base_total_variation'], [0])

    def test_near_tie_can_change_greedy_despite_acceptance(self):
        old = torch.tensor([[0.0, 1e-7]], dtype=torch.float64)
        new = torch.tensor([[1e-7, 0.0]], dtype=torch.float64)
        report = check_update(old, old, new, torch.ones_like(old, dtype=torch.bool))
        self.assertTrue(report['accepted'])
        self.assertEqual(report['greedy_changes_old_to_new'], 1)
        self.assertEqual(report['greedy_changes_base_to_new'], 1)
        self.assertIn('no unseen-state', report['scope'])

    def test_inputs_are_not_mutated_and_report_has_no_tensors(self):
        logits = torch.tensor([[[0.0, -0.1], [0.2, 0.0]]], requires_grad=True)
        mask = torch.ones_like(logits, dtype=torch.bool)
        before, before_mask = logits.detach().clone(), mask.clone()
        report = check_update(logits, logits, logits, mask)
        self.assertTrue(torch.equal(before, logits.detach()))
        self.assertTrue(torch.equal(before_mask, mask))
        self.assertIsNone(logits.grad)
        self.assertEqual(report['state_count'], 2)
        self.assertEqual(report['optimizer_steps_executed'], 0)
        json.dumps(report, allow_nan=False)

    def test_malformed_inputs_fail_closed(self):
        z = torch.zeros((2, 2))
        m = torch.ones_like(z, dtype=torch.bool)
        cases = [
            (z, z, z, torch.zeros_like(m)),
            (z, z, z, m.float()),
            (z, z[:, :1], z, m),
            (z.long(), z, z, m),
            (torch.empty((0, 2)),) * 3 + (torch.empty((0, 2), dtype=torch.bool),),
            (torch.empty((2, 0)),) * 3 + (torch.empty((2, 0), dtype=torch.bool),),
            (torch.tensor(0.0),) * 3 + (torch.tensor(True),),
        ]
        for arguments in cases:
            with self.subTest(shapes=[getattr(a, 'shape', None) for a in arguments]):
                with self.assertRaises(ValueError):
                    check_update(*arguments)
        for which in range(3):
            values = [z.clone(), z.clone(), z.clone()]
            values[which][0, 0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'legal-action logits'):
                check_update(*values, m)

    def test_candidate_limits_require_finite_nonnegative_numbers(self):
        for limits in [dict(max_step_kl=-1.0), dict(max_step_kl=float('nan')),
                       dict(max_step_kl=True), dict(max_base_total_variation=1.1),
                       dict(max_step_probability_change=float('inf'))]:
            with self.subTest(limits=limits):
                with self.assertRaises(ValueError):
                    GuardLimits(**limits)

    def test_rejection_restores_head_and_adam_without_any_step(self):
        head = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(head.parameters(), lr=3e-5)
        # Synthetic Adam buffers are installed directly, without performing updates.
        for parameter in head.parameters():
            optimizer.state[parameter] = dict(step=torch.tensor(7.0),
                exp_avg=torch.full_like(parameter, 0.1),
                exp_avg_sq=torch.full_like(parameter, 0.2))
        before_head = copy.deepcopy(head.state_dict())
        before_opt = copy.deepcopy(optimizer.state_dict())
        immutable_head, immutable_opt = copy.deepcopy(before_head), copy.deepcopy(before_opt)
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.add_(3.0)
                optimizer.state[parameter]['exp_avg'].add_(9.0)
        optimizer.param_groups[0]['lr'] = 0.7
        with patch.object(optimizer, 'step', side_effect=AssertionError('optimizer.step is forbidden')) as forbidden:
            result = restore_if_rejected(head, optimizer, before_head, before_opt, dict(accepted=False))
            forbidden.assert_not_called()
        from .guard import _state_equal
        self.assertTrue(result['restored'])
        self.assertEqual(result['optimizer_steps_executed'], 0)
        self.assertTrue(_state_equal(head.state_dict(), before_head))
        self.assertTrue(_state_equal(optimizer.state_dict(), before_opt))
        self.assertTrue(_state_equal(before_head, immutable_head))
        self.assertTrue(_state_equal(before_opt, immutable_opt))

    def test_accepted_report_does_not_restore(self):
        head = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(head.parameters())
        before = copy.deepcopy(head.state_dict())
        result = restore_if_rejected(head, optimizer, {}, {}, dict(accepted=True))
        self.assertFalse(result['restored'])
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in head.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
