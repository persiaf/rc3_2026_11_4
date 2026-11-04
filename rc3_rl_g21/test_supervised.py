"""Synthetic guards for G2.1; no environment or formal training calls."""
import unittest

import numpy as np
import torch

from rc3_rl_g21.train_supervised import (
    diagnose, root_half_weights, supervision_loss, toy_gate,
)


def sample(*, scene='s', path=(), route=(0, 1, 0), q=(.5, 0., 0.),
           evidence=(True, True, False), ties=(True, False, False),
           legal=(True, True, True), stop=1):
    q = np.asarray(q, dtype=float)
    return dict(scene_id=scene, path=list(path), current_route=list(route),
                teacher_depth=2 if not path else 1,
                legality_mask=np.asarray(legal, dtype=bool),
                evidence_mask=np.asarray(evidence, dtype=bool),
                ties=np.asarray(ties, dtype=bool), q_values=q, stop_id=stop,
                advantages_vs_stop=q - q[stop], tie_tolerance=.001,
                observation={})


class FixedLogitModel:
    def __init__(self, logits):
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def __call__(self, observations):
        return self.logits.repeat(len(observations), 1), torch.zeros(len(observations))


class SupervisedContractTests(unittest.TestCase):
    def test_unknown_excluded_from_ce_and_has_no_direct_gradient(self):
        logits = torch.tensor([[2., 1., 1000.]], requires_grad=True)
        loss, ce, aux = supervision_loss(logits, [sample()])
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(float(loss.detach()), float(torch.nn.functional.softplus(torch.tensor(-1.))), places=6)
        self.assertEqual(float(aux.detach()), 0.)
        loss.backward()
        self.assertEqual(float(logits.grad[0, 2]), 0.)
        self.assertLess(float(logits.grad[0, 0]), 0.)
        self.assertGreater(float(logits.grad[0, 1]), 0.)
        self.assertEqual(float(loss.detach()), float(ce.detach()))

    def test_uniform_ties_have_log_count_loss_floor(self):
        s = sample(q=(0., 0., 0.), ties=(True, True, False))
        logits = torch.tensor([[3., 3., 1000.]], requires_grad=True)
        loss, _, _ = supervision_loss(logits, [s])
        self.assertAlmostEqual(float(loss.detach()), float(np.log(2.)), places=6)
        loss.backward()
        np.testing.assert_allclose(logits.grad.numpy(), 0., atol=1e-7)

    def test_ties_outside_evidence_rejected(self):
        with self.assertRaises(ValueError):
            supervision_loss(torch.zeros(1, 3), [sample(ties=(False, False, True))])

    def test_root_half_balances_scene_then_successor_routes(self):
        samples = [sample(scene='a'), sample(scene='a', path=(0,)),
                   sample(scene='a', path=(1,)),
                   sample(scene='a', path=(2,), route=(0, 2, 0)),
                   sample(scene='b'), sample(scene='b', path=(0,))]
        weights = root_half_weights(samples)
        np.testing.assert_allclose(weights, [.25, .0625, .0625, .125, .25, .25])
        self.assertAlmostEqual(float(weights[:4].sum()), .5)
        self.assertAlmostEqual(float(weights[4:].sum()), .5)
        self.assertAlmostEqual(float(weights[[0, 4]].sum()), .5)

    def test_root_half_requires_one_root_and_successor(self):
        for samples in ([sample()], [sample(), sample(), sample(path=(0,))],
                        [sample(path=(0,))]):
            with self.assertRaises(ValueError):
                root_half_weights(samples)

    def test_full_unknown_has_null_gain_and_null_regret(self):
        result = diagnose(FixedLogitModel([2., 1., 1000.]), [sample()])
        row = result['rows'][0]
        self.assertTrue(row['full_greedy_unknown'])
        self.assertIsNone(row['full_selected_gain'])
        self.assertIsNone(row['observed_full_greedy_Q_gap'])
        self.assertFalse(row['full_known_positive_continue'])
        self.assertTrue(row['evidence_known_positive_continue'])
        self.assertTrue(row['evidence_greedy_near_best_hit'])
        self.assertEqual(result['groups']['roots']['metrics']['observed_full_greedy_Q_gap']['n'], 0)

    def test_full_unknown_placeholder_does_not_supply_quality(self):
        s = sample(q=(.5, 0., 999.))
        row = diagnose(FixedLogitModel([2., 1., 1000.]), [s])['rows'][0]
        self.assertIsNone(row['full_selected_gain'])
        self.assertIsNone(row['observed_full_greedy_Q_gap'])
        self.assertAlmostEqual(row['delta'], .5)

    def test_diagnose_rejects_illegal_finite_logit(self):
        s = sample(legal=(True, True, False))
        with self.assertRaises(ValueError):
            diagnose(FixedLogitModel([2., 1., 0.]), [s])

    def test_gain_auxiliary_only_uses_root_depth2_observed_actions(self):
        root = sample()
        child = sample(path=(0,))
        logits = torch.tensor([[2., 1., 1000.], [2., 1., 1000.]], requires_grad=True)
        total, ce, gain = supervision_loss(logits, [root, child], auxiliary=True)
        self.assertAlmostEqual(float(gain.detach()), .125, places=6)
        total.backward()
        self.assertEqual(float(logits.grad[0, 2]), 0.)
        self.assertEqual(float(logits.grad[1, 2]), 0.)
        root['teacher_depth'] = 1
        with self.assertRaises(ValueError):
            supervision_loss(logits, [root], auxiliary=True)

    @staticmethod
    def gate_rows():
        return [dict(scene_id=str(i), is_root=True, teacher_depth=2,
                     clear_positive=i < 8, clear_stop=i >= 8,
                     evidence_greedy_near_best_hit=True,
                     evidence_known_positive_continue=i < 8) for i in range(16)]

    def test_gate_accepts_prespecified_threshold_boundary(self):
        rows = self.gate_rows()
        rows[0]['evidence_greedy_near_best_hit'] = False
        rows[0]['evidence_known_positive_continue'] = False
        self.assertTrue(toy_gate({'rows': rows})['passed'])

    def test_gate_rejects_below_either_threshold(self):
        rows = self.gate_rows()
        rows[0]['evidence_greedy_near_best_hit'] = False
        rows[1]['evidence_greedy_near_best_hit'] = False
        self.assertFalse(toy_gate({'rows': rows})['passed'])
        rows = self.gate_rows()
        rows[0]['evidence_known_positive_continue'] = False
        rows[1]['evidence_known_positive_continue'] = False
        self.assertFalse(toy_gate({'rows': rows})['passed'])

    def test_gate_rejects_nonroot_or_other_depth(self):
        for key,value in [('is_root',False),('teacher_depth',1)]:
            rows=self.gate_rows();rows[8][key]=value
            self.assertFalse(toy_gate({'rows':rows})['passed'])
            self.assertFalse(toy_gate({'rows':rows})['composition_valid'])

    def test_gate_rejects_duplicate_scenes(self):
        rows=self.gate_rows();rows[8]['scene_id']=rows[0]['scene_id']
        self.assertFalse(toy_gate({'rows':rows})['passed'])

    def test_gate_rejects_missing_stop_or_overlapping_classes(self):
        rows=self.gate_rows();rows[8]['clear_stop']=False
        self.assertFalse(toy_gate({'rows':rows})['passed'])
        rows=self.gate_rows();rows[0]['clear_stop']=True;rows[8]['clear_stop']=False
        self.assertFalse(toy_gate({'rows':rows})['passed'])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
