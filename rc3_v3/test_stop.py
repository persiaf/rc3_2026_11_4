"""Regression checks for global binary STOP and historical compatibility."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from rc3_v3.environment import FEATURE_NAMES
from rc3_v3.model import export_numpy, load_policy, make_policy, save_policy, select_action
from rc3_v3.train import loss_for


class StopTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(31)
        self.rng = np.random.default_rng(31)

    def test_low_global_stop_does_not_compete_with_many_window_logits(self):
        for count in (1, 10, 100, 500):
            with self.subTest(count=count):
                mask = np.ones(count, bool)
                logits = np.r_[np.full(count, -4.), -2.]
                # sigmoid(-2)=0.119; the old categorical rule stops because
                # each individual window logit is lower, regardless of count.
                self.assertIsNone(select_action(logits, mask, 'categorical_argmax'))
                self.assertEqual(select_action(logits, mask, 'binary_threshold', .9), 0)

    def test_binary_threshold_and_mask_are_independent_of_action_scale(self):
        mask = np.asarray([True, False, True])
        self.assertEqual(select_action(np.asarray([-4., 1e6, -3., -2.]), mask), 2)
        self.assertIsNone(select_action(np.asarray([1e6, -np.inf, 0., 3.]), mask))
        with self.assertRaises(ValueError):
            select_action(np.zeros(4), mask, stop_threshold=1.)

    def test_global_stop_bce_has_correct_gradient_for_positive_and_zero_states(self):
        for architecture in ('mlp', 'attention'):
            for has_improvement in (False, True):
                with self.subTest(architecture=architecture, has_improvement=has_improvement):
                    model = make_policy(np.zeros(len(FEATURE_NAMES)),
                                        np.ones(len(FEATURE_NAMES)), architecture, dim=32)
                    with torch.no_grad():
                        model.stop[-1].weight.zero_()
                        model.stop[-1].bias.zero_()
                    row = dict(features=self.rng.normal(size=(4, len(FEATURE_NAMES))).astype(np.float32),
                               mask=np.asarray([True, False, True, True]),
                               gains=np.asarray([1. if has_improvement else 0., 9., 0., 0.], np.float32))
                    # A large gain in a masked slot must not turn a zero-gain
                    # state into a positive state or affect its STOP target.
                    loss, parts = loss_for(model, [row], 1., 1., 'cpu')
                    self.assertTrue(torch.isfinite(loss))
                    self.assertAlmostEqual(parts['stop'], float(np.log(2.)), places=6)
                    loss.backward()
                    self.assertAlmostEqual(float(model.stop[-1].bias.grad),
                                           .5 if has_improvement else -.5, places=6)
                    self.assertTrue(all(torch.isfinite(p.grad).all()
                                        for p in model.parameters() if p.grad is not None))
                    if not has_improvement:
                        self.assertEqual(parts['rank'], 0.)
                        self.assertTrue(torch.equal(model.rank.weight.grad,
                                                    torch.zeros_like(model.rank.weight.grad)))

    def test_legacy_checkpoint_preserves_categorical_rule_in_torch_and_numpy(self):
        model = make_policy(np.zeros(len(FEATURE_NAMES)), np.ones(len(FEATURE_NAMES)), 'mlp', dim=32)
        with torch.no_grad():
            model.rank.weight.zero_()
            model.rank.bias.fill_(-4.)
            model.stop[-1].weight.zero_()
            model.stop[-1].bias.fill_(-2.)
        features = np.zeros((50, len(FEATURE_NAMES)), np.float32)
        mask = np.ones(50, bool)
        self.assertEqual(model.select(features, mask), 0)
        with tempfile.TemporaryDirectory() as tmp:
            pt, npz = Path(tmp)/'legacy.pt', Path(tmp)/'legacy.npz'
            save_policy(pt, model)
            saved = torch.load(pt, weights_only=True)
            del saved['stop_rule'], saved['stop_threshold']
            torch.save(saved, pt)
            export_numpy(pt, npz)
            for path in (pt, npz):
                restored = load_policy(path)
                self.assertEqual(restored.stop_rule, 'categorical_argmax')
                self.assertIsNone(restored.select(features, mask))


if __name__ == '__main__':
    unittest.main()
