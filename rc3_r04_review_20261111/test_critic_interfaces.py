"""Artificial tensors/records only. No optimizer, historical input or weight I/O."""
import copy
from dataclasses import replace
import unittest

import torch

from rc3_conservative_refinement.model import remaining_cost_loss
from .critic_interfaces import (BASELINE_SCHEMA, C1_SCHEMA, BaselineTrainingRecord,
    C1Critic, C1Prediction, ConditionalComponentBaseline, ObservableCondition,
    compose_conditional_prediction, initial_scale, normalized_component_loss)


def artificial_records():
    def row(parent, trajectory, index, remaining, length, stops, credit):
        return BaselineTrainingRecord(parent, trajectory, index, 'fit',
            ObservableCondition(4, 3, remaining), length, stops, credit)
    return [row('physical_A', 'A1', 0, 1., 2., 1., .5),
            row('physical_A', 'A1', 1, .5, 1., .5, .5),
            row('physical_A', 'A2', 0, 1., 6., 3., -.5),
            row('physical_B', 'B1', 0, 1., 4., 2., 1.)]


def artificial_baseline(records=None):
    return ConditionalComponentBaseline.from_fit_records(
        artificial_records() if records is None else records,
        fit_physical_parents={'physical_A', 'physical_B'},
        source_sha256={'artificial_fixture_only': '0' * 64})


class C1InterfaceTests(unittest.TestCase):
    def new_model(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1701)
            return C1Critic()

    def test_268_only_and_finite_input(self):
        with self.assertRaises(ValueError):
            C1Critic(input_dim=269)
        model = self.new_model()
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 267), 10)
        bad = torch.zeros(2, 268);bad[0, 1] = float('nan')
        with self.assertRaises(ValueError):
            model(bad, 10)

    def test_explicit_units_and_reversible_raw_adapter(self):
        prediction = self.new_model()(torch.zeros(3, 268), 20)
        self.assertEqual(prediction.schema, C1_SCHEMA)
        self.assertEqual(prediction.metadata()['normalized_K_unit'], 'stops / initial_task')
        raw = prediction.as_training_dict()
        self.assertTrue(torch.equal(raw['raw_remaining_L'], 20 * prediction.normalized_L))
        self.assertTrue(torch.equal(raw['raw_remaining_K'], 20 * prediction.normalized_K))
        self.assertTrue(torch.equal(raw['terminal_H_credit'], 20 * prediction.normalized_H))
        self.assertTrue(torch.equal(raw['value'], prediction.value))
        self.assertEqual(set(raw), {'raw_remaining_L', 'raw_remaining_K', 'terminal_H_credit', 'value'})

    def test_signed_h_and_composed_value(self):
        prediction = C1Prediction(10, torch.tensor([2., 3.]),
            torch.tensor([1., 2.]), torch.tensor([-.5, .5])).validate()
        self.assertTrue(torch.equal(prediction.value, torch.tensor([-3.25, -4.])))
        self.assertLess(float(prediction.normalized_H[0]), 0)

    def test_loss_equals_old_formula_after_single_normalization(self):
        prediction = C1Prediction(10, torch.tensor([2., 3.]),
            torch.tensor([1., 2.]), torch.tensor([-.5, .5])).validate()
        targets = dict(task_count=10, scale=10., raw_remaining_L=[15., 35.],
            raw_remaining_K=[8., 18.], terminal_H_credit=[-4., 6.],
            returns=[-(15.+.75*8.+4.)/10, -(35.+.75*18.-6.)/10])
        modern = normalized_component_loss(prediction, targets)
        historical = remaining_cost_loss(prediction.as_training_dict(), targets, 10)
        self.assertTrue(torch.allclose(modern, historical, atol=1e-6, rtol=1e-6))
        wrong = {k: getattr(prediction, n) for k, n in
                 [('raw_remaining_L','normalized_L'),('raw_remaining_K','normalized_K'),
                  ('terminal_H_credit','normalized_H')]}
        double_divided = remaining_cost_loss(wrong, targets, 10)
        self.assertFalse(torch.allclose(modern, double_divided))

    def test_fixed_initial_scale_is_checked_against_labels(self):
        prediction = self.new_model()(torch.zeros(2, 268), 10)
        with self.assertRaises(ValueError):
            normalized_component_loss(prediction, {'scale': 2})
        with self.assertRaises(ValueError):
            normalized_component_loss(prediction, {'task_count': 2})
        for bad in (True, -1, 2.5):
            with self.assertRaises(ValueError):
                initial_scale(bad)

    def test_empty_tasks_have_no_policy_states(self):
        model = self.new_model()
        empty = model(torch.zeros(0, 268), 0)
        self.assertEqual(empty.scale, 1)
        self.assertEqual(empty.value.numel(), 0)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 268), 0)

    def test_bad_physical_components_not_silently_clipped(self):
        with self.assertRaises(ValueError):
            C1Prediction(10, torch.tensor([-1.]), torch.tensor([0.]), torch.tensor([0.])).validate()
        with self.assertRaises(ValueError):
            C1Prediction(10, torch.tensor([0.]), torch.tensor([float('inf')]), torch.tensor([0.])).validate()

    def test_gradient_is_finite_and_input_is_detached(self):
        model = self.new_model()
        features = torch.zeros(2, 268, requires_grad=True)
        prediction = model(features, 10)
        prediction.value.square().mean().backward()
        self.assertIsNone(features.grad)
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))


class ConditionalBaselineInterfaceTests(unittest.TestCase):
    def test_fit_statistics_are_parent_trajectory_state_balanced(self):
        baseline = artificial_baseline()
        global_components = baseline.artifact['global_components']
        self.assertEqual(global_components['normalized_L'], 3.875)
        self.assertEqual(global_components['normalized_K'], 1.9375)
        self.assertEqual(global_components['normalized_H'], .5)
        self.assertEqual(baseline.artifact['schema'], BASELINE_SCHEMA)
        self.assertEqual(baseline.artifact['fit_physical_parents'], ['physical_A', 'physical_B'])

    def test_fit_fold_poison_is_rejected_not_ignored(self):
        rows = artificial_records()
        for bad in ('validation', 'holdout', 'test'):
            with self.assertRaises(ValueError):
                artificial_baseline([replace(rows[0], split=bad), *rows[1:]])

    def test_unregistered_physical_parent_is_rejected(self):
        rows = artificial_records()
        with self.assertRaises(ValueError):
            artificial_baseline([replace(rows[0], physical_parent='heldout_physical'), *rows[1:]])

    def test_duplicate_and_missing_parent_fail(self):
        rows = artificial_records()
        with self.assertRaises(ValueError):
            artificial_baseline([*rows, rows[0]])
        with self.assertRaises(ValueError):
            artificial_baseline(rows[:-1])

    def test_forbidden_future_fields_cannot_be_lookup_inputs(self):
        observable = dict(window_width=4, window_height=3, remaining_fraction=.5)
        for forbidden in ('final_H', 'final_J', 'final_route_length', 'returns', 'validation_target'):
            with self.assertRaises(ValueError):
                ObservableCondition.from_mapping(dict(observable, **{forbidden: 1.}))
        with self.assertRaises(TypeError):
            artificial_baseline().lookup(observable)

    def test_observable_indices_match_frozen_schema(self):
        row = torch.zeros(268);row[256]=.5;row[265]=.4;row[266]=.3
        condition = ObservableCondition.from_feature_row(row)
        self.assertEqual(condition.key(), '4x3|q2')
        with self.assertRaises(ValueError):
            ObservableCondition.from_feature_row(torch.zeros(269))
        with self.assertRaises(ValueError):
            ObservableCondition(4, 3, -0.1)

    def test_unknown_condition_uses_fit_global_only(self):
        baseline = artificial_baseline()
        hit = baseline.lookup(ObservableCondition(4, 3, 1.))
        fallback = baseline.lookup(ObservableCondition(6, 6, .5))
        self.assertFalse(hit['fallback']);self.assertTrue(fallback['fallback'])
        self.assertEqual(hit['normalized_L'], 4.)
        self.assertEqual(fallback['normalized_L'], 3.875)
        self.assertEqual(fallback['value'], -(3.875+.75*1.9375-.5))

    def test_artifact_identity_and_external_copy_isolation(self):
        baseline = artificial_baseline()
        artifact = baseline.artifact
        artifact['global_components']['normalized_L'] = 100.
        self.assertEqual(baseline.artifact['global_components']['normalized_L'], 3.875)
        with self.assertRaises(ValueError):
            ConditionalComponentBaseline(artifact)

    def test_zero_residual_is_exact_baseline(self):
        baseline = artificial_baseline().lookup(ObservableCondition(4, 3, 1.))
        zero = {name: torch.zeros(2) for name in ('normalized_L','normalized_K','normalized_H')}
        prediction, residual = compose_conditional_prediction(baseline, zero, 10)
        self.assertTrue(torch.equal(prediction.value, torch.full((2,), baseline['value'])))
        self.assertTrue(torch.equal(residual, torch.zeros(2)))

    def test_value_residual_comes_from_same_components(self):
        baseline = artificial_baseline().lookup(ObservableCondition(4, 3, 1.))
        residuals = dict(normalized_L=torch.tensor([.25,-.1]),
            normalized_K=torch.tensor([.2,-.1]), normalized_H=torch.tensor([-.2,.2]))
        prediction, residual = compose_conditional_prediction(baseline, residuals, 10)
        expected = -(residuals['normalized_L']+.75*residuals['normalized_K']-residuals['normalized_H'])
        self.assertTrue(torch.equal(residual, expected))
        self.assertTrue(torch.allclose(prediction.value, baseline['value']+residual))
        with self.assertRaises(ValueError):
            compose_conditional_prediction(baseline, dict(residuals, value=torch.zeros(2)), 10)

    def test_negative_combined_lk_is_rejected(self):
        baseline = artificial_baseline().lookup(ObservableCondition(4, 3, 1.))
        with self.assertRaises(ValueError):
            compose_conditional_prediction(baseline,
                dict(normalized_L=torch.tensor([-5.]),normalized_K=torch.zeros(1),normalized_H=torch.zeros(1)),10)


if __name__ == '__main__':
    unittest.main()
