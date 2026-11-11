"""Bounded synthetic engineering checks, zero optimizer.step calls.

Do not run before the M1 registration. Actual fitted Adam checkpoint roundtrips
are checked by the admitted runner at real saved endpoints, without extra steps.
"""
import copy
import io
import random
import unittest

import numpy as np
import torch

from .critic import (ADAM, COMPONENTS, IDENTITY_FIELDS, REDUCTION, SEMANTICS,
    ComponentPrediction, MatchedCritic, capture_rng, component_state_loss,
    episode_features, episode_labels, load_export, make_checkpoint, make_export,
    make_matched_models, matched_loss, predict_episode, restore_checkpoint,
    tensor_digest, tree_digest)


def fixture(T=2, n=10):
    v = torch.zeros(T, 268)
    v[:, 256] = torch.linspace(1., .2, T)
    length = torch.linspace(4., 2., T).tolist()
    stops = torch.linspace(2., 1., T).tolist()
    h = [-.5] * T
    returns = [-(ell + .75 * k - credit) / n for ell, k, credit in zip(length, stops, h)]
    return dict(trajectory_id='synthetic_only', parent_id='synthetic_parent',
        split='fit', task_count=n, v=v, targets=dict(scale=n, raw_remaining_L=length,
        raw_remaining_K=stops, terminal_H_credit=h, returns=returns))


def identity():
    return {key: '0' * 64 for key in IDENTITY_FIELDS}


def resources():
    return dict(critic_optimizer_attempts=0, actor_optimizer_attempts=0)


def adam(model):
    return torch.optim.Adam(model.parameters(), **ADAM)


def roundtrip(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    stream.seek(0)
    # This is a trusted self-produced synthetic BytesIO object, not external input.
    return torch.load(stream, map_location='cpu', weights_only=False)


class M1CriticInterfaceTests(unittest.TestCase):
    def test_exact_initial_tensors_and_independent_storage(self):
        models = make_matched_models()
        self.assertEqual(tensor_digest(models['C0'].state_dict()), tensor_digest(models['C1'].state_dict()))
        for left, right in zip(models['C0'].parameters(), models['C1'].parameters()):
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
        self.assertEqual(list(models['C0'].state_dict()),
            ['net.0.weight', 'net.0.bias', 'net.2.weight', 'net.2.bias', 'net.4.weight', 'net.4.bias'])
        with self.assertRaises(ValueError): make_matched_models(11622027)

    def test_output_units_and_initial_physical_difference(self):
        models = make_matched_models();v = fixture()['v'];n = 10
        raw, norm = models['C0'](v, n), models['C1'](v, n)
        self.assertTrue(torch.equal(raw.length, norm.length))
        self.assertTrue(torch.allclose(raw.normalized_L * n, norm.normalized_L, atol=1e-7, rtol=1e-6))
        self.assertTrue(torch.equal(norm.raw_remaining_L, raw.raw_remaining_L * n))
        self.assertTrue(torch.allclose(raw.value * n, norm.value, atol=1e-7, rtol=1e-6))
        self.assertEqual(norm.metadata()['value_unit'], 'negative_cost/initial_task')
        self.assertEqual(norm.metadata()['output_semantics'], SEMANTICS['C1'])

    def test_signed_H_and_complete_value(self):
        prediction = ComponentPrediction('C1', 4, torch.tensor([1.]), torch.tensor([2.]), torch.tensor([-3.]))
        prediction.validate()
        self.assertEqual(float(prediction.value), -5.5)
        self.assertEqual(float(prediction.terminal_H_credit), -12.)
        self.assertFalse(prediction.metadata()['H_is_remaining_distance'])
        self.assertTrue(prediction.metadata()['H_includes_full_path_prefix_revision'])

    def test_input_and_initial_n_validation(self):
        model = make_matched_models()['C0']
        for n in (0, -1, 10., True, float('nan')):
            with self.assertRaises(ValueError): model(torch.zeros(2, 268), n)
        for shape in ((2, 267), (268,), (0, 268)):
            with self.assertRaises(ValueError): model(torch.zeros(*shape), 10)
        bad = torch.zeros(2, 268);bad[0, 2] = float('inf')
        with self.assertRaises(ValueError): model(bad, 10)
        with self.assertRaises(ValueError): model(torch.zeros(2, 268), 10, .5)
        with self.assertRaises(ValueError): MatchedCritic('C2')

    def test_saved_labels_count_scale_and_RTG_validation(self):
        episode = fixture();labels = episode_labels(episode)
        self.assertEqual(set(labels), set(COMPONENTS))
        for field, value in (('scale', 11), ('returns', [1., 2.]), ('raw_remaining_L', [-1., 2.])):
            bad = copy.deepcopy(episode);bad['targets'][field] = value
            with self.assertRaises(ValueError): episode_labels(bad)
        bad = fixture();bad['v'][0, 256] = 2.
        with self.assertRaises(ValueError): episode_features(bad)

    def test_single_division_raw_adapter_matches_original_loss(self):
        from rc3_conservative_refinement.model import remaining_cost_loss
        episode = fixture()
        for name, model in make_matched_models().items():
            prediction = model(episode['v'], 10)
            original = remaining_cost_loss(prediction.as_training_dict(), episode['targets'], 10)
            current = component_state_loss(prediction, episode)
            self.assertTrue(torch.allclose(current, original, atol=1e-7, rtol=1e-6), name)
        normalized = make_matched_models()['C1'](episode['v'], 10)
        labels = episode_labels(episode)
        expected = .5 * sum((getattr(normalized, name) - labels[name]).square()
                            for name in ('normalized_L', 'normalized_K', 'normalized_H'))
        self.assertTrue(torch.allclose(component_state_loss(normalized, episode), expected, atol=1e-7, rtol=1e-6))

    def test_state_SUM_then_episode_MEAN_and_validation_exclusion(self):
        model = make_matched_models()['C1'];episodes = [fixture(1), fixture(3)]
        expected = sum(component_state_loss(model(e['v'], e['task_count']), e).sum()
                       for e in episodes) / 2
        actual = matched_loss(model, episodes)
        self.assertTrue(torch.equal(actual, expected))
        wrong = sum(component_state_loss(model(e['v'], e['task_count']), e).mean()
                    for e in episodes) / 2
        self.assertFalse(torch.equal(actual, wrong))
        bad = fixture();bad['split'] = 'validation'
        with self.assertRaises(ValueError): matched_loss(model, [bad])
        with self.assertRaises(TypeError): component_state_loss(model(fixture()['v'], 10).as_dict(), fixture())
        self.assertEqual(REDUCTION, 'state_SUM_per_trajectory_then_episode_MEAN')

    def test_prediction_dictionary_has_only_detached_tensor_metrics(self):
        output = predict_episode(make_matched_models()['C1'], fixture())
        self.assertEqual(set(output), set(COMPONENTS))
        self.assertTrue(all(isinstance(v, torch.Tensor) and not v.requires_grad for v in output.values()))

    def test_checkpoint_step0_serialization_exact_no_update(self):
        model = make_matched_models()['C1'];optimizer = adam(model)
        package = make_checkpoint(model, optimizer, identity=identity(), cursor=0, resources=resources())
        loaded = roundtrip(package)
        self.assertEqual(tree_digest(package), tree_digest(loaded))
        # Deliberate in-memory perturbation is not an optimizer step.
        with torch.no_grad(): next(model.parameters()).add_(1.)
        answer = restore_checkpoint(loaded, model, optimizer, expected_identity=identity(),
                                   expected_cursor=0, expected_resources=resources())
        self.assertTrue(answer['exact_restore'])
        self.assertEqual(tensor_digest(model.state_dict()), tensor_digest(package['model']))
        self.assertEqual(tree_digest(optimizer.state_dict()), tree_digest(package['optimizer']))
        self.assertEqual(tree_digest(capture_rng()), tree_digest(package['rng']))
        self.assertEqual(len(optimizer.state), 0)

    def test_checkpoint_semantics_source_cursor_resources_refuse(self):
        models = make_matched_models();model = models['C1'];optimizer = adam(model)
        package = make_checkpoint(model, optimizer, identity=identity(), cursor=0, resources=resources())
        with self.assertRaises(ValueError):
            restore_checkpoint(package, models['C0'], adam(models['C0']), expected_identity=identity(),
                               expected_cursor=0, expected_resources=resources())
        changed = identity();changed['source_sha256'] = 'f' * 64
        with self.assertRaises(ValueError):
            restore_checkpoint(package, model, optimizer, expected_identity=changed,
                               expected_cursor=0, expected_resources=resources())
        with self.assertRaises(ValueError):
            restore_checkpoint(package, model, optimizer, expected_identity=identity(),
                               expected_cursor=1, expected_resources=resources())
        with self.assertRaises(ValueError):
            restore_checkpoint(package, model, optimizer, expected_identity=identity(),
                               expected_cursor=0, expected_resources=dict(resources(), critic_optimizer_attempts=1))

    def test_checkpoint_tamper_or_extra_fields_refuse_before_mutation(self):
        model = make_matched_models()['C0'];optimizer = adam(model)
        package = make_checkpoint(model, optimizer, identity=identity(), cursor=0, resources=resources())
        before = tensor_digest(model.state_dict())
        for change in ('model', 'schema', 'extra'):
            bad = copy.deepcopy(package)
            if change == 'model': next(iter(bad['model'].values())).add_(1.)
            elif change == 'schema': bad['schema'] = 'old_critic_unknown_units'
            else: bad['Actor'] = {}
            with self.assertRaises(ValueError):
                restore_checkpoint(bad, model, optimizer, expected_identity=identity(),
                                   expected_cursor=0, expected_resources=resources())
            self.assertEqual(before, tensor_digest(model.state_dict()))

    def test_wrong_adam_or_nonzero_actor_budget_refused(self):
        model = make_matched_models()['C0']
        wrong = torch.optim.Adam(model.parameters(), lr=.01)
        with self.assertRaises(ValueError):
            make_checkpoint(model, wrong, identity=identity(), cursor=0, resources=resources())
        with self.assertRaises(ValueError):
            make_checkpoint(model, adam(model), identity=identity(), cursor=0,
                            resources=dict(resources(), actor_optimizer_attempts=1))
        with self.assertRaises(ValueError):
            make_checkpoint(model, adam(model), identity=identity(), cursor=1, resources=resources())

    def test_critic_only_export_roundtrip_predictions_and_rng(self):
        for name, model in make_matched_models().items():
            package = roundtrip(make_export(model, identity=identity(), cursor=0))
            before_rng = tree_digest(capture_rng())
            loaded = load_export(package, expected_variant=name, expected_identity=identity(), expected_cursor=0)
            self.assertEqual(before_rng, tree_digest(capture_rng()))
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            self.assertFalse(loaded.training)
            before, after = predict_episode(model, fixture()), predict_episode(loaded, fixture())
            for key in COMPONENTS: self.assertTrue(torch.equal(before[key], after[key]))
            self.assertNotIn('optimizer', package)
            self.assertNotIn('Actor', package)

    def test_critic_export_rejects_units_optimizer_unknown_schema(self):
        model = make_matched_models()['C1'];package = make_export(model, identity=identity(), cursor=0)
        with self.assertRaises(ValueError):
            load_export(package, expected_variant='C0', expected_identity=identity(), expected_cursor=0)
        for key in ('optimizer', 'actor', 'unknown'):
            bad = copy.deepcopy(package);bad[key] = {}
            with self.assertRaises(ValueError):
                load_export(bad, expected_variant='C1', expected_identity=identity(), expected_cursor=0)


if __name__ == '__main__':
    unittest.main()
