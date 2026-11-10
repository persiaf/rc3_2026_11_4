"""Synthetic exact-distance lookahead tests; no planner, tasks or weights loaded."""
import time
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest import mock

import numpy as np

from rc3_conservative_refinement.lookahead import (
    post_service_distances, prepare_service_distances, prepared_post_service_distances,
)


class PostServiceDistanceTests(unittest.TestCase):
    def setUp(self):
        self.D = np.array([[0., 20., 2., 8.], [20., 0., 18., 12.],
                           [2., 18., 0., 6.], [8., 12., 6., 0.]])
        self.coverage = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]], dtype=bool)
        self.remaining = np.ones(3, dtype=bool)
        self.mask = np.array([1, 0, 0, 0], dtype=bool)

    def test_removes_served_tasks_then_minimizes_per_task(self):
        result = post_service_distances(self.D, self.coverage, self.remaining, self.mask)
        self.assertEqual(result['remaining_after_count'][0], 2)
        self.assertEqual(result['raw_mean'][0], 5.)
        self.assertEqual(result['raw_max'][0], 8.)
        np.testing.assert_allclose(result['finite_features'][0], [.5, .8, 2 / 3, 0])
        self.assertTrue(result['all_remaining_reachable'][0])
        # Candidate mean (20+2+8)/3 and pre-removal task mean (0+2+8)/3 are wrong.
        self.assertNotEqual(result['raw_mean'][0], (20 + 2 + 8) / 3)
        self.assertNotEqual(result['raw_mean'][0], (0 + 2 + 8) / 3)

    def test_duplicate_service_candidate_does_not_reweight_a_task(self):
        before = post_service_distances(self.D, self.coverage, self.remaining, self.mask)
        indices = np.array([0, 1, 2, 3, 2])
        after = post_service_distances(self.D[np.ix_(indices, indices)], self.coverage[:, indices],
                                       self.remaining, np.array([1, 0, 0, 0, 0], dtype=bool))
        self.assertEqual(before['raw_mean'][0], after['raw_mean'][0])
        self.assertEqual(before['raw_max'][0], after['raw_max'][0])

    def test_unreachable_task_is_not_silently_dropped(self):
        self.D[0, 1:3] = np.inf
        result = post_service_distances(self.D, self.coverage, self.remaining, self.mask)
        self.assertEqual(result['remaining_after_count'][0], 2)
        self.assertEqual(result['unreachable_remaining_count'][0], 1)
        self.assertTrue(np.isposinf(result['raw_mean'][0]))
        self.assertTrue(np.isposinf(result['raw_max'][0]))
        self.assertFalse(result['all_remaining_reachable'][0])
        np.testing.assert_allclose(result['finite_features'][0], [0, 0, 2 / 3, .5])
        self.assertTrue(np.isfinite(result['finite_features']).all())

    def test_no_service_candidate_means_unreachable_not_task_deletion(self):
        self.coverage[2] = False
        result = post_service_distances(self.D, self.coverage, self.remaining, self.mask)
        self.assertEqual(result['remaining_after_count'][0], 2)
        self.assertEqual(result['unreachable_remaining_count'][0], 1)
        self.assertTrue(np.isposinf(result['raw_max'][0]))

    def test_finishing_action_has_zero_future_cost(self):
        self.coverage[:, 0] = True
        result = post_service_distances(self.D, self.coverage, self.remaining, self.mask)
        self.assertEqual(result['raw_mean'][0], 0)
        self.assertEqual(result['raw_max'][0], 0)
        self.assertEqual(result['remaining_after_count'][0], 0)
        self.assertTrue(result['all_remaining_reachable'][0])
        np.testing.assert_array_equal(result['finite_features'][0], np.zeros(4))

    def test_illegal_actions_are_masked_placeholders(self):
        result = post_service_distances(self.D, self.coverage, self.remaining, self.mask)
        np.testing.assert_array_equal(result['valid_action'], self.mask)
        self.assertFalse(result['all_remaining_reachable'][1:].any())
        np.testing.assert_array_equal(result['finite_features'][1:], np.zeros((3, 4)))
        np.testing.assert_array_equal(result['raw_mean'][1:], np.zeros(3))

    def test_exit_filter_only_changes_next_service_candidates(self):
        result = post_service_distances(self.D, self.coverage, self.remaining, self.mask,
                                        exit_distances=np.array([np.inf, 0., np.inf, 0.]))
        self.assertEqual(result['raw_mean'][0], 14.)
        self.assertEqual(result['raw_max'][0], 20.)
        np.testing.assert_array_equal(result['valid_action'], self.mask)
        # Candidate 1 is currently masked but remains an eligible next service.
        self.assertTrue(result['all_remaining_reachable'][0])

    def test_unreachable_exit_does_not_delete_remaining_target(self):
        result = post_service_distances(self.D, self.coverage, self.remaining, self.mask,
                                        exit_distances=np.array([0., np.inf, np.inf, 0.]))
        self.assertEqual(result['remaining_after_count'][0], 2)
        self.assertEqual(result['unreachable_remaining_count'][0], 1)
        self.assertTrue(np.isposinf(result['raw_mean'][0]))

    def test_input_arrays_and_writeability_are_unchanged(self):
        arrays = [self.D, self.coverage, self.remaining, self.mask]
        copies = [array.copy() for array in arrays]
        for array in arrays:
            array.flags.writeable = False
        result = post_service_distances(*arrays)
        result['valid_action'][:] = False
        for array, original in zip(arrays, copies):
            np.testing.assert_array_equal(array, original)
            self.assertFalse(array.flags.writeable)

    def test_candidate_permutation_equivariance(self):
        self.mask[:] = True
        before = post_service_distances(self.D, self.coverage, self.remaining, self.mask)
        permutation = np.array([2, 0, 3, 1])
        after = post_service_distances(self.D[np.ix_(permutation, permutation)],
                                       self.coverage[:, permutation], self.remaining, self.mask[permutation])
        for name in ('raw_mean', 'raw_max', 'remaining_after_count', 'unreachable_remaining_count',
                     'valid_action', 'all_remaining_reachable', 'finite_features'):
            np.testing.assert_array_equal(after[name], before[name][permutation])

    def test_task_permutation_and_completed_history_do_not_reweight_features(self):
        permutation = np.array([2, 0, 1])
        before = post_service_distances(self.D, self.coverage, self.remaining, self.mask)
        after = post_service_distances(self.D, self.coverage[permutation], self.remaining[permutation], self.mask)
        np.testing.assert_array_equal(after['finite_features'], before['finite_features'])
        coverage = np.vstack([self.coverage, np.ones((7, 4), dtype=bool)])
        remaining = np.concatenate([self.remaining, np.zeros(7, dtype=bool)])
        history = post_service_distances(self.D, coverage, remaining, self.mask)
        np.testing.assert_array_equal(history['finite_features'], before['finite_features'])

    def test_empty_remaining_and_zero_dimension_inputs(self):
        result = post_service_distances(self.D, self.coverage, np.zeros(3, dtype=bool), self.mask)
        self.assertTrue(result['all_remaining_reachable'][0])
        np.testing.assert_array_equal(result['finite_features'], np.zeros((4, 4)))
        empty = post_service_distances(np.zeros((0, 0)), np.zeros((0, 0), dtype=bool),
                                       np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))
        self.assertEqual(empty['finite_features'].shape, (0, 4))

    def test_large_finite_distances_do_not_overflow_mean(self):
        self.D[0, 1:] = 1e308
        result = post_service_distances(self.D, self.coverage, self.remaining, self.mask)
        self.assertEqual(result['raw_mean'][0], 1e308)
        self.assertTrue(np.isfinite(result['finite_features']).all())

    def test_rejects_wrong_shapes_and_non_boolean_semantics(self):
        invalid = [
            (self.D[:, :3], self.coverage, self.remaining, self.mask, None),
            (self.D, self.coverage[:, :3], self.remaining, self.mask, None),
            (self.D, self.coverage, self.remaining[:, None], self.mask, None),
            (self.D, self.coverage, self.remaining, self.mask[:3], None),
            (self.D, self.coverage.astype(int), self.remaining, self.mask, None),
            (self.D, self.coverage, self.remaining.astype(int), self.mask, None),
            (self.D, self.coverage, self.remaining, self.mask.astype(int), None),
            (self.D, self.coverage, self.remaining, self.mask, np.zeros(3)),
            (self.D.astype(bool), self.coverage, self.remaining, self.mask, None),
            (self.D.astype(complex), self.coverage, self.remaining, self.mask, None),
            (self.D.astype(str), self.coverage, self.remaining, self.mask, None),
        ]
        for index, args in enumerate(invalid):
            with self.subTest(case=index), self.assertRaises(ValueError):
                post_service_distances(*args)

    def test_rejects_nan_negative_and_invalid_exit_distances(self):
        for value in (np.nan, -1., -np.inf):
            D = self.D.copy()
            D[0, 1] = value
            with self.subTest(D=value), self.assertRaises(ValueError):
                post_service_distances(D, self.coverage, self.remaining, self.mask)
            exit_distances = np.zeros(4)
            exit_distances[1] = value
            with self.subTest(exit=value), self.assertRaises(ValueError):
                post_service_distances(self.D, self.coverage, self.remaining, self.mask, exit_distances)
        with self.assertRaises(ValueError):
            post_service_distances(self.D, self.coverage, self.remaining, self.mask, np.zeros(4, dtype=bool))

    def test_prepared_matches_wrapper_and_reuses_provider_distances(self):
        prepared = prepare_service_distances(self.D, self.coverage)
        original = post_service_distances(self.D, self.coverage, self.remaining, self.mask)
        pending = np.array([False, True, True])
        expected_later = post_service_distances(self.D, self.coverage, pending, self.mask)
        with mock.patch('rc3_conservative_refinement.lookahead.np.min', side_effect=AssertionError('provider rescan')):
            first = prepared_post_service_distances(prepared, self.remaining, self.mask)
            later = prepared_post_service_distances(prepared, pending, self.mask)
        for key in ('raw_mean', 'raw_max', 'finite_features', 'remaining_after_count'):
            np.testing.assert_array_equal(first[key], original[key])
            np.testing.assert_array_equal(later[key], expected_later[key])
        np.testing.assert_array_equal(prepared.task_distances[0], [0., 2., 8.])

    def test_prepared_is_a_detached_irreversibly_readonly_snapshot(self):
        prepared = prepare_service_distances(self.D, self.coverage)
        before = prepared_post_service_distances(prepared, self.remaining, self.mask)
        identity = prepared.source_identity
        self.D[:] = 0
        self.coverage[:] = False
        after = prepared_post_service_distances(prepared, self.remaining, self.mask)
        np.testing.assert_array_equal(before['finite_features'], after['finite_features'])
        self.assertEqual(prepared.source_identity, identity)
        for array in (prepared.task_distances, prepared.coverage):
            with self.assertRaises(ValueError):
                array.flat[0] = 0
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):
            prepared.source_identity = 'changed'
        updated = prepare_service_distances(self.D, self.coverage)
        self.assertNotEqual(updated.source_identity, identity)

    def test_prepared_identity_shape_and_writeability_tampering_rejected(self):
        prepared = prepare_service_distances(self.D, self.coverage)
        wrong_sources = (
            None,
            replace(prepared, source_identity='sha256:' + '0' * 64),
            replace(prepared, task_distances=prepared.task_distances[:, :2]),
            replace(prepared, coverage=prepared.coverage.copy()),
            replace(prepared, candidate_count=3),
        )
        for index, value in enumerate(wrong_sources):
            with self.subTest(case=index), self.assertRaises(ValueError):
                prepared_post_service_distances(value, self.remaining, self.mask)

    def test_prepared_source_binds_exit_input_and_preserves_inf_empty_semantics(self):
        no_exit = prepare_service_distances(self.D, self.coverage)
        explicit = prepare_service_distances(self.D, self.coverage, np.zeros(4))
        self.assertNotEqual(no_exit.source_identity, explicit.source_identity)
        exits = np.array([0., np.inf, np.inf, 0.])
        prepared = prepare_service_distances(self.D, self.coverage, exits)
        exits[:] = 0
        unreachable = prepared_post_service_distances(prepared, self.remaining, self.mask)
        self.assertTrue(np.isposinf(unreachable['raw_mean'][0]))
        self.assertEqual(unreachable['unreachable_remaining_count'][0], 1)
        complete = prepared_post_service_distances(prepared, np.zeros(3, dtype=bool), self.mask)
        self.assertTrue(complete['all_remaining_reachable'][0])
        np.testing.assert_array_equal(complete['finite_features'], np.zeros((4, 4)))


if __name__ == '__main__':
    started = time.perf_counter()
    try:
        unittest.main(verbosity=2)
    finally:
        print(f'pure_numpy_test_wall_s={time.perf_counter() - started:.9f}')
