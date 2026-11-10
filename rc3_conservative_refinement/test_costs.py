"""Synthetic arithmetic invariants only; standard library and no optimizer."""
import math
import time
import unittest

from rc3_conservative_refinement.costs import remaining_cost_targets


class RemainingCostTargetTests(unittest.TestCase):
    def assert_sequence_close(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got, want, places=11)

    def test_remaining_costs_include_exit_and_all_future_stops(self):
        result = remaining_cost_targets([1.0, 2.0, 3.0], 6.0, 5)
        self.assert_sequence_close(result['raw_remaining_L'], [6.0, 5.0])
        self.assertEqual(result['raw_remaining_K'], [2, 1])
        self.assertEqual(result['J_raw'], 7.5)
        self.assertEqual(result['H_gain'], 1.5)
        self.assert_sequence_close(result['returns'], [-1.2, -0.85])
        self.assert_sequence_close(result['dense_rewards'], [-1.75, -4.25])

    def test_late_exit_cost_changes_early_learning_target(self):
        short = remaining_cost_targets([2.0, 3.0, 4.0], 10.5, 2)
        long = remaining_cost_targets([2.0, 3.0, 104.0], 110.5, 2)
        self.assertEqual(short['H_gain'], 0.0)
        self.assertEqual(long['H_gain'], 0.0)
        self.assertAlmostEqual(long['returns'][0] - short['returns'][0], -50.0)
        self.assertAlmostEqual(long['returns'][1] - short['returns'][1], -50.0)

    def test_positive_h_gain_keeps_raw_targets_separate(self):
        raw = remaining_cost_targets([3.0, 4.0], 7.75, 1)
        improved = remaining_cost_targets([3.0, 4.0], 6.0, 1)
        self.assertEqual(raw['raw_remaining_L'], improved['raw_remaining_L'])
        self.assertEqual(raw['raw_remaining_K'], improved['raw_remaining_K'])
        self.assertEqual(improved['H_gain'], 1.75)
        self.assertAlmostEqual(improved['complete_return'], -6.0)
        self.assertNotEqual(-improved['returns'][0], improved['raw_remaining_L'][0])

    def test_negative_h_gain_is_not_silently_clipped(self):
        result = remaining_cost_targets([3.0, 4.0], 9.0, 1)
        self.assertEqual(result['H_gain'], -1.25)
        self.assert_sequence_close(result['dense_rewards'], [-9.0])
        self.assert_sequence_close(result['returns'], [-9.0])

    def test_every_suffix_return_matches_dense_rewards(self):
        result = remaining_cost_targets([0.1, 0.2, 0.3, 0.4], 2.75, 9)
        self.assertAlmostEqual(math.fsum(result['dense_rewards']), -result['final_J'])
        for index, target in enumerate(result['returns']):
            self.assertAlmostEqual(target, math.fsum(result['scaled_dense_rewards'][index:]))

    def test_empty_tasks_have_exit_leg_but_no_fake_action(self):
        result = remaining_cost_targets([4.0], 4.0, 0)
        self.assertEqual(result['raw_route_length'], 4.0)
        self.assertEqual(result['raw_stop_count'], 0)
        for name in ('raw_remaining_L', 'raw_remaining_K', 'returns', 'dense_rewards', 'scaled_dense_rewards'):
            self.assertEqual(result[name], [])
        self.assertEqual(result['no_action_terminal_reward'], -4.0)
        self.assertEqual(result['complete_return'], -4.0)
        self.assertEqual(result['scale'], 1.0)

    def test_empty_tasks_still_report_signed_h_correction(self):
        result = remaining_cost_targets([4.0], 5.0, 0)
        self.assertEqual(result['H_gain'], -1.0)
        self.assertEqual(result['complete_return'], -5.0)
        self.assertEqual(result['scaled_no_action_terminal_reward'], -5.0)

    def test_zero_cost_and_custom_penalty(self):
        result = remaining_cost_targets((0, 0), 0, 1, penalty=0)
        self.assertEqual(result['J_raw'], 0.0)
        self.assertEqual(result['returns'], [0.0])

    def test_rejects_invalid_leg_shapes(self):
        for legs in ([], (), None, '1,2', 3, [[1], 2], [1, [2]]):
            with self.subTest(legs=legs), self.assertRaises(ValueError):
                remaining_cost_targets(legs, 1, 1)

    def test_rejects_nonfinite_negative_and_boolean_numbers(self):
        for value in (True, False, -1, float('inf'), float('-inf'), float('nan'), '2', None):
            with self.subTest(leg=value), self.assertRaises(ValueError):
                remaining_cost_targets([value, 1], 1, 1)
            with self.subTest(final=value), self.assertRaises(ValueError):
                remaining_cost_targets([1, 1], value, 1)
            with self.subTest(penalty=value), self.assertRaises(ValueError):
                remaining_cost_targets([1, 1], 1, 1, penalty=value)

    def test_rejects_invalid_task_counts_and_partial_traces(self):
        for count in (True, False, -1, 1.0, '1', None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                remaining_cost_targets([1, 1], 1, count)
        for legs, count in (([1], 1), ([1, 1], 0), ([1, 1, 1], 1)):
            with self.subTest(legs=legs, count=count), self.assertRaises(ValueError):
                remaining_cost_targets(legs, 1, count)

    def test_rejects_derived_overflow(self):
        for args in (([1e308, 1e308], 1.0, 1), ([1, 1], 1, 10**400)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                remaining_cost_targets(*args)

    def test_inputs_are_not_modified(self):
        legs = [1.0, 2.0]
        result = remaining_cost_targets(legs, 3.75, 1)
        result['raw_leg_lengths'][0] = 99
        self.assertEqual(legs, [1.0, 2.0])


if __name__ == '__main__':
    started = time.perf_counter()
    try:
        unittest.main(verbosity=2)
    finally:
        print(f'pure_stdlib_test_wall_s={time.perf_counter() - started:.9f}')
