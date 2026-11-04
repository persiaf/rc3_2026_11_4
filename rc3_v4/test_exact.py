"""Independent exhaustive checks, including overlapping coverage and bad inputs."""
import itertools
import unittest
from unittest.mock import patch
import numpy as np
from .exact import solve_metric


def brute(cover, costs, penalty):
    n, m = cover.shape
    best = float('inf')
    for k in range(m+1):
        for order in itertools.permutations(range(m), k):
            if not cover[:, order].any(1).all():
                continue
            route = [0]+[j+1 for j in order]+[0]
            best = min(best, sum(costs[a, b] for a, b in zip(route, route[1:]))+penalty*k)
    return best


class ExactTests(unittest.TestCase):
    def test_overlap_matches_independent_permutation_enumeration(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            xy = rng.random((6, 2))
            costs = np.linalg.norm(xy[:, None]-xy[None, :], axis=2)
            cover = rng.random((3, 5)) < .5
            cover[:, :3] |= np.eye(3, dtype=bool)
            for penalty in (0., .75, 2.):
                result = solve_metric(cover, costs, penalty)
                self.assertEqual(result['status'], 'optimal')
                self.assertAlmostEqual(result['objective'], brute(cover, costs, penalty), places=10)
                self.assertEqual(result['lower_bound'], result['objective'])

    def test_directed_metric(self):
        rng = np.random.default_rng(14)
        costs = rng.uniform(.1, 3., (5, 5))
        np.fill_diagonal(costs, 0.)
        for k in range(5):
            costs = np.minimum(costs, costs[:, k, None]+costs[None, k, :])
        cover = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], bool)
        out = solve_metric(cover, costs, .75)
        self.assertEqual(out['status'], 'optimal')
        self.assertAlmostEqual(out['objective'], brute(cover, costs, .75), places=10)

    def test_timeout_unsupported_and_invalid_are_never_optimal(self):
        cover = np.eye(3, dtype=bool)
        costs = np.ones((4, 4))-np.eye(4)
        for kwargs, status in ((dict(time_limit_s=0.), 'time_limit'),
                               (dict(max_tasks=2), 'unsupported'),
                               (dict(penalty=-1.), 'invalid_input')):
            out = solve_metric(cover, costs, **kwargs)
            self.assertEqual(out['status'], status)
            self.assertFalse(out['optimal'])
        bad = costs.copy(); bad[0, 1] = 9.
        self.assertEqual(solve_metric(cover, bad)['status'], 'invalid_input')

    def test_disconnected_and_empty(self):
        costs = np.array([[0., np.inf], [np.inf, 0.]])
        self.assertEqual(solve_metric([[1]], costs)['status'], 'infeasible')
        out = solve_metric(np.zeros((0, 1), bool), costs)
        self.assertEqual(out['route'], [0, 0])
        self.assertEqual(out['objective'], 0.)

    def test_timeout_keeps_valid_incumbent_without_claiming_optimum(self):
        out = solve_metric(np.eye(2), np.ones((3, 3))-np.eye(3),
                           time_limit_s=0., incumbent=[0, 1, 2, 0])
        self.assertEqual(out['route'], [0, 1, 2, 0])
        self.assertEqual(out['status'], 'time_limit')
        self.assertEqual(out['lower_bound'], 0.)
        self.assertEqual(out['objective'], 4.5)

    def test_duplicate_bad_and_incomplete_incumbents_rejected(self):
        for route in ([0, 1, 1, 2, 0], [0, 1, 0], [0, 2, 1, 4, 0]):
            self.assertEqual(solve_metric(np.eye(2), np.ones((3, 3))-np.eye(3),
                                         incumbent=route)['status'], 'invalid_input')

    def test_mid_dp_timeout_reports_valid_conservative_bound(self):
        rng = np.random.default_rng(71)
        xy = rng.random((6, 2))
        costs = np.linalg.norm(xy[:, None]-xy[None, :], axis=2)
        cover = np.column_stack([np.eye(3, dtype=bool), np.array([[1, 0], [1, 1], [0, 1]])])
        clock = itertools.count()
        with patch('rc3_v4.exact.time.perf_counter', side_effect=lambda: next(clock)*.0001):
            out = solve_metric(cover, costs, time_limit_s=.001,
                               incumbent=[0, 1, 2, 3, 0])
        optimum = brute(cover, costs, .75)
        self.assertEqual(out['status'], 'time_limit')
        self.assertTrue(out['metric_verified'])
        self.assertGreater(out['expanded_masks'], 0)
        self.assertLessEqual(out['lower_bound'], optimum+1e-10)
        self.assertGreaterEqual(out['objective'], optimum-1e-10)


if __name__ == '__main__':
    unittest.main()
