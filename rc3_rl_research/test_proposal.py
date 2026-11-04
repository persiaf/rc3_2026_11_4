"""Regression tests for shared menus and actual local repair proposals."""
from __future__ import annotations

import itertools
import unittest
from dataclasses import asdict
from unittest.mock import patch

import numpy as np

from rc3_fast.solver import Problem, route_objective
from rc3_improve.repair import repair_window, route_is_valid

from .action_builder import local_context, build_actions
from .repair_proposal import _exact_replacement, propose
from .types import Action, SearchConfig


def make_problem(coverage, points=None):
    coverage = np.asarray(coverage, dtype=bool)
    n, m = coverage.shape
    if points is None:
        points = np.c_[np.arange(m+1, dtype=float), np.zeros(m+1)]
    points = np.asarray(points, dtype=float)
    costs = np.linalg.norm(points[:, None]-points[None], axis=2)
    reachability = np.zeros((n, m, 2), dtype=bool)
    reachability[:, :, 0] = coverage
    scene = dict(tasks=[dict(x=0., y=0., z=1.) for _ in range(n)],
                 candidates=[dict(x=float(x), y=float(y), theta=0.) for x, y in points[1:]],
                 x_lines=[0., float(max(points[:, 0].max(), 1.))],
                 y_lines=[0., float(max(points[:, 1].max(), 1.))])
    return Problem(scene=scene, F=reachability,
                   arm_dist=np.zeros((n, m, 2)), sparse=[],
                   nav={"costs": costs}, preprocessing_s=0.)


class ExactProposalCoreTests(unittest.TestCase):
    def test_required_tasks_use_entire_retained_route(self):
        problem = make_problem([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
        context = local_context(problem, [0, 1, 2, 3, 0], 1, 2)
        self.assertEqual(context["required"].tolist(), [])
        self.assertEqual(context["retained_counts"].tolist(), [1, 1, 1])

    def test_local_ban_does_not_modify_shared_reachability(self):
        problem = make_problem([[1, 1]])
        before = problem.F.tobytes()
        context = local_context(problem, [0, 1, 0], 1, 2, forbidden_node=1)
        self.assertEqual(context["pool"].tolist(), [2])
        self.assertEqual(problem.F.tobytes(), before)

    def test_sparse_dp_matches_frozen_repair_on_random_small_problems(self):
        rng = np.random.default_rng(417)
        for _ in range(24):
            coverage = rng.random((5, 7)) < .4
            coverage[np.arange(5), np.arange(5)] = True
            problem = make_problem(coverage, rng.uniform(-3, 3, (8, 2)))
            route = [0, 1, 2, 3, 4, 5, 0]
            for start, end in ((1, 3), (2, 5), (1, 6)):
                context = local_context(problem, route, start, end)
                replacement, reason, _ = _exact_replacement(problem, route, context)
                self.assertIsNone(reason)
                raw = route[:start]+replacement+route[end:]
                actual = raw if route_objective(problem, raw) < route_objective(problem, route)-1e-8 else route
                expected = repair_window(problem, route, dict(start=start, end=end))
                self.assertEqual(actual, expected)

    def test_exact_replacement_matches_exhaustive_small_reference(self):
        problem = make_problem([[1, 0, 0, 1, 0], [0, 1, 0, 1, 1], [0, 0, 1, 0, 1]],
                               [[0, 0], [1, 0], [0, 1], [2, 2], [1, 1], [2, 1]])
        route = [0, 1, 2, 3, 0]
        context = local_context(problem, route, 1, 4)
        replacement, reason, counts = _exact_replacement(problem, route, context)
        self.assertIsNone(reason)
        actual = route_objective(problem, [0]+replacement+[0])
        costs = []
        for length in range(1, 6):
            for permutation in itertools.permutations(range(1, 6), length):
                candidate = [0, *permutation, 0]
                if route_is_valid(problem, candidate):
                    costs.append(route_objective(problem, candidate))
        self.assertAlmostEqual(actual, min(costs))
        self.assertGreater(counts["dp_states"], 0)
        self.assertGreater(counts["dp_transitions"], 0)


class PublicProposalTests(unittest.TestCase):
    def test_forced_replacement_really_returns_worse_feasible_route(self):
        problem = make_problem([[1, 1]], [[0, 0], [1, 0], [4, 0]])
        source_f, source_d = problem.F.tobytes(), problem.nav["costs"].tobytes()
        problem.F.flags.writeable = False
        problem.nav["costs"].flags.writeable = False
        route = [0, 1, 0]
        outcome = propose(problem, route, Action("replace", 1, 2, 1))
        self.assertTrue(outcome.valid)
        self.assertTrue(outcome.changed)
        self.assertEqual(outcome.route, [0, 2, 0])
        self.assertGreater(route_objective(problem, outcome.route), route_objective(problem, route))
        self.assertEqual(route, [0, 1, 0])
        self.assertEqual(problem.F.tobytes(), source_f)
        self.assertEqual(problem.nav["costs"].tobytes(), source_d)
        self.assertEqual(outcome.counters,
                         dict(attempts=1, candidate_checks=2, dp_states=1, dp_transitions=2))
        self.assertEqual(outcome.work_units, 6)

    def test_improve_preserves_immediate_improvement_semantics(self):
        problem = make_problem([[1, 1]], [[0, 0], [1, 0], [4, 0]])
        result = propose(problem, [0, 1, 0], Action("improve", 1, 2))
        self.assertTrue(result.valid)
        self.assertFalse(result.changed)
        self.assertEqual(result.route, [0, 1, 0])
        self.assertEqual(result.diagnostics["outcome"], "no_immediate_improvement")

    def test_replacement_can_remove_a_now_redundant_outside_stop(self):
        problem = make_problem([[1, 1, 0], [0, 1, 1]])
        result = propose(problem, [0, 1, 3, 0], Action("replace", 1, 2, 1))
        self.assertTrue(result.valid)
        self.assertEqual(result.raw_route, [0, 2, 3, 0])
        self.assertEqual(result.route, [0, 2, 0])
        self.assertEqual(result.diagnostics["outside_removed_nodes"], [3])
        self.assertEqual(result.diagnostics["required_tasks"], [0])

    def test_failures_charge_work_and_do_not_disguise_original_as_replace(self):
        problem = make_problem([[1]])
        result = propose(problem, [0, 1, 0], Action("replace", 1, 2, 1))
        self.assertFalse(result.valid)
        self.assertIsNone(result.route)
        self.assertEqual(result.failure_reason, "required_tasks_uncovered")
        self.assertEqual(result.work_units, 2)
        malformed = propose(problem, [0, 1, 0], Action("replace", 1, 2, None))
        self.assertFalse(malformed.valid)
        self.assertEqual(malformed.work_units, 1)

    def test_caps_fail_before_dp_and_charged_scan_is_reproducible(self):
        problem = make_problem([[1, 1, 1]])
        with patch("rc3_rl_research.repair_proposal._exact_replacement") as solve:
            result = propose(problem, [0, 1, 0], Action("replace", 1, 2, 1),
                             config=SearchConfig(max_candidates=1))
        solve.assert_not_called()
        self.assertEqual(result.failure_reason, "candidate_cap_exceeded")
        self.assertEqual(result.work_units, 4)

    def test_nonfinite_boundary_failure_accounts_for_dp_query(self):
        problem = make_problem([[1, 0, 0], [0, 1, 1]])
        problem.nav["costs"][1, 3] = np.inf
        result = propose(problem, [0, 1, 2, 0], Action("replace", 2, 3, 2))
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, "no_finite_local_path")
        self.assertEqual(result.counters["dp_transitions"], 1)
        self.assertEqual(result.work_units, 5)

    def test_no_required_tasks_is_a_valid_charged_deletion(self):
        problem = make_problem([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
        result = propose(problem, [0, 1, 2, 3, 0], Action("replace", 1, 2, 1))
        self.assertTrue(result.valid)
        self.assertEqual(result.route, [0, 2, 3, 0])
        self.assertEqual(result.work_units, 1)


class SharedMenuTests(unittest.TestCase):
    def test_menu_order_mask_and_context_replay_without_pre_solving(self):
        problem = make_problem([[1, 0, 1, 0], [0, 1, 0, 1]])
        args = dict(remaining_work=100000, steps_remaining=8, problem_id="fixture")
        with patch("rc3_rl_research.repair_proposal._exact_replacement") as solve:
            first = build_actions(problem, [0, 1, 2, 0], **args)
            replay = build_actions(problem, [0, 1, 2, 0], **args)
        solve.assert_not_called()
        self.assertEqual(asdict(first), asdict(replay))
        self.assertEqual([a.kind for a in first.actions],
                         ["improve", "replace", "improve", "replace", "stop"])
        self.assertTrue(all(first.mask))
        self.assertTrue(all(len(row) == len(first.feature_names) for row in first.features))
        self.assertEqual(first.work_units, first.diagnostics["contexts_evaluated"]
                         + first.diagnostics["candidate_checks"])

    def test_identity_includes_full_route_budget_history_and_constraint_input(self):
        problem = make_problem([[1, 0, 0], [0, 1, 1]])
        args = dict(remaining_work=100000, steps_remaining=8)
        first = build_actions(problem, [0, 1, 2, 0], **args)
        reordered = build_actions(problem, [0, 2, 1, 0], **args)
        smaller = build_actions(problem, [0, 1, 2, 0], remaining_work=99999, steps_remaining=8)
        historical = build_actions(problem, [0, 1, 2, 0], history=(first.action_context_keys[0],), **args)
        problem.F[0, 2, 0] = True
        altered = build_actions(problem, [0, 1, 2, 0], **args)
        self.assertEqual(len({m.context_key for m in (first, reordered, smaller, historical, altered)}), 5)

    def test_budget_mask_stop_and_menu_caps(self):
        problem = make_problem(np.eye(20, dtype=bool))
        route = [0, *range(1, 21), 0]
        menu = build_actions(problem, route, remaining_work=100000, steps_remaining=8)
        self.assertEqual(len(menu.actions), 17)
        self.assertEqual(menu.actions[-1].kind, "stop")
        self.assertGreater(menu.diagnostics["omitted_by_menu_cap"], 0)
        exhausted = build_actions(problem, route, remaining_work=1, steps_remaining=8)
        self.assertFalse(any(exhausted.mask[:-1]))
        self.assertTrue(exhausted.mask[-1])
        with self.assertRaises(ValueError):
            build_actions(problem, route, remaining_work=100000, steps_remaining=8,
                          config=SearchConfig(max_tasks=9))

    def test_menu_exposes_global_and_window_outside_coverage(self):
        problem = make_problem([[1, 0, 1], [0, 1, 0], [0, 0, 1]])
        menu = build_actions(problem, [0, 1, 2, 3, 0], remaining_work=100000, steps_remaining=8)
        self.assertEqual(menu.diagnostics["coverage_counts"], [2, 1, 1])
        index = next(i for i, a in enumerate(menu.actions) if a.kind == "replace" and a.start == 1)
        detail = menu.diagnostics["action_details"][index]
        self.assertEqual(detail["required_tasks"], [])
        self.assertEqual(detail["retained_coverage_counts"], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
