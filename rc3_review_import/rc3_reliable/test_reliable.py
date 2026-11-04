"""Targeted adapter behavior tests; original experiments are not overwritten."""
import copy
import unittest
from unittest.mock import patch

import reliable
from rc3_baseline import baseline
from rc3_fast.scenes import make_complex_scene


class ReliabilityTests(unittest.TestCase):
    def test_empty_tasks_return_explicit_valid_empty_plan(self):
        result = reliable.plan_reliably(baseline.make_scene(1, 1, 3700009000, counts=[0]))
        self.assertEqual(result['code'], 'empty_task_set')
        self.assertEqual(result['certificate']['attempts'], 0)
        self.assertEqual(result['certificate']['returned_objective_m'], 0)
        self.assertTrue(result['plans']['reliable']['validation']['valid_model_plan'])

    def test_no_service_candidate_keeps_original_task(self):
        scene = baseline.make_scene(1, 1, 3700009001, counts=[1])
        scene['tasks'][0]['z'] = 3.0
        result = reliable.plan_reliably(scene)
        self.assertEqual(result['code'], 'no_service_candidate')
        self.assertEqual(result['task_ids'], [0])
        self.assertEqual(result['pending_task_ids'], [0])
        self.assertEqual(result['precheck']['no_service_candidate_task_ids'], [0])
        self.assertTrue(result['certificate']['no_physical_infeasibility_claim'])
        self.assertFalse(result['plans'])

    def test_graph_disconnection_is_distinct_from_arm_proxy_failure(self):
        scene = baseline.make_scene(1, 1, 3700009002, counts=[1])
        table = scene['tables'][0]
        scene['tasks'][0].update(x=table['cx'], y=table['cy'])
        for candidate in scene['candidates']:
            candidate['corridor_anchor'] = [table['cx'], table['cy']]
        result = reliable.plan_reliably(scene)
        self.assertEqual(result['code'], 'disconnected')
        self.assertEqual(result['precheck']['arm_feasible_tasks'], 1)
        self.assertEqual(result['precheck']['navigable_serviceable_tasks'], 0)
        self.assertEqual(result['pending_task_ids'], [0])

    def test_bad_optional_advice_preserves_certified_anchor(self):
        scene = make_complex_scene(4, 3, 3700009003, 'blocked')
        def broken_model(_scene):
            raise RuntimeError('injected suggestion failure')
        for advice in ([999, -1, None, 'bad'], broken_model):
            result = reliable.plan_reliably(scene, max_recipes=4, suggested_recipes=advice)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['certificate']['attempts'], 1)
            self.assertEqual(result['certificate']['reference_objective_m'],
                             result['certificate']['returned_objective_m'])
            self.assertTrue(any(t['event'].startswith('suggestion_') for t in result['trace']))
            self.assertTrue(result['certificate']['fallback_used'])

    def test_failed_or_invalid_candidate_cannot_replace_incumbent(self):
        scene = make_complex_scene(4, 3, 3700009004, 'mixed')
        original = reliable.solve_route
        for failure in ('exception', 'invalid_route'):
            def injected(problem, recipe, failure=failure):
                if recipe == 0:
                    if failure == 'exception':
                        raise RuntimeError('injected solver failure')
                    return [0, 0]
                return original(problem, recipe=recipe)
            with patch('reliable.solve_route', side_effect=injected):
                result = reliable.plan_reliably(scene, mode='quality')
            self.assertEqual(result['status'], 'complete')
            rejected = [t for t in result['trace'] if t['event'] == 'candidate_failed']
            self.assertEqual(len(rejected), 1)
            self.assertEqual(rejected[0]['incumbent_before_m'], rejected[0]['incumbent_after_m'])
            self.assertTrue(result['certificate']['cost_not_worse_than_reference'])
            self.assertTrue(result['certificate']['fallback_used'])

    def test_cost_certificate_names_anchor_and_is_monotone(self):
        scene = make_complex_scene(4, 3, 3700009005, 'blocked')
        for reference, expected in [('fixed1', 'fixed1_recipe_1'),
                                     ('legacy', 'legacy_coverage_cost_2opt')]:
            result = reliable.plan_reliably(scene, mode='quality', reference=reference)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['certificate']['reference'], expected)
            costs = [t['incumbent_after_m'] for t in result['trace'] if t['event'] == 'candidate']
            self.assertTrue(all(b <= a for a, b in zip(costs, costs[1:])))
            self.assertLessEqual(result['certificate']['returned_objective_m'],
                                 result['certificate']['reference_objective_m'])

    def test_one_candidate_budget_returns_only_anchor(self):
        scene = make_complex_scene(4, 3, 3700009006, 'regular')
        result = reliable.plan_reliably(scene, mode='quality', max_recipes=1)
        self.assertEqual(result['certificate']['attempts'], 1)
        self.assertEqual(result['certificate']['winner'], 1)

    def test_variable_lift_metadata_is_explicitly_unsupported(self):
        scene = baseline.make_scene(1, 1, 3700009007, counts=[1])
        scene['candidates'][0]['h'] += 0.1
        result = reliable.plan_reliably(scene)
        self.assertEqual(result['code'], 'input_rejected')
        self.assertIn('variable lift is unsupported', result['detail'])
        self.assertEqual(result['task_ids'], [0])


if __name__ == '__main__':
    unittest.main()
