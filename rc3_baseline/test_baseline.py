"""Behavioral regressions for coverage, safe navigation and saved-plan validation.

Run: python -B -m unittest discover -s rc3_baseline -p test_baseline.py -v
"""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

if __package__:
    from . import baseline
    from .validator import validate, validate_file
else:
    import baseline
    from validator import validate, validate_file


class PlanningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output, cls.rows = baseline.run_case(4, 3, 0)

    def test_complete_routes_and_service_events(self):
        for method, plan in self.output['plans'].items():
            with self.subTest(method=method):
                result = validate(self.output['scene'], plan, self.output['feasibility'])
                self.assertTrue(result['valid_model_plan'], result['errors'])
                self.assertTrue(result['all_tasks_served'])
                self.assertEqual(len(plan['assignments']), len(self.output['scene']['tasks']))
                self.assertEqual(len(list(baseline.schedule_rows(
                    {**self.output, 'plans': {method: plan}}))), len(plan['assignments']))

    def test_2opt_preserves_coverage_and_cannot_increase_length(self):
        for method in ('nearest_feasible', 'coverage_cost'):
            before, after = self.output['plans'][method], self.output['plans'][method+'_2opt']
            self.assertEqual(before['selected_candidate_ids'], after['selected_candidate_ids'])
            self.assertEqual(before['assignments'], after['assignments'])
            self.assertLessEqual(after['route_length_m'], before['route_length_m'] + 1e-8)

    def test_2opt_uncrosses_a_known_tour(self):
        points = np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])
        costs = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        result = baseline.improve_2opt([0, 1, 2, 3, 0], costs)
        self.assertAlmostEqual(sum(costs[a, b] for a, b in zip(result, result[1:])), 4.)
        self.assertEqual(sorted(result[1:-1]), [1, 2, 3])

    def test_empty_and_arm_unreachable_tasks_are_retained(self):
        results = baseline.edge_case_tests()
        self.assertEqual(len(results), 2 * len(baseline.METHODS))
        self.assertTrue(all(result['passed'] for result in results))
        scene = baseline.make_scene(1, 1, 0, counts=[1])
        scene['tasks'][0]['z'] = 3.
        matrix, distances, _ = baseline.feasibility(scene)
        for method in baseline.METHODS:
            plan = baseline.solve(scene, matrix, distances, baseline.navigation(scene), method)
            self.assertEqual(plan['status'], 'partial')
            self.assertEqual(plan['unserved_reasons'],
                             [dict(task_id=0, reason='no_arm_feasible_candidate')])

    def test_blocked_dock_is_removed_before_routing(self):
        scene = baseline.make_scene(1, 1, 0, counts=[1])
        table = scene['tables'][0]
        scene['candidates'][0].update(x=table['cx'], y=table['cy'])
        matrix, distances, sparse = baseline.feasibility(scene)
        nav = baseline.navigation(scene)
        self.assertGreater(nav['rejected_edge_count'], 0)
        self.assertFalse(np.isfinite(nav['costs'][0, 1]))
        for method in baseline.METHODS:
            plan = baseline.solve(scene, matrix, distances, nav, method)
            self.assertNotIn(0, plan['selected_candidate_ids'])
            result = validate(scene, plan, sparse)
            self.assertTrue(result['valid_model_plan'], result['errors'])
            self.assertTrue(result['all_tasks_served'])

    def test_arm_feasible_but_disconnected_tasks_are_not_dropped(self):
        scene = baseline.make_scene(1, 1, 0, counts=[1])
        table = scene['tables'][0]
        scene['tasks'][0].update(x=table['cx'], y=table['cy'])
        for dock in scene['candidates']:
            dock['corridor_anchor'] = [table['cx'], table['cy']]
        matrix, distances, sparse = baseline.feasibility(scene)
        self.assertTrue(matrix.any())
        nav = baseline.navigation(scene)
        self.assertFalse(np.isfinite(nav['costs'][0, 1:]).any())
        for method in baseline.METHODS:
            plan = baseline.solve(scene, matrix, distances, nav, method)
            result = validate(scene, plan, sparse)
            self.assertTrue(result['valid_model_plan'], result['errors'])
            self.assertFalse(result['all_tasks_served'])
            self.assertEqual(plan['unserved_task_ids'], [0])
            self.assertEqual(plan['unserved_reasons'][0]['reason'], 'no_navigable_feasible_candidate')
            self.assertEqual(plan['route_polyline'], [scene['start']])

    def test_invalid_depot_is_rejected_even_with_no_tasks(self):
        scene = baseline.make_scene(1, 1, 0, counts=[0], geometry={'aisle_gap': 0.6})
        with self.assertRaisesRegex(ValueError, 'depot'):
            baseline.navigation(scene)

    def test_zero_penalty_and_zero_distance_service(self):
        scene = baseline.make_scene(1, 1, 0, counts=[1])
        table = scene['tables'][0]
        scene['tasks'][0].update(x=table['cx'], y=table['cy'])
        scene['start'] = [scene['candidates'][0]['x'], scene['candidates'][0]['y']]
        matrix, distances, sparse = baseline.feasibility(scene)
        with np.errstate(divide='raise', invalid='raise'):
            plan = baseline.solve(scene, matrix, distances, baseline.navigation(scene), 'coverage_cost', 0)
        self.assertEqual(plan['route_length_m'], 0.)
        result = validate(scene, plan, sparse)
        self.assertTrue(result['valid_model_plan'], result['errors'])
        self.assertTrue(result['all_tasks_served'])

    def test_bad_input_is_explicitly_rejected(self):
        for kwargs in ({'nx': 0}, {'ny': -1}, {'seed': -1}, {'counts': []},
                       {'counts': [4]}, {'geometry': {'aisle_gap': 0}},
                       {'robot': {'arm_max_reach': float('nan')}}, {'robot': {'typo': 1}}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                baseline.make_scene(**({'nx': 1, 'ny': 1, 'seed': 0} | kwargs))
        scene = baseline.make_scene(1, 1, 0)
        matrix, distances, _ = baseline.feasibility(scene)
        nav = baseline.navigation(scene)
        for method, penalty in (('unknown', 0.75), ('coverage_cost', -1), ('coverage_cost', float('inf'))):
            with self.subTest(method=method, penalty=penalty), self.assertRaises(ValueError):
                baseline.solve(scene, matrix, distances, nav, method, penalty)


class ValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output, _ = baseline.run_case(4, 3, 0, methods=['coverage_cost_2opt'])

    def test_tampered_identifiers_order_geometry_and_metrics_are_rejected(self):
        def wrong_leg_boundary(plan):
            plan['route_leg_costs_m'][0] += 0.01
            plan['route_leg_costs_m'][1] -= 0.01

        mutations = {
            'unknown_task': lambda p: p['assignments'][0].update(task_id=999999),
            'unknown_dock_assignment': lambda p: p['assignments'][0].update(candidate_id=999999),
            'unknown_dock_order': lambda p: p['visit_order_candidate_ids'].__setitem__(0, 999999),
            'unknown_arm': lambda p: p['assignments'][0].update(arm='UNKNOWN'),
            'wrong_visit_order': lambda p: p['visit_order_candidate_ids'].reverse(),
            'same_total_wrong_leg_boundary': wrong_leg_boundary,
            'duplicate_assignment': lambda p: p['assignments'].append(copy.deepcopy(p['assignments'][0])),
            'nonfinite_polyline': lambda p: p['route_polyline'][0].__setitem__(0, float('nan')),
            'nonfinite_leg': lambda p: p['route_leg_costs_m'].__setitem__(0, float('inf')),
            'wrong_objective': lambda p: p.update(objective_length_plus_stops_m=0),
            'wrong_service_pose': lambda p: p['service_stops'][0]['pose'].update(theta=99),
            'missing_service_task': lambda p: p['service_stops'][0]['tasks'].pop(),
            'wrong_arrival_distance': lambda p: p['service_stops'][0].update(arrival_distance_m=999),
            'missing_v2_service_stops': lambda p: p.pop('service_stops'),
        }
        for label, mutate in mutations.items():
            with self.subTest(case=label):
                plan = copy.deepcopy(self.output['plans']['coverage_cost_2opt'])
                mutate(plan)
                result = validate(self.output['scene'], plan, self.output['feasibility'])
                self.assertFalse(result['valid_model_plan'], label)
                self.assertTrue(result['errors'])

    def test_stationary_collision_is_not_missed(self):
        scene = baseline.make_scene(1, 1, 0, counts=[0])
        matrix, distances, sparse = baseline.feasibility(scene)
        plan = baseline.solve(scene, matrix, distances, baseline.navigation(scene), 'coverage_cost')
        table = scene['tables'][0]
        scene['start'] = [table['cx'], table['cy']]
        plan['route_polyline'] = [scene['start']]
        result = validate(scene, plan, sparse)
        self.assertFalse(result['valid_model_plan'])
        self.assertGreater(result['collision_count'], 0)

    def test_collision_alone_invalidates_an_otherwise_consistent_closed_route(self):
        scene = baseline.make_scene(1, 1, 0, counts=[0])
        matrix, distances, sparse = baseline.feasibility(scene)
        plan = baseline.solve(scene, matrix, distances, baseline.navigation(scene), 'coverage_cost')
        table = scene['tables'][0]
        point = [table['cx'], table['cy']]
        distance = 2 * math.dist(scene['start'], point)
        plan.update(route_polyline=[scene['start'], point, scene['start']], route_length_m=distance,
                    route_leg_costs_m=[distance], objective_length_plus_stops_m=distance)
        result = validate(scene, plan, sparse)
        self.assertFalse(result['valid_model_plan'])
        self.assertGreater(result['collision_count'], 0)
        self.assertTrue(all('obstacle' in error for error in result['errors']), result['errors'])

    def test_missing_fields_fail_closed(self):
        result = validate({}, {}, [])
        self.assertFalse(result['valid_model_plan'])


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rc3_test_', dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_custom_scale_seed_and_method_have_replayable_outputs(self):
        output_dir = self.root / 'run'
        with contextlib.redirect_stdout(io.StringIO()):
            code = baseline.main(['--scales', '2x1', '--seeds', '1', '--seed-start', '7',
                                  '--methods', 'nearest_feasible', '--no-plots', '--output', str(output_dir)])
        self.assertEqual(code, 0)
        checks = validate_file(output_dir / 'scene_plan_2x1_seed7.json')
        self.assertTrue(all(c['valid_model_plan'] and c['all_tasks_served'] for c in checks))
        manifest = json.loads((output_dir / 'run_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['seeds'], [7])
        self.assertEqual(manifest['plan_count'], 1)
        self.assertTrue((output_dir / 'service_schedule.csv').is_file())
        self.assertEqual(list(output_dir.glob('*.png')), [])

    def test_zero_seeds_and_existing_outputs_are_rejected(self):
        marker = self.root / 'do_not_overwrite.txt'
        marker.write_text('keep', encoding='utf-8')
        for arguments in (['--seeds', '0'], ['--output', str(self.root)]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                baseline.main(arguments)
            self.assertEqual(caught.exception.code, 2)
        self.assertEqual(marker.read_text(encoding='utf-8'), 'keep')

    def test_incomplete_batch_is_saved_and_returns_nonzero(self):
        config = self.root / 'short_arms.json'
        config.write_text(json.dumps({'robot': {'arm_max_reach': 0.2}}), encoding='utf-8')
        output_dir = self.root / 'partial'
        with contextlib.redirect_stdout(io.StringIO()):
            code = baseline.main(['--scales', '1x1', '--seeds', '1', '--no-plots',
                                  '--config', str(config), '--output', str(output_dir)])
        self.assertEqual(code, 2)
        checks = validate_file(output_dir / 'scene_plan_1x1_seed0.json')
        self.assertTrue(all(c['valid_model_plan'] and not c['all_tasks_served'] for c in checks))
        manifest = json.loads((output_dir / 'run_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['status'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
