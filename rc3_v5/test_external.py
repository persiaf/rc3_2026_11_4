import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import external
from .external import (check_bundle, save_bundle, load_bundle, plan_external,
                       validate_external, make_certificate, prepare_external, export_external, seal_draft)
from .external_example import make_demo, reseal


class ExternalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = make_demo()
        cls.result = plan_external(cls.bundle, 'detour')
        assert cls.result['status'] == 'complete', cls.result
        cls.plan = cls.result['plans']['detour']

    def test_json_npz_roundtrip_and_no_proxy_validation(self):
        draft = dict(scene=self.bundle['scene'], **{r: {k: self.bundle[r][k] for k in ('F', 'source')}
                                                    for r in ('planning', 'audit')})
        self.assertEqual(self.bundle, seal_draft(draft))
        with tempfile.TemporaryDirectory() as directory:
            for ext in ('json', 'npz'):
                path = Path(directory)/('fixture.'+ext)
                save_bundle(self.bundle, path)
                self.assertEqual(self.bundle, load_bundle(path))
        with (patch('rc3_fast.solver.vectorized_feasibility', side_effect=AssertionError('proxy prohibited')),
              patch('rc3_baseline.validator.proxy_reachable', side_effect=AssertionError('proxy prohibited'))):
            self.assertEqual(plan_external(self.bundle)['status'], 'complete')

    def test_height_arm_cost_and_callback(self):
        self.assertEqual({s['pose']['h'] for s in self.plan['service_stops']}, {.4, .7})
        self.assertEqual({a['arm'] for a in self.plan['assignments']}, {'L', 'R'})
        self.assertTrue(self.result['diagnostics']['cost_nonincreasing'])
        calls = []
        def checker(task, candidate, arm, cert):
            calls.append((task['id'], candidate['id'], arm))
            return True
        check = validate_external(self.bundle, self.plan, checker)
        self.assertTrue(check['valid_model_plan'])
        self.assertEqual(check['service_checker_calls'], 4)
        self.assertEqual(len(calls), 4)
        self.assertFalse(check['built_in_physical_ik_recheck'])
        self.assertFalse(validate_external(self.bundle, self.plan, lambda *args: False)['valid_model_plan'])

    def test_tampering_F_scene_order_metadata_and_plan_association(self):
        for mutate in (
            lambda b: b['planning']['F'][0][0].__setitem__(0, 1-b['planning']['F'][0][0][0]),
            lambda b: b['scene']['tasks'].reverse(),
            lambda b: b['scene']['candidates'][0].__setitem__('h', .5),
            lambda b: b['audit']['source'].__setitem__('generator_version', 'tampered'),
            lambda b: b['audit'].__setitem__('arm_order', ['R', 'L']),
        ):
            b = copy.deepcopy(self.bundle)
            mutate(b)
            with self.assertRaises(ValueError):
                check_bundle(b)
        p = copy.deepcopy(self.plan)
        p['audit_certificate_sha256'] = '0'*64
        self.assertFalse(validate_external(self.bundle, p)['valid_model_plan'])

    def test_audit_false_positive_rejected_not_physical_infeasibility(self):
        A = np.asarray(self.bundle['audit']['F'], dtype=bool)
        A[1] = False
        b = reseal(self.bundle, A=A)
        result = plan_external(b)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['physical_infeasibility_claim'])
        self.assertEqual(result['pending_task_ids'], [0, 1, 2, 3])
        p = copy.deepcopy(self.plan)
        p['audit_certificate_sha256'] = b['audit']['certificate_sha256']
        self.assertIn('audit F rejects', validate_external(b, p)['errors'][0])

    def test_invalid_records_and_geometric_tampering_fail_closed(self):
        cases = [
            lambda p: p['assignments'][0].__setitem__('arm', 'both'),
            lambda p: p['assignments'].append(copy.deepcopy(p['assignments'][0])),
            lambda p: p['service_stops'][0]['pose'].__setitem__('h', 99.),
            lambda p: p.__setitem__('objective_length_plus_stops_m', 0.),
            lambda p: p['route_polyline'].__setitem__(1, [1.1, .95]),
            lambda p: p['route_leg_costs_m'].__setitem__(0, float('nan')),
            lambda p: p['visit_order_candidate_ids'].reverse(),
        ]
        for mutate in cases:
            p = copy.deepcopy(self.plan)
            mutate(p)
            self.assertFalse(validate_external(self.bundle, p)['valid_model_plan'])

    def test_penalty_is_bound_to_independent_request(self):
        p = copy.deepcopy(self.plan)
        p['stop_penalty_equivalent_m'] = 0.
        p['objective_length_plus_stops_m'] = p['route_length_m']
        check = validate_external(self.bundle, p)
        self.assertFalse(check['valid_model_plan'])
        self.assertIn('independent planning request', check['errors'][0])
        result = plan_external(self.bundle, penalty=1.5)
        self.assertEqual(result['status'], 'complete')
        custom = result['plans']['fixed3']
        self.assertTrue(validate_external(self.bundle, custom, expected_penalty=1.5)['valid_model_plan'])
        self.assertFalse(validate_external(self.bundle, custom)['valid_model_plan'])
        for penalty in (None, -1., float('nan'), True):
            self.assertFalse(validate_external(self.bundle, custom, expected_penalty=penalty)['valid_model_plan'])

    def test_cooperative_untrusted_assets_and_wrong_shape_rejected(self):
        b = copy.deepcopy(self.bundle)
        b['scene']['tasks'][0]['service_mode'] = 'cooperative_bimanual'
        with self.assertRaises(ValueError):
            check_bundle(b)
        for field, value in [('robot_model_id', 'different'),
                             ('producer_id', b['planning']['source']['producer_id'])]:
            b = copy.deepcopy(self.bundle)
            b['audit']['source'][field] = value
            b['audit'] = make_certificate(b['scene'], b['audit']['F'], b['audit']['source'], role='audit')
            with self.assertRaises(ValueError):
                check_bundle(b)
        F = np.zeros((4, 32, 3), dtype=bool)
        with self.assertRaises(ValueError):
            make_certificate(self.bundle['scene'], F, self.bundle['planning']['source'], role='planning')

    def test_empty_tasks_valid_and_missing_coverage_failed(self):
        b = copy.deepcopy(self.bundle)
        b['scene']['tasks'] = []
        b = reseal(b, [], [])
        result = plan_external(b, 'detour')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['plans']['detour']['objective_length_plus_stops_m'], 0)
        zeros = np.zeros_like(np.asarray(self.bundle['planning']['F']))
        self.assertEqual(plan_external(reseal(self.bundle, zeros, zeros))['status'], 'failed')

    def test_repair_exception_keeps_certified_anchor(self):
        with patch('rc3_v3.environment.Env.step', side_effect=ValueError('injected repair fault')):
            result = plan_external(self.bundle, 'detour')
        self.assertEqual(result['status'], 'complete')
        self.assertTrue(result['diagnostics']['fallback_used'])
        self.assertTrue(validate_external(self.bundle, result['plans']['detour'])['valid_model_plan'])
        self.assertEqual(result['diagnostics']['objective_m'], result['diagnostics']['reference_objective_m'])


class CommonArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import rc3_v4.external as frozen
        cls.bundle = make_demo()
        cls.F, cls.A = external.check_bundle(cls.bundle)
        cls.original_plans = {method: frozen.plan_external(cls.bundle, method)['plans'][method]
                              for method in ('fixed3', 'detour')}

    def arm_case(self, planning_arms, audit_arms):
        f, a = self.F.copy(), self.A.copy()
        allowed = self.F[0, :, 0].copy()
        f[0] = False
        a[0] = False
        for arm in planning_arms:
            f[0, :, 'LR'.index(arm)] = allowed
        for arm in audit_arms:
            a[0, :, 'LR'.index(arm)] = allowed
        return reseal(self.bundle, f, a)

    def check_arm(self, planning_arms, audit_arms, wanted):
        case = self.arm_case(planning_arms, audit_arms)
        result = external.plan_external(case, 'fixed3')
        self.assertEqual(result['status'], 'complete', result)
        plan = result['plans']['fixed3']
        selected = next(a for a in plan['assignments'] if a['task_id'] == 0)
        self.assertEqual(selected['arm'], wanted)
        self.assertTrue(external.validate_external(case, plan)['valid_model_plan'])
        return case, result

    def test_l_only_intersection(self):
        self.check_arm('LR', 'L', 'L')

    def test_r_only_intersection_repairs_counterexample(self):
        self.check_arm('LR', 'R', 'R')

    def test_both_arms_keeps_deterministic_l_preference(self):
        self.check_arm('LR', 'LR', 'L')

    def test_no_intersection_fails_and_preserves_all_task_ids(self):
        calls = []
        case = self.arm_case('L', 'R')
        def checker(*args):
            calls.append(args)
            return True
        result = external.plan_external(case, service_checker=checker)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['pending_task_ids'], [0, 1, 2, 3])
        self.assertFalse(result['physical_infeasibility_claim'])
        self.assertEqual(calls, [])

    def test_audit_cannot_enable_planning_forbidden_arm(self):
        self.check_arm('L', 'LR', 'L')

    def test_task_can_be_deferred_to_later_visited_stop(self):
        order = self.original_plans['fixed3']['visit_order_candidate_ids']
        first, last = order[0], order[-1]
        f, a = self.F.copy(), self.A.copy()
        f[0] = False
        a[0] = False
        f[0, [first, last], 0] = True
        a[0, last, 0] = True
        case = reseal(self.bundle, f, a)
        problem = external.prepare_external(case)
        plan = external.export_external(problem, [0]+[j+1 for j in order]+[0], case)
        service = next(v for v in plan['assignments'] if v['task_id'] == 0)
        self.assertEqual(service['candidate_id'], last)
        self.assertTrue(external.validate_external(case, plan)['valid_model_plan'])

    def test_callback_rejection_is_not_reinterpreted_as_other_arm_success(self):
        calls = []
        case = self.arm_case('LR', 'LR')
        def checker(task, candidate, arm, certificate):
            calls.append((task['id'], candidate['id'], arm))
            return not (task['id'] == 0 and arm == 'L')
        result = external.plan_external(case, service_checker=checker)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['pending_task_ids'], [0, 1, 2, 3])
        self.assertTrue(any(t == 0 for t, _, _ in calls))
        self.assertEqual({arm for t, _, arm in calls if t == 0}, {'L'})

    def test_callback_exception_still_fails_closed(self):
        def checker(*args):
            raise RuntimeError('injected checker error')
        result = external.plan_external(self.bundle, 'detour', service_checker=checker)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['pending_task_ids'], [0, 1, 2, 3])
        self.assertFalse(result['physical_infeasibility_claim'])

    def test_final_callback_rejection_returns_certified_anchor(self):
        calls = []
        def checker(task, candidate, arm, certificate):
            calls.append((task['id'], candidate['id'], arm))
            return len(calls) <= 4
        result = external.plan_external(self.bundle, 'detour', service_checker=checker)
        self.assertEqual(result['status'], 'complete', result)
        self.assertTrue(result['diagnostics']['fallback_used'])
        self.assertEqual(len(calls), 5)
        self.assertEqual(result['diagnostics']['objective_m'],
                         result['diagnostics']['reference_objective_m'])
        self.assertTrue(external.validate_external(self.bundle, result['plans']['detour'])['valid_model_plan'])

    def test_existing_demo_plans_unchanged(self):
        for method in ('fixed3', 'detour'):
            with self.subTest(method=method):
                result = external.plan_external(self.bundle, method)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(result['plans'][method], self.original_plans[method])

    def test_independent_final_validator_still_rejects_disallowed_arm(self):
        case, result = self.check_arm('LR', 'R', 'R')
        plan = copy.deepcopy(result['plans']['fixed3'])
        for a in plan['assignments']:
            if a['task_id'] == 0:
                a['arm'] = 'L'
        for stop in plan['service_stops']:
            for a in stop['tasks']:
                if a['task_id'] == 0:
                    a['arm'] = 'L'
        check = external.validate_external(case, plan)
        self.assertFalse(check['valid_model_plan'])
        self.assertIn('audit F rejects', check['errors'][0])


def later_stop_counterexample():
    """Synthetic tables where F-only canonicalization removes a required A stop."""
    bundle = make_demo()
    F = np.zeros_like(np.asarray(bundle['planning']['F'], dtype=bool))
    A = F.copy()
    first, last = 0, 23
    # F says every task can finish at first. A defers task 0 to last, which
    # otherwise adds no F coverage and is wrongly dropped by F-only pruning.
    F[:, first, 0] = True
    F[0, last, 0] = True
    A[1:, first, 0] = True
    A[0, last, 0] = True
    return reseal(bundle, F, A), [0, first+1, last+1, 0]


class CommonCoverageTests(unittest.TestCase):
    def test_canonicalization_retains_only_audited_later_stop(self):
        from rc3_v4.external import prepare_external as old_prepare
        from rc3_improve.environment import canonical_route
        case, route = later_stop_counterexample()
        legacy = old_prepare(case)
        original_F, original_sparse = legacy.F.copy(), copy.deepcopy(legacy.sparse)
        self.assertEqual(canonical_route(legacy, route), [0, 1, 0])
        plan = export_external(legacy, route, case)
        self.assertEqual(plan['visit_order_candidate_ids'], [0, 23])
        self.assertEqual(next(a for a in plan['assignments'] if a['task_id'] == 0)['candidate_id'], 23)
        self.assertTrue(validate_external(case, plan)['valid_model_plan'])
        np.testing.assert_array_equal(legacy.F, original_F)
        self.assertEqual(legacy.sparse, original_sparse)

    def test_prepare_intersection_and_sparse_do_not_mutate_bundle(self):
        case, _ = later_stop_counterexample()
        untouched = copy.deepcopy(case)
        F, A = check_bundle(case)
        problem = prepare_external(case)
        np.testing.assert_array_equal(problem.F, F & A)
        self.assertEqual(case, untouched)
        for i, record in enumerate(problem.sparse):
            self.assertEqual(record['left'], np.flatnonzero((F & A)[i, :, 0]).tolist())
            self.assertEqual(record['right'], np.flatnonzero((F & A)[i, :, 1]).tolist())

    def test_anchor_and_repair_use_common_coverage_and_exact_export_cost(self):
        from rc3_fast.solver import route_objective
        case, _ = later_stop_counterexample()
        for method in ('fixed3', 'detour'):
            for penalty in (.75, 1.5):
                with self.subTest(method=method, penalty=penalty):
                    result = plan_external(case, method, penalty=penalty)
                    self.assertEqual(result['status'], 'complete', result)
                    plan = result['plans'][method]
                    route = [0]+[j+1 for j in plan['visit_order_candidate_ids']]+[0]
                    self.assertAlmostEqual(plan['objective_length_plus_stops_m'],
                                           route_objective(prepare_external(case, penalty), route))
                    self.assertLessEqual(result['diagnostics']['objective_m'],
                                         result['diagnostics']['reference_objective_m']+1e-8)
                    self.assertTrue(validate_external(case, plan, expected_penalty=penalty)['valid_model_plan'])
                    self.assertTrue(result['diagnostics']['audit_table_used_for_planning'])

    def test_export_rejects_problem_scene_not_bound_to_bundle(self):
        case, route = later_stop_counterexample()
        problem = prepare_external(case)
        problem.scene['candidates'][0]['h'] += .1
        with self.assertRaisesRegex(ValueError, 'scene differs'):
            export_external(problem, route, case)

    def test_export_rejects_caller_enabling_planning_forbidden_entry(self):
        case, route = later_stop_counterexample()
        problem = prepare_external(case)
        problem.F[0, 0, 1] = True
        with self.assertRaisesRegex(ValueError, 'planning-forbidden'):
            export_external(problem, route, case)

    def test_export_preserves_stricter_caller_coverage(self):
        case, route = later_stop_counterexample()
        problem = prepare_external(case)
        problem.F[0] = False
        with self.assertRaisesRegex(ValueError, 'complete valid route'):
            export_external(problem, route, case)

    def test_mismatched_anchor_score_never_becomes_certified_fallback(self):
        from rc3_v3.environment import reference_candidates
        def wrong_score(problem):
            candidates, failures = reference_candidates(problem)
            return [(score+1., recipe, route) for score, recipe, route in candidates], failures
        with patch('rc3_v3.environment.reference_candidates', side_effect=wrong_score):
            result = plan_external(make_demo())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['plans'], {})
        self.assertEqual(result['pending_task_ids'], [0, 1, 2, 3])
        self.assertIn('abstract/export cost mismatch', result['detail'])
        self.assertFalse(result['physical_infeasibility_claim'])

    def test_mismatched_repair_export_cost_keeps_certified_anchor(self):
        original = external.export_external
        exported = []
        def wrong_repair_cost(*args, **kwargs):
            plan = original(*args, **kwargs)
            exported.append(copy.deepcopy(plan))
            if len(exported) > 1:
                plan['objective_length_plus_stops_m'] += .01
            return plan
        with patch('rc3_v5.external.export_external', side_effect=wrong_repair_cost):
            result = plan_external(make_demo(), 'detour')
        self.assertEqual(len(exported), 2)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['plans']['detour'], exported[0])
        self.assertTrue(result['diagnostics']['fallback_used'])
        self.assertIn('abstract/export cost mismatch', result['diagnostics']['failures'][-1]['error'])


if __name__ == '__main__':
    unittest.main()

