import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

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


if __name__ == '__main__':
    unittest.main()
