"""Actual adapter/export/runtime integration; these are G1 checks, not G0 data.

Proxy cases use the existing geometric scene and navigation generator. External
cases deliberately reuse v5 synthetic interoperability/counterexample tables;
they do not establish physical IK or empirical long-horizon learning benefit.
Saved execution/replay evidence goes only to outputs/integration/.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from rc3_baseline.baseline import make_scene
from rc3_baseline.validator import validate
from rc3_fast.scenes import make_complex_scene
from rc3_v5.external import check_bundle, validate_external
from rc3_v5.external_example import make_demo, reseal
from rc3_v5.test_external import later_stop_counterexample

from .adapter import ResearchAdapter
from .runtime_rl import replay_file, run, run_scene
from .types import SearchConfig


OUTPUT = Path(__file__).resolve().parent/'outputs'/'integration'


def save_evidence(name, data):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return path


class RealAdapterIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = make_scene(2, 2, 202611031, counts=[1, 2, 1, 0])
        cls.proxy = ResearchAdapter.from_scene(cls.scene)
        cls.bundle = make_demo()

    def test_proxy_nonempty_certifies_export_and_preserves_source(self):
        scene = copy.deepcopy(self.scene)
        untouched = copy.deepcopy(scene)
        adapter = ResearchAdapter.from_scene(scene)
        certificate = adapter.certify(adapter.anchor.route)
        self.assertEqual(scene, untouched)
        self.assertEqual(len(certificate.plan['assignments']), 4)
        self.assertTrue(validate(scene, certificate.plan, adapter.problem.sparse)['valid_model_plan'])
        self.assertAlmostEqual(certificate.cost, adapter.objective(certificate.route))
        self.assertEqual(adapter.problem_id, adapter.input_identity())
        save_evidence('proxy_nonempty.json', dict(scope='G1 geometric proxy integration; not G0',
                                                adapter=adapter.payload(), plan=certificate.plan))

    def test_proxy_empty_task_batch_returns_certified_depot(self):
        scene = make_scene(2, 1, 202611032, counts=[0, 0])
        adapter = ResearchAdapter.from_scene(scene)
        self.assertEqual(adapter.anchor.route, [0, 0])
        self.assertEqual(adapter.anchor.cost, 0.)
        self.assertEqual(adapter.anchor.plan['assignments'], [])
        path = OUTPUT/'proxy_empty_runtime.json'
        _, diagnostics = run(adapter, 'greedy', SearchConfig(horizon=3), output_path=path)
        self.assertEqual(diagnostics['returned_cost'], 0.)
        self.assertEqual(diagnostics['failures'], [])
        replay = replay_file(path)
        self.assertTrue(replay['replay_valid'])
        save_evidence('proxy_empty_replay.json', replay)

    def test_f_d_are_bytes_backed_and_cannot_be_made_writable(self):
        adapter = ResearchAdapter.from_scene(self.scene)
        untouched = copy.deepcopy(self.scene)
        for array in (adapter.problem.F, adapter.problem.nav['costs'], adapter.problem.arm_dist):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
            with self.assertRaises(ValueError):
                array.flat[0] = 0
        self.assertEqual(self.scene, untouched)
        adapter.source['scene']['tasks'][0]['x'] += 1.
        self.assertEqual(self.scene, untouched)
        self.assertEqual(adapter.problem.scene, untouched)

    def test_payload_roundtrip_and_defensive_copy(self):
        payload = self.proxy.payload()
        restored = ResearchAdapter.from_payload(json.loads(json.dumps(payload)))
        self.assertEqual(restored.payload(), payload)
        self.assertEqual(restored.anchor.plan, self.proxy.anchor.plan)
        payload['source']['scene']['tasks'][0]['x'] += .01
        self.assertEqual(self.proxy.source['scene'], self.scene)

    def test_payload_tampering_fails_closed_including_nan_anchor(self):
        mutations = [
            lambda p: p.__setitem__('problem_id', '0'*64),
            lambda p: p.__setitem__('anchor_cost', p['anchor_cost']+1.),
            lambda p: p.__setitem__('anchor_cost', float('nan')),
            lambda p: p.__setitem__('anchor_cost', float('inf')),
            lambda p: p.__setitem__('anchor_cost', True),
            lambda p: p['anchor_route'].__setitem__(0, False),
            lambda p: p['anchor_route'].__setitem__(1, 9999),
            lambda p: p['source'].__setitem__('kind', 'unknown'),
            lambda p: p.__setitem__('schema', 'unknown'),
            lambda p: p['source']['scene']['tasks'][0].__setitem__('x',
                        p['source']['scene']['tasks'][0]['x']+.001),
        ]
        for mutate in mutations:
            payload = self.proxy.payload()
            mutate(payload)
            with self.subTest(payload_cost=str(payload['anchor_cost'])), self.assertRaises(ValueError):
                ResearchAdapter.from_payload(payload)

    def test_adapter_requires_successful_export_validation_flag(self):
        invalid = copy.deepcopy(self.proxy.anchor.plan)
        invalid['validation']['valid_model_plan'] = False
        with patch('rc3_rl_research.adapter.export_plan', return_value=invalid):
            with self.assertRaisesRegex(ValueError, 'independent validation'):
                self.proxy.certify(self.proxy.anchor.route)

    def test_external_uses_common_f_and_a_without_proxy_calls(self):
        bundle = copy.deepcopy(self.bundle)
        F, A = check_bundle(bundle)
        allowed = F[0, :, 0].copy()
        F[0] = False
        A[0] = False
        F[0, :, 0] = allowed
        F[0, :, 1] = allowed
        A[0, :, 1] = allowed
        bundle = reseal(bundle, F, A)
        untouched = copy.deepcopy(bundle)
        with (patch('rc3_fast.solver.vectorized_feasibility', side_effect=AssertionError('proxy must not run')),
              patch('rc3_baseline.validator.proxy_reachable', side_effect=AssertionError('proxy must not run'))):
            adapter = ResearchAdapter.from_external(bundle)
            restored = ResearchAdapter.from_payload(adapter.payload())
            certificate = restored.certify(restored.anchor.route)
        np.testing.assert_array_equal(adapter.problem.F, F & A)
        self.assertEqual(bundle, untouched)
        assignment = next(a for a in certificate.plan['assignments'] if a['task_id'] == 0)
        self.assertEqual(assignment['arm'], 'R')
        self.assertTrue(validate_external(bundle, certificate.plan)['valid_model_plan'])
        save_evidence('external_common_arm.json', dict(scope='v5 synthetic arm counterexample; not physical IK',
                      adapter=adapter.payload(), plan=certificate.plan))

    def test_v5_later_stop_counterexample_keeps_necessary_stop(self):
        bundle, route = later_stop_counterexample()
        untouched = copy.deepcopy(bundle)
        adapter = ResearchAdapter.from_external(bundle)
        F, A = check_bundle(bundle)
        np.testing.assert_array_equal(adapter.problem.F, F & A)
        certificate = adapter.certify(route)
        self.assertEqual(certificate.route, route)
        self.assertEqual(certificate.plan['visit_order_candidate_ids'], [0, 23])
        self.assertEqual(next(a['candidate_id'] for a in certificate.plan['assignments'] if a['task_id'] == 0), 23)
        self.assertEqual(bundle, untouched)
        path = OUTPUT/'external_later_stop_runtime.json'
        _, diagnostics = run(adapter, 'greedy', SearchConfig(horizon=3), output_path=path)
        self.assertEqual(diagnostics['failures'], [])
        replay = replay_file(path)
        self.assertTrue(replay['exported_plan_verified'])
        save_evidence('external_later_stop_replay.json', replay)

    def test_external_no_intersection_and_bad_checker_fail_closed(self):
        F, A = check_bundle(self.bundle)
        A[0] = False
        with self.assertRaisesRegex(ValueError, 'no certified Fixed3 anchor'):
            ResearchAdapter.from_external(reseal(self.bundle, F, A))
        with self.assertRaisesRegex(ValueError, 'no certified Fixed3 anchor'):
            ResearchAdapter.from_external(self.bundle, service_checker=lambda *args: False)

    def test_external_callback_replay_requires_checker(self):
        checker = lambda *args: True
        adapter = ResearchAdapter.from_external(self.bundle, service_checker=checker)
        with self.assertRaisesRegex(ValueError, 'requires the original external service checker'):
            ResearchAdapter.from_payload(adapter.payload())
        restored = ResearchAdapter.from_payload(adapter.payload(), service_checker=checker)
        self.assertEqual(restored.payload(), adapter.payload())

    def test_three_shared_runtime_policies_save_and_replay_real_geometry(self):
        scene = make_complex_scene(3, 3, 202611033, mode='blocked')
        untouched = copy.deepcopy(scene)
        adapter = ResearchAdapter.from_scene(scene)
        records = {}
        for method in ('random', 'greedy', 'alns'):
            with self.subTest(method=method):
                path = OUTPUT/f'proxy_blocked_{method}.json'
                plan, diagnostics = run_scene(scene, method=method,
                                        config=SearchConfig(horizon=4, work_budget=200000),
                                        seed=0, output_path=path)
                self.assertEqual(diagnostics['failures'], [])
                self.assertEqual(diagnostics['timing']['scope'], 'scene_to_plan')
                self.assertGreater(diagnostics['counters']['proposal_calls'], 0)
                self.assertLessEqual(diagnostics['returned_cost'], diagnostics['anchor_cost']+1e-8)
                self.assertTrue(validate(scene, plan, adapter.problem.sparse)['valid_model_plan'])
                replay = replay_file(path)
                self.assertTrue(replay['replay_valid'])
                self.assertTrue(replay['exported_plan_verified'])
                self.assertTrue(replay['policy_state_verified'])
                records[method] = dict(returned_cost=diagnostics['returned_cost'],
                                       anchor_cost=diagnostics['anchor_cost'],
                                       steps=diagnostics['steps'], work_used=diagnostics['work_used'], replay=replay)
        self.assertEqual(scene, untouched)
        save_evidence('shared_runtime_replay_summary.json', dict(
            scope='G1 integration on one generated 3x3 blocked proxy scene; not G0 or a performance comparison',
            scene_seed=202611033, policies=records))


if __name__ == '__main__':
    unittest.main()
