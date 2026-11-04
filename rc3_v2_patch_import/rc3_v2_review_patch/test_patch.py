import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rc3_baseline import baseline
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare, route_objective, export_plan
from rc3_improve.environment import ImprovementEnv
from rc3_improve.evaluate import run_method
from rc3_improve.efficient import FreshAttemptEnv, run_efficient, plan_scene
from rc3_improve.numpy_policy import NumpyPolicy


class PatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = make_complex_scene(4, 3, 3701169001, 'mixed')
        cls.problem = prepare(cls.scene)
        cls.policy = NumpyPolicy.load(Path(__file__).parent/'weights/ppo.npz', 'ppo')

    def test_changed_external_coverage_reopens_old_window(self):
        x = np.asarray([0, 5.5, 1, 2, 3, 4, 5, 6])
        cover = np.zeros((6, 7, 2), dtype=bool)
        for task in range(6):
            cover[task, task+1, 0] = True
        cover[[0, 4, 5], 0, 0] = True
        problem = SimpleNamespace(F=cover, penalty=.75, nav={'costs': np.abs(x[:, None]-x[None, :])},
            scene={'x_lines': [0, 7], 'y_lines': [0, 1],
                   'candidates': [{'x': float(v), 'y': 0.} for v in x[1:]]})
        env = FreshAttemptEnv(problem, 8, initial_route=[0, 2, 3, 4, 5, 6, 7, 0])
        env.step(0)
        self.assertFalse(env.observe()[1][0])
        env.step(2)
        self.assertEqual(len(env.attempted), 1)  # unchanged failed context retained; successful one not cached
        self.assertTrue(env.observe()[1][0])
        self.assertAlmostEqual(route_objective(problem, env.route), 14.75)
        env.step(0)
        self.assertAlmostEqual(route_objective(problem, env.route), 14.)

    def test_certification_only_preserves_original_policy_actions_and_cost(self):
        original, old = run_method(self.problem, 'ppo', self.policy, 8)
        efficient, new = run_efficient(self.problem, 'ppo', self.policy, 8, refresh_attempts=False)
        self.assertEqual(original['visit_order_candidate_ids'], efficient['visit_order_candidate_ids'])
        self.assertAlmostEqual(old['objective_m'], new['objective_m'])
        self.assertEqual([r['action'] for r in old['trace'] if r['event'] == 'repair'],
                         [r['action'] for r in new['trace'] if r['event'] == 'provisional_repair'])

    def test_fixed3_same_quality_with_one_certificate(self):
        original, old = run_method(self.problem, 'fixed3')
        with patch('rc3_improve.efficient.export_plan', wraps=export_plan) as certify:
            candidate, new = run_efficient(self.problem, 'fixed3')
        self.assertEqual(certify.call_count, 1)
        self.assertAlmostEqual(old['objective_m'], new['objective_m'])
        self.assertTrue(candidate['validation']['valid_model_plan'])

    def test_model_exception_returns_verified_reference(self):
        class Broken:
            def scores(self, features, mask):
                raise RuntimeError('injected model failure')
        plan, info = run_efficient(self.problem, 'ppo', Broken(), reference='fixed3')
        self.assertTrue(info['fallback_used'])
        self.assertEqual(info['objective_m'], info['reference_objective_m'])
        self.assertTrue(plan['validation']['all_tasks_served'])

    def test_final_validation_rejection_retains_verified_reference(self):
        # This fresh 100-table fixture supplies a real strictly improving local
        # proposal. A fault is injected only into its final independent export.
        problem = prepare(make_complex_scene(10, 10, 3701161000, 'regular'))
        original_export = export_plan
        calls = []
        def failed_final(*args):
            calls.append(1)
            if len(calls) > 1:
                raise ValueError('injected independent final validation rejection')
            return original_export(*args)
        with patch('rc3_improve.efficient.export_plan', side_effect=failed_final):
            plan, info = run_efficient(problem, 'ppo', self.policy, 8)
        self.assertGreater(info['provisional_improvements'], 0)
        self.assertGreater(len(calls), 1)
        self.assertTrue(info['fallback_used'])
        self.assertEqual(info['objective_m'], info['reference_objective_m'])
        self.assertTrue(plan['validation']['valid_model_plan'])

    def test_stage_mismatch_does_not_silently_run_ppo_as_bc(self):
        with self.assertRaisesRegex(ValueError, 'checkpoint stage'):
            NumpyPolicy.load(Path(__file__).parent/'weights/ppo.npz', 'bc')

    def test_empty_tasks_and_failed_tasks_are_explicit(self):
        result = plan_scene(baseline.make_scene(1, 1, 18, counts=[0]), 'ppo', self.policy)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['diagnostics']['objective_m'], 0.)
        scene = copy.deepcopy(self.scene)
        scene['tasks'][0]['z'] = 99.
        result = plan_scene(scene)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['code'], 'no_service_candidate')
        self.assertEqual(result['pending_task_ids'], list(range(len(scene['tasks']))))

    def test_hybrid_reference_and_budget(self):
        plan, info = run_efficient(self.problem, 'ppo', self.policy, 3, reference='fixed3')
        self.assertLessEqual(info['attempts'], 3)
        self.assertLessEqual(info['objective_m'], info['reference_objective_m']+1e-8)
        self.assertTrue(plan['validation']['valid_model_plan'])
        self.assertTrue(plan['validation']['all_tasks_served'])


if __name__ == '__main__':
    unittest.main()
