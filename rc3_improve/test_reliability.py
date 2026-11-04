import copy
import json
import unittest
from unittest.mock import patch

import numpy as np

from rc3_baseline import baseline
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare,route_objective
from .environment import ImprovementEnv
from .evaluate import run_method,evaluate_case,safe_json,METHODS


class UniformPolicy:
    def scores(self,features,mask):
        return np.zeros(len(mask))


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.scene=make_complex_scene(4,3,812973,'blocked')
        self.problem=prepare(self.scene)

    def test_invalid_candidate_cannot_replace_validated_anchor(self):
        class BadEnv(ImprovementEnv):
            def step(self,action):
                self.route=[0,0]
                return 100.,False,dict(accepted=True)
        plan,info=run_method(self.problem,'detour',horizon=3,env_factory=BadEnv)
        self.assertTrue(info['fallback_used'])
        self.assertEqual(info['objective_m'],info['reference_objective_m'])
        self.assertTrue(plan['validation']['valid_model_plan'])
        self.assertEqual(len(plan['assignments']),len(self.scene['tasks']))

    def test_worsening_but_complete_candidate_is_rejected(self):
        class WorseEnv(ImprovementEnv):
            def step(self,action):
                rng=np.random.default_rng(73)
                before=route_objective(self.problem,self.route)
                for _ in range(100):
                    candidate=[0,*rng.permutation(self.route[1:-1]).tolist(),0]
                    if route_objective(self.problem,candidate)>before+1e-4:
                        self.route=candidate
                        return 1.,False,dict(accepted=True)
                raise AssertionError('test needs an actually worse candidate')
        plan,info=run_method(self.problem,'detour',horizon=2,env_factory=WorseEnv)
        self.assertTrue(info['fallback_used'])
        self.assertEqual(info['objective_m'],info['reference_objective_m'])
        self.assertTrue(plan['validation']['valid_model_plan'])

    def test_scorer_failure_retains_anchor(self):
        class Broken:
            def scores(self,*args):raise RuntimeError('injected model failure')
        plan,info=run_method(self.problem,'ppo',Broken(),horizon=2)
        self.assertTrue(info['fallback_used'])
        self.assertEqual(info['attempts'],0)
        self.assertTrue(plan['validation']['all_tasks_served'])
        self.assertEqual(info['objective_m'],info['reference_objective_m'])

    def test_budget_and_independent_cost_monotonicity(self):
        for method in ('random','detour','bc','ppo'):
            plan,info=run_method(self.problem,method,UniformPolicy(),horizon=3)
            self.assertLessEqual(info['attempts'],3)
            costs=[t['objective_m'] for t in info['trace']]
            self.assertTrue(all(b<=a+1e-8 for a,b in zip(costs,costs[1:])))
            self.assertTrue(plan['validation']['valid_model_plan'])
            self.assertLessEqual(info['objective_m'],info['reference_objective_m']+1e-8)

    def test_empty_tasks_are_successful_zero_cost(self):
        p=prepare(baseline.make_scene(1,1,19,counts=[0]))
        for method in METHODS:
            plan,info=run_method(p,method,UniformPolicy(),horizon=2)
            self.assertEqual(info['objective_m'],0.)
            self.assertTrue(plan['validation']['valid_model_plan'])
            self.assertFalse(plan['assignments'])

    def test_no_service_failure_keeps_every_task_and_method(self):
        scene=copy.deepcopy(self.scene)
        scene['tasks'][0]['z']=99.
        rows,output=evaluate_case(scene,{},horizon=1,repeats=1)
        self.assertEqual(len(rows),len(METHODS))
        self.assertTrue(all(row['status']=='failed' for row in rows))
        self.assertEqual(output['pending_task_ids'],[t['id'] for t in scene['tasks']])
        self.assertFalse(output['plans'])

    def test_disconnected_failure_keeps_denominator(self):
        p=self.problem
        p.nav['costs'][0,1:]=np.inf
        p.nav['costs'][1:,0]=np.inf
        with patch('rc3_improve.evaluate.prepare',return_value=p):
            rows,output=evaluate_case(self.scene,{},horizon=1,repeats=1)
        self.assertEqual(len(rows),len(METHODS))
        self.assertTrue(all(row['status']=='failed' for row in rows))
        self.assertIn('disconnected=',output['error'])
        self.assertEqual(len(output['pending_task_ids']),len(self.scene['tasks']))

    def test_invalid_input_can_be_serialized_as_failure(self):
        scene=copy.deepcopy(self.scene)
        scene['tasks'][0]['x']=float('nan')
        with patch('rc3_improve.evaluate.prepare',side_effect=ValueError('nonfinite input')):
            rows,output=evaluate_case(scene,{})
        payload=json.dumps(safe_json(output),allow_nan=False)
        self.assertIn('"nan"',payload)
        self.assertEqual(len(rows),len(METHODS))


if __name__=='__main__':unittest.main()
