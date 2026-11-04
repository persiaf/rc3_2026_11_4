import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rc3_baseline.baseline import make_scene
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare, solve_route, route_objective, export_plan
from rc3_improve.environment import canonical_route
from rc3_improve.repair import _local, repair_window, window_features, route_is_valid
from rc3_v3.environment import Env, FEATURE_NAMES, FEATURE_SCHEMA, adaptive_windows
from rc3_v3.runtime import run


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.problem=prepare(make_complex_scene(6,6,931,'blocked'))

    def test_fixed3_anchor_matches_exact_cheap_recipe_comparison(self):
        expected=min(route_objective(self.problem,canonical_route(self.problem,solve_route(self.problem,r)))
                     for r in (0,1,7))
        with patch('rc3_v3.runtime.export_plan',wraps=export_plan) as certify:
            plan,info=run(self.problem,'fixed3')
        self.assertEqual(certify.call_count,1)
        self.assertAlmostEqual(info['objective_m'],expected,places=8)
        self.assertEqual(info['dp_calls'],0)
        self.assertEqual(info['stop_reason'],'reference_only')
        self.assertTrue(plan['validation']['valid_model_plan'])

    def test_adaptive_stride_and_largest_supported_width(self):
        env=Env(self.problem)
        actions,signatures=adaptive_windows(self.problem,env.route)
        self.assertEqual(len(signatures),len(set(signatures)))
        self.assertEqual([w['start'] for w in actions],list(range(1,len(env.route)-1)))
        for w in actions:
            _,_,required,pool,_=_local(self.problem,env.route,w)
            self.assertLessEqual(len(required),8);self.assertLessEqual(len(pool),80)
            width=w['end']-w['start'];self.assertTrue(1<=width<=4)
            for larger in range(width+1,min(4,len(env.route)-1-w['start'])+1):
                _,_,r,p,_=_local(self.problem,env.route,dict(start=w['start'],end=w['start']+larger))
                self.assertTrue(len(r)>8 or len(p)>80)

    def test_budget_features_update_without_recomputing_base_features_in_step(self):
        env=Env(self.problem,horizon=4)
        with patch('rc3_v3.environment.window_features',wraps=window_features) as feature_call:
            first,mask=env.observe();self.assertEqual(feature_call.call_count,1)
            action=int(np.flatnonzero(mask)[0])
            with patch('rc3_v3.environment.repair_window',side_effect=lambda p,r,w:list(r)):
                env.step(action)
            self.assertEqual(feature_call.call_count,1)
            second,_=env.observe();self.assertEqual(feature_call.call_count,1)
        self.assertEqual(FEATURE_SCHEMA,'rc3-v3-budget28')
        self.assertEqual(len(FEATURE_NAMES),28)
        np.testing.assert_allclose(first[0,24:],[1,0,1,0])
        np.testing.assert_allclose(second[0,24:],[.75,.25,1,0])

    def test_external_coverage_context_reopens_failure_without_clearing_cache(self):
        # Candidate 1 can cover task 0 and the final two tasks. Replacing the
        # distant suffix changes required coverage while the local node tuple
        # and its entry/exit remain exactly the same.
        x=np.asarray([0.,6.5,1.,2.,3.,4.,5.,6.,7.])
        f=np.zeros((7,8,2),dtype=bool)
        for task in range(7):f[task,task+1,0]=True
        f[[0,5,6],0,0]=True
        problem=SimpleNamespace(F=f,penalty=.75,nav={'costs':np.abs(x[:,None]-x[None,:])},
            scene={'x_lines':[0.,8.],'y_lines':[0.,1.],
                   'candidates':[dict(x=float(v),y=0.) for v in x[1:]]})
        env=Env(problem,8,initial_route=[0,2,3,4,5,6,7,8,0])
        env.observe();old=env._signatures[0]
        reward,_,info=env.step(0)
        self.assertEqual(reward,0.);self.assertFalse(info['accepted'])
        self.assertIn(old,env.attempted);self.assertFalse(env.observe()[1][0])
        env.route=[0,2,3,4,5,6,1,0]
        _,mask=env.observe();new=env._signatures[0]
        self.assertEqual(old[0],new[0]);self.assertNotEqual(old[1],new[1])
        self.assertTrue(mask[0]);self.assertIn(old,env.attempted)
        reward,_,info=env.step(0)
        self.assertGreater(reward,0.);self.assertTrue(info['accepted'])
        self.assertEqual(env.attempted,{old})

    def test_invalid_and_exceptional_repairs_are_not_cached(self):
        for proposal in ('invalid','exception'):
            env=Env(self.problem);_,mask=env.observe();a=int(np.flatnonzero(mask)[0])
            before=list(env.route)
            kwargs={'return_value':[0,0]} if proposal=='invalid' else {'side_effect':RuntimeError('injected')}
            with patch('rc3_v3.environment.repair_window',**kwargs):
                with self.assertRaises((ValueError,RuntimeError)):env.step(a)
            self.assertEqual(env.route,before);self.assertEqual(env.attempted,set())

    def test_dp_budget_monotonic_output_and_cost_identity(self):
        for method in ('random','detour'):
            plan,info=run(self.problem,method,horizon=3)
            self.assertLessEqual(info['dp_calls'],3)
            self.assertLessEqual(info['objective_m'],info['reference_objective_m']+1e-8)
            self.assertTrue(info['cost_nonincreasing'])
            self.assertTrue(plan['validation']['valid_model_plan'])
            self.assertAlmostEqual(plan['objective_length_plus_stops_m'],info['objective_m'],places=8)
            costs=[t['objective_m'] for t in info['trace'] if 'objective_m' in t]
            self.assertTrue(all(b<=a+1e-8 for a,b in zip(costs,costs[1:])))

    def test_policy_stop_is_not_a_repair_call(self):
        class Stop:
            def select(self,features,mask):return None
        plan,info=run(self.problem,'learned',Stop(),horizon=8)
        self.assertEqual(info['stop_reason'],'policy_stop')
        self.assertEqual(info['dp_calls'],0)
        self.assertEqual(info['objective_m'],info['reference_objective_m'])
        self.assertFalse(info['fallback_used'])

    def test_model_errors_and_bad_actions_fall_back_to_verified_anchor(self):
        class Broken:
            def scores(self,features,mask):raise RuntimeError('injected model failure')
        class Bad:
            def select(self,features,mask):return len(mask)+1
        for policy in (Broken(),Bad()):
            plan,info=run(self.problem,'learned',policy)
            self.assertTrue(info['fallback_used']);self.assertEqual(info['dp_calls'],0)
            self.assertEqual(info['objective_m'],info['reference_objective_m'])
            self.assertTrue(plan['validation']['all_tasks_served'])

    def test_final_certificate_failure_returns_anchor(self):
        calls=[]
        def reject_final(*args,**kwargs):
            calls.append(1)
            if len(calls)>1:raise ValueError('injected final certificate failure')
            return export_plan(*args,**kwargs)
        with patch('rc3_v3.runtime.export_plan',side_effect=reject_final):
            plan,info=run(self.problem,'detour',horizon=8)
        self.assertGreater(info['accepted'],0)
        self.assertEqual(len(calls),2)
        self.assertTrue(info['fallback_used'])
        self.assertEqual(info['objective_m'],info['reference_objective_m'])
        self.assertTrue(plan['validation']['valid_model_plan'])

    def test_zero_deadline_still_returns_a_valid_anchor_without_new_dp(self):
        plan,info=run(self.problem,'detour',time_limit_s=0.)
        self.assertEqual(info['dp_calls'],0)
        self.assertEqual(info['stop_reason'],'deadline')
        self.assertTrue(info['deadline_exceeded']);self.assertGreater(info['time_overrun_s'],0.)
        self.assertTrue(plan['validation']['all_tasks_served'])

    def test_model_time_can_exhaust_deadline_before_repair(self):
        clock=[0.]
        class Slow:
            def select(self,features,mask):
                clock[0]=.2
                return int(np.flatnonzero(mask)[0])
        with patch('rc3_v3.runtime.time.perf_counter',side_effect=lambda:clock[0]):
            plan,info=run(self.problem,'learned',Slow(),time_limit_s=.1)
        self.assertEqual(info['dp_calls'],0);self.assertEqual(info['stop_reason'],'deadline')
        self.assertAlmostEqual(info['model_s'],.2)
        self.assertAlmostEqual(info['time_overrun_s'],.1)

    def test_running_repair_and_final_validation_may_finish_after_soft_deadline(self):
        env=Env(self.problem);_,mask=env.observe()
        choices=[(int(a),canonical_route(self.problem,repair_window(self.problem,env.route,env.windows[int(a)]))) for a in np.flatnonzero(mask)]
        action,route=min(choices,key=lambda pair:route_objective(self.problem,pair[1]))
        self.assertLess(route_objective(self.problem,route),env.reference_cost)
        clock=[0.]
        class Choose:
            def select(self,features,mask):return action
        def slow_repair(*args,**kwargs):clock[0]=.2;return list(route)
        certificates=[]
        def slow_final(*args,**kwargs):
            certificates.append(1)
            if len(certificates)>1:clock[0]=.3
            return export_plan(*args,**kwargs)
        with patch('rc3_v3.runtime.time.perf_counter',side_effect=lambda:clock[0]), \
             patch('rc3_v3.environment.repair_window',side_effect=slow_repair), \
             patch('rc3_v3.runtime.export_plan',side_effect=slow_final):
            plan,info=run(self.problem,'learned',Choose(),horizon=8,time_limit_s=.1)
        self.assertEqual(info['dp_calls'],1)
        self.assertEqual(info['stop_reason'],'deadline')
        self.assertEqual(len(certificates),2)
        self.assertAlmostEqual(info['planning_s'],.3)
        self.assertAlmostEqual(info['time_overrun_s'],.2)
        self.assertGreater(info['final_validation_s'],0.)
        self.assertTrue(plan['validation']['valid_model_plan'])

    def test_empty_terminal_and_old_phase_names_rejected(self):
        empty=prepare(make_scene(1,1,7,counts=[0]))
        env=Env(empty);features,mask=env.observe()
        self.assertTrue(env.done);self.assertEqual(features.shape,(0,28))
        self.assertFalse(mask.any())
        with self.assertRaises(ValueError):env.step(0)
        plan,info=run(empty,'learned',policy=None)
        self.assertEqual(info['objective_m'],0.);self.assertEqual(info['dp_calls'],0)
        self.assertTrue(plan['validation']['valid_model_plan'])
        for method in ('bc','ppo'):
            with self.assertRaises(ValueError):run(self.problem,method)


if __name__=='__main__':
    unittest.main()
