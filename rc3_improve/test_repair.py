import copy
import itertools
import unittest

import numpy as np

from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import Problem, prepare, solve_route, route_objective, export_plan
from rc3_improve.repair import (windows, window_features, repair_window, route_is_valid,
                                FEATURE_NAMES)


def tiny_problem():
    xy=np.array([[0.,0.],[1.,2.],[3.,1.],[4.,0.],[2.,.2],[3.5,.2],[8.,8.]])
    distance=np.sqrt(((xy[:,None]-xy[None,:])**2).sum(-1))
    cover=np.array([[1,0,0,1,0,1],[0,1,0,1,1,1],[0,0,1,0,1,1]],dtype=bool)
    f=np.zeros((*cover.shape,2),dtype=bool);f[:,:,0]=cover
    scene=dict(candidates=[dict(x=float(p[0]),y=float(p[1])) for p in xy[1:]],
               x_lines=[0,10],y_lines=[0,10])
    return Problem(scene,f,np.ones_like(f,dtype=float),[],dict(costs=distance),0.)


def brute_repair(problem, route, start, end):
    retained=set(route[1:start]+route[end:-1])
    pool=[i for i in range(1,problem.F.shape[1]+1) if i not in retained]
    best=route
    for size in range(len(pool)+1):
        for sequence in itertools.permutations(pool,size):
            candidate=route[:start]+list(sequence)+route[end:]
            if route_is_valid(problem,candidate) and route_objective(problem,candidate)<route_objective(problem,best):
                best=candidate
    return best


class RepairTests(unittest.TestCase):
    def test_exact_dp_matches_exhaustive_full_and_fixed_endpoints(self):
        p=tiny_problem();route=[0,1,2,3,0]
        for start,end in ((1,4),(1,3),(2,4),(2,3)):
            expected=brute_repair(p,route,start,end)
            actual=repair_window(p,route,dict(start=start,end=end))
            self.assertAlmostEqual(route_objective(p,actual),route_objective(p,expected),places=9)
            self.assertEqual(actual[:start],route[:start])
            self.assertEqual(actual[-len(route[end:]):],route[end:])
            self.assertTrue(route_is_valid(p,actual))

    def test_joint_replacement_improves_over_incumbent(self):
        p=tiny_problem();route=[0,1,2,3,0]
        after=repair_window(p,route,dict(start=1,end=3))
        self.assertEqual(after,[0,4,3,0])
        self.assertLess(route_objective(p,after),route_objective(p,route))
        self.assertEqual(route,[0,1,2,3,0])

    def test_caps_preserve_original_and_do_not_drop_tasks(self):
        p=tiny_problem();route=[0,1,2,3,0]
        for limits in (dict(max_tasks=1),dict(max_candidates=1)):
            result=repair_window(p,route,dict(start=1,end=3),**limits)
            self.assertEqual(result,route)
            self.assertIsNot(result,route)

    def test_redundant_segment_can_be_removed(self):
        p=tiny_problem();route=[0,1,4,3,0]
        self.assertTrue(route_is_valid(p,route))
        actual=repair_window(p,route,dict(start=1,end=2),max_tasks=0,max_candidates=0)
        self.assertEqual(actual,[0,4,3,0])

    def test_empty_route_and_malformed_inputs(self):
        p=tiny_problem()
        with self.assertRaises(ValueError):windows(p,[0,1,0])
        with self.assertRaises(ValueError):repair_window(p,[0,1,2,3,0],dict(start=0,end=2))
        with self.assertRaises(ValueError):windows(p,[0,1,2,3,0],stride=0)
        self.assertFalse(route_is_valid(p,[0,1,2,3,3,0]))
        self.assertFalse(route_is_valid(p,[0,1,2,3,99,0]))
        self.assertFalse(route_is_valid(p,[0.,1.,2.,3.,0.]))
        p.F=p.F[:0]
        self.assertTrue(route_is_valid(p,[0,0]))
        self.assertEqual(windows(p,[0,0]),[])
        self.assertEqual(window_features(p,[0,0],[]).shape,(0,len(FEATURE_NAMES)))

    def test_disconnected_alternative_cannot_replace_valid_incumbent(self):
        p=tiny_problem();route=[0,1,2,3,0]
        p.nav['costs'][4,:]=np.inf;p.nav['costs'][:,4]=np.inf
        result=repair_window(p,route,dict(start=1,end=3))
        self.assertNotIn(4,result)
        self.assertTrue(route_is_valid(p,result))

    def test_observable_features_and_window_tail(self):
        p=prepare(make_complex_scene(4,3,901,'blocked'))
        route=solve_route(p,1)
        actions=windows(p,route,width=3,stride=2)
        self.assertEqual(actions[-1]['end'],len(route)-1)
        features=window_features(p,route,actions)
        self.assertEqual(features.shape,(len(actions),len(FEATURE_NAMES)))
        self.assertTrue(np.isfinite(features).all())
        p.scene['seed']=123456;p.scene['teacher_gain']=1000
        np.testing.assert_array_equal(features,window_features(p,route,actions))

    def test_complex_scenes_preserve_coverage_and_validate_export(self):
        improvements=0
        for mode in ('regular','mixed','blocked'):
            p=prepare(make_complex_scene(6,6,931,mode))
            route=solve_route(p,1)
            before=route_objective(p,route)
            for window in windows(p,route):
                after=repair_window(p,route,window)
                self.assertLessEqual(route_objective(p,after),before+1e-8)
                self.assertTrue(route_is_valid(p,after))
                self.assertTrue(export_plan(p,after,'exact_local')['validation']['valid_model_plan'])
                improvements+=route_objective(p,after)<before-1e-7
        self.assertGreater(improvements,0)


if __name__=='__main__':
    unittest.main()
