"""Check label semantics, zero-gain retention and fresh manifest partitioning."""
from collections import Counter, defaultdict
import itertools
import unittest
from unittest.mock import patch

import numpy as np

from rc3_fast.solver import route_objective
from rc3_improve.test_repair import tiny_problem
from rc3_improve.repair import route_is_valid
from rc3_v3.data import TeacherLabels, collect_trajectory, generate_cases
from rc3_v3.environment import Env, FEATURE_NAMES


def statistics():
    result=defaultdict(int)
    result['step_counts']=Counter(); result['teacher_counts']=Counter()
    return result


class DataTests(unittest.TestCase):
    def test_labels_match_actual_environment_transitions_and_cache(self):
        problem=tiny_problem(); route=[0,1,2,3,0]
        env=Env(problem,initial_route=route)
        features,mask=env.observe(); stats=statistics()
        labeller=TeacherLabels(problem,env.reference_cost,stats,FEATURE_NAMES)
        gains=labeller.evaluate(env,features,mask)
        for action in np.flatnonzero(mask):
            probe=Env(problem,initial_route=route)
            reward,_,_=probe.step(int(action))
            self.assertAlmostEqual(float(gains[action]),reward,places=5)
            self.assertTrue(route_is_valid(problem,probe.route))
        self.assertGreater(float(gains.max()),0.)
        calls=stats['teacher_repair_calls']
        np.testing.assert_array_equal(gains,labeller.evaluate(env,features,mask))
        self.assertEqual(calls,stats['teacher_repair_calls'])
        self.assertEqual(stats['teacher_cache_hits'],int(mask.sum()))

    def test_zero_gain_states_remain_in_compact_dataset(self):
        problem=tiny_problem(); problem.scene['seed']=901
        # Exhaustively establish a global optimum on this six-candidate fixture.
        routes=([0,*s,0] for size in range(1,7)
                for s in itertools.permutations(range(1,7),size))
        optimum=min((r for r in routes if route_is_valid(problem,r)),
                    key=lambda r:route_objective(problem,r))
        stats=statistics()
        def initialized(p,horizon):return Env(p,horizon,initial_route=optimum)
        # This synthetic fixture has no geometric navigation data; only bypass
        # export serialization, keeping real canonical costs/repair/transition.
        def exported(p,route,label):return {'objective_length_plus_stops_m':route_objective(p,route)}
        with patch('rc3_v3.environment.Env',side_effect=initialized), patch('rc3_v3.data.export_plan',side_effect=exported):
            _,_,states,trajectory=collect_trajectory(problem,0,8,stats)
        self.assertGreater(len(states),0)
        self.assertEqual(stats['zero_gain_states'],len(states))
        self.assertEqual(stats['positive_states'],0)
        self.assertTrue(all(s['oracle_stop'] and not s['gains'][s['mask']].any() for s in states))
        self.assertEqual(trajectory['saved_steps'],[s['step'] for s in states])
        self.assertTrue(all(s['step']%2==0 for s in states))

    def test_seed_partitions_and_stress_metadata(self):
        forbidden=set(range(100000,100100))|{901,931}
        cases=generate_cases(np.random.default_rng(741),forbidden)
        self.assertEqual({key:len(rows) for key,rows in cases.items()},
                         {'train':96,'validation':18,'test':60})
        seeds=[row['seed'] for rows in cases.values() for row in rows]
        self.assertEqual(len(seeds),len(set(seeds)))
        self.assertFalse(set(seeds)&forbidden)
        self.assertTrue(all(row['nx']*row['ny']==100 for key in ('train','validation') for row in cases[key]))
        self.assertEqual(sum(row['split']=='stress' for row in cases['test']),6)


if __name__=='__main__':unittest.main()
