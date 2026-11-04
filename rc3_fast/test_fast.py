import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from rc3_baseline import baseline
from rc3_baseline.validator import validate
from rc3_fast.scenes import make_complex_scene
from rc3_fast.features import scene_features, FEATURE_NAMES
from rc3_fast.planner import Planner, map_key
from rc3_fast.selector import Selector
from rc3_fast.solver import (prepare, vectorized_feasibility, two_opt, construct, joint_search,
                             solve_route, route_objective, export_plan, RECIPES, DEFAULT_RECIPE)


class FastPlannerTests(unittest.TestCase):
    def test_vectorized_geometry_matches_reference(self):
        for mode in ('regular','mixed','blocked'):
            scene=make_complex_scene(4,3,701,mode)
            f,d,s=vectorized_feasibility(scene)
            ff,dd,ss=baseline.feasibility(scene)
            np.testing.assert_array_equal(f,ff)
            np.testing.assert_allclose(d,dd,atol=1e-12)
            self.assertEqual(s,ss)

    def test_fast_two_opt_has_same_result_as_reference(self):
        for seed in range(4):
            p=prepare(make_complex_scene(6,6,700+seed))
            route=construct(p)
            old=baseline.improve_2opt(route,p.nav['costs'])
            new=two_opt(route,p.nav['costs'])
            self.assertEqual(new,old)

    def test_joint_search_preserves_coverage_and_cannot_increase_cost(self):
        for mode in ('regular','mixed','blocked'):
            p=prepare(make_complex_scene(4,3,701,mode))
            for recipe in range(len(RECIPES)):
                before=two_opt(construct(p,recipe),p.nav['costs'])
                after=joint_search(before,p.F.any(2),p.nav['costs'])
                self.assertLessEqual(route_objective(p,after),route_objective(p,before)+1e-7)
                self.assertTrue(p.F.any(2)[:,np.asarray(after[1:-1])-1].any(1).all())
                plan=export_plan(p,after,'joint')
                self.assertTrue(plan['validation']['valid_model_plan'])

    def test_extra_obstacles_are_seen_by_independent_validator(self):
        p=prepare(make_complex_scene(4,3,701))
        plan=export_plan(p,solve_route(p),'fast')
        tampered=copy.deepcopy(p.scene)
        point=plan['route_polyline'][0]
        tampered['obstacles']=[dict(xmin=point[0]-.05,xmax=point[0]+.05,ymin=point[1]-.05,ymax=point[1]+.05)]
        result=validate(tampered,plan,p.sparse)
        self.assertFalse(result['valid_model_plan'])
        self.assertGreater(result['collision_count'],0)

    def test_blocked_scenes_keep_all_original_tasks(self):
        mixed=make_complex_scene(6,6,701,'mixed')
        blocked=make_complex_scene(6,6,701,'blocked')
        self.assertEqual(mixed['tasks'],blocked['tasks'])
        self.assertGreater(len(blocked['obstacles']),0)
        p=prepare(blocked)
        self.assertGreater(p.nav['rejected_edge_count'],0)
        plan=export_plan(p,solve_route(p),'blocked')
        self.assertEqual(len(plan['assignments']),len(mixed['tasks']))

    def test_features_do_not_read_seed_or_teacher(self):
        p=prepare(make_complex_scene(4,3,701,'mixed'))
        x=scene_features(p)
        p.scene['seed']=999999
        p.scene['teacher_cost']=123456
        np.testing.assert_array_equal(x,scene_features(p))

    def test_empty_scene_and_infeasible_scene(self):
        scene=baseline.make_scene(1,1,0,counts=[0])
        p=prepare(scene)
        self.assertEqual(solve_route(p),[0,0])
        self.assertEqual(export_plan(p,[0,0],'empty')['route_length_m'],0)
        self.assertTrue(np.isfinite(scene_features(p)).all())
        scene=baseline.make_scene(1,1,0,counts=[1])
        scene['tasks'][0]['z']=3.
        with self.assertRaises(ValueError):
            solve_route(prepare(scene))

    def test_navigation_cache_reuses_only_unchanged_maps(self):
        scene=make_complex_scene(4,3,701,'blocked')
        planner=Planner(cache_size=1)
        first=planner.plan(scene)
        changed=copy.deepcopy(scene)
        changed['seed']=702
        changed['tasks'][0]['x']+=.001
        second=planner.plan(changed)
        self.assertFalse(first['timing']['cache_hit'])
        self.assertTrue(second['timing']['cache_hit'])
        moved=copy.deepcopy(changed)
        moved['obstacles'][0]['xmin']+=.01
        third=planner.plan(moved)
        self.assertFalse(third['timing']['cache_hit'])
        self.assertEqual(len(planner._cache),1)
        self.assertNotEqual(map_key(changed),map_key(moved))
        self.assertTrue(next(iter(second['plans'].values()))['validation']['valid_model_plan'])

    def test_portable_model_batch_and_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'selector.npz'
            meta=dict(feature_names=FEATURE_NAMES,recipes=RECIPES,members=1)
            arrays=dict(mean=np.zeros(len(FEATURE_NAMES),np.float32),std=np.ones(len(FEATURE_NAMES),np.float32),
                        metadata=np.array(json.dumps(meta)),m0_w0=np.zeros((4,len(FEATURE_NAMES)),np.float32),
                        m0_b0=np.zeros(4,np.float32),m0_w1=np.eye(4,dtype=np.float32),m0_b1=np.zeros(4,np.float32),
                        m0_w2=np.zeros((13,4),np.float32),m0_b2=np.arange(13,dtype=np.float32))
            np.savez(path,**arrays)
            model=Selector(path)
            x=scene_features(prepare(make_complex_scene(4,3,701,'blocked')))
            np.testing.assert_array_equal(model.rank(x,3),[0,1,2])
            np.testing.assert_array_equal(model.predict(np.stack([x,x]))[0],model.predict(x))
            for invalid in (0,14):
                with self.assertRaises(ValueError):
                    model.rank(x,invalid)
            meta['feature_names']=['wrong']
            arrays['metadata']=np.array(json.dumps(meta))
            np.savez(path,**arrays)
            with self.assertRaises(ValueError):
                Selector(path)


if __name__ == '__main__':
    unittest.main()
