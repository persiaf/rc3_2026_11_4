"""Checks for reward accounting, padding/ID mapping and cross-arm intersection."""
import copy
import unittest
import numpy as np
import torch
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import route_objective
from rc3_v5.external_example import make_demo,reseal
from .problem import Cache
from .env import Env,batch,unpack_routes
from .model import Policy
from .baselines import insertion,prior,postprocess


class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_reward_replay_and_export(self):
        item=Cache().prepare(make_complex_scene(3,2,7601000000,'mixed'));route=prior(item)
        env=Env(item);trace=env.replay([j-1 for j in route[1:-1]])
        plan=item.certify(route,'contract')
        self.assertAlmostEqual(-sum(env.rewards),plan['objective_length_plus_stops_m'],places=8)
        self.assertTrue(all(t['served_tasks'] for t in trace));self.assertLessEqual(len(trace),len(item.C))
        with self.assertRaises(ValueError):env.step(0)

    def test_empty_and_unreachable(self):
        scene=make_complex_scene(1,1,7601000001);scene['tasks']=[];item=Cache().prepare(scene)
        env=Env(item);self.assertTrue(env.done);self.assertEqual(env.route,[0,0])
        self.assertEqual(item.certify(env.route,'empty')['objective_length_plus_stops_m'],0.)
        scene=make_complex_scene(1,1,7601000002)
        if not scene['tasks']:
            from rc3_baseline.baseline import make_scene
            scene=make_scene(1,1,7601000002,counts=[1])
        scene['tasks'][0]['x']=10000
        with self.assertRaisesRegex(ValueError,'no depot-connected'):Cache().prepare(scene)

    def test_batch_padding_matches_single_original_ids(self):
        items=[Cache().prepare(make_complex_scene(x,y,7601000010+x,'regular')) for x,y in [(1,1),(3,2)]]
        model=Policy().eval()
        with torch.inference_mode():cost,_,a,l=model(batch(items))
        for item,route,c in zip(items,unpack_routes(a,l),cost.tolist()):
            with torch.inference_mode():c1,_,a1,l1=model(batch([item]))
            self.assertEqual(route,unpack_routes(a1,l1)[0]);self.assertAlmostEqual(c,c1.item(),places=8)
            self.assertAlmostEqual(c,route_objective(item.base,route),places=8)
            Env(item).replay([j-1 for j in route[1:-1]])

    def test_external_arm_intersection_before_union(self):
        bundle=make_demo();F=np.asarray(bundle['planning']['F'],bool);A=np.asarray(bundle['audit']['F'],bool)
        F[0]=False;A[0]=False;F[0,0,0]=True;A[0,0,1]=True
        changed=reseal(bundle,F=F,A=A)
        with self.assertRaisesRegex(ValueError,'no depot-connected'):Cache().prepare(changed)
        item=Cache().prepare(bundle);route=insertion(item);plan=item.certify(postprocess(item,route),'external_contract')
        self.assertTrue(plan['validation']['valid_model_plan'])

    def test_postprocessing_and_invalid_action(self):
        item=Cache().prepare(make_complex_scene(3,3,7601000020,'blocked'));route=insertion(item)
        processed=postprocess(item,route)
        self.assertLessEqual(route_objective(item.base,processed),route_objective(item.base,route)+1e-8)
        env=Env(item)
        with self.assertRaises(ValueError):env.step(item.C.shape[1])
        with self.assertRaises(ValueError):env.replay([])

    def test_summed_log_likelihood_has_real_gradient(self):
        item=Cache().prepare(make_complex_scene(2,2,7601000021,'mixed'));model=Policy();torch.manual_seed(1142026)
        data=batch([item,item]);cost,logp,_,_=model(data,sample=True)
        (cost.float()/100*logp).mean().backward()
        self.assertTrue(torch.isfinite(model.query.weight.grad).all())
        self.assertGreater(model.query.weight.grad.abs().sum().item(),0)

    def test_disconnected_mask_has_finite_parameter_gradient(self):
        item=Cache().prepare(make_complex_scene(2,2,7601000022,'regular'))
        for j in range(item.C.shape[1]):
            if np.delete(item.C,j,axis=1).any(1).all():break
        D=item.D.copy();D[j+1,:]=np.inf;D[:,j+1]=np.inf;D[j+1,j+1]=0.
        item.base.nav=dict(item.base.nav,costs=D);item.check()
        model=Policy();cost,logp,_,_=model(batch([item]),sample=True)
        (cost.float()/100*logp).mean().backward()
        self.assertTrue(all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters()))


if __name__=='__main__':unittest.main()
