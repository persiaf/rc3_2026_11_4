"""Cache semantics and frozen-weight guards; no timing assertions."""
import unittest
from pathlib import Path
import numpy as np
from rc3_v3.environment import Env
from rc3_v4.model import NumpyPolicy
from rc3_v4.features import observation, collate
from rc3_v4.test_features import counterexample
from .policy import load_policy


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = Path('rc3_v4/training/structured.npz')
        cls.original = NumpyPolicy.load(cls.checkpoint)

    def setUp(self):
        self.policy = load_policy(self.checkpoint)
        self.env = Env(counterexample(.1), 8, [0,1,2,3,0])

    def compare(self, env=None, base=None, mask=None):
        env = self.env if env is None else env
        if base is None: base, mask = env.observe()
        with self.original.inference_session(), self.policy.inference_session():
            expected = self.original.outputs(collate([observation(env, base, mask)]))
            actual = self.policy.outputs_env(env, base, mask)
        for left, right in zip(expected, actual): np.testing.assert_array_equal(left, right)
        return actual

    def test_hit_and_dynamic_budget_all_heads(self):
        before = self.compare()
        base, mask = self.env.observe()
        base[:,24] = .25; base[:,25] = .75
        after = self.compare(base=base, mask=mask)
        self.assertEqual((self.policy.cache_misses,self.policy.cache_hits),(1,1))
        self.assertTrue(any(not np.array_equal(x,y) for x,y in zip(before,after)))

    def test_dynamic_action_mask_and_stop_pool(self):
        first = self.compare(); base, mask = self.env.observe(); mask[0] = False
        after = self.compare(base=base,mask=mask)
        self.assertTrue(np.isneginf(after[0][0,0]));self.assertEqual(self.policy.cache_hits,1)
        self.assertNotEqual(float(first[0][0,-1]),float(after[0][0,-1]))

    def test_new_environment_even_same_problem_is_miss(self):
        self.compare();other=Env(self.env.problem,8,[0,1,2,3,0]);self.compare(other)
        self.assertEqual(self.policy.cache_misses,2)

    def test_other_problem_never_reuses_embedding(self):
        self.compare();other=Env(counterexample(10.),8,[0,1,2,3,0]);self.compare(other)
        self.assertEqual(self.policy.cache_misses,2)

    def test_route_change_refreshes(self):
        self.compare();self.env.route=[0,4,3,0];self.compare()
        self.assertEqual(self.policy.cache_misses,2)

    def test_window_order_change_refreshes(self):
        self.compare();base,mask=self.env.observe()
        self.env.windows=list(reversed(self.env.windows))
        self.compare(base=base[::-1].copy(),mask=mask[::-1].copy())
        self.assertEqual(self.policy.cache_misses,2)

    def test_explicit_invalidate_and_model_readonly(self):
        self.compare();self.policy.invalidate();self.compare()
        self.assertEqual((self.policy.cache_misses,self.policy.invalidations),(2,1))
        with self.assertRaises(ValueError):self.policy.model.w['rank.bias'][0]=10.
        with self.assertRaises(ValueError):self.policy.model.w['rank.bias'].setflags(write=True)
        with self.assertRaises(TypeError):self.policy.model.w['rank.bias']=np.zeros(1,np.float32)
        with self.assertRaises(AttributeError):self.policy.model.w={}
        with self.assertRaises(AttributeError):self.policy.model.variant='control'
        with self.assertRaises(AttributeError):self.policy.model=self.original

    def test_invalid_and_empty_observations_rejected_on_miss_and_hit(self):
        base,mask=self.env.observe()
        for primed in (False,True):
            if primed:self.compare()
            invalid=base.copy();invalid[0,0]=np.nan
            for b,m in ((invalid,mask),(base[:,:-1],mask),(base,np.zeros_like(mask)),(base,mask[:-1])):
                with self.assertRaises(ValueError):self.policy.outputs_env(self.env,b,m)
        empty=Env(counterexample(.1),0,[0,1,2,3,0]);b,m=empty.observe()
        with self.assertRaises(ValueError):self.policy.outputs_env(empty,b,m)

    def test_replaced_problem_requires_new_env(self):
        self.compare();self.env.problem.F=self.env.problem.F.copy()
        base,mask=self.env.observe()
        with self.assertRaises(ValueError):self.policy.outputs_env(self.env,base,mask)
        self.compare(Env(self.env.problem,8,[0,1,2,3,0]))

    def test_control_checkpoint_uses_same_cache_contract(self):
        path=Path('rc3_v4/training/control.npz')
        self.policy=load_policy(path);old=self.original
        try:
            self.original=NumpyPolicy.load(path);self.compare();self.compare()
            self.assertEqual(self.policy.cache_hits,1)
        finally:self.original=old

    def test_torch_checkpoint_not_silently_loaded(self):
        with self.assertRaises(ValueError):load_policy('model.pt')


if __name__=='__main__':unittest.main()
