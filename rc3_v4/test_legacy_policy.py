"""Actual frozen v3 weights on all 18 validation scenes, plus adapter guards."""
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from rc3_v3.model import NumpyPolicy
from .legacy_policy import LegacyMLPPolicy,load_policy,validation_states


class LegacyPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=Path('rc3_v3/deployment/policy.npz')
        cls.original=NumpyPolicy.load(cls.path)
        cls.optimized=load_policy(cls.path)
        cls.states=validation_states('rc3_v3/data_v3_20261102/validation_cases.json')

    def test_all_heads_and_actions_equal_in_18_validation_scenes(self):
        self.assertEqual(len({s['seed'] for s in self.states}),18)
        self.assertGreaterEqual(len(self.states),18)
        with self.optimized.inference_session():
            for state in self.states:
                x,m=state['features'],state['mask']
                for old,new in zip(self.original.outputs(x,m),self.optimized.outputs(x,m)):
                    np.testing.assert_array_equal(old,new)
                self.assertEqual(self.original.select(x,m),self.optimized.select(x,m))

    def test_outputs_do_not_create_threadpool_context(self):
        x,m=self.states[0]['features'],self.states[0]['mask']
        with patch('threadpoolctl.threadpool_limits',side_effect=AssertionError('per-action context')):
            self.optimized.outputs(x,m)
            self.optimized.select(x,m)

    def test_reject_non_npz_or_non_mlp(self):
        with self.assertRaises(ValueError):load_policy('weights.pt')
        meta=dict(self.original.metadata,architecture='attention')
        other=NumpyPolicy(self.original.w,meta)
        with self.assertRaises(ValueError):LegacyMLPPolicy(other)

    def test_invalid_observation_rejected(self):
        x,m=self.states[0]['features'],self.states[0]['mask']
        with self.assertRaises(ValueError):self.optimized.outputs(x[:,:-1],m)
        with self.assertRaises(ValueError):self.optimized.outputs(x,np.zeros_like(m))
        bad=x.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):self.optimized.outputs(bad,m)


if __name__=='__main__':unittest.main()
