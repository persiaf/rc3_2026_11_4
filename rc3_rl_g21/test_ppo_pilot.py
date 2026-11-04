"""Pure arrays, tensors and mocked collector tests; no workers or environments."""
import copy
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from .collector_pilot import PilotCollector
from .train_ppo_pilot import (collapse_decision,critic_fit,distribution,
    raw_rollout_metrics,validate_authorization)
from .worker_pilot import root_stop_reference


class PilotGuardTests(unittest.TestCase):
    def authorization(self):
        p=dict(seed=123,ppo=dict(actions_max=4096))
        a=dict(authorized=True,protocol_sha256='p',seed=123,actions_max=4096,
            gates={k:True for k in ('toy','opportunity','early_stop_rules_frozen','one_step_control_ready')},
            required_artifacts=[dict(path='gate.json',sha256='s')],initial_checkpoint='model.pt',
            initial_checkpoint_sha256='m',g2_protocol_sha256='g')
        return a,p

    def test_explicit_authorization_required(self):
        a,p=self.authorization()
        self.assertEqual(validate_authorization(a,p,'p'),4096)
        for key,value in [('authorized',False),('protocol_sha256','wrong'),('seed',124),
                          ('actions_max',8192),('actions_max',True),('required_artifacts',[])]:
            bad=copy.deepcopy(a);bad[key]=value
            with self.assertRaises(ValueError):validate_authorization(bad,p,'p')

    def test_all_gates_require_actual_true(self):
        a,p=self.authorization()
        for value in (False,1,'true'):
            bad=copy.deepcopy(a);bad['gates']['opportunity']=value
            with self.assertRaises(ValueError):validate_authorization(bad,p,'p')

    def test_stop_reference_no_environment(self):
        c=dict(potential_weight=0,lambda_work=.01,work_scale=10000)
        self.assertAlmostEqual(root_stop_reference(25000,c),-.025)
        for changes in ({'potential_weight':1},{'work_scale':0}):
            with self.assertRaises(ValueError):root_stop_reference(25000,{**c,**changes})

    def test_collapse_requires_both_conditions(self):
        kinds=['continue']*10+['stop']*1014
        self.assertTrue(collapse_decision(kinds,.5,.5)['stop'])
        self.assertFalse(collapse_decision(kinds,.6,.5)['stop'])
        self.assertFalse(collapse_decision(['continue']*11+['stop']*1013,.4,.5)['stop'])
        with self.assertRaises(ValueError):collapse_decision(['stop'],.1,.2)

    def test_critic_constant_target_explained_variance_is_null(self):
        result=critic_fit([1,3],[2,2])
        self.assertEqual(result['mse'],1.)
        self.assertIsNone(result['explained_variance'])
        self.assertAlmostEqual(critic_fit([0,1,2],[0,1,2])['explained_variance'],1.)

    def test_raw_advantages_are_not_centered_and_terminal_bootstrap_masked(self):
        batch=dict(rewards=np.array([[1.,2.]],np.float32),values=np.array([[0.,0.]],np.float32),
            next_values=np.array([[99.,3.]],np.float32),terminated=np.array([[True,False]]),
            action_kinds=['stop','improve'])
        result,raw,returns=raw_rollout_metrics(batch,dict(gamma=1.,gae_lambda=.95))
        np.testing.assert_allclose(raw,[1.,5.])
        np.testing.assert_allclose(returns,[1.,5.])
        self.assertEqual(result['raw_advantages']['stop']['mean'],1.)
        self.assertEqual(result['raw_advantages']['continue_actions']['mean'],5.)
        self.assertEqual(result['critic_before']['mse'],13.)

    def test_empty_distribution_is_not_zero_evidence(self):
        d=distribution([])
        self.assertEqual(d['n'],0);self.assertIsNone(d['mean'])

    def test_collector_rejects_overshoot_before_model_or_worker(self):
        c=object.__new__(PilotCollector);c.workers=8;c.committed=3840;c.actions_max=4096
        c.command_all=Mock();model=Mock()
        with self.assertRaises(ValueError):c.collect(model,33)
        model.act.assert_not_called();c.command_all.assert_not_called()

    def test_collector_terminal_bootstrap_and_real_proposal_delta(self):
        c=object.__new__(PilotCollector);c.workers=2;c.committed=0;c.actions_max=4096
        c.observations=[{},{}]
        c.command_all=Mock(return_value=[dict(observation={},reward=1.,terminated=True,summary=None,
            action_kind='stop',route_changed=False,proposal_calls=0,transition={}),
            dict(observation={},reward=2.,terminated=False,summary=None,
            action_kind='improve',route_changed=True,proposal_calls=1,transition={})])
        model=Mock();model.act.return_value=(torch.tensor([1,0]),torch.tensor([-.1,-.2]),torch.tensor([0.,1.]))
        model.return_value=(None,torch.tensor([100.,3.]))
        batch=c.collect(model,1)
        np.testing.assert_array_equal(batch['next_values'],[[0.,3.]])
        np.testing.assert_array_equal(batch['actual_proposal_calls'],[0,1])
        self.assertEqual(c.committed,2)

    def test_resume_disabled_without_worker_query(self):
        c=object.__new__(PilotCollector)
        with self.assertRaises(ValueError):c.load_state_dict([])


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
