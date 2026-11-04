"""Contract and reward tests on synthetic records; no environment/plan calls."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
from .common import METHODS,check_contract,digest,feedback_label,finish_feedback,schedule,teacher_index

class FeedbackTests(unittest.TestCase):
    def contract(self):
        return dict(schema='rc3-g22-frozen-evaluation-v1',evaluation_authorized=True,training_allowed=False,ppo_allowed=False,
            max_plans=72,methods=list(METHODS),cases=[dict(case_id=i,nx=10,ny=10,mode=('regular','mixed','blocked')[i%3]) for i in range(12)],
            config=dict(horizon=16,work_budget=2_000_000),early_stop=dict(max_proposals=4,patience=1,progress_tolerance=1e-12),soft_complete_limit_s=5.)
    def fixture(self):
        observation=dict(problem_id='p',current=[0,3,0],history=[],remaining_work=2000000,
                         actions=[dict(kind='improve'),dict(kind='replace'),dict(kind='stop')],mask=[True,True,True])
        sample=dict(environment_observation=observation,evidence_mask=[True,False,True],legality_mask=[True]*3,
            q_values=[.4,0.,-.1],stop_id=2,teacher_depth=2,state_signature='frozen-finite-label',scene_id='0:123',path=[])
        return observation,{digest(observation):sample}
    def test_evaluation_only_contract(self):self.assertTrue(check_contract(self.contract()))
    def test_no_training_ppo_override(self):
        for field,value in [('training_allowed',True),('training_allowed',0),('ppo_allowed',True),('evaluation_authorized',1),('max_plans',73)]:
            p=self.contract();p[field]=value
            with self.assertRaises(ValueError):check_contract(p)
    def test_missing_or_changed_scope_rejected(self):
        for mutate in (lambda p:p['methods'].reverse(),lambda p:p['cases'].pop(),lambda p:p['early_stop'].update(patience=2),lambda p:p['config'].update(horizon=32)):
            p=self.contract();mutate(p)
            with self.assertRaises(ValueError):check_contract(p)
    def test_cases_balanced_and_unique(self):
        p=self.contract();p['cases'][0]['case_id']=1
        with self.assertRaises(ValueError):check_contract(p)
        p=self.contract();p['cases'][0]['mode']='mixed'
        with self.assertRaises(ValueError):check_contract(p)
    def test_schedule_exact72_rotates_six_positions(self):
        schedule_=schedule(self.contract()['cases']);self.assertEqual(len(schedule_),72)
        for i in range(12):self.assertEqual({m for _,m in schedule_[i*6:i*6+6]},set(METHODS))
        for block in range(2):
            for position in range(6):self.assertEqual({schedule_[(block*6+i)*6+position][1] for i in range(6)},set(METHODS))
    def test_queried_label_uses_only_evidence(self):
        o,index=self.fixture();label=feedback_label(index,o,0)
        self.assertEqual(label['teacher_status'],'action_queried');self.assertAlmostEqual(label['known_delta_vs_stop'],.5)
    def test_unknown_placeholder_never_becomes_zero_label(self):
        o,index=self.fixture();label=feedback_label(index,o,1)
        self.assertEqual(label['teacher_status'],'action_unqueried');self.assertIsNone(label['known_Q']);self.assertIsNone(label['known_delta_vs_stop'])
    def test_changed_history_or_budget_is_unseen_state(self):
        o,index=self.fixture()
        for change in (dict(history=['queried']),dict(remaining_work=1999999)):
            label=feedback_label(index,{**o,**change},0)
            self.assertEqual(label['teacher_status'],'state_unseen');self.assertIsNone(label['known_Q'])
    def test_menu_permutation_is_not_same_action_label(self):
        o,index=self.fixture();o['actions'].reverse()
        self.assertEqual(feedback_label(index,o,0)['teacher_status'],'state_unseen')
    def test_observed_stop_delta_is_zero(self):
        o,index=self.fixture();self.assertEqual(feedback_label(index,o,2)['known_delta_vs_stop'],0.)
    def test_action_illegality_is_rejected(self):
        o,index=self.fixture()
        with self.assertRaises(ValueError):feedback_label(index,o,3)
    def test_suffix_return_is_actual_policy_not_teacher_Q(self):
        rows=[dict(prefix_return=0.,reward=-.1),dict(prefix_return=-.1,reward=.6)]
        result=finish_feedback(rows,.4,[dict(reward=-.1)])
        self.assertAlmostEqual(result[0]['realized_policy_suffix_return'],.4)
        self.assertAlmostEqual(result[1]['realized_policy_suffix_return'],.5)
        self.assertAlmostEqual(result[1]['realized_after_action_return'],-.1)
    def test_reward_event_double_counting_rejected(self):
        with self.assertRaises(ValueError):finish_feedback([dict(prefix_return=0.,reward=.5)],.5,[dict(reward=.1)])
    def test_zero_action_terminal_event(self):self.assertEqual(finish_feedback([],-.2,[dict(reward=-.2)]),[])
    def test_fingerprint_handles_numpy_without_weakening_full_state(self):
        self.assertEqual(digest(dict(a=np.array([1,2]))),digest(dict(a=[1,2])))
    def test_teacher_index_requires_one_root_eight_states(self):
        o,index=self.fixture();s=next(iter(index.values()));samples=[s]+[{**s,'path':[i],'environment_observation':{**o,'steps':i+1}} for i in range(7)]
        self.assertEqual(len(teacher_index(samples,dict(case_id=0,seed=123))),8)
        with self.assertRaises(ValueError):teacher_index(samples[:-1],dict(case_id=0,seed=123))

if __name__=='__main__':unittest.main()
