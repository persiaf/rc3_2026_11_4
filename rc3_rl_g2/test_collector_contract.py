"""Event-ledger integration probes are synthetic semantics, not training data."""
import copy
import unittest
import torch
from rc3_rl_research.test_env import make_env, FixtureAdapter
from .collector_contract import merge_terminal_events
from .ppo_math import compute_gae


def sample(env, action):
    next_obs,reward,terminated,truncated,_=env.step(action)
    return dict(action=action,reward=reward,terminated=terminated,truncated=truncated,
                next_observation=next_obs,next_value=11.,bootstrap_value=11.)


class TerminalEventContractTests(unittest.TestCase):
    def test_pending_menu_fee_is_folded_once_and_terminal_no_bootstrap(self):
        env=make_env();records=[sample(env,0)]
        self.assertAlmostEqual(records[0]['reward'],-.005)
        env.terminate('test_external_end')
        self.assertAlmostEqual(env.events[0]['reward'],-.002)
        original=copy.deepcopy(records)
        merged,seen=merge_terminal_events(records,env.events,env.observe())
        self.assertEqual(records,original)
        self.assertAlmostEqual(merged[0]['reward'],-.007)
        self.assertAlmostEqual(sum(v['reward'] for v in merged),env.total_reward)
        self.assertTrue(merged[-1]['terminated'])
        self.assertEqual(merged[-1]['bootstrap_value'],0.)
        self.assertEqual(merged[-1]['next_value'],0.)
        duplicate,again=merge_terminal_events(merged,env.events,env.observe(),seen)
        self.assertEqual(duplicate,merged);self.assertEqual(again,seen)
        adv,_=compute_gae([merged[0]['reward']],[0.],[999.],[True])
        self.assertAlmostEqual(float(adv[0]),-.007,places=7)

    def test_fallback_retracts_quality_and_potential_zeroing_is_included(self):
        adapter=FixtureAdapter()
        def final_fails(route,method='test'):
            if method=='rl_search_final':raise ValueError('injected final failure')
            return adapter.certify(route,method)
        env=make_env(certifier=final_fails,potential_weight=.7)
        records=[sample(env,0),sample(env,1)]
        self.assertGreater(env.total_quality_reward,0.)
        env.terminate('test_external_end')
        self.assertTrue(env.finalization['fallback_used'])
        merged,_=merge_terminal_events(records,env.events,env.observe())
        self.assertAlmostEqual(sum(v['reward'] for v in merged),env.total_reward)
        self.assertAlmostEqual(env.total_quality_reward,0.)
        self.assertAlmostEqual(env.total_shaping_reward,0.)
        self.assertAlmostEqual(env.total_reward,-env.total_work_penalty)

    def test_zero_action_episode_remains_metrics_only(self):
        env=make_env();env.observe();env.terminate('zero_action_deadline')
        records,seen=merge_terminal_events([],env.events,env.observe())
        self.assertEqual(records,[]);self.assertEqual(len(seen),1)
        self.assertLess(env.total_reward,0.)
        self.assertAlmostEqual(env.total_reward,sum(e['reward'] for e in env.events))

    def test_natural_stop_horizon_budget_have_no_external_merge(self):
        for env,action in ((make_env(),2),(make_env(horizon=1),0),(make_env(work_budget=4),0)):
            records=[sample(env,action)]
            self.assertTrue(env.done)
            self.assertEqual(env.events,[])
            merged,seen=merge_terminal_events(records,env.events,env.observe())
            self.assertEqual(merged,records);self.assertEqual(seen,set())
            self.assertAlmostEqual(sum(v['reward'] for v in records),env.total_reward)

    def test_collector_cut_keeps_search_state_and_true_next_value(self):
        env=make_env();record=sample(env,0)
        before=(env.current.copy(),env.work_used,env.total_reward)
        boundary=env.rollout_boundary(5.)
        self.assertTrue(boundary['truncated']);self.assertFalse(boundary['terminated'])
        self.assertEqual((env.current,env.work_used,env.total_reward),before)
        adv,_=compute_gae([record['reward']],[0.],[boundary['bootstrap_value']],[False])
        self.assertAlmostEqual(float(adv[0]),4.995,places=6)
        # Continuing the same environment after a collection boundary remains
        # a genuine second action rather than a reset-state bootstrap.
        second=sample(env,1)
        self.assertEqual(env.steps,2);self.assertGreater(second['reward'],0.)

    def test_invalid_reward_or_nonterminal_observation_is_rejected(self):
        env=make_env();records=[sample(env,0)];env.terminate('bad_event_test')
        broken=copy.deepcopy(env.events);broken[0]['reward']=float('nan')
        with self.assertRaises(ValueError):merge_terminal_events(records,broken,env.observe())
        with self.assertRaises(ValueError):merge_terminal_events(records,env.events,dict(terminated=False))


if __name__=='__main__':unittest.main()
