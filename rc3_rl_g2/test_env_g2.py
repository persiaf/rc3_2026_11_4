"""Targeted G2 state, shared-memory and certificate-reuse regression checks."""
import copy
from dataclasses import asdict,replace
import json
import unittest
from unittest.mock import patch
import numpy as np
from rc3_fast.scenes import make_complex_scene
from rc3_rl_research.adapter import ResearchAdapter
from rc3_rl_research.types import SearchConfig,ProposalResult,Action
from .adapter_g2 import ResearchAdapterG2
from .action_memory import build_actions,proposal_key
from .env_g2 import SearchEnvG2
from .baselines import ILSPolicy,make_policy


class G2EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter=ResearchAdapterG2.from_scene(make_complex_scene(4,4,2026110312))

    def env(self,**kwargs):return SearchEnvG2(self.adapter,**kwargs)

    def assert_rejected_atomically(self,state):
        target=self.env();target.observe();before=target.state_dict()
        with self.assertRaises((ValueError,TypeError,KeyError)):
            target.load_state_dict(state)
        self.assertEqual(target.state_dict(),before)

    def test_finite_cost_reward_and_timing_guards_are_atomic(self):
        original=self.env();original.observe();base=original.state_dict()
        for key in ('anchor_cost','current_cost','best_cost','total_reward','total_quality_reward',
                    'total_work_penalty','total_shaping_reward'):
            for value in (float('nan'),float('inf'),True):
                bad=copy.deepcopy(base);bad[key]=value
                self.assert_rejected_atomically(bad)
        for value in (float('nan'),float('inf'),True,-1.):
            bad=copy.deepcopy(base);bad['timings']['menu_s']=value
            self.assert_rejected_atomically(bad)

    def test_invalid_counts_and_partial_commit_are_rejected(self):
        env=self.env();env.step(0);base=env.state_dict()
        for key in ('steps','work_used'):
            for value in (float('nan'),-1,True,1.5):
                bad=copy.deepcopy(base);bad[key]=value;self.assert_rejected_atomically(bad)
        bad=copy.deepcopy(base);bad['rng_state']['bit_generator']='invalid'
        self.assert_rejected_atomically(bad)

    def test_menu_identity_dimensions_and_features_are_validated(self):
        env=self.env();env.observe();base=env.state_dict()
        changes=[lambda m:m['features'][0].__setitem__(0,float('nan')),
                 lambda m:m['features'][0].__setitem__(0,True),
                 lambda m:m['mask'].__setitem__(0,False),
                 lambda m:m.__setitem__('context_key','foreign'),
                 lambda m:m.__setitem__('work_units',0),
                 lambda m:m['actions'][0].__setitem__('start',2),
                 lambda m:m['features'].pop()]
        for mutate in changes:
            bad=copy.deepcopy(base);mutate(bad['menu']);self.assert_rejected_atomically(bad)

    def test_active_and_terminal_state_restore_and_continue(self):
        first=self.env(seed=12);first.step(0)
        saved=first.state_dict();second=self.env(seed=99);second.load_state_dict(saved)
        self.assertEqual(first.observe(),second.observe())
        a=next(i for i,v in enumerate(first.observe()['actions']) if v['kind']=='stop')
        x=first.step(a);y=second.step(a)
        self.assertEqual(x[:4],y[:4]);self.assertEqual(first.current,second.current)
        terminal=first.state_dict();third=self.env();third.load_state_dict(terminal)
        self.assertEqual(third.state_dict(),terminal)

    def test_failed_finalization_payload_does_not_partially_commit(self):
        env=self.env();env.step(len(env.observe()['actions'])-1);bad=env.state_dict()
        bad['returned_certificate']['plan']['route_length_m']+=1
        self.assert_rejected_atomically(bad)
        bad=env.state_dict();bad['finalization']['fallback_used']=True
        # Even unchanged anchor output requires consistent correction; corrupt it.
        bad['finalization']['quality_correction']=1.
        self.assert_rejected_atomically(bad)

    def test_proposal_identity_excludes_replay_budget_history(self):
        env=self.env();obs=env.observe();action=env._menu.actions[0]
        key=proposal_key(self.adapter.problem_id,env.current,action,env.config)
        cfg=replace(env.config,work_budget=123456,horizon=3,epsilon=.01)
        self.assertEqual(key,proposal_key(self.adapter.problem_id,env.current,action,cfg))
        m=build_actions(self.adapter.problem,env.current,remaining_work=500000,steps_remaining=3,
                        history=['another'],config=env.config,problem_id=self.adapter.problem_id)
        self.assertNotEqual(obs['menu_context_key'],m.context_key)

    def test_memory_filters_before_cap_and_replenishes(self):
        env=self.env();original=env.observe();self.assertEqual(len(original['actions']),17)
        memory={}
        for action,key in zip(env._menu.actions[:-1],original['proposal_keys'][:-1]):
            memory[key]=dict(attempts=1,no_change=True,failed=False)
        menu=build_actions(self.adapter.problem,env.current,remaining_work=env.remaining_work,
            steps_remaining=16,config=env.config,problem_id=self.adapter.problem_id,attempt_memory=memory)
        self.assertGreater(len(menu.actions),1)
        self.assertFalse({a.key for a in menu.actions[:-1]} & {a['key'] for a in original['actions'][:-1]})
        self.assertEqual(menu.diagnostics['memory_filtered'],16)

    def test_unchanged_and_failed_attempts_block_shared_proposal_key(self):
        for failure in (False,True):
            def custom(problem,route,action,config):
                return ProposalResult(None if failure else list(route),not failure,False,1,
                                      'synthetic_failure' if failure else None)
            env=self.env(proposer=custom);obs=env.observe();key=obs['proposal_keys'][0]
            _,_,_,_,info=env.step(0)
            self.assertNotIn(key,env.observe()['proposal_keys'])
            self.assertEqual(env.attempt_memory[key]['failed'],failure)
            self.assertEqual(env.attempt_memory[key]['no_change'],not failure)
            self.assertGreater(info['work_units'],0)

    def test_fork_memory_and_bound_methods_are_independent(self):
        def noop(problem,route,action,config):return ProposalResult(list(route),True,False,1)
        parent=self.env(proposer=noop);before=parent.observe();child=parent.fork();child.step(0)
        self.assertIs(child.adapter,parent.adapter)
        self.assertEqual(parent.observe(),before);self.assertEqual(parent.attempt_memory,{})
        self.assertEqual(len(child.attempt_memory),1)

    def test_certified_bound_rejection_is_shared_and_charged_before_refill(self):
        from rc3_rl_research.types import Certificate
        real=self.adapter.certify
        checks=[]
        def high_cost(route,method='test'):
            certificate=real(route,method)
            if method=='rl_search_accept':
                checks.append(method)
                return Certificate(certificate.route,certificate.plan,self.adapter.anchor.cost*1.1)
            return certificate
        # Deliberately valid proposal; the independent certifier reports its
        # certified high cost. This exercises acceptance, not DP infeasibility.
        def custom(problem,route,action,config):
            changed=[route[0],*reversed(route[1:-1]),route[-1]]
            return ProposalResult(changed,True,True,7)
        env=self.env(proposer=custom,certifier=high_cost);obs=env.observe();key=obs['proposal_keys'][0]
        expected_work=obs['pending_menu_work']+7
        _,_,_,_,info=env.step(0)
        self.assertEqual(checks,['rl_search_accept'])
        self.assertEqual(info['work_units'],expected_work)
        self.assertFalse(info['accepted'])
        self.assertEqual(info['failure_reason'],'outside_best_plus_epsilon_anchor')
        self.assertTrue(env.attempt_memory[key]['bound_rejected'])
        self.assertFalse(env.attempt_memory[key]['no_change'])
        nextobs=env.observe();self.assertNotIn(key,nextobs['proposal_keys'])
        self.assertEqual(len(nextobs['actions']),17)
        for name in ('random','detour_memory','ils','stop'):
            policy=make_policy(name);picked=policy.select(nextobs,env.rng)
            self.assertTrue(nextobs['mask'][picked])
            self.assertNotEqual(nextobs['proposal_keys'][picked],key)
        restored=self.env();restored.load_state_dict(env.state_dict())
        self.assertEqual(restored.observe(),nextobs)

    def test_light_recording_preserves_restorable_state(self):
        env=self.env(record_transitions=False);env.step(0)
        self.assertFalse(env.transitions)
        other=self.env();other.load_state_dict(env.state_dict())
        self.assertFalse(other.record_transitions)
        self.assertEqual(env.observe(),other.observe())

    def test_stop_only_budget_menu_and_external_event_total(self):
        env=self.env(config=SearchConfig(work_budget=1));obs=env.observe()
        self.assertEqual(sum(obs['mask']),1)
        result=env.step(len(obs['actions'])-1)
        self.assertTrue(result[2]);self.assertEqual(result[0]['mask'],[])
        env=self.env();_,r,_,_,_=env.step(0)
        event_r,event=env.terminate('test_external')
        self.assertAlmostEqual(r+event_r,env.total_reward)
        self.assertIn('event_id',event)
        again,_=env.terminate('test_external');self.assertEqual(again,0.)

    def test_acceptance_cache_reuses_only_exact_route_final_always_checks(self):
        adapter=ResearchAdapterG2.from_scene(make_complex_scene(2,2,2026110313))
        route=adapter.anchor.route
        before=copy.deepcopy(adapter.cache_stats)
        cached=adapter.certify(route,'rl_search_accept')
        self.assertEqual(adapter.cache_stats['hits'],before['hits']+1)
        self.assertTrue(cached.plan['validation']['valid_model_plan'])
        with patch.object(ResearchAdapter,'certify',side_effect=ValueError('forced final checker failure')):
            with self.assertRaisesRegex(ValueError,'forced final'):
                adapter.certify(route,'rl_search_final')
        restored=ResearchAdapterG2.from_payload(adapter.payload())
        self.assertEqual(restored.anchor.route,route)

    def test_ils_noop_does_not_reset_stagnation(self):
        policy=ILSPolicy(stagnation_limit=2,max_perturbations=1)
        info=dict(action=dict(kind='improve'),accepted=True,best_improved=False,route_changed=False)
        policy.update(info);policy.update(info);self.assertEqual(policy.stagnation,2)
        env=self.env();obs=env.observe();a=policy.select(obs,env.rng)
        self.assertEqual(obs['actions'][a]['kind'],'replace')
        policy.update(dict(action=dict(kind='replace'),best_improved=False,route_changed=True))
        policy.update(info);policy.update(info)
        self.assertEqual(obs['actions'][policy.select(obs,env.rng)]['kind'],'stop')
        for name in ('random','stop','detour_memory','ils'):
            candidate=make_policy(name);state=candidate.state_dict();candidate.load_state_dict(state)


if __name__=='__main__':unittest.main()
