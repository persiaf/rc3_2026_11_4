"""Numerical and structural policy contracts, independent of long training."""
import copy
import unittest
import numpy as np
import torch

from rc3_fast.scenes import make_complex_scene
from rc3_rl_research.adapter import ResearchAdapter
from rc3_rl_research.rl_env import SearchEnv
from .graph_observation import build_policy_observation, static_graph, GRAPH_SCHEMA
from .actor_critic import ActorCritic
from .ppo_math import compute_gae


class ModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.adapter=ResearchAdapter.from_scene(make_complex_scene(3,3,2026110311,'mixed'))
        cls.env=SearchEnv(cls.adapter)
        cls.obs=build_policy_observation(cls.env)
        cls.small=build_policy_observation(SearchEnv(ResearchAdapter.from_scene(make_complex_scene(2,2,2026110312,'regular'))))

    def setUp(self):
        torch.manual_seed(17)

    def test_graph_contains_real_coverage_and_navigation_relations(self):
        o=self.obs;p=self.adapter.problem
        self.assertGreater(len(o['edge_task']),0)
        self.assertGreater(len(o['nav_distance']),0)
        for t,c in zip(o['edge_task'],o['edge_candidate']):
            self.assertTrue(p.F[o['task_ids'][t],o['candidate_nodes'][c]-1].any())
            self.assertEqual(o['task_action'][t],o['candidate_action'][c])
        for s,t,d in zip(o['nav_source'],o['nav_target'],o['nav_distance']):
            self.assertEqual(o['candidate_action'][s],o['candidate_action'][t])
            self.assertAlmostEqual(float(d),p.nav['costs'][o['candidate_nodes'][s],o['candidate_nodes'][t]]/2.2,places=5)
        self.assertIs(static_graph(self.adapter),static_graph(self.adapter))
        with self.assertRaises(ValueError):static_graph(self.adapter)['cover'][0,0]=False

    def test_sampled_old_logprob_recomputes_without_update(self):
        for kind in ('mlp','graph'):
            model=ActorCritic(kind=kind)
            actions,old,values=model.act([self.obs,self.small])
            new,entropy,again=model.evaluate_actions([self.obs,self.small],actions)
            torch.testing.assert_close(old,new,atol=1e-7,rtol=1e-6)
            torch.testing.assert_close(torch.exp(new-old),torch.ones_like(old),atol=1e-7,rtol=1e-6)
            torch.testing.assert_close(values,again)
            self.assertTrue(torch.isfinite(entropy).all())

    def test_padding_and_illegal_actions_have_zero_probability(self):
        model=ActorCritic()
        first=copy.deepcopy(self.obs)
        first['mask'][0]=False
        logits,_=model([first,self.small])
        probabilities=torch.softmax(logits,-1)
        self.assertEqual(float(probabilities[0,0].detach()),0.)
        for i,o in enumerate((first,self.small)):
            self.assertTrue(torch.all(probabilities[i,len(o['mask']):]==0))
            self.assertTrue(torch.all(probabilities[i,:len(o['mask'])][~torch.as_tensor(o['mask'])]==0))
        with self.assertRaises(ValueError):model.evaluate_actions([first],[0])

    def test_only_stop_is_valid_distribution(self):
        stop_only=copy.deepcopy(self.obs)
        stop_only['mask'][:]=False
        index=next(i for i,a in enumerate(stop_only['actions']) if a['kind']=='stop')
        stop_only['mask'][index]=True
        for kind in ('mlp','graph'):
            model=ActorCritic(kind=kind)
            action,logp,value=model.act([stop_only])
            self.assertEqual(int(action.item()),index)
            self.assertAlmostEqual(float(logp.item()),0.,places=7)
            _,entropy,_=model.evaluate_actions([stop_only],action)
            self.assertAlmostEqual(float(entropy.detach().item()),0.,places=7)
            self.assertTrue(torch.isfinite(value).all())

    def test_action_permutation_is_equivariant_and_value_invariant(self):
        model=ActorCritic()
        o=copy.deepcopy(self.obs);permutation=np.arange(len(o['mask']))[::-1].copy()
        inverse=np.argsort(permutation)
        o['features']=o['features'][permutation].copy();o['mask']=o['mask'][permutation].copy()
        o['actions']=[o['actions'][i] for i in permutation]
        o['action_keys']=[o['action_keys'][i] for i in permutation]
        o['task_action']=inverse[o['task_action']];o['candidate_action']=inverse[o['candidate_action']]
        original,v1=model([self.obs]);reordered,v2=model([o])
        torch.testing.assert_close(reordered[0],original[0,torch.as_tensor(permutation)],atol=1e-7,rtol=1e-5)
        torch.testing.assert_close(v1,v2,atol=1e-7,rtol=1e-5)

    def test_real_coverage_and_navigation_are_visible_to_graph_policy(self):
        model=ActorCritic()
        original,_=model([self.obs])
        no_edges=copy.deepcopy(self.obs)
        no_edges['edge_task']=np.empty(0,dtype=np.int64)
        no_edges['edge_candidate']=np.empty(0,dtype=np.int64)
        altered,_=model([no_edges])
        self.assertGreater(float((original-altered).detach().abs().max()),1e-6)
        navigation=copy.deepcopy(self.obs)
        navigation['nav_distance']*=4.
        changed,_=model([navigation])
        self.assertGreater(float((original-changed).detach().abs().max()),1e-7)

    def test_finite_actor_critic_gradients_and_fresh_post_update_encoding(self):
        model=ActorCritic()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        action,old,_=model.act([self.obs,self.small])
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        logp,entropy,value=model.evaluate_actions([self.obs,self.small],action)
        loss=-(logp*torch.tensor([.7,-.2])).mean()+.5*(value-torch.tensor([1.,-.3])).square().mean()-.01*entropy.mean()
        optimizer.zero_grad();loss.backward()
        self.assertGreater(float(model.actor.weight.grad.norm()),0.)
        self.assertGreater(float(model.critic[-1].weight.grad.norm()),0.)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()))
        new,_,_=model.evaluate_actions([self.obs,self.small],action)
        self.assertGreater(float((new-old).detach().abs().max()),1e-5)

    def test_single_and_batched_distribution_agree(self):
        model=ActorCritic()
        single,value=model([self.obs]);batch,values=model([self.small,self.obs])
        torch.testing.assert_close(single[0],batch[1,:single.shape[1]],atol=2e-7,rtol=1e-5)
        torch.testing.assert_close(value[0],values[1],atol=2e-7,rtol=1e-5)

    def test_nonfinite_input_and_no_legal_action_rejected(self):
        model=ActorCritic();o=copy.deepcopy(self.obs);o['features'][0,0]=np.nan
        with self.assertRaises(ValueError):model([o])
        o=copy.deepcopy(self.obs);o['mask'][:]=False
        with self.assertRaises(ValueError):model([o])


class GaeContractTests(unittest.TestCase):
    def test_true_terminal_never_bootstraps_and_does_not_cross_episode(self):
        reward=torch.tensor([[1.,2.],[3.,4.]])
        value=torch.zeros_like(reward)
        nxt=torch.tensor([[99.,5.],[99.,7.]])
        term=torch.tensor([[True,False],[True,False]])
        adv,ret=compute_gae(reward,value,nxt,term,gamma=1.,lam=1.)
        torch.testing.assert_close(adv,torch.tensor([[1.,18.],[3.,11.]]))
        torch.testing.assert_close(ret,adv)

    def test_collector_cut_bootstraps_but_optional_trace_cut_does_not_chain(self):
        reward=torch.tensor([1.,2.]);values=torch.tensor([.5,.7]);nxt=torch.tensor([.7,10.])
        terminated=torch.tensor([False,False])
        adv,_=compute_gae(reward,values,nxt,terminated,gamma=1.,lam=.95)
        self.assertAlmostEqual(float(adv[-1]),11.3,places=5)
        self.assertAlmostEqual(float(adv[0]),1.2+.95*11.3,places=5)
        cut,_=compute_gae(reward,values,nxt,terminated,gamma=1.,lam=.95,trace_mask=torch.tensor([0.,1.]))
        self.assertAlmostEqual(float(cut[0]),1.2,places=5)

    def test_gae_rejects_nonfinite_and_nonboolean_terminal(self):
        with self.assertRaises(ValueError):compute_gae([float('nan')],[0.],[0.],[False])
        with self.assertRaises(ValueError):compute_gae([0.],[0.],[0.],[0.])


if __name__=='__main__':unittest.main()
