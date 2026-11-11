"""Deterministic artificial-tensor R04 interfaces; no data/weight file I/O.

No optimizer is constructed or called.  Parameter assignment below creates
test fixtures, not learning updates.  One backward checks the centering
Jacobian; all other calls are forwards or saved-state arithmetic.
"""
import copy
import unittest

import torch
from torch import nn

from rc3_conservative_refinement.guard import check_update, restore_if_rejected
from .relative_actor import (RelativeCorrection, residual_details, rank_response,
                             actor_only_package, load_actor_only_package, SCHEMA)


BASE = {'path':'synthetic/frozen_base.pt','sha256':'a'*64}


def deterministic_head():
    model=RelativeCorrection()
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        model.head[0].weight[0,0]=1.
        model.head[2].weight[0,0]=1.
    return model


def fixture():
    features=torch.zeros((1,3,206),dtype=torch.float32)
    features[0,:,0]=torch.tensor([0.,1.,2.])
    base=torch.tensor([[.25,-.5,2.]],dtype=torch.float32)
    mask=torch.tensor([[True,True,False]])
    return base,features,mask


class NoStepState:
    """Fake Adam-shaped state, solely for the existing restore interface."""
    def __init__(self):
        self.value={'state':{0:{'step':torch.tensor(0.),'exp_avg':torch.tensor([.5])}},
                    'param_groups':[{'lr':3e-5,'params':[0]}]}
    def state_dict(self): return copy.deepcopy(self.value)
    def load_state_dict(self,value): self.value=copy.deepcopy(value)
    def step(self): raise AssertionError('No optimizer step is authorized')


class RelativeActorTests(unittest.TestCase):
    def test_zero_last_layer_exact_supplied_sl_logits_and_mask(self):
        model=RelativeCorrection();z,x,mask=fixture()
        z[0,2]=float('nan')
        out=model(z,x,mask)
        self.assertTrue(torch.equal(out,z.masked_fill(~mask,-torch.inf)))
        self.assertTrue(torch.equal(torch.isfinite(out),mask))

    def test_legal_mean_excludes_illegal_extreme_payload(self):
        raw=torch.tensor([[1.,float('nan'),3.]],dtype=torch.float64)
        mask=torch.tensor([[True,False,True]])
        out=residual_details(raw,mask)
        self.assertEqual(float(out['legal_raw_mean']),2.)
        self.assertTrue(torch.equal(out['centered_scores'],torch.tensor([[-1.,0.,1.]],dtype=torch.float64)))
        self.assertEqual(float(out['delta'][0,1]),0.)

    def test_single_legal_action_zero_even_with_large_raw_score(self):
        raw=torch.tensor([[99.,-17.,float('inf')]],dtype=torch.float32)
        out=residual_details(raw,torch.tensor([[True,False,False]]))
        self.assertTrue(torch.equal(out['delta'],torch.zeros_like(raw)))

    def test_masked_mean_is_per_state_not_across_batch(self):
        raw=torch.tensor([[1.,3.],[10.,10.]],dtype=torch.float64)
        out=residual_details(raw,torch.ones_like(raw,dtype=torch.bool))
        self.assertTrue(torch.equal(out['legal_raw_mean'],torch.tensor([[2.],[10.]],dtype=torch.float64)))
        self.assertTrue(torch.equal(out['delta'][1],torch.zeros(2,dtype=torch.float64)))

    def test_all_illegal_refused(self):
        with self.assertRaises(ValueError):
            residual_details(torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool))

    def test_nonfinite_legal_refused(self):
        with self.assertRaises(ValueError):
            residual_details(torch.tensor([[float('nan')]]),torch.ones(1,1,dtype=torch.bool))
        # Finite raw logits can still overflow log-softmax or exp-ratios.
        huge=torch.tensor([[1e308,-1e308]],dtype=torch.float64)
        legal=torch.ones(1,2,dtype=torch.bool)
        with self.assertRaises(ValueError):
            rank_response(huge,huge,huge,legal)
        before=torch.tensor([[1000.,-1000.]],dtype=torch.float64)
        with self.assertRaises(ValueError):
            rank_response(before,before,-before,legal,
                          actions=torch.tensor([1]),collection_logits=before)

    def test_bound_and_pre_tanh_common_shift_invariance(self):
        raw=torch.tensor([[-8.,-1.,2.,6.]],dtype=torch.float64)
        mask=torch.tensor([[True,True,False,True]])
        a=residual_details(raw,mask);b=residual_details(raw+13.,mask)
        self.assertTrue(torch.equal(a['delta'],b['delta']))
        self.assertTrue(bool((a['delta'].abs()<=.25).all()))

    def test_post_tanh_mean_not_guaranteed_zero(self):
        raw=torch.tensor([[-1.,-1.,2.]],dtype=torch.float64)
        out=residual_details(raw,torch.ones_like(raw,dtype=torch.bool))
        self.assertGreater(abs(float(out['legal_post_tanh_mean'])),.01)

    def test_forbidden_post_centering_can_exceed_bound(self):
        raw=torch.tensor([[-10.,5.,5.]],dtype=torch.float64)
        out=residual_details(raw,torch.ones_like(raw,dtype=torch.bool))
        postcenter=out['delta']-out['legal_post_tanh_mean']
        self.assertGreater(float(postcenter.abs().max()),.25)

    def test_illegal_feature_nan_does_not_affect_mean_or_legal_logits(self):
        model=deterministic_head();z,x,mask=fixture()
        with torch.no_grad(): model.head[-1].weight[0,0]=.1
        a=model(z,x,mask);x[0,2,:]=float('nan');b=model(z,x,mask)
        self.assertTrue(torch.equal(a,b))

    def test_shared_output_bias_cancels_and_frozen_base_has_no_grad(self):
        model=deterministic_head();z,x,mask=fixture()
        with torch.no_grad(): model.head[-1].weight[0,0]=.1
        a=model(z,x,mask)
        with torch.no_grad(): model.head[-1].bias.fill_(9.)
        b=model(z,x,mask)
        self.assertTrue(torch.allclose(a[mask],b[mask],atol=2e-7,rtol=0.))
        frozen=nn.Linear(206,1).eval().requires_grad_(False)
        with torch.no_grad():
            frozen.weight.fill_(.125);frozen.bias.zero_()
            model.head[-1].weight.zero_();model.head[-1].bias.zero_()
        x=x.requires_grad_(True);z=frozen(x).squeeze(-1).detach()
        model.train();model(z,x,mask)[0,0].backward()
        self.assertIsNone(x.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in frozen.parameters()))
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()),0.)
        self.assertTrue(torch.equal(model.head[-1].bias.grad,torch.zeros_like(model.head[-1].bias)))

    def test_existing_guard_accept_and_reject_then_exact_fixture_restore(self):
        model=deterministic_head();_,x,mask=fixture();z=torch.zeros((1,3))
        old=model(z,x,mask).detach()
        with torch.no_grad():model.head[-1].weight[0,0]=.001
        accepted=check_update(z,old,model(z,x,mask),mask)
        self.assertTrue(accepted['accepted'])
        self.assertEqual(accepted['optimizer_steps_executed'],0)
        opt=NoStepState();before=copy.deepcopy(model.state_dict());before_opt=opt.state_dict()
        with torch.no_grad():model.head[-1].weight[0,0]=2.
        opt.value['state'][0]['exp_avg'].fill_(17.)
        rejected=check_update(z,old,model(z,x,mask),mask)
        self.assertFalse(rejected['accepted'])
        receipt=restore_if_rejected(model,opt,before,before_opt,rejected)
        self.assertTrue(receipt['restored'])
        self.assertEqual(receipt['optimizer_steps_executed'],0)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(torch.equal(before_opt['state'][0]['exp_avg'],opt.state_dict()['state'][0]['exp_avg']))

    def test_near_tie_flip_is_recorded_and_can_pass_existing_guard(self):
        model=deterministic_head();_,x,mask=fixture()
        z=torch.tensor([[.0001,0.,0.]])
        before=model(z,x,mask).detach()
        with torch.no_grad():model.head[-1].weight[0,0]=.001
        after=model(z,x,mask).detach()
        guard=check_update(z,before,after,mask)
        self.assertTrue(guard['accepted'])
        self.assertEqual(guard['greedy_changes_old_to_new'],1)
        row=rank_response(z,before,after,mask,actions=torch.tensor([0]),collection_logits=before)[0]
        self.assertTrue(row['top1_changed']);self.assertEqual(row['after_top1'],1)
        self.assertEqual(row['rank_changed_candidates'],2)
        self.assertEqual(row['pre_update_ratio'],1.)
        self.assertNotEqual(row['post_update_ratio'],1.)

    def test_rank_and_ratios_record_actual_pre_post_separately(self):
        base=torch.tensor([[.1,0.,-.2,99.]])
        before=base.clone();after=torch.tensor([[-.1,.2,-.2,float('nan')]])
        mask=torch.tensor([[True,True,True,False]])
        rows=rank_response(base,before,after,mask,actions=torch.tensor([0]),
                           collection_logits=before,candidate_ids=[10,20,30,40])
        row=rows[0]
        self.assertEqual(row['before_top1'],0);self.assertEqual(row['after_top1'],1)
        self.assertEqual(row['before_ranks'],[1,2,3]);self.assertEqual(row['after_ranks'],[2,1,3])
        self.assertEqual(row['rank_changed_candidates'],2)
        self.assertEqual(row['pre_update_ratio'],1.)
        self.assertNotEqual(row['post_update_ratio'],1.)
        self.assertEqual(row['legal_candidate_ids'],[10,20,30])

    def test_missing_collection_ratio_stays_missing_and_ties_stable(self):
        z=torch.zeros((1,3));mask=torch.tensor([[True,True,False]])
        row=rank_response(z,z,z,mask,actions=torch.tensor([0]))[0]
        self.assertEqual(row['base_top1'],0);self.assertEqual(row['before_ranks'],[1,2])
        self.assertIsNone(row['pre_update_ratio']);self.assertIsNone(row['post_update_ratio'])

    def test_package_reload_identical_logits_ranks_and_no_training_objects(self):
        model=deterministic_head();z,x,mask=fixture()
        with torch.no_grad():model.head[-1].weight[0,0]=.05
        model.eval();package=actor_only_package(model,BASE)
        self.assertEqual(package['schema'],SCHEMA)
        self.assertFalse(package['critic_included']);self.assertFalse(package['optimizer_included'])
        loaded=load_actor_only_package(package,expected_base_identity=BASE)
        a=model(z,x,mask);b=loaded(z,x,mask)
        self.assertTrue(torch.equal(a,b))
        rows=rank_response(z,a,b,mask,actions=torch.tensor([0]),collection_logits=a)
        self.assertFalse(rows[0]['top1_changed']);self.assertEqual(rows[0]['rank_changed_candidates'],0)
        self.assertFalse(loaded.training);self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))

    def test_loader_rejects_absolute_schema_training_state_or_bad_base(self):
        model=deterministic_head();package=actor_only_package(model,BASE)
        bad=copy.deepcopy(package);bad['schema']='rc3-conservative-actor-only-206-v1'
        with self.assertRaises(ValueError):load_actor_only_package(bad,expected_base_identity=BASE)
        for key in ('critic','adam','optimizer'):
            bad=copy.deepcopy(package);bad[key]={}
            with self.assertRaises(ValueError):load_actor_only_package(bad,expected_base_identity=BASE)
        with self.assertRaises(ValueError):
            load_actor_only_package(package,expected_base_identity={'path':BASE['path'],'sha256':'b'*64})
        bad=copy.deepcopy(package);bad['correction_state']['head.4.bias'].fill_(.3)
        with self.assertRaises(ValueError):load_actor_only_package(bad,expected_base_identity=BASE)

    def test_width_and_amplitude_cannot_be_silently_changed(self):
        for kwargs in ({'input_dim':202},{'hidden':128},{'max_adjustment':.5},{'max_adjustment':True}):
            with self.assertRaises(ValueError):RelativeCorrection(**kwargs)


if __name__=='__main__':unittest.main()
