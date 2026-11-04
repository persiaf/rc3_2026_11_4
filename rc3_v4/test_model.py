"""Graph input masking, portable parity, and strict deployment metadata tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from .model import (batch_torch, make_policy, save_policy, load_policy,
                    export_numpy, NumpyPolicy)


def fixture(seed=19):
    rng = np.random.default_rng(seed)
    s, w, c, t = 2, 3, 5, 4
    b = dict(base=rng.normal(size=(s,w,28)).astype('f'),
             mask=np.array([[True,True,True],[True,True,False]]),
             cand=rng.normal(size=(s,w,c,10)).astype('f'),
             task=rng.normal(size=(s,w,t,4)).astype('f'),
             reach=rng.integers(2,size=(s,w,t,c,2)).astype('f'),
             dist=rng.random((s,w,c,c),dtype=np.float32),
             cmask=np.ones((s,w,c),bool), tmask=np.ones((s,w,t),bool))
    b['cmask'][0,1,3:] = False
    b['tmask'][0,1,2:] = False
    b['cmask'][1,2] = False
    b['tmask'][1,2] = False
    return b


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def create(self, variant='structured'):
        torch.manual_seed(23)
        return make_policy(np.zeros(28,'f'), np.ones(28,'f'), variant).eval()

    def compare(self, left, right, tol=3e-6):
        for a, b in zip(left, right):
            np.testing.assert_allclose(a, b, atol=tol, rtol=tol)

    def test_numpy_parity_and_metadata(self):
        b = fixture()
        for variant in ('control','structured'):
            model = self.create(variant)
            with tempfile.TemporaryDirectory() as folder:
                pt, npz = Path(folder)/'p.pt', Path(folder)/'p.npz'
                save_policy(pt, model, seed=23, epoch=0)
                export_numpy(pt, npz)
                policy = load_policy(npz)
                with policy.inference_session():
                    self.compare(model.outputs(b), policy.outputs(b))
                cpu = load_policy(pt)
                self.compare(model.outputs(b), cpu.outputs(b), tol=0)
                if torch.cuda.is_available():
                    gpu = load_policy(pt, 'cuda')
                    self.compare(model.outputs(b), gpu.outputs(b), tol=1e-5)
                weights = dict(policy.w)
                bad = dict(policy.metadata, phase='ppo')
                with self.assertRaises(ValueError): NumpyPolicy(weights, bad)
                weights['extra'] = np.zeros(1,'f')
                with self.assertRaises(ValueError): NumpyPolicy(weights, policy.metadata)

    def test_control_hides_all_graph_values(self):
        b = fixture()
        c = fixture(92)
        c['base'], c['mask'] = b['base'].copy(), b['mask'].copy()
        control = self.create('control')
        structured = self.create('structured')
        self.assertEqual(sum(p.numel() for p in control.parameters()),
                         sum(p.numel() for p in structured.parameters()))
        self.compare(control.outputs(b), control.outputs(c), tol=0)
        self.assertGreater(np.max(np.abs(structured.outputs(b)[1]-structured.outputs(c)[1])),1e-5)

    def test_task_and_candidate_permutation_invariant(self):
        b = fixture()
        c = copy.deepcopy(b)
        cp, tp = [3,0,4,1,2], [2,0,3,1]
        c['cand'] = b['cand'][:,:,cp,:]
        c['cmask'] = b['cmask'][:,:,cp]
        c['task'] = b['task'][:,:,tp,:]
        c['tmask'] = b['tmask'][:,:,tp]
        c['reach'] = b['reach'][:,:,tp,:,:][:,:,:,cp,:]
        c['dist'] = b['dist'][:,:,cp,:][:,:,:,cp]
        model = self.create()
        # Only the all-pad window's auxiliary heads can depend on the numerical
        # dummy key. All observable legal outputs and global STOP are invariant.
        left, right = model.outputs(b), model.outputs(c)
        self.compare((left[0],left[1][b['mask']],left[2][b['mask']]),
                     (right[0],right[1][b['mask']],right[2][b['mask']]))

    def test_padding_values_do_not_change_valid_outputs(self):
        b = fixture()
        c = copy.deepcopy(b)
        c['cand'][~b['cmask']] = 1000
        c['task'][~b['tmask']] = -1000
        edges = b['tmask'][..., :, None,None] & b['cmask'][...,None,:,None]
        c['reach'] = np.where(edges, b['reach'], 1.)
        valid_pairs = b['cmask'][..., :,None] & b['cmask'][...,None,:]
        c['dist'] = np.where(valid_pairs, b['dist'], 1000.)
        c['base'][~b['mask']] = 999.
        for variant in ('control','structured'):
            model = self.create(variant)
            left, right = model.outputs(b), model.outputs(c)
            self.compare((left[0],left[1][b['mask']],left[2][b['mask']]),
                         (right[0],right[1][b['mask']],right[2][b['mask']]))

    def test_actual_relation_and_distance_influence_structured(self):
        b = fixture()
        model = self.create()
        original = model.outputs(b)[1]
        for key in ('reach','dist'):
            changed = copy.deepcopy(b)
            changed[key][:] = 0
            delta = np.abs(original-model.outputs(changed)[1])[b['mask']].max()
            self.assertGreater(delta, 1e-7, key)

    def test_backward_finite_and_empty_action_rejected(self):
        model = self.create()
        b = fixture()
        logits, gain, binary = model(batch_torch(b))
        loss = logits[:,-1].sum()+gain[b['mask']].sum()+binary[b['mask']].sum()
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        meta = dict(schema='rc3-v4-local-graph-v1',phase='gain_supervised',
                    feature_names=__import__('rc3_v3.environment',fromlist=['FEATURE_NAMES']).FEATURE_NAMES,
                    variant='structured',dim=32,mean=[0.]*28,std=[1.]*28,
                    stop_rule='binary_threshold',stop_threshold=.9)
        policy = NumpyPolicy({k:v.detach().numpy() for k,v in model.state_dict().items()}, meta)
        bad = copy.deepcopy(b);bad['mask'][0] = False
        with self.assertRaises(ValueError): policy.outputs(bad)


if __name__ == '__main__': unittest.main()
