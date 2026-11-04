import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from rc3_v3.environment import FEATURE_NAMES
from rc3_v3.model import make_policy, batch_states, save_policy, export_numpy, load_policy
from rc3_v3.train import loss_for


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(10)
        self.rng = np.random.default_rng(10)

    def test_padding_and_numpy_export_both_architectures(self):
        names = FEATURE_NAMES
        a = self.rng.normal(size=(7,len(names))).astype(np.float32)
        mask = np.asarray([True, False, True, True, True, False, True])
        b = self.rng.normal(size=(12,len(names))).astype(np.float32)
        for architecture in ('attention', 'mlp'):
            model = make_policy(np.zeros(len(names)),np.ones(len(names)),architecture).eval()
            single = model.outputs(a,mask)
            x,m = batch_states([(a,mask),(b,np.ones(len(b),bool))])
            with torch.no_grad():
                mixed = model(x,m)[0][0].numpy()
            np.testing.assert_allclose(single[0][mask.tolist()+[True]],np.r_[mixed[:7][mask],mixed[-1]],rtol=1e-5,atol=2e-6)
            with tempfile.TemporaryDirectory() as tmp:
                pt,npz=Path(tmp)/'model.pt',Path(tmp)/'model.npz'
                save_policy(pt,model);export_numpy(pt,npz)
                portable=load_policy(npz)
                for before,after in zip(single,portable.outputs(a,mask)):
                    np.testing.assert_allclose(before,after,rtol=3e-5,atol=3e-6)
                self.assertEqual(model.select(a,mask),portable.select(a,mask))

    def test_zero_gain_states_are_learnable_and_stop_is_legal(self):
        model=make_policy(np.zeros(len(FEATURE_NAMES)),np.ones(len(FEATURE_NAMES)),'mlp')
        row=dict(features=np.zeros((3,len(FEATURE_NAMES)),np.float32),mask=np.array([True,False,True]),gains=np.zeros(3,np.float32))
        loss,_=loss_for(model,[row],1.,1.,'cpu')
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(model.stop[-1].bias.grad)
        with torch.no_grad():
            model.stop[-1].bias.fill_(100.)
        self.assertIsNone(model.select(row['features'],row['mask']))

    def test_wrong_phase_and_schema_rejected(self):
        model=make_policy(np.zeros(len(FEATURE_NAMES)),np.ones(len(FEATURE_NAMES)),'mlp')
        with tempfile.TemporaryDirectory() as tmp:
            pt=Path(tmp)/'bad.pt'
            save_policy(pt,model)
            saved=torch.load(pt,weights_only=True)
            saved['phase']='ppo';torch.save(saved,pt)
            with self.assertRaisesRegex(ValueError,'phase'):
                load_policy(pt)


if __name__=='__main__':
    unittest.main()
