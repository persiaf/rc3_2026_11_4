import unittest
from types import SimpleNamespace
import numpy as np
from rc3_v3.environment import Env
from .features import local_graphs,collate,observation


def counterexample(branch):
    a=np.arange(4,dtype=float);d=np.zeros((5,5));d[:4,:4]=abs(a[:,None]-a[None,:])
    d[4,:4]=d[:4,4]=abs(a-2)+branch
    f=np.zeros((3,4,2),bool)
    for i in range(3):f[i,i,0]=True
    f[[0,1],3,0]=True
    return SimpleNamespace(F=f,penalty=.75,nav={'costs':d},scene=dict(start=[0.,0.],
        x_lines=[0.,12.],y_lines=[0.,1.],tasks=[dict(x=float(i+1),y=0.,z=.7) for i in range(3)],
        candidates=[dict(x=x,y=y,theta=0.,h=.4) for x,y in [(1.,0.),(2.,0.),(3.,0.),(2.,.1)]]))


class FeatureTests(unittest.TestCase):
    def test_known_aliasing_separated_by_distance(self):
        envs=[Env(counterexample(v),8,[0,1,2,3,0]) for v in (.1,10.)]
        rows=[observation(e) for e in envs]
        np.testing.assert_array_equal(rows[0]['features'],rows[1]['features'])
        self.assertFalse(np.array_equal(rows[0]['graphs'][0]['dist'],rows[1]['graphs'][0]['dist']))
        np.testing.assert_array_equal(rows[0]['graphs'][0]['reach'],rows[1]['graphs'][0]['reach'])

    def test_graphs_have_no_oracle_dependency(self):
        from unittest.mock import patch
        with patch('rc3_improve.repair.repair_window',side_effect=RuntimeError('oracle forbidden')):
            e=Env(counterexample(.1),8,[0,1,2,3,0]);row=observation(e)
        b=collate([row]);self.assertEqual(b['reach'].shape[-1],2)
        self.assertTrue(b['mask'].all());self.assertTrue(np.isfinite(b['dist']).all())

    def test_padding_does_not_add_legal_actions(self):
        row=observation(Env(counterexample(.1),8,[0,1,2,3,0]))
        short=dict(features=row['features'][:1],mask=row['mask'][:1],graphs=row['graphs'][:1])
        b=collate([row,short]);self.assertEqual(b['mask'][1].tolist(),[True,False,False])


if __name__=='__main__':unittest.main()
