import unittest
from contextlib import contextmanager
from unittest.mock import patch
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare
from rc3_v3.runtime import run as old_run
from .runtime import run


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.problem=prepare(make_complex_scene(3,2,730918,'blocked'))

    def test_fixed_methods_equal_frozen_engine(self):
        for method in ('fixed3','detour','random'):
            new,ni=run(self.problem,method,horizon=8);old,oi=old_run(self.problem,method,horizon=8)
            self.assertEqual(new,old);self.assertEqual(ni['dp_calls'],oi['dp_calls'])

    def test_session_once_and_bad_policy_returns_anchor(self):
        class Bad:
            count=0
            @contextmanager
            def inference_session(self):
                self.count+=1;yield
            def select_env(self,*args):raise ValueError('deliberately bad proposal')
        policy=Bad();plan,info=run(self.problem,'learned',policy)
        self.assertEqual(policy.count,1);self.assertTrue(info['fallback_used'])
        self.assertEqual(info['objective_m'],info['reference_objective_m'])
        self.assertTrue(plan['validation']['valid_model_plan'])

    def test_zero_deadline_no_policy_call(self):
        class Bad:
            def select_env(self,*args):raise AssertionError('must not evaluate')
        plan,info=run(self.problem,'learned',Bad(),time_limit_s=0.)
        self.assertEqual(info['dp_calls'],0);self.assertEqual(info['stop_reason'],'deadline')
        self.assertTrue(info['deadline_exceeded']);self.assertEqual(info['time_limit_s'],0.)

    def test_cli_does_not_discard_external_certificates(self):
        import json,sys,tempfile
        from pathlib import Path
        from .plan import main
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bundle.json';p.write_text(json.dumps({'schema':'rc3-external-reachability-v1','scene':self.problem.scene}),encoding='utf-8')
            with patch.object(sys,'argv',['plan','--scene',str(p),'--output',str(Path(folder)/'out.json')]):
                with self.assertRaises(SystemExit) as caught:main()
            self.assertEqual(caught.exception.code,2)


if __name__=='__main__':unittest.main()
