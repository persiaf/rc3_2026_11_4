"""Synthetic in-memory contract tests; zero real planning requests or weights."""
import dataclasses,json,sys,time,unittest
import numpy as np
from .contracts import ProblemSnapshot,RecordedRoute,JointAction,JointState,ResumeIdentity,action_legality
from .adapters import HOME

def fixture():
    F=np.zeros((2,3,2),bool);F[0,0,0]=F[1,1,1]=F[:,2,0]=True
    p=ProblemSnapshot('toy_problem','toy_parent','toy_physical',(0,1),(10,20,30),F,np.ones_like(F),np.ones((4,4)),0,0,
        ((1.,0.,0.,.4),(1.,0.,3.14,.4),(2.,0.,0.,.4)))
    return JointState(p,RecordedRoute('toy_problem',(10,20),3.5,2.,2,'synthetic_fixture'),4,6,'toy_weights')

class ContractTests(unittest.TestCase):
    def test_joint_cover_replacement(self):
        s=fixture();self.assertFalse(action_legality(s,JointAction(drop=(10,)))[0])
        self.assertTrue(action_legality(s,JointAction(drop=(10,20),add=(30,)))[0])
    def test_same_arm_conjunction(self):
        s=fixture();A=np.zeros_like(s.snapshot.per_arm_allow);A[:,:,1]=True
        p=dataclasses.replace(s.snapshot,per_arm_allow=A)
        self.assertFalse(p.cover[0,0]);self.assertTrue(p.cover[1,1])
    def test_configuration_not_coordinate_count(self):
        s=fixture();self.assertEqual(s.incumbent.K,2);self.assertEqual(len({x[:2] for x in s.snapshot.poses[:2]}),1)
    def test_frozen_snapshot(self):
        with self.assertRaises(ValueError):fixture().snapshot.D[0,0]=2
    def test_foreign_identity(self):
        with self.assertRaises(ValueError):dataclasses.replace(fixture(),incumbent=RecordedRoute('foreign',(10,),1.75,1.,1,'toy'))
    def test_action_ids_and_no_op(self):
        s=fixture();self.assertFalse(action_legality(s,JointAction(drop=(999,)))[0])
        self.assertFalse(action_legality(s,JointAction(add=(10,)))[0])
        with self.assertRaises(ValueError):JointAction(drop=(10,10))
        with self.assertRaises(ValueError):JointAction(stop=True,add=(30,))
    def test_budget_and_stop(self):
        s=dataclasses.replace(fixture(),steps_left=0)
        self.assertFalse(action_legality(s,JointAction(drop=(10,20),add=(30,)))[0]);self.assertTrue(action_legality(s,JointAction(stop=True))[0])
    def test_asymmetric_endpoint_storage(self):
        s=fixture();D=s.snapshot.D.copy();D[0,1]=7.;D[1,0]=3.
        p=dataclasses.replace(s.snapshot,D=D,entry=4,exit=9)
        self.assertEqual(p.D[0,1],7.);self.assertEqual(p.D[1,0],3.)
    def test_unreachable_addition(self):
        s=fixture();D=s.snapshot.D.copy();D[3,0]=np.inf;s=dataclasses.replace(s,snapshot=dataclasses.replace(s.snapshot,D=D))
        self.assertFalse(action_legality(s,JointAction(drop=(10,20),add=(30,)))[0])
    def test_resume_metadata(self):
        q=ResumeIdentity('s','src','menu',4,10,4,'w','adam','rng');q.validate('s','src','menu',11,4)
        with self.assertRaises(ValueError):q.validate('s','src','menu',9,4)
        with self.assertRaises(ValueError):q.validate('s','src','menu',11,5)
        with self.assertRaises(ValueError):q.validate('foreign','src','menu',11,4)
    def test_state_version_changes_fingerprint(self):
        s=fixture();self.assertNotEqual(s.fingerprint(),dataclasses.replace(s,weight_version='next').fingerprint())
    def test_no_weight_or_planner_import(self):
        self.assertFalse(any(m.startswith(('torch','scipy','rc3_global_local.','rc3_fast.','rc3_v4.')) for m in sys.modules))

def main():
    tick=time.perf_counter();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    (HOME/'INTERFACE_TEST_RESULTS.json').write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),wall_s=time.perf_counter()-tick,
        synthetic_in_memory_only=True,new_forward_optimizer_scene_teacher_navigation_native_H_complete_requests=0,
        not_demonstrated='No real solver/MDP transition, network learning or numerical Adam/RNG restore; R08-C remains NOT_RUN'),indent=2)+'\n',encoding='utf-8')
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
