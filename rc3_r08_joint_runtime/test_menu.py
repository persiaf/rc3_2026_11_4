"""Targeted pure-menu tests, zero real input or H/planning requests."""
import dataclasses,json,time,unittest
from rc3_r08_joint_decision_design_20261113.test_interfaces import fixture
from .menu import build_menu
from .common import HOME,put

class MenuTests(unittest.TestCase):
    def make(self):
        return fixture(),[dict(task_id=0,candidate_id=10,arm='L'),dict(task_id=1,candidate_id=20,arm='R')]
    def test_deterministic_order(self):
        s,a=self.make();self.assertEqual(build_menu(s,a)['menu_sha256'],build_menu(s,a)['menu_sha256'])
    def test_real_net_arity(self):
        s,a=self.make();m=build_menu(s,a);self.assertTrue(m['actions']);self.assertTrue(any(x['net_joint'] for x in m['actions']))
        for x in m['actions']:
            self.assertFalse(set(x['drop'])&set(x['add']));self.assertEqual(x['net_joint'],len(x['drop'])==2 or len(x['add'])==2)
    def test_caps_and_missing_not_filled(self):
        s,a=self.make();m=build_menu(s,a);self.assertLessEqual(len(m['actions']),2);self.assertEqual(m['missing_slots'],16-len(m['actions']))
    def test_menu_independent_of_recorded_cost(self):
        s,a=self.make();t=dataclasses.replace(s,incumbent=dataclasses.replace(s.incumbent,J=5.5,L=4.))
        self.assertEqual(build_menu(s,a)['menu_sha256'],build_menu(t,a)['menu_sha256'])
    def test_budget_masks_all_edits(self):
        s,a=self.make();s=dataclasses.replace(s,requests_left=0);m=build_menu(s,a);self.assertEqual(m['actions'],[]);self.assertTrue(m['stop_legal'])

def main():
    tick=time.perf_counter();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MenuTests))
    put(HOME/'MENU_TEST_RESULTS.json',dict(status='PASS' if r.wasSuccessful() else 'FAIL',tests=r.testsRun,wall_s=time.perf_counter()-tick,real_requests=0,new_H_native_navigation_teacher_scene_forward_optimizer=0))
    if not r.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
