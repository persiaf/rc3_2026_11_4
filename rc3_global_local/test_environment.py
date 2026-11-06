"""Independent synthetic MDP contracts: no real scenes/navigation/H/model calls.

F/A arrays and metric-closure costs are hand-specified. Expected task mappings,
costs and rewards are calculated here without the production metrics routine.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rc3_global_local.problem import Parent, LocalProblem, extract
from rc3_global_local.environment import Environment, rule_action


def parent_from(F, A=None, positions=None, nx=3, ny=2, task_tables=None, candidate_tables=None):
    f=np.asarray(F,dtype=bool)
    n,m,_=f.shape
    if A is None:A=np.ones_like(f)
    if positions is None:positions=list(range(m+1))
    coordinates=np.asarray(positions,dtype=float)
    D=np.abs(coordinates[:,None]-coordinates[None,:])
    tables=[dict(id=y*nx+x,ix=x,iy=y,cx=float(x),cy=float(y)) for y in range(ny) for x in range(nx)]
    if task_tables is None:task_tables=[i%(nx*ny) for i in range(n)]
    if candidate_tables is None:candidate_tables=[i%(nx*ny) for i in range(m)]
    scene=dict(nx=nx,ny=ny,start=[float(coordinates[0]),0.],tables=tables,
        tasks=[dict(id=i,table_id=int(task_tables[i]),x=float(i),y=0.,z=.75) for i in range(n)],
        candidates=[dict(id=j,table_id=int(candidate_tables[j]),x=float(coordinates[j+1]),y=0.,theta=0.,h=.4) for j in range(m)],
        obstacles=[dict(xmin=50.,xmax=51.,ymin=50.,ymax=51.)])
    return Parent(SimpleNamespace(scene=scene,nav=dict(costs=D)),f,A)


def two_tasks():
    # Candidate3 covers both tasks, candidate1 only task0, candidate2 only task1.
    f=np.zeros((2,3,2),bool)
    f[0,0,0]=True;f[1,1,1]=True;f[:,2,:]=True
    return parent_from(f,positions=[-3.,0.,4.,10.])


class IdentityTerminal:
    def __init__(self):self.calls=[]
    def __call__(self,p,route):
        self.calls.append(list(route))
        length=sum(float(p.D[a,b]) for a,b in zip(route,route[1:]))
        stops=len(route)-2
        return dict(J=length+.75*stops,L=length,K=stops,route=list(route))


class EnvironmentContracts(unittest.TestCase):
    def test_same_arm_intersection_required(self):
        f=np.array([[[True,False]]]);a=np.array([[[False,True]]])
        self.assertTrue(f.any());self.assertTrue(a.any())
        parent=parent_from(f,a)
        self.assertFalse(parent.effective.any())
        with self.assertRaises(ValueError):LocalProblem(parent,[0],[0])

    def test_same_arm_allow_restriction_is_retained(self):
        f=np.array([[[True,False],[False,True]]])
        a=np.array([[[False,True],[True,True]]])
        view=LocalProblem(parent_from(f,a),[0],[0])
        self.assertEqual(view.candidate_ids.tolist(),[1])
        self.assertEqual(view.F.tolist(),[[[False,True]]])
        self.assertEqual(view.C.tolist(),[[True]])

    def test_boundary_candidate_not_clipped_by_table_or_position(self):
        f=np.zeros((2,3,2),bool)
        f[0,0,0]=True;f[0,2,1]=True;f[1,1,0]=True
        parent=parent_from(f,task_tables=[0,1],candidate_tables=[0,1,5],positions=[0.,1.,2.,100.])
        view=extract(parent,0,0,1,1)
        self.assertEqual(view.task_ids.tolist(),[0])
        self.assertEqual(view.candidate_ids.tolist(),[0,2])
        self.assertEqual(view.boundary_candidate_ids,[2])
        self.assertEqual(view.F.tolist(),[[[True,False],[False,True]]])
        self.assertIs(view.parent.base.nav,parent.base.nav)
        self.assertEqual(parent.scene['obstacles'],[dict(xmin=50.,xmax=51.,ymin=50.,ymax=51.)])

    def test_extract_grid_keeps_original_task_and_candidate_ids(self):
        f=np.zeros((6,6,2),bool)
        for i in range(6):f[i,i,i%2]=True
        parent=parent_from(f)
        view=extract(parent,1,0,1,2)
        self.assertEqual(view.table_ids,(1,4))
        self.assertEqual(view.task_ids.tolist(),[1,4])
        self.assertEqual(view.candidate_ids.tolist(),[1,4])
        self.assertEqual(view.global_route([0,1,2,0]),[0,2,5,0])
        terminal=IdentityTerminal();env=Environment(view,terminal)
        first=env.step(1)
        self.assertEqual(first['served_tasks_global'],[1])
        self.assertEqual(first['remaining_tasks_global'],[4])
        second=env.step(2)
        self.assertEqual(second['served_tasks_global'],[4])
        self.assertTrue(env.done)

    def test_open_endpoint_matrix_and_service_submatrix(self):
        parent=two_tasks();view=LocalProblem(parent,[0,1],[0,1],1,3)
        np.testing.assert_array_equal(view.D[0],[10.,0.,4.,10.])
        np.testing.assert_array_equal(view.D[:,0],[10.,10.,6.,0.])
        np.testing.assert_array_equal(view.D[1:,1:],parent.D[1:,1:])
        self.assertNotEqual(view.D[0,1],view.D[1,0])
        self.assertEqual(view.global_route([0,1,2,0]),[1,1,2,3])
        self.assertEqual(view.metrics([0,1,2,0]),dict(J=11.5,L=10.,K=2))

    def test_closed_endpoint_costs(self):
        view=LocalProblem(two_tasks(),[0,1],[0,1],2,2)
        self.assertEqual(view.global_route([0,1,3,0]),[2,1,3,2])
        self.assertEqual(view.metrics([0,1,3,0]),dict(J=21.5,L=20.,K=2))

    def test_endpoint_may_coincide_with_service_without_repeated_service(self):
        view=LocalProblem(two_tasks(),[0,1],[0,1],1,3)
        self.assertEqual(view.global_route([0,3,0]),[1,3,3])
        terminal=IdentityTerminal();env=Environment(view,terminal);env.step(3)
        self.assertEqual(env.result['J'],10.75)
        self.assertEqual(env.result['K'],1)
        self.assertEqual(sum(env.rewards),-10.75)

    def test_global_mapping_rejects_invalid_internal_nodes(self):
        view=LocalProblem(two_tasks(),[0,1],[0,1])
        for route in ([0,0,0],[0,-1,0],[0,4,0],[0,True,0],[0,1.,0]):
            with self.subTest(route=route):
                with self.assertRaises(ValueError):view.global_route(route)

    def test_global_mapping_rejects_repeated_service_but_not_endpoint_overlap(self):
        view=LocalProblem(two_tasks(),[0,1],[0,1],1,3)
        with self.assertRaises(ValueError):view.global_route([0,1,1,2,0])
        self.assertEqual(view.global_route([0,1,3,0]),[1,1,3,3])

    def test_mask_positive_gain_and_no_repeat(self):
        env=Environment(LocalProblem(two_tasks(),[0,1],[0,1]),IdentityTerminal())
        self.assertEqual(env.mask().tolist(),[True,True,True])
        env.step(1)
        self.assertEqual(env.mask().tolist(),[False,True,True])
        with self.assertRaises(ValueError):env.step(1)
        env.step(2)
        self.assertEqual(env.mask().tolist(),[False,False,False])

    def test_mask_checks_current_and_exit_distance(self):
        view=LocalProblem(two_tasks(),[0,1],[0,1])
        # Deliberately perturb the local synthetic interface to test the mask.
        # No real navigation matrix or parent matrix is changed.
        view.D[3,0]=np.inf
        env=Environment(view,IdentityTerminal())
        self.assertEqual(env.mask().tolist(),[True,True,False])
        view.D[1,3]=np.inf;env.step(1)
        self.assertEqual(env.mask().tolist(),[False,True,False])

    def test_disconnected_candidates_filtered_before_local_decision(self):
        parent=two_tasks();parent.base.nav['costs'][3,:]=np.inf;parent.base.nav['costs'][:,3]=np.inf
        parent.base.nav['costs'][3,3]=0.
        view=LocalProblem(parent,[0,1],[0,1])
        self.assertEqual(view.candidate_ids.tolist(),[0,1])
        self.assertTrue(view.C.any(1).all())

    def test_disconnected_entry_exit_rejected(self):
        parent=two_tasks();parent.base.nav['costs'][1,3]=np.inf;parent.base.nav['costs'][3,1]=np.inf
        with self.assertRaises(ValueError):LocalProblem(parent,[0,1],[0,1],1,3)

    def test_uncoverable_selected_task_fails_closed(self):
        f=np.array([[[True,False]],[[False,False]]])
        with self.assertRaises(ValueError):LocalProblem(parent_from(f),[0,1],[0,1])

    def test_identity_terminal_reward_is_complete_open_route_cost(self):
        view=LocalProblem(two_tasks(),[0,1],[0,1],1,3)
        terminal=IdentityTerminal();env=Environment(view,terminal)
        first=env.step(1);second=env.step(2)
        self.assertEqual(first['reward'],-.75)
        self.assertEqual(second['reward'],-10.75)
        self.assertEqual(second['terminal_H_correction'],0.)
        self.assertEqual(sum(env.rewards),-11.5)
        self.assertEqual(terminal.calls,[[0,1,2,0]])

    def test_terminal_H_correction_sign_and_telescoping_identity(self):
        f=np.zeros((2,3,2),bool);f[0,0,0]=True;f[1,1,1]=True;f[:,2,0]=True
        parent=parent_from(f,positions=[0.,10.,15.,1.])
        view=LocalProblem(parent,[0,1],[0,1])
        calls=[]
        def synthetic_terminal(p,raw):
            calls.append(list(raw))
            # A hand-known one-stop complete route replaces the two-stop raw.
            return dict(J=2.75,L=2.,K=1,route=[0,3,0])
        env=Environment(view,synthetic_terminal)
        env.step(1);last=env.step(2)
        # raw 0->10->15->0 =30m; two stops give Jraw31.5.
        self.assertEqual(last['terminal_H_correction'],28.75)
        self.assertEqual(sum(env.rewards),-2.75)
        self.assertEqual(calls,[[0,1,2,0]])
        self.assertGreater(last['reward'],0.)

    def test_empty_task_problem_moves_directly_to_exit(self):
        parent=parent_from(np.zeros((0,2,2),bool),positions=[0.,2.,9.])
        view=LocalProblem(parent,[],[],1,2)
        terminal=IdentityTerminal();env=Environment(view,terminal)
        self.assertTrue(env.done);self.assertEqual(env.mask().tolist(),[])
        self.assertEqual(view.candidate_ids.tolist(),[])
        self.assertEqual(view.global_route([0,0]),[1,2])
        self.assertEqual(env.result,dict(J=7.,L=7.,K=0,route=[0,0]))
        self.assertEqual(env.rewards,[-7.]);self.assertEqual(terminal.calls,[[0,0]])
        self.assertEqual(env.replay([])['J'],7.)
        with self.assertRaises(ValueError):env.step(1)

    def test_empty_closed_problem_zero_cost(self):
        parent=parent_from(np.zeros((0,2,2),bool))
        env=Environment(LocalProblem(parent,[],[]),IdentityTerminal())
        self.assertTrue(env.done);self.assertEqual(env.result['J'],0.)
        self.assertEqual(env.result['K'],0)

    def test_action_type_and_range_rejected_without_state_mutation(self):
        env=Environment(LocalProblem(two_tasks(),[0,1],[0,1]),IdentityTerminal())
        for value in (True,np.bool_(False),0,-1,4,1.,'1',None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):env.step(value)
                self.assertEqual(env.prefix,[])
                self.assertEqual(env.remaining.tolist(),[True,True])
        env.step(np.int64(3));self.assertTrue(env.done)

    def test_terminal_state_rejects_all_further_actions(self):
        terminal=IdentityTerminal();env=Environment(LocalProblem(two_tasks(),[0,1],[0,1]),terminal)
        env.step(3)
        for node in (1,2,3):
            with self.assertRaises(ValueError):env.step(node)
        self.assertEqual(len(terminal.calls),1)

    def test_replay_is_exact_and_fresh_only(self):
        view=LocalProblem(two_tasks(),[0,1],[0,1],1,3)
        a=Environment(view,IdentityTerminal());a.step(1);a.step(2)
        b=Environment(view,IdentityTerminal());b.replay([1,2])
        self.assertEqual(a.prefix,b.prefix);self.assertEqual(a.trace,b.trace)
        self.assertEqual(a.rewards,b.rewards);self.assertEqual(a.result,b.result)
        with self.assertRaises(ValueError):b.replay([1,2])

    def test_incomplete_replay_fails(self):
        env=Environment(LocalProblem(two_tasks(),[0,1],[0,1]),IdentityTerminal())
        with self.assertRaises(ValueError):env.replay([1])

    def test_observation_arrays_and_prefix_are_copies(self):
        env=Environment(LocalProblem(two_tasks(),[0,1],[0,1]),IdentityTerminal())
        env.step(1);observation=env.observe()
        observation['remaining'][:]=False;observation['action_mask'][:]=False
        observation['ordered_prefix_local'].append(999)
        observation['ordered_prefix_global'].append(999)
        self.assertEqual(env.remaining.tolist(),[False,True]);self.assertEqual(env.prefix,[1])
        self.assertEqual(env.mask().tolist(),[False,True,True])

    def test_parent_scene_F_A_D_unchanged_by_extraction_and_rollout(self):
        parent=two_tasks();scene=copy.deepcopy(parent.scene)
        F=parent.F.copy();A=parent.A.copy();D=parent.D.copy()
        view=extract(parent,0,0,3,2,1,3)
        env=Environment(view,IdentityTerminal());env.replay([1,2])
        self.assertEqual(parent.scene,scene)
        np.testing.assert_array_equal(parent.F,F);np.testing.assert_array_equal(parent.A,A)
        np.testing.assert_array_equal(parent.D,D)

    def test_parent_constraint_input_copied_and_stored_readonly(self):
        rawF=np.ones((1,2,2),bool);rawA=np.ones_like(rawF)
        parent=parent_from(rawF,rawA)
        rawF[:]=False;rawA[:]=False
        self.assertTrue(parent.F.all());self.assertTrue(parent.A.all())
        with self.assertRaises(ValueError):parent.F[0,0,0]=False
        with self.assertRaises(ValueError):parent.A[0,0,0]=False

    def test_window_validation(self):
        parent=two_tasks()
        for bounds in ((-1,0,1,1),(0,0,0,1),(0,0,4,1),(0,2,1,1),(True,0,1,1),(0.,0,1,1)):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):extract(parent,*bounds)

    def test_target_and_endpoint_validation(self):
        parent=two_tasks()
        for tasks in ([0,0],[-1],[2],[True]):
            with self.subTest(tasks=tasks):
                with self.assertRaises(ValueError):LocalProblem(parent,tasks,[0])
        for entry,exit in ((-1,0),(0,4),(True,0),(0,1.)):
            with self.assertRaises(ValueError):LocalProblem(parent,[0,1],[0,1],entry,exit)

    def test_parent_rejects_asymmetric_navigation(self):
        parent=two_tasks();base=SimpleNamespace(scene=copy.deepcopy(parent.scene),nav=dict(costs=parent.D.copy()))
        base.nav['costs'][0,1]+=1.
        with self.assertRaises(ValueError):Parent(base,parent.F,parent.A)

    def test_parent_rejects_noncontiguous_global_ids(self):
        parent=two_tasks()
        for kind in ('tasks','candidates','tables'):
            with self.subTest(kind=kind):
                scene=copy.deepcopy(parent.scene);scene[kind][0]['id']=999
                base=SimpleNamespace(scene=scene,nav=dict(costs=parent.D.copy()))
                with self.assertRaises(ValueError):Parent(base,parent.F,parent.A)

    def test_parent_global_ids_reject_numeric_equal_non_integer_types(self):
        parent=two_tasks()
        for kind in ('tasks','candidates','tables'):
            for bad in (False,0.,np.int64(0)):
                with self.subTest(kind=kind,bad=bad):
                    scene=copy.deepcopy(parent.scene);scene[kind][0]['id']=bad
                    base=SimpleNamespace(scene=scene,nav=dict(costs=parent.D.copy()))
                    with patch('rc3_global_local.problem.prepare') as prepare_mock:
                        with self.assertRaises(ValueError):Parent(base,parent.F,parent.A)
                        prepare_mock.assert_not_called()

    def test_parent_rejects_D_shape_without_preparing_navigation(self):
        parent=two_tasks()
        for shape in ((3,3),(4,3),(5,5),(4,)):
            with self.subTest(shape=shape):
                base=SimpleNamespace(scene=copy.deepcopy(parent.scene),nav=dict(costs=np.zeros(shape)))
                with patch('rc3_global_local.problem.prepare') as prepare_mock:
                    with self.assertRaisesRegex(ValueError,'Global D'):
                        Parent(base,parent.F,parent.A)
                    prepare_mock.assert_not_called()

    def test_proxy_coupled_task_rejected_before_prepare(self):
        for service_mode in ('coupled_dual_arm','simultaneous_dual_arm','unknown_service'):
            with self.subTest(service_mode=service_mode):
                scene=copy.deepcopy(two_tasks().scene)
                scene['robot']=dict(fixed_lift_height=.4)
                scene['tasks'][0]['service_mode']=service_mode
                with patch('rc3_global_local.problem.prepare',side_effect=AssertionError('Real prepare must not run')) as prepare_mock:
                    with self.assertRaisesRegex(ValueError,'Coupled'):
                        Parent.from_scene(scene)
                    prepare_mock.assert_not_called()

    def test_proxy_variable_lift_rejected_before_prepare(self):
        scene=copy.deepcopy(two_tasks().scene);scene['robot']=dict(fixed_lift_height=.4)
        scene['candidates'][0]['h']=.5
        with patch('rc3_global_local.problem.prepare',side_effect=AssertionError('Real prepare must not run')) as prepare_mock:
            with self.assertRaisesRegex(ValueError,'Variable lift'):
                Parent.from_scene(scene)
            prepare_mock.assert_not_called()

    def test_proxy_nonfinite_lift_rejected_before_prepare(self):
        for height in (float('nan'),float('inf'),-float('inf')):
            with self.subTest(height=height):
                scene=copy.deepcopy(two_tasks().scene);scene['robot']=dict(fixed_lift_height=.4)
                scene['candidates'][0]['h']=height
                with patch('rc3_global_local.problem.prepare',side_effect=AssertionError('Real prepare must not run')) as prepare_mock:
                    with self.assertRaisesRegex(ValueError,'Variable lift'):
                        Parent.from_scene(scene)
                    prepare_mock.assert_not_called()

    def test_supported_proxy_input_reaches_only_mocked_prepare(self):
        parent=two_tasks();scene=copy.deepcopy(parent.scene)
        scene['robot']=dict(fixed_lift_height=.4)
        scene['tasks'][0]['service_mode']='independent_single_arm'
        base=SimpleNamespace(scene=scene,nav=dict(costs=parent.D.copy()),F=parent.F.copy())
        with patch('rc3_global_local.problem.prepare',return_value=base) as prepare_mock:
            result=Parent.from_scene(scene,nav=base.nav)
            prepare_mock.assert_called_once()
        self.assertEqual(result.mode,'proxy')
        np.testing.assert_array_equal(result.F,parent.F)
        self.assertEqual(scene['candidates'][0]['h'],.4)

    def test_variable_lift_dispatches_to_supplied_external_tables(self):
        # This is a mocked adapter dispatch test, not external attestation or IK.
        parent=two_tasks();scene=copy.deepcopy(parent.scene)
        scene['robot']=dict(fixed_lift_height=.4);scene['candidates'][0]['h']=.7
        bundle=dict(scene=scene,synthetic_unit_stub=True)
        base=SimpleNamespace(scene=scene,nav=dict(costs=parent.D.copy()))
        with patch('rc3_v5.external.check_bundle',return_value=(parent.F,parent.A)) as check_mock, \
             patch('rc3_v5.external.prepare_external',return_value=base) as external_mock, \
             patch('rc3_global_local.problem.prepare',side_effect=AssertionError('Proxy prepare must not run')) as proxy_mock:
            result=Parent.from_external(bundle)
            check_mock.assert_called_once_with(bundle);external_mock.assert_called_once_with(bundle)
            proxy_mock.assert_not_called()
        self.assertEqual(result.mode,'external_tables')
        self.assertEqual(result.scene['candidates'][0]['h'],.7)
        np.testing.assert_array_equal(result.F,parent.F)

    def test_rule_uses_positive_coverage_and_exact_distance(self):
        env=Environment(LocalProblem(two_tasks(),[0,1],[0,1]),IdentityTerminal())
        # Ratios: 1/3.75, 1/7.75, 2/13.75 => candidate1 wins.
        self.assertEqual(rule_action(env),1)
        env.step(1)
        # Remaining task: candidate2 distance4 vs candidate3 distance10.
        self.assertEqual(rule_action(env),2)
        env.step(2)
        with self.assertRaises(ValueError):rule_action(env)

    def test_every_synthetic_legal_action_sequence_completes_within_N(self):
        fixtures=[]
        fixtures.append(two_tasks())
        f=np.zeros((3,4,2),bool)
        f[0,[0,3],0]=True;f[1,[1,3],1]=True;f[2,[2,3],0]=True
        fixtures.append(parent_from(f))
        completed=0
        for parent in fixtures:
            view=LocalProblem(parent,list(range(len(parent.F))),list(range(len(parent.F))),0,len(parent.D)-1)
            stack=[[]]
            while stack:
                prefix=stack.pop();env=Environment(view,IdentityTerminal())
                for node in prefix:env.step(node)
                self.assertLessEqual(len(env.prefix),len(view.C))
                if env.done:
                    completed+=1;self.assertAlmostEqual(sum(env.rewards),-env.result['J'])
                    self.assertTrue(np.all(~env.remaining));continue
                legal=(np.flatnonzero(env.mask())+1).tolist()
                self.assertTrue(legal)
                for node in legal:
                    # Independent remaining-task test, without using C or mask.
                    global_candidate=int(view.candidate_ids[node-1])
                    covered=set()
                    for old in prefix:
                        gc=int(view.candidate_ids[old-1])
                        covered.update(t for t in view.task_ids if bool((parent.F[t,gc]&parent.A[t,gc]).any()))
                    gain=[t for t in view.task_ids if t not in covered and bool((parent.F[t,global_candidate]&parent.A[t,global_candidate]).any())]
                    self.assertTrue(gain)
                    stack.append(prefix+[node])
        self.assertGreater(completed,10)


if __name__=='__main__':unittest.main(verbosity=2)
