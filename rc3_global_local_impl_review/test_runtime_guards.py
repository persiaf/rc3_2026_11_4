"""Synthetic guard regression. No navigation, H, model computation or optimizer."""
import copy
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np

from rc3_baseline.baseline import ROBOT
from rc3_fast.planner import map_key
from rc3_global_local.problem import Parent, LocalProblem
from rc3_global_local.policy import GlobalLocalPolicy
from rc3_global_local.terminal import TerminalCache
from .runtime_guards import (FrozenDict, bind_navigation, current_navigation_sources,
                             extract_snapshot, file_hash, load_saved_s0_parent,
                             snapshot_local, snapshot_parent)

ROOT = Path(__file__).resolve().parents[1]
COUNTS = dict(score_entry_guard_rejections=0, synthetic_terminal_mock_calls=0,
              synthetic_terminal_cache_hits=0)


def fixture():
    tables = [dict(id=i, ix=i, iy=0, cx=float(i*3+1), cy=1., xmin=float(i*3),
                   xmax=float(i*3+2), ymin=0., ymax=2., z=.75) for i in range(2)]
    scene = dict(nx=2, ny=1, start=[0., -1.], tables=tables, obstacles=[],
                 x_lines=[0., 3., 6.], y_lines=[-1., 3.], robot=copy.deepcopy(ROBOT),
                 tasks=[dict(id=i, table_id=i, x=float(i*3+1), y=1., z=.75) for i in range(2)],
                 candidates=[dict(id=j, table_id=j//2, x=float(j+1), y=-1.,
                                  theta=0., h=.4, corridor_anchor=[float(j+1), -1.]) for j in range(4)])
    x=np.arange(5, dtype=float)
    d=np.abs(x[:,None]-x[None,:])
    f=np.zeros((2,4,2),dtype=bool)
    f[0,0,0]=True;f[0,1,1]=True;f[1,2,0]=True;f[1,3,1]=True
    nav=dict(costs=d,coordinates=[[float(i),-1.] for i in range(5)],
             service_nodes=list(range(5)),predecessors=np.zeros((5,5),dtype=np.int32),
             node_count=5,edge_count=4)
    return scene,f,np.ones_like(f),nav


def binding(scene,nav):
    return bind_navigation(scene,nav,source_map_key=map_key(scene),
        source_sha256=current_navigation_sources(),provenance=dict(producer='synthetic_guard_fixture'))


def parent_for(scene,f,a,nav):
    return snapshot_parent(scene,f,a,navigation=binding(scene,nav))


class RuntimeGuardTests(unittest.TestCase):
    def setUp(self):
        self.scene,self.f,self.a,self.nav=fixture()

    def test_bare_navigation_rejected(self):
        with self.assertRaisesRegex(ValueError,'Bare navigation'):
            snapshot_parent(self.scene,self.f,self.a,navigation=self.nav)

    def test_wrong_geometry_producer_key_rejected(self):
        with self.assertRaisesRegex(ValueError,'geometry key'):
            bind_navigation(self.scene,self.nav,source_map_key='wrong',
                source_sha256=current_navigation_sources(),provenance=dict(producer='fixture'))

    def test_wrong_source_version_rejected(self):
        sources=current_navigation_sources();sources['rc3_baseline/baseline.py']='0'*64
        with self.assertRaisesRegex(ValueError,'source version'):
            bind_navigation(self.scene,self.nav,source_map_key=map_key(self.scene),
                source_sha256=sources,provenance=dict(producer='fixture'))

    def test_binding_cannot_be_reused_for_other_scene(self):
        b=binding(self.scene,self.nav);changed=copy.deepcopy(self.scene)
        changed['candidates'][0]['x']+=.2
        with self.assertRaisesRegex(ValueError,'another geometry'):
            snapshot_parent(changed,self.f,self.a,navigation=b)

    def test_postseal_content_tamper_rejected(self):
        b=replace(binding(self.scene,self.nav),content_sha256='f'*64)
        with self.assertRaisesRegex(ValueError,'changed after'):
            snapshot_parent(self.scene,self.f,self.a,navigation=b)

    def test_snapshot_rechecks_bound_source_version(self):
        b=binding(self.scene,self.nav)
        wrong=dict(b.source_sha256);wrong['rc3_fast/planner.py']='1'*64
        with self.assertRaisesRegex(ValueError,'source version'):
            snapshot_parent(self.scene,self.f,self.a,navigation=replace(b,source_sha256=FrozenDict(wrong)))

    def test_D_change_changes_both_cache_namespaces(self):
        nav2=copy.deepcopy(self.nav);nav2['costs'][0,1]+=1;nav2['costs'][1,0]+=1
        p1=parent_for(self.scene,self.f,self.a,self.nav)
        p2=parent_for(self.scene,self.f,self.a,nav2)
        self.assertNotEqual(p1.identity,p2.identity)
        l1=extract_snapshot(p1,0,0,2,1);l2=extract_snapshot(p2,0,0,2,1)
        self.assertNotEqual(l1.identity,l2.identity)
        self.assertTrue(np.array_equal(l1.C,l2.C))
        self.assertEqual(l2.metrics([0,1,3,0])['J']-l1.metrics([0,1,3,0])['J'],1.)

    def test_numeric_problem_semantics_are_unchanged(self):
        old=Parent(SimpleNamespace(scene=copy.deepcopy(self.scene),nav=copy.deepcopy(self.nav)),self.f,self.a)
        new=parent_for(self.scene,self.f,self.a,self.nav)
        self.assertEqual(new.legacy_identity,old.identity)
        x=LocalProblem(old,[0,1],[0,1],0,4)
        y=snapshot_local(new,[0,1],[0,1],0,4)
        for name in ('C','F','D','candidate_ids','task_ids'):
            self.assertTrue(np.array_equal(getattr(x,name),getattr(y,name)))
        self.assertEqual(x.metrics([0,1,3,0]),y.metrics([0,1,3,0]))
        self.assertEqual(x.global_route([0,1,3,0]),y.global_route([0,1,3,0]))

    def test_external_arrays_and_scene_cannot_mutate_snapshot(self):
        p=parent_for(self.scene,self.f,self.a,self.nav)
        self.f[:]=False;self.a[:]=False;self.nav['costs'][:]=100
        self.scene['candidates'][0]['x']=100
        self.assertTrue(p.F[0,0,0]);self.assertTrue(p.A[0,0,0])
        self.assertEqual(p.D[0,1],1.)
        self.assertEqual(p.scene['candidates'][0]['x'],1.)

    def test_parent_arrays_and_nested_inputs_are_readonly(self):
        p=parent_for(self.scene,self.f,self.a,self.nav)
        for a in (p.F,p.A,p.effective,p.D,p.base.nav['predecessors']):
            with self.assertRaises(ValueError):a.flat[0]=0
            with self.assertRaises(ValueError):a.setflags(write=True)
        with self.assertRaises(TypeError):p.scene['candidates'][0]['x']=10
        with self.assertRaises(TypeError):p.scene['start'].append(10)
        with self.assertRaises(TypeError):p.base.nav['service_nodes'][0]=2
        with self.assertRaises(FrozenInstanceError):p.base.nav={}

    def test_local_arrays_and_window_are_readonly(self):
        p=parent_for(self.scene,self.f,self.a,self.nav)
        window=dict(x0=0,y0=0,width=2,height=1)
        v=snapshot_local(p,[0,1],[0,1],0,4,window)
        window['width']=1
        self.assertEqual(v.window['width'],2)
        for a in (v.F,v.C,v.D,v.task_ids,v.candidate_ids):
            with self.assertRaises(ValueError):a.flat[0]=0
            with self.assertRaises(ValueError):a.setflags(write=True)
        with self.assertRaises(TypeError):v.window['width']=1
        with self.assertRaises(FrozenInstanceError):v.D=np.ones_like(v.D)

    def test_task_changes_reuse_navigation_but_not_parent_cache(self):
        b=binding(self.scene,self.nav);s2=copy.deepcopy(self.scene)
        s2['tasks'][0]['x']+=.01
        p1=snapshot_parent(self.scene,self.f,self.a,navigation=b)
        p2=snapshot_parent(s2,self.f,self.a,navigation=b)
        self.assertNotEqual(p1.identity,p2.identity)
        self.assertEqual(p1.navigation_binding.content_sha256,p2.navigation_binding.content_sha256)

    def test_old_encoded_problem_guard_rejects_new_namespace_before_forward(self):
        old=Parent(SimpleNamespace(scene=copy.deepcopy(self.scene),nav=copy.deepcopy(self.nav)),self.f,self.a)
        old_view=LocalProblem(old,[0,1],[0,1])
        p=parent_for(self.scene,self.f,self.a,self.nav)
        new_view=snapshot_local(p,[0,1],[0,1])
        env=SimpleNamespace(done=False,problem=new_view)
        # The original guard short-circuits before touching tensors or a model.
        with self.assertRaisesRegex(ValueError,'another problem'):
            GlobalLocalPolicy.score(SimpleNamespace(),env,dict(problem_identity=old_view.identity))
        COUNTS['score_entry_guard_rejections']+=1

    def test_D_change_cannot_reuse_terminal_result(self):
        nav2=copy.deepcopy(self.nav);nav2['costs'][0,1]+=1;nav2['costs'][1,0]+=1
        p1=parent_for(self.scene,self.f,self.a,self.nav);p2=parent_for(self.scene,self.f,self.a,nav2)
        v1=extract_snapshot(p1,0,0,2,1);v2=extract_snapshot(p2,0,0,2,1)
        stats={k:0 for k in ('terminal_requests','terminal_cache_hits','terminal_attempts','terminal_completed','terminal_failed')}
        def terminal(view,raw,passed_stats):
            COUNTS['synthetic_terminal_mock_calls']+=1
            return dict(**view.metrics(raw),raw_route=list(raw),route=list(raw),H_s=0.,export_s=0.,validation_s=0.)
        cache=TerminalCache(stats,lambda record:None,'synthetic-H',limit=4)
        with patch('rc3_global_local.terminal.finish',terminal):
            r1=cache(v1,[0,1,3,0]);same=cache(v1,[0,1,3,0]);r2=cache(v2,[0,1,3,0])
        self.assertTrue(same['cache_hit']);self.assertFalse(r2['cache_hit'])
        self.assertEqual(r2['J']-r1['J'],1.)
        self.assertEqual(stats['terminal_attempts'],2)
        self.assertEqual(stats['terminal_cache_hits'],1)
        COUNTS['synthetic_terminal_cache_hits']+=1

    def test_mapping_snapshots_remain_JSON_compatible(self):
        p=parent_for(self.scene,self.f,self.a,self.nav)
        self.assertEqual(json.loads(json.dumps(p.scene)),self.scene)
        v=extract_snapshot(p,0,0,2,1)
        self.assertEqual(json.loads(json.dumps(v.description()))['snapshot_schema'],'rc3-bound-local-v1')

    def test_nonbinary_tables_are_rejected(self):
        wrong=self.a.astype(int);wrong[0,0,0]=2
        with self.assertRaisesRegex(ValueError,'binary'):
            snapshot_parent(self.scene,self.f,wrong,navigation=binding(self.scene,self.nav))

    def test_proxy_variable_height_rejected(self):
        s=copy.deepcopy(self.scene);s['candidates'][0]['h']=.5
        with self.assertRaisesRegex(ValueError,'fixed candidate height'):
            snapshot_parent(s,self.f,self.a,navigation=binding(s,self.nav))

    def test_unsupported_mode_and_missing_external_bundle_rejected(self):
        b=binding(self.scene,self.nav)
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            snapshot_parent(self.scene,self.f,self.a,navigation=b,mode='unknown')
        with self.assertRaisesRegex(ValueError,'requires its original bundle'):
            snapshot_parent(self.scene,self.f,self.a,navigation=b,mode='external_tables')

    def test_invalid_navigation_costs_rejected(self):
        for value in (float('nan'),float('-inf'),-1.):
            nav=copy.deepcopy(self.nav);nav['costs'][0,1]=value
            with self.assertRaisesRegex(ValueError,'invalid or negative'):
                binding(self.scene,nav)

    def test_saved_loader_rejects_unregistered_case_before_file_reads(self):
        with self.assertRaisesRegex(ValueError,'three existing'):
            load_saved_s0_parent(3,Path('does-not-exist'))


def main():
    tick=time.perf_counter();suite=unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeGuardTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    here=Path(__file__).resolve().parent
    receipt=dict(schema='rc3-runtime-guards-synthetic-regression-v1',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
        elapsed_s=time.perf_counter()-tick,operations=dict(COUNTS),
        research_calls=dict(scene_generations=0,navigation_preparations=0,planner_calls=0,H_calls=0,
            model_instantiations=0,model_forwards=0,backward_calls=0,optimizer_updates=0,S3_task_queries=0,original60_reads=0),
        source_sha256={p.relative_to(ROOT).as_posix():file_hash(p) for p in
            (here/'runtime_guards.py',Path(__file__).resolve(),ROOT/'rc3_global_local/problem.py',
             ROOT/'rc3_global_local/policy.py',ROOT/'rc3_global_local/terminal.py')},
        scope='Synthetic inputs only. Two terminal calls are mocked numeric identities, not real H. One score entry rejects before model/tensor computation. Numeric algorithms and S0 files unchanged.')
    (here/'GUARD_TEST_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
