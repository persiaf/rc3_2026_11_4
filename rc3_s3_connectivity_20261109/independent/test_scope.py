"""Independent saved-data/synthetic regression. No real scientific computation."""
import time
START=time.perf_counter()
import copy
import json
import os
import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
HOME=ROOT/'rc3_s3_connectivity_20261109'
sys.path.insert(0,str(ROOT))
CHECKS=0
OBSERVED=Counter({k:0 for k in ('navigation','scene_generation','FA_generation','model','native','H','optimizer','teacher')})


def profile(frame,event,arg):
    if event!='call':return
    filename=frame.f_code.co_filename.replace('\\','/');name=frame.f_code.co_name
    blocked={('rc3_baseline/baseline.py','navigation'):'navigation',
        ('rc3_fast/scenes.py','make_complex_scene'):'scene_generation',
        ('rc3_constructive/problem.py','resample_tasks'):'scene_generation',
        ('rc3_fast/solver.py','vectorized_feasibility'):'FA_generation',
        ('rc3_global_local/policy.py','__init__'):'model',
        ('rc3_global_local_impl_review/endpoint_baselines.py','run_endpoint'):'native',
        ('rc3_global_local/terminal.py','finish'):'H',
        ('torch/optim/optimizer.py','__init__'):'optimizer'}
    for (suffix,fn),key in blocked.items():
        if filename.endswith('/'+suffix) and name==fn:
            OBSERVED[key]+=1
            raise RuntimeError('Forbidden scientific function entry: '+key)


sys.setprofile(profile)
import numpy as np
from rc3_s3_evaluator.common import read,digest,identity,write_new,label,Meter,METHODS
from rc3_fast.planner import map_key
from rc3_global_local.problem import array_hash
from rc3_global_local_impl_review.runtime_guards import (
    bind_navigation,current_navigation_sources,navigation_hash,snapshot_parent,extract_snapshot,load_saved_s0_parent)
from rc3_global_local_trainer.navigation_compat import load_navigation_variants
from rc3_s3_connectivity_20261109.scope import scoped_view,unservable,TargetScopeInfeasible,prepare_view,evaluate,legacy_prepare_view
from rc3_s3_connectivity_20261109.scan import INPUT


def check(condition,message='Independent assertion failed'):
    global CHECKS
    CHECKS+=1
    if not condition:raise AssertionError(message)


def oracle(scene,F,A,D,ids,entry,exit_):
    """Independent loops, not the implementation's vectorized reduction."""
    answer=[]
    for t in ids:
        feasible=False
        for j in range(F.shape[1]):
            same_arm=any(bool(F[t,j,a]) and bool(A[t,j,a]) for a in range(2))
            if same_arm and np.isfinite(D[entry,j+1]) and np.isfinite(D[j+1,exit_]):
                feasible=True;break
        if not feasible:answer.append(int(t))
    return answer


def fixture(disconnected=True,cross_arm=False):
    # Invented test values only. No scene generator or navigation solver.
    scene=dict(nx=10,ny=10,start=[0.,0.],x_lines=[0.,1.,8.],y_lines=[0.,1.,8.],
        robot=dict(fixed_lift_height=.4),obstacles=[],
        tables=[dict(id=0,ix=0,iy=0,cx=1.,cy=1.,xmin=.8,xmax=1.2,ymin=.8,ymax=1.2,z=.75),
                dict(id=1,ix=8,iy=8,cx=8.,cy=8.,xmin=7.8,xmax=8.2,ymin=7.8,ymax=8.2,z=.75)],
        tasks=[dict(id=0,table_id=0,x=1.,y=1.,z=.75),dict(id=1,table_id=1,x=8.,y=8.,z=.75)],
        candidates=[dict(id=0,table_id=0,x=1.,y=0.,h=.4,theta=0.,corridor_anchor=[1.,0.]),
                    dict(id=1,table_id=1,x=2.,y=0.,h=.4,theta=0.,corridor_anchor=[2.,0.]),
                    dict(id=2,table_id=1,x=8.,y=0.,h=.4,theta=0.,corridor_anchor=[8.,0.])])
    F=np.zeros((2,3,2),dtype=bool);A=np.ones_like(F)
    F[0,0,0]=True;F[0,1,1]=True;F[1,2,0]=True
    if cross_arm:A[0,:,:]=False;A[0,:,1]=True;F[0,1,1]=False
    D=np.ones((4,4));np.fill_diagonal(D,0.)
    if disconnected:D[3,:]=D[:,3]=np.inf;D[3,3]=0.
    nav=dict(costs=D,coordinates=[[0.,0.],[1.,0.],[2.,0.],[8.,0.]],service_nodes=[0,1,2,3],
        predecessors=np.full((4,4),-9999,dtype=np.int32),node_count=4,edge_count=0,rejected_edge_count=0)
    bound=bind_navigation(scene,nav,source_map_key=map_key(scene),source_sha256=current_navigation_sources(),
        provenance=dict(producer='independent_synthetic_fixture_not_navigation_output'))
    return snapshot_parent(scene,F,A,navigation=bound),bound


SPEC=dict(x0=0,y0=0,width=4,height=3,endpoints='closed_depot')


class ScopeTests(unittest.TestCase):
    def test_01_exact_saved_diagnosis_and_witness(self):
        receipt=read(INPUT/'RECEIPT.json');scene=read(INPUT/'scene.json');w=read(HOME/'CONNECTIVITY_WITNESS.json')
        check(digest(INPUT/'scene.json')==receipt['scene_sha256']);check(digest(INPUT/'arrays.npz')==receipt['arrays_sha256'])
        with np.load(INPUT/'arrays.npz',allow_pickle=False) as arrays:F,A=arrays['F'].copy(),arrays['A'].copy()
        check(F.shape==A.shape==(233,800,2));check(array_hash(F)==receipt['content']['F']);check(array_hash(A)==receipt['content']['A'])
        variant=load_navigation_variants(scene,HOME/'diagnosis/navigation.npz')
        check(len(variant)==1);bound=variant[0][1];D=bound.nav['costs'];coords=np.asarray(bound.nav['coordinates']);services=bound.nav['service_nodes'];pred=bound.nav['predecessors']
        check(map_key(scene)==w['physical_map_key']);check(navigation_hash(bound.nav)==w['navigation_identity']);check(array_hash(D)==w['diagnostic_D_sha256'])
        check(w['original_D_sha256'] is None);check(w['diagnosis_kind']=='DIAGNOSTIC_RECOMPUTATION_NOT_ORIGINAL_D_RECOVERY')
        check(D.shape==(801,801));check(coords.shape==(1721,2));check(pred.shape==(801,1721))
        check(np.array_equal(np.isfinite(D),np.isfinite(D.T)));check(np.allclose(D[np.isfinite(D)],D.T[np.isfinite(D)],rtol=0,atol=1e-8))
        check(not np.isnan(D).any());check(not np.isneginf(D).any());check(np.all(D[np.isfinite(D)]>=0));check(np.all(np.diag(D)==0))
        for j,c in enumerate(scene['candidates']):check(np.allclose(coords[services[j+1]],[c['x'],c['y']],atol=1e-9,rtol=0))
        spec=receipt['content']['spec'];tables={t['id'] for t in scene['tables'] if spec['x0']<=t['ix']<spec['x0']+spec['width'] and spec['y0']<=t['iy']<spec['y0']+spec['height']}
        ids=[t['id'] for t in scene['tasks'] if t['table_id'] in tables]
        full=oracle(scene,F,A,D,list(range(len(F))),0,0);local=oracle(scene,F,A,D,ids,0,0)
        check(full==[221]);check(local==[]);check(len(ids)==23);check(ids==w['target_task_ids'])
        check(scene['tasks'][221]['table_id']==94);check(94 not in tables);check(221 not in ids)
        effective=[j for j in range(800) if any(F[221,j,a] and A[221,j,a] for a in range(2))]
        check(effective==[752,753]);radius=scene['robot']['base_radius']+scene['robot']['clearance'];check(radius==.4)
        blockers=[]
        for j in effective:
            check(not np.isfinite(D[0,j+1]) and not np.isfinite(D[j+1,0]));check(int(pred[0,services[j+1]])==-9999)
            c=scene['candidates'][j]
            blocked=[i for i,b in enumerate(scene.get('obstacles',[])) if b['xmin']-radius<=c['x']<=b['xmax']+radius and b['ymin']-radius<=c['y']<=b['ymax']+radius]
            check(bool(blocked));blockers.append(blocked)
        check(blockers[0]==blockers[1]);check(np.all((F&A).any((1,2))))
        parent=snapshot_parent(scene,F,A,navigation=bound)
        with self.assertRaisesRegex(ValueError,'Selected tasks cannot all be served'):extract_snapshot(parent,0,0,10,10,0,0)
        view,evidence=scoped_view(parent,spec)
        check(view.task_ids.tolist()==ids);check(evidence['full_parent_closed_depot_unservable_task_ids']==[221]);check(evidence['target_unservable_task_ids']==[])
        check(len(parent.scene['tasks'])==233);check(len(parent.scene['tables'])==100);check(len(parent.scene['candidates'])==800)
        check(np.array_equal(parent.F,F));check(np.array_equal(parent.A,A));check(np.array_equal(parent.D,D));check(not parent.F.flags.writeable and not parent.A.flags.writeable)
        self.saved_summary=dict(full_unservable_task_ids=full,target_unservable_task_ids=local,target_count=len(ids),blocking_task=221,blocking_table=94,
            blocking_candidates=effective,obstacle_indices=blockers[0],diagnostic_D_sha256=array_hash(D))
        # Written below through a class field for the final independent record.
        ScopeTests.saved_summary=self.saved_summary

    def test_02_full_failure_outside_target_and_boundary_candidate(self):
        p,bound=fixture();view,e=scoped_view(p,SPEC)
        check(unservable(p,[0,1],0,0)==[1]);check(view.task_ids.tolist()==[0]);check(view.candidate_ids.tolist()==[0,1])
        check(list(view.boundary_candidate_ids)==[1]);check(e['full_parent_closed_depot_unservable_task_ids']==[1]);check(e['target_unservable_task_ids']==[])
        check(view.parent is p);check(len(p.scene['tasks'])==2 and len(p.scene['tables'])==2 and len(p.scene['candidates'])==3)

    def test_03_real_target_infeasible_rejected_same_arm(self):
        p,bound=fixture(cross_arm=True)
        check(bool(p.F[0].any()) and bool(p.A[0].any()));check(not bool(p.effective[0].any()))
        with self.assertRaises(TargetScopeInfeasible) as caught:scoped_view(p,SPEC)
        check(caught.exception.evidence['target_unservable_task_ids']==[0])
        check(caught.exception.evidence['target_task_ids']==[0])

    def test_04_open_closed_and_invalid_bounds(self):
        p,bound=fixture();closed,ce=scoped_view(p,SPEC);opened,oe=scoped_view(p,dict(SPEC,endpoints='open_first_last_connected'))
        check((closed.entry,closed.exit)==(0,0));check((opened.entry,opened.exit)==(1,2));check(opened.D[0,0]==p.D[1,2])
        check(opened.task_ids.tolist()==closed.task_ids.tolist());check(opened.candidate_ids.tolist()==closed.candidate_ids.tolist())
        with self.assertRaises(ValueError):scoped_view(p,dict(SPEC,x0=9))
        with self.assertRaises(ValueError):scoped_view(p,dict(SPEC,endpoints='unknown'))

    def test_05_full_pass_preserves_original_local_semantics(self):
        # Every original frozen S0 parent is reused without navigation solving.
        comparisons=0
        for case in range(3):
            p=load_saved_s0_parent(case);extract_snapshot(p,0,0,10,10,0,0)
            for w,h in ((4,3),(6,6),(8,8)):
                for endpoint in ('closed_depot','open_first_last_connected'):
                    center=endpoint.startswith('open');spec=dict(width=w,height=h,x0=(10-w)//2 if center else 0,y0=(10-h)//2 if center else 0,endpoints=endpoint)
                    if center:
                        legal=np.flatnonzero(np.isfinite(p.D[0,1:])&np.isfinite(p.D[1:,0]))+1;entry,exit_=int(legal[0]),int(legal[-1])
                    else:entry=exit_=0
                    before=extract_snapshot(p,spec['x0'],spec['y0'],w,h,entry,exit_);after,e=scoped_view(p,spec)
                    check(before.description()==after.description());check(np.array_equal(before.F,after.F));check(np.array_equal(before.C,after.C));check(np.array_equal(before.D,after.D))
                    check(before.task_ids.tolist()==after.task_ids.tolist());check(before.candidate_ids.tolist()==after.candidate_ids.tolist());check(e['full_parent_closed_depot_unservable_task_ids']==[])
                    comparisons+=1
        check(comparisons==18);ScopeTests.old_semantics_comparisons=comparisons

    def test_06_same_input_contract_for_seven_dispatches(self):
        from rc3_s3_evaluator import pipelines
        p,bound=fixture();seen=[]
        class Provider:
            def load(self,case,meter):return copy.deepcopy(dict(p.scene)),np.array(p.F),np.array(p.A),dict(SPEC),{}
        class Pool:
            def get(self,scene,kind,meter):return bound
        def frozen_pipeline_stub(request,*args):
            check(pipelines.prepare_view is prepare_view)
            view,source=pipelines.prepare_view(Provider(),dict(physical_key=map_key(p.scene)),Pool(),'quality',Meter())
            seen.append((request['method'],view.task_ids.tolist(),view.candidate_ids.tolist(),source['scope_evidence']))
            return dict(method=request['method'])
        for method in METHODS:
            with patch.object(pipelines,'evaluate',frozen_pipeline_stub):
                result=evaluate(dict(method=method))
            check(result['input_contract'].startswith('S3_TARGET_SCOPE_R1'));check(pipelines.prepare_view is legacy_prepare_view)
        check([x[0] for x in seen]==list(METHODS));check(all(x[1:]==seen[0][1:] for x in seen))
        with patch.object(pipelines,'prepare_view',lambda *a:None):
            with self.assertRaises(ValueError):evaluate(dict(method=METHODS[0]))

    def test_07_original_bytes_and_profile_guard(self):
        frozen=read(HOME/'FROZEN_EVIDENCE_INDEX.json')['files']
        for name,e in frozen.items():
            p=ROOT/name;check(p.stat().st_size==e['bytes'],name);check(digest(p)==e['sha256'],name)
        check(all(v==0 for v in OBSERVED.values()));ScopeTests.frozen_files_checked=len(frozen)

    def test_08_recovery_identity_and_accounting(self):
        from rc3_s3_connectivity_20261109 import recovery,resume
        permit,cases,ledger=recovery.import_ledger()
        check(ledger['old_debited_requests']==2059);check(ledger['old_successful_results']==2057)
        check(len(ledger['missing_failure_ids'])==2);check(abs(ledger['charged_first_seed_wall_s']-2896.9277158999903)<1e-8)
        check(ledger['remaining_requests']==[3233,5292,5292]);check(ledger['original_exact_wall_s'] is None)
        check(ledger['remaining_wall_s']==[11503.07228410001,14400,14400]);check(ledger['new_paid_attempts_to_refill_old_failures']==0)
        lengths=[]
        for seed in (11622026,11622027,11622028):
            remaining=recovery.next_requests(cases,ledger,seed);lengths.append(len(remaining))
            historical=set(ledger['baselines'][str(seed)]['terminal_ids'])
            check(not historical&{q['id'] for q in remaining})
            check(not set(ledger['missing_failure_ids'])&{q['id'] for q in remaining})
        check(lengths==[3233,5292,5292])
        for failed in ledger['missing_failure_ids']:
            with self.assertRaises(ValueError):recovery.next_requests(cases,ledger,11622026,completed=[failed])
        with self.assertRaises(ValueError):recovery.next_requests(cases,ledger,42)
        fixtures=HOME/'independent/fixtures'
        binding=dict(test='independent_clock_fixture')
        baseline=ledger['baselines']['11622026']
        def clock_folder(name,status='CLOSED_BOUNDARY',other_binding=None):
            folder=fixtures/name;token='f'*64;p=folder/'sessions'/token
            write_new(p/'OPEN.json',dict(token=token,parent_pid=os.getpid(),binding=other_binding or binding))
            write_new(p/'FINAL.json',dict(token=token,status=status,elapsed_s=3.25))
            write_new(p/'FINAL_SEAL.json',dict(final_sha256=digest(p/'FINAL.json')))
            events=[dict(event='session_open',token=token),dict(event='session_close',token=token)]
            return folder,events
        folder,events=clock_folder('closed_clock')
        check(abs(resume.used_wall(folder,baseline,events,binding)-(baseline['wall_carry_s']+3.25))<1e-8)
        with self.assertRaises(ValueError):resume.used_wall(folder,baseline,[],binding)
        with self.assertRaises(ValueError):resume.used_wall(folder,baseline,events,dict(test='wrong_binding'))
        failed,ev=clock_folder('failed_clock',status='FAILED')
        with self.assertRaises(ValueError):resume.used_wall(failed,baseline,ev,binding)
        missing=fixtures/'missing_clock'
        with self.assertRaises(ValueError):resume.used_wall(missing,baseline,[dict(event='session_open',token='absent')],binding)
        # Parent close failure persists a separate sealed clock without a real
        # scientific Journal or a request. It must never be refunded to zero.
        parent_fixture=fixtures/'parent_close_failure';token='e'*64
        def fail_close(elapsed):raise OSError('Synthetic independent parent-close fault')
        f=resume.finish(parent_fixture,token,time.perf_counter(),'FAILED','synthetic',fail_close,120)
        check(f['status']=='FAILED' and f['elapsed_s']>=0 and f['journal_close_error'] is not None)
        check((parent_fixture/'sessions'/token/'FINAL.json').is_file())
        check(read(parent_fixture/'sessions'/token/'FINAL_SEAL.json')['final_sha256']==digest(parent_fixture/'sessions'/token/'FINAL.json'))
        activation=fixtures/'activation';run=activation/'runs'/'scope_r1'
        with patch.object(recovery,'RUN',run),patch.object(recovery,'HOME',activation):
            recovery.initialize(binding,ledger)
            marker=activation/'SCOPE_RECOVERY_ACTIVATION.json';before=digest(marker)
            recovery.initialize(binding,ledger);check(digest(marker)==before)
            with self.assertRaises(ValueError):recovery.initialize(dict(test='foreign_source'),ledger)
            renamed=run.parent/'renamed_scope'
            check(run.resolve().is_relative_to(fixtures.resolve()) and renamed.resolve().is_relative_to(fixtures.resolve()))
            run.rename(renamed)
            with self.assertRaises(ValueError):recovery.initialize(binding,ledger)
            check(not run.exists() and (renamed/'AUTHORITY.json').is_file())
        ScopeTests.recovery_summary=dict(inherited_debits=2059,inherited_success=2057,inherited_failures=2,
            inherited_wall_s=ledger['charged_first_seed_wall_s'],remaining_requests=lengths,
            remaining_wall_s=ledger['remaining_wall_s'],missing_failure_ids=ledger['missing_failure_ids'],
            old_terminal_intersection=0,original_exact_wall_s=None,new_formal_requests_executed=0,
            synthetic_clock_source_and_directory_reset_rejections='PASS')


def main():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ScopeTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',checks=CHECKS,tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),observed_scientific_function_entries=dict(OBSERVED),
        saved_input_summary=getattr(ScopeTests,'saved_summary',None),old_semantics_comparisons=getattr(ScopeTests,'old_semantics_comparisons',None),
        frozen_files_checked=getattr(ScopeTests,'frozen_files_checked',None),recovery_summary=getattr(ScopeTests,'recovery_summary',None),
        elapsed_s=time.perf_counter()-START)
    write_new(HOME/'independent/CHILD_RESULT.json',record)
    sys.setprofile(None)
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
