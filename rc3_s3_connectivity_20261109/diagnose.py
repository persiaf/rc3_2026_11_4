"""One pre-registered navigation diagnosis from byte-pinned saved S3_042 input."""
import time
PROCESS_START=time.perf_counter()
import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
import numpy as np
from rc3_s3_evaluator.common import ROOT,read,digest,identity,write_new,label,require,now
from .scan import HOME,INPUT


def inputs():
    receipt=read(INPUT/'RECEIPT.json');scene=read(INPUT/'scene.json')
    require(digest(INPUT/'scene.json')==receipt['scene_sha256'] and digest(INPUT/'arrays.npz')==receipt['arrays_sha256'] and
        identity(scene)==receipt['content']['scene'],'Saved source changed')
    from rc3_global_local.problem import array_hash
    with np.load(INPUT/'arrays.npz',allow_pickle=False) as arrays:
        require(set(arrays.files)=={'F','A'},'Unexpected saved task fields')
        F,A=arrays['F'].copy(),arrays['A'].copy()
    require(array_hash(F)==receipt['content']['F'] and array_hash(A)==receipt['content']['A'],'F/A identity differs')
    return scene,F,A,receipt['content']['spec']


def endpoint_ids(scene,D,spec):
    if spec['endpoints']=='closed_depot':return 0,0
    require(spec['endpoints']=='open_first_last_connected','Unknown endpoint convention')
    nodes=np.flatnonzero(np.isfinite(D[0,1:])&np.isfinite(D[1:,0]))+1
    require(len(nodes)>=2,'Original open endpoints unavailable')
    return int(nodes[0]),int(nodes[-1])


def coverage(scene,F,A,D,task_ids,entry,exit_,window_ids):
    effective=F&A;connected=np.isfinite(D[entry,1:])&np.isfinite(D[1:,exit_])
    failed=[];rows=[]
    for tid in task_ids:
        candidates=np.flatnonzero(effective[tid].any(1));remaining=candidates[connected[candidates]]
        rows.append(dict(task_id=int(tid),candidate_count=len(candidates),connected_candidate_count=len(remaining)))
        if len(remaining):continue
        detail=[]
        for j in candidates:
            reasons=[]
            if not np.isfinite(D[entry,j+1]):reasons.append('ENTRY_TO_CANDIDATE_UNREACHABLE')
            if not np.isfinite(D[j+1,exit_]):reasons.append('CANDIDATE_TO_EXIT_UNREACHABLE')
            detail.append(dict(global_candidate_id=int(j),global_navigation_service_index=int(j+1),
                candidate_table_id=scene['candidates'][j]['table_id'],same_arm_indices=np.flatnonzero(effective[tid,j]).tolist(),
                entry_reachable=bool(np.isfinite(D[entry,j+1])),exit_reachable=bool(np.isfinite(D[j+1,exit_])),reasons=reasons))
        failed.append(dict(global_task_id=int(tid),global_table_id=scene['tasks'][tid]['table_id'],
            in_target_window=bool(tid in window_ids),same_arm_FA_candidates=detail,
            same_arm_FA_candidate_count=len(candidates),endpoint_filtered_candidate_count=0,
            reason='NO_SAME_ARM_FA_CANDIDATE' if not len(candidates) else 'ALL_EFFECTIVE_CANDIDATES_FILTERED_BY_ENDPOINT_CONNECTIVITY'))
    return dict(entry=entry,exit=exit_,task_count=len(task_ids),unservable_task_ids=[x['global_task_id'] for x in failed],
        unservable_tasks=failed,task_candidate_counts=rows,all_targets_serviceable=not failed)


def worker():
    counters=dict(navigation=0,task_generation=0,proxy_F_A_generation=0,model=0,native=0,H=0,teacher=0,optimizer=0)
    forbidden={('rc3_fast/scenes.py','make_complex_scene'):'task_generation',
        ('rc3_constructive/problem.py','resample_tasks'):'task_generation',
        ('rc3_fast/solver.py','vectorized_feasibility'):'proxy_F_A_generation',
        ('rc3_global_local/policy.py','__init__'):'model',
        ('rc3_global_local_impl_review/endpoint_baselines.py','run_endpoint'):'native',
        ('rc3_global_local/terminal.py','finish'):'H',
        ('torch/optim/optimizer.py','__init__'):'optimizer'}
    def profile(frame,event,arg):
        if event!='call':return
        filename=frame.f_code.co_filename.replace('\\','/');name=frame.f_code.co_name
        if filename.endswith('/rc3_baseline/baseline.py') and name=='navigation':
            counters['navigation']+=1;require(counters['navigation']<=1,'Navigation budget exceeded')
        for (suffix,fn),key in forbidden.items():
            if filename.endswith('/'+suffix) and name==fn:
                counters[key]+=1;raise RuntimeError('Forbidden scientific entry '+key)
    sys.setprofile(profile)
    error=None
    try:
        reg=read(HOME/'DIAGNOSTIC_REGISTRATION.json')
        for name,sha in reg['source_sha256'].items():require(digest(ROOT/name)==sha,'Registered diagnostic source changed')
        scene,F,A,spec=inputs()
        from rc3_fast.planner import map_key
        from rc3_baseline.baseline import navigation
        from rc3_global_local.problem import array_hash
        from rc3_global_local_impl_review.runtime_guards import current_navigation_sources,navigation_hash,bind_navigation,snapshot_parent,extract_snapshot
        require(map_key(scene)==reg['physical_map_key'] and current_navigation_sources()==reg['navigation_sources'],'Navigation source/geometry mismatch')
        write_new(HOME/'diagnosis/NAVIGATION_CALL.json',dict(predebited_calls=1,utc=now(),registration_sha256=digest(HOME/'DIAGNOSTIC_REGISTRATION.json')))
        nav_start=time.perf_counter();nav=navigation(scene);nav_s=time.perf_counter()-nav_start
        from rc3_global_local_trainer.navigation_compat import save_navigation_artifact
        save_navigation_artifact(scene,nav,HOME/'diagnosis/navigation.npz',nav_s)
        D=np.asarray(nav['costs']);coords=np.asarray(nav['coordinates']);services=np.asarray(nav['service_nodes']);pred=nav['predecessors']
        require(D.shape==(len(scene['candidates'])+1,)*2 and np.isfinite(coords).all(),'Navigation dimensions/coordinates invalid')
        require(not np.isnan(D).any() and not np.isneginf(D).any() and (D[np.isfinite(D)]>=0).all(),'Invalid costs')
        require(np.array_equal(np.isfinite(D),np.isfinite(D.T)) and np.allclose(D[np.isfinite(D)],D.T[np.isfinite(D)],rtol=0,atol=1e-8),'Asymmetric frozen navigation')
        require(np.allclose(np.diag(D),0) and pred.shape==(len(services),len(coords)),'Wrong diagonal/predecessor shape')
        for kind in ('tasks','tables','candidates'):require([v['id'] for v in scene[kind]]==list(range(len(scene[kind]))),'Global ID mismatch')
        require(np.allclose(coords[services[0]],scene['start'],rtol=0,atol=1e-9),'Depot mapping differs')
        for j,c in enumerate(scene['candidates']):require(np.allclose(coords[services[j+1]],[c['x'],c['y']],rtol=0,atol=1e-9),'Service node ID differs')
        tables={t['id'] for t in scene['tables'] if spec['x0']<=t['ix']<spec['x0']+spec['width'] and spec['y0']<=t['iy']<spec['y0']+spec['height']}
        target_ids=[t['id'] for t in scene['tasks'] if t['table_id'] in tables];target_set=set(target_ids)
        entry,exit_=endpoint_ids(scene,D,spec);require(np.isfinite(D[entry,exit_]),'Actual endpoints disconnected')
        full=coverage(scene,F,A,D,list(range(len(F))),0,0,target_set)
        target=coverage(scene,F,A,D,target_ids,entry,exit_,target_set)
        path_checks=0
        for service in range(len(services)):
            node=int(services[service]);source=int(services[entry])
            if not np.isfinite(D[entry,service]):
                require(int(pred[entry,node])==-9999,'Unreachable endpoint predecessor not sentinel');continue
            path=[node]
            while path[-1]!=source:
                p=int(pred[entry,path[-1]]);require(p>=0 and p not in path and len(path)<=len(coords),'Invalid predecessor chain');path.append(p)
            distance=sum(float(np.linalg.norm(coords[a]-coords[b])) for a,b in zip(path,path[1:]))
            require(abs(distance-D[entry,service])<=1e-7,'Predecessor path length differs from D');path_checks+=1
        bound=bind_navigation(scene,nav,source_map_key=map_key(scene),source_sha256=current_navigation_sources(),provenance=dict(producer='paid_saved_input_connectivity_diagnosis',original_D_hash=None))
        parent=snapshot_parent(scene,F,A,navigation=bound);old_failure=None;view_description=None
        try:extract_snapshot(parent,0,0,10,10,0,0)
        except ValueError as exc:old_failure=str(exc)
        if target['all_targets_serviceable']:
            view=extract_snapshot(parent,spec['x0'],spec['y0'],spec['width'],spec['height'],entry,exit_);view_description=view.description()
        outcome='A_OUTSIDE_TARGET_ONLY' if full['unservable_tasks'] and target['all_targets_serviceable'] else ('B_TARGET_UNSERVICEABLE' if not target['all_targets_serviceable'] else 'C_FAILURE_NOT_REPRODUCED')
        witness=dict(schema='rc3-s3-042-connectivity-witness-v1',outcome=outcome,diagnosis_kind='DIAGNOSTIC_RECOMPUTATION_NOT_ORIGINAL_D_RECOVERY',
            original_D_sha256=None,diagnostic_D_sha256=array_hash(D),navigation_identity=navigation_hash(nav),navigation_sources=current_navigation_sources(),
            physical_map_key=map_key(scene),saved_input_files=read(HOME/'NAVIGATION_SEARCH.json')['input_files'],
            full_parent_closed_depot=full,actual_target_window=target,window_spec=spec,target_table_ids=sorted(tables),target_task_ids=target_ids,
            local_view=view_description,original_full_parent_error_reproduced=old_failure,
            navigation_validation=dict(cost_shape=list(D.shape),coordinate_shape=list(coords.shape),predecessor_shape=list(pred.shape),
                finite_symmetry=True,finite_cost_symmetry=True,nonnegative=True,diagonal_zero=True,global_ID_mapping=True,
                actual_entry=entry,actual_exit=exit_,entry_exit_connected=True,endpoint_predecessor_paths_checked=path_checks,
                depot_connected_candidates=int(np.isfinite(D[0,1:]).sum()),depot_disconnected_candidate_ids=np.flatnonzero(~np.isfinite(D[0,1:])).tolist(),
                edge_count=nav['edge_count'],rejected_edge_count=nav['rejected_edge_count'],node_count=nav['node_count']),
            actual_function_entries=counters,navigation_wall_s=nav_s,full_parent_not_a_local_route_or_cost_query=True)
        write_new(HOME/'CONNECTIVITY_WITNESS.json',witness)
        write_new(HOME/'diagnosis/WORKER_COMPLETE.json',dict(witness_sha256=digest(HOME/'CONNECTIVITY_WITNESS.json'),counters=counters,worker_elapsed_s=time.perf_counter()-PROCESS_START))
    except BaseException as exc:
        error=repr(exc);write_new(HOME/'diagnosis/WORKER_FAILURE.json',dict(error=error,traceback=traceback.format_exc(),counts=counters,worker_elapsed_s=time.perf_counter()-PROCESS_START));raise
    finally:sys.setprofile(None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker:return worker()
    require(not (HOME/'diagnosis/ATTEMPT.json').exists(),'Diagnostic attempt already charged; no automatic retry')
    reg=read(HOME/'DIAGNOSTIC_REGISTRATION.json')
    write_new(HOME/'diagnosis/ATTEMPT.json',dict(utc=now(),reserved_navigation_attempts=1,registration_sha256=digest(HOME/'DIAGNOSTIC_REGISTRATION.json')))
    error=None;code=None
    try:
        with (HOME/'diagnosis/worker.log').open('xb') as stream:
            p=subprocess.Popen([sys.executable,'-B','-X','utf8','-m','rc3_s3_connectivity_20261109.diagnose','--worker'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            try:code=p.wait(timeout=max(.001,reg['wall_cap_s']-(time.perf_counter()-PROCESS_START)))
            except subprocess.TimeoutExpired:p.kill();p.wait();raise TimeoutError('300s diagnostic cap')
        require(code==0,'Diagnostic worker failed')
    except BaseException as exc:error=repr(exc)
    elapsed=time.perf_counter()-PROCESS_START
    final=dict(status='PASS' if not error and elapsed<=reg['wall_cap_s'] else 'FAILED',error=error,exit_code=code,
        all_diagnostic_wall_s=elapsed,wall_cap_s=reg['wall_cap_s'],navigation_calls=read(HOME/'diagnosis/WORKER_COMPLETE.json')['counters']['navigation'] if (HOME/'diagnosis/WORKER_COMPLETE.json').exists() else None,
        diagnostic_fees_separate_from_formal_S3_ledger=True,original_formal_charged_wall_s=2896.9277158999903,utc=now())
    write_new(HOME/'diagnosis/FINAL.json',final);require(final['status']=='PASS','Diagnosis stopped; evidence retained')
    print(dict(final=final,outcome=read(HOME/'CONNECTIVITY_WITNESS.json')['outcome']))


if __name__=='__main__':main()
