"""Read-only result/resource audit AFTER the authorized process has exited."""
import csv
from collections import Counter
from pathlib import Path
import time
import numpy as np
from rc3_s3_evaluator.common import ROOT,SEEDS,H_METHODS,MODEL_ROLES,read,digest,identity,label,require,write_new,now
from rc3_s3_evaluator.journal import grid
from rc3_global_local.problem import array_hash
from rc3_s3_recovery_20261108.storage import read_chain
from rc3_s3_recovery_20261108.analysis import analyze
from rc3_s3_connectivity_20261109.recovery import RUN,FAILED_OLD,FAILED_SCOPE,admit
from rc3_s3_connectivity_20261109.scope import CONTRACT

OUT=ROOT/'rc3_s3_scope_results_20261109'


def add(target,values):
    for k,v in values.items():
        require(type(v) in (int,float),'Unexpected nonnumeric counter '+k);target[k]+=v


def collect():
    began=time.perf_counter();permit,cases,baseline,binding=admit()
    result_dir=OUT/'results';require(not result_dir.exists(),'Do not overwrite an existing closeout')
    expected={q['id']:q for s in SEEDS for q in grid(cases,s)};case_map={c['id']:c for c in cases}
    states={rid:dict(request=q,state='UNEXECUTED') for rid,q in expected.items()}
    states[FAILED_OLD].update(state='HISTORICAL_INFRASTRUCTURE_FAILURE_MISSING')
    states[FAILED_SCOPE].update(state='HISTORICAL_FULL_PARENT_SCOPE_REJECTION_MISSING')
    old_resource=read(ROOT/'rc3_s3_recovery_20261108/results/RESOURCE_ACCOUNTING.json')
    model_index=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json');model_files=[]
    diagnostic_D=read(ROOT/'rc3_s3_connectivity_20261109/CONNECTIVITY_WITNESS.json')['diagnostic_D_sha256']
    for seed,models in model_index['models'].items():
        for role,m in models.items():
            for pk,hk in (('path','checkpoint_sha256'),('sidecar_path','sidecar_sha256'),('commit_path','commit_sha256')):
                require(digest(ROOT/m[pk])==m[hk],'Frozen model provenance changed')
                model_files.append(dict(seed=int(seed),role=role,path=m[pk],sha256=m[hk]))
    old_index=read(ROOT/'rc3_s3_recovery_20261108/results/RESULT_ARTIFACT_INDEX.json')['artifacts']
    ops=Counter(old_resource['request_operation_counts']);seconds=Counter(old_resource['all_operation_end_seconds_inclusive'])
    pipeline=Counter(old_resource['pipeline_counters']);internal=Counter(old_resource['internal_search_function_entries'])
    inclusive=Counter(old_resource['internal_search_inclusive_s']);exclusive=Counter(old_resource['internal_search_exclusive_s'])
    hot=Counter(old_resource['hot_prime_counts']);method_h=Counter();assigned_arms=Counter()
    records=[];thin=[];artifact_index=[];stable={};pairs={};inputs={};input_signatures={};byseed={};failures=[]
    old_times={r['request_id']:r for r in csv.DictReader((ROOT/'rc3_s3_recovery_20261108/results/REQUEST_METRICS.csv').open(encoding='utf8',newline=''))}
    def saved_input(case_id):
        if case_id in inputs:return inputs[case_id]
        number=int(case_id.split('_')[1])
        parent=ROOT/'rc3_s3_evaluator/runs/rc3_v3_global_local_20261106_01'
        folder=(parent if number<4 else parent/'continuations/persistence_r1')/'inputs'/case_id if number<=42 else RUN/'inputs'/case_id
        receipt=read(folder/'RECEIPT.json');scene=read(folder/'scene.json')
        require(digest(folder/'scene.json')==receipt['scene_sha256'] and digest(folder/'arrays.npz')==receipt['arrays_sha256'],'Saved task artifact changed')
        with np.load(folder/'arrays.npz',allow_pickle=False) as z:F,A=z['F'].copy(),z['A'].copy()
        require(array_hash(F)==receipt['content']['F'] and array_hash(A)==receipt['content']['A'] and identity(scene)==receipt['content']['scene'],'Task content receipt differs')
        inputs[case_id]=(scene,F,A);return inputs[case_id]
    def record(path,event,new=False,components=None):
        require(digest(path)==event['artifact_sha256'],'Result artifact hash changed')
        r=read(path);q=r['request'];rid=q['id'];case=case_map[q['case_id']]
        require(expected.get(rid)==q and states[rid]['state']=='UNEXECUTED','Duplicate/foreign/failed-slot result')
        require(r['valid'] and all(r[k]==case[k] for k in ('group','size','endpoint','physical_key')),'Wrong result stratum')
        if q['method'] in MODEL_ROLES:
            role=MODEL_ROLES[q['method']];m=model_index['models'][str(q['seed'])][role]
            require(r['training_weight_role']==role and all(r['weight'][k]==m[k] for k in ('path','checkpoint_sha256','model_tensor_sha256','cursor','phase')),'Wrong method/seed/fixed endpoint model')
        else:require(r['weight'] is None,'Native baseline used neural weights')
        cert=r['plan']['validation'];require(cert['valid_model_plan'] and cert['all_tasks_served'] and cert['effective_same_arm_F_and_A'] and cert['endpoint_constraints_satisfied'] and cert['collision_count']==0,'Invalid original constraint certificate')
        scene,F,A=saved_input(q['case_id']);source=r['source'];plan=r['plan'];ids=source['global_task_ids'];legal=set(source['global_candidate_ids'])
        require(ids==plan['target_task_ids_global'] and not plan['unserved_task_ids'],'Target set/coverage mismatch')
        require(array_hash(F)==source['F_sha256'] and array_hash(A)==source['A_sha256'],'Result F/A source differs')
        if q['case_id']=='S3_042':require(source['D_sha256']==diagnostic_D,'Formal S3_042 D differs from separately paid diagnostic witness')
        served=[]
        for assignment in plan['assignments']:
            t,j=assignment['task_id'],assignment['candidate_id'];arm=assignment['arm'];a={'L':0,'R':1}.get(arm)
            require(t in ids and j in legal and a is not None and F[t,j,a] and A[t,j,a],'Same-arm F&A assignment invalid')
            served.append(t);assigned_arms[arm]+=1
        require(len(served)==len(ids) and set(served)==set(ids),'Task repeated or not covered')
        sig=(source['evaluation_parent_identity'],source['view_identity'],source['F_sha256'],source['A_sha256'],source['D_sha256'],tuple(ids),tuple(source['global_candidate_ids']))
        require(q['case_id'] not in input_signatures or input_signatures[q['case_id']]==sig,'Input interpretation changed across methods/seeds/versions');input_signatures[q['case_id']]=sig
        expected_h=int(q['method'] in H_METHODS);require(r['counts'].get('common_H_invocations',0)==expected_h,'Wrong H pipeline count');method_h[q['method']]+=expected_h
        if new:require(r['input_contract']==CONTRACT and source['scope_evidence']['target_unservable_task_ids']==[],'Missing/wrong scope evidence')
        require(abs(r['metrics']['J']-r['metrics']['L']-.75*r['metrics']['K'])<=1e-6 and abs(r['metrics']['raw_J']-r['metrics']['raw_L']-.75*r['metrics']['raw_K'])<=1e-6,'Objective inconsistent')
        key=(q['case_id'],q['seed'],q['method']);value=(r['raw_route_sha256'],r['route_sha256'],r['metrics'])
        require(key not in stable or stable[key]==value,'Frozen route/cost changed across repeats');stable[key]=value
        if q['method'] in ('RL_raw','RL_plus_H'):pairs.setdefault((q['case_id'],q['seed'],q['kind'],q['repeat']),{})[q['method']]=r['raw_route_sha256']
        t=event['complete_time_s'];require(np.isfinite(t) and t>0,'Bad complete timing')
        scope=r.get('input_contract','LEGACY_FULL_PARENT_HARD_GATE')
        rec=dict(request_id=rid,case_id=q['case_id'],seed=q['seed'],method=q['method'],kind=q['kind'],repeat=q['repeat'],group=r['group'],size='x'.join(map(str,r['size'])),endpoint=r['endpoint'],physical_key=r['physical_key'],
            **r['metrics'],complete_time_s=t,H_gain_J=r['H_gain_J'],valid=True,task_count=len(ids),tasks_covered=len(served),same_arm_assignments_valid=True,
            scope_revision=scope,persistence_revision=r.get('persistence_revision','legacy_b348'),input_identity=source['view_identity'],D_sha256=source['D_sha256'],
            raw_route_sha256=r['raw_route_sha256'],route_sha256=r['route_sha256'],artifact_path=label(path),artifact_sha256=event['artifact_sha256'])
        win=plan['window'];tables={t['id'] for t in scene['tables'] if win['x0']<=t['ix']<win['x0']+win['width'] and win['y0']<=t['iy']<win['y0']+win['height']}
        rec.update(legal_candidate_count=len(legal),legal_boundary_candidate_count=sum(scene['candidates'][j]['table_id'] not in tables for j in legal),
            full_parent_unservable_count=len(source.get('scope_evidence',{}).get('full_parent_closed_depot_unservable_task_ids',[])),target_unservable_count=0)
        component_values=components if components is not None else r['component_seconds']
        for name in ('input_generation_and_preparation','cold_navigation','navigation_access','whole_parent_and_window',
            'model_load_and_verify','features_and_global_local_encoding','greedy_construction','common_H_pipeline','native_pipeline','export','validation','artifact_write'):
            rec[name+'_s']=component_values.get(name)
        records.append(rec);thin.append({**{k:r[k] for k in ('request','case_id','group','size','endpoint','physical_key','valid','metrics','raw_route_sha256','route_sha256')},'complete_time_s':t,'persistence_revision':rec['persistence_revision']})
        artifact_index.append(dict(path=label(path),sha256=event['artifact_sha256'],bytes=path.stat().st_size))
        states[rid].update(state='COMPLETED',scope_revision=scope,artifact_sha256=event['artifact_sha256'])
        if new:
            add(pipeline,r['pipeline_counters']);add(internal,r['internal_search']['counts']);add(inclusive,r['internal_search']['inclusive_s']);add(exclusive,r['internal_search']['exclusive_s'])
    for e in old_index:
        r=read(ROOT/e['path']);rid=r['request']['id'];record(ROOT/e['path'],dict(artifact_sha256=e['sha256'],complete_time_s=float(old_times[rid]['complete_time_s'])))
    for seed in SEEDS:
        folder=RUN/str(seed);b=baseline['baselines'][str(seed)]
        if not folder.exists():
            byseed[str(seed)]=dict(status='NOT_STARTED',new_debits=0,new_success=0,new_failures=0,total_debits=b['debits'],total_success=b['success'],charged_wall_s=b['wall_carry_s'],sessions=[]);continue
        events,tail=read_chain(folder/'journal.jsonl');head=read(folder/'HEAD.json')
        require(not (folder/'HEAD.pending').exists() and head['sequence']==len(events) and head['sha256']==tail,'Unclosed new anchor')
        starts={};done={};errors={};current=None;outside_counts={};current_seconds={}
        for e in events:
            kind=e['event']
            if kind=='request_start':
                q=e['request'];require(q['id'] not in starts and q['id'] not in b['terminal_ids'] and expected[q['id']]==q,'Duplicate/foreign predebit');starts[q['id']]=q;current={};current_seconds={}
            elif kind=='operation_end':
                seconds[e['operation']]+=e['elapsed_s']
                if current is not None:
                    current=e['counts'];current_seconds[e['operation']]=current_seconds.get(e['operation'],0.)+e['elapsed_s']
                else:outside_counts=e['counts']
            elif kind=='hot_prime':add(hot,e['counts']);outside_counts={}
            elif kind=='request_complete':
                require(e['id'] in starts and e['id'] not in done and current is not None,'Unbilled completion');add(ops,current);current=None;done[e['id']]=e
                record(folder/'results'/(e['id']+'.json'),e,True,current_seconds)
            elif kind=='request_error':
                require(e['id'] in starts and e['id'] not in errors,'Unbilled error');add(ops,e['counts']);current=None;errors[e['id']]=e;failures.append(dict(seed=seed,**e))
                states[e['id']].update(state='NEW_FAILED_REQUEST',error=e['error'],scope_evidence=e.get('scope_evidence'))
        if outside_counts:add(hot,outside_counts)
        require(current is None and set(starts)==set(done)|set(errors) and events[-1]['event']=='session_close','Unclosed request/session')
        require({p.stem for p in (folder/'results').glob('*.json')}==set(done),'Orphan result')
        remaining=[q for q in grid(cases,seed) if q['id'] not in b['terminal_ids']]
        require(list(starts)==[q['id'] for q in remaining[:len(starts)]],'New execution not original unexecuted prefix')
        sessions=[]
        for p in sorted((folder/'sessions').iterdir()):
            final=read(p/'FINAL.json');seal=read(p/'FINAL_SEAL.json');require(digest(p/'FINAL.json')==seal['final_sha256'],'Final clock changed')
            sessions.append(dict(path=label(p),final_sha256=seal['final_sha256'],**final))
        used=b['wall_carry_s']+sum(s['elapsed_s'] for s in sessions)
        schedule=folder/'SCHEDULE_COMPLETE.json';sealed=(folder/'SCHEDULE_SEAL.json').exists()
        if sealed:
            seal=read(folder/'SCHEDULE_SEAL.json');require(seal['schedule_sha256']==digest(schedule) and read(schedule)['journal_sha256']==digest(folder/'journal.jsonl'),'Schedule seal differs')
        byseed[str(seed)]=dict(status='SCHEDULE_DISPOSED_SEALED' if sealed else 'STOPPED_WITH_FAILURE',new_debits=len(starts),new_success=len(done),new_failures=len(errors),
            total_debits=b['debits']+len(starts),total_success=b['success']+len(done),historical_failure_classes=b['failure_classes'],
            measured_new_wall_s=sum(s['elapsed_s'] for s in sessions),charged_wall_s=used,remaining_wall_s=14400-used,wall_limit_s=14400,sessions=sessions,
            journal_events=len(events),journal_tail=tail,anchor_retry_receipts=sum(1 for _ in (folder/'io_receipts').glob('*.json')))
    for pair in pairs.values():
        if len(pair)==2:require(len(set(pair.values()))==1,'RL raw construction mismatch')
    require(states[FAILED_OLD]['state']=='HISTORICAL_INFRASTRUCTURE_FAILURE_MISSING' and states[FAILED_SCOPE]['state']=='HISTORICAL_FULL_PARENT_SCOPE_REJECTION_MISSING','Historical slot was refilled')
    analysis=analyze(thin,cases);analysis.update(scope='V3_PLUS_SCOPE_R1_ACTUAL_GRID_WITH_RETAINED_FAILURES',binding=binding,read_only_postprocessing=True,
        attribution='Common H/native deterministic search gains are separately recorded, not neural-network gains',actual_new_planning_calls=0)
    producer=RUN/'SCIENTIFIC_ANALYSIS_SCOPE_R1.json'
    if producer.exists():
        a=read(producer)
        for key in ('quality_cells','deployment_cells','coverage','independent_RL_pass','deployment_pass','missing_ids'):require(a[key]==analysis[key],'Independent closeout differs from frozen producer analysis: '+key)
    audit=dict(status='PASS',successful_results_hash_checked=len(records),unchanged_original_frozen_files=2515,
        assigned_arms=dict(assigned_arms),task_assignments_verified=sum(assigned_arms.values()),
        all_saved_plans_target_coverage_same_arm_and_endpoint_valid=True,input_cases_verified=len(inputs),input_semantics_stable_across_methods_seeds_versions=True,
        deterministic_case_seed_method_combinations=len(stable),RL_pairs_complete=sum(len(p)==2 for p in pairs.values()),RL_pairs_missing_counterpart=sum(len(p)!=2 for p in pairs.values()),
        historical_failed_slots_not_replayed=True,source_binding_sha256=binding['source_binding_sha256'])
    audit['formal_S3_042_navigation_D_matches_diagnostic_digest']=True
    audit['diagnostic_D_reused_as_runtime_cache']=False
    resource=dict(actual_by_seed=byseed,new_request_debits=sum(s['new_debits'] for s in byseed.values()),new_success=sum(s['new_success'] for s in byseed.values()),new_failures=sum(s['new_failures'] for s in byseed.values()),
        total_request_debits=sum(s['total_debits'] for s in byseed.values()),total_success=len(records),historical_failed_slots=2,
        not_executed=15876-sum(s['total_debits'] for s in byseed.values()),request_operation_counts=dict(ops),inclusive_component_seconds=dict(seconds),
        pipeline_counters=dict(pipeline),internal_search_entries=dict(internal),internal_search_inclusive_s=dict(inclusive),internal_search_exclusive_s=dict(exclusive),hot_prime_counts=dict(hot),
        common_H_by_method=dict(method_h),navigation_solves=ops['navigation_solves']+hot['navigation_solves'],
        charged_total_stage_wall_s=sum(s['charged_wall_s'] for s in byseed.values()),exact_original_stage_wall_s=None,
        old_wall_reservation_s=592,wall_rule='Retain exact2896.9277158999903 first-seed budget carry, including592 conservative historical reservation; add sealed new clocks. Nested component clocks not summed.',
        new_optimizer_updates=0,new_teacher_queries=0,original60_semantic_reads=0,new_diagnostic_navigation_outside_requests=0,
        source_binding_sha256=binding['source_binding_sha256'],read_only_analysis_new_model_navigation_native_H=0)
    result_dir.mkdir()
    with (result_dir/'REQUEST_METRICS.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    for name,value in [('SCIENTIFIC_ANALYSIS.json',analysis),('RESOURCE_ACCOUNTING.json',resource),('CONSTRAINT_SOURCE_AUDIT.json',audit),
        ('REQUEST_STATE_INDEX.json',dict(counts=dict(Counter(s['state'] for s in states.values())),requests=list(states.values()))),
        ('RESULT_ARTIFACT_INDEX.json',dict(artifacts=artifact_index)),('FAILURES.json',dict(new_request_failures=failures,historical_missing_ids=[FAILED_OLD,FAILED_SCOPE]))]:write_new(result_dir/name,value)
    write_new(result_dir/'MODEL_SOURCE_AUDIT.json',dict(status='PASS',fixed_model_source_files=model_files,models=12,
        original_model_index_sha256=digest(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json'),
        every_neural_request_matches_own_seed_fixed_endpoint=True,new_optimizer_updates=0))
    write_new(result_dir/'READ_ONLY_CLOSEOUT.json',dict(utc=now(),wall_s=time.perf_counter()-began,helper_sha256=digest(Path(__file__)),new_scientific_calls=0))
    print(dict(new_debits=resource['new_request_debits'],new_success=resource['new_success'],new_failures=resource['new_failures'],total_success=len(records),output=label(result_dir)))


if __name__=='__main__':collect()
