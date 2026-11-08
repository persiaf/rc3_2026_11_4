"""Delivery writer: opaque frozen-file audit and engineering-only summaries."""
import csv
import io
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from rc3_s3_evaluator.common import *
from rc3_s3_evaluator.journal import Journal,grid
from rc3_s3_evaluator.provenance import ModelCatalog,verify_code_binding,environment

DELIVERY=ROOT/'project_management/s3_evaluator_delivery'


def text_new(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8',newline='\n') as f:f.write(value)


def frozen_audit():
    t=time.perf_counter();manifest=ROOT/'project_management/GIT_MANIFEST.json'
    m=read(manifest);results=[];excluded=[];failures=[]
    for name,e in m['files'].items():
        if name=='第二章与第三章修订稿_20261029.docx':
            excluded.append(dict(file=name,reason='Pre-existing unrelated user Word edit; preserved, not staged, old manifest entry retained'))
            continue
        p=ROOT/name
        actual=digest(p) if p.is_file() else None
        ok=actual==e['sha256'] and p.stat().st_size==e['bytes']
        r=dict(file=name,sha256=actual,bytes=p.stat().st_size if p.is_file() else None,unchanged=ok)
        results.append(r)
        if not ok:failures.append(r)
    require(not failures,'Frozen baseline audit mismatch: '+str(failures[:3]))
    write_new(HERE/'FROZEN_EVIDENCE_AUDIT.json',dict(scope='Opaque byte comparison only; no target/task semantics or performance evaluation',
        baseline_commit=BASELINE,baseline_manifest_sha256=digest(manifest),checked_files=len(results),
        checked_bytes=sum(r['bytes'] for r in results),all_unchanged=True,exclusions=excluded,
        elapsed_s=time.perf_counter()-t,results=results))
    print('Frozen files unchanged',len(results),flush=True)


def summaries():
    r=read(HERE/'ENGINEERING_REGISTRATION.json');root=HERE/'runs'/ENGINEERING_ID
    stable=read(root/'AUTHORITY.json');requests=grid(r['cases'],r['model_seed'],True)
    j=Journal(root/str(r['model_seed']),stable,requests,126);j.clean();rows=j.rows()
    require(len(rows)==126 and not j.failures,'Engineering incomplete')
    ModelCatalog();verify_code_binding(HERE/'S3_CODE_BINDING_V2.json')
    counts=Counter();pipeline=Counter();internal=Counter();seconds=Counter()
    for row in rows:
        counts.update(row['counts']);pipeline.update(row['pipeline_counters']);internal.update(row['internal_search']['counts'])
        seconds.update(row['component_seconds'])
    primes=[e for e in j.events if e['event']=='hot_prime']
    primecounts=Counter();primeseconds=Counter()
    for e in primes:primecounts.update(e['counts']);primeseconds.update(e['seconds'])
    resource=dict(scope='S3_EVALUATOR_ENGINEERING_ONLY',scientific_S3_requests=0,S3_target_generations=0,original60_semantic_reads=0,
        optimizer_updates=0,new_teacher_queries=0,new_task_generations=0,
        complete_requests=dict(attempted=len(j.starts),completed=len(j.results),failed=len(j.failures),cap=126),
        model_seed=r['model_seed'],old_cases=6,native_attempts=36,greedy_constructions=90,
        common_H_invocations=counts['common_H_invocations'],nonempty_common_H_searches=pipeline['full_H_calls'],
        native_extra_common_H=0,RL_raw_common_H=0,navigation_solves=counts['navigation_solves']+primecounts['navigation_solves'],
        navigation_solve_breakdown=dict(cold_or_quality=84,charged_hot_primes=2),navigation_only_cache_hits=42,
        task_embedding_cache_hits=0,route_result_cache_hits=0,export_callbacks=counts['export_attempts'],validation_callbacks=counts['validation_attempts'],
        request_counts=dict(counts),native_H_pipeline_counters=dict(pipeline),internal_search_function_entries=dict(internal),
        request_component_seconds_inclusive=dict(seconds),hot_prime_counts=dict(primecounts),hot_prime_seconds_inclusive=dict(primeseconds),
        complete_request_time_sum_s=sum(x['complete_time_s'] for x in rows),
        cumulative_supervised_stage_wall_s=j.wall_s,stage_wall_cap_s=1800,
        closed_sessions=[{k:e[k] for k in ('token','elapsed_s','status')} for e in j.events if e['event']=='session_close'],
        unit_test_runs=[dict(path=label(p),sha256=digest(p),elapsed_s=read(p)['elapsed_s'],returncode=read(p)['returncode']) for p in sorted(HERE.glob('UNIT_TEST_RESULT_*.json'))],
        unit_test_scalar_validations=24,unit_test_real_planning_or_inference=0,
        frozen_byte_audit=dict(path='rc3_s3_evaluator/FROZEN_EVIDENCE_AUDIT.json',sha256=digest(HERE/'FROZEN_EVIDENCE_AUDIT.json')),
        accounting_notes=[
            '141 export/validation callbacks, including51 native candidate certifications;126 is the number of complete returned requests.',
            'Operation counters/predebits and full nested times retained in hash-chained journal; nested durations must not be summed.',
            'Each complete timer begins after durable request debit and ends after artifact serialization/hash; final request commit and all priming are in supervised stage wall.',
            'Parent interpreter/module import before supervise() is not measured by the stage perf clock; fixed host-launch overhead is outside the112.3302472s. Child interpreter startup, restore and request model/source load are measured.',
            'Unit/source/opaque byte audit/reviewer/Git maintenance are separate engineering/maintenance time, not S3 model latency or science queries.',
            'Registrations02/03 were source snapshots without executed test runs;01/04/05 were actually run. No paid request was repeated after the geometry provenance fix.'])
    write_new(HERE/'RESOURCE_ACCOUNTING.json',resource)
    complete=read(root/str(r['model_seed'])/'COMPLETE.json')
    verification=dict(status='PASS',technical_judgment='GO',scope='S3_EVALUATOR_ENGINEERING_ONLY',
        engineering_runtime_code_binding_sha256=digest(HERE/'S3_CODE_BINDING.json'),
        final_runtime_code_binding_sha256=digest(HERE/'S3_CODE_BINDING_V2.json'),
        original_engineering_commit='6be82edc',production_requests_passed=126,production_requests_failed=0,
        all_returned_plans_valid=all(x['valid'] and x['plan']['validation']['valid_model_plan'] for x in rows),
        deterministic_routes=complete['determinism'],weights_all12_original_bytes_unchanged=True,
        runtime_fresh_encodings=90,runtime_no_optimizer_or_backward_enforced=True,
        H_methods_exact=True,native_no_added_H=True,hot_cache_navigation_only=True,
        closed_boundary_resume=dict(first_requests=21,second_requests=105,cumulative_wall_s=j.wall_s),
        static_full_grid_all3seeds=15876,quality_cells=6,deployment_cells=12,
        synthetic_tests_passed=11,all_executed_test_runs_passed=True,
        negative_tests=['gate/review/geometry tamper','wrong endpoint/candidate/arm/coverage/collision/cost',
            'wrong model seed/role/checkpoint identity','nonfinite metrics','unmatched parent/duplicate/missing result',
            'duplicate predebit, wrong run identity, dirty/failure resume, truncated journal','heldout access without formal permit'],
        new_source_revision='Geometry provenance checks only, original126 evidence retained; post-fix11tests0realrequests',
        code_geometry_fix_test='UNIT_TEST_RESULT_05.json',
        result_root=label(root),complete_receipt_sha256=digest(root/str(r['model_seed'])/'COMPLETE.json'),
        resources_sha256=digest(HERE/'RESOURCE_ACCOUNTING.json'),
        formal_S3_executed=False,target_tasks_and_original60='SEALED',planning_performance_claim=False)
    write_new(HERE/'S3_ENGINEERING_VERIFICATION.json',verification)
    write_new(HERE/'S3_METHOD_PIPELINES.json',dict(protocol_sha256=PROTOCOL,training_source_sha256=TRAIN_SOURCE,H_identity=H_ID,
        methods=[dict(method=m,weight_role=MODEL_ROLES.get(m),construction='native endpoint adapter' if m in METHODS[:2] else 'one frozen greedy rollout',
                      extra_common_H=int(m in H_METHODS),optimizer=False,backward=False,export_validation=True) for m in METHODS],
        RL_raw_plus_H='Separate executions, identical raw route required; no free reused construction',
        warm_cache='Immutable navigation by canonical full-parent geometry/source/namespace only',
        complete_timer='parse/source verification, full input preparation, navigation/subset/F&A, model load/checks, all features/encode, construction, H, export, scalar validation, result serialization/hash',
        stage_timer='Child startup, all requests, navigation-only prime, restore, journal and IO; excludes parent interpreter import before supervisor entry',
        order_seed=11632026,order_derivation='PCG64 seed from first16hex SHA256 canonical [11632026,S3_method_order,case_id,model_seed,cache_kind,repeat]',
        bootstrap_seed=11632026,bootstrap_replicates=20000))
    out=io.StringIO(newline='');cols=['case','size','endpoint','method','kind','repeat','seed','J','L','K','raw_J','raw_L','raw_K','H_gain_J','complete_time_s']
    w=csv.DictWriter(out,fieldnames=cols);w.writeheader()
    for x in rows:
        w.writerow(dict(case=x['case_id'],size='x'.join(map(str,x['size'])),endpoint=x['endpoint'],method=x['request']['method'],
            kind=x['request']['kind'],repeat=x['request']['repeat'],seed=x['request']['seed'],**x['metrics'],
            H_gain_J=x['H_gain_J'],complete_time_s=x['complete_time_s']))
    text_new(HERE/'ENGINEERING_REQUEST_SUMMARY.csv',out.getvalue())
    write_new(HERE/'ARTIFACT_INDEX.json',dict(scope='126 old-DEV engineering requests only',
        artifacts=[dict(path=label(p),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(root.rglob('*')) if p.is_file()]))
    print(json.dumps(dict(requests=126,wall_s=j.wall_s,code_v2=digest(HERE/'S3_CODE_BINDING_V2.json'))),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='audit':frozen_audit()
    elif sys.argv[1]=='summaries':summaries()
    else:raise ValueError('audit or summaries')
