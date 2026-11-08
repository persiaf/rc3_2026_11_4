"""Read-only incident audit; no generation, policy load, planning or evaluation."""
from collections import Counter
from datetime import datetime as DateTime
import csv
import io
import math
from pathlib import Path
from rc3_s3_evaluator.common import *
from rc3_s3_evaluator.provenance import ModelCatalog, verify_code_binding, verify_training_anchors

OUT=ROOT/'rc3_s3_results_20261108'
RUN=HERE/'runs'/EXPERIMENT


def main():
    observed=now()
    verify_training_anchors();verify_code_binding(HERE/'S3_CODE_BINDING_V2.json');ModelCatalog()
    folder=RUN/str(SEEDS[0]);jp=folder/'journal.jsonl'
    events=[];tail='0'*64
    for line in jp.read_text(encoding='utf8').splitlines():
        e=__import__('json').loads(line);h=e.pop('sha256')
        require(e['sequence']==len(events) and e['previous']==tail and identity(e)==h,'Invalid persisted journal chain')
        e['sha256']=h;events.append(e);tail=h
    starts={};complete={};errors=[];rows=[];latest={};prime_counts=Counter()
    operations=Counter();seconds=Counter();current=None
    for e in events:
        if e['event']=='request_start':
            q=e['request'];require(q['id'] not in starts,'Duplicate request debit');starts[q['id']]=q;current=q['id']
        elif e['event']=='request_complete':
            require(e['id'] in starts and e['id'] not in complete,'Duplicate/unbilled completion')
            path=folder/'results'/(e['id']+'.json')
            require(digest(path)==e['artifact_sha256'],'Changed result artifact')
            r=read(path);require(r['request']==starts[e['id']] and r['valid'] and r['plan']['validation']['valid_model_plan'],'Invalid stored result')
            r['complete_time_s']=e['complete_time_s'];rows.append(r);complete[e['id']]=e;current=None
        elif e['event']=='request_error':errors.append(e);latest[e['id']]=e['counts'];current=None
        elif e['event']=='operation_end':
            operations[e['operation']]+=1;seconds[e['operation']]+=e['elapsed_s']
            if current is not None:latest[current]=e['counts']
        elif e['event']=='hot_prime':prime_counts.update(e['counts'])
    counts=Counter();pipeline=Counter();internal=Counter()
    for rid,c in latest.items():counts.update(c)
    for r in rows:pipeline.update(r['pipeline_counters']);internal.update(r['internal_search']['counts'])
    head=read(folder/'HEAD.json');pending=read(folder/'HEAD.pending')
    for value in (head,pending):
        n=value['sequence'];require(n>0 and events[n-1]['sha256']==value['sha256'],'Anchor not a real journal prefix')
    raw={};stable={};pairs={}
    for r in rows:
        q=r['request'];key=(r['case_id'],q['seed'],q['method']);v=(r['raw_route_sha256'],r['route_sha256'],r['metrics'])
        require(key not in stable or stable[key]==v,'Cross-repeat route/cost drift');stable[key]=v
        if q['method'] in ('RL_raw','RL_plus_H'):
            k=(r['case_id'],q['seed']);require(k not in raw or raw[k]==r['raw_route_sha256'],'RL raw route mismatch');raw[k]=r['raw_route_sha256']
            k=(r['case_id'],q['seed'],q['kind'],q['repeat']);pairs.setdefault(k,{})[q['method']]=r['raw_route_sha256']
    require(all(len(set(p.values()))==1 for p in pairs.values()),'RL raw pair mismatch')
    session=next(e for e in events if e['event']=='session_open')
    admission=read(OUT/'ADMISSION_RECEIPT.json')
    # Earlier admission preceded launch; later observation followed process exit.
    # This is an intentionally conservative resource reservation, NOT exact run time.
    reserve=math.ceil((DateTime.fromisoformat(observed)-DateTime.fromisoformat(admission['utc'])).total_seconds())
    row_counts=Counter(r['request']['method'] for r in rows);kinds=Counter(r['request']['kind'] for r in rows)
    files=[dict(path=label(p),sha256=digest(p),bytes=p.stat().st_size) for p in sorted(RUN.rglob('*')) if p.is_file()]
    audit=dict(status='PASS_READ_ONLY_EVIDENCE_AUDIT',observed_after_exit_utc=observed,experiment_id=EXPERIMENT,
        first_seed=SEEDS[0],journal_chain_valid=True,journal_events=len(events),journal_tail_sha256=tail,
        head_json=head,head_pending=pending,head_lag_events=len(events)-head['sequence'],
        head_pending_lag_events=len(events)-pending['sequence'],head_is_current=False,
        scientific_requests_debited=len(starts),completed=len(complete),failed=len(errors),
        incomplete_ids=sorted(set(starts)-set(complete)),failed_records=errors,
        completed_by_method=dict(row_counts),completed_by_kind=dict(kinds),unique_target_cases_started=len({q['case_id'] for q in starts.values()}),
        all_completed_plans_valid=True,returned_valid_plans=len(rows),deterministic_case_seed_methods=len(stable),
        RL_raw_plus_H_complete_pairs=sum(set(v)=={'RL_raw','RL_plus_H'} for v in pairs.values()),
        missing_result_artifacts_for_completed_requests=0,scientific_updates=0,new_teacher_queries=0,
        source_and_all12_checkpoint_hashes_unchanged=True,approved_source_binding_sha256=digest(HERE/'S3_SOURCE_BINDING.json'),
        subsequent_seeds_started={str(s):(RUN/str(s)).exists() for s in SEEDS[1:]},
        complete_receipt_present=(folder/'COMPLETE.json').exists(),formal_analysis_present=(RUN/'FORMAL_ANALYSIS.json').exists(),
        original60_semantic_reads=0,statistical_decision='NOT_EVALUATED_INCOMPLETE_GRID')
    write_new(OUT/'SAVED_RUN_AUDIT.json',audit)
    write_new(OUT/'ARTIFACT_INDEX.json',dict(scope='Original interrupted formal S3 artifacts, unmodified',artifacts=files))
    receipt=dict(status='STOPPED_DIRTY_JOURNAL_NO_AUTOMATIC_RESUME',observed_utc=observed,
        user_authorization_sha256=digest(OUT/'authorizations/S3_AUTHORIZATION.json'),
        approved_evaluation_source_binding_sha256=digest(HERE/'S3_SOURCE_BINDING.json'),
        original_head_and_pending_preserved=True,original_journal_preserved=True,
        source_error='WinError5 during HEAD.pending -> HEAD.json os.replace; secondary FileExistsError during request_error append; parent HEAD consistency rejection prevented session_close',
        original_journal_sha256=digest(jp),request_debits=len(starts),completed=len(rows),failed=len(errors),
        session_open_utc=session['utc'],session_closed=False,stage_wall_s_exact=None,
        known_completed_request_time_sum_s=sum(r['complete_time_s'] for r in rows),
        conservative_carry_forward_reservation_s=reserve,
        reservation_basis='ceil(observed_after_process_exit_utc - earlier_prelaunch_ADMISSION_RECEIPT.utc); includes approval/Git wait and postfailure observation delay, deliberately not a measured stage wall',
        no_budget_reset=True,remaining_wall_after_provisional_reservation_s=max(0,14400-reserve),
        recovery_authorized=False,next_seed_authorized_to_skip_predecessor=False)
    write_new(OUT/'EXTERNAL_FAILURE_RECEIPT.json',receipt)
    resources=dict(scope='ACTUAL_FORMAL_S3_INTERRUPTED',experiment_id=EXPERIMENT,
        planned_requests_per_seed=5292,planned_requests_all_seeds=15876,
        actual_by_seed={str(s):dict(debited=len(starts) if s==SEEDS[0] else 0,completed=len(rows) if s==SEEDS[0] else 0,
            failed=len(errors) if s==SEEDS[0] else 0,status='STOPPED_IO_FAILURE' if s==SEEDS[0] else 'NOT_STARTED') for s in SEEDS},
        total_request_debits=len(starts),complete_requests=len(rows),failed_requests=len(errors),not_executed_requests=15876-len(starts),
        optimizer_updates=0,new_teacher_queries=0,original60_semantic_reads=0,
        request_operation_counts=dict(counts),all_logged_operation_end_seconds_inclusive=dict(seconds),
        pipeline_counters=dict(pipeline),internal_search_function_entries=dict(internal),hot_prime_counts=dict(prime_counts),
        navigation_solves_total=counts['navigation_solves']+prime_counts['navigation_solves'],
        complete_request_time_sum_s=sum(r['complete_time_s'] for r in rows),stage_wall_s_exact=None,
        missing_exact_wall_reason='Parent journal consistency guard raised before session_close; not replaced with zero',
        conservative_frozen_wall_reservation_s=reserve,conservative_reservation_basis=receipt['reservation_basis'],
        partial_data_rule='No six/twelve-cell inferential decisions, no complete-grid performance claims; no dropped or replaced cases',
        postmortem_new_real_generation_navigation_native_H_inference=0,
        original_failure_receipt_sha256=digest(OUT/'EXTERNAL_FAILURE_RECEIPT.json'))
    write_new(OUT/'RESOURCE_ACCOUNTING.json',resources)
    out=io.StringIO();w=csv.DictWriter(out,fieldnames=['case','group','size','endpoint','seed','method','kind','repeat','J','L','K','raw_J','raw_L','raw_K','H_gain_J','complete_time_s'])
    w.writeheader()
    for r in rows:
        q=r['request'];w.writerow(dict(case=r['case_id'],group=r['group'],size='x'.join(map(str,r['size'])),endpoint=r['endpoint'],
            seed=q['seed'],method=q['method'],kind=q['kind'],repeat=q['repeat'],**r['metrics'],H_gain_J=r['H_gain_J'],complete_time_s=r['complete_time_s']))
    with (OUT/'PARTIAL_REQUESTS.csv').open('x',encoding='utf8',newline='') as f:f.write(out.getvalue())
    print(__import__('json').dumps({k:audit[k] for k in ['scientific_requests_debited','completed','failed','unique_target_cases_started','RL_raw_plus_H_complete_pairs']}))
    print(__import__('json').dumps(resources,indent=2))


if __name__=='__main__':main()
