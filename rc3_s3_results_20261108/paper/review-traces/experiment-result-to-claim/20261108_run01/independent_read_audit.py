"""Reviewer-only saved-evidence audit: never invokes a task/model/planner/validator."""
from collections import Counter
from pathlib import Path
import datetime
import json

from rc3_s3_evaluator.common import ROOT, HERE, EXPERIMENT, identity, digest, read, write_new, label, now, require
from rc3_s3_evaluator.pipelines import verify_determinism


def main():
    run=HERE/'runs'/EXPERIMENT
    out=ROOT/'rc3_s3_results_20261108/paper/review-traces/experiment-result-to-claim/20261108_run01'
    before={label(p):digest(p) for p in run.rglob('*') if p.is_file()}
    seed=run/'11622026';journal=seed/'journal.jsonl'
    raw=journal.read_bytes()
    require(raw.endswith(b'\n'),'Journal tail has partial line')
    events=[];tail='0'*64
    for i,line in enumerate(raw.splitlines()):
        e=json.loads(line);h=e.pop('sha256')
        require(e['sequence']==i and e['previous']==tail and identity(e)==h,'Journal chain/hash/order mismatch')
        e['sha256']=h;events.append(e);tail=h
    starts={};completes={};errors=[];rows=[];request_context=None;final_counts={};prime_counts=Counter()
    for e in events:
        if e['event']=='request_start':
            request_context=e['request']['id'];require(request_context not in starts,'Duplicate debit')
            starts[request_context]=e['request']
        elif e['event']=='request_complete':
            rid=e['id'];require(rid in starts and rid not in completes,'Uncharged or repeated complete')
            p=seed/'results'/(rid+'.json');require(digest(p)==e['artifact_sha256'],'Result changed')
            r=read(p);require(r['request']==starts[rid] and r['valid'] is True,'Wrong returned record')
            require(r['plan']['validation']['valid_model_plan'] and r['plan']['validation']['all_tasks_served'],'Returned invalid certificate')
            rows.append(r);completes[rid]=e
            request_context=None
        elif e['event']=='request_error':
            require(e['id'] in starts and e['id'] not in completes,'Uncharged error')
            errors.append(e)
            final_counts[e['id']]=e['counts']
        elif e['event']=='operation_end' and request_context:
            # Major-operation counts are cumulative within a single request.
            final_counts[request_context]=e.get('counts',{})
        elif e['event']=='hot_prime':
            # Priming has its own meter; do not count these as request operations.
            prime_counts.update(e['counts'])
    counts=Counter();pipeline=Counter();internal=Counter()
    # Artifact row counters predate final serialization; last per-request operation meter
    # contains artifact completion as well. Both are retained, not double-counted.
    for rid in starts:counts.update(final_counts[rid])
    for r in rows:
        pipeline.update(r['pipeline_counters']);internal.update(r['internal_search']['counts'])
    head=read(seed/'HEAD.json');pending=read(seed/'HEAD.pending')
    require(head['sequence']<pending['sequence']<len(events),'Unexpected anchor layout')
    require(events[head['sequence']-1]['sha256']==head['sha256'],'HEAD not a valid prefix anchor')
    require(events[pending['sequence']-1]['sha256']==pending['sha256'],'Pending not a valid prefix anchor')
    failed=[r for rid,r in starts.items() if rid not in completes]
    require(len(failed)==len(errors)==1,'Unexpected failure count')
    f=failed[0]
    result_files=sorted((seed/'results').glob('*.json'))
    require(len(result_files)==len(completes),'Orphan result artifacts')
    session_open=[e for e in events if e['event']=='session_open'];session_close=[e for e in events if e['event']=='session_close']
    seeds={s:dict(directory_exists=(run/str(s)).exists(),complete_exists=(run/str(s)/'COMPLETE.json').exists()) for s in (11622026,11622027,11622028)}
    def file_utc(p):return datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat()
    report=dict(schema='rc3-s3-independent-failure-audit-v1',utc=now(),status='READ_ONLY_AUDIT_PASS_EXECUTION_INCOMPLETE',
        experiment_id=EXPERIMENT,seeds=seeds,chain=dict(events=len(events),all_hash_links_valid=True,tail_sha256=tail,
            HEAD=head,pending=pending,HEAD_matches_full_tail=False,HEAD_lag_events=len(events)-head['sequence'],pending_lag_events=len(events)-pending['sequence']),
        charged_requests=len(starts),committed_results=len(completes),request_error_events=len(errors),unclosed_requests=failed,
        request_distribution=dict(started_by_kind=dict(Counter(r['kind'] for r in starts.values())),
            completed_by_kind=dict(Counter(r['request']['kind'] for r in rows)),
            completed_by_method=dict(Counter(r['request']['method'] for r in rows)),
            completed_by_case=dict(Counter(r['case_id'] for r in rows))),
        returned_certificates_valid=len(rows),result_hashes_all_match=True,orphan_result_files=0,
        request_operation_counts=dict(counts),hot_prime_counts=dict(prime_counts),pipeline_counts=dict(pipeline),internal_search_entries=dict(internal),
        complete_request_time_sum_s=sum(e['complete_time_s'] for e in completes.values()),
        failed_request_recorded_elapsed_s=errors[0]['elapsed_s'],stage_monotonic_wall_s_exact=None,
        session_open=session_open,session_close=session_close,
        file_timestamps=dict(journal_mtime_utc=file_utc(journal),session_log_mtime_utc=[file_utc(p) for p in seed.glob('session_*.log')]),
        time_note='No session_close was persisted, so exact cumulative monotonic stage wall is unavailable. Do not report wall0 or use only completed request sums as stage wall.',
        failure_stage='After input_generation_and_preparation completion, while durably recording navigation_access entry; no navigation solver/native/model/H/route validation for failed request entered',
        determinism_saved_subset=verify_determinism(rows),
        real_auditor_requests=0,auditor_model_inferences=0,auditor_navigation=0,auditor_native=0,auditor_H=0,auditor_validator_calls=0,
        optimizer_updates=0,new_teacher_queries=0,original60_semantic_reads=0,
        scientific_quality_decision='NOT_EVALUATED_INCOMPLETE',deployment_decision='NOT_EVALUATED_INCOMPLETE')
    after={label(p):digest(p) for p in run.rglob('*') if p.is_file()}
    require(before==after,'Read-only audit changed original evidence')
    report['original_run_files_unchanged']=len(before)
    report['original_run_file_sha256']=before
    report['supersedes_reviewer_meter_tally']='READ_ONLY_AUDIT.json; first reviewer draft attributed hot-prime operation_end to previous request. Corrected by clearing request context after completion; original scientific journal/results unchanged.'
    write_new(out/'READ_ONLY_AUDIT_V2.json',report)
    print(json.dumps({k:report[k] for k in ['charged_requests','committed_results','request_error_events','request_distribution',
          'request_operation_counts','hot_prime_counts','pipeline_counts','complete_request_time_sum_s','failed_request_recorded_elapsed_s',
          'chain','failure_stage','determinism_saved_subset','original_run_files_unchanged']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
