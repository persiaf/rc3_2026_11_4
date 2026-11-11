"""Saved-evidence presentation and metadata only; no numerical model imports."""
import time
START=time.perf_counter()
import csv
import json
import gzip
from .common import HOME,ROOT,read,put,sha,csvrow

def records(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return [json.loads(x) for x in f if x.strip()]

def main():
    original=records(HOME/'P0/REQUESTS.jsonl.gz');repair=records(HOME/'P0_ACCOUNTING_REPAIR/REQUESTS.jsonl.gz')
    references={tuple(r['spec'][k] for k in ('map_seed','task_seed','width','height','x0','y0','A_condition','endpoints')):r for r in original if r['method']=='FrozenSL'}
    timings={}
    for p in (HOME/'P0/REQUEST_TIMING.csv',HOME/'P0_ACCOUNTING_REPAIR/REQUEST_TIMING.csv'):
        with p.open(encoding='utf8',newline='') as f:timings.update({r['request_id']:float(r['full_request_s']) for r in csv.DictReader(f)})
    for r in original+repair:
        key=tuple(r['spec'][k] for k in ('map_seed','task_seed','width','height','x0','y0','A_condition','endpoints'));base=references[key]
        changes={}
        for name,p in [('raw','raw_plan'),('H','plan')]:
            a=r[p]['visit_order_candidate_ids'];b=base[p]['visit_order_candidate_ids']
            changes[name+'_stop_set_changed']=set(a)!=set(b);changes[name+'_visit_sequence_changed']=a!=b
            changes[name+'_same_set_order_changed']=set(a)==set(b) and a!=b
        csvrow(HOME/'ENGINEERING_REQUEST_SUMMARY.csv',dict(stage='ENGINEERING_ONLY_NOT_TASK_BENEFIT',request_id=r['request_id'],method=r['method'],
            width=r['spec']['width'],height=r['spec']['height'],physical_key=r['physical_key'],**r['metrics'],success=True,full_request_s=timings[r['request_id']],**changes))
    allow=read(HOME/'PROGRAM_PROTOCOL.json')['model']
    put(HOME/'PARAMETER_ALLOWLIST.json',dict(allowlist=allow['allowlist'],shapes={'key.weight':[64,64],'dynamic.weight':[64,64],
        'nav.weight':[64,4],'query.weight':[64,260],'query.bias':[64]},trainable_scalars=25152,total_scalars=65088,frozen_scalars=39936,
        forbidden_initializations=['fixed32 diagnostic weights','P0 engineering weights','CB/CG critics'],optimizer_identity='Exactly owned five parameter objects, no duplicates/foreign tensors'))
    resources=read(HOME/'P0_ACCOUNTING_REPAIR/RESOURCE_ACCOUNTING.json');entry=read(HOME/'ACCOUNTING_REPAIR_ENTRY_CONSUMPTION.json');audit=read(HOME/'INDEPENDENT_NUMERICAL_REVIEW.json')
    charged=entry['cumulative_wall_s']+6.191302
    put(HOME/'RESOURCE_ACCOUNTING.json',dict(status='P6_TECHNICAL_DATA_INCOMPLETE',budget=read(HOME/'PROGRAM_PROTOCOL.json')['budget'],
        counts=resources['counts'],complete_requests=dict(attempted=15,completed=14,failed=1,new_this_program_completed=14,carried_precursor_failed=1),
        Actor=dict(actual_attempts=5,completed=5,formal_scientific_attempts=0),Critic_updates=0,
        teacher_label_replays=2,new_teacher_queries=0,new_scene_generations=0,navigation_solves=0,native_requests=0,
        original60_semantic_reads=0,oldS3_semantic_reads=0,
        forward_accounting=read(HOME/'CORRECTED_FORWARD_ACCOUNTING.json'),
        search_accounting=dict(known_new_aggregate_counts=resources['counts']['internal_search_counts'],precursor_search_counts=None,
            per_request_13_original='UNKNOWN_NOT_PERSISTED',repair_request='KNOWN_ACTUAL',unknown_is_not_zero=True),
        wall=dict(through_accounting_repair_controller_s=entry['cumulative_wall_s'],saved_output_audit_worker_s=audit['elapsed_s'],
            saved_output_audit_whole_job_s=6.191302,charged_compute_job_wall_s_before_metadata_closeout=charged,
            metadata_closeout_wall_s_at_receipt=time.perf_counter()-START,
            P0_previous_failure_included=True,registration_included_in_controllers=True,
            scope='Actual supervised entry jobs plus saved-output audit whole job; prose/reviewer narrative and Git/network maintenance separate, not scientific training'),
        scientific=dict(P1='NOT_RUN',P2='NOT_RUN',P3='NOT_RUN',P4='NOT_RUN',P5='NOT_RUN',results=None,weights=None),
        no_resource_reset=True,old_resources_unchanged=True))
    status=read(HOME/'STATUS.json');status.update(phase='P6',status='P6_TECHNICAL_DATA_INCOMPLETE',
        source_binding_sha256=sha(HOME/'SOURCE_BINDING_R02.json'),P0_execution_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),
        accounting_repair_binding_sha256=sha(HOME/'SOURCE_BINDING_R02.json'),readonly_audit_binding_sha256=sha(HOME/'SAVED_OUTPUT_AUDIT_BINDING.json'),
        logical_scientific_updates=0,actual_Actor_attempts=5,full_requests_attempted=15,full_requests_completed=14,
        scientific_results=None,formal_weights=None,closed=True,stop_reason='Exact legal teacher pool lacks full10x10 labels for required same-task PG/SL schedule',
        next_action='Publish current technical/data closure; no automatic continuation or new teachers')
    put(HOME/'STATUS.json',status,replace=True)
    trace=HOME/'paper/review-traces/experiment-result-to-claim/2026-11-11_run01'
    verdict=dict(claim_supported='partial',confidence='high',supports='Bounded ordinary-task original Actor five-tensor engineering',
        does_not_support='Task benefit, freezing attribution, matched compute, three seeds, unseen layouts, deployment speed',
        missing_evidence='Legal full10x10 labels, identical common schedule, scientific driver and throughput admission',
        next_experiments_needed_within_current_boundary=[],routing='P6 technical data incomplete; do not auto change data or run diagnostics',
        paper_claim_audit='unavailable',writing_verdict='provisional')
    put(trace/'parsed_verdict.json',verdict);put(trace/'routing_action.json',dict(action='CLOSE_CURRENT_PROGRAM_AT_P6',automatic_new_experiments=False))
    put(trace/'status.json',dict(reviewer='Independent paper architect/reviewer sub-agent',pending_external_review=False,review_complete=True,scientific_claim_confirmed=False))
    plan=(HOME/'MASTER_PLAN.md').read_bytes()
    for name in ('2026-11-11_EXPERIMENT_PLAN.md','EXPERIMENT_PLAN.md'):
        with (HOME/'paper'/name).open('xb') as f:f.write(plan)
    put(HOME/'SCIENTIFIC_RESULTS_STATUS.json',dict(P1='NOT_RUN',P2='NOT_RUN',P3='NOT_RUN',P4='NOT_RUN',P5='NOT_RUN',
        FrozenSL=None,D5_SL_U=None,D5_SL_C=None,D5_PG=None,Full_PG=None,Fixed3=None,Detour16=None,
        training_weights=None,task_level_route_benefit=None,independent_generalization=None,deployment_speed=None))
    print('Metadata, engineering request table, truthful NOT_RUN scientific status, review traces and final resource ledger written.')

if __name__=='__main__':main()
