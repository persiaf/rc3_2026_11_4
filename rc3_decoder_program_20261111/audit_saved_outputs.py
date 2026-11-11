"""Independent saved-output arithmetic/byte audit. Never constructs a model."""
import time
START=time.perf_counter()
import csv
import gzip
import math
import subprocess
from .common import HOME,ROOT,BASE,sha,read,put,csvrow

def records(path):
    import json
    with gzip.open(path,'rt',encoding='utf8') as f:return [json.loads(line) for line in f if line.strip()]

def main():
    binding=read(HOME/'SAVED_OUTPUT_AUDIT_BINDING.json')
    if sha(__file__)!=binding['audit_source_sha256']:raise ValueError('Saved-output auditor source changed')
    original=records(HOME/'P0/REQUESTS.jsonl.gz');repair=records(HOME/'P0_ACCOUNTING_REPAIR/REQUESTS.jsonl.gz');all_records=original+repair
    updates=records(HOME/'P0/UPDATES.jsonl.gz');assert len(original)==13 and len(repair)==1 and len(updates)==5
    checks=0
    def equal(a,b,tol=1e-7):
        nonlocal checks
        assert abs(a-b)<=tol,(a,b);checks+=1
    for r in all_records:
        m=r['metrics'];h=r['plan'];raw=r['raw_plan']
        for plan,pre in ((raw,'raw_'),(h,'')):
            equal(m[pre+'J'],m[pre+'L']+.75*m[pre+'K'])
            equal(m[pre+'L'],plan['route_length_m']);equal(m[pre+'J'],plan['objective_length_plus_stops_m'])
            equal(m[pre+'K'],len(plan['visit_order_candidate_ids']))
            assert plan['validation']['valid_model_plan'] and plan['unserved_task_ids']==[]
            assert sorted(a['task_id'] for a in plan['assignments'])==sorted(plan['target_task_ids_global']);checks+=2
        equal(r['complete_return'],-m['J']);equal(math.fsum(r['environment_rewards']),-m['J'])
        assert all(mask[action] for mask,action in zip(r['action_masks'],r['actions']));checks+=1
        H=int(bool(r['raw_route'][1:-1]));known=r in repair
        if known:assert r['counts']['full_H_calls']==H and r['internal_search']['counts']
        csvrow(HOME/'CORRECTED_REQUEST_RESULTS.csv',dict(request_id=r['request_id'],role=r['method'],width=r['spec']['width'],height=r['spec']['height'],
            **m,valid=True,actual_H_calls=H,H_source='Actual persisted R02 counter' if known else 'Frozen terminal.finish exactly one H on nonempty raw route; original writer suppressed field',
            internal_search_counts=r['internal_search']['counts'] if known else None,
            per_request_internal_search_status='KNOWN_ACTUAL' if known else 'UNKNOWN_NOT_PERSISTED; not zero',
            raw_stop_count=len(raw['visit_order_candidate_ids']),H_stop_count=len(h['visit_order_candidate_ids'])))
    index={r['request_id']:r for r in original}
    for i in range(4):
        a=index['p0:init:'+str(i)];b=index['p0:reference:'+str(i)]
        assert a['raw_route']==b['raw_route'] and a['route']==b['route'] and a['metrics']==b['metrics'];checks+=3
    old=read(HOME/'P0/RESOURCE_ACCOUNTING.json');new=read(HOME/'P0_ACCOUNTING_REPAIR/RESOURCE_ACCOUNTING.json')
    assert new['counts']['requests']==15 and new['counts']['requests_completed']==14 and new['counts']['requests_failed']==1
    assert new['counts']['actor_optimizer_attempts']==5 and new['counts']['full_H_calls']==15;checks+=5
    request_score=sum(len(r['actions']) for r in original)
    assert old['counts']['original_forward_calls']==len(original)+request_score
    replay_routes=sum(len(u['loss_sources']) for u in updates);replay_scores=sum(s['action_count'] for u in updates for s in u['loss_sources'])
    init_scores=sum(len(index[f'p0:init:{i}']['actions']) for i in range(4))
    probe_scores=min(2,len(index['p0:pg_sample']['actions']))
    direct_scores=3*init_scores+probe_scores+2*len(index['p0:live_export_check']['actions'])
    direct_encodes=3*4+2+2
    macro=dict(unit='Top-level policy.encode and policy.score calls; nested override/super dispatch is counted once; not Torch operator/kernel FLOPs',
        P0_request_encode_calls=13,P0_request_score_calls=request_score,
        P0_training_replay_encode_calls=replay_routes,P0_training_replay_score_calls=replay_scores,
        P0_direct_engineering_encode_calls=direct_encodes,P0_direct_engineering_score_calls=direct_scores,
        P0_successful_encode_total=13+replay_routes+direct_encodes,P0_successful_score_total=request_score+replay_scores+direct_scores,
        P0_stale_score_rejected_before_tensor_forward=1,
        accounting_repair_encode_calls=1,accounting_repair_score_calls=len(repair[0]['actions']),
        precursor_forward_calls=None,precursor_forward_status='UNKNOWN_NOT_PERSISTED; not assumed zero',
        original_483_field_scope='Request encode+score only, excludes training/direct engineering',
        original_cache_rebuilds_field_scope='Request plus Trainer encode only, excludes direct engineering',
        reconstruction_source='Frozen engineering.py loop structure plus saved request actions and UPDATES.loss_sources.action_count; no model forward')
    put(HOME/'CORRECTED_FORWARD_ACCOUNTING.json',macro)
    # Read tensor payloads only. No GlobalLocalPolicy construction, encode,
    # score, trajectory generation, teacher, navigation, or H invocation.
    import torch
    base_path=ROOT/'rc3_integrated_rppo/runs/rc3_integrated_rppo_v1_20261110_01/actors/11622026/SL256_0256_1817f58aa650.pt'
    assert sha(base_path)=='60f5aa4e59983e92d565f0599823b710965cf2c96a041e7eb483d23a8eb0556c'
    base=torch.load(base_path,map_location='cpu',weights_only=True)['model']
    ep=HOME/'P0/models/D5-PG.engineering.actor.pt';cp=HOME/'P0/models/D5-PG.engineering.training.pt'
    exported=torch.load(ep,map_location='cpu',weights_only=True);checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
    allowed={'key.weight','dynamic.weight','nav.weight','query.weight','query.bias'}
    for name,tensor in exported['model'].items():
        if name not in allowed:assert torch.equal(tensor,base[name]);checks+=1
        assert torch.equal(tensor,checkpoint['model'][name]);checks+=1
    assert exported['engineering_only'] and checkpoint['engineering_only'] and not exported['critic_included'];checks+=3
    models=[dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size,role='D5-PG',usage='engineering_only',scientific_initialization_allowed=False) for p in (ep,cp)]
    put(HOME/'MODEL_INDEX.json',dict(formal_scientific_weights='NOT_RUN',models=models,legal_SL_initialization=dict(path=base_path.relative_to(ROOT).as_posix(),sha256=sha(base_path))))
    frozen=['rc3_global_local/policy.py','rc3_global_local/environment.py','rc3_global_local/terminal.py','rc3_global_local_trainer/engine.py',
        'rc3_task_level_decision_20261111/TASK_LEVEL_COMPARISON.csv','rc3_task_level_decision_20261111/TASK_LEVEL_COMPARISON.md',
        'rc3_task_level_decision_20261111/FIXED32_CLOSURE.json','rc3_task_level_decision_20261111/FIXED32_CLOSURE.md']
    subprocess.run(['git','diff','--quiet',BASE,'--',*frozen],cwd=ROOT,check=True)
    put(HOME/'HISTORY_PRESERVATION_CHECK.json',dict(status='PASS',base_commit=BASE,unchanged_paths=frozen,
        old_diagnostic_weights_not_loaded=True,teacher_brief_only_authorized_additive_history_correction=True,
        original60_and_oldS3_semantic_reads=0,check='Git diff/object identity only; no historical CSV audit or sealed task interpretation'))
    timings=[]
    for path in [HOME/'P0/REQUEST_TIMING.csv',HOME/'P0_ACCOUNTING_REPAIR/REQUEST_TIMING.csv']:
        with path.open(encoding='utf8',newline='') as f:timings.extend(csv.DictReader(f))
    time_index={x['request_id']:float(x['full_request_s']) for x in timings}
    # Engineering-only throughput description, never substitutes for a
    # science schedule/whole-stream throughput admission.
    by_size={}
    for r in original:
        size=f"{r['spec']['width']}x{r['spec']['height']}";by_size.setdefault(size,[]).append(time_index[r['request_id']])
    put(HOME/'ENGINEERING_THROUGHPUT.json',dict(complete_request_s_by_size=by_size,
        formal_4096_stream_throughput_admission='NOT_REACHED_DATA_GATE_BLOCKED',
        limitation='Few ordinary engineering tasks; no complete 8-task logical batch throughput or full training distribution measured. Do not infer 8h feasibility.'))
    put(HOME/'INDEPENDENT_NUMERICAL_REVIEW.json',dict(status='PASS_WITH_EXPLICIT_RECORDING_LIMITATIONS',checks=checks,
        new_output_records_checked=14,updates_read=5,model_payloads_compared=2,
        model_forward_calls=0,optimizer_attempts=0,planning_requests=0,teacher_queries=0,navigation_calls=0,H_calls=0,
        original_per_request_search_for_13_requests='UNKNOWN_NOT_PERSISTED',
        actual_aggregate_new_search_counts=new['counts']['internal_search_counts'],precursor_search_counts=None,
        source_binding_sha256=sha(HOME/'SAVED_OUTPUT_AUDIT_BINDING.json'),elapsed_s=time.perf_counter()-START))
    print('Saved-output review PASS with recording limitations:',checks,'checks;14 records;zero new model forward/planning/optimizer.')

if __name__=='__main__':main()
