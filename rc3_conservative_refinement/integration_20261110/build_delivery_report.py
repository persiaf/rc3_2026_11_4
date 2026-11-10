"""Maintenance: summarize immutable local engineering records; no ML imports."""
from pathlib import Path
import csv
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_new(name,value):
    with (HERE/name).open('x',encoding='utf8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def csv_new(name,rows):
    with (HERE/name).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def main():
    attempts=[];rows=[];totals={};wall=0.
    for folder in sorted(HERE.glob('engineering_R*')):
        r=read(folder/'RESOURCE_ACCOUNTING.json')
        events=[json.loads(line) for line in (folder/'events.jsonl').read_bytes().splitlines()]
        seconds=sum(e['elapsed_s'] for e in events if e['event']=='session_close')
        wall+=seconds
        attempts.append(dict(attempt=folder.name,status=r['status'],counts=r['counts'],
                             closed_wall_s=seconds,source_binding_sha256=r['source_binding_sha256'],
                             resource_path=(folder/'RESOURCE_ACCOUNTING.json').relative_to(ROOT).as_posix()))
        for k,n in r['counts'].items():
            totals[k]=totals.get(k,0)+n
        for event in events:
            if event['event']!='request_done':
                continue
            path=ROOT/event['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=event['artifact_sha256']:
                raise ValueError('Saved request artifact changed')
            d=read(path);m=d['metrics'];native=event['request_id'].startswith('native/')
            rows.append(dict(attempt=folder.name,request_id=event['request_id'],
                             input_identity=d['input_identity'],J=m['J'],L=m['L'],K=m['K'],
                             raw_J=m.get('raw_J',m['J']),raw_L=m.get('raw_L',m['L']),raw_K=m.get('raw_K',m['K']),
                             valid=d['plan']['validation']['valid_model_plan'],
                             raw_valid=d.get('raw_plan',d['plan'])['validation']['valid_model_plan'],
                             raw_semantics='Native final, no extraH' if native else 'Independent raw+H plans',
                             request_before_IO_s=d.get('service_time_before_IO_s',d.get('request_before_IO_s')),
                             request_through_record_s=event['elapsed_through_record_s'],
                             model_startup_s=d.get('model_startup_s',0.),
                             source=event['path']))
    csv_new('ENGINEERING_ROUTE_TABLE.csv',rows)
    updates=read(HERE/'engineering_R03/UPDATE_REPORTS.json');urows=[]
    for key,r in updates.items():
        if 'accepted' not in r:
            continue
        urows.append(dict(case=key,accepted=r['accepted'],rollback_exact=r['rollback_exact'],
                          actor_loss=r['actor_loss'],critic_loss=r.get('critic_loss'),
                          actor_gradient_norm=r['actor_gradient_norm'],critic_gradient_norm=r.get('critic_gradient_norm'),
                          advantage_mean=r.get('advantage_mean'),advantage_rms=r.get('advantage_rms'),
                          max_step_kl=r['guard_maxima']['max_step_kl'],
                          max_probability_change=r['guard_maxima']['max_step_probability_change'],
                          max_base_TV=r['guard_maxima']['max_base_total_variation']))
    csv_new('ENGINEERING_UPDATE_SUMMARY.csv',urows)
    write_new('RESOURCE_ACCOUNTING.json',dict(status='ENGINEERING_PASS_FORMAL_NOT_STARTED',attempts=attempts,
              cumulative_engineering_counts=totals,closed_engineering_wall_s=wall,
              requests_completed=len(rows),requests_failed=totals.get('requests',0)-len(rows),
              absolute_engineering_caps=dict(requests=32,actor_optimizer_attempts=16,critic_optimizer_attempts=16,teacher_episodes=2,wall_s=1800),
              engineering_weight_reused_for_science=False,formal_actor_updates=0,formal_critic_updates=0,
              formal_requests=0,original60_semantic_reads=0,S3_semantic_reads=0,new_tasks=0,new_navigation_solves=0,
              independent_saved_audit='Separate registered read-only check, cap30s; see INDEPENDENT_SAVED_PLAN_AUDIT.json',
              maintenance_scope='Registration, dependency/source/LFS/Git verification and delivery summaries separately identified; no new training or planning'))
    protocol=read(HERE/'PROTOCOL_DRAFT_R03.json')
    write_new('STAGE_STATUS.json',dict(status='REAL_ENGINEERING_PASS_WAITING_EXPLICIT_PARAMETER_APPROVAL',
              source_binding_sha256=protocol['source_binding_sha256'],
              protocol_path=(HERE/'PROTOCOL_DRAFT_R03.json').relative_to(ROOT).as_posix(),
              protocol_sha256=hashlib.sha256((HERE/'PROTOCOL_DRAFT_R03.json').read_bytes()).hexdigest(),
              protocol_approved=False,stage_execution_authorized=False,formal_training='NOT_STARTED',
              ON='NOT_STARTED',OFF='NOT_STARTED',matched_SL='NOT_STARTED',quality='NOT_STARTED',
              critic_calibration='NOT_STARTED',full_timing_comparison='NOT_STARTED',
              engineering='PASS',old_active_protocol_unchanged=True,old_authorization_reused=False))


if __name__=='__main__':
    main()
