"""Strict read-only import of both old S3 segments; no request refunds/replays."""
from rc3_s3_evaluator.common import ROOT,SEEDS,EXPERIMENT,read,digest,identity,label,require,write_new
from rc3_s3_evaluator.inputs import FormalInputs
from rc3_s3_evaluator.journal import grid
from rc3_s3_recovery_20261108.authority import admit as old_admit,CONT as OLD_CONT
from rc3_s3_recovery_20261108.storage import read_chain
from .scan import HOME

FAILED_OLD='68e89307acb21cdff0501aa2c6531990909421d266140a88943876f8fe4b0ea9'
FAILED_SCOPE='88c202d54990cee9fe80d3b8a01fba6c7275a98069a822d3c25e9b701d7eff68'
RUN=HOME/'runs'/EXPERIMENT/'scope_r1'


def frozen():
    index=read(HOME/'FROZEN_EVIDENCE_INDEX.json')
    for name,e in index['files'].items():
        p=ROOT/name;require(p.is_file() and p.stat().st_size==e['bytes'] and digest(p)==e['sha256'],'Frozen evidence differs: '+name)
    return len(index['files'])


def import_ledger():
    frozen();permit,legacy,old_binding=old_admit()
    cases=FormalInputs(permit,RUN).cases
    folder=OLD_CONT/str(SEEDS[0]);events,tail=read_chain(folder/'journal.jsonl')
    require(len(events)==46023 and events[-1]['event']=='session_close' and events[-1]['status']=='FAILED','Unknown old continuation boundary')
    head=read(folder/'HEAD.json');require(head['sequence']==len(events) and head['sha256']==tail and not (folder/'HEAD.pending').exists(),'Old continuation anchor unclosed')
    starts={e['request']['id']:e['request'] for e in events if e['event']=='request_start'}
    results={e['id']:e for e in events if e['event']=='request_complete'}
    errors={e['id']:e for e in events if e['event']=='request_error'}
    require(len(starts)==1896 and len(results)==1895 and set(errors)=={FAILED_SCOPE} and set(starts)==set(results)|set(errors),'Old charged state differs')
    require(not (set(starts)&set(legacy['request_starts'])),'Old segment replay')
    for rid,e in results.items():
        p=folder/'results'/(rid+'.json');require(digest(p)==e['artifact_sha256'],'Old committed result changed')
    terminal=dict(legacy['request_starts'],**starts)
    requests=grid(cases,SEEDS[0]);require([q['id'] for q in requests[:2059]]==list(terminal),'Old request order differs')
    sessions=[]
    for p in sorted((folder/'sessions').iterdir()):
        f=read(p/'FINAL.json');seal=read(p/'FINAL_SEAL.json')
        require(digest(p/'FINAL.json')==seal['final_sha256'] and f['journal_close_error'] is None and f['exit_receipt_error'] is None,'Unproved old clock')
        sessions.append(dict(path=label(p),sha256=seal['final_sha256'],elapsed_s=f['elapsed_s']))
    require(len(sessions)==2,'Unknown old invocation')
    charged=592+sum(s['elapsed_s'] for s in sessions)
    require(abs(charged-2896.9277158999903)<1e-8,'Cannot reset/round old wall')
    failures={FAILED_OLD:'HISTORICAL_INFRASTRUCTURE_FAILURE_MISSING',FAILED_SCOPE:'HISTORICAL_INPUT_SCOPE_REJECTION_MISSING_DISPOSED_UNDER_SCOPE_R1'}
    first=dict(debits=2059,success=2057,infra_failure=2,failure_ids=sorted(failures),terminal_ids=sorted(terminal),
        failure_classes=failures,wall_carry_s=charged,old_exact_wall_s=None,
        original_conservative_carry_s=592,measured_old_continuation_wall_s=charged-592,
        legacy_import_identity=identity(legacy),old_continuation_tail=tail)
    # 'infra_failure' is the frozen storage API's aggregate consumed-failure
    # counter. Use failure_classes above in reports; never label both as IO.
    baselines={str(SEEDS[0]):first}
    for seed in SEEDS[1:]:baselines[str(seed)]=dict(debits=0,success=0,infra_failure=0,failure_ids=[],terminal_ids=[],failure_classes={},wall_carry_s=0,old_exact_wall_s=0)
    remaining={str(seed):[q for q in grid(cases,seed) if q['id'] not in baselines[str(seed)]['terminal_ids']] for seed in SEEDS}
    require([len(remaining[str(s)]) for s in SEEDS]==[3233,5292,5292],'Remaining budgets differ')
    receipt=dict(schema='rc3-s3-target-scope-retained-ledger-v1',experiment_id=EXPERIMENT,baselines=baselines,
        old_binding=old_binding,old_FINAL_receipts=sessions,original_exact_wall_s=None,
        charged_first_seed_wall_s=charged,remaining_requests=[3233,5292,5292],
        remaining_wall_s=[14400-charged,14400,14400],missing_failure_ids=sorted(failures),
        new_paid_attempts_to_refill_old_failures=0,old_successful_results=2057,
        old_debited_requests=2059,unexecuted_request_count=sum(len(v) for v in remaining.values()),
        remaining_request_order_sha256={s:identity(q) for s,q in remaining.items()})
    return permit,cases,receipt


def admit():
    d=read(HOME/'RECOVERY_DECISION.json')
    require(d['status']=='IMPLEMENTED_VERIFIED_READY_NOT_EXECUTED','Scope recovery is not technically admitted')
    require(d['refill_failed_slots'] is False and d['remaining_requests']==[3233,5292,5292],'Changed recovery range')
    require(digest(HOME/'SOURCE_BINDING.json')==d['source_binding_sha256'],'Wrong target-scope source binding')
    s=read(HOME/'SOURCE_BINDING.json')
    for name,sha in s['source_sha256'].items():require(digest(ROOT/name)==sha,'Changed target-scope source: '+name)
    for name,sha in d['evidence_sha256'].items():require(digest(ROOT/name)==sha,'Scope decision evidence changed')
    permit,cases,current=import_ledger()
    saved=read(HOME/'BASELINE_IMPORT.json');require(current==saved,'Old ledger changed after decision')
    binding=dict(experiment_id=EXPERIMENT,original_protocol_sha256=s['protocol_sha256'],
        original_stage_authorization_sha256=permit.authority_sha256,
        source_binding_sha256=d['source_binding_sha256'],scope_amendment_sha256=d['input_scope_amendment_sha256'],
        recovery_decision_sha256=digest(HOME/'RECOVERY_DECISION.json'),baseline_sha256=digest(HOME/'BASELINE_IMPORT.json'))
    return permit,cases,current,binding


def initialize(binding,baseline):
    parent=RUN.parent;marker=HOME/'SCOPE_RECOVERY_ACTIVATION.json'
    if parent.exists():require(all(p.name=='scope_r1' and p.is_dir() and not p.is_symlink() for p in parent.iterdir()),'Renamed/unknown scope run cannot reset budget')
    if marker.exists():
        m=read(marker);require(m['binding']==binding and m['baseline_sha256']==identity(baseline) and RUN.is_dir(),'Scope activation/source/budget changed')
        require(read(RUN/'AUTHORITY.json')==dict(binding=binding,baseline_sha256=identity(baseline)),'Scope authority changed')
    else:
        require(not RUN.exists(),'Unbound scope run cannot reset budget')
        write_new(RUN/'AUTHORITY.json',dict(binding=binding,baseline_sha256=identity(baseline)))
        write_new(marker,dict(binding=binding,baseline_sha256=identity(baseline),canonical_run=label(RUN)))


def next_requests(cases,baseline,seed,completed=()):
    require(seed in SEEDS,'Unregistered seed')
    old=set(baseline['baselines'][str(seed)]['terminal_ids']);done=set(completed)
    require(not old&done,'Completed request cannot refund a historic failed slot')
    return [q for q in grid(cases,seed) if q['id'] not in old|done]
