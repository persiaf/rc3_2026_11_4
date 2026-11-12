"""Register a bounded IO-only repair and debit, without any ML operation."""
import ast,collections,json,time,datetime
from .common import ROOT,HOME,read,put,append,sha,state

def main():
    begin=time.perf_counter();now=time.time()
    target=HOME/'R03_IO_REPAIR_REGISTRATION.json'
    if target.exists():raise FileExistsError('R03 registration must not be repeated')
    receipt=read(HOME/'controls/science-01/SUPERVISOR_RECEIPT.json')
    if not receipt['root_exit_confirmed'] or receipt['job_active_processes_at_end']!=0:raise RuntimeError('Old worker is not exited')
    snapshot=HOME/'source_revisions/io_failure01';evidence=read(snapshot/'FAILURE_RECOVERY_EVIDENCE.json')
    cp=read(HOME/'streams/2026111101/D5-PG/LATEST.json')
    if cp['cursor']!=740 or sha(ROOT/cp['path'])!=cp['sha256'] or sha(snapshot/'step00740_training.pt')!=cp['sha256']:raise ValueError('Checkpoint identity')
    r=read(HOME/'RESOURCE_ACCOUNTING.json');c=r['counts']
    events=[json.loads(x) for x in (HOME/'EVENTS.jsonl').read_text(encoding='utf8').splitlines()]
    if events[-1]['request_id']!='2026111101:D5-PG:741:3' or events[-1]['event']!='request_attempt' or c['requests']!=6713 or c['actor_optimizer_attempts']!=750:raise ValueError('Actual failed-attempt identity')
    before=dict(c);purposes=collections.Counter(e['purpose'] for e in events if e['event']=='request_attempt')
    for purpose,n in purposes.items():
        c['purpose_'+purpose+'_requests']=n+(15 if purpose=='engineering' else 0)
        if purpose!='engineering':c[purpose+'_requests']=n
    c['requests_failed']+=1
    append(HOME/'EVENTS.jsonl',dict(event='request_failure',job='science-01',pool='science',request_id='2026111101:D5-PG:741:3',purpose='train',
        error='WinError5 atomic resource replacement after attempt was charged, before model/H execution',reconstructed_after_confirmed_worker_exit=True,
        exact_stdout_evidence=(snapshot/'WORKER_STDOUT.log').relative_to(ROOT).as_posix(),failed_at_utc=receipt['finish_utc']))
    supplement=read(HOME/'ENGINEERING_WALL_SUPPLEMENT.json')
    post_start=(HOME/'audit_saved_results.py').stat().st_ctime
    setup=supplement['engineering_s_to_add'];post=max(0.,now-post_start)
    r['wall']['engineering_s']+=setup+post;r['wall']['total_s']+=setup+post
    if r['wall']['engineering_s']>=21600-30 or r['wall']['total_s']>=345600-30:raise RuntimeError('Actual engineering work exceeds existing boundary')
    put(target,dict(status='REGISTERED_BEFORE_IO_TEST_AND_SCIENCE_RESUME',root_cause='Windows sharing denial of atomic replace; exact committed740 checkpoint and3 saved same-actor trajectories survive',
        same_root_repair_number=1,max_same_root_repairs=2,source_before_sha256={n:sha(snapshot/n) for n in ('common.py','worker.py')},
        primitive='Same fsynced pending JSON only; atomic os.replace, original destination preserved until success',
        delays_s=[0,.01,.02,.04,.08,.16,.32],retry_only_winerrors=[5,32,33],no_ACL_or_permission_override=True,
        io_test=dict(wall_limit_s=10,new_requests=0,new_Actor=0,new_forward=0,new_H=0,real_Windows_read_share_conflict=True),
        failed_request_kept=True,new_failed_requests=1,actual_Actor_attempts_before=750,actual_scientific_updates=740,
        restore='Same original scientific identity, model, Adam, all RNG, cursor740 and saved same-actor3 traces. Retry only missing scheduled slots; no free attempts or wall refund.',
        original8h_stream_and96h300000_limits_unchanged=True,scientific_method_protocol_schedule_and_thresholds_unchanged=True,
        historical_failure_snapshot=snapshot.relative_to(ROOT).as_posix()))
    put(HOME/'R03_RESOURCE_RECONCILIATION.json',dict(before_counts=before,after_counts=dict(c),no_actual_consumption_refund=True,
        purpose_reconstruction='From durable request_attempt events; first debit survived, second atomic save failed before collection',
        manual_setup_engineering_added_s=setup,setup_supplement_sha256=sha(HOME/'ENGINEERING_WALL_SUPPLEMENT.json'),
        post_engineering_interval_utc=[datetime.datetime.fromtimestamp(post_start,datetime.timezone.utc).isoformat(),datetime.datetime.fromtimestamp(now,datetime.timezone.utc).isoformat()],
        post_engineering_added_s=post,parallel_engineering_and_science_are_charged_additively=True))
    b=read(HOME/'SCIENTIFIC_SOURCE_BINDING_R02.json')
    for name in ('runtime.py','training.py','science.py','data.py','evaluation.py','later_stages.py'):
        path=HOME/name
        if sha(path)!=b['source_sha256'][path.relative_to(ROOT).as_posix()]:raise ValueError('Scientific math/data source changed during IO repair')
    for name in ('common.py','worker.py','io_acceptance.py','register_io_repair.py'):
        path=HOME/name;ast.parse(path.read_text(encoding='utf8'));b['source_sha256'][path.relative_to(ROOT).as_posix()]=sha(path)
    b.update(active_repair='R03_bounded_atomic_IO_only',prior_execution_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING_R02.json'),repair_registration_sha256=sha(target))
    b['revision_artifacts'][target.name]=sha(target)
    put(HOME/'SCIENTIFIC_SOURCE_BINDING_R03.json',b)
    put(HOME/'REPAIR_AUTHORIZATION_R03.json',dict(authorized=True,authority='Current user explicitly allows necessary engineering fixes and same-boundary checkpoint recovery',
        scientific_identity=b['scientific_identity'],source_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING_R03.json'),protocol_sha256=sha(HOME/'REVISION_PROTOCOL.json'),
        requires_R03_IO_acceptance_PASS=True,no_scientific_checkpoint_identity_change=True,no_budget_increase=True))
    put(HOME/'ACTIVE_EXECUTION_SOURCE.json',dict(binding='SCIENTIFIC_SOURCE_BINDING_R03.json',authorization='REPAIR_AUTHORIZATION_R03.json'),True)
    elapsed=time.perf_counter()-begin;r['wall']['engineering_s']+=elapsed;r['wall']['total_s']+=elapsed;put(HOME/'RESOURCE_ACCOUNTING.json',r,True)
    put(HOME/'R03_REGISTRATION_RECEIPT.json',dict(wall_s=elapsed,Actor=0,requests=0,source_binding_sha256=sha(HOME/'SCIENTIFIC_SOURCE_BINDING_R03.json')))
    state(status='REGISTERED_TECHNICAL_IO_RECOVERY',failed_phase='science',failed_job='science-01',error='WinError5 resource atomic replace; all consumption preserved',
        recovery_checkpoint_cursor=740,stage_execution_authority_unchanged=True)
    print(dict(status='R03_REGISTERED',committed_updates=740,requests=c['requests'],Actor=c['actor_optimizer_attempts'],requests_failed=c['requests_failed'],manual_engineering_wall_added_s=setup+post))

if __name__=='__main__':main()
