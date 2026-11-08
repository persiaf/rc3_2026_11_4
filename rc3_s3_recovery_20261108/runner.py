"""User-authorized fixed-grid continuation; only never-debited IDs are computed."""
import argparse
import contextlib
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from unittest.mock import patch
from rc3_s3_evaluator.common import ROOT,SEEDS,Meter,read,write_new,digest,identity,require,now
from rc3_s3_evaluator.inputs import FormalInputs,NavigationPool
from rc3_s3_evaluator.provenance import configure_runtime,ModelCatalog
from rc3_s3_evaluator.journal import grid
from rc3_s3_evaluator.pipelines import evaluate
from .authority import HOME,RUN,CONT,admit,initialize,baseline_for,ledger_binding,disposed_dispatch_sessions
from .storage import Journal,Lease,independent_receipt,verify_supervisor_lease


class Inputs(FormalInputs):
    """Same frozen producer; reuse the original four input artifacts read-only."""
    def __init__(self,permit):super().__init__(permit,CONT)
    def load(self,case,meter):
        self.output=RUN if case['id'] in {'S3_000','S3_001','S3_002','S3_003'} else CONT
        return super().load(case,meter)


def clock_used(folder,events,baseline,current_token=None):
    dirs={p.name:p for p in (folder/'sessions').glob('*') if p.is_dir()}
    tokens={e['token'] for e in events if e['event']=='session_open'}
    if current_token is not None:
        require(current_token in dirs and read(dirs[current_token]/'OPEN.json')['parent_pid']==os.getpid() and
            not any(e['event']=='session_close' and e['token']==current_token for e in events),'Only this live, unclosed parent session may be excluded from prior clocks')
        dirs.pop(current_token);tokens.discard(current_token)
    require(set(dirs)==tokens,'Orphan/missing session clock; resources cannot be reset')
    used=baseline['wall_carry_s']
    for token,p in dirs.items():
        op=read(p/'OPEN.json');final=read(p/'FINAL.json');seal=read(p/'FINAL_SEAL.json')
        disposed=disposed_dispatch_sessions().get(token)
        acknowledged=disposed is not None and final['status']=='FAILED' and digest(p/'FINAL.json')==disposed['final_sha256']
        if acknowledged:
            active=False
            for e in events:
                if e['event']=='session_open' and e['token']==token:active=True
                elif active and e['event']=='session_close' and e['token']==token:active=False;break
                elif active:require(e['event']!='request_start','Paid request failure cannot use dispatch disposition')
            require(not active and final['elapsed_s']==disposed['charged_elapsed_s'],'Unclosed/refunded dispatch session')
        require(op['token']==final['token']==token and seal['final_sha256']==digest(p/'FINAL.json') and
            final['elapsed_s']>=0 and (final['status']=='CLOSED_BOUNDARY' or acknowledged),'Failed/unclosed session needs explicit recovery')
        require(op['carried_wall_s']>=baseline['wall_carry_s'],'Baseline wall reservation lost')
        used+=final['elapsed_s']
    return used


def finish_session(folder,token,start,status,error,close_journal,remaining_wall_s=None):
    """Clock evidence is persisted BEFORE parsing a potentially broken Journal."""
    p=folder/'sessions'/token
    exit_receipt_error=None
    before=dict(token=token,status=status,error=error,elapsed_s=time.perf_counter()-start,utc=now())
    try:write_new(p/'EXIT_BEFORE_JOURNAL.json',before)
    except BaseException as exc:
        exit_receipt_error=repr(exc)
        try:independent_receipt([folder/'emergency',HOME/'runtime_emergency'],'PARENT_CLOCK_FALLBACK',before)
        except BaseException:pass
    close_error=None
    try:close_journal(time.perf_counter()-start)
    except BaseException as exc:
        close_error=repr(exc)
        try:independent_receipt([p/'faults',folder/'emergency'],'PARENT_CLOSE_FAILURE',dict(token=token,error=close_error,original_error=error))
        except BaseException:pass
    final=dict(token=token,status='FAILED' if error or close_error or exit_receipt_error else status,error=error,
        exit_receipt_error=exit_receipt_error,
        journal_close_error=close_error,elapsed_s=time.perf_counter()-start,utc=now(),
        timing_scope='All admission/legacy validation, restore/startup, navigation prime, requests, export/validation/IO and closure before this final tiny receipt; nested clocks not added')
    if remaining_wall_s is not None and final['elapsed_s']>remaining_wall_s:
        final.update(status='FAILED',budget_error='Measured session exceeded remaining wall; no budget extension')
    try:
        write_new(p/'FINAL.json',final);write_new(p/'FINAL_SEAL.json',dict(final_sha256=digest(p/'FINAL.json')))
    except BaseException as exc:
        final['status']='FAILED';final['final_receipt_error']=repr(exc)
        independent_receipt([folder/'emergency',HOME/'runtime_emergency'],'FINAL_CLOCK_FALLBACK',final)
    return final


def worker(args):
    import torch
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=RuntimeError('No optimizers in S3')), \
         patch.object(torch.autograd,'backward',side_effect=RuntimeError('No backward in S3')), \
         patch.object(torch.Tensor,'backward',side_effect=RuntimeError('No backward in S3')):
        _worker(args)


def _worker(args):
    def check():require(time.perf_counter()<args.deadline,'Continuation stage wall exhausted')
    configure_runtime();permit,legacy,binding=admit();check()
    require(args.seed in SEEDS,'Unregistered seed')
    folder=CONT/str(args.seed);session=read(folder/'sessions'/args.token/'OPEN.json')
    require(session['token']==args.token and session['binding']==binding,'Worker session/source identity mismatch')
    verify_supervisor_lease(CONT/'SUPERVISOR.lease',session['parent_pid'],identity(binding))
    provider=Inputs(permit);requests=grid(provider.cases,args.seed);baseline=baseline_for(legacy,args.seed)
    catalog=ModelCatalog();pool=NavigationPool(identity(permit.stable()));cases={c['id']:c for c in provider.cases}
    with Journal(folder,ledger_binding(binding),requests,baseline) as j:
        j.clean(allow_session=args.token);require(j.session==args.token,'No open supervised session')
        try:
            for q in requests:
                if q['id'] in baseline['terminal_ids'] or q['id'] in j.results:continue
                check();case=cases[q['case_id']]
                if q['kind']=='hot':
                    prime=Meter(j.append,check);scene=provider.geometry(case,prime);pool.prime(scene,prime)
                    j.append(dict(event='hot_prime',request_id=q['id'],counts=prime.counts,seconds=prime.seconds))
                j.debit(q);tick=time.perf_counter();meter=Meter(j.append,check)
                try:
                    row=evaluate(q,provider,case,pool,catalog,meter,lambda:max(.000001,args.deadline-time.perf_counter()))
                    row['persistence_revision']=binding['recovery_source_binding_sha256']
                    row['timing_boundary_version']='same-complete-request-boundary-with-repaired-anchor-IO'
                    with meter.span('artifact_write'):
                        path=folder/'results'/(q['id']+'.json');write_new(path,row);digest(path)
                    check();j.complete(q,time.perf_counter()-tick)
                    print(f"{args.seed} scheduled={baseline['debits']+len(j.starts)}/5292 success={baseline['success']+len(j.results)} historical_failure={baseline['infra_failure']} {q['case_id']} {q['kind']} {q['method']}",flush=True)
                except BaseException as exc:
                    evidence=dict(request=q,error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.perf_counter()-tick,
                        counts=meter.counts,seconds=meter.seconds,journal_healthy=j.healthy)
                    independent_receipt([folder/'faults',CONT/'emergency'],'REQUEST_FAILURE',evidence)
                    if j.healthy:
                        try:j.append(dict(event='request_error',id=q['id'],**{k:v for k,v in evidence.items() if k!='request'}))
                        except BaseException:pass
                    raise
        except BaseException as exc:
            independent_receipt([folder/'faults',CONT/'emergency'],'WORKER_FAILURE',dict(seed=args.seed,token=args.token,error=repr(exc),traceback=traceback.format_exc()))
            raise


def run_seed(seed,start,permit,legacy,binding):
    folder=CONT/str(seed);baseline=baseline_for(legacy,seed)
    token=identity([binding,seed,start,now(),os.getpid()]);p=folder/'sessions'/token
    used=None;requests=None;status='CLOSED_BOUNDARY';error=None;schedule_sha=None
    try:
        provider=Inputs(permit);requests=grid(provider.cases,seed)
        with Journal(folder,ledger_binding(binding),requests,baseline) as j:
            j.clean();used=clock_used(folder,j.events,baseline)
            if (folder/'SCHEDULE_COMPLETE.json').exists():
                done=read(folder/'SCHEDULE_COMPLETE.json');seal=read(folder/'SCHEDULE_SEAL.json')
                require(done['journal_sha256']==digest(j.path) and len(j.results)+baseline['debits']==5292 and
                    seal['schedule_sha256']==digest(folder/'SCHEDULE_COMPLETE.json'),'Completed schedule changed')
                return False
            require(used<14400,'No stage wall budget')
            write_new(p/'OPEN.json',dict(token=token,parent_pid=os.getpid(),binding=binding,carried_wall_s=used,remaining_wall_s=14400-used,utc=now()))
            j.append(dict(event='session_open',token=token,remaining_wall_s=14400-used,utc=now()))
        deadline=start+14400-used
        cmd=[sys.executable,'-B','-X','utf8','-m','rc3_s3_recovery_20261108.runner','--worker','--seed',str(seed),'--token',token,'--deadline',str(deadline)]
        log=folder/('session_'+token[:16]+'.log')
        with log.open('xb') as stream:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=max(.001,deadline-time.perf_counter()))
            except subprocess.TimeoutExpired:child.kill();child.wait();raise TimeoutError('Hard continuation wall limit')
        require(code==0,'Worker failed: '+str(log))
    except BaseException as exc:error=repr(exc);status='FAILED'
    # Even failure while creating session_open has an independent per-seed clock.
    if not (p/'OPEN.json').exists():
        try:write_new(p/'OPEN.json',dict(token=token,parent_pid=os.getpid(),binding=binding,
            carried_wall_s=used if used is not None else baseline['wall_carry_s'],prior_clock_known=used is not None,utc=now()))
        except BaseException:
            independent_receipt([CONT/'supervisor_fatal',HOME/'runtime_emergency'],'EARLY_PARENT_SEED_CLOCK',
                dict(seed=seed,token=token,error=error,elapsed_s=time.perf_counter()-start))
    def close_journal(elapsed):
        nonlocal schedule_sha
        require(requests is not None,'No admitted request grid')
        with Journal(folder,ledger_binding(binding),requests,baseline) as j:
            if status=='CLOSED_BOUNDARY':
                j.clean(allow_session=token)
                require(clock_used(folder,j.events,baseline,current_token=token)==used,'Prior resources changed')
                require(len(j.results)+baseline['debits']==5292,'Incomplete scheduling')
            j.append(dict(event='session_close',token=token,status=status,error=error,elapsed_s=elapsed))
            if status=='CLOSED_BOUNDARY':
                write_new(folder/'SCHEDULE_COMPLETE.json',dict(status='ALL_REQUEST_SLOTS_DISPOSED_NOT_FULL_SCIENTIFIC_GRID',seed=seed,
                    debits=5292,success=baseline['success']+len(j.results),historical_infrastructure_failures=baseline['infra_failure'],
                    historical_exact_wall_s=baseline['old_exact_wall_s'],conservative_historical_carry_s=baseline['wall_carry_s'],
                    final_clock_receipt=str(p/'FINAL.json'),journal_sha256=digest(j.path),binding=binding))
                schedule_sha=digest(folder/'SCHEDULE_COMPLETE.json')
    final=finish_session(folder,token,start,status,error,close_journal,remaining_wall_s=14400-used if used is not None else None)
    require(final['status']=='CLOSED_BOUNDARY','Stopped with independent per-seed failure/clock receipts')
    # Only this tiny self-seal follows FINAL: no second full result parse/hash pass.
    write_new(folder/'SCHEDULE_SEAL.json',dict(schedule_sha256=schedule_sha,final_sha256=digest(p/'FINAL.json')))
    return True


def _supervise():
    start=time.perf_counter();permit,legacy,binding=admit();initialize(binding,legacy)
    allowed=binding.get('resolved_parent_fatal_receipts',{})
    for folder in (CONT/'supervisor_fatal',HOME/'runtime_emergency'):
        for p in folder.glob('*.json'):
            from rc3_s3_evaluator.common import label
            require(allowed.get(label(p))==digest(p),'Unknown parent failure receipt; no automatic reset')
    with Lease(CONT/'SUPERVISOR.lease',identity(binding)):
        if (CONT/'SCIENTIFIC_ANALYSIS.json').exists():
            from .analysis import analyze_saved
            analyze_saved();return
        for seed in SEEDS:
            ran=run_seed(seed,start,permit,legacy,binding)
            if ran and seed!=SEEDS[-1]:start=time.perf_counter();permit,legacy,binding=admit()
    # Read-only postprocessing has a separate maintenance clock; no predictions.
    from .analysis import analyze_saved
    analyze_saved()


def supervise():
    began=time.perf_counter()
    try:
        _supervise()
    except BaseException as exc:
        independent_receipt([CONT/'supervisor_fatal',HOME/'runtime_emergency'],'SUPERVISOR_FATAL',
            dict(error=repr(exc),traceback=traceback.format_exc(),outer_invocation_elapsed_s=time.perf_counter()-began,
                note='Supplementary outer clock, not added to per-seed clocks; blocks silent retry and budget reset'))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--seed',type=int);p.add_argument('--token');p.add_argument('--deadline',type=float)
    args=p.parse_args()
    if args.worker:worker(args)
    else:supervise()


if __name__=='__main__':main()
