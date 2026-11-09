"""Explicit future continuation entry. Default --check never computes a request."""
import time
PROCESS_START=time.perf_counter()
import argparse
import os
import subprocess
import sys
import traceback
from unittest.mock import patch
from rc3_s3_evaluator.common import ROOT,SEEDS,Meter,read,write_new,digest,identity,require,now,finite_number
from rc3_s3_evaluator.inputs import FormalInputs,NavigationPool
from rc3_s3_evaluator.provenance import configure_runtime,ModelCatalog
from rc3_s3_evaluator.journal import grid
from rc3_s3_recovery_20261108.storage import Journal,Lease,independent_receipt,verify_supervisor_lease
from rc3_s3_recovery_20261108.authority import RUN as OLD_RUN,CONT as OLD_CONT
from .scan import HOME
from .recovery import RUN,admit,initialize,next_requests
from .scope import evaluate,CONTRACT


class Inputs(FormalInputs):
    def __init__(self,permit):super().__init__(permit,RUN)
    def load(self,case,meter):
        case_number=int(case['id'].split('_')[1])
        # Paid normal regeneration/checking remains identical. These are only
        # immutable artifact locations, never free task/nav/embedding caches.
        self.output=OLD_RUN if case_number<4 else (OLD_CONT if case_number<=42 else RUN)
        return super().load(case,meter)


def used_wall(folder,baseline,events,binding,current_token=None):
    used=baseline['wall_carry_s']
    dirs={p.name:p for p in (folder/'sessions').glob('*') if p.is_dir()}
    tokens={e['token'] for e in events if e['event']=='session_open'}
    require(set(dirs)==tokens,'Missing/orphan clock cannot reset budget')
    if current_token is not None:
        require(current_token in dirs and read(dirs[current_token]/'OPEN.json')['parent_pid']==os.getpid() and
            not any(e['event']=='session_close' and e['token']==current_token for e in events),'Only owned live clock may be excluded')
        dirs.pop(current_token)
    for token,p in dirs.items():
        op=read(p/'OPEN.json')
        final=read(p/'FINAL.json');seal=read(p/'FINAL_SEAL.json')
        finite_number(final['elapsed_s'])
        require(op['binding']==binding and op['token']==final['token']==token and final['elapsed_s']>=0 and
            digest(p/'FINAL.json')==seal['final_sha256'] and final['status']=='CLOSED_BOUNDARY','New failure/unclosed clock needs explicit disposition')
        used+=final['elapsed_s']
    return used


def finish(folder,token,start,status,error,close,remaining):
    p=folder/'sessions'/token;close_error=None;exit_error=None
    before=dict(status=status,error=error,elapsed_s=time.perf_counter()-start,utc=now())
    try:write_new(p/'EXIT_BEFORE_JOURNAL.json',before)
    except BaseException as exc:
        exit_error=repr(exc)
        try:independent_receipt([HOME/'runtime_fatal',HOME/'emergency'],'PARENT_EXIT_CLOCK_FAILURE',dict(before=before,error=exit_error))
        except BaseException:pass  # Continue to independent FINAL even if fallback receipt fails.
    try:close(time.perf_counter()-start)
    except BaseException as exc:
        close_error=repr(exc)
        try:independent_receipt([p/'faults',HOME/'runtime_fatal'],'PARENT_CLOSE_FAILURE',dict(error=close_error,original_error=error))
        except BaseException:pass
    final=dict(token=token,status='FAILED' if error or close_error or exit_error else status,error=error,journal_close_error=close_error,exit_receipt_error=exit_error,
        elapsed_s=time.perf_counter()-start,utc=now(),timing_scope='Includes admission/source/restore/prepare/navigation/inference/H/export/validation/IO/closure; final tiny self-seal cannot time itself')
    if remaining is None:final['remaining_budget_unknown']=True
    elif final['elapsed_s']>remaining:final.update(status='FAILED',budget_error='Hard cumulative wall cap exceeded')
    try:write_new(p/'FINAL.json',final);write_new(p/'FINAL_SEAL.json',dict(final_sha256=digest(p/'FINAL.json')))
    except BaseException as exc:independent_receipt([HOME/'runtime_fatal'],'FINAL_FAILURE',dict(final=final,error=repr(exc)));raise
    return final


def worker(args):
    import torch
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=RuntimeError('No optimizer in S3')),\
         patch.object(torch.autograd,'backward',side_effect=RuntimeError('No backward in S3')),\
         patch.object(torch.Tensor,'backward',side_effect=RuntimeError('No backward in S3')):
        configure_runtime();permit,cases,baseline,binding=admit();require(args.seed in SEEDS,'Unregistered seed')
        folder=RUN/str(args.seed);session=read(folder/'sessions'/args.token/'OPEN.json')
        require(session['binding']==binding and session['token']==args.token,'Source/session mismatch')
        verify_supervisor_lease(HOME/'SCOPE_SUPERVISOR.lease',session['parent_pid'],identity(binding))
        def check():require(time.perf_counter()<args.deadline,'Cumulative stage wall exhausted')
        check();provider=Inputs(permit);catalog=ModelCatalog();pool=NavigationPool(identity(binding));bycase={c['id']:c for c in cases}
        b=baseline['baselines'][str(args.seed)]
        with Journal(folder,binding,grid(cases,args.seed),b) as j:
            j.clean(allow_session=args.token)
            for q in next_requests(cases,baseline,args.seed,j.results):
                check();case=bycase[q['case_id']]
                if q['kind']=='hot':
                    prime=Meter(j.append,check);scene=provider.geometry(case,prime);pool.prime(scene,prime)
                    j.append(dict(event='hot_prime',request_id=q['id'],counts=prime.counts,seconds=prime.seconds))
                j.debit(q);tick=time.perf_counter();meter=Meter(j.append,check)
                try:
                    row=evaluate(q,provider,case,pool,catalog,meter,lambda:max(.000001,args.deadline-time.perf_counter()))
                    row['persistence_revision']=binding['source_binding_sha256'];row['input_contract']=CONTRACT
                    row['timing_boundary_version']='complete-request-scope-r1-extra-parent-diagnostic-measured'
                    with meter.span('artifact_write'):
                        path=folder/'results'/(q['id']+'.json');write_new(path,row);digest(path)
                    check();j.complete(q,time.perf_counter()-tick)
                    print(f"{args.seed} scheduled={b['debits']+len(j.starts)}/5292 successful={b['success']+len(j.results)}",flush=True)
                except BaseException as exc:
                    evidence=dict(id=q['id'],request=q,error=repr(exc),traceback=traceback.format_exc(),counts=meter.counts,
                        seconds=meter.seconds,scope_evidence=getattr(exc,'evidence',None),elapsed_s=time.perf_counter()-tick)
                    independent_receipt([folder/'faults',HOME/'runtime_fatal'],'REQUEST_FAILURE',evidence)
                    if j.healthy:
                        try:j.append(dict(event='request_error',**{k:v for k,v in evidence.items() if k!='request'}))
                        except BaseException:pass
                    raise


def run_seed(seed,start,cases,baseline,binding):
    folder=RUN/str(seed);b=baseline['baselines'][str(seed)];token=identity([binding,seed,start,os.getpid(),now()])
    session=folder/'sessions'/token;error=None;status='CLOSED_BOUNDARY';used=None;child=None
    try:
        with Journal(folder,binding,grid(cases,seed),b) as j:
            j.clean();used=used_wall(folder,b,j.events,binding)
            if (folder/'SCHEDULE_SEAL.json').exists():
                done=read(folder/'SCHEDULE_COMPLETE.json');seal=read(folder/'SCHEDULE_SEAL.json')
                require(seal['schedule_sha256']==digest(folder/'SCHEDULE_COMPLETE.json') and done['journal_sha256']==digest(j.path) and b['debits']+len(j.results)==5292,'Changed completed schedule')
                return False
            require(used<14400,'No remaining stage wall')
            write_new(session/'OPEN.json',dict(token=token,parent_pid=os.getpid(),binding=binding,carried_wall_s=used,utc=now()))
            j.append(dict(event='session_open',token=token,utc=now(),remaining_wall_s=14400-used))
        deadline=start+14400-used
        with (folder/('session_'+token[:16]+'.log')).open('xb') as log:
            child=subprocess.Popen([sys.executable,'-B','-X','utf8','-m','rc3_s3_connectivity_20261109.resume',
                '--worker','--seed',str(seed),'--token',token,'--deadline',str(deadline)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            try:code=child.wait(timeout=max(.001,deadline-time.perf_counter()))
            except subprocess.TimeoutExpired:child.kill();child.wait();raise TimeoutError('Stage hard wall limit')
        require(code==0,'Worker stopped; preserve all paid state')
    except BaseException as exc:error=repr(exc);status='FAILED'
    if not (session/'OPEN.json').exists():
        try:write_new(session/'OPEN.json',dict(token=token,parent_pid=os.getpid(),binding=binding,carried_wall_s=used,utc=now(),early_failure=True))
        except BaseException as exc:
            try:independent_receipt([HOME/'runtime_fatal',HOME/'emergency'],'EARLY_PARENT_CLOCK',dict(token=token,error=repr(exc),original_error=error,elapsed_s=time.perf_counter()-start))
            except BaseException:pass
    def close(elapsed):
        with Journal(folder,binding,grid(cases,seed),b) as j:
            if status=='CLOSED_BOUNDARY':
                j.clean(allow_session=token);require(b['debits']+len(j.results)==5292,'Incomplete request scheduling')
                require(used_wall(folder,b,j.events,binding,current_token=token)==used,'Prior paid clocks changed')
            j.append(dict(event='session_close',token=token,status=status,error=error,elapsed_s=elapsed))
            if status=='CLOSED_BOUNDARY':
                write_new(folder/'SCHEDULE_COMPLETE.json',dict(status='REQUEST_SCHEDULING_FINISHED_WITH_RETAINED_HISTORICAL_FAILURES_NOT_FULL_SCIENTIFIC_GRID',
                    seed=seed,debits=5292,successful=b['success']+len(j.results),historical_failure_classes=b['failure_classes'],
                    original_exact_wall_s=b['old_exact_wall_s'],wall_carry_s=b['wall_carry_s'],journal_sha256=digest(j.path),binding=binding))
    final=finish(folder,token,start,status,error,close,14400-used if used is not None else None)
    require(final['status']=='CLOSED_BOUNDARY','Stopped with paid clock/failure evidence; no automatic recovery')
    write_new(folder/'SCHEDULE_SEAL.json',dict(schedule_sha256=digest(folder/'SCHEDULE_COMPLETE.json'),final_sha256=digest(session/'FINAL.json')))
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--seed',type=int);p.add_argument('--token');p.add_argument('--deadline',type=float);args=p.parse_args()
    if args.worker:return worker(args)
    try:
        permit,cases,baseline,binding=admit()
        if not args.execute:
            print(dict(status='TECHNICAL_CHECK_ONLY_NO_GRID_REQUEST',remaining=baseline['remaining_requests'],remaining_wall_s=baseline['remaining_wall_s'],binding=binding));return
        require(not list((HOME/'runtime_fatal').glob('*.json')),'Unresolved new failure; cannot reset resources')
        with Lease(HOME/'SCOPE_SUPERVISOR.lease',identity(binding)):
            initialize(binding,baseline);start=PROCESS_START
            for seed in SEEDS:
                ran=run_seed(seed,start,cases,baseline,binding)
                if ran and seed!=SEEDS[-1]:
                    start=time.perf_counter();permit,cases,baseline,binding=admit()
        from .postprocess import analyze_saved
        analyze_saved()
        print('All original request slots scheduled. Two historical failed result slots remain missing; scope-r1 analysis is version-labelled and no full original-grid scientific PASS is asserted.')
    except BaseException as exc:
        if args.execute:
            independent_receipt([HOME/'runtime_fatal'],'SUPERVISOR_FAILURE',dict(error=repr(exc),traceback=traceback.format_exc(),
                invocation_elapsed_s=time.perf_counter()-PROCESS_START,note='Do not add overlapping outer clock to sealed per-seed clocks; unresolved pre-admission failure must be disposed before any retry'))
        raise


if __name__=='__main__':main()
