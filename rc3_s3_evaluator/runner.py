"""Bounded subprocess evaluation. Formal mode requires separate explicit S3 authority."""
import argparse
import contextlib
import os
import subprocess
import sys
import time
import traceback
from unittest.mock import patch
from .common import HERE, ROOT, Meter, require, write_new, read, digest, identity, now
from .provenance import (admit_engineering, admit_formal, verify_training_anchors,
                         configure_runtime, ModelCatalog)
from .inputs import SavedDevelopmentInputs, FormalInputs, NavigationPool
from .journal import Journal, grid, run_root
from .pipelines import evaluate, verify_determinism


def admission(args):
    return admit_engineering(HERE/'ENGINEERING_REGISTRATION.json') if args.mode=='engineering' else admit_formal(args.authorization,args.source_approval)


def provider_for(permit):
    cls=SavedDevelopmentInputs if permit.scope=='engineering' else FormalInputs
    return cls(permit,run_root(permit))


def worker(args):
    # Fail closed even if a future accidental dependency attempts scientific updates.
    import torch
    with patch.object(torch.optim.Optimizer,'__init__',side_effect=RuntimeError('S3 forbids optimizers')), \
         patch.object(torch.autograd,'backward',side_effect=RuntimeError('S3 forbids backward')), \
         patch.object(torch.Tensor,'backward',side_effect=RuntimeError('S3 forbids backward')):
        frozen_worker(args)


def frozen_worker(args):
    # perf_counter deadline is the parent's machine monotonic clock, including Python import/startup.
    def check():
        if time.perf_counter()>=args.deadline: raise TimeoutError('Cumulative S3 stage wall reached')
    configure_runtime(); permit=admission(args); verify_training_anchors(); check()
    require(args.seed in permit.seeds, 'Unregistered model seed')
    provider=provider_for(permit); requests=grid(provider.cases,args.seed,permit.scope=='engineering')
    folder=run_root(permit)/str(args.seed)
    ledger=Journal(folder,permit.stable(),requests,permit.requests_per_seed)
    require(ledger.open_session==args.token, 'Worker has no supervised open session')
    ledger.clean(); catalog=ModelCatalog(); pool=NavigationPool(identity(permit.stable()))
    cases={c['id']:c for c in provider.cases}; emitted=0
    for q in requests:
        if q['id'] in ledger.results: continue
        check()
        if args.max_new and emitted>=args.max_new: return
        case=cases[q['case_id']]
        # Priming reads geometry and computes navigation only. Charged to outer stage,
        # excluded from hot per-request timer as required by the approved cache definition.
        if q['kind']=='hot':
            prime=Meter(ledger.append,check)
            scene=provider.geometry(case,prime)
            pool.prime(scene,prime)
            ledger.append(dict(event='hot_prime',request_id=q['id'],counts=prime.counts,seconds=prime.seconds))
        ledger.debit(q); start=time.perf_counter(); meter=Meter(ledger.append,check)
        try:
            row=evaluate(q,provider,case,pool,catalog,meter,lambda: max(.000001,args.deadline-time.perf_counter()))
            with meter.span('artifact_write'):
                write_new(folder/'results'/(q['id']+'.json'),row)
                digest(folder/'results'/(q['id']+'.json'))
            check(); ledger.complete(q,time.perf_counter()-start)
            emitted+=1
            print(f"{args.seed} {len(ledger.results)}/{len(requests)} {q['case_id']} {q['kind']} {q['method']}",flush=True)
        except BaseException as exc:
            ledger.append(dict(event='request_error',id=q['id'],error=type(exc).__name__,message=str(exc),
                elapsed_s=time.perf_counter()-start,counts=meter.counts,seconds=meter.seconds,traceback=traceback.format_exc()))
            raise


def supervise(args):
    invocation_start=time.perf_counter()
    permit=admission(args)
    root=run_root(permit)
    # Canonical immutable authority, before accessing formal declarations; alternate directories are not an API.
    if (root/'AUTHORITY.json').exists(): require(read(root/'AUTHORITY.json')==permit.stable(),'Authority cannot reset spent stage')
    else:
        require(not root.exists(),'Unbound run root cannot reset budget')
        write_new(root/'AUTHORITY.json',permit.stable())
    lock=root/'SUPERVISOR.lock'
    with exclusive_lock(lock,permit):
        execute_seeds(args,permit,root,invocation_start)


@contextlib.contextmanager
def exclusive_lock(path,permit):
    write_new(path,dict(pid=os.getpid(),authority=permit.stable(),utc=now()))
    try: yield
    finally: path.unlink()


def execute_seeds(args,permit,root,invocation_start):
    for si,seed in enumerate(permit.seeds):
        if si: invocation_start=time.perf_counter()
        provider=provider_for(permit); requests=grid(provider.cases,seed,permit.scope=='engineering')
        require(len(requests)==permit.requests_per_seed,'Approved request count mismatch')
        folder=root/str(seed); ledger=Journal(folder,permit.stable(),requests,permit.requests_per_seed)
        require(ledger.open_session is None,'Unclosed supervisor: time/resources unknown, no automatic replay')
        ledger.clean()
        if (folder/'COMPLETE.json').exists():
            c=read(folder/'COMPLETE.json')
            require(c['journal_sha256']==digest(ledger.path) and len(ledger.results)==len(requests),'Completed evidence changed')
            continue
        remaining=permit.wall_s_per_seed-ledger.wall_s
        require(remaining>0,'Stage wall exhausted')
        token=identity([permit.stable(),seed,len(ledger.events),now(),os.getpid()])
        ledger.append(dict(event='session_open',token=token,remaining_wall_s=remaining,utc=now()))
        deadline=invocation_start+remaining
        command=[sys.executable,'-B','-X','utf8','-m','rc3_s3_evaluator.runner',args.mode,'--worker',
            '--seed',str(seed),'--deadline',str(deadline),'--token',token]
        if args.mode=='formal': command+=['--authorization',args.authorization,'--source-approval',args.source_approval]
        if args.max_new: command+=['--max-new',str(args.max_new)]
        logfile=folder/('session_'+token[:16]+'.log')
        status='FAILED'; error=None
        try:
            with logfile.open('xb') as stream:
                p=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                try: code=p.wait(timeout=max(.001,deadline-time.perf_counter()))
                except subprocess.TimeoutExpired:
                    p.kill(); p.wait(); raise TimeoutError('Supervisor stopped worker at cumulative wall cap')
            require(code==0,'Worker failed; see '+str(logfile))
            status='CLOSED_BOUNDARY'
        except BaseException as exc:
            error=str(exc)
        elapsed=time.perf_counter()-invocation_start
        closed=Journal(folder,permit.stable(),requests,permit.requests_per_seed)
        closed.append(dict(event='session_close',token=token,elapsed_s=elapsed,status=status,error=error,
            cap_s=permit.wall_s_per_seed,log_sha256=digest(logfile) if logfile.exists() else None))
        require(error is None,error or 'Failed supervised stage')
        closed=Journal(folder,permit.stable(),requests,permit.requests_per_seed); closed.clean()
        require(closed.wall_s<=permit.wall_s_per_seed,'Stage exceeded cap; preserve charged evidence')
        if len(closed.results)==len(requests):
            determinism=verify_determinism(closed.rows())
            write_new(folder/'COMPLETE.json',dict(status='COMPLETE',scope=permit.scope,seed=seed,
                requests=len(closed.results),wall_s=closed.wall_s,journal_sha256=digest(closed.path),
                determinism=determinism,source=permit.stable()))
        else: return
    if permit.scope=='formal' and not (root/'FORMAL_ANALYSIS.json').exists():
        from .statistics import analyze
        rows=[]
        for seed in permit.seeds:
            j=Journal(root/str(seed),permit.stable(),grid(provider.cases,seed),permit.requests_per_seed)
            rows.extend(j.rows())
        write_new(root/'FORMAL_ANALYSIS.json',analyze(rows,provider.cases))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('engineering','formal'))
    p.add_argument('--authorization');p.add_argument('--source-approval')
    p.add_argument('--max-new',type=int,default=0,help='Closed-boundary pause; never repeats charged requests')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--seed',type=int);p.add_argument('--deadline',type=float);p.add_argument('--token')
    args=p.parse_args()
    require(args.max_new>=0,'Invalid pause count')
    if args.worker: worker(args)
    else: supervise(args)


if __name__=='__main__': main()
