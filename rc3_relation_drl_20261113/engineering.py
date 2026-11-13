"""Registered bounded M0/M1 only. Never enters a scientific training stage."""
from __future__ import annotations
import argparse,copy,json,time,unittest,io
import numpy as np
import torch
from .common import ROOT,HOME,read,put,sha,Ledger,verify_binding
from .actor import RelationActor,tensor_hash
from .data import Pool
from .runtime import (setup,instrument,collect,group_summary,rl_update,sl_update,
    optimizer,checkpoint,restore,export_actor,load_export,memory,sync,rng_state,state_digest)

def unit_tests(ledger):
    from .test_actor import RelationActorTests
    tick=ledger.begin('unit_suite','M0_synthetic')
    original_encode,original_score=RelationActor.encode,RelationActor.score
    def enc(self,*args,**kwargs):ledger.counts['synthetic_encode_calls']+=1;return original_encode(self,*args,**kwargs)
    def score(self,*args,**kwargs):ledger.counts['synthetic_score_calls']+=1;return original_score(self,*args,**kwargs)
    RelationActor.encode,RelationActor.score=enc,score
    try:
        out=io.StringIO();result=unittest.TextTestRunner(stream=out,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RelationActorTests))
        (ledger.folder/'M0_UNIT_TESTS.txt').write_bytes(out.getvalue().encode())
        receipt=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),passed=result.wasSuccessful(),
            synthetic_only=True,optimizer_calls=0,complete_planning_requests=0,backward_calls=1,synthetic_manual_parameter_mutations=1)
        put(ledger.folder/'M0_RESULT.json',receipt)
        if not result.wasSuccessful():raise RuntimeError('Synthetic interface tests failed; see saved output')
        ledger.complete('unit_suite','M0_synthetic',tick,tests=result.testsRun)
    except BaseException as exc:ledger.fail('unit_suite','M0_synthetic',exc);raise
    finally:RelationActor.encode,RelationActor.score=original_encode,original_score

def case(actor,opt,pool,ledger,tid,index):
    device=next(actor.parameters()).device
    pre=[collect(actor,pool,tid,ledger,f'case{index}_pre_{k}',2026111392+100*index+k) for k in range(4)]
    p,_=pool.get(tid)
    with torch.no_grad():stale=actor.encode(p)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    tick=time.perf_counter();rl=rl_update(actor,opt,[pre],pool,ledger,f'case{index}_RL');sync(device)
    backward_measure=dict(wall_s=time.perf_counter()-tick,**memory(device))
    with torch.no_grad():
        from rc3_global_local.environment import Environment
        e=Environment(p,lambda q,r:q.metrics(r))
        try:actor.score(e,stale)
        except ValueError:cache_rejected=True
        else:raise ValueError('Optimizer did not invalidate stale encoding')
    sl=sl_update(actor,opt,[tid],pool,ledger,f'case{index}_SL')
    post=[collect(actor,pool,tid,ledger,f'case{index}_post_{k}',2026111492+100*index+k) for k in range(4)]
    result=dict(task_id=tid,candidates=len(p.candidate_ids),tasks=len(p.task_ids),pre=group_summary(pre),post=group_summary(post),
        RL=rl,SL=sl,stale_cache_rejected=cache_rejected,RL_update_memory_and_time=backward_measure,
        performance_claim=False,note='Engineering mixed updates, not a scientific initialization or quality experiment')
    put(ledger.folder/f'CASE_{tid}.json',result)
    print(json.dumps(dict(case=tid,candidates=len(p.candidate_ids),pre_H=pre[0]['metrics']['J'],rl_gradient=rl['gradient_norm'],
        peak_MB=backward_measure['peak_allocated']/2**20,counts=dict(ledger.counts)),ensure_ascii=False),flush=True)

def run(part):
    registration=read(HOME/'ENGINEERING_REGISTRATION.json');verify_binding(HOME/'ENGINEERING_SOURCE_BINDING.json')
    if registration['authorization']['scope']!='M0_M1_ONLY':raise PermissionError('Wrong engineering scope')
    setup(registration['seed'],registration['device'])
    ledger=Ledger(HOME/'engineering',registration['limits'],'M1:'+sha(HOME/'ENGINEERING_SOURCE_BINDING.json'))
    folder=ledger.folder
    try:
        if part=='first':
            if ledger.counts['request_attempts'] or ledger.counts['optimizer_attempts']:raise ValueError('first cannot reset paid engineering')
            unit_tests(ledger)
            actor=instrument(RelationActor().to(registration['device']),ledger)
            # Unit fixtures use their own seeds; engineering initialization is explicit.
            setup(registration['seed'],registration['device']);actor.load_state_dict(RelationActor().state_dict(),strict=True)
            opt=optimizer(actor);pool=Pool(ledger)
            for i,tid in enumerate(registration['cases'][:3]):case(actor,opt,pool,ledger,tid,i)
            ref=checkpoint(actor,opt,ledger,folder/'M1_boundary.pt',dict(next_case=3),True)
            put(folder/'RESUME_POINTER.json',ref);ledger.close('CLOSED_CHECKPOINT_BOUNDARY')
            print(json.dumps(dict(status='CHECKPOINT_BOUNDARY',ref=ref,counts=dict(ledger.counts))),flush=True)
        elif part=='second':
            ref=read(folder/'RESUME_POINTER.json');actor=instrument(RelationActor().to(registration['device']),ledger);opt=optimizer(actor)
            state=restore(actor,opt,ledger,ref,True)
            if state['cursor']!={'next_case':3}:raise ValueError('Unexpected engineering continuation')
            put(folder/'RECOVERY_RESULT.json',dict(passed=True,checkpoint=ref,model_sha256=tensor_hash(actor),optimizer_sha256=state_digest(opt.state_dict()),
                rng_sha256=state_digest(rng_state()),counts_preserved=True,query_ids_preserved=True,cursor=state['cursor'],new_process=True))
            pool=Pool(ledger)
            for i,tid in enumerate(registration['cases'][3:6],start=3):case(actor,opt,pool,ledger,tid,i)
            empties=[]
            for tid in registration['cases'][6:]:
                rows=[collect(actor,pool,tid,ledger,tid+f'_{k}',2026111592+k) for k in range(4)]
                p,_=pool.get(tid);scale=pool.reference(tid,p,1.)
                if not all(r['metrics']['K']==0 and not r['actions'] and r['valid'] for r in rows):raise ValueError('Empty terminal handling')
                if tid=='EMPTY_CLOSED' and any(r['metrics']['J']!=0 for r in rows):raise ValueError('Closed empty must cost zero')
                if tid=='EMPTY_OPEN' and any(abs(r['metrics']['J']-p.parent.D[p.entry,p.exit])>1e-7 for r in rows):raise ValueError('Open empty must include endpoint navigation')
                empties.append(dict(task_id=tid,group=group_summary(rows),reference_scale=scale,optimizer_called=False))
            put(folder/'EMPTY_BOUNDARIES.json',empties)
            export_ref=export_actor(actor,folder/'M1_ENGINEERING_ONLY.pt',True,ledger.identity)
            try:load_export(export_ref,registration['device'],allow_engineering=False)
            except ValueError:rejected=True
            else:raise ValueError('Engineering export admitted to science')
            frozen=instrument(load_export(export_ref,registration['device'],allow_engineering=True),ledger)
            frozen.requires_grad_(False);frozen.eval();actor.eval()
            for tid in (registration['cases'][0],registration['cases'][2]):
                a=collect(actor,pool,tid,ledger,'original_'+tid,greedy=True)
                b=collect(frozen,pool,tid,ledger,'frozen_'+tid,greedy=True)
                if a['raw_route']!=b['raw_route'] or a['route']!=b['route'] or a['metrics']!=b['metrics']:raise ValueError('Frozen export changed plan')
            final=checkpoint(actor,opt,ledger,folder/'M1_final.pt',dict(next_case=8,closed=True),True)
            result=dict(status='PASS',cases=8,nonempty_training_cases=6,empty_derived_cases=2,teacher_replay_checks=6,
                all_raw_H_valid=True,frozen_routes_exact=True,engineering_export_rejected_for_science=rejected,
                recovery='exact model/Adam/RNG/cursor/query budget restored in a second process',counts=dict(ledger.counts),
                elapsed_s=ledger.elapsed(),final_checkpoint=final,engineering_export=export_ref,scientific_updates=0)
            put(folder/'ENGINEERING_RESULT.json',result);ledger.close('PASS_CLOSED')
            print(json.dumps(result,ensure_ascii=False),flush=True)
        else:raise ValueError(part)
    except BaseException:
        ledger.close('TECHNICAL_FAILURE');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['first','second']);run(p.parse_args().part)
