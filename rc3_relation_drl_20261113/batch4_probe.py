"""Supplementary M1 production-shape check within the original cumulative cap."""
import json,os,time
import torch
from .common import ROOT,HOME,read,put,sha,Ledger,verify_binding
from .actor import RelationActor
from .data import Pool
from .runtime import setup,instrument,optimizer,restore,collect,rl_update,sl_update,checkpoint,memory,sync,group_summary

def worker():
    verify_binding(HOME/'ENGINEERING_SOURCE_BINDING.json')
    reg=read(HOME/'M1_BATCH4_REGISTRATION.json')
    if reg['source_sha256']!=sha(__file__) or reg['prior_result_sha256']!=sha(HOME/'engineering/ENGINEERING_RESULT.json'):raise ValueError('Batch probe binding')
    setup(2026111391,'cuda');ledger=Ledger(HOME/'engineering',reg['limits'],'M1:'+sha(HOME/'ENGINEERING_SOURCE_BINDING.json'))
    try:
        if ledger.counts['request_attempts']!=90 or ledger.counts['optimizer_attempts']!=12:raise ValueError('Exact preserved original engineering boundary required')
        actor=instrument(RelationActor().to('cuda'),ledger);opt=optimizer(actor)
        restore(actor,opt,ledger,read(HOME/'engineering/M1_final.json'),engineering_only=True)
        pool=Pool(ledger);groups=[];start=time.perf_counter()
        for i,tid in enumerate(reg['tasks']):
            groups.append([collect(actor,pool,tid,ledger,f'batch4_sample_{i}_{k}',2026111792+i*10+k) for k in range(4)])
        collection_s=time.perf_counter()-start;torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        rl=rl_update(actor,opt,groups,pool,ledger,'batch4_RL');sync('cuda');rl_s=time.perf_counter()-start;peak=memory('cuda')
        start=time.perf_counter();sl=sl_update(actor,opt,reg['tasks'],pool,ledger,'batch4_SL');sync('cuda');sl_s=time.perf_counter()-start
        if not rl['weight_changed'] or not sl['weight_changed']:raise ValueError('Production-shape update did not move parameters')
        ref=checkpoint(actor,opt,ledger,HOME/'engineering/M1_production_batch.pt',dict(closed=True,stage='M1_batch4'),True)
        result=dict(status='ENGINEERING_PASS',prior_result=dict(path='rc3_relation_drl_20261113/engineering/ENGINEERING_RESULT.json',sha256=reg['prior_result_sha256']),
            batch4_tasks=reg['tasks'],groups=[group_summary(g) for g in groups],RL=rl,SL=sl,memory=peak,
            timing=dict(collection_s=collection_s,RL_update_s=rl_s,SL_update_s=sl_s),counts=dict(ledger.counts),elapsed_s=ledger.elapsed(),
            checkpoint=ref,scientific_updates=0,performance_claim=False,
            conclusion='B4G4 physical execution, finite gradients, weight changes and bounded microbatch memory pass; engineering quality is not a scientific result')
        put(HOME/'engineering/M1_FINAL_RESULT.json',result);ledger.close('PASS_CLOSED')
        print(json.dumps({k:result[k] for k in ('status','memory','timing','counts','elapsed_s')},ensure_ascii=False),flush=True)
    except BaseException:ledger.close('TECHNICAL_FAILURE');raise

def main():
    from rc3_r04_m1_recovery_20261111 import supervisor
    supervisor.STAGE=HOME/'engineering'
    prior=sum(read(p)['elapsed_s'] for p in (HOME/'engineering/controls').glob('*/SUPERVISOR_RECEIPT.json'))
    prior=max(prior,read(HOME/'engineering/RESOURCE_ACCOUNTING.json')['elapsed_s'])
    folder=HOME/'engineering/controls/batch4'
    if folder.exists():raise RuntimeError('Batch probe already attempted')
    env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',RELATION_BATCH4_WORKER='1')
    receipt=supervisor.launch_supervised([str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.batch4_probe'],
        folder,'relation-M1-batch4',wall_limit_s=7200-prior,closeout_reserve_s=10,termination_tolerance_s=2,cwd=ROOT,env=env,
        entry_monotonic=time.monotonic(),entry_wall=time.time())
    print({k:receipt[k] for k in ('status','elapsed_s','worker_exit_code')},flush=True)
    if receipt['status']!='COMPLETE':raise RuntimeError('Batch probe failed; original costs retained')

if __name__=='__main__':
    if os.environ.get('RELATION_BATCH4_WORKER')=='1':worker()
    else:main()
