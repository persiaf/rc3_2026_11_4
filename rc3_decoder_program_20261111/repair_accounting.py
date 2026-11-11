"""One registered recorder-interface repair; no optimizer or repeated P0 suite."""
import time
START_M,START_W=time.monotonic(),time.time()
import ast
import csv
import copy
import os
import io
import argparse
from collections import Counter
from pathlib import Path
from .common import HOME,ROOT,RUN,read,put,sha,Ledger,verify_binding,current_binding

def register():
    if (HOME/'SOURCE_BINDING_R02.json').exists():raise FileExistsError('No repair-registration reset')
    b=read(HOME/'SOURCE_BINDING.json')
    for p in HOME.glob('*.py'):
        ast.parse(p.read_text(encoding='utf8'));b['source_sha256'][p.relative_to(ROOT).as_posix()]=sha(p)
    b.update(revision='R02_recorder_only',prior_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),
        change='Write actual per-request H/internal search values; remove inherited aggregation duplication. Actor,Trainer,MDP,loss,protocol,data and gates unchanged.')
    put(HOME/'SOURCE_BINDING_R02.json',b)
    put(HOME/'ACTIVE_SOURCE_BINDING.json',dict(path='SOURCE_BINDING_R02.json',sha256=sha(HOME/'SOURCE_BINDING_R02.json')))
    put(HOME/'ACCOUNTING_REPAIR_REGISTRATION.json',dict(root_cause='Adapter suppressed persisted per-request counts to avoid old writer double aggregation',
        repair_attempt=1,source_binding_sha256=sha(HOME/'SOURCE_BINDING_R02.json'),
        scope='One ordinary training 4x3 FrozenSL complete request, no optimizer, no P0 suite rerun; same input/method/MDP/H',
        additional_complete_request_cap=1,additional_Actor_attempt_cap=0,wall_s=time.perf_counter()-START_M,
        cumulative_prior_wall_s=read(HOME/'P0_ENTRY_CONSUMPTION.json')['cumulative_wall_s'],
        original_P0_files_preserved=True,missing_old_per_request_search_details='UNKNOWN; do not recompute routes to fill logs',
        authorization='Current user already permits source-compatible engineering fixes within P0 caps'))
    print('R02 recorder-only source registered; one bounded ordinary request, zero optimizer.')

def worker():
    b=verify_binding();old=read(HOME/'P0/RESOURCE_ACCOUNTING.json');reg=read(HOME/'ACCOUNTING_REPAIR_REGISTRATION.json')
    ledger=Ledger.__new__(Ledger);ledger.stage='engineering_accounting_repair';ledger.path=HOME/'P0_ACCOUNTING_REPAIR';ledger.path.mkdir(exist_ok=False)
    ledger.start_m=START_M;ledger.start_w=START_W;ledger.prior=reg['cumulative_prior_wall_s']+reg['wall_s'];ledger.cap=old['caps']
    ledger.counts=Counter(copy.deepcopy(old['counts']));ledger.status='RUNNING'
    with (HOME/'P0/REQUEST_INDEX.csv').open(encoding='utf8',newline='') as f:ledger.ids={r['request_id'] for r in csv.DictReader(f)}
    before=copy.deepcopy(ledger.counts)
    from rc3_r04_m1_real_20261111.provenance_use import ScientificTracedProvenance,verify_real_environment
    trace=ScientificTracedProvenance(ROOT,ledger.path,b,ledger.check,RUN+'-accounting-r02');trace.install_import_guard()
    with trace.stage('recorder_repair_exact_saved_input'):
        import torch
        import numpy as np
        from . import actor,trainer,runtime
        from rc3_conservative_refinement import saved_inputs as si,runtime as source
        actual=verify_real_environment(trace,torch,dict(actor=actor,trainer=trainer,runtime=runtime))
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        si.digest=source.file_sha256=trace.hash_file;si.read=trace.read_json
        from rc3_global_local_trainer import navigation_compat,approval
        from rc3_global_local_impl_review import runtime_guards
        navigation_compat.read=approval.read=trace.read_json
        navigation_compat.digest=approval.digest=runtime_guards.file_hash=trace.hash_file
        np_load_original=np.load;torch_load_original=torch.load
        def np_load(p,*a,**kw):return np_load_original(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,os.PathLike)) else p,*a,**kw)
        def torch_load(p,*a,**kw):return torch_load_original(io.BytesIO(trace.read_bytes(p)) if isinstance(p,(str,os.PathLike)) and not Path(p).resolve().is_relative_to(HOME) else p,*a,**kw)
        np.load=np_load;torch.load=torch_load
        inputs=si.SavedInputs();spec=read(HOME/'PROGRAM_PROTOCOL.json')['P0']['ordinary_training_specs'][1]
        inputs.allowed[si.spec_key(spec)]=spec
        result=runtime.request(ledger,inputs,spec,'p0:accounting-r02:1','FrozenSL',greedy=True)
    record=result['record']
    assert ledger.counts['requests']==before['requests']+1 and ledger.counts['actor_optimizer_attempts']==before['actor_optimizer_attempts']
    assert record['counts']['full_H_calls']==ledger.counts['full_H_calls']-before['full_H_calls']==1
    for k in ('counts','inclusive_s','exclusive_s'):
        for name,value in record['internal_search'][k].items():
            delta=ledger.counts['internal_search_'+k].get(name,0)-before['internal_search_'+k].get(name,0)
            assert abs(delta-value)<1e-9
    trace.check_loaded_modules();put(ledger.path/'ACTUAL_DEPENDENCY_VERIFICATION.json',dict(trace.summary(),real_Torch=actual))
    ledger.status='COMPLETE';put(ledger.path/'RESOURCE_ACCOUNTING.json',dict(ledger.snapshot(),input_counts=inputs.counts))
    put(ledger.path/'CHECKS.json',dict(status='PASS',new_requests=1,new_Actor_attempts=0,
        per_request_H_equals_actual_global_delta=True,per_request_internal_search_equals_actual_global_delta=True,
        original_P0_not_repeated=True,original_missing_details_not_fabricated=True))
    print(dict(status='PASS',new_requests=1,Actor_attempts=0,H_calls=record['counts']['full_H_calls'],search_counts=record['internal_search']['counts']),flush=True)

def launch():
    verify_binding();reg=read(HOME/'ACCOUNTING_REPAIR_REGISTRATION.json')
    if (HOME/'P0_ACCOUNTING_CONTROL').exists():raise FileExistsError('No repeated accounting repair')
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    prior=reg['cumulative_prior_wall_s']+reg['wall_s'];remaining=21600-prior
    command=[str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.repair_accounting','worker']
    receipt=launch_supervised(command,HOME/'P0_ACCOUNTING_CONTROL',RUN+'-accounting-r02',wall_limit_s=remaining-2,closeout_reserve_s=15,termination_tolerance_s=2,cwd=ROOT,
        entry_monotonic=START_M,entry_wall=START_W,worker_heartbeat_path=HOME/'P0_ACCOUNTING_REPAIR/WORKER_HEARTBEAT.json')
    wall=max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/'ACCOUNTING_REPAIR_ENTRY_CONSUMPTION.json',dict(status=receipt['status'],entry_wall_s=wall,prior_charged_s=prior,cumulative_wall_s=prior+wall,
        root_exit_confirmed=receipt['root_exit_confirmed'],active_processes_at_end=receipt['job_active_processes_at_end']))
    print(dict(status=receipt['status'],cumulative_wall_s=prior+wall));raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run','worker']);a=p.parse_args()
    {'register':register,'run':launch,'worker':worker}[a.action]()
