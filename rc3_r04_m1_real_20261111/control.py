"""Register/launch one bounded entry, using the accepted Windows supervisor.

No Torch import here. Existing source verifier and supervisor are reused.
All registration and launch consumption is retained across eligible repairs.
"""
import time
ENTRY_M=time.monotonic()
ENTRY_W=time.time()
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parent
BASE='992981414239da51c662b5060921ceea2312b7a1'
PRIOR_WALL=2885.2303396420666

def elapsed():return max(time.monotonic()-ENTRY_M,time.time()-ENTRY_W)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def consumption():
    entries=[]
    for p in sorted((PACKAGE/'attempts').glob('*/REGISTRATION_RECEIPT.json')):
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),kind='registration',wall_s=read(p)['wall_s']))
    for p in sorted((PACKAGE/'attempts').glob('*/ENTRY_CONSUMPTION.json')):
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),kind='supervised_entry',wall_s=read(p)['wall_s']))
    return entries,sum(x['wall_s'] for x in entries)+.555

def register(index):
    from .provenance_use import register_manifest
    attempt=PACKAGE/'attempts'/index
    if attempt.exists():raise RuntimeError('Registration already exists: no overwrite/reset')
    if index!='01':
        previous=PACKAGE/'attempts'/f'{int(index)-1:02d}'
        if not (previous/'ENTRY_REPAIR_AUTHORIZATION.json').exists():
            raise RuntimeError('Explicit eligibility review of preceding failure required')
        decision=read(previous/'ENTRY_REPAIR_AUTHORIZATION.json')
        if decision['eligible'] is not True:raise RuntimeError('Previous failure not eligible')
    attempt.mkdir(parents=True)
    manifest=register_manifest(ROOT,PACKAGE,attempt)
    old=[]
    for name in ('rc3_r04_m1_20261111','rc3_r04_m1_recovery_20261111'):
        path=ROOT/name/'RESOURCE_ACCOUNTING.json';data=read(path)
        old.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),wall_s=data['wall_s'],
                        status=data['status'],critic_optimizer_attempts=data['counts']['critic_optimizer_attempts']))
    run_id='rc3-r04-m1-real-20261111-'+index
    put(attempt/'ATTEMPT_LINK.json',dict(base_commit=BASE,run_id=run_id,old_attempts=old,
        old_engineering_wall_s=14.386909658434888,all_prior_wall_s=PRIOR_WALL,
        prior_consumption_not_reset=True,new_task_wall_cap_s=1800,max_worker_entries=3,
        old_budget_balance_reused=False,prior_new_task_entries=consumption()[0],
        scientific_files_unchanged={n:manifest[k] for n,k in (
            ('M1_EXECUTION_PROTOCOL.json','protocol_sha256'),('M1_DATA_AND_SOURCE_IDENTITY.json','data_sha256'),('BATCH_SCHEDULE.json','schedule_sha256'))}))
    put(attempt/'EXECUTION_CONTROLS.json',dict(run_id=run_id,scientific_modifications=[],
        total_new_task_wall_cap_s=1800,work_admission_cutoff_s=1650,cooperative_deadline_s=1780,
        supervisor_force_stop_s=1790,termination_tolerance_s=5,
        max_entries=3,critic_optimizer_attempts=512,per_configuration_attempts=256,actor_optimizer_attempts=0,
        retry_conditions=['zero Actor/Critic attempts','step0 not started','import/source classification/recording interface only',
            'no SHA mismatch, data corruption, sealed access or scientific semantics change'],
        scientific_inputs='original 33 fit / 16 holdout / 116 trajectories, unchanged batches',
        accepted_supervisor='existing Windows Job / temporary sleep prevention; no new synthetic suite',
        clock='Cumulative registration + full supervisor/worker elapsed across all entries; no truncation',
        pre_registration_stdlib_AST_check_s=.555))
    receipt=ROOT/'rc3_r04_m1_recovery_20261111/engineering/ENGINEERING_TEST_RESULTS.json'
    manifest.update(base_commit=BASE,run_id=run_id,
        accepted_engineering_receipt=dict(path=receipt.relative_to(ROOT).as_posix(),sha256=sha(receipt)),
        startup_files_sha256={n:sha(attempt/n) for n in ('ATTEMPT_LINK.json','EXECUTION_CONTROLS.json')})
    put(attempt/'SOURCE_BINDING.json',manifest)
    put(attempt/'STAGE_AUTHORIZATION.json',dict(authorized=True,run_id=run_id,
        source_binding_sha256=sha(attempt/'SOURCE_BINDING.json'),scientific_protocol_sha256=manifest['protocol_sha256'],
        authority='Explicit user task: base992981; existing classifier integration, real preflight and continuous M1; up to3 eligible entries, cumulative1800s; no G2/M3',
        total_new_task_wall_cap_s=1800,max_worker_entries=3,critic_optimizer_attempts=512,Actor=0,G2=False,M3=False))
    # Small source snapshots preserve every entry implementation through repairs.
    snapshot=attempt/'entry_source';snapshot.mkdir()
    for p in sorted(PACKAGE.glob('*.py')):
        with (snapshot/p.name).open('xb') as f:f.write(p.read_bytes())
    put(attempt/'REGISTRATION_RECEIPT.json',dict(status='REGISTERED',run_id=run_id,wall_s=elapsed(),
        source_binding_sha256=sha(attempt/'SOURCE_BINDING.json'),model_imports=0,optimizer_attempts=0,
        source_snapshots='entry_source: exact small new entry sources, not a repository backup'))
    print(json.dumps(dict(status='REGISTERED',run_id=run_id,new_task_consumption_s=consumption()[1])),flush=True)

def launch(index):
    from rc3_r04_m1_recovery_20261111 import supervisor
    supervisor.STAGE=PACKAGE  # Claims are new outputs; accepted implementation unchanged.
    attempt=PACKAGE/'attempts'/index
    if (attempt/'ENTRY_CONSUMPTION.json').exists():raise RuntimeError('Entry already consumed')
    binding=read(attempt/'SOURCE_BINDING.json');review=read(attempt/'IMPLEMENTATION_REVIEW.json')
    if review['status']!='GO' or review['source_binding_sha256']!=sha(attempt/'SOURCE_BINDING.json'):
        raise RuntimeError('Independent implementation review does not match')
    entries,prior=consumption()
    remaining=1800-prior
    if remaining<=150:raise RuntimeError('No admission budget remaining')
    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    put(attempt/'EXECUTION_SOURCE_COMMIT.json',dict(commit=git_head,source_binding_sha256=sha(attempt/'SOURCE_BINDING.json')))
    receipt=supervisor.launch_supervised(
        [sys.executable,'-B','-X','utf8','-m','rc3_r04_m1_real_20261111.worker'],attempt/'control',
        binding['run_id'],remaining,10,5,cwd=ROOT,
        env=dict(RC3_M1_ATTEMPT_DIR=str(attempt),RC3_M1_PRIOR_TASK_WALL_S=str(prior)),
        entry_monotonic=ENTRY_M,entry_wall=ENTRY_W,worker_heartbeat_path=attempt/'WORKER_HEARTBEAT.json')
    # Preserve raw worker resource. Parent-owned receipt includes exit and I/O.
    resource=read(attempt/'RESOURCE_ACCOUNTING.json') if (attempt/'RESOURCE_ACCOUNTING.json').exists() else None
    total=elapsed()
    put(attempt/'ENTRY_CONSUMPTION.json',dict(run_id=binding['run_id'],wall_s=total,
        supervisor_receipt_sha256=sha(attempt/'control/SUPERVISOR_RECEIPT.json'),
        prior_new_task_consumption_s=prior,cumulative_new_task_wall_s=prior+total,
        cumulative_all_research_wall_s=PRIOR_WALL+prior+total,
        old_consumption_preserved_s=PRIOR_WALL,worker_resource_sha256=sha(attempt/'RESOURCE_ACCOUNTING.json') if resource else None,
        counts=resource['counts'] if resource else None,step0_started=resource.get('step0_started',False) if resource else None,
        status=receipt['status'],execution_wall_violation=prior+total>1800,
        root_exit_confirmed=receipt['root_exit_confirmed'],job_active_processes_at_end=receipt['job_active_processes_at_end']))
    print(json.dumps(dict(status=receipt['status'],new_task_consumption_s=prior+total,
        worker_status=resource['status'] if resource else None,counts=resource['counts'] if resource else None)),flush=True)
    raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','launch'));parser.add_argument('index',choices=('01','02','03'))
    args=parser.parse_args();globals()[args.action](args.index)
