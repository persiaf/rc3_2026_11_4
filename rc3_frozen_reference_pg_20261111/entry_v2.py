"""Import-registry repair only: preserve all science/source/protocol bytes."""
import time
START_M,START_W=time.monotonic(),time.time()
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from . import common
PACKAGE=common.HOME
HOME=PACKAGE/'execution_v2'
ROOT=common.ROOT
read,put,sha=common.read,common.put,common.sha

def register():
    failure=read(PACKAGE/'engineering/FAILURE.json')
    assert failure['error']=='Unregistered actual dependency: rc3_r04_review_20261111/__init__.py'
    r=read(PACKAGE/'engineering/RESOURCE_ACCOUNTING.json')
    assert r['counts']['requests']==r['counts']['actor_optimizer_attempts']==0
    original=read(PACKAGE/'SOURCE_BINDING.json')
    for name,expected in original['source_sha256'].items():
        if name.startswith(PACKAGE.name+'/'):assert sha(ROOT/name)==expected
    HOME.mkdir(exist_ok=False)
    for name in ('PROTOCOL.json','SCHEDULE.json'):
        with (HOME/name).open('xb') as f:f.write((PACKAGE/name).read_bytes())
    original['source_sha256'][(PACKAGE/'entry_v2.py').relative_to(ROOT).as_posix()]=sha(__file__)
    additions={}
    for p in sorted((ROOT/'rc3_r04_review_20261111').glob('*.py')):
        name=p.relative_to(ROOT).as_posix();payload=subprocess.check_output(['git','show',common.BASE+':'+name],cwd=ROOT)
        expected=hashlib.sha256(payload).hexdigest();assert sha(p)==expected
        if name in original['source_sha256']:assert original['source_sha256'][name]==expected
        original['source_sha256'][name]=expected;additions[name]=expected
    for name in ('PROTOCOL.json','SCHEDULE.json'):
        original['artifact_sha256'][(HOME/name).relative_to(ROOT).as_posix()]=sha(HOME/name)
    original['entry_repair']=dict(kind='missing_immutable_A1_package_source_registration',added_baseline_sources=additions,
        prior_source_binding_sha256=sha(PACKAGE/'SOURCE_BINDING.json'),failure_sha256=sha(PACKAGE/'engineering/FAILURE.json'),
        scientific_protocol_and_worker_bytes_unchanged=True,old_requests=0,old_optimizer_attempts=0)
    put(HOME/'SOURCE_BINDING.json',original)
    put(HOME/'IMPLEMENTATION_REVIEW.json',dict(status='GO',source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),
        scope='Exact existing A1 package registered, no algorithm/schedule/budget/gate change'))
    for stage in ('engineering','science'):
        put(HOME/(stage.upper()+'_AUTHORIZATION.json'),dict(authorized=True,authority='same explicit user engineering+science task',
            source_binding_sha256=sha(HOME/'SOURCE_BINDING.json'),protocol_sha256=sha(HOME/'PROTOCOL.json'),
            prerequisite='real engineering acceptance PASS' if stage=='science' else 'same-budget source interface repair',stage=stage))
    prior=read(PACKAGE/'ENGINEERING_ENTRY_CONSUMPTION.json')['stage_cumulative_wall_s']
    own=max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/'REGISTRATION_RECEIPT.json',dict(status='BOUND_BEFORE_ACTUAL_REQUEST_OR_UPDATE',wall_s=prior+own,
        prior_failed_engineering_wall_s=prior,own_repair_registration_s=own,prior_requests=0,prior_optimizer_attempts=0))
    print('A1 package exact baseline source registration repaired; original protocol/source preserved')

def launch(stage):
    if (HOME/(stage+'_control')).exists():raise FileExistsError('No repeat entry')
    if stage=='science' and read(HOME/'engineering/CHECKS.json')['status']!='PASS':raise ValueError('Real acceptance required')
    binding=read(HOME/'SOURCE_BINDING.json')
    for name in ('__init__.py','supervisor.py'):
        p=ROOT/'rc3_r04_m1_recovery_20261111'/name
        assert sha(p)==binding['source_sha256'][p.relative_to(ROOT).as_posix()]
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    prior=read(HOME/'REGISTRATION_RECEIPT.json')['wall_s'] if stage=='engineering' else 0.
    cap=read(HOME/'PROTOCOL.json')['caps'][stage]['wall_s']
    receipt=launch_supervised([str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',PACKAGE.name+'.entry_v2','worker',stage],
        HOME/(stage+'_control'),'frozen-reference-PG-20261111-'+stage+'-02',wall_limit_s=cap-prior-2,
        closeout_reserve_s=15,termination_tolerance_s=2,cwd=ROOT,entry_monotonic=START_M,entry_wall=START_W,
        worker_heartbeat_path=HOME/stage/'WORKER_HEARTBEAT.json')
    elapsed=max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/(stage.upper()+'_ENTRY_CONSUMPTION.json'),dict(status=receipt['status'],wall_s=elapsed,prior_engineering_wall_s=prior,
        stage_cumulative_wall_s=prior+elapsed,root_exit_confirmed=receipt['root_exit_confirmed'],active_processes_at_end=receipt['job_active_processes_at_end']))
    print({'stage':stage,'status':receipt['status'],'wall_s':prior+elapsed})
    raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)

def worker(stage):
    binding=read(HOME/'SOURCE_BINDING.json');assert sha(__file__)==binding['source_sha256'][Path(__file__).resolve().relative_to(ROOT).as_posix()]
    common.HOME=HOME
    from .worker import main
    main(stage)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('register','engineering','science','worker'));p.add_argument('stage',nargs='?')
    a=p.parse_args()
    if a.action=='register':register()
    elif a.action=='worker':worker(a.stage)
    else:launch(a.action)
