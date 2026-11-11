"""Registered import-only entry repair; original M2 sources/protocol untouched."""
import time
START_M, START_W = time.monotonic(), time.time()
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from . import common

PACKAGE, ROOT = common.HOME, common.ROOT
ATTEMPT = PACKAGE/'attempts/02'
RUN_ID = 'rc3-r04-m2-conditional-anchor-20261111-02'
read, put, sha = common.read, common.put, common.sha

def register():
    failure = read(PACKAGE/'WORKER_FAILURE.json')
    assert failure['error_type'] == 'ModuleNotFoundError' and failure['error'] == "No module named 'torch'"
    assert not any(failure['resources']['counts'].values())
    assert not (PACKAGE/'models').exists()
    previous = read(PACKAGE/'SOURCE_BINDING.json')
    for path, expected in previous['source_sha256'].items():
        if path.startswith(PACKAGE.name+'/'): assert sha(ROOT/path) == expected
    original = read(ROOT/'rc3_r04_m1_real_20261111/attempts/02/control/SUPERVISOR_RECEIPT.json')['command'][0]
    assert Path(sys.executable).resolve() == Path(original).resolve()
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    for name in ('M2_PROTOCOL.json','OLD_RESOURCE_SNAPSHOT.json'):
        with (ATTEMPT/name).open('xb') as f: f.write((PACKAGE/name).read_bytes())
    binding = json.loads(json.dumps(previous))
    binding['source_sha256'][(PACKAGE/'entry2.py').relative_to(ROOT).as_posix()] = sha(PACKAGE/'entry2.py')
    binding['artifact_sha256'][(ATTEMPT/'M2_PROTOCOL.json').relative_to(ROOT).as_posix()] = sha(ATTEMPT/'M2_PROTOCOL.json')
    binding['entry_repair'] = dict(kind='exact_M1_interpreter_and_distinct_output_directory',
        initial_binding_sha256=sha(PACKAGE/'SOURCE_BINDING.json'),
        preserved_failure_sha256=sha(PACKAGE/'WORKER_FAILURE.json'),
        interpreter=original, same_scientific_protocol_bytes=True,
        previous_optimizer_attempts=0, previous_step0_started=False, max_task_worker_launches=2)
    put(ATTEMPT/'SOURCE_BINDING.json',binding)
    put(ATTEMPT/'IMPLEMENTATION_REVIEW.json',dict(status='GO',source_binding_sha256=sha(ATTEMPT/'SOURCE_BINDING.json'),
        scope='import-only repair; frozen scientific worker/model/analysis/audit/protocol unchanged'))
    put(ATTEMPT/'STAGE_AUTHORIZATION.json',dict(authorized=True,authority='same explicit user M2 task; original import failure consumed zero optimizer/data',
        run_id=RUN_ID,source_binding_sha256=sha(ATTEMPT/'SOURCE_BINDING.json'),protocol_sha256=sha(ATTEMPT/'M2_PROTOCOL.json'),
        max_critic_optimizer_attempts=256,actor_optimizer_attempts=0,task_wall_cap_s=1800))
    prior = read(PACKAGE/'ENTRY_CONSUMPTION.json')['cumulative_new_task_wall_s']
    own = max(time.monotonic()-START_M,time.time()-START_W)
    put(ATTEMPT/'REGISTRATION_RECEIPT.json',dict(status='REGISTERED_BEFORE_NUMERICAL_COMPUTATION',
        wall_s=prior+own,own_registration_wall_s=own,prior_task_consumption_s=prior,
        source_binding_sha256=sha(ATTEMPT/'SOURCE_BINDING.json'),optimizer_attempts=0))
    print('Exact M1 Python import repair registered; failure/time preserved; no science change')

def launch():
    if (ATTEMPT/'control').exists(): raise FileExistsError('No third automatic worker')
    binding = read(ATTEMPT/'SOURCE_BINDING.json')
    assert Path(sys.executable).resolve() == Path(binding['entry_repair']['interpreter']).resolve()
    for name in ('__init__.py','supervisor.py'):
        p = ROOT/'rc3_r04_m1_recovery_20261111'/name
        assert sha(p) == binding['source_sha256'][p.relative_to(ROOT).as_posix()]
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    prior = read(ATTEMPT/'REGISTRATION_RECEIPT.json')['wall_s']
    receipt = launch_supervised([sys.executable,'-B','-X','utf8','-m',PACKAGE.name+'.entry2','worker'],
        ATTEMPT/'control',RUN_ID,wall_limit_s=1790-prior,closeout_reserve_s=20,termination_tolerance_s=5,
        cwd=ROOT,entry_monotonic=START_M,entry_wall=START_W,worker_heartbeat_path=ATTEMPT/'WORKER_HEARTBEAT.json')
    wall = max(time.monotonic()-START_M,time.time()-START_W)
    put(ATTEMPT/'ENTRY_CONSUMPTION.json',dict(status=receipt['status'],wall_s=wall,prior_task_consumption_s=prior,
        cumulative_new_task_wall_s=prior+wall,root_exit_confirmed=receipt['root_exit_confirmed'],
        job_active_processes_at_end=receipt['job_active_processes_at_end']))
    print(json.dumps(dict(status=receipt['status'],wall_s=wall,task_wall_s=prior+wall)))
    raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)

def worker():
    binding = read(ATTEMPT/'SOURCE_BINDING.json')
    assert sha(__file__) == binding['source_sha256'][Path(__file__).resolve().relative_to(ROOT).as_posix()]
    common.HOME, common.RUN_ID = ATTEMPT, RUN_ID
    from .worker import main
    main()

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('register','launch','worker'))
    globals()[parser.parse_args().action]()
