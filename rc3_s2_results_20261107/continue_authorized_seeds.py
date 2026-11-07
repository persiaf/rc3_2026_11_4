"""Sequentially invoke the two remaining already-authorized production seeds.

No algorithm, data, model or optimizer code lives here. The production authority
and stage chain revalidate every input. A failure never starts a later seed.
"""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01'
AUTH=ROOT/'rc3_global_local_execution_v3/authorizations/20261107_S2/S2_AUTHORIZATION.json'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def new(p,d):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def completed(seed):
    p=RUN/f'stage_admission/S2/{seed}/COMPLETE.json'
    failed=p.with_name('FAILED.json')
    if failed.exists():raise RuntimeError(f'Production S2 seed{seed} failed; later seeds remain unstarted')
    if not p.exists():return None
    c=read(p)
    if c['status']!='COMPLETE' or c['stage_key']!=f'S2/{seed}' or c['error'] is not None:
        raise RuntimeError('Not a successful fixed S2 endpoint')
    for r in c['cumulative_stage_resources'].values():
        if r['elapsed_wall_s']>7200:raise RuntimeError('Completed seed exceeds fixed wall cap')
    final=read(RUN/f'seed_{seed}/S2/FINAL.json')
    if final['status']!='COMPLETE_FIXED_ENDPOINTS':raise RuntimeError('Missing fixed RL/SL endpoints')
    for phase in ['RL','continued_SL']:
        cp=RUN/f'seed_{seed}/{phase}/update_0512.pt'
        m=read(cp.with_suffix('.pt.json'))
        if m['cursors'][phase]!=512 or m['sha256']!=digest(cp):raise RuntimeError('Endpoint cursor/checksum mismatch')
    return digest(p)

def main():
    folder=HERE/'production_invocations';folder.mkdir(exist_ok=True)
    start=folder/'ORCHESTRATOR_STARTED.json'
    new(start,dict(started_utc=now(),authorization_sha256=digest(AUTH),first_seed_already_running=11622026,
        remaining_seeds=[11622027,11622028],automatic_retry=False,scope='Exact authorized production CLI; no scientific recipe or budget changes'))
    try:
        previous=11622026
        while not completed(previous):time.sleep(2)
        for seed in [11622027,11622028]:
            parent_hash=completed(previous)
            if (RUN/f'stage_admission/S2/{seed}').exists():raise RuntimeError('Next stage already admitted; refuse duplicate invocation')
            argv=[str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-m','rc3_global_local_execution_v3.runner',
                '--stage','S2','--seed',str(seed),'--experiment-id','rc3_v3_global_local_20261106_01',
                '--output',RUN.relative_to(ROOT).as_posix(),'--existing-run',
                '--source-binding','rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json',
                '--protocol-approval','rc3_global_local_execution_v3/authorizations/20261106_S1_B/SOURCE_REVISION_APPROVAL.json',
                '--stage-authorization',AUTH.relative_to(ROOT).as_posix()]
            t=time.perf_counter();utc=now()
            new(folder/f'{seed}_START.json',dict(seed=seed,started_utc=utc,argv=argv,
                previous_seed=previous,previous_completion_sha256=parent_hash,authorization_sha256=digest(AUTH)))
            print(f'Starting authorized S2 seed {seed}',flush=True)
            with (folder/f'{seed}_stdout.log').open('x',encoding='utf8') as out,(folder/f'{seed}_stderr.log').open('x',encoding='utf8') as err:
                result=subprocess.run(argv,cwd=ROOT,stdout=out,stderr=err)
            new(folder/f'{seed}_PROCESS_RESULT.json',dict(seed=seed,returncode=result.returncode,
                process_wall_s=time.perf_counter()-t,started_utc=utc,ended_utc=now(),no_retry=True))
            if result.returncode!=0:raise RuntimeError(f'Production seed{seed} exited {result.returncode}; no retry or next seed')
            if not completed(seed):raise RuntimeError('Process ended without a complete fixed endpoint; no next seed')
            previous=seed
        new(folder/'ALL3_COMPLETE.json',dict(status='COMPLETE',seeds=[11622026,11622027,11622028],
            completion_sha256={str(s):completed(s) for s in [11622026,11622027,11622028]},ended_utc=now()))
    except Exception as exc:
        new(folder/'STOPPED.json',dict(status='STOPPED_NO_RETRY',error=type(exc).__name__,message=str(exc),ended_utc=now()))
        raise

if __name__=='__main__':main()
