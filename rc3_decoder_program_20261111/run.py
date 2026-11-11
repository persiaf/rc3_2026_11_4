import time
START_M,START_W=time.monotonic(),time.time()
import argparse
import subprocess
from .common import HOME,ROOT,RUN,read,put,sha,verify_binding

def continue_plan(run_id,resume=False):
    if run_id!=RUN:raise ValueError('Exact approved run_id required')
    verify_binding();status=read(HOME/'STATUS.json');admission=read(HOME/'DATA_ADMISSION.json')
    if admission['status']!='PASS':
        print(dict(run_id=RUN,status=status['status'],P1_admission=admission['status'],scientific_training='NOT_RUN',
            model_forward_calls_added=0,optimizer_attempts_added=0,planning_attempts_added=0,
            action='P6 closeout. No implicit teacher generation or unequal-task SL substitution.'));return
    raise RuntimeError('This source version is bound to the failed data-admission closure; it cannot launch another data design')

def engineering():
    b=verify_binding()
    if (HOME/'P0_CONTROL').exists() or (HOME/'P0').exists():raise FileExistsError('Existing attempt: do not reset or repeat P0')
    for name in ('__init__.py','supervisor.py'):
        p=ROOT/'rc3_r04_m1_recovery_20261111'/name
        if sha(p)!=b['source_sha256'][p.relative_to(ROOT).as_posix()]:raise ValueError('Supervisor source changed')
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    previous=read(HOME/'PRECURSOR_CONSUMPTION.json')['wall_s']+read(HOME/'REGISTRATION_RECEIPT.json')['wall_s']
    cap=read(HOME/'PROGRAM_PROTOCOL.json')['budget']['P0_wall_s']
    command=[str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.worker']
    receipt=launch_supervised(command,HOME/'P0_CONTROL',RUN+'-P0',wall_limit_s=cap-previous-2,closeout_reserve_s=15,termination_tolerance_s=2,cwd=ROOT,
        entry_monotonic=START_M,entry_wall=START_W,worker_heartbeat_path=HOME/'P0/WORKER_HEARTBEAT.json')
    elapsed=max(time.monotonic()-START_M,time.time()-START_W)
    put(HOME/'P0_ENTRY_CONSUMPTION.json',dict(status=receipt['status'],entry_wall_s=elapsed,prior_charged_s=previous,cumulative_wall_s=elapsed+previous,
        root_exit_confirmed=receipt['root_exit_confirmed'],active_processes_at_end=receipt['job_active_processes_at_end']))
    print(dict(controller=receipt['status'],wall_s=elapsed+previous,program=read(HOME/'STATUS.json')['status']));raise SystemExit(0 if receipt['status']=='COMPLETE' else 1)

def main():
    p=argparse.ArgumentParser(description='Approved D5 program: exact source-bound P0 and conditional continuation, no force switch')
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--stage',choices=['engineering']);g.add_argument('--continue-authorized-plan');g.add_argument('--resume');g.add_argument('--status',action='store_true')
    a=p.parse_args()
    if a.stage:engineering()
    elif a.continue_authorized_plan:continue_plan(a.continue_authorized_plan)
    elif a.resume:continue_plan(a.resume,True)
    else:print(read(HOME/'STATUS.json'))

if __name__=='__main__':main()
