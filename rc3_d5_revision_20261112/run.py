import os,time,argparse
from .common import ROOT,HOME,RUN,read,put,state,sha

def launch(phase):
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    resource=read(HOME/'RESOURCE_ACCOUNTING.json');protocol=read(HOME/'REVISION_PROTOCOL.json')
    if read(HOME/'STATUS.json').get('closed'):raise PermissionError('This execution has reached its registered stopping exit')
    pool='data' if phase in ('data','joint-data') else 'engineering' if phase=='acceptance' else 'deployment' if phase=='deployment' else 'independent' if phase in ('confirmation','final') else 'training'
    limit=(3600-resource['wall']['data_s']) if phase in ('data','joint-data') else protocol['wall_limits'][pool]-resource['wall'][pool+'_s']
    limit=min(limit,345600-resource['wall']['total_s'])
    if limit<=40:raise RuntimeError('Insufficient approved remaining pool time')
    if phase!='data' and read(HOME/'FULL_TEACHER_MANIFEST.json')['status']!='PASS':raise PermissionError('Data not admitted')
    if phase in ('reference','science') and read(HOME/'TARGETED_ACCEPTANCE.json')['status']!='PASS':raise PermissionError('Acceptance not passed')
    if phase=='science' and read(HOME/'REFERENCE_MANIFEST.json')['status']!='PASS':raise PermissionError('Complete frozen-reference table required before P1')
    jobs=HOME/'controls';jobs.mkdir(exist_ok=True);attempt=len(list(jobs.glob(phase+'-*')))+1;job=f'{phase}-{attempt:02}'
    previous=read(HOME/'STATUS.json').get('active_job')
    if previous and not (jobs/previous/'SUPERVISOR_RECEIPT.json').exists():raise RuntimeError('Previous supervised job lacks an exit receipt; do not duplicate')
    state(active_job=job,active_phase=phase,controller_pid=os.getpid())
    env=dict(os.environ,D5_JOB_ID=job,D5_JOB_DEADLINE=str(time.time()+limit))
    command=[str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.worker',phase]
    receipt=launch_supervised(command,jobs/job,RUN+'-'+job,wall_limit_s=limit-2,closeout_reserve_s=20,termination_tolerance_s=2,cwd=ROOT,env=env,
        entry_monotonic=time.monotonic(),entry_wall=time.time(),worker_heartbeat_path=HOME/'jobs'/job/'WORKER_HEARTBEAT.json')
    # Reconcile outer supervisor time, including time before/after worker ledger.
    r=read(HOME/'RESOURCE_ACCOUNTING.json');recorded=r.get('jobs',{}).get(job,{}).get('wall_s',0);actual=float(receipt['elapsed_s']);extra=max(0,actual-recorded)
    r['wall']['total_s']+=extra;wp='training' if phase in ('data','joint-data','reference','science') else pool;r['wall'][wp+'_s']+=extra
    if phase in ('data','joint-data'):r['wall']['data_s']+=extra
    r.setdefault('jobs',{})[job]=dict(pool=pool,wall_s=max(actual,recorded),status=receipt['status']);put(HOME/'RESOURCE_ACCOUNTING.json',r,True)
    state(active_job=None,last_job=job,last_job_status=receipt['status'])
    if receipt['status']!='COMPLETE':raise RuntimeError('Worker failed; evidence and consumption preserved: '+job)
    return receipt

def advance():
    while True:
        s=read(HOME/'STATUS.json')
        if s.get('closed'):print(s);return
        if s['phase'] in ('P4','P5_CONFIRM','P5_FINAL'):
            launch({'P4':'deployment','P5_CONFIRM':'confirmation','P5_FINAL':'final'}[s['phase']]);continue
        if not (HOME/'FULL_TEACHER_MANIFEST.json').exists() or read(HOME/'FULL_TEACHER_MANIFEST.json')['status']!='PASS':launch('data');continue
        if not (HOME/'SCIENTIFIC_SOURCE_BINDING.json').exists():raise RuntimeError('Scientific source must be frozen before targeted optimizer/scientific execution')
        if not (HOME/'JOINT_DATA_ADMISSION.json').exists():launch('joint-data');continue
        if not (HOME/'TARGETED_ACCEPTANCE.json').exists():launch('acceptance');continue
        if read(HOME/'TARGETED_ACCEPTANCE.json')['status']!='PASS':print(read(HOME/'TARGETED_ACCEPTANCE.json'));return
        if not (HOME/'REFERENCE_MANIFEST.json').exists() or read(HOME/'REFERENCE_MANIFEST.json')['status']!='PASS':launch('reference');continue
        launch('science')

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--stage',choices=['data','joint-data','acceptance','reference','science','deployment','confirmation','final']);g.add_argument('--continue-authorized-plan');g.add_argument('--resume');g.add_argument('--status',action='store_true');a=p.parse_args()
    if a.stage:launch(a.stage)
    elif a.status:print(read(HOME/'STATUS.json'))
    else:
        if (a.resume or a.continue_authorized_plan)!=RUN:raise ValueError('Exact revision run_id required')
        advance()
