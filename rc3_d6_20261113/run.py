import os,time,sys,argparse
from .common import ROOT,HOME,RUN,read,put,sha,state

PHASES=['diagnose','A_128','B_128','SL_128','eval128','A_256','B_256','SL_256','eval256']
def main(resume=False):
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    protocol=read(HOME/'PROTOCOL.json')
    for phase in PHASES:
        marker=HOME/'controls'/phase/'SUPERVISOR_RECEIPT.json'
        if marker.exists():
            if read(marker)['status']=='COMPLETE':continue
            raise RuntimeError('Failure preserved; no automatic free retry: '+phase)
        if read(HOME/'STATUS.json').get('closed'):raise RuntimeError('Scientific identity already closed')
        if phase!='diagnose' and read(HOME/'COST_DECOMPOSITION_VALIDATION.json')['status']!='PASS':raise PermissionError('Cost decomposition not passed')
        remaining=protocol['deadline_unix_s']-time.time()-600
        if remaining<40:raise TimeoutError('D6 remaining original wall insufficient')
        state(phase=phase,status='RUNNING',active_phase=phase)
        env=dict(os.environ,D6_JOB_DEADLINE=str(time.time()+remaining))
        cmd=[str(ROOT/'rc3_learning/.venv/Scripts/python.exe'),'-B','-X','utf8','-m',HOME.name+'.worker',phase]
        receipt=launch_supervised(cmd,HOME/'controls'/phase,RUN+'-'+phase,wall_limit_s=remaining,closeout_reserve_s=20,
            termination_tolerance_s=2,cwd=ROOT,env=env,entry_monotonic=time.monotonic(),entry_wall=time.time())
        if receipt['status']!='COMPLETE':raise RuntimeError('D6 phase actual failure preserved: '+phase)
    state(active_phase=None,status='COMPLETE',closed=True)
    print(read(HOME/'STATUS.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--run');g.add_argument('--resume');g.add_argument('--status',action='store_true');a=p.parse_args()
    if a.status:print(read(HOME/'STATUS.json'))
    else:
        if (a.run or a.resume)!=RUN:raise ValueError('Exact D6 identity required')
        main(bool(a.resume))
