import os,sys,time
from .common import ROOT,HOME,RUN,read,put

def main():
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    if (HOME/'controls/SUPERVISOR_RECEIPT.json').exists():raise RuntimeError('Existing R08-B attempt retained; no reset/free rerun')
    remaining=read(HOME/'PROTOCOL.json')['deadline_unix_s']-time.time()-300
    if remaining<30:raise TimeoutError('Original R08-B remaining wall exhausted')
    receipt=launch_supervised([sys.executable,'-B','-X','utf8','-m',HOME.name+'.worker'],HOME/'controls',RUN,
        wall_limit_s=remaining,closeout_reserve_s=20,termination_tolerance_s=2,cwd=ROOT,env=os.environ.copy(),entry_monotonic=time.monotonic(),entry_wall=time.time())
    print(dict(status=receipt['status'],wall_s=receipt['elapsed_s']),flush=True)
    if receipt['status']!='COMPLETE':raise RuntimeError('R08-B actual failure preserved')
if __name__=='__main__':main()
