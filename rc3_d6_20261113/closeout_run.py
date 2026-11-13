"""Bound read-only closeout commands, retaining the original7200s deadline."""
import os,sys,time
from .common import ROOT,HOME,RUN,read,put,sha

def main(mode):
    from .run import PHASES
    for phase in PHASES:
        r=read(HOME/'controls'/phase/'SUPERVISOR_RECEIPT.json')
        if r['status']!='COMPLETE' or r['job_active_processes_at_end']!=0:raise RuntimeError('Actual scientific phase incomplete: '+phase)
    if not read(HOME/'STATUS.json')['closed']:raise ValueError('Scientific identity still open')
    if not (HOME/'RESOURCE_SCIENTIFIC_FINAL_RAW.json').exists():put(HOME/'RESOURCE_SCIENTIFIC_FINAL_RAW.json',read(HOME/'RESOURCE_ACCOUNTING.json'))
    raw=read(HOME/'RESOURCE_SCIENTIFIC_FINAL_RAW.json')
    for name,value in read(HOME/'SOURCE_BINDING.json')['source_sha256'].items():
        if name.startswith(HOME.name+'/') and sha(ROOT/name)!=value:raise ValueError('Preregistered core source changed: '+name)
    source='postprocess.py' if mode=='audit' else 'verify_remote_lfs.py'
    remaining=read(HOME/'PROTOCOL.json')['deadline_unix_s']-time.time()-120
    if remaining<40:raise TimeoutError('Original D67200s closeout budget exhausted')
    put(HOME/f'{mode.upper()}_SOURCE_BINDING.json',dict(source=source,sha256=sha(HOME/source),wrapper_sha256=sha(HOME/'closeout_run.py'),
        remaining_original_wall_s=remaining,new_model_forward_or_optimizer=False,science_counters_preserved=True))
    from rc3_r04_m1_recovery_20261111.supervisor import launch_supervised
    cmd=[sys.executable,'-B','-X','utf8','-m',HOME.name+'.postprocess'] if mode=='audit' else [sys.executable,'-B','-X','utf8',str(HOME/'verify_remote_lfs.py')]
    receipt=launch_supervised(cmd,HOME/'closeout_controls'/mode,RUN+'-'+mode,wall_limit_s=remaining,closeout_reserve_s=20,
        termination_tolerance_s=2,cwd=ROOT,env=os.environ.copy(),entry_monotonic=time.monotonic(),entry_wall=time.time())
    current=read(HOME/'RESOURCE_ACCOUNTING.json')
    if current['counts']!=raw['counts']:raise ValueError('Read-only closeout changed scientific request/optimizer counters')
    current['wall']['total_s']=time.time()-read(HOME/'PROTOCOL.json')['origin_unix_s'];current['jobs'][mode+'-closeout']=dict(status=receipt['status'],wall_s=receipt['elapsed_s'])
    put(HOME/'RESOURCE_ACCOUNTING.json',current,True)
    print(dict(mode=mode,status=receipt['status'],wall_s=receipt['elapsed_s']),flush=True)
    if receipt['status']!='COMPLETE':raise RuntimeError('Actual closeout failure preserved: '+mode)
if __name__=='__main__':main(sys.argv[1])
