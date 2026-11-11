"""New-attempt journal, linked to prior consumption, independent hard supervisor."""
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

PACKAGE=Path(__file__).resolve().parent
ROOT=PACKAGE.parent
HOME=Path(os.environ['RC3_M1_ATTEMPT_DIR']).resolve()
if HOME.parent!=PACKAGE/'attempts':raise ValueError('Unregistered attempt output directory')
CAPS=dict(wall_s=1800,critic_optimizer_attempts=512,actor_optimizer_attempts=0,
          saved_trajectories=116,new_scenes=0,navigation_requests=0,teacher_requests=0,
          native_requests=0,H_requests=0,original60_reads=0,oldS3_reads=0)
ACTIVE_BUDGET=None

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(path):
    """Uncached fresh output/control hash, charged to the active attempt clock.

    This helper never changes scientific EVENTS or resource counters: the
    numerical reviewer must be able to prove their bytes stayed unchanged.
    Every invocation reads actual bytes; dynamic outputs never use input cache.
    """
    budget=ACTIVE_BUDGET
    check=budget.check if budget is not None else lambda:None
    trace=budget.tracer.event if budget is not None and budget.tracer is not None else lambda event,**fields:None
    source=Path(path);tick=time.perf_counter();size=0;h=hashlib.sha256()
    check()
    trace('file_hash_start',path=str(source),role='fresh_output_or_control',cache=False)
    try:
        with source.open('rb') as stream:
            while True:
                check();block=stream.read(1<<20);check()
                if not block:break
                size+=len(block);h.update(block)
        check();actual=h.hexdigest()
        trace('file_hash_end',path=str(source),bytes=size,seconds=time.perf_counter()-tick,
              sha256=actual,status='GO',role='fresh_output_or_control',cache=False)
        return actual
    except BaseException as exc:
        trace('file_hash_end',path=str(source),bytes=size,seconds=time.perf_counter()-tick,
              status='FAIL',error=repr(exc),role='fresh_output_or_control',cache=False)
        raise

def write_new(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

class Budget:
    def __init__(self,phase,started=None):
        global ACTIVE_BUDGET
        self.started_unix=float(os.environ['RC3_SUPERVISOR_START_UNIX'])
        self.started_monotonic=float(os.environ['RC3_SUPERVISOR_START_MONOTONIC'])
        self.run_id=os.environ['RC3_SUPERVISOR_RUN_ID']
        if self.run_id!='rc3-r04-m1-real-20261111-'+HOME.name or HOME.name not in ('01','02','03'):raise RuntimeError('Unregistered science run_id')
        self.prior_new_task_consumption_s=float(os.environ['RC3_M1_PRIOR_TASK_WALL_S'])
        self.path=HOME/'RESOURCE_ACCOUNTING.json'
        if self.path.exists():raise RuntimeError('Attempt already consumed; no retry/reset')
        self.phase=phase;self.tracer=None
        self.data=dict(schema='rc3-r04-m1-real-cumulative-resource-v1',prior_new_task_consumption_s=self.prior_new_task_consumption_s,step0_started=False,run_id=self.run_id,caps=CAPS,
            counts={k:0 for k in CAPS if k!='wall_s'},wall_s=0.,sessions=[],sequence=0,
            event_sha256='0'*64,failures=[],session_open=True,status='RUNNING',
            prior_attempt=dict(path='rc3_r04_m1_20261111/RESOURCE_ACCOUNTING.json',wall_s=2859.2566758999983,
                included_in_new_1800_limit=False,retained_original_bytes=True),
            clock='max(supervisor-origin monotonic elapsed, UTC elapsed); imports/preflight/restoration/fit/evaluate/save/recheck included',
            work_admission_cutoff_s=1650,cooperative_deadline_s=1780,supervisor_force_stop_s=1790,
            read_counter_scope='Instrumented scientific entry paths; complete file provenance timings in PREFLIGHT_TRACE.jsonl',
            opened_utc=datetime.now(timezone.utc).isoformat())
        ACTIVE_BUDGET=self
        self.emit('session_start')

    def elapsed(self):return self.prior_new_task_consumption_s+max(time.monotonic()-self.started_monotonic,time.time()-self.started_unix)

    def check(self):
        if self.elapsed()>=1780:raise RuntimeError('Reserved closeout deadline reached; no further work')

    def admit(self,operation):
        self.check()
        if self.elapsed()>=1650:raise RuntimeError('New work admission closed: '+operation)

    def persist(self):
        self.data['wall_s']=self.elapsed()
        temp=self.path.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8',newline='\n') as f:
            json.dump(self.data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        temp.replace(self.path)

    def emit(self,event,**info):
        self.data['sequence']+=1
        row=dict(sequence=self.data['sequence'],event=event,phase=self.phase,cumulative_wall_s=self.elapsed(),
                 previous_sha256=self.data['event_sha256'],**info)
        row['sha256']=hashlib.sha256(canonical(row)).hexdigest()
        with (HOME/'EVENTS.jsonl').open('a',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        self.data['event_sha256']=row['sha256'];self.persist()

    def charge(self,kind,amount=1,**info):
        self.check()
        if kind=='timing':
            times=self.data.setdefault('component_seconds',{});key=info['category']
            times[key]=times.get(key,0.)+info['seconds'];return
        count=self.data['counts'].get(kind,0)
        if kind in CAPS and count+amount>CAPS[kind]:raise RuntimeError('Resource cap exceeded: '+kind)
        self.data['counts'][kind]=count+amount
        if kind in ('critic_optimizer_attempts','saved_trajectories'):
            self.emit('predebit',kind=kind,amount=amount,**info)

    def close(self,status,error=None):
        if error is not None:self.data['failures'].append(dict(phase=self.phase,error=str(error)))
        self.data['sessions'].append(dict(phase=self.phase,wall_s=self.elapsed(),status=status))
        self.data.update(status=status,session_open=False)
        self.emit('session_close',status=status,error=str(error) if error else None)
