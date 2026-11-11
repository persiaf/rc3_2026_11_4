import json
import os
import time
import hashlib
from pathlib import Path
from collections import Counter
from rc3_frozen_reference_pg_20261111.common import archive, append, csvrow, Ledger as OldLedger

ROOT=Path(__file__).resolve().parents[1]
HOME=Path(__file__).resolve().parent
BASE='2f6dc749ba3533d18f12254dd2ba50e1d52bcc40'
RUN='decoder-program-20261111-v1'
ROLES=('D5-PG','D5-SL','Full-PG')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    target=p.with_name(p.name+'.pending') if replace else p
    with target.open('x',encoding='utf8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    if replace:os.replace(target,p)

class Ledger(OldLedger):
    def __init__(self,start_m,start_w):
        self.stage='engineering';self.path=HOME/'P0';self.path.mkdir(exist_ok=False)
        self.start_m=start_m;self.start_w=start_w
        self.prior=read(HOME/'PRECURSOR_CONSUMPTION.json')['wall_s']+read(HOME/'REGISTRATION_RECEIPT.json')['wall_s']
        self.cap=dict(requests=60,actor_optimizer_attempts=24,critic_optimizer_attempts=0,wall_s=21600)
        self.counts=Counter(requests=1,requests_completed=0,requests_failed=1,
            actor_optimizer_attempts=0,actor_optimizer_completed=0,critic_optimizer_attempts=0,
            teacher_replays=0,new_teacher_requests=0,new_scenes=0,navigation_solves=0,
            original60_reads=0,oldS3_reads=0,full_H_calls=1,native_requests=0,feature_states=0,
            feature_s=0.,input_prepare_s=0.,request_io_s=0.,cache_rebuilds=0,backward_calls=0,
            internal_search_counts={},internal_search_inclusive_s={},internal_search_exclusive_s={})
        self.ids=set();self.status='RUNNING'
    def check(self):
        if self.elapsed()>self.cap['wall_s']-15:raise TimeoutError('Cumulative P0 wall boundary')
        if os.environ.get('RC3_CONTROLLER_DEADLINE') and time.time()>float(os.environ['RC3_CONTROLLER_DEADLINE'])-15:
            raise TimeoutError('Supervised job deadline')
    def request(self,rid,compute,**kw):
        # Terminal/search/forward costs are charged by runtime finally even
        # on failure. Persist their actual per-request values, without a
        # second aggregation in this writer.
        self.check();role=kw['role'];kind=kw.get('kind','engineering');op=kw.get('op',0)
        if rid in self.ids or self.counts['requests']>=self.cap['requests']:raise RuntimeError('Duplicate request or total request cap')
        self.ids.add(rid);self.counts['requests']+=1;self.event('request_debit',request_id=rid,role=role,kind=kind)
        tick=time.perf_counter()
        try:
            result=compute();r=result['record'];r.update(request_id=rid,method=role,opportunity=op,request_kind=kind,stage=self.stage)
            io_start=time.perf_counter();ref=archive(self.path/'REQUESTS.jsonl.gz',r);io_s=time.perf_counter()-io_start
            self.counts['request_io_s']+=io_s;self.counts['requests_completed']+=1
            self.counts['input_prepare_s']+=r.get('timing',{}).get('input_prepare_s',0.)
            row=dict(request_id=rid,role=role,width=r['spec']['width'],height=r['spec']['height'],physical_key=r['physical_key'],
                **r['metrics'],valid=bool(r['plan']['validation']['valid_model_plan']),H_calls=r['counts']['full_H_calls'],
                internal_search=r['internal_search']['counts'],**ref)
            csvrow(self.path/'REQUEST_INDEX.csv',row);self.event('request_complete',request_id=rid,record=ref)
            csvrow(self.path/'REQUEST_TIMING.csv',dict(request_id=rid,full_request_s=time.perf_counter()-tick,archive_IO_s=io_s))
            return result
        except BaseException as exc:
            self.counts['requests_failed']+=1;self.event('request_failure',request_id=rid,error=repr(exc));raise

def current_binding():
    return HOME/read(HOME/'ACTIVE_SOURCE_BINDING.json')['path'] if (HOME/'ACTIVE_SOURCE_BINDING.json').exists() else HOME/'SOURCE_BINDING.json'

def binding_identity():
    return dict(protocol_sha256=sha(HOME/'PROGRAM_PROTOCOL.json'),plan_sha256=sha(HOME/'MASTER_PLAN.md'),source_binding_sha256=sha(current_binding()))

def verify_binding():
    b=read(current_binding())
    for name,expected in b['source_sha256'].items():
        if name.startswith(HOME.name+'/') and sha(ROOT/name)!=expected:raise ValueError('New executable source changed: '+name)
    for name in ('MASTER_PLAN.md','PROGRAM_PROTOCOL.json','DATA_ADMISSION.json','APPROVAL_RECEIPT.json'):
        if sha(HOME/name)!=b['program_artifacts'][name]:raise ValueError('Frozen program identity changed: '+name)
    if not read(HOME/'APPROVAL_RECEIPT.json')['execution_approved']:raise PermissionError('Program not approved')
    return b
