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
        # H/search calls are transferred by the caller in finally, including failures.
        try:
            result=super().request(rid,compute,**kw)
        except BaseException:raise
        # The old request writer also adds action rows; our finally already
        # counts them so failed and successful requests use the same boundary.
        self.counts['feature_states']-=len(result['record'].get('actions',[]))
        return result

def binding_identity():
    return {k:sha(HOME/f) for k,f in [('protocol_sha256','PROGRAM_PROTOCOL.json'),('plan_sha256','MASTER_PLAN.md'),('source_binding_sha256','SOURCE_BINDING.json')]}

def verify_binding():
    b=read(HOME/'SOURCE_BINDING.json')
    for name,expected in b['source_sha256'].items():
        if name.startswith(HOME.name+'/') and sha(ROOT/name)!=expected:raise ValueError('New executable source changed: '+name)
    for name in ('MASTER_PLAN.md','PROGRAM_PROTOCOL.json','DATA_ADMISSION.json','APPROVAL_RECEIPT.json'):
        if sha(HOME/name)!=b['program_artifacts'][name]:raise ValueError('Frozen program identity changed: '+name)
    if not read(HOME/'APPROVAL_RECEIPT.json')['execution_approved']:raise PermissionError('Program not approved')
    return b
