import os,json,time,hashlib,gzip
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
HOME=Path(__file__).resolve().parent
BASE='27deb2daf46f091cc2b9a7e3c475b7242eb2753e'
RUN='rc3-r08-b-joint-opportunity-20261113-r1'
D5=ROOT/'rc3_d5_revision_20261112'
H_ID='5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14'
def _json_default(v):
    if hasattr(v,'tolist'):return v.tolist()
    raise TypeError('Non-JSON evidence: '+type(v).__name__)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.pending') if replace else p
    with tmp.open('w' if replace else 'x',encoding='utf8',newline='\n') as f:
        json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False,default=_json_default);f.write('\n');f.flush();os.fsync(f.fileno())
    if replace:
        # A concurrent Windows reader can briefly deny delete sharing. Keep the
        # atomic destination intact and retry this exact completed write only.
        delays=(0.,.01,.02,.04,.08,.16,.32)
        for attempt,delay in enumerate(delays):
            if delay:time.sleep(delay)
            try:
                os.replace(tmp,p)
                break
            except OSError as exc:
                if getattr(exc,'winerror',None) not in (5,32,33) or attempt==len(delays)-1:raise
                append(HOME/'ATOMIC_IO_RETRIES.jsonl',dict(path=p.relative_to(ROOT).as_posix(),attempt=attempt+1,
                    winerror=exc.winerror,next_delay_s=delays[attempt+1],wall_unix_s=time.time(),operation='replace_same_fsynced_pending_file'))
def append(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a',encoding='utf8',newline='\n') as f:f.write(json.dumps(v,ensure_ascii=False,allow_nan=False,default=_json_default)+'\n');f.flush();os.fsync(f.fileno())
def archive(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False,default=_json_default)+'\n').encode()
    data=gzip.compress(raw,compresslevel=3,mtime=0)
    with p.open('ab') as f:offset=f.tell();f.write(data);f.flush();os.fsync(f.fileno())
    return dict(path=p.relative_to(ROOT).as_posix(),offset=offset,compressed_bytes=len(data),raw_sha256=hashlib.sha256(raw).hexdigest())
def unarchive(ref):
    with (ROOT/ref['path']).open('rb') as f:f.seek(ref['offset']);raw=gzip.decompress(f.read(ref['compressed_bytes']))
    if hashlib.sha256(raw).hexdigest()!=ref['raw_sha256']:raise ValueError('Archive identity')
    return json.loads(raw)
def gzput(p,v):
    raw=json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False,default=_json_default).encode();p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(gzip.compress(raw,compresslevel=3,mtime=0));f.flush();os.fsync(f.fileno())
def gzread(p):return json.loads(gzip.decompress(Path(p).read_bytes()))
def state(**kw):
    s=read(HOME/'STATUS.json');s.update(kw);put(HOME/'STATUS.json',s,True)
def identity():return read(HOME/'SCIENTIFIC_IDENTITY.json')['identity']

class Budget:
    def __init__(self):
        self.r=read(HOME/'RESOURCE_ACCOUNTING.json');self.counts=Counter(self.r['counts']);self.p=read(HOME/'PROTOCOL.json')
    def check(self):
        if time.time()>=self.p['deadline_unix_s']-300:raise TimeoutError('R08-B 1800s global wall with300s delivery reserve')
    def save(self):
        self.r['counts']=dict(self.counts);self.r['wall_s']=time.time()-self.p['origin_unix_s'];put(HOME/'RESOURCE_ACCOUNTING.json',self.r,True)
    def begin(self,rid):
        self.check()
        if self.counts['requests']>=900:raise RuntimeError('R08-B 900-request cap')
        self.counts['requests']+=1;append(HOME/'EVENTS.jsonl',dict(event='request_attempt',request_id=rid,attempt=self.counts['requests']));self.save();return time.perf_counter()
    def complete(self,rid,record,tick):
        ref=archive(HOME/'rollouts/REQUESTS.jsonl.gz',record);duration=time.perf_counter()-tick;self.counts['requests_completed']+=1
        append(HOME/'rollouts/REQUEST_POINTERS.jsonl',dict(request_id=rid,record=ref,full_request_s=duration));self.save();return dict(record=ref,full_request_s=duration)
    def fail(self,rid,e):
        self.counts['requests_failed']+=1;append(HOME/'EVENTS.jsonl',dict(event='request_failed',request_id=rid,error=repr(e)));self.save()
    def internal(self,value):
        for kind in ('counts','inclusive_s','exclusive_s'):
            for k,v in value.get(kind,{}).items():self.counts['internal_'+k+'_'+kind]+=v

