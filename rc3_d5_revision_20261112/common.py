import os,json,time,hashlib,gzip
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
HOME=Path(__file__).resolve().parent
BASE='20478a4dde2888f86abb9526c9d39ab71a27ee52'
RUN='d5-data-revision-20261112-r1'
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
    if replace:os.replace(tmp,p)
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
    """Persist attempts before work; resource consumption never follows checkpoint rollback."""
    def __init__(self,pool,job):
        self.pool=pool;self.job=job;self.stage=pool;self.path=HOME/'jobs'/job;self.path.mkdir(parents=True,exist_ok=True)
        self.base=read(HOME/'RESOURCE_ACCOUNTING.json');self.counts=Counter(self.base['counts'])
        self.start_m=time.monotonic();self.start_w=time.time();self.ids=set();self.status='RUNNING'
        self.check();self.save()
    def elapsed(self):return max(time.monotonic()-self.start_m,time.time()-self.start_w)
    def check(self):
        elapsed=self.elapsed();pool='training' if self.pool in ('data','reference','science') else self.pool
        limits=read(HOME/'REVISION_PROTOCOL.json')['wall_limits']
        if self.base['wall']['total_s']+elapsed>=345600-20:raise TimeoutError('96h cumulative boundary')
        if self.base['wall'].get(pool+'_s',0)+elapsed>=limits[pool]-20:raise TimeoutError('Cumulative '+pool+' time boundary')
        if self.pool=='data' and self.base['wall'].get('data_s',0)+elapsed>=3600-20:raise TimeoutError('3600s data boundary')
        if os.environ.get('D5_JOB_DEADLINE') and time.time()>=float(os.environ['D5_JOB_DEADLINE'])-20:raise TimeoutError('Supervised job boundary')
    def event(self,event,**kw):append(HOME/'EVENTS.jsonl',dict(event=event,job=self.job,pool=self.pool,job_wall_s=self.elapsed(),**kw))
    def save(self):
        value=dict(self.base);value['counts']=dict(self.counts);value['wall']=dict(self.base['wall']);elapsed=self.elapsed()
        value['wall']['total_s']+=elapsed;pool='training' if self.pool in ('data','reference','science') else self.pool
        value['wall'][pool+'_s']=value['wall'].get(pool+'_s',0)+elapsed
        if self.pool=='data':value['wall']['data_s']=value['wall'].get('data_s',0)+elapsed
        value['jobs']=dict(self.base.get('jobs',{}));value['jobs'][self.job]=dict(pool=self.pool,wall_s=elapsed,status=self.status)
        put(HOME/'RESOURCE_ACCOUNTING.json',value,True)
    def debit(self,key,n=1,limit=None):
        self.check()
        if limit is not None and self.counts[key]+n>limit:raise RuntimeError(key+' budget exhausted')
        self.counts[key]+=n;self.save()
    def optimizer(self,role,op):
        if self.pool=='engineering' and self.counts['engineering_optimizer_attempts']>=24:raise RuntimeError('Remaining P0 Actor quota exhausted')
        self.debit('actor_optimizer_attempts',limit=80000)
        self.counts['engineering_optimizer_attempts' if self.pool=='engineering' else 'scientific_optimizer_attempts']+=1
        self.event('optimizer_attempt',role=role,logical_update=op);self.save()
    def begin_request(self,purpose,rid):
        if purpose not in ('teacher','reference','train','eval','engineering','deployment','confirmation','final'):raise ValueError('Exact request purpose')
        if self.pool=='engineering' and self.counts['engineering_requests']>=60:raise RuntimeError('Remaining P0 request quota exhausted')
        self.debit('requests',limit=300000);self.counts[purpose+'_requests']+=1
        if self.pool=='engineering':self.counts['engineering_requests']+=1
        self.event('request_attempt',request_id=rid,purpose=purpose,attempt_number=self.counts['requests']);self.save();return time.perf_counter()
    def complete(self,purpose,rid,record,tick,folder):
        ref=archive(HOME/folder/'REQUESTS.jsonl.gz',record)
        self.counts['requests_completed']+=1;full=time.perf_counter()-tick
        append(HOME/folder/'REQUEST_POINTERS.jsonl',dict(request_id=rid,purpose=purpose,full_request_s=full,record=ref))
        self.event('request_complete',request_id=rid,purpose=purpose,full_request_s=full,record=ref);self.save();return ref,full
    def failed(self,purpose,rid,error):
        self.counts['requests_failed']+=1;self.event('request_failure',request_id=rid,purpose=purpose,error=repr(error));self.save()
    def internal(self,info):
        for k,v in info.get('counts',{}).items():self.counts['internal_'+k+'_calls']+=v
        for k,v in info.get('inclusive_s',{}).items():self.counts['internal_'+k+'_inclusive_s']+=v
        for k,v in info.get('exclusive_s',{}).items():self.counts['internal_'+k+'_exclusive_s']+=v
    def snapshot(self):return dict(counts=dict(self.counts),pool=self.pool,job=self.job,job_wall_s=self.elapsed())
    def close(self,status='COMPLETE'):self.status=status;self.save()
