import os,json,time,hashlib,gzip
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
HOME=Path(__file__).resolve().parent
BASE='539ead5ff52cb110a91736d6550cb988171d4d9b'
RUN='rc3-d6-causal-tail-20261113-r1'
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
    def __init__(self,job):
        self.job=job;self.pool='science';self.path=HOME/'jobs'/job;self.path.mkdir(parents=True,exist_ok=True)
        self.resource=read(HOME/'RESOURCE_ACCOUNTING.json');self.counts=Counter(self.resource['counts']);self.ids=set()
        protocol=read(HOME/'PROTOCOL.json');self.deadline=protocol['deadline_unix_s'];self.origin=protocol['origin_unix_s']
        self.start=time.time();self.status='RUNNING'
        if (HOME/'EVENTS.jsonl').exists():
            for line in (HOME/'EVENTS.jsonl').open(encoding='utf-8'):
                e=json.loads(line)
                if e['event']=='request_attempt':self.ids.add(e['request_id'])
    def elapsed(self):return time.time()-self.start
    def check(self):
        now=time.time()
        if now>=self.deadline-600:raise TimeoutError('D6 total7200s boundary with600s closeout reserve')
        if os.environ.get('D6_JOB_DEADLINE') and now>=float(os.environ['D6_JOB_DEADLINE'])-20:raise TimeoutError('D6 supervised tree deadline')
    def event(self,event,**kw):append(HOME/'EVENTS.jsonl',dict(event=event,job=self.job,job_wall_s=self.elapsed(),**kw))
    def save(self):
        self.resource['counts']=dict(self.counts)
        self.resource['wall']['total_s']=time.time()-self.origin
        self.resource.setdefault('jobs',{})[self.job]=dict(status=self.status,elapsed_s=self.elapsed())
        put(HOME/'RESOURCE_ACCOUNTING.json',self.resource,True)
    def debit(self,key,n=1,limit=None):
        self.check();caps={'requests':4920,'actor_optimizer_attempts':768,'teacher_replays':2048}
        cap=caps.get(key,limit)
        if cap is not None and self.counts[key]+n>cap:raise RuntimeError(key+' D6 quota exhausted')
        if key in ('new_scenes','navigation_solves','new_teacher_requests','critic_optimizer_attempts') and n:raise PermissionError('Forbidden D6 computation')
        self.counts[key]+=n;self.save()
    def optimizer(self,stream,op):
        if self.counts['optimizer_'+stream]>=256:raise RuntimeError('Per-stream256 Actor cap')
        self.debit('actor_optimizer_attempts');self.counts['optimizer_'+stream]+=1
        self.event('optimizer_attempt',stream=stream,logical_update=op);self.save()
    def begin_request(self,purpose,rid):
        if purpose not in ('train','eval'):raise ValueError('No new teacher/reference/engineering request is permitted')
        self.debit('requests');self.counts[purpose+'_requests']+=1;self.ids.add(rid)
        self.event('request_attempt',request_id=rid,purpose=purpose,attempt_number=self.counts['requests']);self.save();return time.perf_counter()
    def complete(self,purpose,rid,record,tick,folder):
        ref=archive(HOME/folder/'REQUESTS.jsonl.gz',record);full=time.perf_counter()-tick
        self.counts['requests_completed']+=1
        append(HOME/folder/'REQUEST_POINTERS.jsonl',dict(request_id=rid,purpose=purpose,full_request_s=full,record=ref))
        self.event('request_complete',request_id=rid,purpose=purpose,full_request_s=full,record=ref);self.save();return ref,full
    def failed(self,purpose,rid,error):
        self.counts['requests_failed']+=1;self.event('request_failure',request_id=rid,purpose=purpose,error=repr(error));self.save()
    def internal(self,info):
        for k,v in info.get('counts',{}).items():self.counts['internal_'+k+'_calls']+=v
        for k,v in info.get('inclusive_s',{}).items():self.counts['internal_'+k+'_inclusive_s']+=v
        for k,v in info.get('exclusive_s',{}).items():self.counts['internal_'+k+'_exclusive_s']+=v
    def close(self,status='COMPLETE'):self.status=status;self.save()

