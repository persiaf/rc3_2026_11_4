import copy
import errno
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid
import datetime
import numpy as np
import torch
from rc3_s3_evaluator.common import ROOT,read,write_new,digest,identity,canonical,require,now,label,H_ID,REGISTRY
from rc3_s3_recovery_20261108.storage import Lease,read_chain,commit_anchor,independent_receipt,regular

HOME=Path(__file__).resolve().parent
BASE='7d2cfe75919b25bac59c5274321baca4d8e98f77'
EXP='rc3_integrated_rppo_v1_20261110_01'
OUT=HOME/'runs'/EXP
OLD=ROOT/'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01'
SEEDS=(11622026,11622027,11622028)
CAPS=dict(engineering=96,warm=576,train=36864,selection=1392,promotion=528,quality=3024,timing=3024,ablation=3168)
TRAIN_DEVICE='cuda' if (HOME/'GPU_ENGINEERING_REGISTRATION.json').exists() else 'cpu'


def source_path():
    pointer=HOME/'EXECUTION_SOURCE.json'
    if not pointer.exists():return HOME/'SOURCE_BINDING.json'
    p=read(pointer);path=(ROOT/p['binding_path']).resolve()
    require(path.parent==HOME and digest(path)==p['binding_sha256'],'Execution source pointer changed or escaped')
    return path


def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,torch.Tensor):return x.detach().cpu().tolist()
    return x


def tensor_digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    require(not tmp.exists(),'Foreign temporary checkpoint')
    with tmp.open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    for attempt in range(8):
        try:os.replace(tmp,path);break
        except PermissionError:
            if attempt==7:raise
            time.sleep(.01*min(2**attempt,8))
    return dict(path=label(path),sha256=digest(path))


def object_new(value):
    raw=canonical(plain(value));h=hashlib.sha256(raw).hexdigest();p=OUT/'objects'/h[:2]/(h+'.json.gz')
    if not p.exists():
        p.parent.mkdir(parents=True,exist_ok=True)
        b=io.BytesIO()
        with gzip.GzipFile(fileobj=b,mode='wb',mtime=0) as f:f.write(raw)
        tmp=p.with_name(p.name+'.part.'+uuid.uuid4().hex)
        with tmp.open('xb') as f:f.write(b.getvalue());f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    require(hashlib.sha256(gzip.decompress(p.read_bytes())).hexdigest()==h,'Content-addressed object collision')
    return dict(path=label(p),sha256=digest(p),content_sha256=h,bytes=p.stat().st_size)


def object_read(ref):
    p=ROOT/ref['path'];require(digest(p)==ref['sha256'],'Object file changed')
    b=gzip.decompress(p.read_bytes());require(hashlib.sha256(b).hexdigest()==ref['content_sha256'],'Object content changed')
    return json.loads(b)


def seed_for(*parts):return int(identity([EXP,*parts])[:15],16)


def boot_identity():
    if os.name=='nt':
        import ctypes
        f=ctypes.windll.kernel32.GetTickCount64;f.restype=ctypes.c_ulonglong
        return dict(host=os.environ.get('COMPUTERNAME'),boot_epoch_s=time.time()-f()/1000.)
    p=Path('/proc/sys/kernel/random/boot_id')
    return dict(uuid=p.read_text().strip()) if p.exists() else None


def same_boot(a,b):
    if a is None or b is None:return False
    if 'uuid' in a:return a==b
    return a.get('host')==b.get('host') and abs(a['boot_epoch_s']-b['boot_epoch_s'])<2.


class Log:
    def __init__(self,folder=OUT/'ledger'):
        self.folder=folder;folder.mkdir(parents=True,exist_ok=True);self.lease=Lease(folder/'WRITER.lease',EXP)
        self.path=folder/'events.jsonl';self.head=folder/'HEAD.json';self.pending=folder/'HEAD.pending'
        self.events,self.tail=read_chain(self.path);self.anchor=read(self.head) if self.head.exists() else None
        if self.pending.exists():
            pending=read(self.pending)
            require(pending['sequence']==len(self.events) and pending['sha256']==self.tail and pending['experiment']==EXP,'Unproven pending')
            require(self.anchor is None or self.anchor['sequence']==len(self.events)-1,'Pending not immediate successor')
            commit_anchor(self.pending,self.head,pending,self.anchor);self.anchor=pending
        require((not self.events and self.anchor is None) or (self.anchor['sequence']==len(self.events) and self.anchor['sha256']==self.tail),'Broken ledger anchor')
        self.size=self.path.stat().st_size if self.path.exists() else 0;self.healthy=True
    def __call__(self,row):
        require(self.healthy,'Failed writer');self.lease.assert_owner()
        require((self.path.stat().st_size if self.path.exists() else 0)==self.size and (read(self.head) if self.head.exists() else None)==self.anchor,'Foreign ledger mutation')
        e=dict(plain(row),previous=self.tail,sequence=len(self.events));e['sha256']=identity(e)
        try:
            with self.path.open('ab') as f:f.write(canonical(e)+b'\n');f.flush();os.fsync(f.fileno())
            self.events.append(e);self.tail=e['sha256'];self.size=self.path.stat().st_size
            a=dict(sequence=len(self.events),sha256=self.tail,experiment=EXP);write_new(self.pending,a)
            commit_anchor(self.pending,self.head,a,self.anchor,notify=lambda r:independent_receipt([self.folder/'io_receipts',OUT/'emergency'],'ANCHOR_RETRY',r));self.anchor=a
        except BaseException as exc:
            self.healthy=False;independent_receipt([self.folder/'faults',OUT/'emergency'],'JOURNAL_FAILURE',dict(event=e,error=repr(exc)));raise
    def close(self):self.lease.close()


class Clock:
    def __init__(self,log,name,limit):
        self.log,self.name,self.limit=log,name,float(limit)
        closed={e['open_sequence'] for e in log.events if e['event']=='clock_close'}
        for e in list(log.events):
            if e['event']=='clock_open' and e['name']==name and e['sequence'] not in closed:
                candidates=[]
                for path in (log.folder/'orphan_clocks').glob('*.json'):
                    r=read(path)
                    if r['experiment']==EXP and r['open_sequence']==e['sequence']:candidates.append(r['elapsed_s'])
                if candidates:delta=min(candidates)
                elif same_boot(e.get('boot_identity'),boot_identity()):delta=time.perf_counter()-e['start_perf']
                else:
                    require('utc' in e,'No provable elapsed time across boot boundary')
                    delta=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(e['utc'])).total_seconds()
                require(np.isfinite(delta) and delta>=0,'Unprovable interrupted monotonic clock; cannot reset')
                log(dict(event='clock_close',name=name,open_sequence=e['sequence'],elapsed_s=delta,
                    interrupted_recovery=True,exact_wall_s=delta if candidates else None,
                    scope='Parent exit receipt' if candidates else 'Conservative includes downtime until recovery; boot-bound monotonic or UTC upper charge'))
        self.carry=sum(e['elapsed_s'] for e in log.events if e['event']=='clock_close' and e['name']==name)
        self.start=time.perf_counter();self.closed=False
        log(dict(event='clock_open',name=name,cap_s=limit,carry_s=self.carry,start_perf=self.start,utc=now(),boot_identity=boot_identity()))
        self.open_sequence=log.events[-1]['sequence']
        atomic(log.folder/'ACTIVE_CLOCK.pt',dict(name=name,cap_s=limit,carry_s=self.carry,start_perf=self.start,open_sequence=self.open_sequence))
    def elapsed(self):return self.carry+time.perf_counter()-self.start
    def check(self):require(self.elapsed()<self.limit,'Cumulative wall budget exhausted: '+self.name)
    def remaining(self):self.check();return self.limit-self.elapsed()
    def close(self):
        if not self.closed:
            self.log(dict(event='clock_close',name=self.name,open_sequence=self.open_sequence,elapsed_s=time.perf_counter()-self.start,cumulative_s=self.elapsed()));self.closed=True
            atomic(self.log.folder/'ACTIVE_CLOCK.pt',dict(name=self.name,open_sequence=self.open_sequence,closed=True))


class Store:
    def __init__(self,log):
        self.log=log;self.starts={};self.done={};self.failed={};self.counts={k:0 for k in CAPS};self.retry=0;self.timings={}
        for e in log.events:
            if e['event']=='request_start':
                k=e['key'];require(k not in self.starts,'Repeated predebit');self.starts[k]=e;self.counts[e['query']['kind']]+=1;self.retry+=int(e['attempt']>0)
            elif e['event']=='request_done':self.done[e['key']]=e
            elif e['event']=='request_failed':self.failed[e['key']]=e
            elif e['event']=='request_close_timing':self.timings[e['key']]=e['full_audit_s']
        for k,e in list(self.starts.items()):
            if k in self.done or k in self.failed:continue
            receipt=log.folder/'request_receipts'/(k.replace(':','_')+'.json')
            if receipt.exists():
                r=read(receipt);require(r['query']==e['query'] and r['source_binding_sha256']==e['source_binding_sha256'],'Foreign orphan result receipt')
                value=object_read(r['completion']['object']);require(value['request']==e['query'],'Orphan object request mismatch')
                log(r['completion']);self.done[k]=r['completion']
    def request(self,q,fn,clock):
        require(q['kind'] in CAPS,'Unknown request category');rid=identity(q)
        for attempt in range(2):
            k=rid+':'+str(attempt)
            if k in self.done:
                v=object_read(self.done[k]['object']);v['_audit_time_s']=self.timings.get(k);return v
            if k in self.failed:
                require(self.failed[k]['infrastructure'] and attempt==0,'Numerical/input failure or repeat retry forbidden');continue
            require(k not in self.starts,'Unclosed request must be explicitly recovered, not replayed')
            initial=sum(e['query']['kind']==q['kind'] and e['attempt']==0 for e in self.starts.values())
            require(sum(self.counts.values())<50000 and (attempt>0 or initial<CAPS[q['kind']]),'Request cap')
            if q['kind']=='engineering':require(self.counts['engineering']<96,'Engineering hard cap including retries')
            if attempt:require(self.retry<32,'Infrastructure retry pool exhausted');self.retry+=1
            clock.check();event=dict(event='request_start',key=k,query=q,attempt=attempt,source_binding_sha256=digest(source_path()))
            self.log(event);self.starts[k]=event;self.counts[q['kind']]+=1;t=time.perf_counter()
            try:
                v=fn();pi=v.get('plan',v.get('result',{}).get('plan'))
                require(pi is not None and pi.get('validation',{}).get('valid_model_plan') and pi['validation'].get('all_tasks_served'),'Unvalidated/incomplete result')
                metrics=v.get('metrics',v.get('result',{}))
                require(all(np.isfinite(metrics[k]) for k in ('J','L','K','raw_J','raw_L','raw_K')),'Nonfinite request metrics')
                v['request']=q;v['attempt']=attempt;ref=object_new(v);clock.check()
                e=dict(event='request_done',key=k,object=ref,complete_time_s=time.perf_counter()-t)
                receipt=self.log.folder/'request_receipts'/(k.replace(':','_')+'.json')
                write_new(receipt,dict(query=q,source_binding_sha256=digest(source_path()),completion=e))
                self.log(e);self.done[k]=e
                elapsed=time.perf_counter()-t
                self.log(dict(event='request_close_timing',key=k,full_audit_s=elapsed,
                    boundary='Through committed result journal; this timing event itself outside request and inside stage wall'))
                self.timings[k]=elapsed;v['_audit_time_s']=elapsed;return v
            except BaseException as exc:
                if k in self.done:
                    independent_receipt([self.log.folder/'faults',OUT/'emergency'],'POST_COMPLETION_TIMING_FAILURE',dict(key=k,error=repr(exc)));raise
                infra=isinstance(exc,OSError) and not isinstance(exc,(TimeoutError,FileNotFoundError)) and (getattr(exc,'winerror',None) in (5,32,33) or getattr(exc,'errno',None) in (errno.EACCES,errno.EBUSY,errno.EPERM))
                e=dict(event='request_failed',key=k,error=repr(exc),infrastructure=infra,elapsed_s=time.perf_counter()-t)
                self.log(e);self.failed[k]=e
                if not infra or attempt:raise
        raise RuntimeError('No successful request attempt')


def configure():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False


def verify_sources():
    b=read(source_path())
    require(all(digest(ROOT/n)==h for n,h in b['source_sha256'].items()),'Bound source changed')
    for n,h in b['artifacts'].items():require(digest(ROOT/n)==h,'Registration changed: '+n)
    return b
