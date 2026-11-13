"""Local immutable identities and debit-before-work resource journal."""
from __future__ import annotations
import hashlib,json,os,time
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
HOME=Path(__file__).resolve().parent
H_ID='5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+'.pending')
    with pending.open('w',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(pending,path)
def append(path,value):
    with Path(path).open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

class Ledger:
    """One cumulative stage; failed/pending attempts are never refunded."""
    def __init__(self,folder,limits,identity):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.path=self.folder/'RESOURCE_ACCOUNTING.json';self.journal=self.folder/'EVENTS.jsonl'
        self.limits=limits;self.identity=identity;self.start=time.perf_counter();self.start_wall=time.time()
        if self.path.exists():
            old=read(self.path)
            if old['identity']!=identity or old['limits']!=limits:raise ValueError('Cannot reset resource identity/limits')
            self.counts=Counter(old['counts']);self.prior_wall=old['elapsed_s'];self.pending=old['pending'];self.status=old['status']
        else:self.counts=Counter();self.prior_wall=0.;self.pending={};self.status='CREATED'
        self.ids=set()
        journal_counts=Counter();journal_pending={};journal_wall=0.
        if self.journal.exists():
            for line in self.journal.read_text(encoding='utf-8').splitlines():
                e=json.loads(line)
                journal_wall=max(journal_wall,e['elapsed_s'])
                if e['event']=='begin':
                    self.ids.add(e['id']);journal_pending[e['id']]=e['kind'];journal_counts[e['kind']+'_attempts']+=1
                elif e['event'] in ('complete','failed'):
                    journal_pending.pop(e['id'],None);journal_counts[e['kind']+('_completed' if e['event']=='complete' else '_failed')]+=1
        for key,n in journal_counts.items():self.counts[key]=max(self.counts[key],n)
        self.pending.update(journal_pending);self.prior_wall=max(self.prior_wall,journal_wall)
        if self.pending:
            self.status='DIRTY_INTERRUPTION';self.save();raise RuntimeError('Dirty prior attempt retained; explicit recovery disposition required')
        self.status='RUNNING';self.save()
    def elapsed(self):return self.prior_wall+max(time.perf_counter()-self.start,time.time()-self.start_wall)
    def check(self):
        if self.elapsed()>=self.limits['wall_s']:raise TimeoutError('Cumulative stage wall limit')
    def save(self):
        put(self.path,dict(identity=self.identity,limits=self.limits,elapsed_s=self.elapsed(),counts=dict(self.counts),pending=self.pending,status=self.status,scientific_updates=0 if self.identity.startswith('M1:') else self.counts['optimizer_completed']))
    def event(self,event,**fields):append(self.journal,dict(event=event,elapsed_s=self.elapsed(),**fields))
    def begin(self,kind,rid,**fields):
        entered=time.perf_counter();self.check()
        if rid in self.ids:raise ValueError('Attempt ID already consumed: '+rid)
        key=kind+'_attempts'
        if key in self.limits and self.counts[key]>=self.limits[key]:raise RuntimeError(key+' exhausted')
        self.counts[key]+=1;self.ids.add(rid);self.pending[rid]=kind
        self.event('begin',kind=kind,id=rid,**fields);self.save();return entered
    def complete(self,kind,rid,tick,**fields):
        if self.pending.get(rid)!=kind:raise ValueError('Completion without exact debit')
        self.pending.pop(rid);self.counts[kind+'_completed']+=1
        self.event('complete',kind=kind,id=rid,wall_s=time.perf_counter()-tick,**fields);self.save()
    def fail(self,kind,rid,exc):
        if self.pending.get(rid)==kind:self.pending.pop(rid)
        self.counts[kind+'_failed']+=1;self.status='TECHNICAL_FAILURE'
        self.event('failed',kind=kind,id=rid,error=repr(exc));self.save()
    def close(self,status):self.status=status;self.save()

def verify_binding(path):
    b=read(path)
    for name,expected in b['files'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Source binding changed: '+name)
    return b
