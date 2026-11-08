"""Single-writer, bounded same-anchor retry and nonrecursive failure receipts."""
import contextlib
import errno
import json
import os
from pathlib import Path
import stat
import sys
import time
import uuid
from rc3_s3_evaluator.common import canonical, identity, require, read, write_new, digest, now


class WriterBusy(RuntimeError): pass
class PersistenceFault(RuntimeError): pass


def regular(path):
    s=Path(path).lstat()
    require(stat.S_ISREG(s.st_mode) and not Path(path).is_symlink() and s.st_nlink==1,'Unsafe journal/anchor file identity')
    return (s.st_dev,s.st_ino)


class Lease:
    """OS lock released on process exit; the small identity file is retained."""
    def __init__(self,path,scope):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.exists():regular(self.path)
        try:self.file=self.path.open('x+b')
        except FileExistsError:self.file=self.path.open('r+b')
        self.held=False
        self.file.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:
            self.file.close();raise WriterBusy('Another writer holds '+str(self.path)) from e
        self.held=True;self.pid=os.getpid();self.token=identity([scope,self.pid,uuid.uuid4().hex])
        # Only a locked owner can publish its identity. No journal is opened yet.
        self.file.seek(0);self.file.write(b'\0');self.file.write(canonical(dict(scope=scope,pid=self.pid,token=self.token)));self.file.truncate();self.file.flush();os.fsync(self.file.fileno())

    def assert_owner(self):
        require(self.held and self.pid==os.getpid(),'Foreign/unheld process writer')
        require(regular(self.path)==(os.fstat(self.file.fileno()).st_dev,os.fstat(self.file.fileno()).st_ino),'Lease path replaced')
        self.file.seek(1);require(json.loads(self.file.read())['token']==self.token,'Writer lease identity changed')

    def close(self):
        if not self.held:return
        self.file.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(self.file.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.file.fileno(),fcntl.LOCK_UN)
        self.file.close();self.held=False

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()


def independent_receipt(folders,kind,payload):
    """Immutable first-success receipt; never calls Journal or replaces HEAD."""
    name=kind+'_'+uuid.uuid4().hex+'.json';errors=[]
    record=dict(schema='rc3-s3-independent-io-receipt-v1',kind=kind,utc=now(),payload=payload)
    for folder in folders:
        try:
            p=Path(folder)/name;write_new(p,record)
            return dict(path=str(p),sha256=digest(p),preceding_receipt_errors=errors)
        except BaseException as e:errors.append(type(e).__name__+': '+str(e))
    sys.stderr.write('INDEPENDENT_RECEIPT_FAILED '+json.dumps(record,ensure_ascii=False)+' '+str(errors)+'\n')
    raise PersistenceFault('All independent receipt destinations failed: '+str(errors))


def commit_anchor(pending,head,anchor,previous,*,attempts=8,delay=.01,notify=lambda x:None):
    """Retry only os.replace for THIS exact event; no event append occurs here."""
    pending,head=Path(pending),Path(head)
    fid=regular(pending);sha=digest(pending)
    require(read(pending)==anchor,'Wrong prepared anchor')
    for attempt in range(attempts):
        require(regular(pending)==fid and digest(pending)==sha and read(pending)==anchor,'Pending anchor changed during commit')
        current=read(head) if head.exists() else None
        if head.exists():regular(head)
        require(current==previous,'Foreign/stale HEAD; not overwriting it')
        try:
            os.replace(pending,head)
            require(read(head)==anchor and not pending.exists(),'Committed anchor identity differs')
            return attempt
        except OSError as e:
            # An OS may commit then report an error: accept only exact new HEAD.
            if not pending.exists() and head.exists() and read(head)==anchor:return attempt
            transient=(getattr(e,'winerror',None) in (5,32,33) or e.errno in (errno.EACCES,errno.EBUSY,errno.EPERM))
            notify(dict(attempt=attempt+1,error=type(e).__name__,message=str(e),same_event_sha256=anchor['sha256']))
            if not transient or attempt+1==attempts:raise
            time.sleep(delay*min(2**attempt,8))
    raise AssertionError('Unreachable retry loop')


def read_chain(path):
    events=[];tail='0'*64
    if not Path(path).exists():return events,tail
    regular(path)
    raw=Path(path).read_bytes()
    require(not raw or raw.endswith(b'\n'),'Unclosed journal line')
    for line in raw.splitlines():
        e=json.loads(line);h=e.pop('sha256')
        require(e['sequence']==len(events) and e['previous']==tail and identity(e)==h,'Broken journal hash chain')
        e['sha256']=h;events.append(e);tail=h
    return events,tail


class Journal:
    """New continuation journal only. Legacy files are never opened for writing."""
    def __init__(self,folder,binding,requests,baseline,*,retry_attempts=8,retry_delay=.01):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.binding=binding;self.allowed={q['id']:q for q in requests};self.baseline=baseline
        require(len(self.allowed)==len(requests) and len(baseline['terminal_ids'])==baseline['debits'] and
            set(baseline['terminal_ids'])<=set(self.allowed) and baseline['success']+baseline['infra_failure']==baseline['debits'] and
            len(baseline.get('failure_ids',[]))==baseline['infra_failure'] and set(baseline.get('failure_ids',[]))<=set(baseline['terminal_ids']), 'Invalid inherited budget/state')
        self.jid=identity(dict(binding=binding,requests=requests,baseline=baseline))
        self.lease=Lease(self.folder/'WRITER.lease',self.jid)
        self.path=self.folder/'journal.jsonl';self.head=self.folder/'HEAD.json';self.pending=self.folder/'HEAD.pending'
        self.retry_attempts,self.retry_delay=retry_attempts,retry_delay;self.healthy=True
        try:
            manifest=dict(schema='rc3-s3-continuation-journal-v1',journal_identity=self.jid,binding=binding,
                grid_sha256=identity(requests),baseline=baseline)
            mp=self.folder/'MANIFEST.json'
            if mp.exists():require(read(mp)==manifest,'Continuation identity/budget/source changed')
            else:
                require(not self.path.exists() and not self.head.exists() and not self.pending.exists(),'Unbound journal cannot reset state')
                write_new(mp,manifest)
            self.events,self.tail=read_chain(self.path)
            require(not self.pending.exists(),'Unclosed new pending retained; no implicit repair/replay')
            self.anchor=read(self.head) if self.head.exists() else None
            if self.head.exists():regular(self.head)
            require((not self.events and self.anchor is None) or
                (self.anchor is not None and self.anchor['journal_identity']==self.jid and
                 self.anchor['sequence']==len(self.events) and self.anchor['sha256']==self.tail),'HEAD/chain inconsistency')
            self.size=self.path.stat().st_size if self.path.exists() else 0
            self.starts={};self.results={};self.errors={};self.session=None
            for e in self.events:
                kind=e['event']
                if kind=='session_open':
                    require(self.session is None,'Overlapping sessions');self.session=e['token']
                elif kind=='session_close':
                    require(self.session==e['token'] and e['elapsed_s']>=0,'Bad session close');self.session=None
                elif kind=='request_start':
                    q=e['request'];rid=q['id']
                    require(self.allowed.get(rid)==q and rid not in self.starts and rid not in baseline['terminal_ids'],'Duplicate/foreign/already consumed request')
                    self.starts[rid]=q
                elif kind=='request_complete':
                    rid=e['id'];require(rid in self.starts and rid not in self.results and rid not in self.errors,'Unbilled/repeated completion')
                    p=self.folder/'results'/(rid+'.json');require(digest(p)==e['artifact_sha256'],'Result modified')
                    r=read(p);require(r['request']==self.starts[rid] and r['valid'] and r['plan']['validation']['valid_model_plan'],'Invalid committed result')
                    self.results[rid]=e
                elif kind=='request_error':
                    rid=e['id'];require(rid in self.starts and rid not in self.results and rid not in self.errors,'Wrong failure');self.errors[rid]=e
                else:require(kind in ('operation_start','operation_end','hot_prime'),'Unknown event')
            artifacts={p.stem for p in (self.folder/'results').glob('*.json')}
            require(artifacts==set(self.results),'Orphan or missing result; no implicit adoption/replay')
            require(len(self.starts)+baseline['debits']<=len(requests),'Budget exceeded')
        except BaseException:
            self.lease.close();raise

    def close(self):self.lease.close()
    def __enter__(self):return self
    def __exit__(self,*exc):self.close()

    def append(self,event):
        require(self.healthy and self.lease.held,'Failed or unleased writer')
        self.lease.assert_owner()
        require((self.path.stat().st_size if self.path.exists() else 0)==self.size,'Journal changed under writer')
        require((read(self.head) if self.head.exists() else None)==self.anchor,'HEAD changed under writer')
        e=dict(event,previous=self.tail,sequence=len(self.events));e['sha256']=identity(e)
        try:
            with self.path.open('ab') as f:f.write(canonical(e)+b'\n');f.flush();os.fsync(f.fileno())
            self.events.append(e);self.tail=e['sha256'];self.size=self.path.stat().st_size
            anchor=dict(schema='rc3-s3-anchor-v2',journal_identity=self.jid,writer_identity=self.lease.token,
                sequence=len(self.events),sha256=self.tail)
            write_new(self.pending,anchor)
            commit_anchor(self.pending,self.head,anchor,self.anchor,attempts=self.retry_attempts,delay=self.retry_delay,
                notify=lambda x:independent_receipt([self.folder/'io_receipts',self.folder.parent/'emergency'], 'ANCHOR_RETRY',x))
            self.anchor=anchor
        except BaseException as exc:
            self.healthy=False
            try:independent_receipt([self.folder/'faults',self.folder.parent/'emergency'], 'JOURNAL_FAILURE',
                dict(journal_identity=self.jid,event=e,first_error=repr(exc),journal_bytes=self.path.stat().st_size if self.path.exists() else None))
            except BaseException:pass # Original exception remains primary; stderr is the final fallback.
            raise

    def clean(self,allow_session=None):
        require(not self.errors and set(self.starts)==set(self.results),'New failed/incomplete request cannot auto-resume')
        require(self.session is None or self.session==allow_session,'Unclosed session needs explicit disposition')

    def debit(self,q):
        require(self.allowed.get(q['id'])==q and q['id'] not in self.baseline['terminal_ids'] and q['id'] not in self.starts,'Not an unexecuted request')
        require(len(self.starts)+self.baseline['debits']<len(self.allowed),'No request budget')
        self.append(dict(event='request_start',request=q));self.starts[q['id']]=q

    def complete(self,q,elapsed):
        p=self.folder/'results'/(q['id']+'.json');e=dict(event='request_complete',id=q['id'],artifact_sha256=digest(p),complete_time_s=elapsed)
        self.append(e);self.results[q['id']]=e

    def request_state(self,rid):
        require(rid in self.allowed,'Unknown request state')
        if rid in self.baseline['terminal_ids']:
            return 'INFRASTRUCTURE_FAILED_DISPOSED' if rid in self.baseline.get('failure_ids',[]) else 'COMPLETED'
        if rid in self.results:return 'COMPLETED'
        if rid in self.errors:return 'INFRASTRUCTURE_FAILED'
        if rid in self.starts:return 'RUNNING'
        return 'UNEXECUTED'
