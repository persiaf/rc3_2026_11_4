import os
import time
from pathlib import Path
from rc3_s3_evaluator.common import (ROOT,read,write_new,digest,identity,canonical,require,now,label,
    PROTOCOL as OLD_PROTOCOL,REGISTRY,H_ID,METHODS,MODEL_ROLES,Meter)
from rc3_s3_recovery_20261108.storage import Lease,read_chain,commit_anchor,independent_receipt

HOME=Path(__file__).resolve().parent
BASE='925a0a1a8957299e3e849b29ac501a99212aa902'
EXPERIMENT='rc3_actor_pg_screen_20261109_01'
SEED=11622026
OLD_RUN=ROOT/'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01'
EMPTY_BASELINE=dict(debits=0,success=0,infra_failure=0,terminal_ids=[],failure_ids=[],wall_carry_s=0)


class Log:
    """Generic chained predebits; same-event atomic-anchor retries only."""
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.path=self.folder/'events.jsonl';self.head=self.folder/'HEAD.json';self.pending=self.folder/'HEAD.pending'
        self.lease=Lease(self.folder/'WRITER.lease',EXPERIMENT)
        try:
            self.events,self.tail=read_chain(self.path)
            self.anchor=read(self.head) if self.head.exists() else None
            require(not self.pending.exists(),'Unclosed pending evidence; no automatic restart')
            require((not self.events and self.anchor is None) or
                (self.anchor['sequence']==len(self.events) and self.anchor['sha256']==self.tail),'Bad event anchor')
        except BaseException:self.lease.close();raise
        self.size=self.path.stat().st_size if self.path.exists() else 0
        self.healthy=True

    def __call__(self,row):
        require(self.healthy,'Failed log cannot be reused');self.lease.assert_owner()
        require((self.path.stat().st_size if self.path.exists() else 0)==self.size,'Log changed')
        require((read(self.head) if self.head.exists() else None)==self.anchor,'Foreign anchor')
        e=dict(row,sequence=len(self.events),previous=self.tail);e['sha256']=identity(e)
        try:
            with self.path.open('ab') as f:f.write(canonical(e)+b'\n');f.flush();os.fsync(f.fileno())
            self.events.append(e);self.tail=e['sha256'];self.size=self.path.stat().st_size
            anchor=dict(sequence=len(self.events),sha256=self.tail,experiment=EXPERIMENT)
            write_new(self.pending,anchor)
            commit_anchor(self.pending,self.head,anchor,self.anchor,
                notify=lambda e:independent_receipt([self.folder/'io_receipts',HOME/'emergency'],'ANCHOR_RETRY',e))
            self.anchor=anchor
        except BaseException as exc:
            self.healthy=False
            independent_receipt([self.folder/'faults',HOME/'emergency'],'JOURNAL_FAILURE',dict(event=e,error=repr(exc)))
            raise

    def close(self):self.lease.close()


class Deadline:
    def __init__(self,end):self.end=float(end)
    def check(self):require(time.perf_counter()<self.end,'Phase wall cap reached')
    def remaining(self):self.check();return self.end-time.perf_counter()


def admit():
    p=read(HOME/'PROTOCOL.json');a=read(HOME/'AUTHORIZATION.json');b=read(HOME/'SOURCE_BINDING.json')
    require(a['authorized_by']=='user' and a['protocol_sha256']==digest(HOME/'PROTOCOL.json'),'New experiment authorization mismatch')
    require(a['experiment_id']==p['experiment_id']==EXPERIMENT and a['old_budget_reused'] is False,'Wrong experiment/budget')
    require(all(digest(ROOT/n)==h for n,h in b['source_sha256'].items()),'Execution source changed')
    require(digest(ROOT/'rc3_global_local_execution_v3/REGISTRY.json')==REGISTRY,'Registry changed')
    require(digest(ROOT/'rc3_global_local_execution_v3/V3_EXECUTION_CANDIDATE.json')==OLD_PROTOCOL,'Old protocol changed')
    verify=read(HOME/'ENGINEERING_VERIFICATION.json');require(verify['status']=='PASS' and verify['source_binding_sha256']==digest(HOME/'SOURCE_BINDING.json'),'Bound tests not passed')
    from rc3_s3_evaluator.provenance import environment
    require(environment()==b['environment'],'Runtime environment differs from frozen registration')
    return p


def torch_new(path,value):
    from rc3_global_local_trainer.checkpoint import _atomic_new
    import torch
    _atomic_new(path,lambda f:torch.save(value,f))
    return dict(path=label(path),sha256=digest(path),bytes=path.stat().st_size)
