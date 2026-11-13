"""Read only previously admitted inputs; no scene/teacher/navigation generation."""
from collections import Counter
import gzip
import numpy as np
from rc3_d5_revision_20261112.data import Pool as OriginalPool
from rc3_global_local.problem import LocalProblem
from .common import ROOT,HOME,read,sha

class Meter:
    def __init__(self):self.counts=Counter()
    def check(self):pass

class Pool:
    def __init__(self,ledger=None):
        self.ledger=ledger or Meter();self.pool=OriginalPool(self.ledger)
        self.training=read(ROOT/'rc3_d5_revision_20261112/JOINT_DATA_INDEX.json')
        self.references=read(ROOT/'rc3_d5_revision_20261112/REFERENCE_MANIFEST.json')['records']
    def get(self,tid):
        if tid=='EMPTY_CLOSED':
            p,e=self.pool.get('F012');q=LocalProblem(p.parent,[],[],0,0,window=dict(engineering_boundary='empty_closed'))
            return q,dict(e,boundary_parent_task='F012',empty_boundary=True)
        if tid=='EMPTY_OPEN':
            p,e=self.pool.get('L030');q=LocalProblem(p.parent,[],[],p.entry,p.exit,window=dict(engineering_boundary='empty_open'))
            return q,dict(e,boundary_parent_task='L030',empty_boundary=True)
        return self.pool.get(tid)
    def teacher(self,tid):
        if tid not in self.training['full'] and tid not in self.training['local']:raise ValueError('Training labels only')
        return self.pool.teacher(tid)
    def reference(self,tid,p,floor=1.):
        if not np.isfinite(floor) or floor<=0:raise ValueError('Positive fixed reference floor required')
        if tid.startswith('EMPTY_'):return max(float(p.metrics([0,0])['J']),floor)
        r=self.references[tid];path=ROOT/r['path']
        if sha(path)!=r['sha256']:raise ValueError('Frozen reference bytes changed')
        obj=__import__('json').loads(gzip.decompress(path.read_bytes()))
        if obj['problem_identity']!=p.identity or not obj['plan']['validation']['valid_model_plan']:raise ValueError('Frozen reference identity/validation')
        value=float(obj['metrics']['J'])
        if not np.isfinite(value) or value<0 or (len(p.task_ids)>0 and value==0):raise ValueError('Invalid or nonempty zero reference cost')
        return max(value,floor)
