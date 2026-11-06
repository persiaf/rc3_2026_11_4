"""Task windows on an unchanged global navigation/constraint model."""
from dataclasses import dataclass
import copy
import hashlib
import json
import operator
import numpy as np
from rc3_fast.solver import prepare


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def array_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()


def integer(value,name):
    if isinstance(value,(bool,np.bool_)):raise ValueError(name+' must be integer')
    try:return int(operator.index(value))
    except TypeError as exc:raise ValueError(name+' must be integer') from exc


@dataclass
class Parent:
    base: object
    F: np.ndarray
    A: np.ndarray
    mode: str='proxy'
    bundle: object=None

    def __post_init__(self):
        for kind in ('tasks','candidates','tables'):
            if any(type(x['id']) is not int for x in self.scene[kind]) or [x['id'] for x in self.scene[kind]]!=list(range(len(self.scene[kind]))):raise ValueError('Noncontiguous global '+kind+' IDs')
        self.F=np.asarray(self.F,dtype=bool).copy();self.A=np.asarray(self.A,dtype=bool).copy()
        shape=(len(self.scene['tasks']),len(self.scene['candidates']),2)
        if self.F.shape!=shape or self.A.shape!=shape:raise ValueError('F/A shape mismatch')
        if self.D.shape!=(shape[1]+1,shape[1]+1):raise ValueError('Global D must include depot and original candidates')
        if not np.array_equal(np.isfinite(self.D),np.isfinite(self.D.T)) or not np.allclose(self.D[np.isfinite(self.D)],self.D.T[np.isfinite(self.D)],rtol=0,atol=1e-8):
            raise ValueError('Frozen navigation must be undirected for endpoint-aware H')
        self.effective=self.F & self.A
        self.identity=signature(dict(scene=self.scene,F=array_hash(self.F),A=array_hash(self.A),mode=self.mode))
        self.F.flags.writeable=False;self.A.flags.writeable=False;self.effective.flags.writeable=False

    @property
    def scene(self):return self.base.scene
    @property
    def D(self):return self.base.nav['costs']

    @classmethod
    def from_scene(cls,scene,A=None,nav=None):
        if any(t.get('service_mode','independent_single_arm')!='independent_single_arm' for t in scene['tasks']):
            raise ValueError('Coupled dual-arm tasks require a joint-service model')
        if any(not np.isfinite(c['h']) or abs(c['h']-scene['robot']['fixed_lift_height'])>1e-9 for c in scene['candidates']):
            raise ValueError('Variable lift requires supplied external reachability tables')
        base=prepare(copy.deepcopy(scene),nav=nav)
        if A is None:A=np.ones_like(base.F,dtype=bool)
        raw=np.asarray(A)
        if raw.shape!=base.F.shape or raw.dtype.kind not in 'biu' or not np.isin(raw,[0,1]).all():
            raise ValueError('A must be a boolean/0-1 per-arm allow table')
        return cls(base,base.F,raw)

    @classmethod
    def from_external(cls,bundle):
        from rc3_v5.external import check_bundle,prepare_external
        F,A=check_bundle(bundle)
        return cls(prepare_external(bundle),F,A,'external_tables',copy.deepcopy(bundle))


class LocalProblem:
    """Local node0 denotes entry on outgoing edges and exit on incoming edges."""
    penalty=.75

    def __init__(self,parent,task_ids,table_ids,entry_node=0,exit_node=0,window=None):
        self.parent=parent;self.entry=integer(entry_node,'entry');self.exit=integer(exit_node,'exit')
        if not 0<=self.entry<len(parent.D) or not 0<=self.exit<len(parent.D):raise ValueError('Endpoint node out of range')
        if not np.isfinite(parent.D[self.entry,self.exit]):raise ValueError('Disconnected entry/exit')
        self.task_ids=np.asarray([integer(x,'task ID') for x in task_ids],dtype=np.int64)
        if len(set(self.task_ids))!=len(self.task_ids) or np.any(self.task_ids<0) or np.any(self.task_ids>=len(parent.F)):
            raise ValueError('Invalid target task IDs')
        self.table_ids=tuple(sorted(table_ids));self.window=window
        target=parent.effective[self.task_ids]
        positive=target.any((0,2))
        connected=np.isfinite(parent.D[self.entry,1:]) & np.isfinite(parent.D[1:,self.exit])
        self.candidate_ids=np.flatnonzero(positive & connected)
        self.F=target[:,self.candidate_ids,:].copy();self.C=self.F.any(2)
        if not self.C.any(1).all():raise ValueError('Selected tasks cannot all be served under F&A and endpoints')
        nodes=self.candidate_ids+1;m=len(nodes)
        self.D=np.zeros((m+1,m+1),dtype=np.float64)
        self.D[0,0]=parent.D[self.entry,self.exit]
        self.D[0,1:]=parent.D[self.entry,nodes]
        self.D[1:,0]=parent.D[nodes,self.exit]
        self.D[1:,1:]=parent.D[np.ix_(nodes,nodes)]
        self.boundary_candidate_ids=[int(j) for j in self.candidate_ids if parent.scene['candidates'][j]['table_id'] not in self.table_ids]
        self.identity=signature(dict(parent=parent.identity,tasks=self.task_ids.tolist(),candidates=self.candidate_ids.tolist(),
            entry=self.entry,exit=self.exit,window=window,penalty=self.penalty))

    def global_route(self,route):
        if len(route)<2 or route[0]!=0 or route[-1]!=0:raise ValueError('Expected symbolic endpoint route')
        if any(not 1<=integer(n,'service node')<=len(self.candidate_ids) for n in route[1:-1]):raise ValueError('Service node out of range')
        if len(set(route[1:-1]))!=len(route[1:-1]):raise ValueError('Repeated internal service')
        return [self.entry,*[int(self.candidate_ids[integer(n,'service node')-1])+1 for n in route[1:-1]],self.exit]

    def metrics(self,route):
        if len(route)<2 or route[0]!=0 or route[-1]!=0:raise ValueError('Invalid symbolic route')
        stops=route[1:-1]
        if any(type(n) is not int or not 1<=n<=len(self.candidate_ids) for n in stops) or len(set(stops))!=len(stops):
            raise ValueError('Invalid/repeated service node')
        if len(self.C) and (not stops or not self.C[:,np.asarray(stops)-1].any(1).all()):raise ValueError('Incomplete coverage')
        L=float(sum(self.D[a,b] for a,b in zip(route,route[1:])))
        if not np.isfinite(L):raise ValueError('Nonfinite leg')
        return dict(J=L+self.penalty*len(stops),L=L,K=len(stops))

    def description(self):
        return dict(identity=self.identity,parent_identity=self.parent.identity,window=self.window,
            task_ids_global=self.task_ids.tolist(),candidate_ids_global=self.candidate_ids.tolist(),
            boundary_candidate_ids_global=self.boundary_candidate_ids,entry_global_node=self.entry,exit_global_node=self.exit,
            coverage_sha256=array_hash(self.C),effective_F_sha256=array_hash(self.F),endpoint_D_sha256=array_hash(self.D),
            semantics='Task window on full parent navigation; paths may leave window; no physical-map clipping')


def extract(parent,x0,y0,width,height,entry_node=0,exit_node=0):
    x0,y0,width,height=[integer(v,n) for v,n in zip((x0,y0,width,height),('x0','y0','width','height'))]
    if min(x0,y0)<0 or min(width,height)<1 or x0+width>parent.scene['nx'] or y0+height>parent.scene['ny']:
        raise ValueError('Window outside parent grid')
    tables=[t['id'] for t in parent.scene['tables'] if x0<=t['ix']<x0+width and y0<=t['iy']<y0+height]
    tasks=[t['id'] for t in parent.scene['tasks'] if t['table_id'] in set(tables)]
    return LocalProblem(parent,tasks,tables,entry_node,exit_node,dict(x0=x0,y0=y0,width=width,height=height))
