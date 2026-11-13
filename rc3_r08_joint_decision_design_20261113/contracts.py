"""Pure interfaces: no Actor, optimizer, solver, scene generation or H."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import hashlib,json
import numpy as np

def array_hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape,a.dtype.str)).encode()+a.tobytes()).hexdigest()

@dataclass(frozen=True)
class ProblemSnapshot:
    problem_id:str
    parent_id:str
    physical_key:str
    task_ids:tuple[int,...]
    candidate_ids:tuple[int,...]
    per_arm_F:np.ndarray
    per_arm_allow:np.ndarray
    D:np.ndarray
    entry:int
    exit:int
    poses:tuple[tuple[float,...],...]

    def __post_init__(self):
        n,m=len(self.task_ids),len(self.candidate_ids)
        if len(set(self.task_ids))!=n or len(set(self.candidate_ids))!=m:raise ValueError('Duplicate global ID')
        for name,dtype,shape in [('per_arm_F',bool,(n,m,2)),('per_arm_allow',bool,(n,m,2)),('D',np.float64,(m+1,m+1))]:
            a=np.array(getattr(self,name),dtype=dtype,copy=True)
            if a.shape!=shape:raise ValueError('Wrong '+name+' shape')
            a.flags.writeable=False;object.__setattr__(self,name,a)
        if np.isnan(self.D).any() or (self.D<0).any():raise ValueError('Invalid navigation costs')
        if len(self.poses)!=m:raise ValueError('Pose/configuration ID mismatch')

    @property
    def legal_service(self):return self.per_arm_F & self.per_arm_allow
    @property
    def cover(self):return self.legal_service.any(2)

@dataclass(frozen=True)
class RecordedRoute:
    """An existing evidenced plan, not a newly solved route."""
    problem_id:str
    visit_ids:tuple[int,...]
    J:float
    L:float
    K:int
    evidence_id:str
    def __post_init__(self):
        if len(set(self.visit_ids))!=len(self.visit_ids) or self.K!=len(self.visit_ids):raise ValueError('K counts distinct configuration visits')
        if not np.isfinite([self.J,self.L]).all() or abs(self.J-self.L-.75*self.K)>1e-7:raise ValueError('Wrong cost units')

@dataclass(frozen=True)
class JointAction:
    """Whole-configuration set edit; lower solver must supply ordering later."""
    drop:tuple[int,...]=()
    add:tuple[int,...]=()
    stop:bool=False
    def __post_init__(self):
        if len(self.drop)>2 or len(self.add)>2:raise ValueError('At most two configuration removals/additions')
        if len(set(self.drop))!=len(self.drop) or len(set(self.add))!=len(self.add) or set(self.drop)&set(self.add):raise ValueError('Duplicate/no-op action IDs')
        if self.stop and (self.drop or self.add):raise ValueError('STOP has no edits')
        if not self.stop and not (self.drop or self.add):raise ValueError('Empty edit is not STOP')

@dataclass(frozen=True)
class JointState:
    snapshot:ProblemSnapshot
    incumbent:RecordedRoute
    steps_left:int
    requests_left:int
    weight_version:str
    def __post_init__(self):
        if self.snapshot.problem_id!=self.incumbent.problem_id:raise ValueError('Problem identity mismatch')
        if self.steps_left<0 or self.requests_left<0:raise ValueError('Negative budget')
    def fingerprint(self):
        return hashlib.sha256(json.dumps(dict(problem=self.snapshot.problem_id,route=self.incumbent.visit_ids,
            steps=self.steps_left,requests=self.requests_left,weights=self.weight_version),sort_keys=True).encode()).hexdigest()

def action_legality(state:JointState,action:JointAction):
    """Coverage/endpoints precheck only. Does NOT solve or price a new route."""
    if action.stop:return True,'STOP'
    if state.steps_left==0 or state.requests_left==0:return False,'budget'
    selected=set(state.incumbent.visit_ids);p=state.snapshot;index={j:k for k,j in enumerate(p.candidate_ids)}
    if not set(action.drop)<=selected:return False,'drop_not_selected'
    if set(action.add)&selected or not set(action.add)<=set(index):return False,'add_not_available'
    proposed=sorted((selected-set(action.drop))|set(action.add))
    cols=[index[j] for j in proposed]
    if p.task_ids and (not cols or not p.cover[:,cols].any(1).all()):return False,'uncovered_task'
    nodes=np.asarray(cols,dtype=int)+1
    if not np.isfinite(p.D[0,nodes]).all() or not np.isfinite(p.D[nodes,0]).all():return False,'endpoint_unreachable'
    # Pairwise sequencing is a separate billable lower-solver feasibility check.
    return True,'coverage_precheck_only'

@dataclass(frozen=True)
class SolveLimits:
    insertion_count:int=2
    two_opt_passes:int=8
    complete_request_limit:int=1

class LowerOrderingSolver(Protocol):
    def solve(self,state:JointState,action:JointAction,limits:SolveLimits)->RecordedRoute:...

@dataclass(frozen=True)
class ResumeIdentity:
    scientific_id:str
    source_sha:str
    menu_sha:str
    cursor:int
    requests:int
    optimizer_attempts:int
    weights_sha:str
    Adam_sha:str
    RNG_sha:str
    def validate(self,current_scientific_id,source_sha,menu_sha,durable_requests,durable_optimizers):
        if (self.scientific_id,self.source_sha,self.menu_sha)!=(current_scientific_id,source_sha,menu_sha):raise ValueError('Foreign checkpoint identity')
        if self.requests>durable_requests or self.optimizer_attempts!=durable_optimizers:raise ValueError('Resource rollback or ambiguous update')
        if min(self.cursor,self.requests,self.optimizer_attempts)<0:raise ValueError('Negative cursor')
