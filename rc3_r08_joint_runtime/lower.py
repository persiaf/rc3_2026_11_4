"""Only registered two insertions, eight 2-opt passes, then positive-gain cleanup."""
import time
import numpy as np
from rc3_fast.solver import two_opt
from rc3_constructive.baselines import normalize
from rc3_r08_joint_decision_design_20261113.contracts import JointAction,action_legality

def solve(problem,state,record):
    tick=time.perf_counter();a=JointAction(tuple(record['drop']),tuple(record['add']));valid,reason=action_legality(state,a)
    if not valid:raise ValueError('Registered precheck failed: '+reason)
    index={int(j):k+1 for k,j in enumerate(problem.candidate_ids)};removed=set(a.drop)
    route=[0,*[index[j] for j in state.incumbent.visit_ids if j not in removed],0];insertions=[]
    for j in sorted(a.add):
        node=index[j];options=[]
        for pos in range(1,len(route)):
            before,after=route[pos-1],route[pos];value=float(problem.D[before,node]+problem.D[node,after]-problem.D[before,after])
            if np.isfinite(value):options.append((value,pos))
        if not options:raise ValueError('No finite registered insertion')
        change,pos=min(options);route.insert(pos,node);insertions.append(dict(global_id=j,position=pos,navigation_delta=change))
    before_opt=list(route)
    if len(set(route[1:-1]))!=len(route)-2 or not np.isfinite(problem.D[np.asarray(route[:-1]),np.asarray(route[1:])]).all():raise ValueError('Invalid route after insertion')
    if len(problem.C) and not problem.C[:,np.asarray(route[1:-1])-1].any(1).all():raise ValueError('Complete edited coverage failed')
    opt=two_opt(route,problem.D,passes=8);raw=normalize(problem,opt)
    return raw,dict(insertions=insertions,after_insertion_local=before_opt,after_two_opt_local=list(opt),
        cleanup_removed_global=[int(problem.candidate_ids[n-1]) for n in opt[1:-1] if n not in raw[1:-1]],two_opt_pass_cap=8,lower_s=time.perf_counter()-tick)
