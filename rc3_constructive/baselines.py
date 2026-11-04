"""Coverage-aware insertion, prior, shared postprocessing, and frozen references."""
from __future__ import annotations
import time
from contextlib import contextmanager
from unittest.mock import patch
import numpy as np
from rc3_fast import solver
from rc3_v4 import runtime
from .env import Env


def normalize(item, route):
    left=np.ones(len(item.C),bool);kept=[0]
    for node in route[1:-1]:
        if (left & item.C[:,node-1]).any():
            kept.append(int(node));left &= ~item.C[:,node-1]
    if left.any():raise ValueError('incomplete route')
    return kept+[0]


def postprocess(item, route):
    route=list(route)
    for _ in range(2):
        route=normalize(item,route);route=solver.two_opt(route,item.D,passes=60)
    return normalize(item,route)


def insertion(item):
    remaining=np.ones(len(item.C),bool);route=[0,0];D=item.D
    while remaining.any():
        gains=item.C[remaining].sum(0)
        nodes=np.arange(1,item.C.shape[1]+1)
        u=np.asarray(route[:-1]);v=np.asarray(route[1:])
        delta=D[u[:,None],nodes]+D[nodes,v[:,None]]-D[u,v][:,None]+item.base.penalty
        best=delta.min(0);positions=delta.argmin(0)
        allowed=(gains>0)&np.isfinite(D[0,1:])&np.isfinite(D[1:,0])&np.isfinite(best)
        if not allowed.any():raise ValueError('insertion cannot cover remaining tasks')
        scores=np.full(len(gains),np.inf);scores[allowed]=best[allowed]/gains[allowed]
        j=int(scores.argmin());route.insert(int(positions[j])+1,j+1);remaining &= ~item.C[:,j]
    return normalize(item,route)


def prior(item):
    env=Env(item)
    while not env.done:
        gains=item.C[env.remaining].sum(0)
        mask=env.get_action_mask()
        score=np.where(mask,gains/(item.D[env.current,1:]+item.base.penalty),-np.inf)
        env.step(int(score.argmax()))
    return env.route


@contextmanager
def reference_instrumentation(item):
    """Exclusive timers: nested two_opt inside joint_search is counted once."""
    totals=dict(postprocess_s=0.,export_validation_s=0.);depth=0
    def wrap_post(fn):
        def wrapped(*args,**kwargs):
            nonlocal depth
            outer=depth==0;tick=time.perf_counter();depth+=1
            try:return fn(*args,**kwargs)
            finally:
                depth-=1
                if outer:totals['postprocess_s']+=time.perf_counter()-tick
        return wrapped
    def export(problem,route,method):
        tick=time.perf_counter()
        try:return item.certify(route,method)
        finally:totals['export_validation_s']+=time.perf_counter()-tick
    with patch.object(solver,'two_opt',wrap_post(solver.two_opt)),patch.object(solver,'joint_search',wrap_post(solver.joint_search)),patch.object(runtime,'export_plan',export):
        yield totals


def reference(item, method):
    with reference_instrumentation(item) as timers:
        tick=time.perf_counter()
        plan,info=runtime.run(item.base,'fixed3' if method=='fixed3' else 'detour',
                             horizon=0 if method=='fixed3' else int(method.removeprefix('detour')))
        total=time.perf_counter()-tick
    return plan,info,{**timers,'decision_s':max(0.,total-sum(timers.values())), 'solver_total_s':total}
