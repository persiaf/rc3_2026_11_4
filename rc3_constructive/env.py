"""Cover-all then auto-return MDP. No early success STOP and no rule fallback."""
from __future__ import annotations
import numpy as np
import torch
from .problem import features


class Env:
    def __init__(self, problem): self.problem = problem; self.reset()

    def reset(self):
        self.remaining = np.ones(len(self.problem.C), dtype=bool)
        self.current = 0; self.route = [0]; self.rewards = []; self.done = not self.remaining.any()
        if self.done: self.route.append(0)
        return self.remaining.copy()

    def get_action_mask(self):
        D = self.problem.D
        return (self.problem.C[self.remaining].any(0) & np.isfinite(D[self.current,1:]) & np.isfinite(D[1:,0]))

    def step(self, candidate_id):
        if self.done or isinstance(candidate_id, (bool,np.bool_)) or not isinstance(candidate_id,(int,np.integer)):
            raise ValueError('invalid action or terminated episode')
        if not 0 <= candidate_id < self.problem.C.shape[1] or not self.get_action_mask()[candidate_id]:
            raise ValueError('masked candidate')
        served = np.flatnonzero(self.remaining & self.problem.C[:,candidate_id]).tolist()
        node = int(candidate_id)+1
        reward = -(self.problem.D[self.current,node]+self.problem.base.penalty)
        self.remaining &= ~self.problem.C[:,candidate_id]; self.current = node; self.route.append(node)
        self.done = not self.remaining.any()
        if self.done:
            reward -= self.problem.D[node,0]; self.route.append(0)
        self.rewards.append(float(reward))
        return dict(candidate_id=int(candidate_id), served_tasks=served, reward=float(reward),
                    remaining_tasks=np.flatnonzero(self.remaining).tolist(), done=self.done)

    def replay(self, candidate_ids):
        self.reset(); trace = [self.step(int(j)) for j in candidate_ids]
        if not self.done: raise ValueError('incomplete construction')
        return trace


def batch(items, device='cpu'):
    B = len(items); N = max(1,max(len(x.C) for x in items)); M = max(x.C.shape[1] for x in items)
    tf=np.zeros((B,N,5),np.float32); cf=np.zeros((B,M,7),np.float32)
    C=np.zeros((B,N,M),np.float32); D=np.full((B,M+1,M+1),np.inf,np.float64)
    tv=np.zeros((B,N),bool); cv=np.zeros((B,M),bool); scales=[]; penalties=[]
    for b,item in enumerate(items):
        n,m=item.C.shape; t,c,s=features(item); tf[b,:n]=t;cf[b,:m]=c;C[b,:n,:m]=item.C
        D[b,:m+1,:m+1]=item.D;tv[b,:n]=True;cv[b,:m]=True;scales.append(s);penalties.append(item.base.penalty)
    values=dict(task=tf,candidate=cf,C=C,D=D,task_valid=tv,candidate_valid=cv,
                scale=np.asarray(scales,np.float32),penalty=np.asarray(penalties,np.float64))
    return {k:torch.as_tensor(v,device=device) for k,v in values.items()}


def unpack_routes(actions, lengths):
    arr=actions.detach().cpu().numpy(); ls=lengths.detach().cpu().tolist()
    return [[0]+(arr[b,:k]+1).tolist()+[0] for b,k in enumerate(ls)]
