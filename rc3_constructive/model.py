"""Two-layer bipartite encoder and dynamic coverage attention with a declared prior."""
from __future__ import annotations
import math
import torch
from torch import nn


class Policy(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__(); self.hidden=hidden; self.layers=layers
        self.task_in=nn.Linear(5,hidden); self.candidate_in=nn.Linear(7,hidden)
        self.to_task=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(layers)])
        self.to_candidate=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(layers)])
        self.task_norm=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.candidate_norm=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.key=nn.Linear(hidden,hidden,bias=False);self.dynamic=nn.Linear(hidden,hidden,bias=False)
        self.navigation=nn.Linear(4,hidden,bias=False)
        self.query=nn.Linear(2*hidden+3,hidden)
        nn.init.zeros_(self.query.weight);nn.init.zeros_(self.query.bias)
        self.prior_weight=nn.Parameter(torch.tensor(math.log(math.expm1(4.))))

    def encode(self, data):
        C=data['C']; t=self.task_in(data['task']); c=self.candidate_in(data['candidate'])
        dt=C.sum(2).clamp_min(1).unsqueeze(-1);dc=C.sum(1).clamp_min(1).unsqueeze(-1)
        for i in range(self.layers):
            tnext=self.task_norm[i](t+torch.relu(self.to_task[i](torch.bmm(C,c)/dt)))
            cnext=self.candidate_norm[i](c+torch.relu(self.to_candidate[i](torch.bmm(C.transpose(1,2),t)/dc)))
            t=tnext*data['task_valid'].unsqueeze(-1);c=cnext*data['candidate_valid'].unsqueeze(-1)
        return t,c,self.key(c)

    def forward(self, data, sample=False):
        t,c,keys=self.encode(data);C=data['C'];B,N,M=C.shape
        remaining=data['task_valid'];current=torch.zeros(B,dtype=torch.long,device=C.device)
        lengths=torch.zeros_like(current);cost=torch.zeros(B,dtype=torch.float64,device=C.device)
        logsum=torch.zeros(B,device=C.device);actions=[];rows=torch.arange(B,device=C.device)
        allnodes=torch.cat([torch.zeros_like(c[:,:1]),c],1)
        totals=remaining.sum(1).clamp_min(1)
        for _ in range(N):
            active=remaining.any(1)
            if not bool(active.any()): break
            rf=remaining.to(C.dtype);gains=torch.bmm(rf.unsqueeze(1),C).squeeze(1)
            pool=torch.bmm(C.transpose(1,2),t*rf.unsqueeze(-1))/gains.clamp_min(1).unsqueeze(-1)
            global_t=(t*rf.unsqueeze(-1)).sum(1)/rf.sum(1).clamp_min(1).unsqueeze(-1)
            travel=data['D'][rows,current,1:];home=data['D'][:,1:,0]
            fraction=rf.sum(1)/totals
            context=torch.cat([global_t,allnodes[rows,current],fraction[:,None],
                (lengths/totals)[:,None],(data['penalty'].float()/data['scale'])[:,None]],1)
            query=self.query(context)
            # Physical d and lambda share units; both are normalized by the same scale.
            nav=torch.stack([travel.float()/data['scale'][:,None],home.float()/data['scale'][:,None],
                             gains/totals[:,None],(data['penalty'].float()/data['scale'])[:,None].expand(-1,M)],-1)
            nav=torch.where(torch.isfinite(nav),nav,0.)
            relation=(torch.tanh(keys+self.dynamic(pool)+self.navigation(nav))*query[:,None,:]).sum(-1)/math.sqrt(self.hidden)
            # Direct numeric navigation bias: no Euclidean obstacle approximation.
            safe_travel=torch.where(torch.isfinite(travel),travel,0.)
            prior=torch.log(gains.clamp_min(1))-torch.log((safe_travel+data['penalty'][:,None]).float().clamp_min(1e-6))
            score=10*torch.tanh(relation)+torch.nn.functional.softplus(self.prior_weight)*prior
            mask=(gains>0)&data['candidate_valid']&torch.isfinite(travel)&torch.isfinite(home)
            if bool((active & ~mask.any(1)).any()):raise ValueError('uncovered tasks but no valid action')
            mask=mask.clone();mask[~active,0]=True
            score=torch.where(active[:,None],score,0.)
            score=score.masked_fill(~mask,-torch.inf)
            logp=torch.log_softmax(score,-1)
            selected=torch.multinomial(logp.exp(),1).squeeze(1) if sample else score.argmax(1)
            logsum=logsum+torch.where(active,logp[rows,selected],0.)
            served=C[rows,:,selected]>0
            new_remaining=remaining & ~served
            finished=active & ~new_remaining.any(1)
            move=travel[rows,selected]+data['penalty']+torch.where(finished,home[rows,selected],0.)
            cost=cost+torch.where(active,move,0.);lengths=lengths+active
            current=torch.where(active,selected+1,current);remaining=new_remaining;actions.append(selected)
        if bool(remaining.any()):raise ValueError('horizon exceeded')
        out=torch.stack(actions,1) if actions else torch.empty((B,0),dtype=torch.long,device=C.device)
        return cost,logsum,out,lengths
