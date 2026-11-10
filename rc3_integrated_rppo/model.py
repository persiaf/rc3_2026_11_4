"""Frozen SL base, zero-output residual head; independent268-dim value model."""
import math
import numpy as np
import torch
from torch import nn
from .common import require,tensor_digest,seed_for
from rc3_global_local.policy import GlobalLocalPolicy

ARCH=dict(actor_input=202,critic_input=268,actor_hidden=[64,64],critic_hidden=[128,128],amplitude=2.,dtype='float32',device='cpu')


class Actor(nn.Module):
    def __init__(self,base_state,seed,device='cpu'):
        super().__init__()
        with torch.device('meta'):self.base=GlobalLocalPolicy()
        self.base.load_state_dict(base_state,strict=True,assign=True);self.base.eval().requires_grad_(False)
        with torch.random.fork_rng():
            torch.manual_seed(seed_for(seed,'residual_init'))
            self.head=nn.Sequential(nn.Linear(202,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        self.head.to(device)
        self.base_sha=tensor_digest(self.base.state_dict())
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def encode(self,p):
        from rc3_global_local.problem import array_hash
        with torch.no_grad():encoded=self.base.encode(p)
        encoded['integrated_D_sha256']=array_hash(p.D)
        return encoded
    def features(self,env,encoded):
        with torch.no_grad():
            z0=self.base.score(env,encoded);C=encoded['C'];rf=torch.as_tensor(env.remaining.copy(),dtype=torch.float32);gains=rf@C;n=max(len(rf),1)
            pool=C.T@(encoded['t']*rf[:,None])/gains.clamp_min(1)[:,None]
            remaining=(encoded['t']*rf[:,None]).sum(0)/rf.sum().clamp_min(1)
            current=encoded['c'][env.current-1] if env.current else torch.zeros(64)
            h=encoded['history_hidden'];oldscalars=torch.tensor([float(env.remaining.mean()),len(env.prefix)/n,env.raw_L/100,env.problem.D[env.current,0]/10],dtype=torch.float32)
            q=torch.tanh(self.base.query(torch.cat([encoded['g'],remaining,current,h,oldscalars])))
            scarcity=(rf/C.sum(1).clamp_min(1))@C/n;p=env.problem
            common=[float(rf.mean()),len(env.prefix)/n,env.raw_L/100,p.window['width']/10,p.window['height']/10,float(p.entry!=p.exit)]
            u=torch.stack([gains/n,scarcity,encoded['D'][env.current,1:].float()/10,encoded['D'][1:,0].float()/10,*[torch.full_like(gains,v) for v in common]],1)
            u=torch.where(torch.isfinite(u),u,0.)
            x=torch.cat([q.expand(len(gains),-1),encoded['c'],pool,u],1)
            scene=p.parent.scene
            def xy(node):return scene['start'] if node==0 else [scene['candidates'][node-1][k] for k in ('x','y')]
            global_current=p.entry if env.current==0 else int(p.candidate_ids[env.current-1])+1
            context=[float(rf.mean()),len(env.prefix)/n,env.raw_L/100,*[v/10 for node in (global_current,p.entry,p.exit) for v in xy(node)],p.window['width']/10,p.window['height']/10,float(p.entry!=p.exit)]
            v=torch.cat([encoded['g'],remaining,current,h,torch.tensor(context,dtype=torch.float32)])
            mask=torch.tensor(env.mask().copy());require(x.shape[1]==202 and v.shape==(268,),'Feature dimensions')
            key=(p.identity,tuple(env.prefix),self.base_sha,encoded['integrated_D_sha256'])
            return dict(x=x.detach(),v=v.detach(),z0=z0.detach(),mask=mask,key=key)
    def logits(self,features):
        device=next(self.head.parameters()).device
        adjustment=2*torch.tanh(self.head(features['x'].to(device)).squeeze(-1))
        return (features['z0']+adjustment.to(features['z0'].device)).masked_fill(~features['mask'],-torch.inf)
    def score(self,env,encoded):return self.logits(self.features(env,encoded))


class Critic(nn.Module):
    def __init__(self,seed):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed_for(seed,'critic_init'));self.net=nn.Sequential(nn.Linear(268,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def pack_states(states):
    if not states:return None
    require(all(len({s['key'][k] for s in states})==1 for k in (0,2,3)),'Foreign task/base/D cache')
    return dict(x=torch.stack([s['x'] for s in states]),v=torch.stack([s['v'] for s in states]),
        z0=torch.stack([s['z0'] for s in states]),mask=torch.stack([s['mask'] for s in states]),keys=[s['key'] for s in states])


def batch_logits(actor,b):
    T,M,F=b['x'].shape
    device=next(actor.head.parameters()).device
    return (b['z0'].to(device)+2*torch.tanh(actor.head(b['x'].reshape(T*M,F).to(device)).reshape(T,M))).masked_fill(~b['mask'].to(device),-torch.inf)
