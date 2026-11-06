"""Small global-context / local-coverage / ordered-prefix stochastic DRL policy."""
import math
import numpy as np
import torch
from torch import nn


GLOBAL_DIM=14


def global_tokens(parent):
    scene=parent.scene;D=parent.D;nx=scene['nx'];ny=scene['ny']
    tasks=np.bincount([t['table_id'] for t in scene['tasks']],minlength=len(scene['tables']))
    by_table={t['id']:[c['id']+1 for c in scene['candidates'] if c['table_id']==t['id']] for t in scene['tables']}
    rows=[]
    for t in scene['tables']:
        tid=t['id'];nodes=by_table[tid];adj=[]
        for dx,dy in ((-1,0),(1,0),(0,-1),(0,1)):
            x,y=t['ix']+dx,t['iy']+dy
            if 0<=x<nx and 0<=y<ny:
                costs=D[np.ix_(nodes,by_table[y*nx+x])];finite=costs[np.isfinite(costs)]
                adj.extend([float(finite.min())/10 if len(finite) else 0.,float(bool(len(finite)))])
            else:adj.extend([0.,0.])
        rows.append([t['cx']/10,t['cy']/10,(t['xmax']-t['xmin'])/10,(t['ymax']-t['ymin'])/10,
            tasks[tid]/3,len(scene.get('obstacles',[]))/100,*adj])
    return np.asarray(rows,dtype=np.float32)


def static_data(p):
    parent=p.parent;scene=parent.scene;n,m=p.C.shape
    entry=np.asarray(scene['start'] if p.entry==0 else [scene['candidates'][p.entry-1][k] for k in ('x','y')])
    tasks=np.asarray([[scene['tasks'][i][k] for k in ('x','y','z')] for i in p.task_ids],float).reshape(n,3)
    tf=np.column_stack([(tasks[:,:2]-entry)/10,tasks[:,2]/10,p.F[:,:,0].sum(1)/max(m,1),p.F[:,:,1].sum(1)/max(m,1)]).astype(np.float32)
    candidates=[scene['candidates'][j] for j in p.candidate_ids]
    cf=np.asarray([[(c['x']-entry[0])/10,(c['y']-entry[1])/10,np.cos(c['theta']),np.sin(c['theta']),c['h']/10,
                   p.C[:,k].sum()/max(n,1),p.D[k+1,0]/10] for k,c in enumerate(candidates)],np.float32).reshape(m,7)
    return dict(global_=torch.tensor(global_tokens(parent)),task=torch.tensor(tf),candidate=torch.tensor(cf),
        C=torch.tensor(p.C.astype(np.float32)),D=torch.tensor(p.D,dtype=torch.float64))


class GlobalLocalPolicy(nn.Module):
    def __init__(self,hidden=64):
        super().__init__();self.hidden=hidden
        self.global_net=nn.Sequential(nn.Linear(GLOBAL_DIM,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU())
        self.task_in=nn.Linear(5,hidden);self.candidate_in=nn.Linear(7,hidden)
        self.to_task=nn.Linear(hidden,hidden);self.to_candidate=nn.Linear(hidden,hidden)
        self.task_norm=nn.LayerNorm(hidden);self.candidate_norm=nn.LayerNorm(hidden)
        self.key=nn.Linear(hidden,hidden,bias=False);self.dynamic=nn.Linear(hidden,hidden,bias=False)
        self.nav=nn.Linear(4,hidden,bias=False)
        self.prefix=nn.GRUCell(hidden+2,hidden)
        self.query=nn.Linear(4*hidden+4,hidden)

    def encode(self,p):
        device=next(self.parameters()).device
        data={k:v.to(device) for k,v in static_data(p).items()};C=data['C'];t=self.task_in(data['task']);c=self.candidate_in(data['candidate'])
        tn=self.task_norm(t+torch.relu(self.to_task(C@c/C.sum(1).clamp_min(1)[:,None])))
        cn=self.candidate_norm(c+torch.relu(self.to_candidate(C.T@tn/C.sum(0).clamp_min(1)[:,None])))
        return dict(**data,t=tn,c=cn,keys=self.key(cn),g=self.global_net(data['global_']).mean(0),
            problem_identity=p.identity,model_identity=id(self),parameter_versions=tuple(w._version for w in self.parameters()),
            history_prefix=(),history_hidden=torch.zeros(self.hidden,device=device))

    def score(self,env,encoded):
        if env.done:raise ValueError('No action distribution after terminal')
        if encoded['problem_identity']!=env.problem.identity or encoded['model_identity']!=id(self) or encoded['parameter_versions']!=tuple(w._version for w in self.parameters()):
            raise ValueError('Static/prefix cache belongs to another problem or weight version')
        device=encoded['C'].device
        C=encoded['C'];rf=torch.tensor(env.remaining,dtype=torch.float32,device=device)
        gains=rf@C
        pool=C.T@(encoded['t']*rf[:,None])/gains.clamp_min(1)[:,None]
        remaining=(encoded['t']*rf[:,None]).sum(0)/rf.sum().clamp_min(1)
        current=encoded['c'][env.current-1] if env.current else torch.zeros(self.hidden,device=device)
        old=encoded['history_prefix']
        if tuple(env.prefix[:len(old)])==old:
            h=encoded['history_hidden'];start=len(old)
        else:h=torch.zeros(self.hidden,device=device);start=0
        for k in range(start,len(env.prefix)):
            node=env.prefix[k];previous=env.prefix[k-1] if k else 0
            extra=torch.tensor([env.problem.D[previous,node]/10,(k+1)/max(len(env.remaining),1)],dtype=torch.float32,device=device)
            h=self.prefix(torch.cat([encoded['c'][node-1],extra]),h)
        encoded['history_prefix']=tuple(env.prefix);encoded['history_hidden']=h
        scalars=torch.tensor([float(env.remaining.mean()),len(env.prefix)/max(len(rf),1),env.raw_L/100,env.problem.D[env.current,0]/10],dtype=torch.float32,device=device)
        q=torch.tanh(self.query(torch.cat([encoded['g'],remaining,current,h,scalars])))
        nav=torch.stack([encoded['D'][env.current,1:].float()/10,encoded['D'][1:,0].float()/10,
            gains/max(len(rf),1),torch.full_like(gains,.075)],1)
        nav=torch.where(torch.isfinite(nav),nav,0.)
        logits=(torch.tanh(encoded['keys']+self.dynamic(pool)+self.nav(nav))*q).sum(1)/math.sqrt(self.hidden)
        logits=10*torch.tanh(logits)
        mask=torch.tensor(env.mask(),device=device)
        if not bool(mask.any()):raise ValueError('Nonterminal state has no legal action')
        return logits.masked_fill(~mask,-torch.inf)


def tensor_hash(model):
    import hashlib
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()
