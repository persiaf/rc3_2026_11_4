"""Independent numeric STOP-advantage heads, two typed arm graph encoders."""
import copy
import numpy as np
import torch
from torch import nn
from rc3_rl_g2.actor_critic import ActorCritic
from rc3_rl_g2.graph_observation import build_policy_observation

def observation(env):
    raw=build_policy_observation(env)
    t=raw['task_ids'][raw['edge_task']];c=raw['candidate_nodes'][raw['edge_candidate']]-1
    raw['edge_left']=np.asarray(env.adapter.problem.F[t,c,0],bool)
    raw['edge_right']=np.asarray(env.adapter.problem.F[t,c,1],bool)
    e=env.observe();n=len(e['mask'])
    raw['extra']=np.asarray([[float(i==0 and env.rule_info is not None),e['steps_remaining']/16,e['work_after_menu']/2e6,
        e['current_best_gap']/max(e['anchor_cost'],1e-8),100*(e['anchor_cost']-e['best_cost'])/max(e['anchor_cost'],1e-8),
        len(e['history'])/16,len(env.attempt_memory)/32,len(env.rule_failed)/64] for i in range(n)],np.float32)
    return raw

class Value(nn.Module):
    def __init__(self):
        super().__init__();self.left=ActorCritic(kind='graph',hidden=32,message_steps=1);self.right=ActorCritic(kind='graph',hidden=32,message_steps=1)
        self.context=nn.Sequential(nn.Linear(8,32),nn.Tanh(),nn.Linear(32,1))
    def forward(self,batch):
        arms=[]
        for arm,model in [('left',self.left),('right',self.right)]:
            own=[]
            for raw in batch:
                item=dict(raw);mask=raw['edge_'+arm]
                item['edge_task']=raw['edge_task'][mask];item['edge_candidate']=raw['edge_candidate'][mask];own.append(item)
            arms.append(model(own)[0])
        values=(arms[0]+arms[1])/2
        for i,raw in enumerate(batch):
            n=len(raw['mask']);values[i,:n]=values[i,:n]+self.context(torch.as_tensor(raw['extra'])).squeeze(-1)
        return values
    @torch.no_grad()
    def select(self,env):
        raw=observation(env);scores=self([raw])[0,:len(raw['mask'])].numpy().copy()
        stop=next(i for i,a in enumerate(raw['actions']) if a['kind']=='stop');scores[stop]=0.
        chosen=int(scores.argmax());return chosen if scores[chosen]>0 else stop

def fit(model,data,target,updates,ledger,path):
    optimizer=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(731)
    logs=[]
    for iteration in range(updates):
        own=[data[int(i)] for i in rng.integers(0,len(data),16)]
        values=model([s['observation'] for s in own]);selected=values[torch.arange(len(own)),torch.as_tensor([s['action'] for s in own])]
        loss=nn.functional.smooth_l1_loss(selected,torch.tensor([s[target] for s in own],dtype=torch.float32))
        if not torch.isfinite(loss):raise ValueError('Nonfinite value loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
        ledger.use('optimizer_updates');optimizer.step();logs.append(dict(update=iteration+1,loss=float(loss.detach()),grad=float(norm)))
    from .core import write
    write(path,logs)
    return model

def save(path,model,**meta):torch.save(dict(state=model.state_dict(),**meta),path)
