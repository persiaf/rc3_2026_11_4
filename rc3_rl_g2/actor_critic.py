"""Small shared-action MLP and explicit sparse local-graph actor-critic."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .graph_observation import ACTION_DIM, TASK_DIM, CANDIDATE_DIM, GRAPH_SCHEMA


def _mean_pool(values, index, groups):
    result = values.new_zeros((groups, values.shape[-1]))
    if len(index):
        result.index_add_(0, index, values)
        counts = torch.bincount(index, minlength=groups).clamp_min(1).to(values.dtype)
        result = result / counts[:, None]
    return result


class ActorCritic(nn.Module):
    def __init__(self, kind='graph', hidden=64, message_steps=1):
        super().__init__()
        if kind not in ('graph', 'mlp') or hidden not in (32, 64) or message_steps not in (1, 2):
            raise ValueError('unsupported G2 architecture')
        self.kind, self.hidden, self.message_steps = kind, hidden, message_steps
        # Fixed engineering scales preserve absolute physical units. They are
        # not fitted to the validation set and do not depend on batch contents.
        scales = [1.,1.,1.,1.,4.,8.,80.,10.,10.,10.,1.,7.,7.,8.,4.,4.,8.,1.,16.,1.,1.,16.,
                  2.,2.,1.,1.,1.,3.,3.,3.]
        self.register_buffer('feature_scale', torch.tensor(scales, dtype=torch.float32))
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM,hidden),nn.Tanh(),
                                            nn.Linear(hidden,hidden),nn.Tanh())
        if kind == 'graph':
            self.task_encoder = nn.Sequential(nn.Linear(TASK_DIM,hidden),nn.Tanh())
            self.candidate_encoder = nn.Sequential(nn.Linear(CANDIDATE_DIM,hidden),nn.Tanh())
            self.task_updates = nn.ModuleList(nn.Sequential(nn.Linear(2*hidden,hidden),nn.Tanh()) for _ in range(message_steps))
            self.candidate_updates = nn.ModuleList(nn.Sequential(nn.Linear(3*hidden,hidden),nn.Tanh()) for _ in range(message_steps))
            self.fusion = nn.Sequential(nn.Linear(3*hidden,hidden),nn.Tanh())
        self.actor = nn.Linear(hidden,1)
        self.critic = nn.Sequential(nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.)

    @property
    def device(self):return next(self.parameters()).device

    def architecture(self):
        """Constructor kwargs, so ActorCritic(**checkpoint['architecture']) works."""
        return dict(kind=self.kind,hidden=self.hidden,message_steps=self.message_steps)

    def _tensor(self, values, dtype=torch.float32):
        return torch.as_tensor(values, dtype=dtype, device=self.device)

    def forward(self, observations):
        if not isinstance(observations, (list,tuple)) or not observations:
            raise ValueError('forward requires a nonempty list of raw observations')
        counts = [len(o['mask']) for o in observations]
        if any(n == 0 for n in counts):raise ValueError('terminal observations have no actor distribution')
        for o,n in zip(observations,counts):
            if o.get('schema') != GRAPH_SCHEMA:raise ValueError('unknown policy observation schema')
            if np.asarray(o['features']).shape != (n,ACTION_DIM):raise ValueError('invalid action feature shape')
            if np.asarray(o['mask']).dtype != np.dtype(bool) or not np.asarray(o['mask']).any():
                raise ValueError('policy requires at least one legal action')
        total = sum(counts)
        features = self._tensor(np.concatenate([o['features'] for o in observations]))
        if not torch.isfinite(features).all():raise ValueError('nonfinite action input')
        h = self.action_encoder(features/self.feature_scale)
        if self.kind == 'graph':
            tasks, candidates, task_actions, candidate_actions = [],[],[],[]
            edge_t, edge_c, nav_s, nav_t, nav_d = [],[],[],[],[]
            action_offset = task_offset = candidate_offset = 0
            for o,n in zip(observations,counts):
                tasks.append(o['task_features']);candidates.append(o['candidate_features'])
                task_actions.append(o['task_action']+action_offset)
                candidate_actions.append(o['candidate_action']+action_offset)
                edge_t.append(o['edge_task']+task_offset);edge_c.append(o['edge_candidate']+candidate_offset)
                nav_s.append(o['nav_source']+candidate_offset);nav_t.append(o['nav_target']+candidate_offset)
                nav_d.append(o['nav_distance'])
                action_offset += n;task_offset += len(o['task_features']);candidate_offset += len(o['candidate_features'])
            tx=self._tensor(np.concatenate(tasks));cx=self._tensor(np.concatenate(candidates))
            if not torch.isfinite(tx).all() or not torch.isfinite(cx).all():raise ValueError('nonfinite graph node input')
            ht,hc=self.task_encoder(tx),self.candidate_encoder(cx)
            ti=self._tensor(np.concatenate(edge_t),torch.long);ci=self._tensor(np.concatenate(edge_c),torch.long)
            ns=self._tensor(np.concatenate(nav_s),torch.long);nt=self._tensor(np.concatenate(nav_t),torch.long)
            nd=self._tensor(np.concatenate(nav_d))
            if not torch.isfinite(nd).all() or (nd<0).any():raise ValueError('invalid navigation edge distances')
            ta=self._tensor(np.concatenate(task_actions),torch.long);ca=self._tensor(np.concatenate(candidate_actions),torch.long)
            for tupdate,cupdate in zip(self.task_updates,self.candidate_updates):
                tmessage=_mean_pool(hc[ci],ti,len(ht))
                cmessage=_mean_pool(ht[ti],ci,len(hc))
                nmessage=_mean_pool(hc[ns]/(1.+nd[:,None]),nt,len(hc))
                ht,hc=tupdate(torch.cat((ht,tmessage),-1)),cupdate(torch.cat((hc,cmessage,nmessage),-1))
            h=self.fusion(torch.cat((h,_mean_pool(ht,ta,total),_mean_pool(hc,ca,total)),-1))
        logits_flat=self.actor(h).squeeze(-1)
        max_actions=max(counts)
        logits=logits_flat.new_full((len(observations),max_actions),-torch.inf)
        pooled=[];offset=0
        for batch,(o,n) in enumerate(zip(observations,counts)):
            mask=self._tensor(np.asarray(o['mask']),torch.bool)
            logits[batch,:n]=logits_flat[offset:offset+n].masked_fill(~mask,-torch.inf)
            pooled.append(h[offset:offset+n][mask].mean(0))
            offset+=n
        values=self.critic(torch.stack(pooled)).squeeze(-1)
        return logits,values

    def evaluate_actions(self, observations, actions):
        logits,values=self(observations)
        actions=self._tensor(actions,torch.long)
        if actions.shape != values.shape:raise ValueError('one action per observation is required')
        if (actions<0).any() or (actions>=logits.shape[1]).any():raise ValueError('action index outside padded distribution')
        if not torch.isfinite(logits.gather(1,actions[:,None])).all():raise ValueError('selected action was masked or padding')
        distribution=Categorical(logits=logits)
        return distribution.log_prob(actions),distribution.entropy(),values

    @torch.no_grad()
    def act(self, observations, deterministic=False):
        logits,values=self(observations)
        distribution=Categorical(logits=logits)
        actions=logits.argmax(-1) if deterministic else distribution.sample()
        return actions,distribution.log_prob(actions),values
