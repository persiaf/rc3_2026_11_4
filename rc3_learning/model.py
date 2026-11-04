"""Size-independent attention policy with roadmap distance in both encoder and decoder."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class DistanceAttentionLayer(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3*dim)
        self.output = nn.Linear(dim, dim)
        self.distance_weight = nn.Parameter(torch.zeros(heads))
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2*dim), nn.ReLU(), nn.Linear(2*dim, dim))

    def forward(self, nodes, distances, valid):
        b, m, d = nodes.shape
        q, k, v = self.qkv(nodes).reshape(b, m, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        score = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        score = score - F.softplus(self.distance_weight)[None, :, None, None]*distances[:, None]
        score = score.masked_fill(~valid[:, None, None, :], -torch.inf)
        attended = (score.softmax(-1) @ v).transpose(1, 2).reshape(b, m, d)
        nodes = self.norm1(nodes + self.output(attended))
        nodes = self.norm2(nodes + self.ff(nodes))
        return nodes.masked_fill(~valid.unsqueeze(-1), 0.)


class AttentionPolicy(nn.Module):
    def __init__(self, embed_dim=128, layers=2, heads=4):
        super().__init__()
        if embed_dim < 1 or heads < 1 or embed_dim % heads or layers < 1:
            raise ValueError('positive embed_dim must be divisible by heads; layers must be positive')
        self.config = dict(embed_dim=embed_dim, layers=layers, heads=heads)
        self.task_embedding = nn.Sequential(nn.Linear(6, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.node_embedding = nn.Linear(9+embed_dim, embed_dim)
        self.layers = nn.ModuleList([DistanceAttentionLayer(embed_dim, heads) for _ in range(layers)])
        self.depot = nn.Parameter(torch.zeros(embed_dim))
        self.query = nn.Linear(3*embed_dim+1, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.remaining_key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dynamic_score = nn.Sequential(nn.Linear(6, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def encode(self, batch):
        task = self.task_embedding(batch['task_features'])*batch['task_mask'].unsqueeze(-1)
        cover = batch['cover'].float()*batch['task_mask'].unsqueeze(-1)
        aggregate = cover.transpose(1, 2) @ task / cover.sum(1).clamp_min(1).unsqueeze(-1)
        nodes = self.node_embedding(torch.cat((batch['node_features'], aggregate), dim=-1))
        distances = batch['costs'][:, 1:, 1:] / batch['length_scale'][:, None, None]
        distances = torch.nan_to_num(distances, posinf=4.)
        valid = batch['candidate_mask']
        for layer in self.layers:
            nodes = layer(nodes, distances, valid)
        graph = (nodes*valid.unsqueeze(-1)).sum(1)/valid.sum(1).clamp_min(1).unsqueeze(-1)
        depot = self.depot[None, None].expand(nodes.shape[0], 1, -1)
        return dict(task=task, nodes=nodes, graph=graph, keys=self.key(nodes),
                    all_nodes=torch.cat((depot, nodes), dim=1))

    def forward(self, td, cache):
        remaining = td['remaining'].float()*td['task_mask']
        cover = td['cover'].float()*remaining.unsqueeze(-1)
        gains = cover.sum(1)
        count = remaining.sum(1).clamp_min(1)
        total = td['task_mask'].sum(1).clamp_min(1)
        task_pool = (cache['task']*remaining.unsqueeze(-1)).sum(1)/count.unsqueeze(-1)
        local = cover.transpose(1, 2) @ cache['task']/gains.clamp_min(1).unsqueeze(-1)
        rows = torch.arange(td.batch_size[0], device=td.device)
        current = cache['all_nodes'][rows, td['current_node']]
        query = self.query(torch.cat((current, cache['graph'], task_pool, (count/total).unsqueeze(-1)), -1))
        score = ((cache['keys']+self.remaining_key(local))*query.unsqueeze(1)).sum(-1)/math.sqrt(query.shape[-1])
        travel = td['costs'][rows, td['current_node'], 1:]/td['length_scale'].unsqueeze(-1)
        returning = td['costs'][:, 1:, 0]/td['length_scale'].unsqueeze(-1)
        arm_gain = (td['feasibility'] & td['remaining'][:, :, None, None]).sum(1).float()
        dynamic = torch.stack((gains/total[:, None], gains/count[:, None],
                               torch.nan_to_num(travel, posinf=4.), torch.nan_to_num(returning, posinf=4.),
                               arm_gain[:, :, 0]/count[:, None], arm_gain[:, :, 1]/count[:, None]), -1)
        logits = 10*torch.tanh(score+self.dynamic_score(dynamic).squeeze(-1))
        mask = td['action_mask'].clone()
        # A numerical sentinel for finished rows only. It is never a service action.
        mask[td['done'].squeeze(-1), 0] = True
        return logits.masked_fill(~mask, -torch.inf)


def rollout(policy, env, batch, mode='greedy'):
    if mode not in ('greedy', 'sample', 'teacher'):
        raise ValueError('mode must be greedy, sample or teacher')
    td = env.reset(batch.clone(recurse=False))
    cache = policy.encode(td)
    b = td.batch_size[0]
    log_likelihood = torch.zeros(b, device=td.device)
    reward_sum = torch.zeros(b, device=td.device)
    entropy_sum = torch.zeros(b, device=td.device)
    history = []
    limit = int(td['task_mask'].sum(1).max().item())
    for step in range(limit):
        active = ~td['done'].squeeze(-1)
        if not active.any():
            break
        logits = policy(td, cache)
        distribution = torch.distributions.Categorical(logits=logits)
        if mode == 'teacher':
            if step >= batch['teacher_actions'].shape[1]:
                raise ValueError('teacher ended before task completion')
            if (batch['teacher_actions'][active, step] < 0).any():
                raise ValueError('missing teacher action for an active instance')
            action = batch['teacher_actions'][:, step].clamp_min(0)
        else:
            action = distribution.sample() if mode == 'sample' else logits.argmax(-1)
        log_likelihood = log_likelihood + torch.where(active, distribution.log_prob(action), 0.)
        entropy_sum = entropy_sum + torch.where(active, distribution.entropy(), 0.)
        history.append(torch.where(active, action, -1))
        td['action'] = action
        td = env.step(td)['next']
        reward_sum += td['reward'].squeeze(-1)
    if not td['done'].all():
        raise RuntimeError('rollout exceeded the number of tasks')
    actions = torch.stack(history, 1) if history else torch.empty(b, 0, device=td.device, dtype=torch.long)
    return dict(state=td, actions=actions, log_likelihood=log_likelihood,
                reward=reward_sum, normalized_cost=td['total_cost']/td['reference_cost'],
                cost=td['total_cost'], entropy=entropy_sum/td['service_count'].clamp_min(1))
