"""Parameter-matched coverage/navigation value model for the R1 protocol.

The representation variant is determined only by caller-supplied edgefeatures:
Euclidean distances for the basic control and real navigation distances for the
navigation variant. Both use the identical geometric neighbor indices, network
parameters, prefix representation, current/home distances and hard action set.
No rollout, planner, training loop, environment mutation or policy fallback is
defined in this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 64
    layers: int = 2
    task_dim: int = 5
    candidate_dim: int = 7
    edge_dim: int = 2
    distance_scale: float = 10.0
    score_scale: float = 0.2


class CNRValueNetwork(nn.Module):
    """Predict relative final-cost advantage; a larger score is preferred.

    Static input shapes:
      C [B,N,M] float coverage matrix
      D [B,M+1,M+1] physical navigation distances (depot is node0)
      task [B,N,5], candidate [B,M,7]
      task_valid [B,N], candidate_valid [B,M] Boolean padding masks
      neighbors [B,M,K] ZERO-based candidate indices, K normally20
      edgefeatures [B,M,K,2] = [distance/10m, finite_relation_flag]
      neighbor_valid [B,M,K] optional Boolean relation/padding mask

    The exact action mask belongs to the constraint kernel outside this model.
    Scores are finite for padded/inaccessible positions, which must be masked by
    the caller before selection. Only actually queried legal actions enter loss.
    """

    def __init__(self, hidden=64, layers=2, task_dim=5, candidate_dim=7,
                 edge_dim=2, distance_scale=10.0, score_scale=0.2):
        super().__init__()
        self.config = ModelConfig(hidden, layers, task_dim, candidate_dim,
                                  edge_dim, distance_scale, score_scale)
        h = hidden
        self.task_in = nn.Linear(task_dim, h)
        self.candidate_in = nn.Linear(candidate_dim, h)
        self.cover_to_task = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.cover_to_candidate = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.task_norm = nn.ModuleList(nn.LayerNorm(h) for _ in range(layers))
        self.candidate_norm = nn.ModuleList(nn.LayerNorm(h) for _ in range(layers))
        self.edge_in = nn.Linear(edge_dim, h)
        self.nav_message = nn.ModuleList(nn.Linear(h, h) for _ in range(layers))
        self.nav_norm = nn.ModuleList(nn.LayerNorm(h) for _ in range(layers))
        self.prefix_gru = nn.GRU(h + 1, h, batch_first=True)
        self.candidate_key = nn.Linear(h, h, bias=False)
        self.dynamic_task = nn.Linear(h, h, bias=False)
        self.context = nn.Linear(3*h + 5, h)
        self.dynamic_numbers = nn.Linear(6, h)
        self.head = nn.Sequential(nn.Linear(h, h//2), nn.ReLU(), nn.Linear(h//2, 1))
        # Both representations share a neutral starting policy before warm-up.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @property
    def hidden(self):
        return self.config.hidden

    def architecture_config(self):
        return asdict(self.config)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def _neighbors(values: Tensor, indices: Tensor) -> Tensor:
        """Gather [B,M,H] by [B,M,K] without a dense M-by-M relation tensor."""
        rows = torch.arange(values.shape[0], device=values.device)[:, None, None]
        return values[rows, indices]

    def static_encode(self, data: Mapping[str, Tensor]) -> dict[str, Tensor]:
        C = data['C'].float()
        task_valid = data['task_valid'].bool()
        candidate_valid = data['candidate_valid'].bool()
        C = C * task_valid[..., None] * candidate_valid[:, None, :]
        indices = data['neighbors'].long()
        edge = data['edgefeatures'].float()
        relation_valid = data.get('neighbor_valid')
        if relation_valid is None:
            relation_valid = edge[..., 1] > 0
        relation_valid = relation_valid.bool()
        neighbor_padding_valid = self._neighbors(candidate_valid[..., None], indices).squeeze(-1)
        relation_valid = relation_valid & neighbor_padding_valid & candidate_valid[..., None]
        # Sanitize masked infinities before any affine map. The true D below is
        # retained unchanged for exact current/home features and external masks.
        edge = torch.where(torch.isfinite(edge), edge, torch.zeros_like(edge))
        t = self.task_in(data['task'].float()) * task_valid[..., None]
        c = self.candidate_in(data['candidate'].float()) * candidate_valid[..., None]
        task_degree = C.sum(2).clamp_min(1)[..., None]
        candidate_degree = C.sum(1).clamp_min(1)[..., None]
        nav_edge = torch.relu(self.edge_in(edge))
        relation_weight = relation_valid.to(C.dtype)[..., None]
        relation_degree = relation_weight.sum(2).clamp_min(1)
        for i in range(self.config.layers):
            to_t = torch.bmm(C, c) / task_degree
            t_next = self.task_norm[i](t + torch.relu(self.cover_to_task[i](to_t)))
            to_c = torch.bmm(C.transpose(1, 2), t) / candidate_degree
            c_covered = self.candidate_norm[i](c + torch.relu(self.cover_to_candidate[i](to_c)))
            neighbors = self._neighbors(c_covered, indices)
            messages = torch.relu(self.nav_message[i](neighbors + nav_edge))
            pooled = (messages * relation_weight).sum(2) / relation_degree
            c = self.nav_norm[i](c_covered + pooled) * candidate_valid[..., None]
            t = t_next * task_valid[..., None]
        return dict(task=t, candidate=c, C=C, D=data['D'],
                    task_valid=task_valid, candidate_valid=candidate_valid,
                    neighbors=indices, neighbor_valid=relation_valid,
                    total_tasks=task_valid.sum(1).clamp_min(1).to(C.dtype),
                    total_candidates=candidate_valid.sum(1).clamp_min(1).to(C.dtype))

    @staticmethod
    def expand_encoded(encoded: Mapping[str, Tensor], batch_size: int) -> dict[str, Tensor]:
        """Broadcast a frozen single-instance encoding to parallel completions."""
        result = {}
        for key, value in encoded.items():
            if value.shape[0] == batch_size:
                result[key] = value
            elif value.shape[0] == 1:
                result[key] = value.expand((batch_size,) + value.shape[1:])
            else:
                raise ValueError('Cannot broadcast encoded batch dimension')
        return result

    @staticmethod
    def _all_nodes(encoded: Mapping[str, Tensor]) -> Tensor:
        c = encoded['candidate']
        return torch.cat((torch.zeros_like(c[:, :1]), c), 1)

    def encode_prefix(self, encoded: Mapping[str, Tensor], prefix_nodes: Tensor,
                      prefix_lengths: Tensor) -> Tensor:
        """Encode the ordered, unexecuted planning prefix; pad/depot node is0.

        prefix_nodes is [B,T] with node IDs1..M and0 padding. Length0 is legal.
        Every real prefix edge must be legal in the caller's exact state kernel.
        """
        c = encoded['candidate']
        b = c.shape[0]
        if prefix_nodes.shape[1] == 0:
            return c.new_zeros((b, self.hidden))
        nodes = prefix_nodes.long()
        rows = torch.arange(b, device=c.device)[:, None]
        all_nodes = self._all_nodes(encoded)
        embeddings = all_nodes[rows, nodes]
        previous = torch.cat((torch.zeros_like(nodes[:, :1]), nodes[:, :-1]), 1)
        distance = encoded['D'][rows, previous, nodes].float() / self.config.distance_scale
        distance = torch.where(torch.isfinite(distance), distance, torch.zeros_like(distance))
        valid = torch.arange(nodes.shape[1], device=c.device)[None, :] < prefix_lengths[:, None]
        inputs = torch.cat((embeddings, distance[..., None]), -1) * valid[..., None]
        output, _ = self.prefix_gru(inputs)
        last = (prefix_lengths.long()-1).clamp_min(0)
        hidden = output[torch.arange(b, device=c.device), last]
        return torch.where((prefix_lengths>0)[:, None], hidden, torch.zeros_like(hidden))

    # An explicit alias makes the prefix API discoverable for orchestration.
    prefix_hidden = encode_prefix

    def update_prefix(self, encoded: Mapping[str, Tensor], hidden: Tensor,
                      node: Tensor, current: Tensor, active: Tensor | None = None) -> Tensor:
        """One exact chosen-node update, compatible with encode_prefix()."""
        node = node.long()
        current = current.long()
        rows = torch.arange(node.shape[0], device=node.device)
        embedding = self._all_nodes(encoded)[rows, node]
        distance = encoded['D'][rows, current, node].float()/self.config.distance_scale
        distance = torch.where(torch.isfinite(distance), distance, torch.zeros_like(distance))
        inputs = torch.cat((embedding, distance[:, None]), -1)[:, None, :]
        _, result = self.prefix_gru(inputs, hidden[None, :, :])
        updated = result[0]
        return updated if active is None else torch.where(active[:, None], updated, hidden)

    def score(self, encoded: Mapping[str, Tensor], remaining: Tensor,
              current: Tensor, prefix_hidden: Tensor, prefix_rawL: Tensor,
              step_count: Tensor) -> Tensor:
        """Score the complete candidate menu; caller applies the hard mask."""
        C = encoded['C']
        rf = remaining.to(C.dtype) * encoded['task_valid'].to(C.dtype)
        b, _, m = C.shape
        rows = torch.arange(b, device=C.device)
        gains = torch.bmm(rf[:, None, :], C).squeeze(1)
        task_sum = torch.bmm(C.transpose(1,2), encoded['task']*rf[..., None])
        pool = task_sum/gains.clamp_min(1)[..., None]
        left_count = rf.sum(1)
        global_task = (encoded['task']*rf[...,None]).sum(1)/left_count.clamp_min(1)[:,None]
        current_embedding = self._all_nodes(encoded)[rows, current.long()]
        totals = encoded['total_tasks']
        context_numbers = torch.stack((left_count/totals, step_count.float()/totals,
            prefix_rawL.float()/100., totals/200., encoded['total_candidates']/800.), -1)
        context = torch.cat((global_task, current_embedding, prefix_hidden, context_numbers), -1)
        travel = encoded['D'][rows, current.long(), 1:].float()/self.config.distance_scale
        home = encoded['D'][:,1:,0].float()/self.config.distance_scale
        travel = torch.where(torch.isfinite(travel), travel, torch.zeros_like(travel))
        home = torch.where(torch.isfinite(home), home, torch.zeros_like(home))
        task_choices = C.sum(2).clamp_min(1)
        scarcity = torch.bmm((rf/task_choices)[:,None,:],C).squeeze(1)/left_count.clamp_min(1)[:,None]
        neighbor_gain = self._neighbors((gains/totals[:,None])[...,None],encoded['neighbors']).squeeze(-1)
        mask = encoded['neighbor_valid'].to(C.dtype)
        neighbor_gain = (neighbor_gain*mask).sum(2)/mask.sum(2).clamp_min(1)
        numbers = torch.stack((travel, home, gains/totals[:,None], scarcity,
                               neighbor_gain, (gains>0).to(C.dtype)), -1)
        mixed = torch.tanh(self.candidate_key(encoded['candidate']) + self.dynamic_task(pool)
                           + self.context(context)[:,None,:] + self.dynamic_numbers(numbers))
        return self.config.score_scale*torch.tanh(self.head(mixed).squeeze(-1))

    def forward_state(self, data: Mapping[str, Tensor], remaining: Tensor,
                      current: Tensor, prefix_nodes: Tensor, prefix_lengths: Tensor,
                      prefix_rawL: Tensor) -> Tensor:
        encoded = self.static_encode(data)
        hidden = self.encode_prefix(encoded,prefix_nodes,prefix_lengths)
        return self.score(encoded,remaining,current,hidden,prefix_rawL,prefix_lengths)

    def forward(self, data: Mapping[str, Tensor], remaining: Tensor,
                current: Tensor, prefix_nodes: Tensor, prefix_lengths: Tensor,
                prefix_rawL: Tensor) -> Tensor:
        return self.forward_state(data,remaining,current,prefix_nodes,prefix_lengths,prefix_rawL)


# Short public name for experiment code and portable checkpoint config loading.
Model = CNRValueNetwork
