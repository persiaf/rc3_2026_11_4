"""RC3 relation Actor: navigation edges and dynamic remaining coverage.

This module does not modify the RC3 environment, action set, cost, H, or
independent validator.  Construction and complete-H return already existed.
No model/optimizer/rollout executes on import.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

from rc3_global_local.policy import GLOBAL_DIM, static_data, tensor_hash


MODEL_ID = "rc3-navigation-remaining-coverage-relation-v1"


def _masked_weights(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Softmax with exactly zero mass (and no NaN) for unsupported rows."""
    if logits.shape[-1] == 0:
        return torch.zeros_like(logits)
    mask = allowed.unsqueeze(0)
    masked = logits.masked_fill(~mask, -torch.inf)
    supported = allowed.any(-1)[None, :, None]
    safe = torch.where(supported, masked, torch.zeros_like(masked))
    return torch.softmax(safe, dim=-1) * mask.to(logits.dtype)


class EdgeAttention(nn.Module):
    """Multihead attention over explicit allowed edges, with learned edge bias."""

    def __init__(self, hidden: int, heads: int, edge_dim: int, chunk_size: int):
        super().__init__()
        self.hidden, self.heads = hidden, heads
        self.width, self.chunk_size = hidden // heads, chunk_size
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.value = nn.Linear(hidden, hidden, bias=False)
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.output = nn.Linear(hidden, hidden, bias=False)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], self.heads, self.width).transpose(0, 1)

    def prepare(self, target, source, allowed, edges):
        """Return normalized support weights/values, reusable for remaining masks."""
        q = self._heads(self.query(target))
        k = self._heads(self.key(source))
        v = self._heads(self.value(source))
        chunks = []
        for start in range(0, len(target), self.chunk_size):
            end = min(start + self.chunk_size, len(target))
            logits = torch.matmul(q[:, start:end], k.transpose(-1, -2))
            logits = logits / math.sqrt(self.width)
            logits = logits + self.edge_bias(edges[start:end]).permute(2, 0, 1)
            chunks.append(_masked_weights(logits, allowed[start:end]))
        weights = (torch.cat(chunks, dim=1) if chunks else
                   target.new_empty((self.heads, 0, len(source))))
        return weights, v

    def aggregate(self, weights, values):
        mixed = torch.matmul(weights, values).transpose(0, 1)
        return self.output(mixed.reshape(mixed.shape[0], self.hidden))

    def forward(self, target, source, allowed, edges):
        # Query chunks bound temporary allocations.  All legal edges remain;
        # there is no candidate or nearest-neighbor pruning.
        q = self._heads(self.query(target))
        k = self._heads(self.key(source))
        v = self._heads(self.value(source))
        chunks = []
        for start in range(0, len(target), self.chunk_size):
            end = min(start + self.chunk_size, len(target))
            logits = torch.matmul(q[:, start:end], k.transpose(-1, -2))
            logits = logits / math.sqrt(self.width)
            logits = logits + self.edge_bias(edges[start:end]).permute(2, 0, 1)
            weights = _masked_weights(logits, allowed[start:end])
            chunks.append(torch.matmul(weights, v).transpose(0, 1))
        mixed = (torch.cat(chunks, dim=0) if chunks else
                 target.new_empty((0, self.heads, self.width)))
        return self.output(mixed.reshape(len(target), self.hidden))


class RelationBlock(nn.Module):
    def __init__(self, hidden, heads, chunk_size):
        super().__init__()
        self.task_support = EdgeAttention(hidden, heads, 2, chunk_size)
        self.candidate_support = EdgeAttention(hidden, heads, 2, chunk_size)
        self.navigation = EdgeAttention(hidden, heads, 2, chunk_size)
        self.task_norm = nn.LayerNorm(hidden)
        self.cover_norm = nn.LayerNorm(hidden)
        self.nav_norm = nn.LayerNorm(hidden)
        self.task_ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(),
                                     nn.Linear(2 * hidden, hidden))
        self.candidate_ff = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.ReLU(),
                                          nn.Linear(2 * hidden, hidden))
        self.task_ff_norm = nn.LayerNorm(hidden)
        self.candidate_ff_norm = nn.LayerNorm(hidden)

    def forward(self, tasks, candidates, E, C, nav_allowed, nav_edges):
        tasks = self.task_norm(tasks + self.task_support(tasks, candidates, C, E))
        tasks = self.task_ff_norm(tasks + self.task_ff(tasks))
        candidates = self.cover_norm(candidates + self.candidate_support(
            candidates, tasks, C.T, E.permute(1, 0, 2)))
        candidates = self.nav_norm(candidates + self.navigation(
            candidates, candidates, nav_allowed, nav_edges))
        candidates = self.candidate_ff_norm(candidates + self.candidate_ff(candidates))
        return tasks, candidates


class RelationActor(nn.Module):
    """One variable-cardinality Actor with the historical encode/score API.

    `encode` is per problem and parameter version. `fork_cache` shares its
    differentiable static tensors but gives a trajectory independent prefix.
    One backward may aggregate the four same-problem trajectories; reusing a
    consumed autograd graph across separate backwards is the caller's error.
    """

    def __init__(self, hidden=64, heads=4, blocks=2, chunk_size=64):
        super().__init__()
        if hidden <= 0 or heads <= 0 or hidden % heads or blocks < 1 or chunk_size < 1:
            raise ValueError("Invalid relation Actor dimensions")
        self.hidden = int(hidden)
        self.config = dict(model_id=MODEL_ID, hidden=hidden, heads=heads,
                           blocks=blocks, chunk_size=chunk_size)
        self.task_in = nn.Linear(5, hidden)
        self.candidate_in = nn.Linear(7, hidden)
        self.global_net = nn.Sequential(nn.Linear(GLOBAL_DIM, hidden), nn.ReLU(),
                                        nn.Linear(hidden, hidden), nn.ReLU())
        self.blocks = nn.ModuleList(RelationBlock(hidden, heads, chunk_size)
                                    for _ in range(blocks))
        self.remaining_attention = EdgeAttention(hidden, heads, 2, chunk_size)
        self.remaining_norm = nn.LayerNorm(hidden)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.dynamic = nn.Linear(hidden, hidden, bias=False)
        self.nav = nn.Linear(4, hidden, bias=False)
        self.prefix = nn.GRUCell(hidden + 2, hidden)
        self.query = nn.Linear(4 * hidden + 4, hidden)
        self._cache_epoch = 0

    def parameter_fingerprint(self):
        # _version changes on normal optimizer steps and copy_/load_state_dict.
        # Identity/dtype/device/requires_grad also catch replacements and freezes.
        # Unsafely mutating parameter.data is unsupported and never used here.
        return tuple((name, id(p), p._version, str(p.dtype), str(p.device),
                      p.requires_grad) for name, p in self.named_parameters())

    def load_state_dict(self, state_dict, strict=True, assign=False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._cache_epoch += 1
        return result

    def encode(self, problem) -> dict[str, Any]:
        device = next(self.parameters()).device
        data = {k: v.to(device) for k, v in static_data(problem).items()}
        # LocalProblem.F is already the effective same-arm F&A table.
        E = torch.tensor(np.array(problem.F, dtype=np.float32, copy=True), device=device)
        C = data["C"].bool()
        D = data["D"]
        if E.shape != (*C.shape, 2) or not torch.equal(E.bool().any(-1), C):
            raise ValueError("Effective E and coverage C disagree")
        if bool(torch.isnan(D).any()) or bool((D[torch.isfinite(D)] < 0).any()):
            raise ValueError("Navigation costs must be nonnegative finite values or +Inf")
        if bool(torch.isneginf(D).any()):
            raise ValueError("Negative infinite navigation cost")
        nav_allowed = torch.isfinite(D[1:, 1:])
        distance = torch.where(nav_allowed, D[1:, 1:].float() / 10,
                               torch.zeros_like(D[1:, 1:], dtype=torch.float32))
        nav_edges = torch.stack((distance, torch.log1p(distance)), dim=-1)
        tasks, candidates = self.task_in(data["task"]), self.candidate_in(data["candidate"])
        for block in self.blocks:
            tasks, candidates = block(tasks, candidates, E, C, nav_allowed, nav_edges)
        dynamic_weights, dynamic_values = self.remaining_attention.prepare(
            candidates, tasks, C.T, E.permute(1, 0, 2))
        global_emb = self.global_net(data["global_"])
        g = (global_emb.mean(0) if len(global_emb) else
             candidates.new_zeros(self.hidden))
        encoded = dict(**data, E=E, t=tasks, c=candidates, keys=self.key(candidates), g=g,
                       dynamic_weights=dynamic_weights, dynamic_values=dynamic_values,
                       navigation_edge_mask=nav_allowed, problem_identity=problem.identity,
                       problem_object_identity=id(problem), model_identity=id(self),
                       parameter_versions=tuple(p._version for p in self.parameters()),
                       parameter_fingerprint=self.parameter_fingerprint(),
                       cache_epoch=self._cache_epoch, grad_enabled=torch.is_grad_enabled(),
                       training_mode=self.training)
        return self.fork_cache(encoded)

    def fork_cache(self, encoded):
        """Share static encoding and start an independent prefix (zero forward calls)."""
        result = dict(encoded)
        result["history_prefix"] = ()
        result["history_hidden"] = encoded["c"].new_zeros(self.hidden)
        return result

    def _check_cache(self, env, encoded):
        if (encoded["problem_identity"] != env.problem.identity or
                encoded["problem_object_identity"] != id(env.problem) or
                encoded["model_identity"] != id(self) or
                encoded["parameter_fingerprint"] != self.parameter_fingerprint() or
                encoded["cache_epoch"] != self._cache_epoch or
                encoded["grad_enabled"] != torch.is_grad_enabled() or
                encoded["training_mode"] != self.training):
            raise ValueError("Encoding/prefix cache does not match problem, model, parameters, or mode")

    def score(self, env, encoded):
        if env.done:
            raise ValueError("No action distribution after terminal")
        self._check_cache(env, encoded)
        rf = torch.as_tensor(env.remaining, dtype=encoded["C"].dtype,
                             device=encoded["C"].device)
        gains = rf @ encoded["C"]
        remaining_weights = encoded["dynamic_weights"] * rf[None, None, :]
        denominator = remaining_weights.sum(-1, keepdim=True)
        remaining_weights = remaining_weights / denominator.clamp_min(torch.finfo(rf.dtype).tiny)
        cover = self.remaining_attention.aggregate(remaining_weights, encoded["dynamic_values"])
        cover = self.remaining_norm(cover)
        remaining = (encoded["t"] * rf[:, None]).sum(0) / rf.sum().clamp_min(1)
        current = (encoded["c"][env.current - 1] if env.current
                   else encoded["c"].new_zeros(self.hidden))
        old = encoded["history_prefix"]
        if tuple(env.prefix[:len(old)]) == old:
            h, start = encoded["history_hidden"], len(old)
        else:
            h, start = encoded["c"].new_zeros(self.hidden), 0
        for k in range(start, len(env.prefix)):
            node = env.prefix[k]
            previous = env.prefix[k - 1] if k else 0
            extra = encoded["c"].new_tensor([env.problem.D[previous, node] / 10,
                                              (k + 1) / max(len(rf), 1)])
            h = self.prefix(torch.cat((encoded["c"][node - 1], extra)), h)
        encoded["history_prefix"] = tuple(env.prefix)
        encoded["history_hidden"] = h
        scalars = encoded["c"].new_tensor([float(env.remaining.mean()),
                                           len(env.prefix) / max(len(rf), 1),
                                           env.raw_L / 100, env.problem.D[env.current, 0] / 10])
        q = torch.tanh(self.query(torch.cat((encoded["g"], remaining, current, h, scalars))))
        nav = torch.stack((encoded["D"][env.current, 1:].float() / 10,
                           encoded["D"][1:, 0].float() / 10,
                           gains / max(len(rf), 1), torch.full_like(gains, .075)), dim=1)
        nav = torch.where(torch.isfinite(nav), nav, torch.zeros_like(nav))
        logits = (torch.tanh(encoded["keys"] + self.dynamic(cover) + self.nav(nav)) * q).sum(1)
        logits = 10 * torch.tanh(logits / math.sqrt(self.hidden))
        mask = torch.as_tensor(env.mask(), dtype=torch.bool, device=logits.device)
        if not bool(mask.any()):
            raise ValueError("Nonterminal state has no legal action")
        return logits.masked_fill(~mask, -torch.inf)


__all__ = ["MODEL_ID", "RelationActor", "tensor_hash"]
