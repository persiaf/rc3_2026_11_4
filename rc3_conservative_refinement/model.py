"""Small residual correction and training-only remaining-cost accounting critic.

No planner, dataset, historical checkpoint or optimizer is imported or invoked.
This candidate is deliberately not wired into the closed Integrated-RPPO runner.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def freeze_supervised_base(base: nn.Module) -> nn.Module:
    """Freeze the existing model without replacing or reinitializing its tensors."""
    base.eval().requires_grad_(False)
    for parameter in base.parameters():
        parameter.grad = None
    return base


class SmallCorrection(nn.Module):
    """Zero output initially; the positive amplitude leaves a usable gradient.

    0.25 is a candidate engineering bound, not an approved scientific setting.
    A bound on logits alone does not bound trajectory cost or prevent a near-tie
    argmax flip. Every accepted future update also needs check_update().
    """
    def __init__(self, input_dim=202, hidden=64, max_adjustment=0.25):
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0 or type(hidden) is not int or hidden <= 0:
            raise ValueError('Feature dimensions must be positive integers')
        if isinstance(max_adjustment, bool) or not math.isfinite(max_adjustment) or max_adjustment <= 0:
            raise ValueError('Use a finite positive amplitude; a zero multiplier blocks learning')
        self.input_dim = input_dim
        self.max_adjustment = float(max_adjustment)
        self.head = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, base_logits, features, mask):
        device = next(self.head.parameters()).device
        if base_logits.device != device or features.device != device or mask.device != device:
            raise ValueError('Place logits, detached features, mask and correction on the same device')
        if mask.dtype != torch.bool or base_logits.shape != mask.shape:
            raise ValueError('Base logits and bool mask must have identical shapes')
        if features.shape != (*mask.shape, self.input_dim):
            raise ValueError('Candidate features do not match logits/feature dimension')
        if not bool(mask.any(-1).all()) or not bool(torch.isfinite(base_logits[mask]).all()):
            raise ValueError('Each state needs finite legal logits and at least one legal action')
        if not features.is_floating_point() or not bool(torch.isfinite(features[mask]).all()):
            raise ValueError('Legal candidate features must be finite floating point')
        # Ignore illegal feature payloads before the MLP, including NaN sentinels.
        safe = features.detach().masked_fill(~mask[..., None], 0.)
        delta = self.max_adjustment * torch.tanh(self.head(safe).squeeze(-1))
        return (base_logits.detach().masked_fill(~mask, 0.) + delta).masked_fill(~mask, -torch.inf)

    def actor_only_package(self, base_identity):
        if not isinstance(base_identity, dict) or not base_identity.get('path') or not base_identity.get('sha256'):
            raise ValueError('An exact frozen base path and SHA256 are required')
        return dict(schema='rc3-conservative-correction-candidate-v1',
                    status='UNTRAINED_ENGINEERING_CANDIDATE_NOT_OLD_ACTOR_SCHEMA',
                    base=dict(base_identity), input_dim=self.input_dim,
                    hidden=self.head[0].out_features, max_adjustment=self.max_adjustment,
                    correction_state={k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
                    critic_included=False, optimizer_included=False)


class RemainingCostCritic(nn.Module):
    """Predict RAW remaining length/stops and separate signed terminal H credit.

    H can alter the already-built prefix. Its credit is accounting, not a
    physical distance remaining from the current position. Only the combined
    negative cost / fixed initial task count is comparable to the old G_t.
    No actor parameters or actor instance are stored by this module.
    """
    def __init__(self, input_dim=268, hidden=128):
        super().__init__()
        if type(input_dim) is not int or input_dim <= 0 or type(hidden) is not int or hidden <= 0:
            raise ValueError('Feature dimensions must be positive integers')
        self.input_dim = input_dim
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, features, task_count, penalty=0.75):
        if features.shape[-1] != self.input_dim or not bool(torch.isfinite(features).all()):
            raise ValueError('Finite critic features of the declared dimension are required')
        if type(task_count) is not int or task_count < 0:
            raise ValueError('Use the fixed initial task count, never a changing remaining count')
        if isinstance(penalty, bool) or not math.isfinite(penalty) or penalty < 0:
            raise ValueError('Invalid stop penalty')
        output = self.net(features.detach())
        length = torch.nn.functional.softplus(output[..., 0])
        stops = torch.nn.functional.softplus(output[..., 1])
        h_credit = output[..., 2]
        value = -(length + penalty * stops - h_credit) / max(1, task_count)
        return dict(raw_remaining_L=length, raw_remaining_K=stops,
                    terminal_H_credit=h_credit, value=value)


def remaining_cost_loss(prediction, targets, task_count):
    """Detached complete-episode targets; a candidate loss, never executed here.

    Every component is expressed per fixed initial task count. This auxiliary
    decomposition changes the critic fitting objective and needs its own study.
    """
    if type(task_count) is not int or task_count < 0:
        raise ValueError('Invalid fixed task count')
    scale = max(1, task_count)
    if 'scale' in targets and targets['scale'] != scale:
        raise ValueError('Trajectory targets use another initial task count')
    terms = []
    for key in ('raw_remaining_L', 'raw_remaining_K', 'terminal_H_credit'):
        source = targets.get(key)
        if key == 'terminal_H_credit' and source is None and 'H_gain' in targets:
            source = [targets['H_gain']] * len(targets['returns'])
        expected = torch.as_tensor(source, device=prediction[key].device,
                                   dtype=prediction[key].dtype).detach()
        if expected.shape != prediction[key].shape or not bool(torch.isfinite(expected).all()):
            raise ValueError('Critic target shape or values are invalid')
        terms.append(((prediction[key] - expected) / scale).square())
    return 0.5 * torch.stack(terms).sum(0)


def full_return_advantage(returns, old_value):
    if returns.shape != old_value.shape or not bool(torch.isfinite(returns).all()) or not bool(torch.isfinite(old_value).all()):
        raise ValueError('Finite matching complete reward-to-go and old values are required')
    return (returns.detach() - old_value.detach()).detach()
