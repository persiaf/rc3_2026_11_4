"""Candidate per-state update guard; no training or planning is performed here.

These limits are UNAPPROVED CANDIDATE values, not an active scientific protocol.
The guard constrains only the supplied states. It does not certify unseen states,
unchanged greedy routes, or monotonic planning cost.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from typing import Any

import torch


@dataclass(frozen=True)
class GuardLimits:
    """Unapproved candidate limits; equality at a bound is accepted."""

    max_step_kl: float = 0.001
    max_step_probability_change: float = 0.02
    max_base_total_variation: float = 0.10

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f'{name} must be a finite nonnegative number')
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be a finite nonnegative number')
        if self.max_step_probability_change > 1 or self.max_base_total_variation > 1:
            raise ValueError('Probability-change and total-variation limits cannot exceed one')


def _checked_distributions(base_logits, old_logits, new_logits, mask):
    values = (base_logits, old_logits, new_logits)
    if any(not isinstance(value, torch.Tensor) for value in values):
        raise ValueError('Every logits argument must be a floating tensor')
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
        raise ValueError('mask must be a boolean tensor')
    if not all(value.is_floating_point() for value in values):
        raise ValueError('Every logits argument must be a floating tensor')
    shape, device = base_logits.shape, base_logits.device
    if base_logits.ndim < 1 or base_logits.numel() == 0 or shape[-1] == 0:
        raise ValueError('A nonempty final candidate dimension and nonempty states are required')
    if any(value.shape != shape for value in values) or mask.shape != shape:
        raise ValueError('base, old, new and the shared mask must have identical shapes')
    if any(value.device != device for value in values) or mask.device != device:
        raise ValueError('All logits and the shared mask must be on the same device')
    legal = mask.detach().reshape(-1, shape[-1])
    if not bool(legal.any(dim=-1).all()):
        raise ValueError('Every checked state must have at least one legal action')
    result = []
    for label, value in zip(('base', 'old', 'new'), values):
        value = value.detach().to(dtype=torch.float64).reshape(-1, shape[-1])
        if not bool(torch.isfinite(value[legal]).all()):
            raise ValueError(f'{label} legal-action logits must all be finite')
        # Nonfinite values on illegal actions are discarded before softmax.
        logp = torch.log_softmax(value.masked_fill(~legal, -torch.inf), dim=-1)
        if not bool(torch.isfinite(logp[legal]).all()):
            raise ValueError(f'{label} legal-action log-probabilities overflowed')
        result.append((logp.masked_fill(~legal, 0.0), logp.exp()))
    return legal, result


def check_update(base_logits, old_logits, new_logits, mask, limits=None):
    """Return a JSON-safe decision for every supplied state using one exact mask.

    The last dimension indexes candidates; all preceding dimensions are flattened
    into states. Malformed inputs raise ValueError, so callers must fail closed.
    Legal logits must be finite; illegal-action values do not affect any metric.
    No input tensor or model is modified and no optimizer is called.
    """
    if limits is None:
        limits = GuardLimits()
    if not isinstance(limits, GuardLimits):
        raise ValueError('limits must be a GuardLimits instance')
    with torch.no_grad():
        legal, distributions = _checked_distributions(base_logits, old_logits, new_logits, mask)
        (_, base_p), (old_logp, old_p), (new_logp, new_p) = distributions
        kl = (old_p * (old_logp - new_logp)).sum(dim=-1)
        # Roundoff may make an otherwise zero KL slightly negative.
        kl = kl.clamp_min(0.0)
        probability_change = (new_p - old_p).abs().amax(dim=-1)
        base_tv = 0.5 * (new_p - base_p).abs().sum(dim=-1)
        metrics = (kl, probability_change, base_tv)
        if not all(bool(torch.isfinite(value).all()) for value in metrics):
            raise ValueError('Guard metrics must be finite')
        checks = (
            ('max_step_kl', kl, limits.max_step_kl),
            ('max_step_probability_change', probability_change, limits.max_step_probability_change),
            ('max_base_total_variation', base_tv, limits.max_base_total_variation),
        )
        maxima, rejected, reasons = {}, {}, []
        for name, value, bound in checks:
            maxima[name] = float(value.max().cpu())
            indices = (value > bound).nonzero(as_tuple=False).flatten().cpu().tolist()
            rejected[name] = indices
            if indices:
                reasons.append(f'{name} exceeded in {len(indices)} checked state(s)')
        old_greedy = old_p.argmax(dim=-1)
        new_greedy = new_p.argmax(dim=-1)
        base_greedy = base_p.argmax(dim=-1)
        return dict(
            accepted=not reasons,
            limits_status='UNAPPROVED_CANDIDATE',
            limits={name: float(value) for name, value in asdict(limits).items()},
            state_count=int(legal.shape[0]),
            legal_candidate_count=int(legal.sum().cpu()),
            maxima=maxima,
            rejected_state_indices=rejected,
            greedy_changes_old_to_new=int((old_greedy != new_greedy).sum().cpu()),
            greedy_changes_base_to_new=int((base_greedy != new_greedy).sum().cpu()),
            rejection_reasons=reasons,
            scope='Supplied states only; no unseen-state, greedy-route, or cost-improvement guarantee',
            optimizer_steps_executed=0,
        )


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(_state_equal(value, right[key]) for key, value in left.items()))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(_state_equal(a, b) for a, b in zip(left, right)))
    return left == right


def restore_if_rejected(head, optimizer, before_head, before_opt, report):
    """Restore only the caller-supplied Actor head and optimizer after rejection.

    Snapshots are deep-copied, never modified, and verified after restoration.
    An accepted report does nothing. This helper never executes optimizer.step.
    """
    if not isinstance(report, dict) or type(report.get('accepted')) is not bool:
        raise ValueError('A report with a boolean accepted field is required')
    if report['accepted']:
        return dict(restored=False, optimizer_steps_executed=0)
    saved_head, saved_opt = copy.deepcopy(before_head), copy.deepcopy(before_opt)
    current_head, current_opt = copy.deepcopy(head.state_dict()), copy.deepcopy(optimizer.state_dict())
    try:
        head.load_state_dict(copy.deepcopy(saved_head), strict=True)
        optimizer.load_state_dict(copy.deepcopy(saved_opt))
        if not (_state_equal(head.state_dict(), saved_head)
                and _state_equal(optimizer.state_dict(), saved_opt)):
            raise RuntimeError('Restored Actor head or optimizer differs from its snapshot')
    except Exception:
        head.load_state_dict(current_head, strict=True)
        optimizer.load_state_dict(current_opt)
        raise
    if not (_state_equal(before_head, saved_head) and _state_equal(before_opt, saved_opt)):
        raise RuntimeError('Caller snapshots were modified')
    return dict(restored=True, actor_head_restored=True, optimizer_restored=True,
                optimizer_steps_executed=0)
