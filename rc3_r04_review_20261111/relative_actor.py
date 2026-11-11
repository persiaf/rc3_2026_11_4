"""R04 relative correction interface candidate; no planner or optimizer entry.

The correction is exactly .25*tanh(f - legal_mean(f)).  Pre-tanh additive
common shifts cancel.  The post-tanh correction need not have zero legal mean:
tanh is nonlinear.  Subtracting that latter mean is intentionally NOT done,
because it can make an individual correction exceed the fixed .25 bound.

Only the new head is implemented here.  A future runtime must separately load
and verify the exact frozen SL encoder, preserve its candidate order and mask,
and guard every proposed Actor update.  No old checkpoint schema is reused.
"""
from __future__ import annotations

import hashlib
import math
from numbers import Real
import re
from typing import Mapping

import torch
from torch import nn


SCHEMA = 'relative_pre_tanh_centered_v1'
INPUT_DIM = 206
HIDDEN_DIM = 64
AMPLITUDE = .25


def _validate_masked_values(values, mask):
    if (not isinstance(values, torch.Tensor) or not values.is_floating_point()
            or values.ndim < 1 or values.numel() == 0 or values.shape[-1] == 0):
        raise ValueError('Nonempty floating candidate scores are required')
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
            or mask.shape != values.shape or mask.device != values.device):
        raise ValueError('Scores and boolean legal mask must have identical shape/device')
    if not bool(mask.any(-1).all()):
        raise ValueError('Every state requires at least one legal candidate')
    if not bool(torch.isfinite(values[mask]).all()):
        raise ValueError('All legal candidate scores must be finite')


def residual_details(raw_scores, mask):
    """Return the exact legal-mask centering and bounded correction.

    Float16/32 means are accumulated in float64 and the bounded output is cast
    back to the input dtype.  Illegal payloads, including NaN/Inf, are discarded
    before reduction.  This function does not detach the head scores: head-only
    gradients retain the legal-mean Jacobian.  Masked returned entries are zero.
    """
    _validate_masked_values(raw_scores, mask)
    dtype = raw_scores.dtype
    work = raw_scores.masked_fill(~mask, 0.).to(torch.float64)
    count = mask.sum(-1, keepdim=True).to(torch.float64)
    mean = work.sum(-1, keepdim=True) / count
    centered = (work - mean).masked_fill(~mask, 0.)
    if not bool(torch.isfinite(centered[mask]).all()):
        raise ValueError('Legal centering overflowed; refuse the candidate')
    delta = (AMPLITUDE * torch.tanh(centered)).to(dtype).masked_fill(~mask, 0.)
    post_mean = delta.to(torch.float64).sum(-1, keepdim=True) / count
    return dict(raw_scores=raw_scores, legal_raw_mean=mean,
                centered_scores=centered, delta=delta,
                legal_post_tanh_mean=post_mean, legal_count=count.to(torch.long))


class RelativeCorrection(nn.Module):
    """Same 206-64-64-1 size as R03; new legal-relative output semantics.

    The last layer is exactly zero initially.  No SL module is owned here, so
    calling train() on the correction cannot change a caller's frozen encoder.
    The caller supplies detached encoder features and exact base logits.
    """
    def __init__(self, input_dim=INPUT_DIM, hidden=HIDDEN_DIM, max_adjustment=AMPLITUDE):
        super().__init__()
        if type(input_dim) is not int or input_dim != INPUT_DIM:
            raise ValueError('R04 requires the fixed 206-dimensional feature schema')
        if type(hidden) is not int or hidden != HIDDEN_DIM:
            raise ValueError('R04 preserves the 64/64 hidden-layer dimensions')
        if (isinstance(max_adjustment, bool) or not isinstance(max_adjustment, Real)
                or not math.isfinite(max_adjustment) or max_adjustment != AMPLITUDE):
            raise ValueError('R04 keeps the fixed +/-0.25 correction bound')
        self.input_dim = input_dim
        self.max_adjustment = float(max_adjustment)
        self.correction_schema = SCHEMA
        self.head = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def details(self, base_logits, features, mask):
        _validate_masked_values(base_logits, mask)
        first = next(self.parameters())
        if (not isinstance(features, torch.Tensor) or not features.is_floating_point()
                or features.shape != (*mask.shape, self.input_dim)
                or features.device != first.device or base_logits.device != first.device
                or features.dtype != first.dtype or base_logits.dtype != first.dtype):
            raise ValueError('Base logits/features/head must share dtype/device and T x M x 206 shape')
        if not bool(torch.isfinite(features[mask]).all()):
            raise ValueError('Legal features must be finite')
        safe = features.detach().masked_fill(~mask[..., None], 0.)
        raw = self.head(safe).squeeze(-1)
        out = residual_details(raw, mask)
        out['logits'] = (base_logits.detach().masked_fill(~mask, 0.) + out['delta']).masked_fill(~mask, -torch.inf)
        return out

    def forward(self, base_logits, features, mask):
        return self.details(base_logits, features, mask)['logits']


def _validated_base_identity(identity):
    if (not isinstance(identity, Mapping) or set(identity) != {'path', 'sha256'}
            or not isinstance(identity['path'], str) or not identity['path']
            or not isinstance(identity['sha256'], str)
            or re.fullmatch('[0-9a-f]{64}', identity['sha256']) is None):
        raise ValueError('Explicit frozen base path and lowercase SHA256 are required')
    return dict(identity)


def _state_identity(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError('Deployment head weights must be float32 tensors only')
        if not bool(torch.isfinite(value).all()):
            raise ValueError('Deployment head weights must be finite')
        digest.update(name.encode('utf-8'))
        digest.update(str(tuple(value.shape)).encode('ascii'))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def actor_only_package(correction, base_identity):
    """A strict correction-only mapping: no base tensors, Critic, or Adam.

    This is an engineering deployment interface, not an authorized trained
    policy or a full historical-runtime loader.  It never opens the base path.
    """
    if not isinstance(correction, RelativeCorrection):
        raise ValueError('Only the new relative Actor schema can be exported')
    base_identity = _validated_base_identity(base_identity)
    state = {name: value.detach().cpu().clone() for name, value in correction.state_dict().items()}
    return dict(schema=SCHEMA, status='R04_INTERFACE_CANDIDATE_NOT_AUTHORIZED_RUN',
                input_dim=INPUT_DIM, hidden=HIDDEN_DIM, max_adjustment=AMPLITUDE,
                dtype='float32', base_identity=base_identity,
                correction_state=state, correction_tensor_sha256=_state_identity(state),
                critic_included=False, optimizer_included=False)


def load_actor_only_package(package, *, expected_base_identity):
    """Strict mapping loader; rejects absolute-head/foreign training schemas.

    The caller must hash and deserialize any file before calling this mapping
    interface, and separately verify/load the exact frozen SL artifact.  No
    Critic, optimizer, model path, scene, or H implementation is loaded here.
    """
    allowed = {'schema', 'status', 'input_dim', 'hidden', 'max_adjustment', 'dtype',
               'base_identity', 'correction_state', 'correction_tensor_sha256',
               'critic_included', 'optimizer_included'}
    if not isinstance(package, Mapping) or set(package) != allowed:
        raise ValueError('Unexpected training state or missing relative Actor fields')
    if (package['schema'] != SCHEMA
            or package['status'] != 'R04_INTERFACE_CANDIDATE_NOT_AUTHORIZED_RUN'
            or package['input_dim'] != INPUT_DIM or type(package['input_dim']) is not int
            or package['hidden'] != HIDDEN_DIM or type(package['hidden']) is not int
            or package['max_adjustment'] != AMPLITUDE or isinstance(package['max_adjustment'], bool)
            or package['dtype'] != 'float32'
            or package['critic_included'] is not False or package['optimizer_included'] is not False):
        raise ValueError('Incompatible relative Actor-only schema')
    base = _validated_base_identity(package['base_identity'])
    if base != _validated_base_identity(expected_base_identity):
        raise ValueError('Frozen base identity differs')
    state = package['correction_state']
    if not isinstance(state, Mapping) or _state_identity(state) != package['correction_tensor_sha256']:
        raise ValueError('Correction tensor identity differs')
    model = RelativeCorrection()
    model.load_state_dict(state, strict=True)
    if _state_identity(model.state_dict()) != package['correction_tensor_sha256']:
        raise ValueError('Loaded correction tensors differ')
    model.eval().requires_grad_(False)
    model.base_identity = base
    return model


@torch.no_grad()
def rank_response(base_logits, before_logits, after_logits, mask, *,
                  actions=None, collection_logits=None, candidate_ids=None):
    """Record exact legal ranks and separate pre/post actual-action ratios.

    Ratios require explicit collection logits.  They are not silently inferred
    from a current policy.  Ties use the original candidate-index order.  This
    helper computes no policy, action choice, route, loss, or optimizer update.
    """
    for value in (base_logits, before_logits, after_logits):
        _validate_masked_values(value, mask)
        if value.shape != base_logits.shape or value.device != base_logits.device:
            raise ValueError('Base/pre/post logits must use identical state/candidate layout')
    if collection_logits is not None:
        _validate_masked_values(collection_logits, mask)
    shape = base_logits.shape
    M = shape[-1]
    legal = mask.reshape(-1, M)
    values = [v.detach().to(torch.float64).reshape(-1, M)
              for v in (base_logits, before_logits, after_logits)]
    collection = None if collection_logits is None else collection_logits.detach().to(torch.float64).reshape(-1, M)
    if candidate_ids is not None:
        candidate_ids = list(candidate_ids)
        if (len(candidate_ids) != M or len(set(candidate_ids)) != M
                or any(type(x) is not int for x in candidate_ids)):
            raise ValueError('Ordered unique global candidate IDs must match M')
    if actions is not None:
        actions = torch.as_tensor(actions, device=mask.device)
        if actions.dtype != torch.long or tuple(actions.shape) != tuple(shape[:-1]):
            raise ValueError('Actions must be integer local indices for all state dimensions')
        actions = actions.reshape(-1)
        if not bool(((actions >= 0) & (actions < M)).all()) or not bool(legal.gather(1, actions[:, None]).all()):
            raise ValueError('A recorded actual action is illegal')
    result = []
    for t in range(len(legal)):
        ids = legal[t].nonzero(as_tuple=False).flatten()
        logits = [value[t, ids] for value in values]
        order = [torch.argsort(z, descending=True, stable=True) for z in logits]
        ranks = []
        for seq in order:
            rank = torch.empty_like(seq)
            rank[seq] = torch.arange(1, len(seq)+1, device=seq.device)
            ranks.append(rank)
        lps = [torch.log_softmax(z, -1) for z in logits]
        if any(not bool(torch.isfinite(lp).all()) for lp in lps):
            raise ValueError('Legal ranking log-probabilities overflowed')
        action_pos = None if actions is None else int((ids == actions[t]).nonzero(as_tuple=False)[0])
        old_lp = None if collection is None else torch.log_softmax(collection[t, ids], -1)
        if old_lp is not None and not bool(torch.isfinite(old_lp).all()):
            raise ValueError('Legal collection log-probabilities overflowed')
        gap = None if len(ids) == 1 else float(logits[0][order[0][0]]-logits[0][order[0][1]])
        row = dict(state_index=t, legal_candidate_indices=ids.cpu().tolist(),
                   legal_candidate_ids=None if candidate_ids is None else [candidate_ids[int(j)] for j in ids],
                   legal_count=len(ids), single_legal_action=len(ids)==1,
                   base_top1_top2_gap=gap, strict_bound_blocks_unique_top1=None if gap is None else gap>.5,
                   equality_bound_at_most_tie=None if gap is None else gap==.5,
                   base_top1=int(ids[order[0][0]]), before_top1=int(ids[order[1][0]]),
                   after_top1=int(ids[order[2][0]]),
                   before_ranks=ranks[1].cpu().tolist(), after_ranks=ranks[2].cpu().tolist(),
                   rank_changed_candidates=int((ranks[1]!=ranks[2]).sum()),
                   top1_changed=bool(order[1][0]!=order[2][0]),
                   recorded_actual_action=None if actions is None else int(actions[t]),
                   pre_update_ratio=None, post_update_ratio=None,
                   ratio_status='missing_explicit_collection_logits_or_actions')
        if action_pos is not None and old_lp is not None:
            pre_ratio=float((lps[1][action_pos]-old_lp[action_pos]).exp())
            post_ratio=float((lps[2][action_pos]-old_lp[action_pos]).exp())
            if not math.isfinite(pre_ratio) or not math.isfinite(post_ratio):
                raise ValueError('Actual-action probability ratio overflowed')
            row.update(pre_update_ratio=pre_ratio,
                       post_update_ratio=post_ratio,
                       ratio_status='explicit_collection_logits')
        result.append(row)
    return result
