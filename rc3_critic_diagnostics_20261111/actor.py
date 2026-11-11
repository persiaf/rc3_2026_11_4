"""Saved-state Actor ranking diagnostics; no data, checkpoint, or optimizer I/O.

The caller admits the exact trajectories and checkpoint pairs before calling.
Every actual head forward is charged by a mandatory callback.  No planner,
environment, frozen encoder, H implementation, or optimization path is imported.
"""
from __future__ import annotations

import math
import time
from typing import Mapping

import numpy as np
import torch


AMPLITUDE = 0.25
PAIRWISE_BOUND = 2 * AMPLITUDE


def off_features(features):
    """Ablate only the admitted mean/max columns; preserve count/unreachable."""
    if features.ndim != 3 or features.shape[-1] != 206:
        raise ValueError('Expected saved T x M x 206 features')
    result = features.detach().clone()
    result[..., 202:204] = 0
    return result


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


def _distribution(values):
    shifted = values - values.max()
    logp = shifted - np.log(np.exp(shifted).sum())
    return logp, np.exp(logp)


def _ranks(values):
    # Stable ties reproduce torch.argmax's first-index convention.  Ranks are
    # ordinal ranks with index tie breaking, not averaged statistical ranks.
    order = np.argsort(-values, kind='stable')
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(1, len(values) + 1)
    return order, ranks


def _delta_stats(delta, prefix):
    mean = float(delta.mean())
    centered = delta - mean
    square = float(np.mean(delta * delta))
    centered_square = float(np.mean(centered * centered))
    common_square = mean * mean
    return {
        prefix + 'min': float(delta.min()), prefix + 'max': float(delta.max()),
        prefix + 'span': float(np.ptp(delta)), prefix + 'mean': mean,
        prefix + 'rms': math.sqrt(square),
        prefix + 'centered_rms': math.sqrt(centered_square),
        prefix + 'mean_square': square,
        prefix + 'common_mean_square': common_square,
        prefix + 'centered_mean_square': centered_square,
        prefix + 'energy_closure_abs_error': abs(square-common_square-centered_square),
        prefix + 'centered_max_abs': float(np.max(np.abs(centered))),
        prefix + 'centered_min': float(centered.min()),
        prefix + 'centered_max': float(centered.max()),
        prefix + 'common_energy_fraction': min(1., max(0., mean * mean / square))
        if square else None,
        prefix + 'zero_total_energy': square == 0.,
    }


def _validate_batch(episode):
    batch = episode['batch']
    z0, mask = batch['z0'].detach(), batch['mask'].detach()
    if z0.ndim != 2 or mask.shape != z0.shape or mask.dtype != torch.bool:
        raise ValueError('Saved logits and legal mask must be matching T x M')
    if not bool(mask.any(-1).all()) or not bool(torch.isfinite(z0[mask]).all()):
        raise ValueError('Every saved state requires finite legal base logits')
    actions = episode.get('actions')
    actions = None if actions is None else np.asarray(_numpy(actions), dtype=np.int64)
    if actions is not None:
        if actions.shape != (len(mask),):
            raise ValueError('Recorded actual actions must have one entry per state')
        for t, action in enumerate(actions):
            if not 0 <= action < mask.shape[1] or not bool(mask[t, action]):
                raise ValueError('Recorded actual action is not a legal local candidate')
    old = episode.get('old_logits')
    if old is not None:
        old = _numpy(old)
        if old.shape != tuple(z0.shape) or not np.isfinite(old[_numpy(mask).astype(bool)]).all():
            raise ValueError('Saved collection logits must match all legal states')
    return batch, z0, mask, actions, old


def diagnose_trajectory(episode, *, identity: Mapping, variants: Mapping,
                        source_before: str, source_after: str, charge,
                        contrasts=()):
    """Return (state_rows, candidate_iterator) for one pre-admitted trajectory.

    ``variants`` maps stable labels to ``{'correction': module,
    'features': T x M x 206 tensor}``.  ``features=None`` explicitly records a
    missing feature view; it does not invent ON distances for an OFF-only cache.
    The caller supplies before/after modules from the exact update's snapshots,
    and can additionally supply the ON/OFF weights/features response matrix.

    ``charge(kind, amount=1, **context)`` is called BEFORE each actual head forward.
    Reusing the
    identical module and tensor objects shares a forward and records the alias.
    ``contrasts`` contains (name, left_variant, right_variant) tuples.  These are
    same-state comparisons, never comparisons between different online routes.

    All legal candidate values/ranks are retained in the second return iterator;
    consume it before discarding the trajectory.  Illegal candidates are excluded.
    True tanh residuals and float32-effective logit differences are both saved.
    The ratio columns use the recorded actual action and collection likelihood;
    absent collection logits are reported as missing, never replaced by logits
    from the reconstructed current policy.
    """
    if not callable(charge):
        raise ValueError('A pre-forward resource charge callback is mandatory')
    if source_before not in variants or source_after not in variants:
        raise ValueError('Explicit before/after variant labels are required')
    batch, z0, mask, actions, old = _validate_batch(episode)
    base, legal = _numpy(z0), _numpy(mask).astype(bool)
    prepared, cache = {}, {}
    for label, spec in variants.items():
        correction, features = spec.get('correction'), spec.get('features')
        if correction is None or features is None:
            prepared[label] = None
            continue
        if (correction.input_dim != 206 or correction.max_adjustment != AMPLITUDE
                or tuple(features.shape) != (*z0.shape, 206)):
            raise ValueError('Exact 206-dimension, +/-0.25 correction required')
        if correction.training or any(p.requires_grad for p in correction.parameters()):
            raise ValueError('Diagnostic correction must be eval() and frozen')
        if not bool(torch.isfinite(features[mask]).all()):
            raise ValueError('Saved legal candidate features must be finite')
        key = id(correction), id(features)
        if key not in cache:
            charge('actor_head_forwards', 1, variant=label,
                   states=int(z0.shape[0]), legal_candidate_states=int(mask.sum()),
                   candidate_slots=int(mask.numel()))
            tick = time.perf_counter()
            try:
                with torch.no_grad():
                    # This is the exact uncombined residual from SmallCorrection.
                    # Computing only (actor_logits - z0) would lose sub-ULP changes.
                    safe = features.detach().masked_fill(~mask[..., None], 0.)
                    delta = AMPLITUDE * torch.tanh(correction.head(safe).squeeze(-1))
                    logits = (z0.masked_fill(~mask, 0.) + delta).masked_fill(~mask, -torch.inf)
            finally:
                charge('timing', 0, category='actor_head_forward', seconds=time.perf_counter()-tick)
            if not bool(torch.isfinite(logits[mask]).all()):
                raise ValueError('Nonfinite diagnostic Actor output')
            if bool((delta[mask].abs() > AMPLITUDE).any()):
                raise ValueError('Residual exceeded the frozen structure bound')
            cache[key] = (_numpy(logits), _numpy(delta))
        prepared[label] = cache[key]
    candidates = episode.get('candidate_ids')
    if candidates is not None and len(candidates) != z0.shape[1]:
        raise ValueError('Candidate global IDs must match the saved local order')
    states, rows_by_state = [], []
    for t in range(len(z0)):
        ids = np.flatnonzero(legal[t])
        b = base[t, ids]
        base_order, base_ranks = _ranks(b)
        base_lp, base_p = _distribution(b)
        top = int(base_order[0])
        gap = float(b[top] - b[base_order[1]]) if len(ids) > 1 else None
        action_pos = None if actions is None else int(np.flatnonzero(ids == actions[t])[0])
        old_lp, old_p = (None, None) if old is None else _distribution(old[t, ids])
        row = dict(identity, state_index=t, legal_count=len(ids),
                   single_legal_action=len(ids) == 1,
                   base_top1_local_index=int(ids[top]),
                   base_top2_local_index=int(ids[base_order[1]]) if len(ids) > 1 else None,
                   base_top1_logit=float(b[top]),
                   base_top2_logit=float(b[base_order[1]]) if len(ids) > 1 else None,
                   base_top1_top2_gap=gap,
                   gap_gt_0_5=None if gap is None else gap > PAIRWISE_BOUND,
                   gap_eq_0_5=None if gap is None else gap == PAIRWISE_BOUND,
                   structurally_cannot_change_unique_top1=None if gap is None else gap > PAIRWISE_BOUND,
                   at_bound_can_at_most_tie=None if gap is None else gap == PAIRWISE_BOUND,
                   max_pairwise_adjustment=PAIRWISE_BOUND,
                   actual_action_local_index=None if actions is None else int(actions[t]),
                   actual_action_base_rank=None if action_pos is None else int(base_ranks[action_pos]),
                   collection_likelihood_status='missing_logits' if old is None else
                       ('missing_action' if actions is None else 'present'),
                   ratio_source=episode.get('ratio_source', 'saved_collection_logits'),
                   collection_actual_logp=None if old_lp is None or action_pos is None else float(old_lp[action_pos]))
        saved_audit = episode.get('record', {}).get('post_service_audit', [])
        saved_probability = None if t >= len(saved_audit) else saved_audit[t].get('selected_probability')
        saved_logit = None if t >= len(saved_audit) else saved_audit[t].get('selected_actor_logit')
        reconstructed_probability = None if old_p is None or action_pos is None else float(old_p[action_pos])
        row['saved_selected_probability'] = saved_probability
        row['reconstructed_collection_selected_probability'] = reconstructed_probability
        row['saved_collection_probability_abs_difference'] = None if saved_probability is None or reconstructed_probability is None else abs(saved_probability-reconstructed_probability)
        row['saved_selected_actor_logit'] = saved_logit
        row['saved_collection_logit_abs_difference'] = None if saved_logit is None or old is None or actions is None else abs(saved_logit-float(old[t, actions[t]]))
        variant_states = {}
        for label, result in prepared.items():
            prefix = label + '__'
            row[prefix + 'status'] = 'missing_features_or_weights' if result is None else 'present'
            row[prefix + 'is_actual_update_source'] = label in (source_before, source_after)
            if result is None:
                continue
            z, delta = result[0][t, ids], result[1][t, ids]
            order, ranks = _ranks(z)
            lp, p = _distribution(z)
            selected_lp = None if action_pos is None else float(lp[action_pos])
            row.update(_delta_stats(delta, prefix + 'residual_'))
            row.update(_delta_stats(z - b, prefix + 'effective_logit_delta_'))
            row.update({prefix + 'top1_local_index': int(ids[order[0]]),
                        prefix + 'top1_changed_from_base': bool(order[0] != top),
                        prefix + 'rank_changed_candidates': int(np.count_nonzero(ranks != base_ranks)),
                        prefix + 'entire_rank_unchanged_from_base': bool(np.array_equal(order, base_order)),
                        prefix + 'probability_max_change_from_base': float(np.max(np.abs(p - base_p))),
                        prefix + 'total_variation_from_base': float(.5 * np.abs(p - base_p).sum()),
                        prefix + 'actual_action_logp': selected_lp,
                        prefix + 'actual_action_ratio_to_collection':
                            None if selected_lp is None or old_lp is None else
                            math.exp(selected_lp - float(old_lp[action_pos]))})
            variant_states[label] = dict(z=z, delta=delta, lp=lp, p=p, ranks=ranks,
                                         order=order, centered=delta-delta.mean())
        for output_name, label in [('pre_update_ratio', source_before), ('post_update_ratio', source_after)]:
            row[output_name] = row.get(label + '__actual_action_ratio_to_collection')
        before, after = variant_states.get(source_before), variant_states.get(source_after)
        row['pre_to_post_top1_changed'] = None if before is None or after is None else bool(before['order'][0] != after['order'][0])
        row['pre_to_post_rank_changed_candidates'] = None if before is None or after is None else int(np.count_nonzero(before['ranks'] != after['ranks']))
        row['pre_to_post_max_probability_difference'] = None if before is None or after is None else float(np.max(np.abs(before['p']-after['p'])))
        row['pre_to_post_total_variation'] = None if before is None or after is None else float(.5*np.abs(before['p']-after['p']).sum())
        for name, left, right in contrasts:
            a, c = variant_states.get(left), variant_states.get(right)
            prefix = name + '__'
            row[prefix + 'status'] = 'present' if a is not None and c is not None else 'missing_variant'
            if a is None or c is None:
                continue
            d = a['z'] - c['z']
            row.update({prefix + 'top1_differs': bool(a['order'][0] != c['order'][0]),
                        prefix + 'rank_changed_candidates': int(np.count_nonzero(a['ranks'] != c['ranks'])),
                        prefix + 'total_variation': float(.5*np.abs(a['p']-c['p']).sum()),
                        prefix + 'max_probability_difference': float(np.max(np.abs(a['p']-c['p']))),
                        prefix + 'centered_logit_difference_rms': float(np.sqrt(np.mean((d-d.mean())**2)))})
        states.append(row)
        rows_by_state.append((ids, b, base_ranks, base_p, action_pos, old_p, variant_states))

    def candidate_iterator():
        for t, (ids, b, ranks, probabilities, action_pos, old_p, variants_for_state) in enumerate(rows_by_state):
            for j, local_id in enumerate(ids):
                row = dict(identity, state_index=t, candidate_local_index=int(local_id),
                           candidate_global_id=None if candidates is None else int(candidates[local_id]),
                           is_recorded_actual_action=action_pos == j,
                           base_logit=float(b[j]), base_rank=int(ranks[j]),
                           base_probability=float(probabilities[j]),
                           collection_logit=None if old is None else float(old[t, local_id]),
                           collection_probability=None if old_p is None else float(old_p[j]))
                for label in prepared:
                    result = variants_for_state.get(label)
                    prefix = label + '__'
                    for key in ('logit', 'residual', 'centered_residual', 'probability', 'rank'):
                        row[prefix + key] = None
                    if result is not None:
                        row.update({prefix + 'logit': float(result['z'][j]),
                                    prefix + 'residual': float(result['delta'][j]),
                                    prefix + 'centered_residual': float(result['centered'][j]),
                                    prefix + 'probability': float(result['p'][j]),
                                    prefix + 'rank': int(result['ranks'][j])})
                yield row
    return states, candidate_iterator()


def tidy_state_rows(state_rows, variant_labels):
    """Yield the equivalent state x variant table without new model operations.

    Common base margins, actual pre/post ratios, and same-state contrasts are
    retained on every variant row; only each variant's diagnostic keys are
    unprefixed.  Missing variants remain explicit rows with a missing status.
    """
    prefixes = tuple(label + '__' for label in variant_labels)
    for row in state_rows:
        common = {key: value for key, value in row.items()
                  if not key.startswith(prefixes)}
        for label in variant_labels:
            prefix = label + '__'
            result = dict(common, variant=label)
            result.update({key[len(prefix):]: value for key, value in row.items()
                           if key.startswith(prefix)})
            yield result
