"""Matched offline 268-input Critic parameterization diagnostic.

This module never reads run files, starts a planner, calls H, or constructs an
optimizer. The caller admits saved episodes, charges every actual forward and
optimizer attempt, and fixes the data/split/schedule before invoking it.

The three variants share exactly the same initial ``net`` tensors. C0 retains
the historical raw component parameterization and component loss. C1 represents
the three components per *initial* task count. C2 adds a unit-weight value loss
to C1. These are offline fits; none can establish a route-quality improvement.
"""
from __future__ import annotations

import copy
from collections import defaultdict
import math

import numpy as np
import torch

from rc3_conservative_refinement.model import RemainingCostCritic, remaining_cost_loss


SEED = 11622026
INPUT_DIM = 268
HIDDEN_DIM = 128
PENALTY = 0.75
# Frozen rc3_integrated_rppo/model.py: Actor.features concatenates four
# 64-vectors [g, remaining, current, history_hidden] and the 12-value context.
# context[0] is float(rf.mean()), so v[:, 256] is the observable remaining-task
# fraction. The caller source-binds that old file; no future trace length is
# used to condition a reference prediction.
REMAINING_FRACTION_INDEX = 4 * 64
assert REMAINING_FRACTION_INDEX == 256 and INPUT_DIM - REMAINING_FRACTION_INDEX == 12
COMPONENTS = (
    'raw_remaining_L', 'raw_remaining_K', 'terminal_H_credit',
    'normalized_L', 'normalized_K', 'normalized_H', 'value',
)
NORMALIZED_COMPONENTS = ('normalized_L', 'normalized_K', 'normalized_H')
VARIANTS = ('C0', 'C1', 'C2')


def episode_features(episode):
    """Return detached saved features; never reconstruct a state or a label."""
    source = episode['v'] if 'v' in episode else episode['batch']['v']
    features = torch.as_tensor(source, dtype=torch.float32).detach()
    if features.ndim != 2 or features.shape[1] != INPUT_DIM or not len(features):
        raise ValueError('A nonempty saved T x 268 state matrix is required')
    if not bool(torch.isfinite(features).all()):
        raise ValueError('Saved Critic features must be finite')
    remaining = features[:, REMAINING_FRACTION_INDEX]
    if not bool(((remaining >= -1e-6) & (remaining <= 1 + 1e-6)).all()):
        raise ValueError('Frozen context[0] / v[:,256] remaining fraction is outside [0,1]')
    return features


def episode_labels(episode, *, device=None, dtype=torch.float32):
    """Read complete saved L/K/H/RTG labels without producing H labels."""
    count = episode['task_count']
    if type(count) is not int or count <= 0:
        raise ValueError('A positive fixed initial task count is required')
    targets = episode['targets']
    if 'scale' in targets and targets['scale'] != count:
        raise ValueError('Saved label scale differs from the initial task count')
    returns = torch.as_tensor(targets['returns'], device=device, dtype=dtype).detach()
    if returns.ndim != 1 or not len(returns):
        raise ValueError('Saved RTG must contain one complete label per state')
    out = {'value': returns}
    for key in COMPONENTS[:3]:
        source = targets.get(key)
        if key == 'terminal_H_credit' and source is None:
            if 'H_gain' not in targets:
                raise ValueError('Saved signed H credit is missing; no new H call is allowed')
            source = [targets['H_gain']] * len(returns)
        value = torch.as_tensor(source, device=device, dtype=dtype).detach()
        if value.shape != returns.shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f'Saved {key} shape/finiteness differs from RTG')
        out[key] = value
    if not bool(torch.isfinite(returns).all()):
        raise ValueError('Saved RTG must be finite')
    if not bool((out['raw_remaining_L'] >= 0).all()) or not bool((out['raw_remaining_K'] >= 0).all()):
        raise ValueError('Saved raw L/K targets must be nonnegative')
    out['normalized_L'] = out['raw_remaining_L'] / count
    out['normalized_K'] = out['raw_remaining_K'] / count
    out['normalized_H'] = out['terminal_H_credit'] / count
    composed = -(out['normalized_L'] + PENALTY * out['normalized_K'] - out['normalized_H'])
    if not torch.allclose(composed, returns, atol=2e-5, rtol=2e-5):
        raise ValueError('Saved full RTG does not match saved L/K/H accounting')
    return out


class MatchedCritic(RemainingCostCritic):
    """The old 268-128-128-3 MLP with one fixed output interpretation."""
    def __init__(self, variant):
        if variant not in VARIANTS:
            raise ValueError('Only preregistered C0/C1/C2 variants are allowed')
        super().__init__(input_dim=INPUT_DIM, hidden=HIDDEN_DIM)
        self.variant = variant

    def forward(self, features, task_count, penalty=PENALTY):
        if penalty != PENALTY:
            raise ValueError('The saved 0.75 stop penalty is frozen')
        if type(task_count) is not int or task_count <= 0:
            raise ValueError('Use the fixed positive initial task count')
        if self.variant == 'C0':
            prediction = super().forward(features, task_count, penalty)
            prediction['normalized_L'] = prediction['raw_remaining_L'] / task_count
            prediction['normalized_K'] = prediction['raw_remaining_K'] / task_count
            prediction['normalized_H'] = prediction['terminal_H_credit'] / task_count
            return prediction
        if features.ndim != 2 or features.shape[-1] != INPUT_DIM or not bool(torch.isfinite(features).all()):
            raise ValueError('Finite saved T x 268 features are required')
        output = self.net(features.detach())
        length = torch.nn.functional.softplus(output[..., 0])
        stops = torch.nn.functional.softplus(output[..., 1])
        credit = output[..., 2]
        value = -(length + PENALTY * stops - credit)
        return dict(raw_remaining_L=length * task_count,
                    raw_remaining_K=stops * task_count,
                    terminal_H_credit=credit * task_count,
                    normalized_L=length, normalized_K=stops,
                    normalized_H=credit, value=value)


def make_matched_models(seed=SEED):
    """Identical default initial tensors; no target-informed bias adjustment.

    Caller RNG is preserved. Each configuration receives its own independent
    parameter tensors. No old trained Critic or Actor is mutated or referenced.
    """
    if type(seed) is not int:
        raise ValueError('An exact integer initialization seed is required')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        initial = MatchedCritic('C0')
    result = {}
    for variant in VARIANTS:
        model = copy.deepcopy(initial)
        model.variant = variant
        result[variant] = model
    return result


def matched_loss(model, episodes):
    """State mean within trajectory, then equal mean across batch trajectories.

    The caller fixes parent-balanced trajectory batches once and reuses them
    across C0/C1/C2. This reduction is explicit and differs from the historical
    PPO trainer's per-trajectory *sum*; it is fixed for all offline variants.
    C0 otherwise uses the unmodified historical component loss function.
    """
    if not episodes:
        raise ValueError('An admitted nonempty saved batch is required')
    parts = []
    for episode in episodes:
        features = episode_features(episode)
        labels = episode_labels(episode, device=features.device, dtype=features.dtype)
        if len(features) != len(labels['value']):
            raise ValueError('Saved state and complete label counts differ')
        prediction = model(features, episode['task_count'])
        if model.variant == 'C0':
            state_loss = remaining_cost_loss(prediction, episode['targets'], episode['task_count'])
        else:
            state_loss = 0.5 * torch.stack([
                (prediction[key] - labels[key]).square() for key in NORMALIZED_COMPONENTS
            ]).sum(0)
            if model.variant == 'C2':
                state_loss = state_loss + 0.5 * (prediction['value'] - labels['value']).square()
        if not bool(torch.isfinite(state_loss).all()):
            raise RuntimeError('Nonfinite matched offline Critic loss')
        parts.append(state_loss.mean())
    return torch.stack(parts).mean()


def window_key(episode):
    """A saved window identity, never inferred from a held-out target."""
    if 'window_key' in episode:
        return str(episode['window_key'])
    source = episode.get('window_size', episode.get('window'))
    if isinstance(source, (list, tuple)) and len(source) == 2:
        return f'{source[0]}x{source[1]}'
    if isinstance(source, str) and source:
        return source
    raise ValueError('An explicit saved window_key/window_size is required')


def progress_bin(remaining_fraction):
    """Four fixed bins from an observable state feature, not final trace T."""
    value = float(remaining_fraction)
    if not math.isfinite(value) or not -1e-6 <= value <= 1 + 1e-6:
        raise ValueError('Observable remaining fraction must lie in [0,1]')
    value = min(1., max(0., value))
    return min(3, max(0, math.floor(4 * (1 - value))))


def condition_key(episode, step, state_count):
    # state_count validates the saved index only; it does not condition the
    # constant. No t/T or eventual path length enters this bin.
    if type(step) is not int or type(state_count) is not int or not 0 <= step < state_count:
        raise ValueError('Invalid saved state index')
    source = episode['v'] if 'v' in episode else episode['batch']['v']
    return f'{window_key(episode)}|q{progress_bin(source[step][REMAINING_FRACTION_INDEX])}'


def _balanced_means(entries):
    """Fit-only parent equal, trajectory equal, within-condition state mean."""
    grouped = defaultdict(lambda: defaultdict(list))
    for entry in entries:
        grouped[str(entry['parent_id'])][str(entry['trajectory_id'])].append(entry['values'])
    if not grouped:
        raise ValueError('No fitting labels for reference estimation')
    per_parent = []
    for trajectories in grouped.values():
        per_parent.append(np.mean([np.mean(values, axis=0) for values in trajectories.values()], axis=0))
    values = np.mean(per_parent, axis=0)
    return dict(values={key: float(value) for key, value in zip(COMPONENTS, values)},
                parents=len(grouped), trajectories=sum(len(x) for x in grouped.values()), states=len(entries))


def fit_references(fit_episodes):
    """Global and window x observable-progress constants from fitting parents.

    All raw/n/value references are estimated separately on the same fitting
    fold. Unknown validation strata use the fitting global reference; no
    validation label, even a mean, participates in reference fitting.
    """
    all_entries, by_condition = [], defaultdict(list)
    parents, trajectories = set(), set()
    for episode in fit_episodes:
        if episode.get('split', 'fit') != 'fit':
            raise ValueError('Reference estimation accepts only the fixed fitting fold')
        parent = str(episode['parent_id'])
        identifier = str(episode['trajectory_id'])
        if identifier in trajectories:
            raise ValueError('A trajectory is duplicated in the reference pool')
        parents.add(parent)
        trajectories.add(identifier)
        features = episode_features(episode)
        labels = episode_labels(episode, dtype=torch.float64)
        if len(features) != len(labels['value']):
            raise ValueError('Saved feature and label counts differ')
        values = np.column_stack([labels[key].cpu().numpy() for key in COMPONENTS])
        for step, row in enumerate(values):
            entry = dict(parent_id=parent, trajectory_id=identifier, values=row)
            all_entries.append(entry)
            by_condition[condition_key(episode, step, len(values))].append(entry)
    return dict(schema='rc3-offline-critic-fit-references-v1',
                parent_ids=sorted(parents), trajectory_ids=sorted(trajectories),
                reduction='parent equal / trajectory equal / state mean within stratum',
                condition='saved window size x min(3,max(0,floor(4*(1-clamp(v[256],0,1))))); v[256]=observable remaining task fraction',
                unknown_condition='fitting global constant',
                global_constant=_balanced_means(all_entries),
                conditional_constants={key: _balanced_means(value) for key, value in sorted(by_condition.items())})


@torch.no_grad()
def predict_episode(model, episode):
    features = episode_features(episode)
    prediction = model(features, episode['task_count'])
    if any(not bool(torch.isfinite(prediction[key]).all()) for key in COMPONENTS):
        raise RuntimeError('Nonfinite offline Critic prediction')
    return {key: prediction[key].detach().cpu() for key in COMPONENTS}


def evaluate_rows(model, episodes, configuration, endpoint, split, references):
    """One row per saved state/component; sufficient for independent arithmetic."""
    if configuration != model.variant or configuration not in VARIANTS:
        raise ValueError('Evaluation configuration differs from the actual model')
    if split not in ('fit', 'validation'):
        raise ValueError('Use the preregistered whole-parent fit/validation split')
    fit_parents = set(references['parent_ids'])
    result = []
    for episode in episodes:
        parent = str(episode['parent_id'])
        if (parent in fit_parents) != (split == 'fit'):
            raise ValueError('Parent layout crossed the fitting/validation boundary')
        if episode.get('split', split) != split:
            raise ValueError('Episode split does not match the requested evaluation')
        prediction = predict_episode(model, episode)
        labels = episode_labels(episode, dtype=torch.float64)
        T = len(labels['value'])
        if len(prediction['value']) != T:
            raise ValueError('Prediction and complete saved label counts differ')
        for step in range(T):
            key = condition_key(episode, step, T)
            condition = references['conditional_constants'].get(key)
            fallback = condition is None
            if fallback:
                condition = references['global_constant']
            for component in COMPONENTS:
                target = float(labels[component][step])
                predicted = float(prediction[component][step])
                global_constant = references['global_constant']['values'][component]
                conditional_constant = condition['values'][component]
                result.append(dict(configuration=configuration, endpoint=endpoint, split=split,
                                   parent_id=parent, trajectory_id=str(episode['trajectory_id']),
                                   cohort=str(episode.get('cohort', 'not_recorded')),
                                   source_condition=str(episode.get('condition', 'not_recorded')),
                                   source_stage=str(episode.get('stage', 'not_recorded')),
                                   window=window_key(episode), state_index=step, state_count=T,
                                   task_count=episode['task_count'],
                                   remaining_fraction=float((episode['v'] if 'v' in episode else episode['batch']['v'])[step][REMAINING_FRACTION_INDEX]),
                                   progress_bin=int(key.rsplit('q', 1)[1]),
                                   component=component, target=target, prediction=predicted,
                                   error=predicted - target, global_constant=global_constant,
                                   conditional_constant=conditional_constant,
                                   conditional_fallback=fallback))
    return result


def saved_online_prediction_rows(episodes, references):
    """Read original saved predictions, with no Critic model forward.

    Returns ``(rows, missing)``. The collection Critic changes over the historical
    schedule, so these rows are labelled mixed checkpoints and are never a
    fourth matched fitting configuration. Missing saved states are retained in
    ``missing`` and are not imputed or replaced with a fresh model evaluation.
    """
    result, missing = [], []
    fit_parents = set(references['parent_ids'])
    for episode in episodes:
        identifier = str(episode['trajectory_id'])
        parent = str(episode['parent_id'])
        split = episode['split']
        if split not in ('fit', 'validation') or (parent in fit_parents) != (split == 'fit'):
            raise ValueError('Saved online prediction parent crossed the fixed split')
        labels = episode_labels(episode, dtype=torch.float64)
        T = len(labels['value'])
        features = episode_features(episode)
        if len(features) != T:
            raise ValueError('Saved online feature/label counts differ')
        predictions = episode.get('record', {}).get('critic_predictions')
        if not isinstance(predictions, (list, tuple)) or len(predictions) != T:
            missing.append(dict(trajectory_id=identifier, field='record.critic_predictions',
                                expected_states=T, actual_states=len(predictions) if isinstance(predictions, (list, tuple)) else None))
            continue
        for step, source in enumerate(predictions):
            required = ('raw_remaining_L', 'raw_remaining_K', 'terminal_H_credit', 'value')
            if not isinstance(source, dict) or any(key not in source for key in required):
                missing.append(dict(trajectory_id=identifier, state_index=step,
                                    field='record.critic_predictions[state]',
                                    missing_fields=list(required) if not isinstance(source, dict) else [key for key in required if key not in source]))
                continue
            prediction = {key: float(source[key]) for key in required}
            if not all(math.isfinite(value) for value in prediction.values()):
                raise ValueError('Nonfinite original saved Critic prediction')
            n = episode['task_count']
            prediction.update(normalized_L=prediction['raw_remaining_L'] / n,
                              normalized_K=prediction['raw_remaining_K'] / n,
                              normalized_H=prediction['terminal_H_credit'] / n)
            expected_value = -(prediction['normalized_L'] + PENALTY * prediction['normalized_K'] - prediction['normalized_H'])
            if not math.isclose(prediction['value'], expected_value, abs_tol=2e-5, rel_tol=2e-5):
                raise ValueError('Saved Critic value differs from its saved component accounting')
            key = condition_key(episode, step, T)
            condition = references['conditional_constants'].get(key)
            fallback = condition is None
            if fallback:
                condition = references['global_constant']
            for component in COMPONENTS:
                target, predicted = float(labels[component][step]), prediction[component]
                result.append(dict(configuration='R03_SAVED_ONLINE_CRITIC',
                                   endpoint='collection_mixed_checkpoints', split=split,
                                   parent_id=parent, trajectory_id=identifier,
                                   cohort=str(episode.get('cohort', 'not_recorded')),
                                   source_condition=str(episode.get('condition', 'not_recorded')),
                                   source_stage=str(episode.get('stage', 'not_recorded')),
                                   window=window_key(episode), state_index=step, state_count=T,
                                   task_count=n, remaining_fraction=float(features[step, REMAINING_FRACTION_INDEX]),
                                   progress_bin=int(key.rsplit('q', 1)[1]), component=component,
                                   target=target, prediction=predicted, error=predicted-target,
                                   global_constant=references['global_constant']['values'][component],
                                   conditional_constant=condition['values'][component],
                                   conditional_fallback=fallback))
    return result, missing


def _moments(rows):
    if not rows:
        raise ValueError('Cannot aggregate an empty state group')
    error = np.asarray([row['prediction'] - row['target'] for row in rows], dtype=np.float64)
    global_error = np.asarray([row['global_constant'] - row['target'] for row in rows], dtype=np.float64)
    condition_error = np.asarray([row['conditional_constant'] - row['target'] for row in rows], dtype=np.float64)
    return dict(mse=float(np.mean(error * error)), bias=float(np.mean(error)),
                mae=float(np.mean(np.abs(error))),
                target_mean=float(np.mean([row['target'] for row in rows])),
                prediction_mean=float(np.mean([row['prediction'] for row in rows])),
                global_constant_mse=float(np.mean(global_error * global_error)),
                conditional_constant_mse=float(np.mean(condition_error * condition_error)),
                conditional_fallback_fraction=float(np.mean([bool(row['conditional_fallback']) for row in rows])))


def summarize_rows(rows, group_by=('configuration', 'endpoint', 'split', 'component')):
    """Both state-pooled and parent-equal metrics, retaining bias decomposition.

    Parent-equal means: first state-average inside each parent, then equal
    average over the admitted parents. It never substitutes for the historical
    R03 stopping metric. In parent_equal, MSE = mean_parent_bias_squared +
    mean_within_parent_centered_error. The separate bias_squared is the square
    of the parent-mean signed bias; cancellation between opposite parent biases
    can make it much smaller than mean_parent_bias_squared.
    """
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in group_by)].append(row)
    output = []
    for identity, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        by_parent = defaultdict(list)
        for row in group:
            by_parent[row['parent_id']].append(row)
        parent_moments = [_moments(parent_rows) for parent_rows in by_parent.values()]
        parent_bias_sq = float(np.mean([value['bias'] ** 2 for value in parent_moments]))
        for aggregation in ('state_pooled', 'parent_equal'):
            if aggregation == 'state_pooled':
                moments = _moments(group)
                weight = np.asarray([len(parent_rows) for parent_rows in by_parent.values()], dtype=np.float64)
                parent_bias_sq_actual = float(np.average([value['bias'] ** 2 for value in parent_moments], weights=weight))
            else:
                moments = {key: float(np.mean([value[key] for value in parent_moments])) for key in parent_moments[0]}
                parent_bias_sq_actual = parent_bias_sq
            mse, bias = moments['mse'], moments['bias']
            record = dict(zip(group_by, identity))
            record.update(aggregation=aggregation, parents=len(by_parent),
                          trajectories=len({row['trajectory_id'] for row in group}), states=len(group),
                          **moments, rmse=math.sqrt(mse), bias_squared=bias * bias,
                          centered_error_variance=max(0., mse - bias * bias),
                          mean_parent_bias_squared=parent_bias_sq_actual,
                          mean_within_parent_centered_error=max(0., mse - parent_bias_sq_actual),
                          bias_squared_fraction_of_mse=(bias * bias / mse) if mse > 0 else 0.,
                          mse_over_global_constant=(mse / moments['global_constant_mse']) if moments['global_constant_mse'] > 0 else None,
                          mse_over_conditional_constant=(mse / moments['conditional_constant_mse']) if moments['conditional_constant_mse'] > 0 else None)
            output.append(record)
    return output
