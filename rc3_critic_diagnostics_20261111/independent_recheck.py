"""Independent arithmetic audit of this run's already-saved diagnostic tables.

Only stdlib and NumPy are imported.  This file never imports analysis helpers,
loads a model/checkpoint, reconstructs a state, fits a parameter, or reads any
historical trajectory.  ``recheck(folder)`` reads the six named output artifacts
and writes NUMERICAL_RECHECK.json.  It must itself be admitted before execution.

Coverage is all Actor state/variant rows, all legal candidates (streamed one
state at a time), all Critic state/component rows and all calibration summaries.
This establishes internal arithmetic consistency, not truth of model outputs or
planning quality.  Unavailable values are not replaced by reconstructed labels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np


ATOL = 1e-9
RTOL = 1e-7
COMPONENTS = (
    'raw_remaining_L', 'raw_remaining_K', 'terminal_H_credit',
    'normalized_L', 'normalized_K', 'normalized_H', 'value',
)
VARIANTS = ('C0', 'C1', 'C2')
FILES = (
    'ACTOR_RANK_RESPONSE.csv', 'ACTOR_CANDIDATES.csv.gz',
    'CRITIC_STATE_PREDICTIONS.csv.gz', 'CRITIC_MATCHED_CALIBRATION.csv',
    'CALIBRATION_SPLIT.json', 'TRAINING_LOG.csv',
)


def _missing(value):
    return value is None or str(value).strip().lower() in ('', 'none', 'null', 'nan')


def _number(value):
    return None if _missing(value) else float(value)


def _bool(value):
    if _missing(value):
        return None
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ('true', '1', 'yes', 'pass'):
        return True
    if str(value).strip().lower() in ('false', '0', 'no', 'fail'):
        return False
    raise ValueError('Invalid Boolean value: ' + str(value))


def _rows(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size}


class Checks:
    def __init__(self):
        self.groups = {}

    def add(self, name, condition, *, context=None, actual=None, expected=None):
        group = self.groups.setdefault(name, {'checks': 0, 'failures': 0, 'examples': []})
        group['checks'] += 1
        if not condition:
            group['failures'] += 1
            if len(group['examples']) < 8:
                group['examples'].append(dict(context=context, actual=actual, expected=expected))

    def equal(self, name, actual, expected, *, context=None, atol=ATOL, rtol=RTOL):
        if expected is None:
            okay = _missing(actual)
        elif isinstance(expected, (bool, np.bool_)):
            try:
                okay = _bool(actual) == bool(expected)
            except (ValueError, TypeError):
                okay = False
        elif isinstance(expected, str):
            okay = actual == expected
        else:
            try:
                av = float(actual)
                okay = math.isfinite(av) and math.isfinite(float(expected)) and math.isclose(av, float(expected), abs_tol=atol, rel_tol=rtol)
            except (ValueError, TypeError):
                okay = False
        self.add(name, okay, context=context, actual=actual, expected=expected)


def _state_key(row):
    return str(row['trajectory_id']), int(row['state_index'])


def _distribution(logits):
    shifted = logits - np.max(logits)
    denominator = np.sum(np.exp(shifted), dtype=np.float64)
    log_probabilities = shifted - math.log(float(denominator))
    return log_probabilities, np.exp(log_probabilities)


def _rank(logits):
    order = np.argsort(-logits, kind='stable')
    ranks = np.empty(len(logits), dtype=np.int64)
    ranks[order] = np.arange(1, len(logits) + 1)
    return order, ranks


def _delta(delta):
    mean = float(np.mean(delta))
    centered = delta - mean
    total = float(np.mean(delta ** 2))
    common = mean ** 2
    energy = float(np.mean(centered ** 2))
    return dict(min=float(np.min(delta)), max=float(np.max(delta)),
                span=float(np.max(delta) - np.min(delta)), mean=mean,
                rms=math.sqrt(total), centered_rms=math.sqrt(energy),
                mean_square=total, common_mean_square=common,
                centered_mean_square=energy,
                energy_closure_abs_error=abs(total - common - energy),
                centered_max_abs=float(np.max(np.abs(centered))),
                centered_min=float(np.min(centered)), centered_max=float(np.max(centered)),
                common_energy_fraction=min(1., max(0., common / total)) if total else None,
                zero_total_energy=total == 0.)


def _describe(values):
    if not values:
        return {'n': 0}
    array = np.asarray(values, dtype=np.float64)
    return dict(n=len(array), mean=float(np.mean(array)), min=float(np.min(array)),
                p10=float(np.quantile(array, .1)), median=float(np.median(array)),
                p90=float(np.quantile(array, .9)), max=float(np.max(array)))


def _actor(folder, split, check):
    state_rows = defaultdict(list)
    for row in _rows(folder / FILES[0]):
        state_rows[_state_key(row)].append(row)
    check.add('actor.nonempty_state_table', bool(state_rows))
    seen = set()
    candidate_count = 0
    single_count = 0
    multi_count = 0
    locked_count = 0
    values = defaultdict(list)
    variant_coverage = defaultdict(int)
    actual_after_deltas = []
    common_fields = ('legal_count', 'single_legal_action', 'base_top1_local_index',
                     'base_top2_local_index', 'base_top1_logit', 'base_top2_logit',
                     'base_top1_top2_gap', 'gap_gt_0_5', 'gap_eq_0_5',
                     'structurally_cannot_change_unique_top1', 'at_bound_can_at_most_tie',
                     'max_pairwise_adjustment', 'actual_action_base_rank',
                     'collection_actual_logp', 'pre_update_ratio', 'post_update_ratio',
                     'pre_to_post_top1_changed', 'pre_to_post_rank_changed_candidates',
                     'pre_to_post_max_probability_difference', 'pre_to_post_total_variation')
    for key, iterator in itertools.groupby(_rows(folder / FILES[1]), key=_state_key):
        candidates = sorted(list(iterator), key=lambda row: int(row['candidate_local_index']))
        candidate_count += len(candidates)
        check.add('actor.candidate_groups_unique', key not in seen, context=key)
        seen.add(key)
        responses = state_rows.get(key, [])
        check.add('actor.candidate_state_has_response', bool(responses), context=key)
        if not responses:
            continue
        labels = sorted({name[:-7] for name in candidates[0] if name.endswith('__logit')})
        check.add('actor.variant_columns_exist', bool(labels), context=key)
        response = responses[0]
        local_ids = np.asarray([int(row['candidate_local_index']) for row in candidates])
        check.add('actor.legal_candidates_unique', len(set(local_ids.tolist())) == len(local_ids), context=key)
        base = np.asarray([float(row['base_logit']) for row in candidates])
        check.add('actor.base_finite', bool(np.isfinite(base).all()), context=key)
        bo, br = _rank(base)
        blp, bp = _distribution(base)
        actual = [i for i, row in enumerate(candidates) if _bool(row.get('is_recorded_actual_action'))]
        check.add('actor.exactly_one_recorded_action', len(actual) == 1, context=key, actual=len(actual), expected=1)
        ap = actual[0] if len(actual) == 1 else None
        old = None
        if all(not _missing(row.get('collection_logit')) for row in candidates):
            old = np.asarray([float(row['collection_logit']) for row in candidates])
            check.add('actor.collection_logits_finite', bool(np.isfinite(old).all()), context=key)
            olp, op = _distribution(old)
        else:
            olp = op = None
        gap = float(base[bo[0]] - base[bo[1]]) if len(base) > 1 else None
        single_count += len(base) == 1
        multi_count += len(base) > 1
        locked_count += gap is not None and gap > .5
        expected = dict(legal_count=len(base), single_legal_action=len(base) == 1,
                        base_top1_local_index=int(local_ids[bo[0]]),
                        base_top2_local_index=int(local_ids[bo[1]]) if len(base) > 1 else None,
                        base_top1_logit=float(base[bo[0]]),
                        base_top2_logit=float(base[bo[1]]) if len(base) > 1 else None,
                        base_top1_top2_gap=gap, gap_gt_0_5=None if gap is None else gap > .5,
                        gap_eq_0_5=None if gap is None else gap == .5,
                        structurally_cannot_change_unique_top1=None if gap is None else gap > .5,
                        at_bound_can_at_most_tie=None if gap is None else gap == .5,
                        max_pairwise_adjustment=.5,
                        actual_action_base_rank=None if ap is None else int(br[ap]),
                        collection_actual_logp=None if olp is None or ap is None else float(olp[ap]))
        predictions = {}
        for i, row in enumerate(candidates):
            for field, value in [('base_probability', bp[i]), ('base_rank', br[i]),
                                 ('collection_probability', None if op is None else op[i])]:
                check.equal('actor.candidate_' + field, row.get(field), value, context=(key, i))
        for label in labels:
            prefix = label + '__'
            present = [not _missing(row.get(prefix + 'logit')) for row in candidates]
            check.add('actor.variant_missing_is_whole_state', all(present) or not any(present), context=(key, label))
            if not all(present):
                continue
            z = np.asarray([float(row[prefix + 'logit']) for row in candidates])
            residual = np.asarray([float(row[prefix + 'residual']) for row in candidates])
            check.add('actor.variant_finite', bool(np.isfinite(z).all() and np.isfinite(residual).all()), context=(key, label))
            check.add('actor.residual_amplitude_bound', bool((np.abs(residual) <= .25).all()), context=(key, label))
            # The helper saves both the true tanh residual and float32-effective logits.
            effective_sum = (base.astype(np.float32) + residual.astype(np.float32)).astype(np.float64)
            check.add('actor.float32_logit_composition', bool(np.array_equal(z, effective_sum)), context=(key, label))
            order, ranks = _rank(z)
            lp, prob = _distribution(z)
            centered = residual - np.mean(residual)
            for i, row in enumerate(candidates):
                for field, value in [('probability', prob[i]), ('rank', ranks[i]), ('centered_residual', centered[i])]:
                    check.equal('actor.candidate_variant_' + field, row.get(prefix + field), value, context=(key, label, i))
            out = {**{'residual_' + k: v for k, v in _delta(residual).items()},
                   **{'effective_logit_delta_' + k: v for k, v in _delta(z - base).items()},
                   'top1_local_index': int(local_ids[order[0]]),
                   'top1_changed_from_base': bool(order[0] != bo[0]),
                   'rank_changed_candidates': int(np.sum(ranks != br)),
                   'entire_rank_unchanged_from_base': bool(np.array_equal(order, bo)),
                   'probability_max_change_from_base': float(np.max(np.abs(prob - bp))),
                   'total_variation_from_base': float(np.sum(np.abs(prob - bp)) / 2),
                   'actual_action_logp': None if ap is None else float(lp[ap]),
                   'actual_action_ratio_to_collection': None if ap is None or olp is None else math.exp(float(lp[ap] - olp[ap]))}
            variant_coverage[label] += 1
            for metric in ('residual_mean', 'residual_centered_mean_square', 'residual_common_energy_fraction',
                           'top1_changed_from_base', 'rank_changed_candidates', 'total_variation_from_base'):
                if out[metric] is not None:
                    values[label + '__' + metric].append(float(out[metric]))
            source = [row for row in responses if row.get('variant') == label]
            if 'variant' not in response:
                source = responses
            check.add('actor.variant_has_unique_response', len(source) == 1, context=(key, label), actual=len(source), expected=1)
            if source:
                for field, value in out.items():
                    name = field if 'variant' in source[0] else prefix + field
                    check.equal('actor.state_variant_' + field, source[0].get(name), value, context=(key, label))
            if gap is not None and gap > .5:
                check.add('actor.theoretical_locked_top1', order[0] == bo[0], context=(key, label))
            predictions[label] = dict(order=order, ranks=ranks, prob=prob, lp=lp, out=out, z=z)
        before_label = response.get('source_before')
        after_label = response.get('source_after')
        if _missing(before_label):
            before_label = candidates[0].get('source_before')
        if _missing(after_label):
            after_label = candidates[0].get('source_after')
        check.add('actor.explicit_before_after_labels', before_label in predictions and after_label in predictions,
                  context=key, actual=[before_label, after_label], expected='two present saved variant labels')
        before, after = predictions.get(before_label), predictions.get(after_label)
        if after is not None:
            actual_after_deltas.append(after['out'])
        expected.update(pre_update_ratio=None if before is None else before['out']['actual_action_ratio_to_collection'],
                        post_update_ratio=None if after is None else after['out']['actual_action_ratio_to_collection'],
                        pre_to_post_top1_changed=None if before is None or after is None else bool(before['order'][0] != after['order'][0]),
                        pre_to_post_rank_changed_candidates=None if before is None or after is None else int(np.sum(before['ranks'] != after['ranks'])),
                        pre_to_post_max_probability_difference=None if before is None or after is None else float(np.max(np.abs(before['prob'] - after['prob']))),
                        pre_to_post_total_variation=None if before is None or after is None else float(np.sum(np.abs(before['prob'] - after['prob'])) / 2))
        for contrast, left, right in (
            ('ON_feature_switch', 'ON_after', 'ON_after_OFF'),
            ('OFF_feature_switch', 'OFF_after_ON', 'OFF_after'),
            ('ON_OFF_weights_fixed_ON', 'ON_after', 'OFF_after_ON'),
            ('ON_OFF_weights_fixed_OFF', 'ON_after_OFF', 'OFF_after'),
        ):
            a, b = predictions.get(left), predictions.get(right)
            for row in responses:
                prefix = contrast + '__'
                check.equal('actor.contrast_status', row.get(prefix + 'status'), 'present' if a is not None and b is not None else 'missing_variant', context=(key, contrast))
                if a is None or b is None:
                    continue
                difference = a['z'] - b['z']
                contrast_metrics = dict(top1_differs=bool(a['order'][0] != b['order'][0]),
                                        rank_changed_candidates=int(np.sum(a['ranks'] != b['ranks'])),
                                        total_variation=float(np.sum(np.abs(a['prob'] - b['prob'])) / 2),
                                        max_probability_difference=float(np.max(np.abs(a['prob'] - b['prob']))),
                                        centered_logit_difference_rms=float(np.sqrt(np.mean((difference - np.mean(difference)) ** 2))))
                for field, value in contrast_metrics.items():
                    check.equal('actor.contrast_' + field, row.get(prefix + field), value, context=(key, contrast))
        for field in ('pre_update_ratio', 'post_update_ratio', 'pre_to_post_top1_changed', 'pre_to_post_total_variation'):
            if expected[field] is not None:
                values[field].append(float(expected[field]))
        for row in responses:
            for field in common_fields:
                check.equal('actor.state_' + field, row.get(field), expected[field], context=key)
            if ap is not None:
                check.equal('actor.recorded_action_identity', row.get('actual_action_local_index'), int(local_ids[ap]), context=key)
            saved_probability = _number(row.get('saved_selected_probability'))
            reconstructed = None if op is None or ap is None else float(op[ap])
            check.equal('actor.reconstructed_collection_probability', row.get('reconstructed_collection_selected_probability'), reconstructed, context=key)
            difference = None if saved_probability is None or reconstructed is None else abs(saved_probability - reconstructed)
            check.equal('actor.saved_collection_probability_difference', row.get('saved_collection_probability_abs_difference'), difference, context=key)
            saved_logit = _number(row.get('saved_selected_actor_logit'))
            difference = None if saved_logit is None or old is None or ap is None else abs(saved_logit - float(old[ap]))
            check.equal('actor.saved_collection_logit_difference', row.get('saved_collection_logit_abs_difference'), difference, context=key)
    check.add('actor.state_candidate_coverage_equal', seen == set(state_rows), actual=len(seen), expected=len(state_rows))
    trajectories = {key[0] for key in state_rows}
    check.add('actor.trajectory_cap_128', len(trajectories) <= 128)
    selected = set(split.get('selected_trajectory_ids', []))
    check.add('actor.registered_trajectory_set_exact', bool(selected) and trajectories == selected, actual=len(trajectories), expected=len(selected))
    total_energy = sum(row['residual_mean_square'] for row in actual_after_deltas)
    common_energy = sum(row['residual_common_mean_square'] for row in actual_after_deltas)
    centered_energy = sum(row['residual_centered_mean_square'] for row in actual_after_deltas)
    actual_summary = dict(states=len(actual_after_deltas),
                          residual_state_equal_common_energy_fraction=common_energy / total_energy if total_energy else None,
                          residual_rms=math.sqrt(total_energy / len(actual_after_deltas)) if actual_after_deltas else None,
                          residual_centered_rms=math.sqrt(centered_energy / len(actual_after_deltas)) if actual_after_deltas else None,
                          residual_min=min((row['residual_min'] for row in actual_after_deltas), default=None),
                          residual_max=max((row['residual_max'] for row in actual_after_deltas), default=None),
                          residual_max_span=max((row['residual_span'] for row in actual_after_deltas), default=None))
    return dict(states=len(state_rows), state_variant_rows=sum(map(len, state_rows.values())),
                legal_candidate_rows=candidate_count, trajectories=len({key[0] for key in state_rows}),
                single_legal_states=single_count, multi_legal_states=multi_count,
                gap_gt_0_5_states=locked_count,
                gap_gt_0_5_fraction_among_multi=(locked_count / multi_count) if multi_count else None,
                gap_gt_0_5_fraction_among_all=(locked_count / len(state_rows)) if state_rows else None,
                single_or_strictly_locked_fraction=(single_count + locked_count) / len(state_rows) if state_rows else None,
                actual_source_after=actual_summary,
                variant_present_states=dict(variant_coverage),
                distributions={key: _describe(val) for key, val in sorted(values.items())})


def _moments(rows):
    target = np.asarray([float(row['target']) for row in rows])
    prediction = np.asarray([float(row['prediction']) for row in rows])
    error = prediction - target
    return dict(mse=float(np.mean(error ** 2)), bias=float(np.mean(error)),
                mae=float(np.mean(np.abs(error))), target_mean=float(np.mean(target)),
                prediction_mean=float(np.mean(prediction)),
                global_constant_mse=float(np.mean((np.asarray([float(row['global_constant']) for row in rows]) - target) ** 2)),
                conditional_constant_mse=float(np.mean((np.asarray([float(row['conditional_constant']) for row in rows]) - target) ** 2)),
                conditional_fallback_fraction=float(np.mean([bool(_bool(row['conditional_fallback'])) for row in rows])))


def _summary(rows, aggregation):
    parents = defaultdict(list)
    for row in rows:
        parents[row['parent_id']].append(row)
    parent_moments = [_moments(group) for group in parents.values()]
    if aggregation == 'state_pooled':
        result = _moments(rows)
        parent_bias_sq = float(np.average([mom['bias'] ** 2 for mom in parent_moments], weights=[len(group) for group in parents.values()]))
    elif aggregation in ('parent_equal', 'parent_macro'):
        result = {key: float(np.mean([mom[key] for mom in parent_moments])) for key in parent_moments[0]}
        parent_bias_sq = float(np.mean([mom['bias'] ** 2 for mom in parent_moments]))
    else:
        raise ValueError('Unknown aggregation: ' + aggregation)
    mse, bias = result['mse'], result['bias']
    result.update(parents=len(parents), trajectories=len({row['trajectory_id'] for row in rows}), states=len(rows),
                  rmse=math.sqrt(mse), bias_squared=bias ** 2,
                  centered_error_variance=max(0., mse - bias ** 2),
                  mean_parent_bias_squared=parent_bias_sq,
                  mean_within_parent_centered_error=max(0., mse - parent_bias_sq),
                  bias_squared_fraction_of_mse=(bias ** 2 / mse) if mse else 0.,
                  mse_over_global_constant=mse / result['global_constant_mse'] if result['global_constant_mse'] > 0 else None,
                  mse_over_conditional_constant=mse / result['conditional_constant_mse'] if result['conditional_constant_mse'] > 0 else None)
    return result


def _canonical_split(value):
    return 'validation' if value in ('validation', 'holdout') else value


def _balanced_reference(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row['parent_id']][row['trajectory_id']].append(float(row['target']))
    return float(np.mean([np.mean([np.mean(vals) for vals in trajectories.values()]) for trajectories in grouped.values()]))


def _critic(folder, split, check):
    rows = list(_rows(folder / FILES[2]))
    summaries = list(_rows(folder / FILES[3]))
    check.add('critic.nonempty_prediction_and_summary_tables', bool(rows) and bool(summaries))
    fit_parents = set(split.get('fit_physical_parents', split.get('fit_parent_ids', [])))
    validation_parents = set(split.get('holdout_physical_parents', split.get('validation_physical_parents', split.get('validation_parent_ids', []))))
    check.add('split.both_parent_folds_nonempty', bool(fit_parents) and bool(validation_parents))
    check.add('split.physical_parent_intersection_empty', not (fit_parents & validation_parents), actual=sorted(fit_parents & validation_parents), expected=[])
    folds = split.get('trajectory_fold', {})
    check.add('split.trajectory_folds_saved', bool(folds))
    unique = set()
    state_components = defaultdict(dict)
    source_labels = {}
    endpoint_states = defaultdict(set)
    fit_reference_rows = {}
    for row in rows:
        key = (row['configuration'], row['endpoint'], row['trajectory_id'], int(row['state_index']), row['component'])
        check.add('critic.state_component_keys_unique', key not in unique, context=key)
        unique.add(key)
        component = row['component']
        check.add('critic.known_component', component in COMPONENTS, context=key)
        numbers = [float(row[name]) for name in ('target', 'prediction', 'error', 'global_constant', 'conditional_constant', 'remaining_fraction')]
        check.add('critic.all_prediction_numbers_finite', all(math.isfinite(value) for value in numbers), context=key)
        check.equal('critic.saved_error', row['error'], float(row['prediction']) - float(row['target']), context=key)
        parent = row['parent_id']
        fold = _canonical_split(row['split'])
        check.add('split.prediction_parent_membership', (parent in fit_parents and fold == 'fit') or (parent in validation_parents and fold == 'validation'), context=key)
        check.equal('split.trajectory_fold_membership', fold, _canonical_split(folds.get(row['trajectory_id'])), context=key)
        remaining = float(row['remaining_fraction'])
        progress_bin = min(3, max(0, math.floor(4 * (1 - min(1., max(0., remaining))))))
        check.add('critic.remaining_fraction_in_range', -1e-6 <= remaining <= 1 + 1e-6, context=key)
        check.equal('critic.observable_progress_bin', row['progress_bin'], progress_bin, context=key)
        state = key[:-1]
        state_components[state][component] = row
        endpoint_states[(row['configuration'], row['endpoint'])].add((row['trajectory_id'], int(row['state_index'])))
        source_key = (row['trajectory_id'], int(row['state_index']), component)
        if source_key in source_labels:
            check.equal('critic.targets_identical_across_models', row['target'], source_labels[source_key], context=key)
        else:
            source_labels[source_key] = float(row['target'])
        if fold == 'fit':
            fit_reference_rows.setdefault(source_key, row)
    for state, components in state_components.items():
        check.add('critic.complete_seven_components', set(components) == set(COMPONENTS), context=state)
        if set(components) != set(COMPONENTS):
            continue
        n = int(components['value']['task_count'])
        check.add('critic.positive_initial_task_count', n > 0, context=state)
        for field in ('target', 'prediction'):
            for raw, normal in zip(COMPONENTS[:3], COMPONENTS[3:6]):
                check.equal('critic.' + field + '_normalization', components[normal][field], float(components[raw][field]) / n, context=(state, raw), atol=2e-5, rtol=2e-5)
            expected_value = -(float(components['normalized_L'][field]) + .75 * float(components['normalized_K'][field]) - float(components['normalized_H'][field]))
            check.equal('critic.' + field + '_value_composition', components['value'][field], expected_value, context=state, atol=2e-5, rtol=2e-5)
    # References are independently estimated from one copy of each fitting label.
    fit_by_component = defaultdict(list)
    fit_by_condition = defaultdict(list)
    for row in fit_reference_rows.values():
        fit_by_component[row['component']].append(row)
        condition = (row['component'], row['window'], int(row['progress_bin']))
        fit_by_condition[condition].append(row)
    global_reference = {key: _balanced_reference(group) for key, group in fit_by_component.items()}
    conditional_reference = {key: _balanced_reference(group) for key, group in fit_by_condition.items()}
    for row in rows:
        component = row['component']
        condition = (component, row['window'], int(row['progress_bin']))
        fallback = condition not in conditional_reference
        reference = global_reference.get(component)
        check.equal('critic.fit_only_global_reference', row['global_constant'], reference, context=(row['configuration'], row['trajectory_id'], row['state_index'], component))
        check.equal('critic.fit_only_conditional_reference', row['conditional_constant'], conditional_reference.get(condition, reference), context=(row['configuration'], row['trajectory_id'], row['state_index'], component))
        check.equal('critic.conditional_fallback', row['conditional_fallback'], fallback)
    # Summary rows may additionally be stratified by any of these saved fields.
    selectors = ('configuration', 'endpoint', 'split', 'component', 'window', 'cohort', 'source_condition', 'source_stage')
    recomputed = []
    summary_identities = set()
    grouped_cache = {}
    for summary in summaries:
        dims = tuple(name for name in selectors if not _missing(summary.get(name)))
        identity = tuple(summary[name] for name in dims)
        marker = (dims, identity, summary.get('aggregation'))
        check.add('critic.summary_keys_unique', marker not in summary_identities, context=marker)
        summary_identities.add(marker)
        if dims not in grouped_cache:
            groups = defaultdict(list)
            for row in rows:
                groups[tuple(row.get(name) for name in dims)].append(row)
            grouped_cache[dims] = groups
        selected = grouped_cache[dims].get(identity, [])
        check.add('critic.summary_group_nonempty', bool(selected), context=marker)
        if not selected:
            continue
        metrics = _summary(selected, summary['aggregation'])
        for name, value in metrics.items():
            check.equal('critic.summary_' + name, summary.get(name), value, context=marker)
        check.equal('critic.mse_bias_variance_closure', metrics['mse'], metrics['bias_squared'] + metrics['centered_error_variance'], context=marker)
        check.equal('critic.mse_parent_bias_variance_closure', metrics['mse'], metrics['mean_parent_bias_squared'] + metrics['mean_within_parent_centered_error'], context=marker)
        recomputed.append(dict(zip(dims, identity), aggregation=summary['aggregation'], **metrics))
    matched = {config: sorted(endpoint for variant, endpoint in endpoint_states if variant == config) for config in VARIANTS}
    check.add('critic.three_matched_configurations', all(matched[config] for config in VARIANTS))
    for config in VARIANTS:
        check.add('critic.two_matched_endpoints_' + config, len(matched[config]) == 2, actual=matched[config], expected='init and 256-update endpoints')
        check.add('critic.required_0_256_endpoints_' + config, set(matched[config]) == {'0', '256'}, actual=matched[config], expected=['0', '256'])
    all_matched_sets = [states for (config, _), states in endpoint_states.items() if config in VARIANTS]
    check.add('critic.matched_endpoint_state_sets_equal', bool(all_matched_sets) and all(states == all_matched_sets[0] for states in all_matched_sets))
    check.add('critic.registered_trajectory_set_exact', {row['trajectory_id'] for row in rows} == set(split.get('selected_trajectory_ids', [])))
    originals = {endpoint: states for (config, endpoint), states in endpoint_states.items() if config == 'R03_SAVED_ONLINE_CRITIC'}
    check.add('critic.saved_online_rows_explicitly_descriptive', bool(originals) and set(originals) == {'collection_mixed_checkpoints'}, actual=list(originals), expected=['collection_mixed_checkpoints'])
    # At identical initial tensors C0 raw outputs equal C1/C2 normalized outputs.
    init_names = {'init', 'initial', 'initialization', 'step0', 'step_0', '000', '0', 'initial_000'}
    initial = {config: next((name for name in matched[config] if name.lower() in init_names), None) for config in VARIANTS}
    check.add('critic.initial_endpoint_identifiable', all(initial.values()), actual=initial)
    if all(initial.values()) and all_matched_sets:
        for trajectory, step in all_matched_sets[0]:
            states = {config: state_components.get((config, initial[config], trajectory, step), {}) for config in VARIANTS}
            if all(set(value) == set(COMPONENTS) for value in states.values()):
                for raw, normal in zip(COMPONENTS[:3], COMPONENTS[3:6]):
                    for config in ('C1', 'C2'):
                        check.equal('critic.identical_initial_net_tensors_output_identity', states[config][normal]['prediction'], float(states['C0'][raw]['prediction']), context=(trajectory, step, config, normal), atol=2e-6, rtol=2e-6)
    return dict(state_component_rows=len(rows), calibration_summary_rows=len(summaries),
                trajectories=len({row['trajectory_id'] for row in rows}),
                physical_fit_parents=len(fit_parents), physical_validation_parents=len(validation_parents),
                matched_endpoints=matched,
                saved_R03_description={'configuration': 'R03_SAVED_ONLINE_CRITIC',
                                       'endpoint': 'collection_mixed_checkpoints',
                                       'scope': 'saved historical changing Critic predictions; descriptive only, not a fourth fitted configuration',
                                       'states_by_endpoint': {key: len(value) for key, value in originals.items()}},
                independently_recomputed_calibration=recomputed)


def _training(folder, split, check):
    rows = list(_rows(folder / FILES[5]))
    by_configuration = defaultdict(dict)
    fields = rows[0] if rows else {}
    check.add('training.nonempty_log', bool(rows))
    aliases = {
        'step': ('step', 'update', 'optimizer_step'),
        'batch': ('batch_trajectory_ids', 'batch_ids', 'trajectory_ids', 'batch_sha256'),
        'loss': ('loss', 'loss_before', 'training_loss'),
        'gradient': ('grad_norm', 'gradient_norm'),
        'finite': ('gradient_finite', 'gradients_finite', 'finite_gradients', 'finite_gradient'),
        'attempt': ('optimizer_attempt', 'optimizer_attempted'),
        'completed': ('optimizer_completed', 'optimizer_step_completed', 'step_completed'),
    }
    names = {key: next((name for name in options if name in fields), None) for key, options in aliases.items()}
    for key, name in names.items():
        check.add('training.required_column_' + key, name is not None, actual=list(fields), expected=list(aliases[key]))
    fit_trajectories = {trajectory for trajectory, fold in split.get('trajectory_fold', {}).items() if fold == 'fit'}
    for row in rows:
        config = row.get('configuration')
        check.add('training.only_C0_C1_C2', config in VARIANTS, actual=config)
        if not all(names.values()) or config not in VARIANTS:
            continue
        step = int(row[names['step']])
        check.add('training.unique_configuration_step', step not in by_configuration[config], context=(config, step))
        by_configuration[config][step] = row[names['batch']]
        check.add('training.finite_loss', math.isfinite(float(row[names['loss']])), context=(config, step))
        norm = float(row[names['gradient']])
        check.add('training.finite_gradient_norm', math.isfinite(norm) and norm >= 0, context=(config, step))
        check.equal('training.finite_gradient_flag', row[names['finite']], True, context=(config, step))
        check.equal('training.optimizer_attempt', row[names['attempt']], True, context=(config, step))
        check.equal('training.optimizer_completed', row[names['completed']], True, context=(config, step))
        if names['batch'] != 'batch_sha256':
            try:
                batch = json.loads(row[names['batch']])
                check.add('training.batch_is_nonempty_trajectory_list', isinstance(batch, list) and bool(batch), context=(config, step))
                if isinstance(batch, list):
                    check.add('training.fit_trajectories_only', all(identifier in fit_trajectories for identifier in batch), context=(config, step))
                    by_configuration[config][step] = tuple(batch)
            except (ValueError, TypeError):
                check.add('training.batch_is_valid_json', False, context=(config, step), actual=row[names['batch']])
    for config in VARIANTS:
        steps = set(by_configuration[config])
        check.add('training.exact_256_updates_' + config, steps == set(range(1, 257)), actual=len(steps), expected=256)
    for step in range(1, 257):
        batches = [by_configuration[config].get(step) for config in VARIANTS]
        check.add('training.same_batch_schedule_all_three', batches[0] is not None and batches[0] == batches[1] == batches[2], context=step, actual=batches)
    check.add('training.exact_768_rows', len(rows) == 768, actual=len(rows), expected=768)
    return dict(rows=len(rows), updates_by_configuration={config: len(by_configuration[config]) for config in VARIANTS},
                schedule_comparison='all 256 steps across C0/C1/C2', log_columns=names)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def recheck(folder) -> dict:
    """Recompute saved evidence only and write a self-contained PASS/FAIL receipt."""
    start = time.perf_counter()
    folder = Path(folder).resolve()
    output = folder / 'NUMERICAL_RECHECK.json'
    if output.exists():
        raise RuntimeError('Independent arithmetic receipt already exists; no unregistered replay')
    checks = Checks()
    result = dict(schema='rc3-critic-diagnostic-independent-arithmetic-v1',
                  imports='stdlib + numpy; no actor/critic/data helper imports',
                  tolerances={'default_absolute': ATOL, 'default_relative': RTOL,
                              'float32_component_composition_absolute_and_relative': 2e-5,
                              'initial_output_identity_absolute_and_relative': 2e-6},
                  coverage='all Actor state/variant and legal-candidate rows; all Critic predictions and summaries; all optimizer log rows',
                  scientific_limit='Internal arithmetic audit of already-saved outputs; no independent model evaluation or route-quality evidence.',
                  input_files={})
    for name in FILES:
        path = folder / name
        checks.add('inputs.' + name + '_exists', path.is_file())
        if path.is_file():
            result['input_files'][name] = _hash(path)
    if all((folder / name).is_file() for name in FILES):
        try:
            split = json.loads((folder / FILES[4]).read_text(encoding='utf-8-sig'))
            result['actor'] = _actor(folder, split, checks)
            result['critic'] = _critic(folder, split, checks)
            result['training'] = _training(folder, split, checks)
        except Exception as error:
            # Malformed evidence is a failure; never silently skip a table.
            checks.add('audit.completed_without_parse_or_schema_error', False,
                       actual=type(error).__name__ + ': ' + str(error), expected='complete arithmetic audit')
    result['checks'] = checks.groups
    total = sum(group['checks'] for group in checks.groups.values())
    failed = sum(group['failures'] for group in checks.groups.values())
    result.update(status='PASS' if failed == 0 else 'FAIL', check_count=total,
                  pass_count=total - failed, fail_count=failed,
                  wall_seconds=time.perf_counter() - start)
    temporary = output.with_suffix('.json.tmp')
    result = _json_safe(result)
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    receipt = recheck(args.folder)
    print(json.dumps({key: receipt[key] for key in ('status', 'check_count', 'pass_count', 'fail_count', 'wall_seconds')}, ensure_ascii=False))
    raise SystemExit(0 if receipt['status'] == 'PASS' else 1)
