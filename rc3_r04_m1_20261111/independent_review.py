"""Independent saved-table arithmetic review for the authorized M1 block.

This module never imports torch, a model, a planner, a training loss or the
production summary implementation. Invoke only inside the registered cumulative
wall-clock budget. Inputs are this stage's saved predictions, summaries and gate.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


SEED = 11622026
COMPONENTS = ('raw_remaining_L', 'raw_remaining_K', 'terminal_H_credit',
              'normalized_L', 'normalized_K', 'normalized_H', 'value')
SIZES = ('10x10', '4x3', '6x6', '8x8')
AGGREGATIONS = ('state_pooled', 'parent_equal')


def _rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def _bool(value):
    if value in (True, 'True', 'true', '1', 1):
        return True
    if value in (False, 'False', 'false', '0', 0):
        return False
    raise ValueError('Noncanonical saved boolean')


def _close(a, b):
    return math.isclose(float(a), float(b), rel_tol=2e-6, abs_tol=2e-8)


def _file_identity(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return dict(path=path.name, bytes=path.stat().st_size, sha256=digest.hexdigest())


def recheck(folder, *, heartbeat=lambda: None):
    """Return a review receipt; do not overwrite any prior evidence.

    ``heartbeat`` is the root run's already-started durable budget check. This
    audit uses the complete physical parent as the only resampling unit. It
    estimates constants independently from step0/C0 fit prediction targets.
    """
    start = time.perf_counter()
    folder = Path(folder)
    state_path = folder / 'M1_STATE_PREDICTIONS.csv.gz'
    if not state_path.exists():
        state_path = folder / 'M1_STATE_PREDICTIONS.csv'
    summary_path = folder / 'M1_METRICS.csv'
    gate_path = folder / 'M1_GATE_DECISION.json'
    registry_path = folder / 'M1_DATA_AND_SOURCE_IDENTITY.json'
    schedule_path = folder / 'BATCH_SCHEDULE.json'
    training_path = folder / 'TRAINING_LOG.csv'
    resource_path = folder / 'RESOURCE_ACCOUNTING.json'
    event_path = folder / 'EVENTS.jsonl'
    files = [state_path, summary_path, gate_path, registry_path, schedule_path,
             training_path, resource_path, event_path]
    identities = [_file_identity(path) for path in files]
    checks = 0
    failures = []

    def check(condition, description, context=None):
        nonlocal checks
        checks += 1
        if not condition and len(failures) < 100:
            failures.append(dict(check=description, context=context))
        if checks % 25000 == 0:
            heartbeat()

    registry = json.loads(registry_path.read_text(encoding='utf-8-sig'))
    schedule = json.loads(schedule_path.read_text(encoding='utf-8-sig'))
    old_split_path = folder.parent/'rc3_critic_diagnostics_20261111/CALIBRATION_SPLIT.json'
    old_split = json.loads(old_split_path.read_text(encoding='utf-8-sig'))
    old_fit = set(old_split['fit_physical_parents'])
    old50 = old_fit | set(old_split['holdout_physical_parents'])
    training_parents = set(registry['all_R03_training_physical_parents'])
    reserve_hash = lambda parent: hashlib.sha256(f'{SEED}|R04-validation|{parent}'.encode()).hexdigest()
    expected_reserve = sorted(training_parents - old50, key=reserve_hash)[:16]
    check(len(training_parents) == 96 and len(training_parents-old50) == 46,
          'metadata 96 training / 46 outside previous50')
    check(set(registry['fit_physical_parents']) == old_fit and len(old_fit) == 33,
          'unchanged previous33 fit parents')
    check(registry['holdout_physical_parents'] == expected_reserve,
          'exact prior SHA256 reserve rule')
    selected = {row['trajectory_id']: row for row in registry['trajectories']}
    check(len(selected) == len(registry['trajectories']) <= 128,
          'unique metadata selection within128')
    check(len(schedule['batches']) == 256, 'full fixed256 batch schedule')
    for number, batch in enumerate(schedule['batches'], 1):
        check(len(batch) == 8 and len(set(batch)) == 8, 'eight distinct trajectories per batch', number)
        check(all(rid in selected and selected[rid]['split'] == 'fit' for rid in batch),
              'all scheduled trajectories from fit fold', number)
        check(len({selected[rid]['parent_id'] for rid in batch}) == 8,
              'eight distinct fit physical parents per batch', number)

    # Each group/parent holds state count, squared error, absolute error,
    # signed error, target sum, prediction sum, and two reference squared sums.
    moments = defaultdict(lambda: defaultdict(lambda: np.zeros(9, dtype=np.float64)))
    group_trajectories = defaultdict(set)
    parent_sets = defaultdict(set)
    state_seen = set()
    trajectories = {}
    reference_targets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    reference_values = {}
    endpoint_sets = defaultdict(set)
    composition_rows = {}
    initial_predictions = defaultdict(dict)
    rows_count = 0
    for row in _rows(state_path):
        rows_count += 1
        model, endpoint, split = row['configuration'], int(row['endpoint']), row['split']
        parent, traj = row['parent_id'], row['trajectory_id']
        window, phase, comp = row['window'], row['source_stage'], row['component']
        state = int(row['state_index'])
        identity = (model, endpoint, traj, state, comp)
        check(identity not in state_seen, 'unique state/component row', identity)
        state_seen.add(identity)
        check(model in ('C0', 'C1') and endpoint in (0, 128, 256), 'registered model/endpoint', identity)
        check(split in ('fit', 'validation') and window in SIZES and comp in COMPONENTS,
              'registered split/window/component', identity)
        endpoint_sets[model].add(endpoint)
        parent_sets[split].add(parent)
        previous = trajectories.setdefault(traj, (parent, split, window, phase, int(row['task_count'])))
        check(previous == (parent, split, window, phase, int(row['task_count'])),
              'trajectory identity remains fixed', traj)
        meta = selected[traj]
        check((parent, split, window, phase) ==
              (meta['parent_id'], meta['split'], f"{meta['width']}x{meta['height']}", meta['phase']),
              'prediction row matches locked metadata', identity)
        y, pred = float(row['target']), float(row['prediction'])
        global_pred, cond_pred = float(row['global_constant']), float(row['conditional_constant'])
        check(all(map(math.isfinite, (y, pred, global_pred, cond_pred))), 'finite values', identity)
        error = pred - y
        if endpoint == 0:
            initial_predictions[(traj, state, comp)][model] = pred
        fallback_value = _bool(row['conditional_fallback'])
        values = np.asarray((1., error * error, abs(error), error, y, pred,
                             (global_pred-y)**2, (cond_pred-y)**2, float(fallback_value)), dtype=np.float64)
        for size_group in ('ALL', window):
            for phase_group in ('ALL', phase):
                key = (model, endpoint, split, size_group, phase_group, comp)
                moments[key][parent] += values
                group_trajectories[key].add(traj)
        composition_key = (model, endpoint, traj, state)
        compound = composition_rows.setdefault(composition_key, {})
        compound[comp] = (y, pred, global_pred, cond_pred)
        if len(compound) == 7:
            n = int(row['task_count'])
            check(n > 0, 'positive fixed initial task count', composition_key)
            # Model and target raw/n units and value compose. Reference raw
            # constants were fit independently and need not equal n * norm.
            for i, role in ((0, 'target'), (1, 'prediction')):
                for raw, normalized in zip(COMPONENTS[:3], COMPONENTS[3:6]):
                    check(_close(compound[raw][i] / n, compound[normalized][i]),
                          'raw/n unit equivalence ' + role, composition_key)
            for i, role in enumerate(('target', 'prediction', 'global', 'conditional')):
                composed = -(compound['normalized_L'][i] + .75*compound['normalized_K'][i]
                             - compound['normalized_H'][i])
                check(math.isclose(composed, compound['value'][i], rel_tol=2e-5, abs_tol=2e-5),
                      'composed value ' + role, composition_key)
            del composition_rows[composition_key]
        # All configurations/endpoints must share identical labels/references.
        cross_key = (traj, state, comp)
        cross_value = (y, global_pred, cond_pred, _bool(row['conditional_fallback']))
        if cross_key in reference_values:
            check(reference_values[cross_key] == cross_value, 'fixed labels and constants', cross_key)
        else:
            reference_values[cross_key] = cross_value
        # Conditioning is observable current fraction, not t/T. Required state
        # field permits an independent four-bin calculation for fit-only means.
        if model == 'C0' and endpoint == 0 and split == 'fit':
            fraction = float(row['remaining_fraction'])
            check(math.isfinite(fraction) and -1e-6 <= fraction <= 1+1e-6,
                  'observable remaining fraction', cross_key)
            bucket = min(3, max(0, math.floor(4*(1-min(1., max(0., fraction))))))
            cond_key = f'{window}|q{bucket}'
            reference_targets[('GLOBAL', comp)][parent][traj].append(y)
            reference_targets[(cond_key, comp)][parent][traj].append(y)
    heartbeat()
    check(not composition_rows, 'all state rows have seven components')
    check(not parent_sets['fit'] & parent_sets['validation'], 'complete parent isolation')
    check(len(parent_sets['fit']) == 33 and len(parent_sets['validation']) == 16,
          'fixed parent counts', {k: len(v) for k, v in parent_sets.items()})
    check(len(trajectories) <= 128, 'saved trajectory cap', len(trajectories))
    check(set(trajectories) == set(selected), 'all locked trajectories represented, none replaced')
    check(endpoint_sets['C0'] == endpoint_sets['C1'], 'paired complete endpoints')
    check(0 in endpoint_sets['C0'] and 128 in endpoint_sets['C0'], 'initial and midpoint saved')
    for (traj, state, comp), values in initial_predictions.items():
        check(set(values) == {'C0', 'C1'}, 'paired step0 predictions', (traj, state, comp))
        if comp in COMPONENTS[:3]:
            normalized = COMPONENTS[3 + COMPONENTS[:3].index(comp)]
            check(_close(values['C0'], initial_predictions[(traj, state, normalized)]['C1']),
                  'same initial net outputs, different physical units', (traj, state, comp))
    constants = {}
    for key, parents in reference_targets.items():
        constants[key] = float(np.mean([
            np.mean([np.mean(states) for states in trajs.values()])
            for trajs in parents.values()]))
    # A second streaming pass checks reference means and unknown fallback for
    # every prediction, including held-out parents, with no held-out estimates.
    for row in _rows(state_path):
        fraction = min(1., max(0., float(row['remaining_fraction'])))
        bucket = min(3, max(0, math.floor(4*(1-fraction))))
        cond = (f"{row['window']}|q{bucket}", row['component'])
        glob = constants[('GLOBAL', row['component'])]
        fallback = cond not in constants
        check(_close(row['global_constant'], glob), 'fit-only global constant')
        check(_close(row['conditional_constant'], constants.get(cond, glob)), 'fit-only conditional constant')
        check(_bool(row['conditional_fallback']) == fallback, 'unknown condition fallback')
    expected_metrics = {}
    metric_keys = ('mse', 'mae', 'bias', 'mean_target', 'mean_prediction', 'global_mse', 'conditional_mse')
    for key, parents in moments.items():
        mat = np.stack([parents[p] for p in sorted(parents)])
        for aggregation in AGGREGATIONS:
            stats = mat.sum(0)[1:] / mat[:, 0].sum() if aggregation == 'state_pooled' else (mat[:, 1:] / mat[:, :1]).mean(0)
            record = dict(zip(metric_keys, map(float, stats[:7])))
            weights = mat[:, 0] if aggregation == 'state_pooled' else np.ones(len(mat))
            parent_bias_sq = float(np.average((mat[:, 3]/mat[:, 0])**2, weights=weights))
            record.update(n_states=int(mat[:, 0].sum()), n_parents=len(parents),
                          n_trajectories=len(group_trajectories[key]),
                          ratio_global=record['mse']/record['global_mse'],
                          ratio_conditional=record['mse']/record['conditional_mse'],
                          bias_squared=record['bias']**2,
                          mean_parent_bias_squared=parent_bias_sq,
                          within_parent_centered_error=max(0., record['mse']-parent_bias_sq),
                          conditional_fallback_fraction=float(stats[7]))
            expected_metrics[key+(aggregation,)] = record
    received = set()
    for row in _rows(summary_path):
        key = (row['configuration'], int(row['endpoint']), row['split'], row['window'],
               row['source_stage'], row['component'], row['aggregation'])
        check(key not in received, 'summary unique group', key)
        received.add(key)
        check(key in expected_metrics, 'summary registered group', key)
        if key in expected_metrics:
            for name, value in expected_metrics[key].items():
                check(_close(row[name], value), 'independent metric ' + name, key)
    check(received == set(expected_metrics), 'complete summary groups', [len(received), len(expected_metrics)])
    heartbeat()
    last = max(endpoint_sets['C1'])
    def metric(model, component, aggregation, size='ALL', step=last):
        return expected_metrics[(model, step, 'validation', size, 'ALL', component, aggregation)]
    value_key = ('C1', last, 'validation', 'ALL', 'ALL', 'value')
    parents = sorted(moments[value_key])
    c1 = np.stack([moments[value_key][parent] for parent in parents])
    rng = np.random.Generator(np.random.PCG64(SEED))
    draws = rng.integers(0, len(parents), size=(20000, len(parents)))
    pooled_ratio = c1[:, 1][draws].sum(1) / c1[:, 7][draws].sum(1)
    parent_ratio = (c1[:, 1]/c1[:, 0])[draws].mean(1) / (c1[:, 7]/c1[:, 0])[draws].mean(1)
    ci = {agg: dict(lower=float(np.quantile(values, .025)), upper=float(np.quantile(values, .975)))
          for agg, values in zip(AGGREGATIONS, (pooled_ratio, parent_ratio))}
    components_ok = all(metric('C1', comp, agg)['mse'] / metric('C0', comp, agg)['mse'] <= 1.10
                        for comp in COMPONENTS[3:6] for agg in AGGREGATIONS)
    value_point_ok = all(metric('C1', 'value', agg)['ratio_conditional'] <= .95 for agg in AGGREGATIONS)
    ci_ok = all(ci[agg]['upper'] < 1. for agg in AGGREGATIONS)
    c0_ok = all(metric('C1', 'value', agg)['mse'] < metric('C0', 'value', agg)['mse'] for agg in AGGREGATIONS)
    size_rows = []
    for size in SIZES:
        for agg in AGGREGATIONS:
            item = metric('C1', 'value', agg, size=size)
            size_rows.append(dict(window=size, aggregation=agg, parents=item['n_parents'],
                                  ratio_conditional=item['ratio_conditional']))
    coverage_ok = all(row['parents'] >= 3 for row in size_rows)
    sizes_ok = all(row['ratio_conditional'] <= 1 for row in size_rows)
    intermediate_stop = all(metric('C1', 'value', agg, step=128)['ratio_conditional'] > 1.25 for agg in AGGREGATIONS)
    check(not intermediate_stop or last == 128, 'registered midpoint stop obeyed')
    decision = ('INSUFFICIENT' if not coverage_ok else
                'GO' if last == 256 and all((value_point_ok, ci_ok, c0_ok, components_ok, sizes_ok)) else 'FAIL')
    gate = json.loads(gate_path.read_text(encoding='utf-8-sig'))
    reported = gate.get('status', gate.get('decision', gate.get('G3_status')))
    check(reported == decision, 'independent G3 decision', {'reported': reported, 'independent': decision})
    check(gate['endpoint'] == last, 'G3 endpoint is actual last saved endpoint')
    check(gate['midpoint_stop_triggered'] == intermediate_stop, 'G3 midpoint decision')
    leaves = {leaf['check']: leaf for leaf in gate['checks']}
    expected_leaf_names = set()
    for agg in AGGREGATIONS:
        a, b = metric('C1', 'value', agg), metric('C0', 'value', agg)
        checks_and_values = [
            ('value_vs_condition_'+agg, a['ratio_conditional'] <= .95, 'actual', a['ratio_conditional']),
            ('paired_upper_'+agg, ci[agg]['upper'] < 1., 'actual', ci[agg]['upper']),
            ('value_vs_C0_'+agg, a['mse'] < b['mse'], 'ratio', a['mse']/b['mse']),
        ]
        for comp in COMPONENTS[3:6]:
            ratio = metric('C1', comp, agg)['mse'] / metric('C0', comp, agg)['mse']
            checks_and_values.append((comp+'_vs_C0_'+agg, ratio <= 1.10, 'actual', ratio))
        for name, passed, field, value in checks_and_values:
            expected_leaf_names.add(name)
            check(leaves[name]['status'] == ('GO' if passed else 'FAIL'), 'G3 leaf status', name)
            check(_close(leaves[name][field], value), 'G3 leaf numeric value', name)
        for size in SIZES:
            name = 'size_'+size+'_'+agg
            expected_leaf_names.add(name)
            item = metric('C1', 'value', agg, size=size)
            expected_status = ('INSUFFICIENT' if item['n_parents'] < 3 else
                               'GO' if item['ratio_conditional'] <= 1 else 'FAIL')
            check(leaves[name]['status'] == expected_status, 'G3 size status', name)
            check(leaves[name]['parents'] == item['n_parents'], 'G3 size parent count', name)
            if item['n_parents'] >= 3:
                check(_close(leaves[name]['actual'], item['ratio_conditional']), 'G3 size ratio', name)
        leaf_interval = leaves['paired_upper_'+agg]['interval']
        check(_close(leaf_interval['ci95_low'], ci[agg]['lower']), 'independent paired CI lower', agg)
        check(_close(leaf_interval['ci95_high'], ci[agg]['upper']), 'independent paired CI upper', agg)
    check(set(leaves) == expected_leaf_names, 'exact G3 leaf set')
    training_counts = defaultdict(int)
    consumed = 0
    for row in _rows(training_path):
        consumed += 1
        model, step = row['configuration'], int(row['step'])
        training_counts[model] += 1
        check(model in ('C0', 'C1') and step == training_counts[model],
              'contiguous unique fitting steps', [model, step])
        check(model == ('C0' if consumed % 2 else 'C1'), 'paired update order', consumed)
        check(json.loads(row['batch_trajectory_ids']) == schedule['batches'][step-1],
              'exact common preregistered batch', [model, step])
        check(int(row['cumulative_optimizer_attempts']) == consumed,
              'training log never resets attempts', consumed)
        check(row['reduction'] == 'state_SUM_per_trajectory_then_episode_MEAN',
              'fit reduction metadata remains SUM then MEAN', [model, step])
        expected_units = 'raw_L_K_signed_H_v1' if model == 'C0' else 'per_fixed_initial_n_L_K_signed_H_v1'
        check(row['output_semantics'] == expected_units, 'training log explicit output units', [model, step])
        check(math.isfinite(float(row['loss'])) and float(row['loss']) >= 0 and
              math.isfinite(float(row['grad_norm'])) and float(row['grad_norm']) >= 0,
              'finite logged real loss/gradient', [model, step])
        check(float(row['cumulative_wall_s']) <= 1800, 'fitting log respects wall limit', [model, step])
    check(training_counts == {'C0': last, 'C1': last} and consumed == 2*last <= 512,
          'complete paired update counts', dict(training_counts))
    resource = json.loads(resource_path.read_text(encoding='utf-8-sig'))
    counts = resource['counts']
    check(counts['critic_optimizer_attempts'] == consumed, 'resource optimizer attempts agree')
    check(counts['critic_optimizer_completed'] == consumed, 'resource completed steps agree')
    check(counts['saved_trajectories'] == len(selected), 'resource saved trajectory count agrees')
    for counter in ('actor_optimizer_attempts', 'new_scenes', 'navigation_requests',
                    'teacher_requests', 'native_requests', 'H_requests',
                    'original60_reads', 'oldS3_reads'):
        check(counts[counter] == 0, 'forbidden resource remains zero', counter)
    previous_sha = '0'*64
    event_count, optimizer_debits = 0, 0
    with event_path.open('r', encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            event_count += 1
            claimed_sha = event.pop('sha256')
            actual_sha = hashlib.sha256(json.dumps(event, ensure_ascii=False, sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            check(actual_sha == claimed_sha and event['previous_sha256'] == previous_sha,
                  'durable event hash chain', event_count)
            check(event['sequence'] == event_count, 'durable event sequence', event_count)
            previous_sha = claimed_sha
            if event['event'] == 'predebit' and event.get('kind') == 'critic_optimizer_attempts':
                optimizer_debits += event['amount']
                expected_model = 'C0' if optimizer_debits % 2 else 'C1'
                check(event['amount'] == 1 and event['configuration'] == expected_model and
                      event['step'] == (optimizer_debits+1)//2,
                      'before-update debit exact paired slot', optimizer_debits)
    check(optimizer_debits == consumed, 'all real optimizer steps durably predebited')
    check(previous_sha == resource['event_sha256'] and event_count == resource['sequence'],
          'live resource points to exact current ledger tail')
    heartbeat()
    after = [_file_identity(path) for path in files]
    check(identities == after, 'input bytes unchanged during review')
    return dict(schema='r04-m1-independent-numerical-review-v1',
                status='PASS' if not failures else 'FAIL', checks=checks, failures=failures,
                implementation_independence='fresh saved-table arithmetic; no production metric/loss/model imports',
                source_identity=_file_identity(Path(__file__)), inputs=identities,
                previous_split_identity=_file_identity(old_split_path),
                state_component_rows=rows_count, summary_groups=len(received),
                fit_parents=len(parent_sets['fit']), holdout_parents=len(parent_sets['validation']),
                saved_trajectories=len(trajectories), endpoints=sorted(endpoint_sets['C1']),
                independently_verified_optimizer_steps=dict(training_counts),
                live_resource_snapshot_note='Captured while registered execution is RUNNING; later close/final hash maintenance advances resource ledger without extra optimizer steps.',
                last_endpoint=last, independent_G3=decision,
                gate_checks=dict(value_point=value_point_ok, paired_CI=ci_ok, versus_C0=c0_ok,
                                 components=components_ok, size_coverage=coverage_ok, size_error=sizes_ok),
                paired_parent_CI=ci, size_rows=size_rows, midpoint_stop=intermediate_stop,
                optimizer_attempts=0, model_forwards=0, new_requests=0,
                wall_s=time.perf_counter()-start)
