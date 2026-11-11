"""One admitted saved-only arithmetic audit for the R04 engineering review.

No top-level execution, torch import, model loading, state reconstruction,
optimizer, planner, navigation, teacher or terminal-H request occurs here.
The frozen predecessor's pure arithmetic functions are reused without calling
its writing ``recheck`` entry. All writes remain in this new R04 directory.
"""
from __future__ import annotations

from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'rc3_critic_diagnostics_20261111'
BASE_COMMIT = '853eb6b016df3aee08c233894ab6c24beb3765d8'
SOURCE_FILES = ('actor.py', 'data.py', 'critic.py', 'run.py', 'independent_recheck.py')
TABLE_FILES = (
    'ACTOR_RANK_RESPONSE.csv', 'ACTOR_CANDIDATES.csv.gz',
    'CRITIC_STATE_PREDICTIONS.csv.gz', 'CRITIC_MATCHED_CALIBRATION.csv',
    'CALIBRATION_SPLIT.json', 'TRAINING_LOG.csv',
    'CRITIC_STRATIFIED_CALIBRATION.csv',
)
AUDIT_INPUTS = tuple('rc3_critic_diagnostics_20261111/' + name
                    for name in SOURCE_FILES + TABLE_FILES
                    + ('NUMERICAL_RECHECK.json', 'PREREGISTRATION.json'))
# Frozen metadata entries at the exact user-specified base commit. The six
# principal table hashes come from the hash-pinned original numerical receipt.
PINNED_METADATA = {
    'NUMERICAL_RECHECK.json': ('56fc2d03d73b3092e80fdf56cc5bb94b8f40cbeb9b2714358b8f135668f8b19b', 245355),
    'PREREGISTRATION.json': ('3d251c3c16e0b643f072faa4eb32aff9bbd0cdf1b9beefcc853e3648a0c760ea', 174695),
    'CRITIC_STRATIFIED_CALIBRATION.csv': ('3fd5e2517fa4e265494e4afa93046ec3ad4fdab7c4f4362de560c181cf9b4310', 570088),
}
STRATIFIED_DIMS = ('configuration', 'endpoint', 'split', 'component', 'window', 'source_stage')


def _nothing(kind, amount=1, **context):
    pass


def _hash(path, charge):
    charge('source_reads', 1, path=str(path.relative_to(ROOT)), role='hash_file_read')
    charge('source_hashes', 1, path=str(path.relative_to(ROOT)))
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return {'sha256': h.hexdigest(), 'bytes': path.stat().st_size}


def _read_json(path, charge):
    charge('source_reads', 1, path=str(path.relative_to(ROOT)), role='json_parse')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _new_json(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(data)


def _stratified(legacy, check, charge):
    """Independent streaming moments for every window/phase summary row."""
    # Counts, sum(error), sum(error²), sum(abs(error)), sum(target), sum(pred),
    # sum(global-error²), sum(conditional-error²), sum(fallback) per parent.
    groups = defaultdict(dict)
    trajectories = defaultdict(set)
    source = OLD / 'CRITIC_STATE_PREDICTIONS.csv.gz'
    charge('source_reads', 1, path=str(source.relative_to(ROOT)), role='stratified_prediction_stream')
    prediction_rows = 0
    for row in legacy._rows(source):
        identity = tuple(row[key] for key in STRATIFIED_DIMS)
        parent = row['parent_id']
        values = groups[identity].setdefault(parent, [0., 0., 0., 0., 0., 0., 0., 0., 0.])
        y, pred = float(row['target']), float(row['prediction'])
        error = pred - y
        global_error, conditional_error = float(row['global_constant']) - y, float(row['conditional_constant']) - y
        increments = (1., error, error * error, abs(error), y, pred,
                      global_error * global_error, conditional_error * conditional_error,
                      float(legacy._bool(row['conditional_fallback'])))
        for index, value in enumerate(increments):
            values[index] += value
        trajectories[identity].add(row['trajectory_id'])
        prediction_rows += 1
        if prediction_rows % 10000 == 0:
            charge('heartbeat', 0, phase='stratified_prediction_stream', rows=prediction_rows)
    summary_file = OLD / 'CRITIC_STRATIFIED_CALIBRATION.csv'
    charge('source_reads', 1, path=str(summary_file.relative_to(ROOT)), role='stratified_summary_parse')
    markers, summary_rows = set(), 0
    for row in legacy._rows(summary_file):
        identity = tuple(row[key] for key in STRATIFIED_DIMS)
        aggregation = row['aggregation']
        marker = identity + (aggregation,)
        check.add('stratified.unique_summary_key', marker not in markers, context=marker)
        markers.add(marker)
        by_parent = groups.get(identity)
        check.add('stratified.summary_has_exact_prediction_group', bool(by_parent), context=marker)
        if not by_parent:
            continue
        values = list(by_parent.values())
        total = sum(v[0] for v in values)
        counts = [v[0] for v in values]
        moments = [[v[i] / v[0] for i in range(1, 9)] for v in values]
        if aggregation == 'state_pooled':
            mean = [sum(v[i] for v in values) / total for i in range(1, 9)]
            parent_bias_squared = sum(m[0] ** 2 * n for m, n in zip(moments, counts)) / total
        elif aggregation == 'parent_equal':
            mean = [sum(m[i] for m in moments) / len(moments) for i in range(8)]
            parent_bias_squared = sum(m[0] ** 2 for m in moments) / len(moments)
        else:
            check.add('stratified.only_registered_aggregations', False, context=marker, actual=aggregation)
            continue
        bias, mse, mae, target_mean, prediction_mean, global_mse, conditional_mse, fallback = mean
        expected = dict(
            parents=len(values), trajectories=len(trajectories[identity]), states=int(total),
            mse=mse, bias=bias, mae=mae, target_mean=target_mean, prediction_mean=prediction_mean,
            global_constant_mse=global_mse, conditional_constant_mse=conditional_mse,
            conditional_fallback_fraction=fallback, rmse=math.sqrt(mse), bias_squared=bias * bias,
            centered_error_variance=max(0., mse - bias * bias),
            mean_parent_bias_squared=parent_bias_squared,
            mean_within_parent_centered_error=max(0., mse - parent_bias_squared),
            bias_squared_fraction_of_mse=(bias * bias / mse) if mse else 0.,
            mse_over_global_constant=(mse / global_mse) if global_mse > 0 else None,
            mse_over_conditional_constant=(mse / conditional_mse) if conditional_mse > 0 else None,
        )
        for key, value in expected.items():
            check.equal('stratified.' + key, row.get(key), value, context=marker)
        summary_rows += 1
    required = {identity + (aggregation,) for identity in groups
                for aggregation in ('state_pooled', 'parent_equal')}
    check.add('stratified.complete_group_and_aggregation_coverage', markers == required,
              actual=len(markers), expected=len(required))
    return dict(prediction_component_rows=prediction_rows, summary_rows=summary_rows,
                groups=len(groups), full_stream=True,
                independent_formula='Per-parent sums from all prediction rows; pooled or equal-parent moment aggregation')


def audit(outfolder, charge=None):
    """Execute exactly one registered saved-only audit and write only new files.

    Root owns authorization/resource registration. This function does not run
    implicitly. A start marker prevents free replay after partial failure.
    ``charge`` is optional only for external use; the owning R04 runner supplies
    its durable engineering counter callback.
    """
    charge = charge or _nothing
    start = time.perf_counter()
    folder = Path(outfolder).resolve()
    if not folder.is_relative_to(REVIEW_ROOT) or folder == OLD:
        raise ValueError('Audit output must stay inside the new R04 review directory')
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'EVIDENCE_AUDIT.json'
    marker = folder / 'EVIDENCE_AUDIT_STARTED.json'
    if output.exists() or marker.exists():
        raise RuntimeError('The admitted saved-evidence audit was already attempted; no replay')
    _new_json(marker, dict(started_utc=datetime.now(timezone.utc).isoformat(),
                           base_commit=BASE_COMMIT, scope='Saved arithmetic only; no model/task calls'))
    result = dict(schema='rc3-r04-saved-evidence-audit-v1', base_commit=BASE_COMMIT,
                  scope='Engineering saved-output audit; no scientific training or fresh performance evaluation',
                  old_receipt_overwritten=False, input_files={}, source_files={},
                  optimizer_updates=0, model_loads=0, model_forwards=0,
                  new_scene_navigation_teacher_native_H_requests=0,
                  original60_or_S3_semantic_reads=0,
                  inherited_arithmetic_implementation='Hash-pinned predecessor independent_recheck pure functions; not a second independent implementation',
                  additional_arithmetic_implementation='New streaming window/phase aggregation, no predecessor analysis helper imports')
    # Importing this module imports only stdlib/NumPy, never torch or model code.
    from rc3_critic_diagnostics_20261111 import independent_recheck as legacy
    class WatchedChecks(legacy.Checks):
        def __init__(self):
            super().__init__()
            self._add_calls = 0

        def add(self, name, condition, *, context=None, actual=None, expected=None):
            self._add_calls += 1
            if self._add_calls % 100000 == 0:
                charge('heartbeat', 0, phase='saved_arithmetic_checks', checks=self._add_calls)
            return super().add(name, condition, context=context, actual=actual, expected=expected)

    checks = WatchedChecks()
    try:
        for name, (sha, count) in PINNED_METADATA.items():
            actual = _hash(OLD / name, charge)
            result['input_files'][name] = actual
            checks.add('identity.' + name, actual == {'sha256': sha, 'bytes': count},
                       actual=actual, expected={'sha256': sha, 'bytes': count})
            if actual != {'sha256': sha, 'bytes': count}:
                raise ValueError('Frozen evidence metadata differs: ' + name)
        old_receipt = _read_json(OLD / 'NUMERICAL_RECHECK.json', charge)
        old_registration = _read_json(OLD / 'PREREGISTRATION.json', charge)
        checks.equal('historical.status_PASS', old_receipt['status'], 'PASS')
        checks.equal('historical.check_count_preserved', old_receipt['check_count'], 41591999)
        for name in SOURCE_FILES:
            actual = _hash(OLD / name, charge)
            source_key = 'rc3_critic_diagnostics_20261111/' + name
            expected = old_registration['source_sha256'][source_key]
            result['source_files'][name] = actual
            checks.equal('source.' + name + '_registered_hash', actual['sha256'], expected)
            if actual['sha256'] != expected:
                raise ValueError('Frozen executable arithmetic source differs: ' + name)
        for name, expected in old_receipt['input_files'].items():
            actual = _hash(OLD / name, charge)
            result['input_files'][name] = actual
            checks.add('identity.original_receipt.' + name, actual == expected,
                       actual=actual, expected=expected)
            if actual != expected:
                raise ValueError('Saved table differs from original numerical receipt: ' + name)
        split = _read_json(OLD / 'CALIBRATION_SPLIT.json', charge)
        # Each underlying reader has exactly one full scan for the declared
        # files: Actor response/candidates, Critic prediction/summary, training.
        charge('source_reads', 5, role='legacy_full_saved_table_scans')
        result['actor'] = legacy._actor(OLD, split, checks)
        result['critic'] = legacy._critic(OLD, split, checks)
        result['training'] = legacy._training(OLD, split, checks)
        result['stratified'] = _stratified(legacy, checks, charge)
        charge('heartbeat', 0, phase='saved_evidence_final_wall_check')
    except Exception as error:
        checks.add('audit.completed_without_parse_or_schema_error', False,
                   actual=type(error).__name__ + ': ' + str(error), expected='complete arithmetic audit')
        result['error'] = dict(type=type(error).__name__, message=str(error))
    total = sum(group['checks'] for group in checks.groups.values())
    failures = sum(group['failures'] for group in checks.groups.values())
    result.update(checks=checks.groups, check_count=total, pass_count=total-failures,
                  fail_count=failures, status='PASS' if failures == 0 else 'FAIL',
                  wall_seconds=time.perf_counter()-start,
                  statement='A PASS supports saved arithmetic/provenance only, never route improvement or scientific significance')
    result = legacy._json_safe(result)
    _new_json(output, result)
    charge('timing', 0, category='saved_evidence_audit', seconds=time.perf_counter()-start)
    return result
