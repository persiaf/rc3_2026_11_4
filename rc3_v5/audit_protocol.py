"""Read-only protocol/statistic audit of the frozen 400 ms MLP supplement.

Complements audit_supplement.py (geometry/cost certification); does not rerun
planners, change experiments, or measure performance. Timing values can only be
checked for internal consistency, not independently remeasured from saved data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

import numpy as np


SPECS = [dict(id='detour_wall400', method='detour', policy=None, kind='wall',
              horizon=256, seconds=.4, repeats=3),
         dict(id='v3_mlp_wall400', method='learned', policy='v3_mlp', kind='wall',
              horizon=256, seconds=.4, repeats=3)]
CI = dict(method='percentile paired scene bootstrap', samples=20000, seed=731203,
          sorting=['seed', 'nx', 'ny', 'mode'], training_seed_uncertainty=False,
          repeated_runs_treated_as_independent=False, multiple_comparison_adjustment=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative_path(value):
    """Accept Windows/POSIX relative manifest paths without accepting escape paths."""
    path = PurePosixPath(value.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or not path.parts or ':' in str(path):
        raise ValueError('unsafe relative source path: '+value)
    return str(path)


class Checks:
    def __init__(self):
        self.errors = []
        self.count = 0

    def require(self, condition, label):
        self.count += 1
        if not condition:
            self.errors.append(label)

    def same(self, actual, expected, label, tolerance=1e-7):
        if type(expected) is float:
            self.require(type(actual) in (int, float) and math.isfinite(actual)
                         and math.isclose(actual, expected, rel_tol=0., abs_tol=tolerance), label)
        elif isinstance(expected, dict):
            self.require(isinstance(actual, dict) and set(actual) == set(expected), label+':keys')
            if isinstance(actual, dict):
                for key, value in expected.items():
                    self.same(actual.get(key), value, label+'.'+key, tolerance)
        elif isinstance(expected, list):
            self.require(isinstance(actual, list) and len(actual) == len(expected), label+':length')
            if isinstance(actual, list):
                for index, (a, b) in enumerate(zip(actual, expected)):
                    self.same(a, b, label+f'[{index}]', tolerance)
        else:
            self.require(type(actual) is type(expected) and actual == expected, label)

    def nonnegative(self, value, label):
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
        self.require(valid, label)
        return valid


def check_record(record, spec, repeat, order, plan, reference, checks, label):
    checks.same(record['config'], spec, label+':config', tolerance=1e-12)
    checks.same(record['repeat'], repeat, label+':repeat')
    checks.same(record['execution_order'], order, label+':execution_order')
    checks.same(record['status'], 'complete', label+':status')
    checks.same(plan['status'], 'complete', label+':plan status')
    checks.same(record['bound_against_revalidated_saved_fixed3'], True, label+':saved Fixed3 bound')
    checks.same(plan['stop_penalty_equivalent_m'], .75, label+':penalty', tolerance=1e-12)
    objective = sum(math.dist(a, b) for a, b in zip(plan['route_polyline'], plan['route_polyline'][1:]))
    objective += .75*len(plan['selected_candidate_ids'])
    checks.same(record['objective_m'], objective, label+':objective')
    checks.same(plan['objective_length_plus_stops_m'], objective, label+':plan objective')
    checks.same(record['reference_objective_m'], reference, label+':reference')
    checks.require(objective <= reference+1e-7, label+':cost bound')
    inner = record['runtime']
    for key in ('planning_s', 'inner_planning_s', 'overrun_s'):
        checks.nonnegative(record[key], label+':'+key)
    checks.same(record['overrun_s'], max(0., record['planning_s']-.4), label+':overrun', tolerance=1e-9)
    checks.same(record['deadline_exceeded'], record['planning_s'] > .4, label+':deadline flag')
    checks.require(record['planning_s']+1e-9 >= record['inner_planning_s'], label+':outer/inner time ordering')
    checks.same(record['inner_planning_s'], inner['planning_s'], label+':inner time', tolerance=1e-9)
    checks.same(inner['elapsed_s'], inner['planning_s'], label+':runtime elapsed', tolerance=1e-9)
    checks.same(inner['time_limit_s'], .4, label+':runtime budget', tolerance=1e-12)
    checks.same(inner['time_overrun_s'], max(0., inner['planning_s']-.4), label+':runtime overrun', tolerance=1e-9)
    checks.same(inner['deadline_exceeded'], inner['planning_s'] > .4, label+':runtime deadline flag')
    for key in ('reference_s', 'feature_s', 'model_s', 'repair_s', 'final_validation_s', 'thread_session_s'):
        checks.nonnegative(inner[key], label+':runtime '+key)
    checks.require(sum(inner[k] for k in ('reference_s', 'feature_s', 'model_s', 'repair_s',
                    'final_validation_s', 'thread_session_s')) <= inner['planning_s']+1e-7,
                   label+':runtime component sum')
    for key in ('dp_calls', 'accepted'):
        checks.require(type(record[key]) is int and 0 <= record[key] <= spec['horizon'], label+':'+key)
        checks.same(inner[key], record[key], label+':runtime '+key)
    checks.require(record['accepted'] <= record['dp_calls'], label+':accepted/calls')
    checks.same(inner['attempts'], record['dp_calls'], label+':attempts')
    checks.same(inner['provisional_improvements'], record['accepted'], label+':improvements')
    for key in ('objective_m', 'reference_objective_m', 'fallback_used', 'stop_reason', 'status'):
        checks.same(inner[key], record[key], label+':runtime '+key)
    checks.same(inner['method'], spec['method'], label+':runtime method')
    checks.same(inner['reference'], 'fixed3', label+':runtime reference')
    checks.same(inner['cost_nonincreasing'], True, label+':runtime cost bound')
    checks.same(record['fallback_used'], bool(inner['failures']), label+':fallback/failures')
    checks.same(inner['budget_exhausted'], inner['stop_reason'] in ('deadline', 'horizon'), label+':budget exhaustion')
    checks.require(inner['stop_reason'] in ('deadline', 'horizon', 'policy_stop', 'no_legal_action',
                                          'proposal_failure', 'reference_only'), label+':stop reason')
    return dict(method=spec['id'], cost=objective, reference=reference, time=record['planning_s'],
                calls=record['dp_calls'], overrun=record['overrun_s'],
                exceeded=record['deadline_exceeded'], fallback=record['fallback_used'], stop=record['stop_reason'])


def rebuild_summary(rows, scene_rows):
    methods = {}
    for spec in SPECS:
        ss = [r for r in scene_rows if r['method'] == spec['id']]
        rr = [r for r in rows if r['method'] == spec['id']]
        times = [r['time']*1000 for r in rr]
        methods[spec['id']] = dict(scenes=len(ss), runs=len(rr),
            mean_objective_m=float(np.mean([s['cost'] for s in ss])),
            mean_reduction_vs_fixed3_pct=float(np.mean([100*(s['reference']-s['cost'])/s['reference'] for s in ss])),
            median_scene_planning_ms=float(np.median([s['time_ms'] for s in ss])),
            p95_scene_planning_ms=float(np.quantile([s['time_ms'] for s in ss], .95)),
            p95_all_run_planning_ms=float(np.quantile(times, .95)), max_run_planning_ms=max(times),
            mean_dp_calls=float(np.mean([r['calls'] for r in rr])), median_dp_calls=float(np.median([r['calls'] for r in rr])),
            deadline_exceeded_runs=sum(r['exceeded'] for r in rr),
            max_overrun_ms=max(r['overrun']*1000 for r in rr),
            mean_overrun_ms=float(np.mean([r['overrun']*1000 for r in rr])),
            fallback_runs=sum(r['fallback'] for r in rr), stop_reasons=dict(Counter(r['stop'] for r in rr)))
    lookup = {(r['key'], r['method']): r for r in scene_rows}
    differences, ratios = [], []
    for key in sorted({r['key'] for r in scene_rows}):
        a, b = lookup[key, 'v3_mlp_wall400'], lookup[key, 'detour_wall400']
        differences.append(100*(b['cost']-a['cost'])/b['cost'])
        ratios.append(a['time_ms']/b['time_ms'])
    values = np.asarray(differences)
    indices = np.random.default_rng(731203).integers(0, len(values), size=(20000, len(values)))
    paired = dict(scenes=len(values), mean_reduction_pct=float(values.mean()),
        ci95_pct=np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist(),
        wins=int((values>1e-9).sum()), ties=int((np.abs(values)<=1e-9).sum()), losses=int((values < -1e-9).sum()),
        median_paired_time_ratio=float(np.median(ratios)),
        ratio_of_median_times=methods['v3_mlp_wall400']['median_scene_planning_ms']/methods['detour_wall400']['median_scene_planning_ms'])
    return methods, paired


def audit(folder, root):
    folder, root = Path(folder), Path(root)
    checks = Checks()
    manifest, summary = read(folder/'manifest.json'), read(folder/'summary.json')
    all_cases = read(root/'rc3_v4/data/test_cases.json')
    expected_cases = [dict(original_case_index=i, **c) for i, c in enumerate(all_cases) if c['nx']*c['ny'] == 100]
    checks.same(len(all_cases), 39, 'original case count')
    checks.same(len(expected_cases), 18, 'supplement case count')
    checks.same(manifest['cases'], expected_cases, 'manifest cases')
    checks.same(manifest['configurations'], SPECS, 'manifest configurations', tolerance=1e-12)
    checks.same(manifest['ci'], CI, 'manifest CI')
    for key, value in dict(status='complete', repeats=3, planned_runs=108, attempted_runs=108,
                           saved_plans=108, no_training_or_tuning=True, original_adoption_gate_unchanged=True,
                           changed_frozen_files=[]).items():
        checks.same(manifest[key], value, 'manifest '+key)
    checks.same(manifest['script_sha256'], sha(root/'rc3_v5/budget_supplement.py'), 'frozen experiment script')
    checks.same(manifest['model_sha256'], sha(root/'rc3_v3/deployment/policy.npz'), 'frozen model')
    hashes = {relative_path(k): v for k, v in manifest['frozen_source_sha256'].items()}
    checks.same(len(hashes), len(manifest['frozen_source_sha256']), 'unique normalized source paths')
    for path, expected in hashes.items():
        checks.same(sha(root/path), expected, 'source hash '+path)
    checks.same(manifest['original_platform'], read(root/'rc3_v4/results/manifest.json')['platform'], 'original platform')
    expected_files = {f'scene_plan_{c["original_case_index"]:03d}_{c["seed"]}.json' for c in expected_cases}
    checks.require({p.name for p in (folder/'plans').glob('*.json')} == expected_files, 'exact saved scene inventory')
    rows, scene_rows, tamper_checks = [], [], []
    for case in expected_cases:
        name = f'scene_plan_{case["original_case_index"]:03d}_{case["seed"]}.json'
        source_path = 'rc3_v4/results/plans/'+name
        data, original = read(folder/'plans'/name), read(root/source_path)
        checks.same(relative_path(data['source_plan']), source_path, name+':source path')
        checks.same(data['source_plan_sha256'], hashes[source_path], name+':source hash binding')
        checks.require(data['scene'] == original['scene'], name+':exact original scene')
        checks.require(data['metadata'] == original['metadata'], name+':exact original metadata')
        checks.require(data['feasibility'] == original['feasibility'], name+':original sparse reachability')
        for key in ('seed', 'nx', 'ny', 'mode'):
            checks.same(data['metadata'][key], case[key], name+':case '+key)
        checks.same(data['metadata']['case_index'], case['original_case_index'], name+':case index')
        checks.same(data['status'], 'complete', name+':status')
        checks.nonnegative(data['preprocessing_s'], name+':preprocessing')
        fixed_labels = sorted(k for k, v in original['runs'].items() if v['config']['id'] == 'fixed3')
        ref = original['plans'][fixed_labels[0]]
        checks.require(data['fixed3_reference_plan'] == ref, name+':exact saved Fixed3 reference')
        checks.same(ref['stop_penalty_equivalent_m'], .75, name+':reference penalty', tolerance=1e-12)
        reference = sum(math.dist(a, b) for a, b in zip(ref['route_polyline'], ref['route_polyline'][1:]))+.75*len(ref['selected_candidate_ids'])
        labels = [f'{s["id"]}_rep{r}' for s in SPECS for r in range(3)]
        scheduled = [labels[int(i)] for i in np.random.default_rng(case['seed']+753911).permutation(6)]
        checks.same(data['execution_order'], scheduled, name+':randomized schedule')
        checks.require(set(data['runs']) == set(labels) == set(data['plans']), name+':exact run and plan labels')
        current = []
        for spec in SPECS:
            for repeat in range(3):
                label = f'{spec["id"]}_rep{repeat}'
                record, plan = data['runs'][label], data['plans'][label]
                row = check_record(record, spec, repeat, scheduled.index(label), plan, reference, checks, name+':'+label)
                current.append(row)
                if not tamper_checks:
                    # No disk mutation and no planner rerun: prove the new guards reject
                    # four independent corruptions of a genuine saved record.
                    for field, value in [('planning_s', float('nan')), ('overrun_s', -1.),
                                         ('repeat', 99), ('config', {**spec, 'seconds': .8})]:
                        corrupt, trial = copy.deepcopy(record), Checks()
                        corrupt[field] = value
                        check_record(corrupt, spec, repeat, scheduled.index(label), plan, reference, trial, 'tamper '+field)
                        tamper_checks.append(dict(field=field, rejected=bool(trial.errors), errors=trial.errors))
        rows.extend(current)
        for spec in SPECS:
            rs = [r for r in current if r['method'] == spec['id']]
            scene_rows.append(dict(key=tuple(case[k] for k in ('seed', 'nx', 'ny', 'mode')), method=spec['id'],
                                   cost=float(np.mean([r['cost'] for r in rs])), reference=reference,
                                   time_ms=float(np.median([r['time'] for r in rs]))*1000))
    methods, paired = rebuild_summary(rows, scene_rows)
    checks.same(summary['methods'], methods, 'all method summary statistics')
    checks.same(summary['paired_mlp_vs_detour'], paired, 'all paired summary statistics')
    checks.same(summary['bootstrap'], CI, 'summary bootstrap')
    for key, value in dict(status='complete', scenes=18, runs=108, valid_runs=108, fixed3_bounds=108,
                           changed_frozen_files=[], all_passed=True).items():
        checks.same(summary[key], value, 'summary '+key)
    checks.same(summary['platform'], manifest['platform'], 'summary platform')
    checks.same(summary['inference_scope'], manifest['inference_scope'], 'summary inference scope')
    checks.nonnegative(summary['elapsed_s'], 'summary loop elapsed')
    checks.same(summary['elapsed_s'], manifest['elapsed_s'], 'loop elapsed consistency')
    checks.require(all(t['rejected'] for t in tamper_checks), 'four corruption probes')
    return dict(all_passed=not checks.errors, errors=checks.errors, checks=checks.count,
        scenes=len(expected_cases), runs=len(rows), methods=methods, paired=paired,
        platform=manifest['platform'], model_sha256=manifest['model_sha256'],
        auditor_sha256=sha(Path(__file__)), budget_script_sha256=manifest['script_sha256'],
        tamper_checks=tamper_checks, experiment_modified=False, planner_rerun=False,
        scope='Read-only source/scene/config/order/timing consistency and complete summary reconstruction. '
              'Complements geometric audit; saved clocks are not independent hardware measurements. '
              'Post-v4 descriptive evidence, with platform-separated scene-level inference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.input, args.root)
    except Exception as error:
        result = dict(all_passed=False, errors=[f'{type(error).__name__}: {error}'], experiment_modified=False, planner_rerun=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('methods', 'paired', 'tamper_checks')}, ensure_ascii=False))
    return 0 if result['all_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
