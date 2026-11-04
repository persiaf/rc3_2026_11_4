"""Read-only G2.3 result audit using only the Python standard library.

Reads saved JSON plans, results, protocol and validation receipts. Does not import
project code, deserialize models/pickles, generate scenes, plan, train or time a
planner. The confidence-interval check enumerates all 6**6 scene resamples;
it does not reproduce NumPy's 10,000 random draws.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from collections import Counter
from pathlib import Path


TOL = 1e-9


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def method_class(method, seeds):
    for seed in seeds:
        prefix = str(seed) + '_'
        if method.startswith(prefix):
            return method[len(prefix):]
    return method


def audit(source):
    failures = []

    def check(condition, label):
        if not condition:
            failures.append(label)

    def near(a, b, label, tol=1e-7):
        check(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol, label)

    out = source / 'outputs'
    protocol = read(out / 'protocol.json')
    rows = read(out / 'test_results.json')
    published = read(out / 'analysis.json')
    published_classes = read(out / 'class_summary.json')
    validation_receipt = read(out / 'plan_validation.json')
    seeds = protocol['seeds']
    cases = protocol['splits']['test']
    case_ids = [case['case_id'] for case in cases]
    repetitions = range(protocol['repetitions'])
    methods = sorted({row['method'] for row in rows})
    index = {(row['case_id'], row['rep'], row['method']): row for row in rows}
    plan_receipts = {row['file']: row for row in validation_receipt}
    check(len(rows) == 180 and len(methods) == 15, '180 records and 15 method instances')
    check(len(cases) == 6 and len(seeds) == 3 and len(repetitions) == 2, '6 scenes, 3 seeds, 2 repetitions')
    check(len(index) == len(rows), 'no duplicate scene/repetition/method records')
    check(set(index) == set(itertools.product(case_ids, repetitions, methods)), 'complete registered factorial')
    check(len(plan_receipts) == 180, '180 unique plan verification receipts')
    all_split_seeds = [case['seed'] for split in protocol['splits'].values() for case in split]
    check(len(all_split_seeds) == len(set(all_split_seeds)), 'G2.3 train/calibration/selection/test seeds disjoint')
    trial_text = (source / 'trial.py').read_text(encoding='utf-8')
    check("cases=p['splits']['train'][i*4:(i+1)*4]" in trial_text,
          'registered execution assigns a distinct consecutive block of four scenes to each seed')
    training_sets = {}
    for position, seed in enumerate(seeds):
        own_cases = protocol['splits']['train'][position * 4:(position + 1) * 4]
        training_sets[str(seed)] = {
            'case_ids': [case['case_id'] for case in own_cases],
            'scene_seeds': [case['seed'] for case in own_cases],
            'mode_counts': dict(Counter(case['mode'] for case in own_cases)),
        }
        check(len(own_cases) == 4, 'four distinct training scenes for seed ' + str(seed))
    check(sha(out / 'protocol.json') == (out / 'protocol.sha256').read_text().strip(), 'protocol content hash')
    for relative, expected in protocol['source_sha256'].items():
        check(sha(source / relative) == expected, 'registered source hash: ' + relative)

    anchor_task_ids = {}
    for case_id in case_ids:
        plan = read(out / 'plans' / f'{case_id}_0_original_fixed3.json')
        anchor_task_ids[case_id] = {item['task_id'] for item in plan['assignments']}

    checked_plans = []
    scalar_rows = []
    for row in rows:
        identity = f"{row['case_id']}_{row['rep']}_{row['method']}"
        registered = next(case for case in cases if case['case_id'] == row['case_id'])
        check(row['seed'] == registered['seed'] and row['mode'] == registered['mode'], identity + ': registered scene')
        path = out / 'plans' / (identity + '.json')
        plan = read(path)
        points = plan['route_polyline']
        length = math.fsum(math.dist(a, b) for a, b in zip(points, points[1:]))
        stops = plan['service_stops']
        count = len(stops)
        cost = length + 0.75 * count
        near(row['L'], length, identity + ': polyline L')
        near(plan['route_length_m'], length, identity + ': recorded plan L')
        near(math.fsum(plan['route_leg_costs_m']), length, identity + ': sum of route legs')
        near(row['J'], cost, identity + ': L+0.75K')
        near(plan['objective_length_plus_stops_m'], cost, identity + ': recorded plan J')
        near(plan['stop_penalty_equivalent_m'], 0.75, identity + ': stop penalty')
        check(row['K'] == count, identity + ': K')
        check(len(plan['route_leg_costs_m']) == count + 1, identity + ': round-trip leg count')
        assigned = [(a['task_id'], a['candidate_id'], a['arm']) for a in plan['assignments']]
        from_stops = [(t['task_id'], s['candidate_id'], t['arm']) for s in stops for t in s['tasks']]
        task_ids = [a[0] for a in assigned]
        stop_ids = [s['candidate_id'] for s in stops]
        check(sorted(assigned) == sorted(from_stops), identity + ': assignment/stop agreement')
        check(len(task_ids) == len(set(task_ids)), identity + ': each listed task assigned once')
        check(set(task_ids) == anchor_task_ids[row['case_id']], identity + ': task set equals saved Fixed3 anchor')
        check(all(a[2] in ('L', 'R') for a in assigned), identity + ': arm identifiers')
        check(plan['visit_order_candidate_ids'] == stop_ids, identity + ': visit order')
        check(set(plan['selected_candidate_ids']) == set(stop_ids), identity + ': selected candidate set')
        check(len(stop_ids) == len(set(stop_ids)), identity + ': unique service candidates')
        check([s['sequence'] for s in stops] == list(range(1, count + 1)), identity + ': stop sequence')
        check(row['valid'] and plan['validation']['valid_model_plan'], identity + ': saved valid flags')
        check(plan['validation']['all_tasks_served'], identity + ': saved coverage certificate')
        check(not plan['unserved_task_ids'] and not plan['validation']['errors'], identity + ': no saved coverage/errors')
        check(plan['validation']['collision_count'] == 0, identity + ': saved collision certificate')
        near(plan['validation']['independent_polyline_length_m'], length, identity + ': certificate length')
        receipt = plan_receipts.get(path.name)
        check(receipt is not None and receipt['valid'] and receipt['sha256'] == sha(path), identity + ': validation receipt/hash')
        anchor = index[(row['case_id'], row['rep'], 'original_fixed3')]['J']
        gain = 100 * (anchor - row['J']) / anchor
        check(row['J'] <= anchor + 1e-7, identity + ': Fixed3 bound')
        check(row['overrun'] == (row['complete_s'] > protocol['complete_soft_s']), identity + ': soft deadline flag')
        if row['return_'] is not None:
            near(row['return_'], gain - row['work'] / 1e6, identity + ': final reward')
        scalar_rows.append({**row, 'method_class': method_class(row['method'], seeds), 'gain_vs_fixed3_percent': gain})
        checked_plans.append({'file': path.name, 'sha256': sha(path), 'polyline_length_m': length,
                              'stops': count, 'J': cost, 'assigned_tasks': len(task_ids)})

    def summarize(group):
        times = [row['complete_s'] * 1000 for row in group]
        return {
            'n': len(group), 'mean_J': statistics.mean(row['J'] for row in group),
            'mean_gain_percent': statistics.mean(row['gain_vs_fixed3_percent'] for row in group),
            'median_complete_ms': statistics.median(times), 'mean_complete_ms': statistics.mean(times),
            'p95_ms': percentile(times, 0.95), 'overruns': sum(row['overrun'] for row in group),
            'calls': sum(row['calls'] for row in group), 'fallbacks': sum(row['fallback'] for row in group),
            'termination_reasons': dict(Counter(row['reason'] for row in group)),
            'work_budget_overruns': sum(row['work'] is not None and row['work'] > protocol['config']['work_budget'] for row in group),
        }

    summaries = {}
    for method in methods:
        summary = summarize([row for row in scalar_rows if row['method'] == method])
        summaries[method] = summary
        for key in ('n', 'mean_J', 'mean_gain_percent', 'median_complete_ms', 'p95_ms', 'overruns', 'calls', 'fallbacks'):
            near(summary[key], published['summary'][method][key], 'published method summary: ' + method + '/' + key)

    classes = {}
    for name in sorted({row['method_class'] for row in scalar_rows}):
        summary = summarize([row for row in scalar_rows if row['method_class'] == name])
        classes[name] = summary
        for own, theirs in [('mean_gain_percent', 'mean_gain'), ('median_complete_ms', 'median_ms'), ('overruns', 'overrun'), ('n', 'n')]:
            near(summary[own], published_classes[name][theirs], 'published class summary: ' + name + '/' + own)

    comparisons = {}
    for comparator in protocol['main_gate']['comparators']:
        scene_values = []
        by_seed = {str(seed): [] for seed in seeds}
        for case_id in case_ids:
            paired = []
            for seed in seeds:
                reference = comparator if comparator in ('same_interface_rule', 'original_detour16') else f'{seed}_{comparator}'
                for rep in repetitions:
                    base = index[(case_id, rep, reference)]['J']
                    proposed = index[(case_id, rep, f'{seed}_multi_policy_iteration')]['J']
                    gain = 100 * (base - proposed) / base
                    paired.append(gain)
                    by_seed[str(seed)].append(gain)
            scene_values.append(statistics.mean(paired))
        values = scene_values
        n = len(values)
        resampled_means = [math.fsum(values[i] for i in sample) / n
                           for sample in itertools.product(range(n), repeat=n)]
        exact_ci = [percentile(resampled_means, prob) for prob in (0.025, 0.975)]
        average = statistics.mean(values)
        registered_pass = average >= protocol['main_gate']['mean_gain_percent'] and published['comparisons'][comparator]['ci95'][0] > 0
        exact_pass = average >= protocol['main_gate']['mean_gain_percent'] and exact_ci[0] > 0
        recorded = published['comparisons'][comparator]
        for i, value in enumerate(values):
            near(value, recorded['scene_values'][i], comparator + ': scene paired gain', tol=1e-12)
        near(average, recorded['mean_gain_percent'], comparator + ': paired mean', tol=1e-12)
        check(recorded['passed'] == registered_pass == exact_pass, comparator + ': robust quality gate')
        comparisons[comparator] = {
            'scene_values_percent': values, 'mean_gain_percent': average,
            'recorded_10000_bootstrap_ci95_percent': recorded['ci95'],
            'independent_exhaustive_bootstrap_ci95_percent': exact_ci,
            'exhaustive_resamples': len(resampled_means),
            'per_seed_mean_gain_percent': {seed: statistics.mean(gains) for seed, gains in by_seed.items()},
            'scene_win_tie_loss': {'wins': sum(v > TOL for v in values),
                                  'ties': sum(abs(v) <= TOL for v in values),
                                  'losses': sum(v < -TOL for v in values)},
            'recorded_quality_passed': registered_pass, 'exhaustive_quality_passed': exact_pass,
            'statistical_unit': 'Six scenes after averaging three trained policies and two repetitions within each scene.',
        }

    proposed_rows = [row for row in scalar_rows if row['method_class'] == 'multi_policy_iteration']
    joint_counts = {'improved_and_on_time': 0, 'improved_and_late': 0, 'no_improvement_and_on_time': 0, 'no_improvement_and_late': 0}
    for row in proposed_rows:
        anchor = index[(row['case_id'], row['rep'], 'original_fixed3')]['J']
        improved = row['J'] < anchor - TOL
        on_time = row['complete_s'] <= protocol['complete_soft_s']
        name = ('improved' if improved else 'no_improvement') + '_and_' + ('on_time' if on_time else 'late')
        joint_counts[name] += 1
    on_time = [row for row in proposed_rows if row['complete_s'] <= protocol['complete_soft_s']]
    joint = {
        'n': len(proposed_rows), 'soft_budget_s': protocol['complete_soft_s'],
        'strict_J_tolerance': TOL, 'counts': joint_counts,
        'on_time_n': len(on_time), 'late_n': sum(row['overrun'] for row in proposed_rows),
        'on_time_all_equal_fixed3': all(abs(row['J'] - index[(row['case_id'], row['rep'], 'original_fixed3')]['J']) <= TOL for row in on_time),
        'scope': 'Joint descriptive distribution of all 36 saved runs; no sample exclusion, new acceptance rule or re-scoring.',
    }

    quality_changes = []
    if len(repetitions) == 2:
        for case_id, method in itertools.product(case_ids, methods):
            a, b = index[(case_id, 0, method)], index[(case_id, 1, method)]
            if abs(a['J'] - b['J']) > TOL:
                quality_changes.append({'case_id': case_id, 'method': method, 'J_rep0': a['J'], 'J_rep1': b['J'],
                                        'calls_rep0': a['calls'], 'calls_rep1': b['calls'],
                                        'reason_rep0': a['reason'], 'reason_rep1': b['reason']})
    total_overruns = sum(row['overrun'] for row in rows)
    quality_passed = all(item['recorded_quality_passed'] for item in comparisons.values())
    primary_passed = quality_passed and total_overruns == 0 and all(row['valid'] for row in rows)
    check(total_overruns == published['overruns'] == 105, '105 complete-time overruns')
    check(joint['late_n'] == 14, '14 multi-model overruns')
    check(primary_passed == published['primary_passed'] == False, 'registered primary gate failed')
    check(sum(row['calls'] for row in rows) == 404, '404 actual final-test repair calls')

    return {
        'schema': 'independent-g23-readonly-results-audit-v1',
        'scope': {'source': str(source.resolve()), 'new_environment_queries': 0, 'new_plans': 0,
                  'new_model_inferences': 0, 'new_training_updates': 0, 'new_performance_measurements': 0,
                  'method': 'Standard-library arithmetic from saved JSON only; no project imports or pickle/model deserialization.'},
        'input_sha256': {name: sha(out / name) for name in ('protocol.json', 'test_results.json', 'analysis.json', 'class_summary.json', 'plan_validation.json')},
        'all_passed': not failures, 'failures': failures,
        'counts': {'independent_test_scenes': len(cases), 'training_seeds': len(seeds), 'method_instances': len(methods),
                   'repetitions_per_scene_method': len(repetitions), 'saved_plans': len(rows), 'final_test_repairs': sum(row['calls'] for row in rows),
                   'original_detour_actual_test_repairs': summaries['original_detour16']['calls'], 'all_overruns': total_overruns},
        'formulas': {'J': 'L + 0.75 K', 'gain_vs_fixed3_percent': '100*(J_fixed3-J)/J_fixed3',
                     'main_comparison_percent': '100*(J_comparator-J_multi)/J_comparator, then average inside each scene',
                     'work_reward': 'gain_vs_fixed3_percent - W/1e6',
                     'exhaustive_ci': 'Percentile endpoints with linear interpolation over all 6**6 equally likely scene bootstrap resamples.'},
        'method_summary': summaries, 'class_summary': classes, 'main_comparisons': comparisons,
        'training_sets_from_registered_order_and_trial_slice': training_sets,
        'proposed_method_joint_quality_time': joint,
        'quality_varies_across_timing_repetitions': quality_changes,
        'plan_checks': checked_plans, 'rows': scalar_rows,
        'decision': {'quality_gate_passed': quality_passed, 'all_runs_on_time': total_overruns == 0,
                     'proposed_method_all_runs_on_time': joint['late_n'] == 0, 'registered_primary_passed': primary_passed,
                     'action': 'Respect the recorded stop decision; no extension of the local-window-plus-STOP training direction.'},
        'limitations': [
            'Collision/reachability geometry was not rerun; original saved certificates and hashes are the geometric evidence.',
            'Task coverage is checked against the saved Fixed3 task set; scenes are not regenerated.',
            'Old G2/G21/G22 payloads and old split seeds were not restored or independently rehashed here.',
            'Specific NumPy 10,000 bootstrap draws are not rerun; exact enumeration checks the same six-scene empirical resampling distribution.',
            'Scene bootstrap conditions on these trained models and does not separately resample training-seed uncertainty.',
            'Six new scene IDs within G2.3, two timing repetitions, and different training sets across seeds limit external validity.',
            'A stopping investment decision is not a proof that all local-search reinforcement learning methods fail.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.is_relative_to(source):
        raise ValueError('Audit output must be outside the original source package')
    data = audit(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'all_passed': data['all_passed'], 'failures': data['failures'], 'counts': data['counts'],
                      'joint_quality_time': data['proposed_method_joint_quality_time']['counts'],
                      'primary_passed': data['decision']['registered_primary_passed']}, ensure_ascii=False))
    if not data['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
