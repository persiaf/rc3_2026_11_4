"""Frozen held-out comparison: paired costs, equal-budget controls, complete-plan timing."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from rc3_baseline import baseline
from rc3_baseline.validator import validate
from .planner import choose_and_solve, map_key
from .scenes import make_complex_scene
from .selector import Selector
from .solver import prepare, export_plan, two_opt


METHODS = ('legacy', 'legacy_nearest', 'vectorized', 'default_joint', 'fixed_top1', 'fixed_top3',
           'random_top3', 'learned_top1', 'learned_top3')


def one_method(problem, method, selector, random_ids):
    start = time.perf_counter()
    if method in ('legacy', 'legacy_nearest', 'vectorized'):
        plan = baseline.solve(problem.scene, problem.F, problem.arm_dist, problem.nav,
                              'nearest_feasible_2opt' if method == 'legacy_nearest' else 'coverage_cost_2opt',
                              two_opt_fn=two_opt if method == 'vectorized' else None)
        plan['method'] = method
        before_check = time.perf_counter()
        plan['validation'] = validate(problem.scene, plan, problem.sparse)
        details = dict(feature_s=0., model_s=0., search_s=before_check-start,
                       export_and_validation_s=time.perf_counter()-before_check)
    else:
        if method.startswith('fixed_'):
            recipe_ids = selector.meta['fixed_portfolios'][method.replace('fixed_', '')]
        elif method == 'random_top3':
            recipe_ids = random_ids
        else:
            recipe_ids = None
        model = selector if method.startswith('learned_') else None
        k = 3 if method.endswith('top3') else 1
        route, details = choose_and_solve(problem, model, k, recipe_ids)
        before_export = time.perf_counter()
        plan = export_plan(problem, route, method)
        details['export_and_validation_s'] = time.perf_counter()-before_export
    elapsed = time.perf_counter()-start
    if not plan['validation']['valid_model_plan'] or not plan['validation']['all_tasks_served']:
        raise AssertionError(f'{method}: {plan["validation"]}')
    return plan, elapsed, details


def mean_ci(values, rng):
    values = np.asarray(values, dtype=float)
    means = values[rng.integers(len(values), size=(5000,len(values)))].mean(1)
    return [float(v) for v in np.quantile(means,[.025,.975])]


def summarize(rows):
    groups = {'all': lambda r: True,
              **{f'tables_{n}': lambda r,n=n: r['tables']==n for n in sorted({r['tables'] for r in rows})},
              **{f'mode_{m}': lambda r,m=m: r['mode']==m for m in ('regular','mixed','blocked')},
              '100_complex': lambda r: r['tables']==100 and r['mode']!='regular',
              '100_blocked': lambda r: r['tables']==100 and r['mode']=='blocked'}
    summaries, comparisons = [], []
    rng = np.random.default_rng(314159)
    pairs = [('learned_top1','legacy'),('learned_top3','legacy'),('fixed_top1','legacy'),
             ('learned_top1','legacy_nearest'),('learned_top3','legacy_nearest'),
             ('learned_top1','fixed_top1'),('learned_top3','fixed_top3'),
             ('default_joint','vectorized'),('vectorized','legacy')]
    for group, predicate in groups.items():
        subset = [r for r in rows if predicate(r)]
        if not subset:
            continue
        lookup = {(r['seed'],r['method']):r for r in subset}
        for method in METHODS:
            current = [r for r in subset if r['method']==method]
            summaries.append(dict(group=group,method=method,n=len(current),
                                  objective_mean_m=float(np.mean([r['objective_m'] for r in current])),
                                  length_mean_m=float(np.mean([r['length_m'] for r in current])),
                                  stops_mean=float(np.mean([r['stops'] for r in current])),
                                  planning_median_ms=float(np.median([r['planning_s'] for r in current])*1000),
                                  planning_p95_ms=float(np.quantile([r['planning_s'] for r in current],.95)*1000),
                                  cold_total_median_ms=float(np.median([r['cold_total_s'] for r in current])*1000),
                                  warm_total_median_ms=float(np.median([r['warm_total_s'] for r in current])*1000),
                                  all_valid=all(r['valid'] for r in current)))
        for method, reference in pairs:
            current = [r for r in subset if r['method']==method]
            delta = [(lookup[r['seed'],reference]['objective_m']-r['objective_m']) /
                     max(lookup[r['seed'],reference]['objective_m'],1e-8)*100 for r in current]
            comparisons.append(dict(group=group,method=method,reference=reference,n=len(current),
                                    mean_cost_reduction_pct=float(np.mean(delta)),
                                    reduction_ci95_pct=mean_ci(delta,rng),
                                    wins=sum(d>1e-7 for d in delta),ties=sum(abs(d)<=1e-7 for d in delta),
                                    losses=sum(d < -1e-7 for d in delta),
                                    median_planning_speedup=float(np.median([
                                        lookup[r['seed'],reference]['planning_s']/r['planning_s'] for r in current])),
                                    median_cold_speedup=float(np.median([
                                        lookup[r['seed'],reference]['cold_total_s']/r['cold_total_s'] for r in current]))))
    return summaries, comparisons


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='optional separately frozen OOD case manifest')
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error('repeats must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'plans').mkdir()
    case_path = args.cases or args.data/'test_cases.json'
    cases = json.loads(case_path.read_text(encoding='utf-8'))
    selector = Selector(args.model)
    manifest = dict(status='running',test_instances=len(cases),methods=METHODS,repeats=args.repeats,
                    model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    test_cases_sha256=hashlib.sha256(case_path.read_bytes()).hexdigest(),
                    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                   [*Path(__file__).parent.glob('*.py'),Path(baseline.__file__)]},
                    python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                    timing_scope='planning includes features/model when applicable, full polyline/assignments and independent validation; excludes JSON I/O and process/model loading',
                    preprocessing='All methods share vectorized reachability and exactly the same navigation graph. Cold total adds new graph; warm total adds reachability and map key but reuses graph.',
                    independence='Model/checkpoint/fixed portfolios frozen before opening test cases; all methods use the same solver passes. Model is a heuristic selector, not a direct route decoder.')
    baseline.write_json(args.output/'manifest.json',manifest)
    warmup = prepare(make_complex_scene(4,3,991734,'mixed'))
    for method in METHODS:
        one_method(warmup,method,selector,[0,4,7])
    started, rows = time.perf_counter(), []
    for index, case in enumerate(cases):
        scene = make_complex_scene(case['nx'],case['ny'],case['seed'],case['mode'])
        problem = prepare(scene)
        warm_start = time.perf_counter()
        map_key(scene)
        prepare(scene, nav=problem.nav)
        warm_preprocess = time.perf_counter()-warm_start
        random_ids = np.random.default_rng(case['seed']+9871).choice(13,3,replace=False).tolist()
        timings, components, plans = {m:[] for m in METHODS}, {m:[] for m in METHODS}, {}
        order_rng = np.random.default_rng(case['seed']+4123)
        for _ in range(args.repeats):
            for method in order_rng.permutation(METHODS).tolist():
                plan, seconds, details = one_method(problem,method,selector,random_ids)
                if method in plans and abs(plan['objective_length_plus_stops_m']-plans[method]['objective_length_plus_stops_m'])>1e-7:
                    raise AssertionError('nondeterministic route objective')
                plans[method] = plan
                timings[method].append(seconds)
                components[method].append(details)
        if abs(plans['legacy']['objective_length_plus_stops_m']-plans['vectorized']['objective_length_plus_stops_m'])>1e-7:
            raise AssertionError('vectorized control differs from reference')
        for method in METHODS:
            plan, details = plans[method], components[method]
            elapsed = float(np.median(timings[method]))
            rows.append(dict(seed=case['seed'],nx=case['nx'],ny=case['ny'],tables=case['nx']*case['ny'],
                             mode=case['mode'],tasks=len(scene['tasks']),obstacles=len(scene['obstacles']),method=method,
                             objective_m=plan['objective_length_plus_stops_m'],length_m=plan['route_length_m'],
                             stops=len(plan['service_stops']),valid=True,planning_s=elapsed,
                             cold_preprocessing_s=problem.preprocessing_s,warm_preprocessing_s=warm_preprocess,
                             cold_total_s=problem.preprocessing_s+elapsed,warm_total_s=warm_preprocess+elapsed,
                             **{key:float(np.median([d[key] for d in details])) for key in
                                ('feature_s','model_s','search_s','export_and_validation_s')}))
        baseline.write_json(args.output/'plans'/f'scene_plan_{case["nx"]}x{case["ny"]}_{case["mode"]}_{case["seed"]}.json',
                            dict(scene=scene,feasibility=problem.sparse,plans=plans))
        if (index+1)%12==0:
            print(f'test {index+1}/{len(cases)} all valid; elapsed={time.perf_counter()-started:.1f}s',flush=True)
    summaries, comparisons = summarize(rows)
    baseline.write_csv(args.output/'metrics.csv',rows)
    baseline.write_json(args.output/'summary.json',dict(summaries=summaries,comparisons=comparisons))
    baseline.write_csv(args.output/'summary.csv',summaries)
    baseline.write_json(args.output/'comparisons.json',comparisons)
    manifest.update(status='complete',valid_plans=len(rows),benchmark_s=time.perf_counter()-started)
    baseline.write_json(args.output/'manifest.json',manifest)
    print(json.dumps(dict(status='complete',valid_plans=len(rows),benchmark_s=manifest['benchmark_s'])),flush=True)


if __name__ == '__main__':
    main()
