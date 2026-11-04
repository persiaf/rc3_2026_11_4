"""Fresh, predeclared small-instance quality reference; no training or tuning.

python -B -m rc3_v4.small_benchmark --output rc3_v4/small_results
Timing is diagnostic (one execution per method), not a speed benchmark.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from rc3_baseline.baseline import expand_route
from rc3_baseline.validator import validate, validate_file
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare, export_plan
from rc3_v3.runtime import run
from rc3_v3.data import historical_seeds
from .exact import solve, solve_metric


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def table_route(problem):
    """Visit tables by ID, greedily cover that table using its own eight docks.

    Select max remaining same-table coverage, then nearest from current node,
    then lower candidate ID. Arrival also serves other reachable pending tasks.
    """
    scene, cover, costs = problem.scene, problem.F.any(2), problem.nav['costs']
    left = np.ones(len(cover), bool)
    route = [0]
    for table in scene['tables']:
        task_ids = np.asarray([t['id'] for t in scene['tasks'] if t['table_id'] == table['id']], int)
        candidate_ids = np.asarray([c['id'] for c in scene['candidates'] if c['table_id'] == table['id']], int)
        while left[task_ids].any():
            remaining = task_ids[left[task_ids]]
            gains = cover[np.ix_(remaining, candidate_ids)].sum(0)
            good = ((gains > 0) & np.isfinite(costs[route[-1], candidate_ids+1])
                    & np.isfinite(costs[candidate_ids+1, 0]))
            if not good.any():
                raise ValueError('table-by-table baseline cannot cover a task at an own-table dock')
            choices = np.flatnonzero(good)
            pick = min(choices, key=lambda k: (-int(gains[k]), costs[route[-1], candidate_ids[k]+1], candidate_ids[k]))
            candidate = int(candidate_ids[pick])
            route.append(candidate+1)
            left &= ~cover[:, candidate]
    if left.any():
        raise ValueError('table baseline incomplete')
    return route+[0]


def reorder_fixed_plan(problem, initial, route, method):
    """Preserve the table baseline's service set AND task/arm assignments.

    Unlike canonical export, this intentionally does not move assignments to an
    earlier overlapping stop after reordering. It isolates visit-order choice.
    """
    result = copy.deepcopy(initial)
    selected = [node-1 for node in route[1:-1]]
    if set(selected) != set(initial['selected_candidate_ids']) or len(selected) != len(set(selected)):
        raise ValueError('fixed-set ordering changed selected candidates')
    polyline, legs = expand_route(route, problem.nav)
    by_id = {s['candidate_id']: s for s in result['service_stops']}
    arrival, stops = 0., []
    for sequence, j in enumerate(selected, 1):
        arrival += legs[sequence-1]
        stop = by_id[j]
        stop.update(sequence=sequence, arrival_distance_m=arrival)
        stops.append(stop)
    result.update(method=method, selected_candidate_ids=selected,
                  visit_order_candidate_ids=selected, service_stops=stops,
                  route_polyline=polyline, route_leg_costs_m=legs,
                  route_length_m=sum(legs),
                  objective_length_plus_stops_m=sum(legs)+problem.penalty*len(selected))
    result['validation'] = validate(problem.scene, result, problem.sparse)
    if not result['validation']['valid_model_plan']:
        raise ValueError(result['validation']['errors'])
    return result


def method(problem, name, exact_limit_s):
    if name in ('fixed3', 'detour8'):
        plan, info = run(problem, 'detour' if name == 'detour8' else 'fixed3', horizon=8)
        return plan, info
    if name == 'table_order':
        return export_plan(problem, table_route(problem), name), {'status': 'heuristic'}
    if name == 'fixed_set_exact_order':
        initial = export_plan(problem, table_route(problem), 'table_order')
        pool = [0]+[j+1 for j in initial['selected_candidate_ids']]
        result = solve_metric(np.eye(len(pool)-1, dtype=bool),
                              problem.nav['costs'][np.ix_(pool, pool)], problem.penalty,
                              time_limit_s=exact_limit_s,
                              incumbent=[0]+list(range(1, len(pool)))+[0])
        if result['route'] is None:
            return None, result
        route = [pool[j] for j in result['route']]
        result['scope'] = 'Exact visit order for the fixed table-baseline stop set only; task/arm assignment fixed.'
        return reorder_fixed_plan(problem, initial, route, name), result
    if name == 'joint_exact':
        # Construction of this warm start is included in complete method timing.
        initial, info = run(problem, 'fixed3')
        route = [0]+[j+1 for j in initial['selected_candidate_ids']]+[0]
        result = solve(problem, incumbent=route, time_limit_s=exact_limit_s)
        if result['route'] is None:
            return None, result
        plan = export_plan(problem, result['route'], name)
        if abs(plan['objective_length_plus_stops_m']-result['objective']) > 1e-7:
            raise AssertionError('exact abstract tour and exported cost differ')
        return plan, result
    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('rc3_v4/small_results'))
    parser.add_argument('--exact-time-limit', type=float, default=30.)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output/'metrics.csv').exists():
        raise ValueError('output already has results; use a fresh directory to preserve evidence')
    output.mkdir(parents=True, exist_ok=True)
    cases = [dict(case_id=k, nx=nx, ny=ny, mode=mode, seed=840001+k)
             for k, (nx, ny, mode, repeat) in enumerate(
                 (nx, ny, mode, repeat)
                 for nx, ny in ((1, 2), (2, 2), (3, 2))
                 for mode in ('regular', 'mixed', 'blocked') for repeat in range(3))]
    previous, seed_scan = historical_seeds(Path.cwd(), exclude=output)
    reused = sorted(previous & {r['seed'] for r in cases})
    if reused:
        raise ValueError(f'predeclared seed collision: {reused}; do not silently resample')
    sources = [Path(__file__), Path(__file__).with_name('exact.py'),
               Path('rc3_fast/solver.py'), Path('rc3_v3/runtime.py'), Path('rc3_improve/repair.py')]
    protocol = dict(cases=cases, primary_penalty=.75,
                    sensitivity_penalties=[0., 1.5], sensitivity_subset='all nine 2x2 cases',
                    methods=['table_order', 'fixed_set_exact_order', 'fixed3', 'detour8', 'joint_exact'],
                    max_tasks=14, max_candidates=128, exact_time_limit_s=args.exact_time_limit,
                    task_resampling=False, unsupported_policy='retain explicit row and exclude only uncertified optimal gaps',
                    repeats=1, timing_scope='single execution, common navigation preprocessing excluded; diagnostic only, no speed claim',
                    seed_collision_check=dict(reused=reused, historical_unique=len(previous), unreadable=seed_scan['unreadable']),
                    source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                    frozen_before_results=True)
    write(output/'protocol.json', protocol)
    metrics, details, checks = [], [], []
    started = time.perf_counter()
    rng = np.random.default_rng(840031)
    for case in cases:
        scene = make_complex_scene(case['nx'], case['ny'], case['seed'], case['mode'])
        nav = None
        penalties = [.75, 0., 1.5] if case['nx'] == case['ny'] == 2 else [.75]
        for penalty in penalties:
            problem = prepare(scene, penalty=penalty, nav=nav)
            nav = problem.nav
            plans, records = {}, {}
            for name in rng.permutation(protocol['methods']):
                name = str(name)
                tick = time.perf_counter()
                try:
                    plan, info = method(problem, name, args.exact_time_limit)
                    elapsed = time.perf_counter()-tick
                    if plan is not None:
                        validation = validate(scene, plan, problem.sparse)
                        if not validation['valid_model_plan'] or not validation['all_tasks_served']:
                            raise AssertionError(validation)
                        plans[name] = plan
                    records[name] = dict(info=info, elapsed_s=elapsed)
                except Exception as error:
                    records[name] = dict(info={'status': 'failed', 'reason': str(error)},
                                         elapsed_s=time.perf_counter()-tick)
            exact_info = records['joint_exact']['info']
            optimum = exact_info.get('objective') if exact_info.get('status') == 'optimal' else None
            for name in protocol['methods']:
                rec, plan = records[name], plans.get(name)
                cost = plan['objective_length_plus_stops_m'] if plan else None
                gap = (100*(cost-optimum)/max(abs(optimum), 1e-8)
                       if cost is not None and optimum is not None else None)
                if gap is not None and gap < -1e-6:
                    raise AssertionError(f'{name} cost below proved optimum')
                metrics.append(dict(**case, tables=case['nx']*case['ny'], tasks=len(scene['tasks']),
                                    penalty=penalty, method=name, status=rec['info']['status'],
                                    objective=cost, stops=len(plan['selected_candidate_ids']) if plan else None,
                                    optimal_objective=optimum, optimal_gap_pct=max(0., gap) if gap is not None else None,
                                    bound=exact_info.get('lower_bound'), method_s=rec['elapsed_s'],
                                    validated=bool(plan and plan['validation']['valid_model_plan']),
                                    bound_status=exact_info['status']))
            filename = f"scene_plan_{case['seed']}_lambda{penalty:g}.json"
            write(output/'plans'/filename, dict(scene=scene, feasibility=problem.sparse,
                                                plans=plans, diagnostics=records))
            checks.extend(validate_file(output/'plans'/filename))
            details.append(dict(**case, penalty=penalty, methods=records))
        print(json.dumps(dict(case=case['case_id'], tasks=len(scene['tasks']), done=True)), flush=True)
    with (output/'metrics.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    aggregates = []
    for penalty in (.75, 0., 1.5):
        for name in protocol['methods']:
            rows = [r for r in metrics if r['penalty'] == penalty and r['method'] == name]
            valid = [r for r in rows if r['optimal_gap_pct'] is not None]
            aggregates.append(dict(penalty=penalty, method=name, cases=len(rows),
                                   certified_optimum_cases=len(valid), valid_plans=sum(r['validated'] for r in rows),
                                   mean_optimal_gap_pct=float(np.mean([r['optimal_gap_pct'] for r in valid])) if valid else None,
                                   median_optimal_gap_pct=float(np.median([r['optimal_gap_pct'] for r in valid])) if valid else None,
                                   max_optimal_gap_pct=max((r['optimal_gap_pct'] for r in valid), default=None),
                                   mean_stops=float(np.mean([r['stops'] for r in rows if r['stops'] is not None])),
                                   median_method_ms=1000*float(np.median([r['method_s'] for r in rows]))))
    bound_rows = [r for r in metrics if r['method'] == 'joint_exact']
    summary = dict(scene_count=len(cases), scenario_penalty_count=len(details), rows=len(metrics),
                   saved_plans=len(checks), independently_valid=sum(c['valid_model_plan'] for c in checks),
                   all_tasks_served=sum(c['all_tasks_served'] for c in checks),
                   exact_status_counts={status: sum(r['status'] == status for r in bound_rows)
                                        for status in sorted({r['status'] for r in bound_rows})},
                   aggregates=aggregates, elapsed_s=time.perf_counter()-started,
                   scope='Fresh generated proxy scenes; global optimum only for supported finite candidates and supplied shortest-path metric; timing is diagnostic one-shot.')
    write(output/'details.json', details)
    write(output/'verification.json', dict(all_valid=all(c['valid_model_plan'] and c['all_tasks_served'] for c in checks), checks=checks))
    write(output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=True), flush=True)
    if summary['independently_valid'] != summary['saved_plans'] or any(r['status'] == 'failed' for r in metrics):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
