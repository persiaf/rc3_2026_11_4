"""Exact small-instance reference for the current discrete cover-routing model.

State: (covered task bitmask, last service candidate); positive-gain transitions
cost roadmap_distance + stop_penalty and completion adds the depot return.
This is exact only for the supplied discrete F/D model, not continuous robotics.

The nonnegative shortest-path metric permits deleting every zero-new-coverage
service stop without increasing cost. Thus an optimal solution exists with at
most N positive-gain stops; this DP enumerates all such possibilities. Every
transition strictly increases the bitmask, so numeric mask order is topological.

No neural training, changes to original sources, or historical timings are used.

Portable layout: place this script in audits/exact/ below the directory that
contains rc3_baseline/ and rc3_fast/, or provide --source-root explicitly.
The fixed portfolios are frozen constants; selector.npz is not required.
Use --check-only for the original synthetic/empty checks. New runs default to
rerun_N beside this script so the archived run_N results remain separate.
When importing functions, call configure_source(...) before running an audit.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

# These portfolios were frozen before the original test evaluation. The model
# archive is provenance only: it is deliberately not a runtime dependency.
FIXED1 = (1,)
FIXED3 = (0, 1, 7)
FROZEN_PORTFOLIO_PROVENANCE = {'source_file': 'rc3_fast/deployment/selector.npz', 'source_field': 'metadata.fixed_portfolios', 'source_model_sha256': 'a553b653257e46a1661bc5d6e9b18ce815f064c569deca39a3a62c83897cdefd', 'source_metadata_utf8_sha256': '25371fe3eeea1e2a3f38fe0994320b77e78c02fc010fb567b1f50e0c0b2921d9', 'selection_manifest': 'rc3_fast/deployment/selection.json', 'frozen_values': {'top1': [1], 'top3': [0, 1, 7]}, 'runtime_requires_model_file': False}
SOURCE = None


def locate_source_root(explicit=None):
    """Find the directory containing adjacent rc3_baseline and rc3_fast."""
    required = ('rc3_baseline/baseline.py', 'rc3_baseline/validator.py',
                'rc3_fast/solver.py')
    candidates = ([Path(explicit).expanduser().resolve()] if explicit is not None
                  else list(Path(__file__).resolve().parents))
    for candidate in candidates:
        if all((candidate/name).is_file() for name in required):
            return candidate
    raise FileNotFoundError(
        'Cannot locate adjacent rc3_baseline and rc3_fast; pass --source-root '
        'with the directory containing those two packages.')


def configure_source(explicit=None):
    global SOURCE, baseline, RECIPES, prepare, solve_route, route_objective, export_plan
    SOURCE = locate_source_root(explicit)
    sys.path.insert(0, str(SOURCE))
    from rc3_baseline import baseline
    from rc3_fast.solver import RECIPES, prepare, solve_route, route_objective, export_plan
    return SOURCE


def exact_cover_dp(problem, max_tasks=12):
    started = time.perf_counter()
    cover = problem.F.any(2)
    n, m = cover.shape
    if n > max_tasks:
        raise ValueError(f'exact DP is intentionally limited to {max_tasks} tasks')
    if not np.isfinite(problem.penalty) or problem.penalty < 0:
        raise ValueError('requires a finite nonnegative stop penalty')
    if n == 0:
        return [0, 0], 0., dict(solver_s=time.perf_counter()-started,
                                states_expanded=0, useful_candidates=0)
    full = (1 << n)-1
    costs = np.asarray(problem.nav['costs'], dtype=np.float64)
    if costs.shape != (m+1, m+1) or np.any(costs[np.isfinite(costs)] < -1e-9):
        raise ValueError('invalid roadmap distances')
    useful = cover.any(0) & np.isfinite(costs[0, 1:]) & np.isfinite(costs[1:, 0])
    nodes = np.r_[0, np.flatnonzero(useful)+1]
    small = costs[np.ix_(nodes, nodes)]
    covers = np.asarray([sum(1 << int(i) for i in np.flatnonzero(cover[:, j-1]))
                         for j in nodes[1:]], dtype=np.int64)
    if np.bitwise_or.reduce(covers, initial=0) != full:
        raise ValueError('some tasks have no reachable service candidate')
    # Verify the property used in the exactness proof on this complete tiny D.
    if np.any(small[:, None, :] > small[:, :, None]+small[None, :, :]+1e-8):
        raise ValueError('D is not a shortest-path metric; zero-gain pruning is unjustified')
    k = len(nodes)
    dp = np.full((1 << n, k), np.inf, dtype=np.float64)
    previous_mask = np.full(dp.shape, -1, dtype=np.int32)
    previous_node = np.full(dp.shape, -1, dtype=np.int16)
    dp[0, 0] = 0.
    expanded = 0
    for mask in range(full):
        live = np.flatnonzero(np.isfinite(dp[mask]))
        if not len(live):
            continue
        expanded += len(live)
        merged = mask | covers
        allowed = np.flatnonzero(merged != mask)
        if not len(allowed):
            continue
        transitions = dp[mask, live, None]+small[np.ix_(live, allowed+1)]+problem.penalty
        best_from = transitions.argmin(0)
        values = transitions[best_from, np.arange(len(allowed))]
        dest_mask, dest_node = merged[allowed], allowed+1
        improve = values < dp[dest_mask, dest_node]-1e-12
        dp[dest_mask[improve], dest_node[improve]] = values[improve]
        previous_mask[dest_mask[improve], dest_node[improve]] = mask
        previous_node[dest_mask[improve], dest_node[improve]] = live[best_from[improve]]
    total = dp[full]+small[:, 0]
    last = int(total.argmin())
    optimum = float(total[last])
    if not np.isfinite(optimum):
        raise ValueError('no complete route')
    mask, reversed_nodes = full, []
    while mask:
        reversed_nodes.append(int(nodes[last]))
        old_mask = int(previous_mask[mask, last])
        old_last = int(previous_node[mask, last])
        if old_mask < 0 or old_last < 0:
            raise RuntimeError('invalid DP predecessor chain')
        mask, last = old_mask, old_last
    route = [0]+list(reversed(reversed_nodes))+[0]
    if not np.isclose(route_objective(problem, route), optimum, atol=1e-9):
        raise RuntimeError('DP reconstruction objective mismatch')
    return route, optimum, dict(solver_s=time.perf_counter()-started,
                                states_expanded=expanded, useful_candidates=k-1,
                                allocated_states=int(dp.size))


def portfolio(problem, ids):
    started = time.perf_counter()
    routes = [solve_route(problem, i) for i in ids]
    objectives = [route_objective(problem, route) for route in routes]
    winner = int(np.argmin(objectives))
    return routes[winner], objectives[winner], time.perf_counter()-started, ids[winner]


def small_correctness_checks():
    # An independently exhaustive tiny combinatorial case catches DP indexing,
    # charging and reconstruction errors. Geometric cases are also validated.
    from types import SimpleNamespace
    positions = np.asarray([[0., 0.], [1., 0.], [1., 1.], [2., 0.], [0., 2.]])
    distance = np.linalg.norm(positions[:, None]-positions[None], axis=-1)
    f = np.zeros((3, 4, 2), dtype=bool)
    f[0, [0, 1], 0] = True
    f[1, [1, 2], 0] = True
    f[2, [2, 3], 1] = True
    problem = SimpleNamespace(F=f, nav={'costs': distance}, penalty=.75)
    route, value, _ = exact_cover_dp(problem)
    brute = np.inf
    for length in range(1, 5):
        for sequence in itertools.permutations(range(1, 5), length):
            if f[:, np.asarray(sequence)-1].any((1, 2)).all():
                brute = min(brute, route_objective(problem, [0, *sequence, 0]))
    if not np.isclose(value, brute, atol=1e-9):
        raise AssertionError((value, brute))
    empty = prepare(baseline.make_scene(2, 2, 3_100_001_000, counts=[0, 0, 0, 0]))
    empty_route, empty_value, _ = exact_cover_dp(empty)
    empty_plan = export_plan(empty, empty_route, 'exact_dp_empty')
    if empty_value != 0 or empty_route != [0, 0] or not empty_plan['validation']['valid_model_plan']:
        raise AssertionError('empty-instance handling failed')
    return dict(exhaustive_synthetic_cost=value, exhaustive_synthetic_match=True,
                empty_cost=empty_value, empty_valid=True)


def run(count, output):
    if not 1 <= count <= 12:
        raise ValueError('audit limited to 1..12 random small scenes')
    output.mkdir(parents=True, exist_ok=True)
    fixed1, fixed3 = list(FIXED1), list(FIXED3)
    checks = small_correctness_checks()
    rows, plans, skipped = [], [], []
    cursor = 0
    while len(rows) < count:
        nx, ny = (2, 2) if len(rows) % 2 == 0 else (3, 2)
        seed = 3_100_000_000+cursor
        cursor += 1
        scene = baseline.make_scene(nx, ny, seed)
        n = len(scene['tasks'])
        if n > 12 or n == 0:
            skipped.append(dict(nx=nx, ny=ny, seed=seed, tasks=n,
                                reason='audit conditions on 1 <= N <= 12; task layout unchanged'))
            continue
        problem = prepare(scene)
        route, exact, info = exact_cover_dp(problem)
        exact_plan = export_plan(problem, route, 'exact_dp')
        row = dict(scale=f'{nx}x{ny}', seed=seed, tasks=n,
                   candidates=len(scene['candidates']), useful_candidates=info['useful_candidates'],
                   preprocessing_s=problem.preprocessing_s, exact_objective=exact,
                   exact_stops=len(route)-2, exact_solver_s=info['solver_s'],
                   dp_states_expanded=info['states_expanded'])
        case_plans = {'exact_dp': exact_plan}
        for name, ids in [('fixed1', fixed1), ('fixed3', fixed3), ('oracle13', list(range(13)))]:
            candidate_route, value, elapsed, winner = portfolio(problem, ids)
            plan = export_plan(problem, candidate_route, name)
            if value < exact-1e-8:
                raise AssertionError('heuristic below claimed exact optimum')
            if not np.isclose(value, plan['objective_length_plus_stops_m'], atol=1e-9):
                raise AssertionError('export unexpectedly pruned the measured route')
            row.update({name+'_objective': value, name+'_gap_pct': 100*(value/exact-1),
                        name+'_solver_s': elapsed, name+'_winning_recipe': winner})
            case_plans[name] = plan
        plans.append(dict(scene=scene, feasibility=problem.sparse, plans=case_plans,
                          exactness_scope='current discrete candidate/feasibility/roadmap model only'))
        rows.append(row)
        print(json.dumps(row), flush=True)
    with (output/'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for case in plans:
        scene = case['scene']
        (output/f'scene_plan_{scene["nx"]}x{scene["ny"]}_{scene["seed"]}.json').write_text(json.dumps(case, indent=2))
    summary = dict(cases=len(rows), checks=checks, skipped_candidates=skipped,
                   fixed_portfolios=dict(top1=fixed1, top3=fixed3), recipes=RECIPES,
                   fixed_portfolio_provenance=FROZEN_PORTFOLIO_PROVENANCE,
                   sampling='alternating 2x2 and 3x2, sequential audit seeds >= 3100000000; conditioned on 1<=tasks<=12',
                   task_generation='original make_scene, independent uniform count 0..3 per table; no task resampling',
                   exactness_scope='current discrete candidate/feasibility/roadmap model only; oracle13 is a finite-portfolio reference',
                   timing_scope='new CPU timings only; solve includes route construction/DP; shared preprocessing and export/validation excluded',
                   python=platform.python_version(), numpy=np.__version__,
                   source_sha256={str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [SOURCE/'rc3_baseline/baseline.py',SOURCE/'rc3_baseline/validator.py',SOURCE/'rc3_fast/solver.py']},
                   all_plans_valid=all(p['validation']['valid_model_plan'] for c in plans for p in c['plans'].values()))
    for name in ['fixed1', 'fixed3', 'oracle13']:
        values = np.asarray([r[name+'_gap_pct'] for r in rows])
        times = np.asarray([r[name+'_solver_s'] for r in rows])
        summary[name] = dict(mean_gap_pct=float(values.mean()), max_gap_pct=float(values.max()),
                             exact_match_cases=int((values < 1e-8).sum()),
                             mean_solver_s=float(times.mean()))
    summary['exact'] = dict(mean_solver_s=float(np.mean([r['exact_solver_s'] for r in rows])),
                            max_solver_s=max(r['exact_solver_s'] for r in rows))
    (output/'summary.json').write_text(json.dumps(summary, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=int, default=5)
    parser.add_argument('--output', type=Path,
                        help='new output folder; defaults to rerun_N beside this script')
    parser.add_argument('--source-root', type=Path,
                        help='directory containing rc3_baseline and rc3_fast; otherwise search above this file')
    parser.add_argument('--check-only', action='store_true',
                        help='run the original tiny exhaustive and empty-task checks, without generating cases')
    args = parser.parse_args()
    configure_source(args.source_root)
    if args.check_only:
        print(json.dumps(dict(source_root=str(SOURCE), checks=small_correctness_checks(),
                              fixed_portfolio_provenance=FROZEN_PORTFOLIO_PROVENANCE), indent=2))
    else:
        run(args.cases, args.output or Path(__file__).resolve().parent/f'rerun_{args.cases}')
