"""Exploratory, same-process v2 versus additive adapter comparison.

The cases are fresh development cases, NOT a replacement frozen test set.
Both versions use the same NumPy export of the uploaded frozen weights.
"""
import argparse
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy

from rc3_baseline.baseline import write_json
from rc3_baseline.validator import validate
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare
from rc3_improve.environment import ImprovementEnv
from rc3_improve.evaluate import run_method
from rc3_improve.efficient import run_efficient
from rc3_improve.numpy_policy import NumpyPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('patch_benchmark_repeat'))
    parser.add_argument('--cases', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.cases < 1 or args.repeats < 1 or (args.output.exists() and any(args.output.iterdir())):
        parser.error('positive counts and new/empty output directory required')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'plans').mkdir()
    policies = {stage: NumpyPolicy.load(Path(__file__).parent/'weights'/f'{stage}.npz', stage) for stage in ('bc', 'ppo')}
    variants = {
        'v2_fixed1': lambda p: run_method(p, 'fixed1'),
        'v2_fixed3': lambda p: run_method(p, 'fixed3'),
        'efficient_fixed3': lambda p: run_efficient(p, 'fixed3'),
        'v2_ppo': lambda p: run_method(p, 'ppo', policies['ppo']),
        'deferred_ppo_same_mask': lambda p: run_efficient(p, 'ppo', policies['ppo'], refresh_attempts=False),
        'refreshed_ppo_fixed1': lambda p: run_efficient(p, 'ppo', policies['ppo']),
        'refreshed_ppo_fixed3': lambda p: run_efficient(p, 'ppo', policies['ppo'], reference='fixed3'),
        'refreshed_bc_fixed3': lambda p: run_efficient(p, 'bc', policies['bc'], reference='fixed3'),
        'refreshed_detour_fixed3': lambda p: run_efficient(p, 'detour', reference='fixed3'),
    }
    rows, checks = [], []
    started = time.perf_counter()
    for i in range(args.cases):
        seed = 3701161000+i
        mode = ('regular', 'mixed', 'blocked')[i % 3]
        scene = make_complex_scene(10, 10, seed, mode)
        problem = prepare(scene)
        # One tiny network warmup is excluded from timed solver calls, as loading is.
        if i == 0:
            features, mask = ImprovementEnv(problem).observe()
            for policy in policies.values():
                policy.scores(features, mask)
        samples = {name: [] for name in variants}
        last = {}
        rng = np.random.default_rng(seed)
        for repeat in range(args.repeats):
            for name in rng.permutation(list(variants)):
                tick = time.perf_counter()
                plan, info = variants[name](problem)
                samples[name].append(time.perf_counter()-tick)
                last[name] = (plan, info)
        fixed1 = last['v2_fixed1'][1]['objective_m']
        fixed3 = last['v2_fixed3'][1]['objective_m']
        old_actions = [r['action'] for r in last['v2_ppo'][1]['trace'] if r['event'] == 'repair']
        deferred_actions = [r['action'] for r in last['deferred_ppo_same_mask'][1]['trace'] if r['event'] == 'provisional_repair']
        check = dict(seed=seed, fixed3_cost_unchanged=abs(fixed3-last['efficient_fixed3'][1]['objective_m']) < 1e-7,
                     same_mask_ppo_actions_unchanged=old_actions == deferred_actions,
                     same_mask_ppo_cost_unchanged=abs(last['v2_ppo'][1]['objective_m']-last['deferred_ppo_same_mask'][1]['objective_m']) < 1e-7)
        checks.append(check)
        if not all(v for k, v in check.items() if k != 'seed'):
            raise AssertionError(check)
        output = dict(scene=scene, feasibility=problem.sparse, plans={}, diagnostics={})
        for name, (plan, info) in last.items():
            verified = validate(scene, plan, problem.sparse)
            if not verified['valid_model_plan'] or not verified['all_tasks_served']:
                raise AssertionError((name, verified))
            output['plans'][name] = plan
            output['diagnostics'][name] = info
            duration = float(np.median(samples[name]))
            row = dict(seed=seed, mode=mode, tables=100, tasks=len(scene['tasks']), method=name,
                objective_m=info['objective_m'], planning_median_ms=1000*duration,
                preprocessing_ms=1000*problem.preprocessing_s,
                cold_total_median_ms=1000*(duration+problem.preprocessing_s),
                gain_over_fixed1_pct=100*(fixed1-info['objective_m'])/fixed1,
                gain_over_fixed3_pct=100*(fixed3-info['objective_m'])/fixed3,
                valid=verified['valid_model_plan'], all_tasks_served=verified['all_tasks_served'],
                fallback_used=info['fallback_used'], attempts=info['attempts'])
            rows.append(row)
        write_json(args.output/'plans'/f'scene_plan_{i:02d}_{seed}.json', output)
        print(f'{i+1}/{args.cases} scenes, {len(rows)} plans, elapsed={time.perf_counter()-started:.1f}s', flush=True)
    with (args.output/'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = []
    for method in variants:
        subset = [r for r in rows if r['method'] == method]
        summary.append(dict(method=method, n=len(subset), mean_cost=float(np.mean([r['objective_m'] for r in subset])),
            median_planning_ms=float(np.median([r['planning_median_ms'] for r in subset])),
            median_cold_total_ms=float(np.median([r['cold_total_median_ms'] for r in subset])),
            mean_gain_fixed1_pct=float(np.mean([r['gain_over_fixed1_pct'] for r in subset])),
            mean_gain_fixed3_pct=float(np.mean([r['gain_over_fixed3_pct'] for r in subset]))))
    payload = dict(scope='8 fresh 100-table development scenes by default; exploratory, no new training, no test-set superiority claim.',
        python=platform.python_version(), platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        cases=args.cases, repeats=args.repeats, elapsed_s=time.perf_counter()-started, plans=len(rows),
        all_valid=all(r['valid'] and r['all_tasks_served'] for r in rows),
        summary=summary, equality_checks=checks,
        timing='Single-process CPU, identical NumPy policy backend on both versions; scene median of repeated full solver/export/certification; preprocessing separate; excludes loading and writing.')
    write_json(args.output/'summary.json', payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == '__main__':
    main()
