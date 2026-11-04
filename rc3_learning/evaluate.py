"""Greedy checkpoint inference, shared-instance baselines and geometric certification."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from rc3_baseline import baseline
from rc3_baseline.validator import validate
from .data import collate, export_instance, make_instance, plan_from_actions
from .env import ReachabilityCoverRoutingEnv
from .model import AttentionPolicy, rollout
from .plots import save_route_plots


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def load_policy(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    model = AttentionPolicy(**saved['model_config']).to(device)
    model.load_state_dict(saved['model_state'])
    return model.eval(), saved


@torch.no_grad()
def greedy_costs(policy, env, instances, batch_size, device):
    policy.eval()
    costs = []
    for start in range(0, len(instances), batch_size):
        result = rollout(policy, env, collate(instances[start:start+batch_size], device))
        if not result['state']['completed'].all() or result['state']['failed'].any():
            raise ValueError('validation policy failed to complete a feasible scene')
        costs.extend(result['cost'].cpu().tolist())
    return np.asarray(costs)


@torch.no_grad()
def benchmark(policy, instances, output_dir, device='cpu', plots=True):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.eval()
    env = ReachabilityCoverRoutingEnv(device)
    # Warm up before latency measurement. No optimization or parameter updates.
    rollout(policy, env, collate(instances[:1], device))
    synchronize(device)
    rows, samples, schedules = [], [], []
    for item in instances:
        scene, plans = item.scene, {}
        for method in baseline.METHODS:
            start = time.perf_counter()
            plan = baseline.solve(scene, item.feasibility, item.arm_distance, item.navigation, method, item.penalty)
            solve_s = time.perf_counter()-start
            start = time.perf_counter()
            plan['validation'] = validate(scene, plan, item.sparse)
            validation_s = time.perf_counter()-start
            plans[method] = plan
            rows.append(metric(item, plan, solve_s, validation_s))
        if item.teacher:
            plans['teacher_replayed'] = item.teacher
            # Teacher creation includes original solve, replay and independent check.
            rows.append(metric(item, item.teacher, item.teacher_s, 0.))
        synchronize(device)
        start = time.perf_counter()
        batch = collate([item], device)
        synchronize(device)
        feature_s = time.perf_counter()-start
        start = time.perf_counter()
        result = rollout(policy, env, batch)
        synchronize(device)
        solver_s = time.perf_counter()-start
        start = time.perf_counter()
        actions = result['actions'][0].cpu().tolist()
        plan = plan_from_actions(item, [a for a in actions if a >= 0])
        export_validation_s = time.perf_counter()-start
        if not np.isclose(result['cost'][0].item(), plan['objective_length_plus_stops_m'], atol=2e-4, rtol=1e-5):
            raise ValueError('environment cost differs from independently exported route')
        plans['attention_greedy'] = plan
        rows.append(metric(item, plan, solver_s, export_validation_s, feature_s))
        output = export_instance(item, plans)
        baseline.write_json(output_dir/f'scene_plan_{scene["nx"]}x{scene["ny"]}_seed{scene["seed"]}.json', output)
        schedules.extend(baseline.schedule_rows(output))
        if not samples or (samples[-1]['scene']['nx'], samples[-1]['scene']['ny']) != (scene['nx'], scene['ny']):
            samples.append(output)
    baseline.write_csv(output_dir/'metrics.csv', rows)
    baseline.write_csv(output_dir/'service_schedule.csv', schedules, baseline.SCHEDULE_COLUMNS)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['scale'], row['method'])].append(row)
    summary = [dict(scale=scale, method=method, instances=len(group),
                    avg_objective_m=float(np.mean([r['objective_m'] for r in group])),
                    avg_route_m=float(np.mean([r['route_length_m'] for r in group])),
                    avg_stops=float(np.mean([r['stops'] for r in group])),
                    avg_solver_s=float(np.mean([r['solver_s'] for r in group])),
                    avg_total_s=float(np.mean([r['total_s'] for r in group])),
                    all_valid=all(r['valid'] for r in group), all_covered=all(r['all_tasks_served'] for r in group))
               for (scale, method), group in groups.items()]
    report = dict(instances=len(instances), plans=len(rows), all_valid=all(r['valid'] for r in rows),
                  all_tasks_served=all(r['all_tasks_served'] for r in rows), summary=summary,
                  timing='One instance at a time, warmed model; CUDA synchronized. Total includes scene/F/navigation, '
                         'features/transfer, inference/solve and export/check. Excludes teacher generation for learner, '
                         'model loading, imports, plots and disk I/O. teacher_replayed solver_s includes its replay/check.')
    baseline.write_json(output_dir/'summary.json', report)
    if plots:
        save_route_plots(samples, output_dir)
    return report


def metric(item, plan, solver_s, validation_s, feature_s=0.):
    scene, check = item.scene, plan['validation']
    return dict(scale=f'{scene["nx"]}x{scene["ny"]}', seed=scene['seed'], method=plan['method'],
                tasks=len(scene['tasks']), served_tasks=len(plan['assignments']),
                stops=len(plan['visit_order_candidate_ids']), route_length_m=plan['route_length_m'],
                objective_m=plan['objective_length_plus_stops_m'], preprocessing_s=item.preprocessing_s,
                feature_s=feature_s, solver_s=solver_s, export_validation_s=validation_s,
                total_s=item.preprocessing_s+feature_s+solver_s+validation_s,
                valid=check['valid_model_plan'], all_tasks_served=check['all_tasks_served'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scales', nargs='+', type=baseline.scale_argument, default=[(2, 2), (3, 2), (4, 3)])
    parser.add_argument('--seeds', type=baseline.positive_int, default=4)
    parser.add_argument('--seed-start', type=int, default=1000000)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args(argv)
    if args.seed_start < 0:
        parser.error('--seed-start must be nonnegative')
    if len(set(args.scales)) != len(args.scales):
        parser.error('--scales must not contain duplicates')
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error('output directory must be empty or new')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu')
    torch.set_num_threads(4)
    model, saved = load_policy(args.checkpoint, device)
    config = saved['scene_config']
    items = [make_instance(nx, ny, args.seed_start+i, penalty=saved['stop_penalty'], **config)
             for nx, ny in args.scales for i in range(args.seeds)]
    report = benchmark(model, items, args.output, device, not args.no_plots)
    print(json.dumps({k: v for k, v in report.items() if k != 'summary'}))
    return 0 if report['all_valid'] and report['all_tasks_served'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
