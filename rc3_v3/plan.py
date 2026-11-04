"""Reusable planner; default fast Fixed3, optional frozen gain-supervised policy."""
import argparse
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
import numpy as np
from rc3_baseline.baseline import write_json, write_csv, schedule_rows, SCHEDULE_COLUMNS, scale_argument
from rc3_fast.planner import map_key
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare
from rc3_improve.evaluate import safe_json
from .runtime import run

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent/'deployment'/'policy.npz'


class Planner:
    def __init__(self, checkpoint=None, method='fixed3', horizon=8, cache_size=2, time_limit_s=None):
        if method not in ('fixed3', 'random', 'detour', 'learned') or type(horizon) is not int or horizon < 0 or type(cache_size) is not int or cache_size < 0:
            raise ValueError('invalid method, repair budget or cache size')
        if time_limit_s is not None and (not math.isfinite(time_limit_s) or time_limit_s < 0):
            raise ValueError('time_limit_s must be finite and nonnegative')
        self.policy = None
        if method == 'learned':
            from .model import load_policy
            self.policy = load_policy(checkpoint or DEFAULT_CHECKPOINT)
        self.method, self.horizon, self.cache_size, self.time_limit_s = method, horizon, cache_size, time_limit_s
        self.cache = OrderedDict()

    def plan(self, scene):
        started = time.perf_counter()
        code, ids = 'unsupported_input', []
        try:
            if not isinstance(scene, dict):
                raise ValueError('scene must be a mapping')
            ids = [t.get('id') for t in scene.get('tasks', []) if isinstance(t, dict)]
            for kind in ('tasks', 'tables', 'candidates'):
                if [v['id'] for v in scene[kind]] != list(range(len(scene[kind]))) or any(type(v['id']) is not int for v in scene[kind]):
                    raise ValueError(f'{kind} IDs must be contiguous integers from zero')
            for task in scene['tasks']:
                if type(task['table_id']) is not int or not 0 <= task['table_id'] < len(scene['tables']):
                    raise ValueError('invalid task table ID')
                if not np.isfinite([task[k] for k in ('x','y','z')]).all():
                    raise ValueError('nonfinite task')
            for candidate in scene['candidates']:
                if not np.isfinite([candidate[k] for k in ('x','y','theta','h')]).all():
                    raise ValueError('nonfinite candidate')
                if abs(candidate['h']-scene['robot']['fixed_lift_height']) > 1e-9:
                    raise ValueError('variable lift unsupported by current proxy')
            key = map_key(scene)
            hit = key in self.cache
            problem = prepare(scene, nav=self.cache.get(key))
            if self.cache_size:
                self.cache[key] = problem.nav
                self.cache.move_to_end(key)
                while len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
            cover = problem.F.any(2)
            missing = np.flatnonzero(~cover.any(1)).tolist()
            connected = np.isfinite(problem.nav['costs'][0,1:]) & np.isfinite(problem.nav['costs'][1:,0])
            disconnected = np.flatnonzero(cover.any(1) & ~(cover & connected).any(1)).tolist()
            if missing:
                code = 'no_service_candidate'; raise ValueError(f'tasks: {missing}')
            if disconnected:
                code = 'disconnected'; raise ValueError(f'tasks: {disconnected}')
            code = 'solver_failure'
            plan, info = run(problem, self.method, self.policy, self.horizon, self.time_limit_s)
            return dict(status='complete', scene=scene, feasibility=problem.sparse, plans={self.method:plan},
                diagnostics=info, timing=dict(preprocessing_s=problem.preprocessing_s, planning_s=info['planning_s'],
                    total_s=time.perf_counter()-started, cache_hit=hit),
                guarantee='Complete independently validated proxy plan; objective <= verified Fixed3 anchor. Soft deadline only.')
        except Exception as error:
            return safe_json(dict(status='failed', code=code, detail=str(error), scene=scene, plans={},
                task_ids=ids, pending_task_ids=ids, physical_infeasibility_claim=False, total_s=time.perf_counter()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['fixed3','random','detour','learned'], default='fixed3')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--scale', type=scale_argument, default=(10,10))
    parser.add_argument('--mode', choices=['regular','mixed','blocked'], default='blocked')
    parser.add_argument('--seed', type=int, default=771901)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--time-limit-ms', type=float)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text(encoding='utf-8')) if args.scene else make_complex_scene(*args.scale,args.seed,args.mode)
    scene = scene.get('scene',scene)
    planner = Planner(args.checkpoint,args.method,args.budget,time_limit_s=None if args.time_limit_ms is None else args.time_limit_ms/1000)
    result = planner.plan(scene)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,safe_json(result))
    if result['status']=='complete':
        write_csv(args.output.with_suffix('.csv'),list(schedule_rows(result)),SCHEDULE_COLUMNS)
        print(json.dumps(dict(status='complete',timing=result['timing'],objective=result['diagnostics']['objective_m'],
            reference=result['diagnostics']['reference_objective_m'],calls=result['diagnostics']['dp_calls'],
            stop_reason=result['diagnostics']['stop_reason'])),flush=True)
    else:
        print(json.dumps({k:result[k] for k in ('status','code','detail','pending_task_ids')}),flush=True)
        raise SystemExit(2)


if __name__=='__main__':
    main()
