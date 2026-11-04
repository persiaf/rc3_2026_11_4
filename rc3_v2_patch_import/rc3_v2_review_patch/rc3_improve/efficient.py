"""Fast planning with a verified reference and deferred final certification.

This additive adapter leaves the uploaded v2 experiment code unchanged. During
search, routes are provisional; only fully exported, independently validated
plans can be returned. A failed final candidate falls back to a previously
verified reference. The budget counts repair calls, not wall-clock time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from rc3_baseline.baseline import write_json, write_csv, schedule_rows, SCHEDULE_COLUMNS, scale_argument
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare, solve_route, route_objective, export_plan
from .environment import ImprovementEnv, canonical_route
from .evaluate import safe_json
from .repair import FEATURE_NAMES, route_is_valid, _local


class FreshAttemptEnv(ImprovementEnv):
    """Invalidate attempted windows only when their local repair inputs change."""

    def __init__(self, problem, horizon=8, initial_route=None, refresh_attempts=True):
        if initial_route is None:
            initial_route = solve_route(problem, 1)
        self.problem, self.horizon = problem, horizon
        self.initial_route = canonical_route(problem, initial_route)
        self.route = list(self.initial_route)
        self.reference_cost = route_objective(problem, self.route)
        self.steps, self.attempted, self.done = 0, set(), False
        self._key = None
        self.refresh_attempts = refresh_attempts
        self._context_key = None

    def observe(self):
        features, mask = super().observe()
        if not self.refresh_attempts:
            return features, mask
        key = tuple(self.route)
        if key != self._context_key:
            cover = self.problem.F.any(2)
            counts = cover[:, np.asarray(self.route[1:-1], dtype=np.int64)-1].sum(1)
            self._signatures = []
            for window in self.windows:
                start, end, required, pool, _ = _local(self.problem, self.route, window, cover, counts)
                # For the static F/D instance, these are all inputs to the fixed-
                # endpoint DP. Nested tuples avoid separator/ID ambiguity.
                self._signatures.append((tuple(self.route[start-1:end+1]),
                                         tuple(required.tolist()), tuple(pool.tolist())))
            self._context_key = key
        mask = np.asarray([signature not in self.attempted for signature in self._signatures], dtype=bool)
        if len(mask):
            mask &= features[:, -1] > 0
        self.done = self.steps >= self.horizon or not mask.any()
        return features, mask

    def step(self, action):
        # Only a proven no-change subproblem may be memoized. A successful
        # context can recur after an overlapping repair and must remain usable.
        self.observe()
        signature = self._signatures[action] if isinstance(action, (int, np.integer)) and 0 <= action < len(self._signatures) else None
        reward, done, info = super().step(action)
        if self.refresh_attempts and info['accepted']:
            self.attempted.discard(signature)
            self.observe()
            done = self.done
        return reward, done, info


def _verified_reference(problem, reference, method):
    """Compare cheap route costs, then certify in cost order until one succeeds."""
    recipes = (1,) if reference == 'fixed1' else (1, 0, 7)
    routes, failures = [], []
    for recipe in recipes:
        try:
            route = canonical_route(problem, solve_route(problem, recipe))
            routes.append((route_objective(problem, route), recipe, route))
        except Exception as error:
            failures.append(dict(stage='construct_reference', recipe=recipe, error=str(error)))
    for score, recipe, route in sorted(routes, key=lambda item: item[0]):
        try:
            plan = export_plan(problem, route, method)
            cost = plan['objective_length_plus_stops_m']
            if not (plan['validation']['valid_model_plan'] and plan['validation']['all_tasks_served']):
                raise ValueError('reference failed independent completeness/validity check')
            if abs(cost-score) > 1e-7:
                raise ValueError('abstract and independently recomputed reference cost differ')
            return plan, route, recipe, failures
        except Exception as error:
            failures.append(dict(stage='certify_reference', recipe=recipe, error=str(error)))
    raise ValueError(f'no independently validated reference: {failures}')


def run_efficient(problem, method='fixed3', policy=None, horizon=8, *,
                  reference='fixed1', refresh_attempts=True, env_factory=FreshAttemptEnv):
    """Return a complete certified plan whose cost is at most its named reference.

    Fixed1/Fixed3 construct their corresponding reference and return it directly.
    Learned and nonlearned local selectors use identical repair/certification.
    Setting refresh_attempts=False isolates certification-only speed changes.
    """
    if method not in ('fixed1', 'fixed3', 'random', 'detour', 'bc', 'ppo'):
        raise ValueError('unknown method')
    if reference not in ('fixed1', 'fixed3') or type(horizon) is not int or horizon < 0:
        raise ValueError('invalid reference or repair budget')
    started = time.perf_counter()
    reference = method if method in ('fixed1', 'fixed3') else reference
    tick = time.perf_counter()
    anchor, anchor_route, winning_recipe, failures = _verified_reference(problem, reference, method)
    reference_s = time.perf_counter()-tick
    anchor_cost = anchor['objective_length_plus_stops_m']
    certified = anchor
    trace = [dict(event='validated_reference', reference=reference, recipe=winning_recipe, objective_m=anchor_cost)]
    provisional, attempts = [], 0
    timing = dict(feature_s=0., model_s=0., repair_s=0., final_validation_s=0.)
    if method not in ('fixed1', 'fixed3') and horizon:
        env = env_factory(problem, horizon, initial_route=anchor_route, refresh_attempts=refresh_attempts)
        rng = np.random.default_rng(int(problem.scene.get('seed', 0))+82991)
        for _ in range(horizon):
            previous = list(env.route)
            before = route_objective(problem, previous)
            try:
                tick = time.perf_counter()
                features, mask = env.observe()
                timing['feature_s'] += time.perf_counter()-tick
                if env.done or not mask.any():
                    break
                tick = time.perf_counter()
                if method == 'random':
                    action = int(rng.choice(np.flatnonzero(mask)))
                elif method == 'detour':
                    action = int(np.where(mask, features[:, FEATURE_NAMES.index('detour_excess')], -np.inf).argmax())
                else:
                    if policy is None:
                        raise ValueError('learned selector requires a policy')
                    scores = np.asarray(policy.scores(features.copy(), mask.copy()))
                    if scores.shape != mask.shape or not np.isfinite(scores[mask]).all():
                        raise ValueError('invalid learned scores')
                    action = int(np.where(mask, scores, -np.inf).argmax())
                timing['model_s'] += time.perf_counter()-tick
                attempts += 1
                tick = time.perf_counter()
                _, _, info = env.step(action)
                timing['repair_s'] += time.perf_counter()-tick
                if not route_is_valid(problem, env.route):
                    raise ValueError('provisional route failed combinatorial guard')
                # ImprovementEnv.step already canonicalizes every accepted
                # route. Keep one cheap coverage guard here; repeating the full
                # canonicalization would add redundant F reductions per action.
                candidate_route = list(env.route)
                after = route_objective(problem, candidate_route)
                if not np.isfinite(after) or after > before+1e-8:
                    raise ValueError('provisional route worsens objective')
                changed = after < before-1e-8
                env.route = candidate_route if changed else previous
                if changed:
                    provisional.append((after, list(candidate_route)))
                trace.append(dict(event='provisional_repair', action=action, accepted=changed,
                                  abstract_objective_m=after, independently_certified=False, info=safe_json(info)))
            except Exception as error:
                env.route = previous
                failures.append(dict(stage='proposal', error=str(error)))
                break

    # Cheap search scores are never treated as a certificate. Try the best saved
    # route first; rejected final plans cannot overwrite the verified reference.
    for score, route in sorted(provisional, key=lambda item: item[0]):
        tick = time.perf_counter()
        try:
            candidate = export_plan(problem, route, method)
            cost = candidate['objective_length_plus_stops_m']
            if not (candidate['validation']['valid_model_plan'] and candidate['validation']['all_tasks_served']):
                raise ValueError('final plan failed independent completeness/validity check')
            if abs(cost-score) > 1e-7 or cost > anchor_cost+1e-8:
                raise ValueError('final independent cost disagrees or worsens reference')
            certified = candidate
            trace.append(dict(event='validated_final', objective_m=cost))
            break
        except Exception as error:
            failures.append(dict(stage='certify_final', error=str(error)))
        finally:
            timing['final_validation_s'] += time.perf_counter()-tick
    cost = certified['objective_length_plus_stops_m']
    return certified, dict(status='complete', method=method, reference=reference,
        reference_objective_m=anchor_cost, winning_reference_recipe=winning_recipe,
        objective_m=cost, cost_nonincreasing=cost <= anchor_cost+1e-8,
        attempts=attempts, provisional_improvements=len(provisional),
        refresh_attempts=refresh_attempts, fallback_used=bool(failures), failures=failures,
        planning_s=time.perf_counter()-started, reference_s=reference_s, **timing, trace=trace,
        guarantee_scope='Fixed discrete reachability proxy and static navigation graph; final certified plan only; no optimality or hard-time guarantee.')


def plan_scene(scene, method='fixed3', policy=None, horizon=8, reference='fixed1', refresh_attempts=True):
    started = time.perf_counter()
    ids, code = [], 'unsupported_input'
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
            if not np.isfinite([task[k] for k in ('x', 'y', 'z')]).all():
                raise ValueError('nonfinite task')
        for candidate in scene['candidates']:
            if not np.isfinite([candidate[k] for k in ('x', 'y', 'theta', 'h')]).all():
                raise ValueError('nonfinite candidate')
            if abs(candidate['h']-scene['robot']['fixed_lift_height']) > 1e-9:
                raise ValueError('variable lift not supported by current reachability proxy')
        problem = prepare(scene)
        cover = problem.F.any(2)
        missing = np.flatnonzero(~cover.any(1)).tolist()
        connected = np.isfinite(problem.nav['costs'][0, 1:]) & np.isfinite(problem.nav['costs'][1:, 0])
        disconnected = np.flatnonzero(cover.any(1) & ~(cover & connected).any(1)).tolist()
        if missing:
            code = 'no_service_candidate'
            raise ValueError(f'tasks: {missing}')
        if disconnected:
            code = 'disconnected'
            raise ValueError(f'tasks: {disconnected}')
        code = 'solver_failure'
        plan, info = run_efficient(problem, method, policy, horizon, reference=reference,
                                   refresh_attempts=refresh_attempts)
        return dict(status='complete', scene=scene, feasibility=problem.sparse, plans={method: plan},
                    diagnostics=info, timing=dict(preprocessing_s=problem.preprocessing_s,
                    planning_s=info['planning_s'], total_s=time.perf_counter()-started))
    except Exception as error:
        return safe_json(dict(status='failed', code=code, detail=str(error), scene=scene, plans={},
            task_ids=ids, pending_task_ids=ids, physical_infeasibility_claim=False,
            total_s=time.perf_counter()-started))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['fixed1', 'fixed3', 'random', 'detour', 'bc', 'ppo'], default='fixed3')
    parser.add_argument('--reference', choices=['fixed1', 'fixed3'], default='fixed1')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--scale', type=scale_argument, default=(10, 10))
    parser.add_argument('--mode', choices=['regular', 'mixed', 'blocked'], default='blocked')
    parser.add_argument('--seed', type=int, default=6241030)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--legacy-attempt-mask', action='store_true', help='isolate certification-only speed change')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.budget < 0:
        parser.error('budget must be nonnegative')
    policy = None
    if args.method in ('bc', 'ppo'):
        from .numpy_policy import NumpyPolicy
        path = args.checkpoint or Path(__file__).resolve().parents[1]/'weights'/f'{args.method}.npz'
        policy = NumpyPolicy.load(path, expected_stage=args.method)
    scene = json.loads(args.scene.read_text(encoding='utf-8')) if args.scene else make_complex_scene(*args.scale, args.seed, args.mode)
    scene = scene.get('scene', scene) if isinstance(scene, dict) else scene
    output = plan_scene(scene, args.method, policy, args.budget, args.reference, not args.legacy_attempt_mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, safe_json(output))
    if output['status'] == 'complete':
        write_csv(args.output.with_suffix('.csv'), list(schedule_rows(output)), SCHEDULE_COLUMNS)
        print(json.dumps(dict(status='complete', objective=output['diagnostics']['objective_m'],
              reference=output['diagnostics']['reference_objective_m'], timing=output['timing'])))
    else:
        print(json.dumps({k: output[k] for k in ('status', 'code', 'detail', 'pending_task_ids')}))


if __name__ == '__main__':
    main()
