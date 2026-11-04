"""One verified Fixed3 anchor, provisional local search, one final certificate."""
from __future__ import annotations

import math
import time
import numpy as np

from rc3_fast.solver import export_plan, route_objective
from rc3_improve.repair import route_is_valid
from .environment import Env, FEATURE_NAMES, reference_candidates


def _certify(problem, route, method, expected):
    plan = export_plan(problem, route, method)
    cost = plan['objective_length_plus_stops_m']
    check = plan.get('validation', {})
    if not check.get('valid_model_plan') or not check.get('all_tasks_served'):
        raise ValueError('independent plan validation failed')
    if not math.isfinite(cost) or abs(cost-expected) > 1e-7:
        raise ValueError('exported and abstract costs differ')
    return plan


def run(problem, method='fixed3', policy=None, horizon=8, time_limit_s=None):
    """Soft planning deadline includes anchor construction and final validation.

    A new DP call never starts after its deadline. A running repair and mandatory
    final validation may finish after it. Only independently validated plans are
    returned; final failure returns the already validated anchor.
    """
    if method not in ('fixed1', 'fixed3', 'random', 'detour', 'learned'):
        raise ValueError('unknown planning method')
    if type(horizon) is not int or horizon < 0:
        raise ValueError('horizon must be a nonnegative integer')
    if time_limit_s is not None and (not math.isfinite(time_limit_s) or time_limit_s < 0):
        raise ValueError('time_limit_s must be finite and nonnegative')
    started = time.perf_counter()
    deadline = started+time_limit_s if time_limit_s is not None else math.inf
    timing = dict(reference_s=0., feature_s=0., model_s=0., repair_s=0., final_validation_s=0.)
    tick = time.perf_counter()
    candidates, failures = reference_candidates(problem, (1,) if method == 'fixed1' else (0, 1, 7))
    anchor, anchor_route, winning_recipe = None, None, None
    for score, recipe, route in candidates:
        try:
            anchor = _certify(problem, route, method, score)
            anchor_route, winning_recipe = list(route), recipe
            break
        except Exception as error:
            failures.append(dict(stage='certify_reference', recipe=recipe, error=str(error)))
    timing['reference_s'] = time.perf_counter()-tick
    if anchor is None:
        raise ValueError(f'no independently verified reference: {failures}')
    anchor_cost = float(anchor['objective_length_plus_stops_m'])
    certified, best_route = anchor, list(anchor_route)
    trace = [dict(event='validated_reference', reference='fixed1' if method == 'fixed1' else 'fixed3',
                  recipe=winning_recipe, objective_m=anchor_cost)]
    dp_calls, accepted = 0, 0
    stop_reason = 'reference_only'
    if method not in ('fixed1', 'fixed3'):
        env = Env(problem, horizon, initial_route=anchor_route)
        rng = np.random.default_rng(int(problem.scene.get('seed', 0))+82991)
        stop_reason = 'horizon' if horizon == 0 else 'no_legal_action'
        while env.steps < horizon:
            if time.perf_counter() >= deadline:
                stop_reason = 'deadline'; break
            previous = list(env.route)
            try:
                tick = time.perf_counter()
                try:
                    features, mask = env.observe()
                finally:
                    timing['feature_s'] += time.perf_counter()-tick
                if env.done or not mask.any():
                    stop_reason = 'no_legal_action'; break
                if time.perf_counter() >= deadline:
                    stop_reason = 'deadline'; break
                tick = time.perf_counter()
                try:
                    if method == 'random':
                        action = int(rng.choice(np.flatnonzero(mask)))
                    elif method == 'detour':
                        action = int(np.where(mask, features[:, FEATURE_NAMES.index('detour_excess')], -np.inf).argmax())
                    else:
                        if policy is None:
                            raise ValueError('learned method requires a policy')
                        if hasattr(policy, 'select'):
                            action = policy.select(features.copy(), mask.copy())
                        else:
                            scores = np.asarray(policy.scores(features.copy(), mask.copy()))
                            if scores.shape != mask.shape or not np.isfinite(scores[mask]).all():
                                raise ValueError('invalid model scores')
                            action = int(np.where(mask, scores, -np.inf).argmax())
                finally:
                    timing['model_s'] += time.perf_counter()-tick
                if action is None:
                    stop_reason = 'policy_stop'
                    trace.append(dict(event='policy_stop', after_dp_calls=dp_calls)); break
                if not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)) or not 0 <= action < len(mask) or not mask[action]:
                    raise ValueError('policy selected an invalid or masked repair action')
                # Model/feature time can consume the remainder of the budget.
                if time.perf_counter() >= deadline:
                    stop_reason = 'deadline'; break
                dp_calls += 1
                tick = time.perf_counter()
                try:
                    _, done, info = env.step(int(action))
                finally:
                    timing['repair_s'] += time.perf_counter()-tick
                current = route_objective(problem, env.route)
                before = route_objective(problem, previous)
                if not route_is_valid(problem, env.route) or not math.isfinite(current) or current > before+1e-8:
                    raise ValueError('provisional route violates coverage or nonincrease guard')
                if current < before-1e-8:
                    best_route = list(env.route); accepted += 1
                trace.append(dict(event='provisional_repair', action=int(action), accepted=bool(info['accepted']),
                                  objective_m=float(current), independently_certified=False,
                                  elapsed_s=time.perf_counter()-started))
                if done:
                    stop_reason = 'horizon' if env.steps >= horizon else 'no_legal_action'; break
            except Exception as error:
                env.route = previous
                failures.append(dict(stage='proposal', error=str(error)))
                stop_reason = 'proposal_failure'; break
        else:
            stop_reason = 'horizon'
    best_cost = route_objective(problem, best_route)
    if best_cost < anchor_cost-1e-8:
        tick = time.perf_counter()
        try:
            candidate = _certify(problem, best_route, method, best_cost)
            if candidate['objective_length_plus_stops_m'] > anchor_cost+1e-8:
                raise ValueError('final plan worsens the verified anchor')
            certified = candidate
            trace.append(dict(event='validated_final', objective_m=float(best_cost)))
        except Exception as error:
            failures.append(dict(stage='certify_final', error=str(error)))
            trace.append(dict(event='validated_anchor_fallback', objective_m=anchor_cost))
        finally:
            timing['final_validation_s'] += time.perf_counter()-tick
    elapsed = time.perf_counter()-started
    objective = float(certified['objective_length_plus_stops_m'])
    overrun = max(0., elapsed-time_limit_s) if time_limit_s is not None else 0.
    return certified, dict(status='complete', method=method,
        reference='fixed1' if method == 'fixed1' else 'fixed3', reference_objective_m=anchor_cost,
        winning_reference_recipe=winning_recipe, objective_m=objective,
        cost_nonincreasing=objective <= anchor_cost+1e-8, attempts=dp_calls, dp_calls=dp_calls,
        accepted=accepted, provisional_improvements=accepted, fallback_used=bool(failures),
        failures=failures, stop_reason=stop_reason, budget_exhausted=stop_reason in ('deadline', 'horizon'),
        deadline_exceeded=overrun>0., time_overrun_s=overrun, time_limit_s=time_limit_s,
        planning_s=elapsed, elapsed_s=elapsed, **timing, trace=trace,
        guarantee_scope='Only the returned independently verified plan is certified. Cost is bounded by the verified reference under the discrete proxy model. Deadline is soft, not hard real time.')
