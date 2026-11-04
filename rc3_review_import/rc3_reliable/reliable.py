"""Validated incumbents for the fixed-height tabletop proxy planning problem.

This adapter adds no learning and makes no real-robot or hard-real-time claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import numbers
from pathlib import Path
import time

import numpy as np

from rc3_baseline import baseline
from rc3_baseline.validator import validate
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import DEFAULT_RECIPE, RECIPES, prepare, solve_route, export_plan, two_opt

FIXED_ORDER = (1, 0, 7)
SCOPE = ('Fixed-height spherical-sector independent-arm proxy; collision-checked '
         'static undirected corridor roadmap; no IK, arm collision, dynamics, or hardware certification.')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return (not isinstance(value, (bool, np.bool_))
            and isinstance(value, numbers.Real) and math.isfinite(value))


def _scene_contract(scene):
    """Reject unsupported schemas before positional IDs reach the copied solver."""
    _require(isinstance(scene, dict), 'scene must be an object')
    for key in ('tasks', 'tables', 'candidates'):
        records = scene[key]
        _require(isinstance(records, list), f'{key} must be a list')
        _require([r['id'] for r in records] == list(range(len(records)))
                 and all(type(r['id']) is int for r in records),
                 f'{key} IDs must be consecutive integers starting at zero')
    table_ids = set(range(len(scene['tables'])))
    for task in scene['tasks']:
        _require(task['table_id'] in table_ids, 'unknown task table_id')
        _require(all(_finite(task[k]) for k in ('x', 'y', 'z')), 'nonfinite task pose')
    robot = scene['robot']
    keys = ('base_radius', 'clearance', 'arm_forward_offset', 'arm_spacing',
            'arm_min_reach', 'arm_max_reach', 'arm_sector_half_angle_deg',
            'shoulder_height', 'fixed_lift_height')
    _require(all(_finite(robot[k]) for k in keys), 'nonfinite or missing robot parameter')
    _require(robot['base_radius'] >= 0 and robot['clearance'] >= 0
             and robot['arm_spacing'] >= 0 and robot['fixed_lift_height'] >= 0
             and 0 <= robot['arm_min_reach'] < robot['arm_max_reach']
             and 0 < robot['arm_sector_half_angle_deg'] <= 180,
             'invalid proxy geometry')
    for candidate in scene['candidates']:
        _require(all(_finite(candidate[k]) for k in ('x', 'y', 'theta', 'h')),
                 'nonfinite candidate pose')
        _require(math.isclose(candidate['h'], robot['fixed_lift_height'], abs_tol=1e-9, rel_tol=0),
                 'variable lift is unsupported: candidate h must equal fixed_lift_height metadata')
        point = candidate['corridor_anchor']
        _require(len(point) == 2 and all(_finite(v) for v in point), 'invalid corridor anchor')
    for rectangle in [*scene['tables'], *scene.get('obstacles', [])]:
        _require(all(_finite(rectangle[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax')),
                 'nonfinite obstacle rectangle')
        _require(rectangle['xmin'] < rectangle['xmax'] and rectangle['ymin'] < rectangle['ymax'],
                 'empty or inverted obstacle rectangle')
    _require(len(scene['start']) == 2 and all(_finite(v) for v in scene['start']), 'invalid depot')
    for key in ('x_lines', 'y_lines'):
        _require(bool(scene[key]) and all(_finite(v) for v in scene[key]), 'invalid corridor lines')


def _task_ids(scene):
    if not isinstance(scene, dict) or not isinstance(scene.get('tasks'), list):
        return []
    return [task.get('id') if isinstance(task, dict) else None for task in scene['tasks']]


def _failure(scene, started, code, detail, *, precheck=None, trace=None, sparse=None):
    return dict(status='failed', code=code, detail=detail, scene=scene, feasibility=sparse or [],
                plans={}, task_ids=_task_ids(scene), pending_task_ids=_task_ids(scene),
                precheck=precheck, trace=trace or [],
                timing=dict(total_s=time.perf_counter()-started),
                certificate=dict(validated_complete_plan=False, scope=SCOPE,
                                 no_physical_infeasibility_claim=True,
                                 task_records_deleted=0, execution_attempted=False,
                                 fallback_used=False))


def _certify(problem, plan):
    """Do not trust a saved validation flag; recompute the final exported plan."""
    check = validate(problem.scene, plan, problem.sparse)
    if not check['valid_model_plan'] or not check['all_tasks_served']:
        raise ValueError(f'candidate rejected by final validation: {check["errors"]}')
    objective = check['independent_polyline_length_m'] + problem.penalty * len(plan['selected_candidate_ids'])
    _require(math.isfinite(objective), 'candidate has nonfinite objective')
    _require(math.isclose(objective, plan['objective_length_plus_stops_m'], rel_tol=1e-8, abs_tol=1e-7),
             'independent objective differs from exported objective')
    plan['validation'] = check
    return float(objective)


def plan_reliably(scene, *, mode='fast', reference='fixed1', max_recipes=None,
                  suggested_recipes=None, stop_penalty=0.75):
    """Return a certified incumbent or a structured failure, retaining every task.

    max_recipes counts complete candidate attempts, including the reference.
    suggested_recipes may be an iterable or a callable receiving a scene copy.
    Suggestions can only append candidates after the fixed prescribed sequence.
    """
    started = time.perf_counter()
    trace = []
    try:
        _require(mode in ('fast', 'quality'), 'mode must be fast or quality')
        _require(reference in ('fixed1', 'legacy'), 'reference must be fixed1 or legacy')
        _require(_finite(stop_penalty) and stop_penalty >= 0, 'invalid stop penalty')
        _scene_contract(scene)
        fixed = list(FIXED_ORDER[:1] if mode == 'fast' else FIXED_ORDER)
        labels = fixed if reference == 'fixed1' else ['legacy', DEFAULT_RECIPE, *fixed]
        if max_recipes is None:
            max_recipes = len(labels)
        _require(type(max_recipes) is int and 1 <= max_recipes <= 14,
                 'max_recipes must be an integer in 1..14')
    except Exception as error:
        return _failure(scene, started, 'input_rejected', str(error))

    try:
        problem = prepare(scene, penalty=stop_penalty)
    except Exception as error:
        return _failure(scene, started, 'preprocessing_failed', str(error))
    cover = problem.F.any(axis=2)
    navigable = np.isfinite(problem.nav['costs'][0, 1:]) & np.isfinite(problem.nav['costs'][1:, 0])
    arm_feasible = cover.any(axis=1)
    jointly_feasible = (cover & navigable[None, :]).any(axis=1)
    no_service = np.flatnonzero(~arm_feasible).tolist()
    disconnected = np.flatnonzero(arm_feasible & ~jointly_feasible).tolist()
    precheck = dict(tasks=len(scene['tasks']), candidates=len(scene['candidates']),
                    arm_feasible_tasks=int(arm_feasible.sum()),
                    navigable_serviceable_tasks=int(jointly_feasible.sum()),
                    no_service_candidate_task_ids=no_service, disconnected_task_ids=disconnected,
                    all_tasks_have_navigable_service_candidate=bool(jointly_feasible.all()),
                    interpretation='Only the current finite candidate set, proxy, and roadmap are tested.')
    if no_service or disconnected:
        code = ('mixed_task_precheck_failure' if no_service and disconnected
                else 'no_service_candidate' if no_service else 'disconnected')
        return _failure(scene, started, code,
                        'No complete plan is available in the declared candidate/proxy/roadmap model; '
                        'this is not a proof of global physical infeasibility.',
                        precheck=precheck, sparse=problem.sparse)

    def generate(label):
        if label == 'legacy':
            return baseline.solve(scene, problem.F, problem.arm_dist, problem.nav,
                                  'coverage_cost_2opt', stop_penalty, two_opt_fn=two_opt)
        route = solve_route(problem, recipe=label)
        return export_plan(problem, route, f'reliable_recipe_{label}', validate_output=False)

    incumbent = None
    incumbent_j = math.inf
    winner = None
    attempted = 0
    reference_j = None
    if not len(scene['tasks']):
        try:
            incumbent = export_plan(problem, [0, 0], 'empty_task_set', validate_output=False)
            incumbent_j = _certify(problem, incumbent)
        except Exception as error:
            return _failure(scene, started, 'empty_plan_validation_failed', str(error),
                            precheck=precheck, sparse=problem.sparse)
        reference_j, winner = incumbent_j, 'empty_task_set'
        trace.append(dict(event='empty_task_set', validated=True, incumbent_objective_m=incumbent_j))
    else:
        # Always produce and validate the anchor before consulting optional advice.
        anchor = labels[0]
        candidate_started = time.perf_counter()
        attempted += 1
        try:
            incumbent = generate(anchor)
            incumbent_j = _certify(problem, incumbent)
            reference_j, winner = incumbent_j, anchor
            trace.append(dict(event='candidate', recipe=anchor, accepted=True, validated=True,
                              objective_m=incumbent_j, incumbent_before_m=None,
                              incumbent_after_m=incumbent_j,
                              elapsed_s=time.perf_counter()-candidate_started))
        except Exception as error:
            trace.append(dict(event='anchor_failed', recipe=anchor, error=str(error)))
            return _failure(scene, started, 'anchor_failed', str(error), precheck=precheck,
                            trace=trace, sparse=problem.sparse)

        if suggested_recipes is not None and max_recipes > len(labels):
            try:
                advice = suggested_recipes(copy.deepcopy(scene)) if callable(suggested_recipes) else suggested_recipes
                _require(isinstance(advice, (list, tuple, np.ndarray)), 'suggestions must be a sequence of recipe IDs')
                for value in advice:
                    if (not isinstance(value, (bool, np.bool_)) and isinstance(value, numbers.Integral)
                            and 0 <= int(value) < len(RECIPES)):
                        if int(value) not in labels:
                            labels.append(int(value))
                    else:
                        trace.append(dict(event='suggestion_rejected', value=repr(value)))
            except Exception as error:
                trace.append(dict(event='suggestion_failed', error=str(error)))

        for label in labels[1:]:
            if attempted >= max_recipes:
                break
            attempted += 1
            candidate_started = time.perf_counter()
            before = incumbent_j
            try:
                candidate = generate(label)
                candidate_j = _certify(problem, candidate)
                accepted = candidate_j < incumbent_j - 1e-9
                if accepted:
                    incumbent, incumbent_j, winner = candidate, candidate_j, label
                trace.append(dict(event='candidate', recipe=label, accepted=accepted, validated=True,
                                  objective_m=candidate_j, incumbent_before_m=before,
                                  incumbent_after_m=incumbent_j,
                                  elapsed_s=time.perf_counter()-candidate_started))
            except Exception as error:
                trace.append(dict(event='candidate_failed', recipe=label, error=str(error),
                                  incumbent_before_m=before, incumbent_after_m=incumbent_j,
                                  elapsed_s=time.perf_counter()-candidate_started))

    candidate_hash = hashlib.sha256(json.dumps(scene, sort_keys=True, separators=(',', ':'),
                                              allow_nan=False).encode()).hexdigest()
    reference_label = 'fixed1_recipe_1' if reference == 'fixed1' else 'legacy_coverage_cost_2opt'
    certificate = dict(validated_complete_plan=True, scope=SCOPE, reference=reference_label,
                       reference_objective_m=reference_j, returned_objective_m=incumbent_j,
                       cost_not_worse_than_reference=bool(incumbent_j <= reference_j),
                       fallback_used=any(item['event'] in ('candidate_failed', 'suggestion_failed', 'suggestion_rejected')
                                         for item in trace),
                       winner=winner, scene_sha256=candidate_hash, attempts=attempted,
                       task_records_deleted=0, task_ids=_task_ids(scene),
                       no_hard_realtime_claim=True, no_global_optimality_claim=True,
                       conditional_completeness='Static finite proxy coverage; a bidirectionally depot-reachable '
                       'candidate for every task; independent tasks; no time, capacity, or resource constraints.',
                       cost_bound='Only against the named, validated anchor on this identical instance and objective.')
    return dict(status='complete', code='empty_task_set' if not len(scene['tasks']) else 'validated_plan',
                scene=scene, feasibility=problem.sparse, plans={'reliable': incumbent},
                task_ids=_task_ids(scene), pending_task_ids=[], precheck=precheck,
                certificate=certificate, trace=trace,
                timing=dict(preprocessing_s=problem.preprocessing_s,
                            total_s=time.perf_counter()-started,
                            scope='New CPU wall time for this adapter call, including preprocessing, '
                            'all attempted searches and validations; excludes process imports and JSON file I/O.'),
                settings=dict(mode=mode, reference=reference, max_recipes=max_recipes,
                              fixed_order=list(FIXED_ORDER), stop_penalty=stop_penalty))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--scale', default='10x10')
    parser.add_argument('--seed', type=int, default=3700000000)
    parser.add_argument('--scene-mode', choices=('regular', 'mixed', 'blocked'), default='blocked')
    parser.add_argument('--mode', choices=('fast', 'quality'), default='fast')
    parser.add_argument('--reference', choices=('fixed1', 'legacy'), default='fixed1')
    parser.add_argument('--max-recipes', type=int)
    parser.add_argument('--suggest', nargs='*', type=int)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.scene:
        source = json.loads(args.scene.read_text(encoding='utf-8-sig'))
        scene = source.get('scene', source)
    else:
        nx, ny = map(int, args.scale.lower().split('x'))
        scene = make_complex_scene(nx, ny, args.seed, args.scene_mode)
    result = plan_reliably(scene, mode=args.mode, reference=args.reference,
                           max_recipes=args.max_recipes, suggested_recipes=args.suggest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], code=result['code'],
                          certificate=result['certificate'], timing=result['timing']), ensure_ascii=False))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
