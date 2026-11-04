"""Shared fast nonlearning building blocks and coverage-preserving local search."""
from dataclasses import dataclass
import math
import time

import numpy as np

from rc3_baseline import baseline
from rc3_baseline.validator import validate


RECIPES = [dict(name='nearest', alpha=0., beta=.75)] + [
    dict(name=f'gain{a:g}_offset{b:g}', alpha=a, beta=b)
    for a in (.5, 1., 2., 4.) for b in (.75, 2., 6.)]
DEFAULT_RECIPE = next(i for i, r in enumerate(RECIPES) if r['alpha'] == 1 and r['beta'] == .75)


@dataclass
class Problem:
    scene: dict
    F: np.ndarray
    arm_dist: np.ndarray
    sparse: list
    nav: dict
    preprocessing_s: float
    penalty: float = .75


def vectorized_feasibility(scene):
    n, m = len(scene['tasks']), len(scene['candidates'])
    tasks = np.asarray([[t['x'], t['y'], t['z']] for t in scene['tasks']], float).reshape(n, 3)
    docks = np.asarray([[c['x'], c['y'], c['theta']] for c in scene['candidates']], float).reshape(m, 3)
    robot = scene['robot']
    dx, dy = tasks[:, None, 0]-docks[None, :, 0], tasks[:, None, 1]-docks[None, :, 1]
    cos, sin = np.cos(docks[:, 2]), np.sin(docks[:, 2])
    x = cos*dx+sin*dy-robot['arm_forward_offset']
    y = (-sin*dx+cos*dy)[..., None]-np.array([1., -1.])*robot['arm_spacing']/2
    z = tasks[:, None, 2]-robot['shoulder_height']
    distance = np.sqrt(x[..., None]**2+y**2+z[..., None]**2)
    matrix = ((distance >= robot['arm_min_reach']) & (distance <= robot['arm_max_reach'])
              & (np.abs(np.arctan2(y, x[..., None])) <= math.radians(robot['arm_sector_half_angle_deg'])))
    sparse = [dict(task_id=i, left=np.flatnonzero(matrix[i, :, 0]).tolist(),
                   right=np.flatnonzero(matrix[i, :, 1]).tolist()) for i in range(n)]
    return matrix, distance, sparse


def prepare(scene, penalty=.75, nav=None):
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError('penalty must be finite and nonnegative')
    start = time.perf_counter()
    matrix, distance, sparse = vectorized_feasibility(scene)
    if nav is None:
        nav = baseline.navigation(scene)
    return Problem(scene, matrix, distance, sparse, nav, time.perf_counter()-start, penalty)


def two_opt(route, costs, passes=60):
    route = np.asarray(route, dtype=np.int64).copy()
    i, j = np.triu_indices(len(route)-2, 1)
    i, j = i+1, j+1
    if not len(i):
        return route.tolist()
    for _ in range(passes):
        change = (costs[route[i-1], route[j]]+costs[route[i], route[j+1]]
                  -costs[route[i-1], route[i]]-costs[route[j], route[j+1]])
        best = int(change.argmin())
        if change[best] >= -1e-9:
            break
        lo, hi = i[best], j[best]
        route[lo:hi+1] = route[lo:hi+1][::-1]
    return route.tolist()


def construct(problem, recipe=DEFAULT_RECIPE):
    config = RECIPES[recipe] if isinstance(recipe, int) else recipe
    cover, costs = problem.F.any(2), problem.nav['costs']
    left = np.ones(len(cover), dtype=bool)
    route = [0]
    navigable = np.isfinite(costs[0, 1:]) & np.isfinite(costs[1:, 0])
    while left.any():
        gains = cover[left].sum(0)
        travel = costs[route[-1], 1:]
        mask = (gains > 0) & navigable & np.isfinite(travel)
        if not mask.any():
            raise ValueError('infeasible scene: uncovered tasks have no accessible candidate')
        if config['alpha'] == 0:
            score = np.where(mask, -travel, -np.inf)
        else:
            score = np.where(mask, gains.astype(float)**config['alpha']/(travel+config['beta']), -np.inf)
        j = int(score.argmax())
        left &= ~cover[:, j]
        route.append(j+1)
    return route+[0]


def joint_search(route, cover, costs, passes=2):
    """Replace service candidates and remove redundant stops without losing coverage."""
    route = list(route)
    if len(route) <= 2:
        return route
    count = cover[:, np.asarray(route[1:-1])-1].sum(1).astype(np.int64)
    reachable = np.isfinite(costs[0, 1:]) & np.isfinite(costs[1:, 0])
    for _ in range(passes):
        changed, position = False, 1
        while position < len(route)-1:
            old = route[position]-1
            unique = cover[:, old] & (count == 1)
            if not unique.any():
                count -= cover[:, old]
                route.pop(position)
                changed = True
                continue
            allowed = cover[unique].all(0) & reachable
            allowed[np.asarray(route[1:-1])-1] = False
            allowed[old] = True
            previous, following = route[position-1], route[position+1]
            lengths = costs[previous, 1:]+costs[1:, following]
            best = int(np.where(allowed, lengths, np.inf).argmin())
            if lengths[best] < lengths[old]-1e-9:
                count += cover[:, best].astype(np.int64)-cover[:, old]
                route[position] = best+1
                changed = True
            position += 1
        route = two_opt(route, costs)
        if not changed:
            break
    return route


def solve_route(problem, recipe=DEFAULT_RECIPE, search_passes=2):
    route = two_opt(construct(problem, recipe), problem.nav['costs'])
    if search_passes:
        route = joint_search(route, problem.F.any(2), problem.nav['costs'], search_passes)
    left = np.ones(len(problem.F), dtype=bool)
    cover, kept = problem.F.any(2), [0]
    for node in route[1:-1]:
        if (left & cover[:, node-1]).any():
            kept.append(node)
            left &= ~cover[:, node-1]
    return kept+[0]


def route_objective(problem, route):
    return float(sum(problem.nav['costs'][a, b] for a, b in zip(route, route[1:]))+problem.penalty*(len(route)-2))


def export_plan(problem, route, method):
    cover, scene = problem.F.any(2), problem.scene
    left = np.ones(len(cover), dtype=bool)
    selected, assignments = [], []
    for node in route[1:-1]:
        j = int(node)-1
        served = np.flatnonzero(left & cover[:, j])
        if not len(served):
            continue
        selected.append(j)
        for i in served:
            arm = int(np.argmin(np.where(problem.F[i, j], problem.arm_dist[i, j], np.inf)))
            assignments.append(dict(task_id=int(i), candidate_id=j, arm='L' if arm == 0 else 'R'))
        left[served] = False
    if left.any():
        raise ValueError('export would leave tasks unserved')
    route = [0]+[j+1 for j in selected]+[0]
    polyline, legs = baseline.expand_route(route, problem.nav)
    stops, arrival = [], 0.
    for k, j in enumerate(selected):
        arrival += legs[k]
        stops.append(dict(sequence=k+1, candidate_id=j,
                          pose={key: scene['candidates'][j][key] for key in ('x', 'y', 'theta', 'h')},
                          arrival_distance_m=arrival,
                          tasks=[dict(task_id=a['task_id'], arm=a['arm']) for a in assignments if a['candidate_id'] == j]))
    plan = dict(method=method, status='complete', selected_candidate_ids=selected,
                visit_order_candidate_ids=selected, assignments=assignments, service_stops=stops,
                unserved_task_ids=[], unserved_reasons=[], route_polyline=polyline,
                route_leg_costs_m=legs, route_length_m=sum(legs), stop_penalty_equivalent_m=problem.penalty,
                objective_length_plus_stops_m=sum(legs)+problem.penalty*len(selected))
    plan['validation'] = validate(scene, plan, problem.sparse)
    if not plan['validation']['valid_model_plan']:
        raise ValueError(plan['validation']['errors'])
    return plan
