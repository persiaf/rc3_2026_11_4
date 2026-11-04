"""Use the existing baseline's scenes, arm proxy and exact roadmap costs."""
from dataclasses import dataclass
import math
import time

import numpy as np
import torch
from tensordict import TensorDict

from rc3_baseline import baseline
from rc3_baseline.validator import validate


@dataclass
class Instance:
    scene: dict
    feasibility: np.ndarray
    arm_distance: np.ndarray
    sparse: list
    navigation: dict
    teacher: dict | None = None
    teacher_original_stops: int = 0
    preprocessing_s: float = 0.0
    teacher_s: float = 0.0
    penalty: float = 0.75


def plan_from_actions(instance, actions, method='attention_greedy', prune=False):
    """Replay auto-service semantics; optionally remove zero-gain teacher stops."""
    scene, matrix, nav = instance.scene, instance.feasibility, instance.navigation
    cover = matrix.any(axis=2)
    remaining = np.ones(len(scene['tasks']), dtype=bool)
    route, assignments, selected = [0], [], []
    for raw_action in actions:
        j = int(raw_action)
        if not 0 <= j < len(scene['candidates']):
            raise ValueError(f'candidate out of range: {j}')
        served = np.flatnonzero(remaining & cover[:, j])
        if not len(served):
            if prune:
                continue
            raise ValueError(f'action {j} has zero new coverage')
        if not np.isfinite(nav['costs'][route[-1], j+1]) or not np.isfinite(nav['costs'][j+1, 0]):
            raise ValueError(f'action {j} cannot be reached and returned from')
        for i in served:
            arm = int(np.argmin(np.where(matrix[i, j], instance.arm_distance[i, j], np.inf)))
            assignments.append(dict(task_id=int(i), candidate_id=j, arm='L' if arm == 0 else 'R'))
        remaining[served] = False
        selected.append(j)
        route.append(j+1)
    route.append(0)
    polyline, legs = baseline.expand_route(route, nav)
    missing = np.flatnonzero(remaining).tolist()
    stops, arrival = [], 0.0
    for k, j in enumerate(selected):
        arrival += legs[k]
        stops.append(dict(sequence=k+1, candidate_id=j,
                          pose={key: scene['candidates'][j][key] for key in ('x', 'y', 'theta', 'h')},
                          arrival_distance_m=arrival,
                          tasks=[dict(task_id=a['task_id'], arm=a['arm']) for a in assignments if a['candidate_id'] == j]))
    plan = dict(method=method, status='partial' if missing else 'complete',
                selected_candidate_ids=selected, visit_order_candidate_ids=selected.copy(),
                assignments=assignments, service_stops=stops, unserved_task_ids=missing,
                unserved_reasons=[dict(task_id=i, reason='no_arm_feasible_candidate' if not cover[i].any()
                                       else 'no_navigable_feasible_candidate') for i in missing],
                route_polyline=polyline, route_leg_costs_m=legs,
                route_length_m=sum(legs), stop_penalty_equivalent_m=instance.penalty,
                objective_length_plus_stops_m=sum(legs)+instance.penalty*len(selected))
    # A prematurely stopped policy is distinct from a genuinely infeasible scene.
    navigable = np.isfinite(nav['costs'][0, 1:]) & np.isfinite(nav['costs'][1:, 0])
    if any((cover[i] & navigable).any() for i in missing):
        raise ValueError('action sequence ended while legal service actions remained')
    plan['validation'] = validate(scene, plan, instance.sparse)
    if not plan['validation']['valid_model_plan']:
        raise ValueError(plan['validation']['errors'])
    return plan


def make_instance(nx, ny, seed, penalty=0.75, geometry=None, robot=None, counts=None,
                  require_feasible=True, with_teacher=True):
    start = time.perf_counter()
    scene = baseline.make_scene(nx, ny, seed, counts=counts, geometry=geometry, robot=robot)
    matrix, arm_distance, sparse = baseline.feasibility(scene)
    nav = baseline.navigation(scene)
    item = Instance(scene, matrix, arm_distance, sparse, nav, penalty=penalty,
                    preprocessing_s=time.perf_counter()-start)
    if not with_teacher:
        if require_feasible:
            navigable = np.isfinite(nav['costs'][0, 1:]) & np.isfinite(nav['costs'][1:, 0])
            if not (matrix.any(2) & navigable).any(1).all():
                raise ValueError('infeasible scene; tasks retained, not resampled')
        return item
    start = time.perf_counter()
    source = baseline.solve(scene, matrix, arm_distance, nav, 'coverage_cost_2opt', penalty)
    item.teacher_original_stops = len(source['visit_order_candidate_ids'])
    item.teacher = plan_from_actions(item, source['visit_order_candidate_ids'], 'teacher_replayed', prune=True)
    item.teacher_s = time.perf_counter()-start
    if require_feasible and item.teacher['unserved_task_ids']:
        raise ValueError(f'infeasible training scene {nx}x{ny} seed={seed}; tasks retained, not resampled')
    return item


def collate(instances, device='cpu'):
    """Pad tasks/candidates separately; all masks are explicit and size independent."""
    if not instances:
        raise ValueError('cannot collate an empty batch')
    b = len(instances)
    n = max(1, max(len(i.scene['tasks']) for i in instances))
    m = max(len(i.scene['candidates']) for i in instances)
    tasks = np.zeros((b, n, 6), dtype=np.float32)
    nodes = np.zeros((b, m, 9), dtype=np.float32)
    task_mask = np.zeros((b, n), dtype=bool)
    candidate_mask = np.zeros((b, m), dtype=bool)
    feasibility = np.zeros((b, n, m, 2), dtype=bool)
    costs = np.full((b, m+1, m+1), np.inf, dtype=np.float32)
    scales, reference, penalties = [], [], []
    max_steps = max(1, max(len(i.teacher['visit_order_candidate_ids']) if i.teacher else 0 for i in instances))
    teachers = np.full((b, max_steps), -1, dtype=np.int64)
    for index, item in enumerate(instances):
        scene = item.scene
        nn, mm = len(scene['tasks']), len(scene['candidates'])
        length_scale = max(scene['x_lines'][-1], scene['y_lines'][-1], 1.)
        scales.append(length_scale)
        reference.append(max(item.teacher['objective_length_plus_stops_m'], 1.) if item.teacher else 1.)
        penalties.append(item.penalty)
        task_mask[index, :nn] = True
        candidate_mask[index, :mm] = True
        feasibility[index, :nn, :mm] = item.feasibility
        costs[index, :mm+1, :mm+1] = item.navigation['costs']
        for t in scene['tasks']:
            table = scene['tables'][t['table_id']]
            tasks[index, t['id']] = [t['x']/length_scale, t['y']/length_scale, t['z'],
                                      table['cx']/length_scale, table['cy']/length_scale, 1.]
        for j, c in enumerate(scene['candidates']):
            feasible = item.feasibility[:, j]
            distances = item.arm_distance[:, j][feasible]
            quality = float(distances.mean()/scene['robot']['arm_max_reach']) if len(distances) else 0.
            nodes[index, j] = [c['x']/length_scale, c['y']/length_scale, math.sin(c['theta']),
                               math.cos(c['theta']), c['h'], feasible[:, 0].sum()/max(nn, 1),
                               feasible[:, 1].sum()/max(nn, 1), feasible.any(axis=1).sum()/max(nn, 1), quality]
        teacher = item.teacher['visit_order_candidate_ids'] if item.teacher else []
        teachers[index, :len(teacher)] = teacher
    arrays = dict(task_features=tasks, node_features=nodes, task_mask=task_mask,
                  candidate_mask=candidate_mask, feasibility=feasibility,
                  cover=feasibility.any(axis=-1), costs=costs, teacher_actions=teachers,
                  length_scale=np.asarray(scales, dtype=np.float32),
                  reference_cost=np.asarray(reference, dtype=np.float32),
                  stop_penalty=np.asarray(penalties, dtype=np.float32))
    return TensorDict({key: torch.as_tensor(value, device=device) for key, value in arrays.items()},
                      batch_size=[b], device=device)


def export_instance(item, plans):
    return dict(scene=item.scene, feasibility=item.sparse, plans=plans,
                navigation=dict(model='same collision-filtered undirected corridor roadmap as rc3_baseline',
                                nodes=item.navigation['node_count'], edges=item.navigation['edge_count']),
                teacher_original_stops=item.teacher_original_stops,
                teacher_replayed_stops=len(item.teacher['visit_order_candidate_ids']) if item.teacher else None)
