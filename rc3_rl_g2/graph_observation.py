"""Raw, reconstructible local coverage graphs for G2 policy inputs.

No learned embedding is cached. Immutable scene coordinates and C=max_arm(F)
may be reused for one problem identity; every dynamic route/menu is rebuilt.
"""
from __future__ import annotations

import copy
import hashlib
import math
import weakref
import numpy as np

from rc3_rl_research.action_builder import FEATURE_NAMES as BASE_FEATURE_NAMES

GRAPH_SCHEMA = 'rc3-g2-local-coverage-v1'
ACTION_FEATURE_NAMES = list(BASE_FEATURE_NAMES) + [
    'current_best_gap_percent', 'best_gain_percent', 'current_anchor_ratio',
    'work_after_menu_fraction', 'pending_menu_fraction', 'attempts_log',
    'unchanged_attempts_log', 'failed_attempts_log',
]
TASK_FEATURE_NAMES = [
    'entry_relative_x', 'entry_relative_y', 'height', 'required',
    'retained_coverage_log', 'route_coverage_log', 'pool_options_fraction',
    'mean_route_service_rank', 'covered_by_removed',
]
CANDIDATE_FEATURE_NAMES = [
    'entry_relative_x', 'entry_relative_y', 'cos_theta', 'sin_theta', 'lift',
    'entry_graph_distance', 'exit_graph_distance', 'depot_graph_distance',
    'current_route_rank', 'is_retained', 'required_coverage_fraction',
    'outside_coverage_fraction', 'is_removed', 'retained_cover_mean_log',
]
ACTION_DIM = len(ACTION_FEATURE_NAMES)
TASK_DIM = len(TASK_FEATURE_NAMES)
CANDIDATE_DIM = len(CANDIDATE_FEATURE_NAMES)
_STATIC = weakref.WeakKeyDictionary()


def _frozen(array, dtype=None):
    array = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def static_graph(adapter):
    """Cache raw immutable data, tied to the exact versioned problem identity."""
    cached = _STATIC.get(adapter)
    identity = hashlib.sha256((GRAPH_SCHEMA + ':' + adapter.problem_id).encode()).hexdigest()
    if cached is not None:
        if cached['identity'] != identity:
            raise ValueError('static graph identity changed on a live adapter')
        return cached
    p = adapter.problem
    tasks = np.asarray([[t['x'], t['y'], t.get('z', 0.)] for t in p.scene['tasks']], dtype=np.float64).reshape(-1, 3)
    candidates = np.asarray([[c['x'], c['y'], c.get('theta', 0.), c.get('h', 0.)]
                             for c in p.scene['candidates']], dtype=np.float64).reshape(-1, 4)
    positions = np.concatenate((np.asarray(p.scene['start'], dtype=np.float64)[None, :2], candidates[:, :2]))
    result = dict(identity=identity, problem_id=adapter.problem_id,
                  tasks=_frozen(tasks), candidates=_frozen(candidates),
                  positions=_frozen(positions), cover=_frozen(p.F.any(2), bool))
    _STATIC[adapter] = result
    return result


def _distance(value, scale):
    # A finite indicator is provided by legal navigation/pool construction; a
    # nonfinite value is never silently imputed into a policy feature.
    if not math.isfinite(float(value)):
        raise ValueError('nonfinite navigation distance in a legal action graph')
    return float(value) / scale


def build_policy_observation(env):
    """Snapshot exact actions plus explicit local task/candidate/nav relations.

    Node arrays from all local action graphs are concatenated; *_action maps
    each node to its menu entry. Coverage and sparse within-pool navigation
    relations carry real structure rather than only aggregate summaries.
    STOP has no artificial graph nodes and uses the dynamic context features.
    """
    obs = env.observe()
    p = env.adapter.problem
    raw = static_graph(env.adapter)
    cover = raw['cover']
    scale = float(env.config.physical_scale)
    route = np.asarray(obs['current'], dtype=np.int64)
    n_tasks, n_candidates = cover.shape
    count = len(obs['actions'])
    details = [] if env._menu is None else env._menu.diagnostics.get('action_details', [])
    if count and len(details) != count:
        raise ValueError('action graph details do not align with the sampled menu')
    attempts = obs.get('action_attempts', [{} for _ in range(count)])
    if len(attempts) != count:
        raise ValueError('attempt features do not align with sampled actions')
    anchor = max(float(obs['anchor_cost']), 1e-8)
    budget = max(int(obs['work_budget']), 1)
    common = [100. * (obs['current_cost'] - obs['best_cost']) / anchor,
              100. * (obs['anchor_cost'] - obs['best_cost']) / anchor,
              obs['current_cost'] / anchor,
              obs['work_after_menu'] / budget, obs['pending_menu_work'] / budget]
    features = []
    task_rows, candidate_rows = [], []
    task_action, candidate_action = [], []
    task_ids, candidate_nodes = [], []
    edge_task, edge_candidate = [], []
    nav_source, nav_target, nav_distance = [], [], []
    route_counts = np.asarray(obs['task_coverage_counts'], dtype=np.int64)
    service_rank = np.full(n_candidates + 1, -1., dtype=float)
    if len(route) > 2:
        service_rank[route[1:-1]] = np.arange(1, len(route)-1) / max(len(route)-2, 1)
    task_rank = np.zeros(n_tasks, dtype=float)
    if len(route) > 2 and n_tasks:
        visited_cover = cover[:, route[1:-1]-1]
        task_rank = (visited_cover * service_rank[route[1:-1]][None, :]).sum(1) / np.maximum(route_counts, 1)

    for action_index, action in enumerate(obs['actions']):
        attempt = attempts[action_index]
        if not isinstance(attempt, dict):
            raise ValueError('attempt features must be dictionaries')
        attempt_values = [attempt.get('attempts', 0), attempt.get('no_change', 0), attempt.get('failed', 0)]
        if (isinstance(attempt_values[0], (bool,np.bool_)) or
                any(not math.isfinite(float(v)) or float(v) < 0 for v in attempt_values)):
            raise ValueError('invalid attempt counters')
        features.append(list(obs['features'][action_index]) + common + [math.log1p(float(v)) for v in attempt_values])
        if action['kind'] == 'stop':
            continue
        detail = details[action_index]
        required = np.asarray(detail['required_tasks'], dtype=np.int64)
        pool = np.asarray(detail['pool_nodes'], dtype=np.int64)
        retained_counts = np.asarray(detail['retained_coverage_counts'], dtype=np.int64)
        start, end = int(action['start']), int(action['end'])
        entrance, exit_node = int(route[start-1]), int(route[end])
        removed = route[start:end]
        retained = np.asarray(detail['retained_nodes'], dtype=np.int64)
        touched = np.flatnonzero(cover[:, pool-1].any(1)) if len(pool) else np.empty(0, dtype=np.int64)
        selected_tasks = np.union1d(required, touched)
        local_cover = cover[np.ix_(selected_tasks, pool-1)]
        removed_cover = cover[:, removed-1].any(1)
        origin = raw['positions'][entrance]
        task_offset, candidate_offset = len(task_rows), len(candidate_rows)
        required_flag = np.isin(selected_tasks, required)
        for local_id, task in enumerate(selected_tasks):
            xyz = raw['tasks'][task]
            task_rows.append([
                (xyz[0]-origin[0])/scale, (xyz[1]-origin[1])/scale, xyz[2]/scale,
                float(required_flag[local_id]), math.log1p(int(retained_counts[task])),
                math.log1p(int(route_counts[task])), local_cover[local_id].sum()/max(env.config.max_candidates, 1),
                float(task_rank[task]), float(removed_cover[task]),
            ])
            task_action.append(action_index); task_ids.append(int(task))
        required_global = np.zeros(n_tasks, dtype=bool); required_global[required] = True
        for node in pool:
            c = raw['candidates'][node-1]
            ccover = cover[:, node-1]
            covered_counts = retained_counts[ccover]
            candidate_rows.append([
                (c[0]-origin[0])/scale, (c[1]-origin[1])/scale, math.cos(c[2]), math.sin(c[2]), c[3]/scale,
                _distance(p.nav['costs'][entrance, node], scale),
                _distance(p.nav['costs'][node, exit_node], scale),
                _distance(p.nav['costs'][0, node], scale),
                float(service_rank[node]), float(node in retained),
                float((ccover & required_global).sum())/max(env.config.max_tasks, 1),
                float((ccover & (retained_counts > 0)).sum())/max(n_tasks, 1),
                float(node in removed), math.log1p(float(covered_counts.mean())) if len(covered_counts) else 0.,
            ])
            candidate_action.append(action_index); candidate_nodes.append(int(node))
        ti, ci = np.nonzero(local_cover)
        edge_task.extend((ti+task_offset).tolist()); edge_candidate.extend((ci+candidate_offset).tolist())
        # At most two directed nearest-neighbour edges per candidate, ordered
        # by true navigation distance, never Euclidean distance through tables.
        if len(pool) > 1:
            distances = np.asarray(p.nav['costs'][np.ix_(pool, pool)])
            for target in range(len(pool)):
                sources = [int(i) for i in np.argsort(distances[:, target], kind='stable')
                           if i != target and math.isfinite(float(distances[i, target]))][:2]
                for source in sources:
                    nav_source.append(candidate_offset+source); nav_target.append(candidate_offset+target)
                    nav_distance.append(_distance(distances[source, target], scale))

    result = dict(schema=GRAPH_SCHEMA, static_graph_id=raw['identity'], problem_id=env.adapter.problem_id,
        state_key=obs['menu_context_key'], actions=copy.deepcopy(obs['actions']),
        action_keys=list(obs.get('proposal_keys', obs['action_context_keys'])),
        features=np.asarray(features, dtype=np.float32).reshape(count, ACTION_DIM),
        mask=np.asarray(obs['mask'], dtype=bool),
        task_features=np.asarray(task_rows, dtype=np.float32).reshape(-1, TASK_DIM),
        candidate_features=np.asarray(candidate_rows, dtype=np.float32).reshape(-1, CANDIDATE_DIM),
        task_action=np.asarray(task_action, dtype=np.int64), candidate_action=np.asarray(candidate_action, dtype=np.int64),
        task_ids=np.asarray(task_ids, dtype=np.int64), candidate_nodes=np.asarray(candidate_nodes, dtype=np.int64),
        edge_task=np.asarray(edge_task, dtype=np.int64), edge_candidate=np.asarray(edge_candidate, dtype=np.int64),
        nav_source=np.asarray(nav_source, dtype=np.int64), nav_target=np.asarray(nav_target, dtype=np.int64),
        nav_distance=np.asarray(nav_distance, dtype=np.float32),
        observation_scope='Independent-arm route coverage C=max_arm(F); no simultaneous dual-arm coordination claim')
    for key in ('features', 'task_features', 'candidate_features', 'nav_distance'):
        if not np.isfinite(result[key]).all():
            raise ValueError('nonfinite policy input: '+key)
    return result
