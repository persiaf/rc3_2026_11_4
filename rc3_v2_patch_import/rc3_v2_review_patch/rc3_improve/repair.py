"""Exact fixed-endpoint repair of short coverage-route segments.

The incumbent is always recoverable. Exactness is conditional on the declared
task/candidate caps and on nonnegative shortest-path navigation distances.
No learned scores or teacher improvements enter the observable features.
"""
import math
import numpy as np

from rc3_fast.solver import route_objective


MAX_TASKS = 8
MAX_CANDIDATES = 80
FEATURE_NAMES = [
    'route_stops_log', 'relative_position', 'window_stops', 'required_tasks',
    'candidate_count', 'tasks_per_stop', 'local_travel', 'direct_distance',
    'detour_excess', 'detour_fraction', 'edge_mean', 'edge_std', 'edge_max',
    'coverage_mean', 'coverage_max', 'options_mean', 'options_min',
    'retained_coverage_fraction', 'x_span', 'y_span', 'centroid_x_fraction',
    'centroid_y_fraction', 'both_arms_fraction', 'exact_supported',
]


def route_is_valid(problem, route):
    """Cheap combinatorial guard; final exported polylines need independent validation."""
    try:
        raw = np.asarray(route)
        if raw.ndim != 1 or len(raw) < 2 or raw.dtype.kind not in 'iu':
            return False
        nodes = raw.astype(np.int64)
        m = problem.F.shape[1]
        if nodes[0] != 0 or nodes[-1] != 0:
            return False
        middle = nodes[1:-1]
        if np.any((middle < 1) | (middle > m)) or len(np.unique(middle)) != len(middle):
            return False
        if not np.isfinite(problem.nav['costs'][nodes[:-1], nodes[1:]]).all():
            return False
        cover = problem.F.any(2)
        return bool(cover[:, middle-1].any(1).all())
    except (IndexError, TypeError, ValueError):
        return False


def windows(problem, route, width=3, stride=2):
    """Contiguous candidate intervals, including the final possible interval."""
    if not isinstance(width, int) or not isinstance(stride, int) or width < 1 or stride < 1:
        raise ValueError('width and stride must be positive integers')
    if not route_is_valid(problem, route):
        raise ValueError('local repair requires a complete valid incumbent route')
    count = len(route)-2
    if not count:
        return []
    actual = min(width, count)
    last = count-actual+1
    starts = sorted(set([*range(1, last+1, stride), last]))
    return [dict(start=start, end=start+actual) for start in starts]


def _interval(route, window):
    start, end = window['start'], window['end']
    if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
        raise ValueError('window indices must be integers')
    if not 1 <= start < end <= len(route)-1:
        raise ValueError('window must exclude both depot endpoints')
    return int(start), int(end)


def _local(problem, route, window, cover=None, counts=None):
    start, end = _interval(route, window)
    cover = problem.F.any(2) if cover is None else cover
    nodes = np.asarray(route, dtype=np.int64)
    removed = nodes[start:end]
    if counts is None:
        counts = cover[:, nodes[1:-1]-1].sum(1)
    retained_counts = counts-cover[:, removed-1].sum(1)
    required = np.flatnonzero(retained_counts == 0)
    if not len(required):
        return start, end, required, np.empty(0, dtype=np.int64), cover
    eligible = cover[required].any(0)
    eligible[removed-1] = True
    retained = np.r_[nodes[1:start], nodes[end:-1]]
    eligible[retained-1] = False
    costs = problem.nav['costs']
    eligible &= np.isfinite(costs[0, 1:]) & np.isfinite(costs[1:, 0])
    pool = np.flatnonzero(eligible)+1
    return start, end, required, pool, cover


def window_features(problem, route, candidates):
    """Return fixed-width observable features for variable-size action sets."""
    if not route_is_valid(problem, route):
        raise ValueError('features require a complete valid incumbent route')
    cover = problem.F.any(2)
    counts = cover[:, np.asarray(route[1:-1], dtype=np.int64)-1].sum(1)
    costs = problem.nav['costs']
    scene = problem.scene
    x_extent = max(float(scene['x_lines'][-1]), 1.)
    y_extent = max(float(scene['y_lines'][-1]), 1.)
    values = []
    for window in candidates:
        start, end, required, pool, _ = _local(problem, route, window, cover, counts)
        removed = np.asarray(route[start:end])-1
        local = np.asarray(route[start-1:end+1])
        edges = costs[local[:-1], local[1:]]
        travel = float(edges.sum())
        direct = float(costs[local[0], local[-1]])
        xyz = np.asarray([[scene['candidates'][i]['x'], scene['candidates'][i]['y']] for i in removed])
        gains = cover[np.ix_(required, pool-1)].sum(0)
        options = cover[np.ix_(required, pool-1)].sum(1)
        removed_coverage = cover[:, removed].any(1)
        retained_covered = removed_coverage & (counts-cover[:, removed].sum(1) > 0)
        both = problem.F[required][:, pool-1].all(2) if len(required) and len(pool) else np.empty(0)
        values.append([
            math.log1p(len(route)-2), (start-1)/max(len(route)-2, 1), end-start,
            len(required), len(pool), len(required)/(end-start), travel, direct,
            travel-direct, (travel-direct)/max(travel, 1e-8),
            float(edges.mean()), float(edges.std()), float(edges.max()),
            float(gains.mean()) if len(gains) else 0., float(gains.max()) if len(gains) else 0.,
            float(options.mean()) if len(options) else 0., float(options.min()) if len(options) else 0.,
            float(retained_covered.sum()/max(removed_coverage.sum(), 1)),
            float(np.ptp(xyz[:, 0])), float(np.ptp(xyz[:, 1])),
            float(xyz[:, 0].mean()/x_extent), float(xyz[:, 1].mean()/y_extent),
            float(both.mean()) if both.size else 0.,
            float(len(required) <= MAX_TASKS and len(pool) <= MAX_CANDIDATES),
        ])
    result = np.asarray(values, dtype=np.float32).reshape(-1, len(FEATURE_NAMES))
    if not np.isfinite(result).all():
        raise ValueError('nonfinite window features')
    return result


def repair_window(problem, route, window, *, max_tasks=MAX_TASKS, max_candidates=MAX_CANDIDATES):
    """Return an exact improving local replacement, or an unchanged incumbent.

    Required tasks are precisely those losing coverage after removing the slice.
    DP state is (covered task mask, last service configuration). A transition must
    add coverage; because navigation is a shortest-path metric and stop penalties
    are nonnegative, omitting zero-gain intermediate service stops is lossless.
    """
    incumbent = list(route)
    if not route_is_valid(problem, incumbent):
        raise ValueError('local repair requires a complete valid incumbent route')
    if max_tasks < 0 or max_candidates < 0 or not math.isfinite(problem.penalty) or problem.penalty < 0:
        raise ValueError('caps and service penalty must be nonnegative')
    start, end, required, pool, cover = _local(problem, incumbent, window)
    if len(required) > max_tasks or len(pool) > max_candidates:
        return incumbent
    costs = problem.nav['costs']
    entrance, exit_node = incumbent[start-1], incumbent[end]
    if not len(required):
        candidate = incumbent[:start]+incumbent[end:]
    else:
        n, m = len(required), len(pool)
        bits = (cover[np.ix_(required, pool-1)].astype(np.int64) * (1 << np.arange(n))[:, None]).sum(0)
        full = (1 << n)-1
        dp = np.full((full+1, m), np.inf)
        predecessor = np.full((full+1, m), -1, dtype=np.int16)
        previous_mask = np.full((full+1, m), -1, dtype=np.int32)
        valid = bits > 0
        ids = np.flatnonzero(valid)
        dp[bits[ids], ids] = costs[entrance, pool[ids]]+problem.penalty
        previous_mask[bits[ids], ids] = 0
        inside = costs[np.ix_(pool, pool)]
        columns = np.arange(m)
        for mask in range(1, full):
            if not np.isfinite(dp[mask]).any():
                continue
            merged = mask | bits
            adds = merged != mask
            if not adds.any():
                continue
            transitions = dp[mask, :, None]+inside
            best_previous = transitions.argmin(0)
            best_cost = transitions[best_previous, columns]+problem.penalty
            improve = adds & (best_cost < dp[merged, columns]-1e-10)
            targets = columns[improve]
            dp[merged[improve], targets] = best_cost[improve]
            predecessor[merged[improve], targets] = best_previous[improve]
            previous_mask[merged[improve], targets] = mask
        final = dp[full]+costs[pool, exit_node]
        best = int(np.argmin(final))
        if not np.isfinite(final[best]):
            return incumbent
        replacement, mask = [], full
        while mask:
            replacement.append(int(pool[best]))
            old_mask = int(previous_mask[mask, best])
            old_node = int(predecessor[mask, best])
            if old_mask < 0:
                return incumbent
            mask, best = old_mask, old_node
        candidate = incumbent[:start]+list(reversed(replacement))+incumbent[end:]
    if not route_is_valid(problem, candidate):
        return incumbent
    if route_objective(problem, candidate) < route_objective(problem, incumbent)-1e-8:
        return candidate
    return incumbent
