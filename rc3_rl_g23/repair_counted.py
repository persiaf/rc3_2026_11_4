"""Original dense repair with statement-local work counters; no semantic changes."""
import math
import numpy as np
from rc3_fast.solver import route_objective
from rc3_improve.repair import MAX_TASKS,MAX_CANDIDATES,route_is_valid,_local

def _repair(problem, route, window, *, max_tasks=MAX_TASKS, max_candidates=MAX_CANDIDATES, counters=None):
    """Return an exact improving local replacement, or an unchanged incumbent.

    Required tasks are precisely those losing coverage after removing the slice.
    DP state is (covered task mask, last service configuration). A transition must
    add coverage; because navigation is a shortest-path metric and stop penalties
    are nonnegative, omitting zero-gain intermediate service stops is lossless.
    """
    counters.update(attempts=1,candidate_checks=0,dp_states=0,dp_transitions=0)
    incumbent = list(route)
    if not route_is_valid(problem, incumbent):
        raise ValueError('local repair requires a complete valid incumbent route')
    if max_tasks < 0 or max_candidates < 0 or not math.isfinite(problem.penalty) or problem.penalty < 0:
        raise ValueError('caps and service penalty must be nonnegative')
    start, end, required, pool, cover = _local(problem, incumbent, window)
    counters['candidate_checks']=int(problem.F.shape[1]) if len(required) else 0
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
        counters['dp_transitions']+=len(ids)
        for mask in range(1, full):
            counters['dp_states']+=m
            if not np.isfinite(dp[mask]).any():
                continue
            merged = mask | bits
            adds = merged != mask
            if not adds.any():
                continue
            counters['dp_transitions']+=m*m
            transitions = dp[mask, :, None]+inside
            best_previous = transitions.argmin(0)
            best_cost = transitions[best_previous, columns]+problem.penalty
            improve = adds & (best_cost < dp[merged, columns]-1e-10)
            targets = columns[improve]
            dp[merged[improve], targets] = best_cost[improve]
            predecessor[merged[improve], targets] = best_previous[improve]
            previous_mask[merged[improve], targets] = mask
        counters['dp_transitions']+=m
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
