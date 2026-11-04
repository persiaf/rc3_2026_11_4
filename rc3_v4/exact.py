"""Certified small-instance coverage-tour reference for a discrete metric model.

The DP state is (covered task mask, last candidate), and every transition adds
at least one task. Any closed feasible tour can drop visits with no new coverage:
the triangle inequality cannot increase travel, and the nonnegative stop penalty
cannot increase service cost. Thus an optimal tour has at most N service visits
and is represented by this DP. This is a *global* statement for the supplied
finite candidate set and distance matrix, not for a continuous robot workspace.

Float64 comparisons use the supplied matrix without epsilon pruning. Metric
validation tolerates floating point roundoff (1e-8 absolute, 1e-10 relative).
"""
from __future__ import annotations

import math
import time
import numpy as np


def solve_metric(coverage, distances, penalty=.75, *, max_tasks=14,
                 max_candidates=128, time_limit_s=30., incumbent=None):
    """Return status, primal tour, lower bound, and explicit proof scope.

    Candidate column j has navigation index j+1; navigation index 0 is depot.
    Time limits include validation and DP, checked between atomic NumPy calls.
    On timeout, the bound is a conservative geometric/count bound, *not* a
    partially searched DP value. Unsupported inputs never carry an optimal flag.
    """
    started = time.perf_counter()
    best_route, upper, lower, states = None, math.inf, 0., 0
    metric_verified = False
    def result(status, reason=None):
        gap = ((upper-lower)/max(abs(upper), 1e-12)
               if math.isfinite(upper) else None)
        return dict(status=status, optimal=status == 'optimal', route=best_route,
                    objective=upper if math.isfinite(upper) else None,
                    lower_bound=lower if math.isfinite(lower) else None,
                    relative_gap=max(0., gap) if gap is not None else None,
                    elapsed_s=time.perf_counter()-started, expanded_masks=states,
                    metric_verified=metric_verified, reason=reason,
                    scope='Global for the supplied discrete coverage and nonnegative metric distances only; float64 numerical tolerance, no continuous/IK claim.')
    try:
        raw = np.asarray(coverage)
        if raw.ndim != 2 or not np.isin(raw, [0, 1]).all():
            raise ValueError('coverage must be a binary task-by-candidate matrix')
        cover = raw.astype(bool)
        n, m = cover.shape
        costs = np.asarray(distances, dtype=np.float64)
        if costs.shape != (m+1, m+1) or np.isnan(costs).any() or (costs < 0).any():
            raise ValueError('distance shape, NaN, or negative navigation cost')
        if not np.allclose(np.diag(costs), 0., rtol=0., atol=1e-10):
            raise ValueError('distance diagonal must be zero')
        if not math.isfinite(penalty) or penalty < 0:
            raise ValueError('penalty must be finite and nonnegative')
        if time_limit_s is not None and (not math.isfinite(time_limit_s) or time_limit_s < 0):
            raise ValueError('time limit must be finite and nonnegative')
        if type(max_tasks) is not int or type(max_candidates) is not int or max_tasks < 0 or max_candidates < 1:
            raise ValueError('caps must be nonnegative task / positive candidate integers')
        deadline = math.inf if time_limit_s is None else started+time_limit_s
        if incumbent is not None:
            route = list(incumbent)
            if (len(route) < 2 or route[0] != 0 or route[-1] != 0
                    or any(type(v) is not int for v in route)
                    or any(v < 1 or v > m for v in route[1:-1])
                    or len(set(route[1:-1])) != len(route)-2
                    or not cover[:, np.asarray(route[1:-1], dtype=int)-1].any(1).all()):
                raise ValueError('incumbent is not a complete simple depot tour')
            value = sum(costs[a, b] for a, b in zip(route, route[1:]))+penalty*(len(route)-2)
            if not math.isfinite(value):
                raise ValueError('incumbent uses disconnected navigation edges')
            best_route, upper = route, float(value)
        if n > max_tasks or m > max_candidates:
            return result('unsupported', f'{n} tasks / {m} candidates exceed caps {max_tasks} / {max_candidates}')
        if time.perf_counter() >= deadline:
            return result('time_limit', 'deadline reached before metric verification')
        # Verify the full supplied matrix, including disconnected components.
        for k in range(m+1):
            through = costs[:, k, None]+costs[None, k, :]
            tolerance = 1e-8+1e-10*np.where(np.isfinite(through), through, 0.)
            if np.any(costs > through+tolerance):
                return result('invalid_input', 'distances violate the triangle inequality')
            if time.perf_counter() >= deadline:
                return result('time_limit', 'deadline reached during metric verification')
        metric_verified = True
        if n == 0:
            best_route, upper, lower = [0, 0], 0., 0.
            return result('optimal')
        connected = np.isfinite(costs[0, 1:]) & np.isfinite(costs[1:, 0])
        ids = np.flatnonzero(connected & cover.any(0))
        if not cover[:, ids].any(1).all():
            best_route, upper, lower = None, math.inf, math.inf
            return result('infeasible', 'a task has no candidate in the depot round-trip component')
        # Each task requires a tour at least as long as its cheapest depot round
        # trip; every tour needs ceil(N / max_tasks_per_stop) service stops.
        rt = costs[0, ids+1]+costs[ids+1, 0]
        travel_lb = float(np.max(np.min(np.where(cover[:, ids], rt[None, :], np.inf), axis=1)))
        stops_lb = math.ceil(n/int(cover[:, ids].sum(0).max()))
        lower = travel_lb+penalty*stops_lb
        if math.isfinite(upper):
            lower = min(lower, upper)  # roundoff only
        pool = ids+1
        p = len(pool)
        bits = (cover[:, ids].astype(np.int64)*(1 << np.arange(n))[:, None]).sum(0)
        full = (1 << n)-1
        dp = np.full((full+1, p), np.inf)
        predecessor = np.full((full+1, p), -1, np.int16)
        previous_mask = np.full((full+1, p), -1, np.int32)
        columns = np.arange(p)
        dp[bits, columns] = costs[0, pool]+penalty
        previous_mask[bits, columns] = 0
        inside = costs[np.ix_(pool, pool)]
        finished = True
        for mask in range(1, full):
            if time.perf_counter() >= deadline:
                finished = False
                break
            if not np.isfinite(dp[mask]).any():
                continue
            states += 1
            merged = mask | bits
            adds = merged != mask
            if not adds.any():
                continue
            transitions = dp[mask, :, None]+inside
            prev = transitions.argmin(0)
            values = transitions[prev, columns]+penalty
            improve = adds & (values < dp[merged, columns])
            targets = columns[improve]
            dp[merged[improve], targets] = values[improve]
            predecessor[merged[improve], targets] = prev[improve]
            previous_mask[merged[improve], targets] = mask
        totals = dp[full]+costs[pool, 0]
        last = int(totals.argmin())
        if math.isfinite(totals[last]) and totals[last] <= upper+1e-10:
            rev, mask = [], full
            while mask:
                rev.append(int(pool[last]))
                mask, last = int(previous_mask[mask, last]), int(predecessor[mask, last])
            best_route = [0]+list(reversed(rev))+[0]
            upper = float(sum(costs[a, b] for a, b in zip(best_route, best_route[1:]))+penalty*(len(best_route)-2))
        if finished:
            if best_route is None:
                return result('infeasible', 'exhaustive DP found no complete tour')
            lower = upper
            return result('optimal')
        return result('time_limit', 'DP interrupted; conservative bound reported')
    except (ValueError, TypeError, IndexError, OverflowError) as error:
        return result('invalid_input', str(error))


def solve(problem, **kwargs):
    """Convenience adapter for rc3_fast.solver.Problem (no changes to v3)."""
    return solve_metric(problem.F.any(2), problem.nav['costs'], problem.penalty, **kwargs)
