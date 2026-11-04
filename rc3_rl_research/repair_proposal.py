"""Coverage-preserving local proposals with explicit, deterministic work.

The forced replacement operator has no incumbent-cost upper bound.  Acceptance
of a feasible proposal, including the allowed deterioration from best, belongs
to the environment and is deliberately absent from this module.
"""
from __future__ import annotations

import math

import numpy as np

from rc3_fast.solver import route_objective
from rc3_improve.environment import canonical_route
from rc3_improve.repair import route_is_valid

from .action_builder import local_context, validate_caps
from .types import ProposalResult, SearchConfig


def _exact_replacement(problem, route, context):
    """Solve a fixed-boundary set-cover path; return nodes, reason, counters.

    Ties follow the frozen solver's ascending pool, mask, and predecessor order.
    We only materialize finite-predecessor / coverage-adding transitions; the
    optimum is the same as its dense ``inf``-filled transition calculation.
    Counts describe evaluated entries, not a wall-time estimate or FLOPs.
    """
    required, pool, cover = (context[k] for k in ("required", "pool", "cover"))
    start, end = context["start"], context["end"]
    counters = dict(dp_states=0, dp_transitions=0)
    if not len(required):
        return [], None, counters
    if not len(pool) or not cover[np.ix_(required, pool-1)].any(1).all():
        return None, "required_tasks_uncovered", counters
    costs = problem.nav["costs"]
    n, m = len(required), len(pool)
    bits = (cover[np.ix_(required, pool-1)].astype(np.int64)
            * (1 << np.arange(n))[:, None]).sum(0)
    full = (1 << n)-1
    dp = np.full((full+1, m), np.inf)
    predecessor = np.full((full+1, m), -1, dtype=np.int16)
    previous_mask = np.full((full+1, m), -1, dtype=np.int32)
    ids = np.flatnonzero(bits > 0)
    counters["dp_transitions"] += len(ids)
    dp[bits[ids], ids] = costs[route[start-1], pool[ids]]+problem.penalty
    previous_mask[bits[ids], ids] = 0
    inside = costs[np.ix_(pool, pool)]
    for mask in range(1, full):
        sources = np.flatnonzero(np.isfinite(dp[mask]))
        if not len(sources):
            continue
        counters["dp_states"] += len(sources)
        merged = mask | bits
        targets = np.flatnonzero(merged != mask)
        if not len(targets):
            continue
        counters["dp_transitions"] += len(sources)*len(targets)
        transitions = dp[mask, sources, None]+inside[np.ix_(sources, targets)]
        best_local = transitions.argmin(0)
        best_sources = sources[best_local]
        best_cost = transitions[best_local, np.arange(len(targets))]+problem.penalty
        improve = best_cost < dp[merged[targets], targets]-1e-10
        kept = targets[improve]
        dp[merged[kept], kept] = best_cost[improve]
        predecessor[merged[kept], kept] = best_sources[improve]
        previous_mask[merged[kept], kept] = mask
    finals = np.flatnonzero(np.isfinite(dp[full]))
    counters["dp_states"] += len(finals)
    counters["dp_transitions"] += len(finals)
    if not len(finals):
        return None, "no_finite_local_path", counters
    final_costs = dp[full, finals]+costs[pool[finals], route[end]]
    best_local = int(np.argmin(final_costs))
    if not np.isfinite(final_costs[best_local]):
        return None, "no_finite_local_path", counters
    best, mask, replacement = int(finals[best_local]), full, []
    while mask:
        replacement.append(int(pool[best]))
        old_mask = int(previous_mask[mask, best])
        old_node = int(predecessor[mask, best])
        if old_mask < 0 or old_mask >= mask:
            return None, "invalid_dp_predecessor", counters
        mask, best = old_mask, old_node
    return list(reversed(replacement)), None, counters


def propose(problem, route, action, *, config=SearchConfig()):
    """Query precisely one action; feasibility here is not physical certification.

    ``improve`` retains the frozen primitive's strict raw-route improvement
    condition. ``replace`` must ban its one removed old configuration and may
    return a worse route. Both then use the frozen full-route visit semantics.
    Failed queries always charge their actual work, at least the attempt unit.
    """
    validate_caps(config)
    counters = dict(attempts=1, candidate_checks=0, dp_states=0, dp_transitions=0)
    diagnostics = dict(operator=action.kind, forbidden_node=action.forbidden_node,
                       max_tasks=config.max_tasks, max_candidates=config.max_candidates,
                       independent_certification=False)

    def result(candidate=None, reason=None, raw=None):
        integer_counters = {name: int(value) for name, value in counters.items()}
        return ProposalResult(route=candidate, valid=candidate is not None,
                              changed=candidate is not None and candidate != list(route),
                              work_units=sum(integer_counters.values()), failure_reason=reason,
                              raw_route=raw, counters=integer_counters,
                              diagnostics=dict(diagnostics))

    if not route_is_valid(problem, route):
        return result(reason="invalid_incumbent")
    if not math.isfinite(problem.penalty) or problem.penalty < 0:
        return result(reason="invalid_service_penalty")
    incumbent = [int(node) for node in route]
    if action.kind not in ("improve", "replace"):
        return result(reason="unsupported_proposal_operator")
    if action.kind == "improve" and action.forbidden_node is not None:
        return result(reason="improve_must_not_ban_candidate")
    if (type(action.start) is int and type(action.end) is int and
            action.end-action.start > config.max_width):
        return result(reason="window_width_cap_exceeded")
    if action.kind == "replace":
        if (type(action.start) is not int or type(action.end) is not int or
                action.end != action.start+1 or not 1 <= action.start < len(route)-1 or
                type(action.forbidden_node) is not int or
                action.forbidden_node != incumbent[action.start]):
            return result(reason="replace_requires_one_removed_forbidden_node")
    try:
        context = local_context(problem, incumbent, action.start, action.end,
                                action.forbidden_node)
    except ValueError as error:
        diagnostics["error"] = str(error)
        return result(reason="invalid_window")
    counters["candidate_checks"] = context["candidate_checks"]
    diagnostics.update(required_tasks=context["required"].tolist(),
                       pool_nodes=context["pool"].tolist(),
                       retained_nodes=context["retained"].tolist(),
                       retained_coverage_counts=context["retained_counts"].tolist())
    if len(context["required"]) > config.max_tasks:
        return result(reason="task_cap_exceeded")
    if len(context["pool"]) > config.max_candidates:
        return result(reason="candidate_cap_exceeded")
    replacement, failure, dp_counters = _exact_replacement(problem, incumbent, context)
    counters.update(dp_counters)
    if failure is not None:
        return result(reason=failure)
    raw = incumbent[:action.start]+replacement+incumbent[action.end:]
    if action.forbidden_node is not None and action.forbidden_node in raw:
        return result(reason="forbidden_node_survived", raw=raw)
    if not route_is_valid(problem, raw):
        return result(reason="invalid_complete_repair", raw=raw)
    before, raw_cost = route_objective(problem, incumbent), route_objective(problem, raw)
    diagnostics.update(before_cost=float(before), raw_cost=float(raw_cost))
    if not math.isfinite(raw_cost):
        return result(reason="nonfinite_repair_cost", raw=raw)
    if action.kind == "improve" and not raw_cost < before-1e-8:
        diagnostics.update(outcome="no_immediate_improvement", outside_removed_nodes=[])
        return result(incumbent, raw=raw)
    try:
        candidate = canonical_route(problem, raw)
    except ValueError as error:
        diagnostics["error"] = str(error)
        return result(reason="global_canonicalization_failed", raw=raw)
    if not route_is_valid(problem, candidate):
        return result(reason="invalid_canonical_repair", raw=raw)
    final_cost = float(route_objective(problem, candidate))
    if not math.isfinite(final_cost):
        return result(reason="nonfinite_canonical_cost", raw=raw)
    diagnostics.update(after_cost=final_cost,
                       outcome="feasible_proposal",
                       outside_removed_nodes=[int(node) for node in context["retained"] if node not in candidate],
                       current_cost_change=final_cost-before)
    return result(candidate, raw=raw)
