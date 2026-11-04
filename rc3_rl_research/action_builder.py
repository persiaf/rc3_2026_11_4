"""Deterministic, cheap action construction shared by all research policies.

No function in this module calls the exact repair solver.  The complete retained
route, rather than just the local boundary, determines which tasks need repair.
Navigation node 0 is the depot; service node ``q + 1`` indexes candidate ``q``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict

import numpy as np

from rc3_improve.repair import route_is_valid
from .types import Action, ActionMenu, SearchConfig


FEATURE_NAMES = [
    "is_improve", "is_replace", "is_stop", "relative_position", "window_stops",
    "required_tasks", "candidate_count", "local_travel", "direct_distance",
    "detour_excess", "detour_fraction", "route_stops_log", "task_count_log",
    "global_candidate_count_log", "coverage_count_mean", "coverage_count_max",
    "removed_covered_tasks", "retained_overlap_fraction", "pool_extra_covered_tasks",
    "remaining_work_fraction", "steps_remaining_fraction", "attempt_history_count",
]


def validate_caps(config):
    if config.max_tasks > 8 or config.max_candidates > 80:
        raise ValueError("G0/G1 hard caps are 8 tasks and 80 candidates")
    if config.max_width > 4 or config.max_actions > 16:
        raise ValueError("G0/G1 hard caps are width 4 and 16 non-STOP actions")


def stable_digest(value):
    """A process-independent identifier; Python's salted hash is unsuitable."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def local_context(problem, route, start, end, forbidden_node=None, *,
                  cover=None, counts=None):
    """Compute a local candidate pool without changing problem.F or any mask.

    The optional cover/counts inputs only share a read-only calculation within
    one menu construction.  A returned pool is a new ascending integer array.
    Old removed candidates stay eligible for ``improve``.  ``replace`` excludes
    its forbidden old node *only* in the newly allocated local eligibility mask.
    """
    if (type(start) is not int or type(end) is not int or
            not 1 <= start < end <= len(route)-1):
        raise ValueError("window must be an integer slice excluding depots")
    if forbidden_node is not None:
        if type(forbidden_node) is not int or forbidden_node not in route[start:end]:
            raise ValueError("forbidden node must be a removed service node")
    cover = problem.F.any(2) if cover is None else cover
    nodes = np.asarray(route, dtype=np.int64)
    removed = nodes[start:end]
    retained = np.r_[nodes[1:start], nodes[end:-1]]
    counts = cover[:, nodes[1:-1]-1].sum(1) if counts is None else counts
    retained_counts = counts-cover[:, removed-1].sum(1)
    required = np.flatnonzero(retained_counts == 0)
    candidate_checks = 0
    if len(required):
        candidate_checks = int(cover.shape[1])
        eligible = cover[required].any(0)
        eligible[removed-1] = True
        eligible[retained-1] = False
        if forbidden_node is not None:
            eligible[forbidden_node-1] = False
        costs = problem.nav["costs"]
        eligible &= np.isfinite(costs[0, 1:]) & np.isfinite(costs[1:, 0])
        pool = np.flatnonzero(eligible)+1
    else:
        pool = np.empty(0, dtype=np.int64)
    return dict(start=start, end=end, required=required, pool=pool, cover=cover,
                retained=retained, removed=removed, counts=counts,
                retained_counts=retained_counts, candidate_checks=candidate_checks)


def _problem_identity(problem, problem_id):
    if problem_id:
        return str(problem_id)
    digest = hashlib.sha256()
    for array in (problem.F, problem.nav["costs"]):
        value = np.asarray(array)
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
    digest.update(repr(float(problem.penalty)).encode("ascii"))
    return digest.hexdigest()


def build_actions(problem, route, *, remaining_work, steps_remaining, history=(),
                  config=SearchConfig(), problem_id=""):
    """Create a replayable menu without looking up any repair outcome.

    Work is ``contexts_evaluated + candidate_checks``. Each evaluated window,
    including cap-rejected widths, costs one context unit. Its candidate checks
    count M exactly when a nonempty required set causes all M configurations to
    be scanned; an empty required set needs no candidate scan. STOP adds no work.
    A caller must cache this snapshot; repeated observations must not rebuild it.

    For every start, improve keeps the largest supported width. Replace removes
    exactly one stop. Known unsupported pools are omitted before size truncation.
    Each operator is sorted by cheap detour then by slice; their lists interleave.
    No true improvement or DP result participates in ordering or masking.
    """
    validate_caps(config)
    if not route_is_valid(problem, route):
        raise ValueError("action construction requires a complete valid route")
    if type(remaining_work) is not int or type(steps_remaining) is not int:
        raise ValueError("remaining work and steps must be integers")
    route = [int(node) for node in route]
    history = tuple(str(key) for key in history)
    cover = problem.F.any(2)
    nodes = np.asarray(route, dtype=np.int64)
    counts = cover[:, nodes[1:-1]-1].sum(1)
    costs = problem.nav["costs"]
    contexts_evaluated = candidate_checks = unsupported = 0
    groups = {"improve": [], "replace": []}

    def evaluate(kind, start, end, forbidden=None):
        nonlocal contexts_evaluated, candidate_checks, unsupported
        context = local_context(problem, route, start, end, forbidden,
                                cover=cover, counts=counts)
        contexts_evaluated += 1
        candidate_checks += context["candidate_checks"]
        required, pool = context["required"], context["pool"]
        supported = len(required) <= config.max_tasks and len(pool) <= config.max_candidates
        if len(required):
            supported = supported and bool(len(pool)) and bool(cover[np.ix_(required, pool-1)].any(1).all())
        if not supported:
            unsupported += 1
            return None
        local_nodes = nodes[start-1:end+1]
        travel = float(costs[local_nodes[:-1], local_nodes[1:]].sum())
        direct = float(costs[local_nodes[0], local_nodes[-1]])
        # A missing direct connection is a known local boundary obstruction.
        # Keep the candidate: an intermediate covering stop can connect it.
        detour = travel-direct if math.isfinite(direct) else 0.
        action = Action(kind, start, end, forbidden,
                        f"{kind}:{start}:{end}:{forbidden if forbidden is not None else '-'}")
        return dict(action=action, context=context, travel=travel,
                    direct=direct if math.isfinite(direct) else 0., detour=detour)

    if steps_remaining > 0 and remaining_work > 0:
        for start in range(1, len(route)-1):
            for width in range(min(config.max_width, len(route)-1-start), 0, -1):
                row = evaluate("improve", start, start+width)
                if row is not None:
                    groups["improve"].append(row)
                    break
            row = evaluate("replace", start, start+1, route[start])
            if row is not None:
                groups["replace"].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: (-row["detour"], row["action"].start, row["action"].end))
    ordered = []
    for i in range(max(map(len, groups.values()), default=0)):
        for kind in ("improve", "replace"):
            if i < len(groups[kind]):
                ordered.append(groups[kind][i])
    count_before = len(ordered)
    ordered = ordered[:config.max_actions]
    work_units = contexts_evaluated+candidate_checks
    context_key = stable_digest(dict(problem_id=_problem_identity(problem, problem_id),
                                     route=route, remaining_work=remaining_work,
                                     steps_remaining=steps_remaining, history=history,
                                     config=asdict(config)))
    common = [math.log1p(len(route)-2), math.log1p(cover.shape[0]),
              math.log1p(cover.shape[1]), float(counts.mean()) if len(counts) else 0.,
              float(counts.max()) if len(counts) else 0.]
    dynamic = [max(remaining_work, 0)/max(config.work_budget, 1),
               max(steps_remaining, 0)/max(config.horizon, 1), float(len(history))]
    actions, features, action_details = [], [], []
    for row in ordered:
        action, context = row["action"], row["context"]
        removed_covered = cover[:, context["removed"]-1].any(1)
        overlap = removed_covered & (context["retained_counts"] > 0)
        pool_covered = cover[:, context["pool"]-1].any(1)
        extra = pool_covered & (context["retained_counts"] > 0)
        actions.append(action)
        features.append([
            float(action.kind == "improve"), float(action.kind == "replace"), 0.,
            (action.start-1)/max(len(route)-2, 1), float(action.end-action.start),
            float(len(context["required"])), float(len(context["pool"])),
            row["travel"]/config.physical_scale, row["direct"]/config.physical_scale,
            row["detour"]/config.physical_scale, row["detour"]/max(row["travel"], 1e-8),
            *common, float(removed_covered.sum()),
            float(overlap.sum()/max(removed_covered.sum(), 1)), float(extra.sum()), *dynamic])
        action_details.append(dict(required_tasks=context["required"].tolist(),
                                   pool_nodes=context["pool"].tolist(),
                                   retained_coverage_counts=context["retained_counts"].tolist(),
                                   retained_nodes=context["retained"].tolist()))
    actions.append(Action("stop", key="stop"))
    features.append([0., 0., 1., *([0.]*8), *common, 0., 0., 0., *dynamic])
    action_details.append({})
    # At least one unit is required to launch an atomic selected proposal.
    can_search = steps_remaining > 0 and remaining_work > work_units
    mask = [can_search]*(len(actions)-1)+[True]
    if not np.isfinite(np.asarray(features)).all():
        raise ValueError("action features must be finite")
    keys = [stable_digest(dict(context=context_key, action=a.to_dict())) for a in actions]
    return ActionMenu(actions, mask, features, list(FEATURE_NAMES), context_key, keys,
                      int(work_units), diagnostics=dict(
                          contexts_evaluated=contexts_evaluated, candidate_checks=candidate_checks,
                          unsupported_contexts=unsupported, supported_before_cap=count_before,
                          omitted_by_menu_cap=max(count_before-len(ordered), 0),
                          ordered_nonstop_count=len(ordered), route=route,
                          coverage_counts=counts.tolist(), action_details=action_details,
                          cache_enabled=False, ordering="detour_desc_interleave_improve_replace"))
