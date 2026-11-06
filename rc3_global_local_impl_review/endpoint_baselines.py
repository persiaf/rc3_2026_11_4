"""Endpoint-aware adapters for the frozen Fixed3 and Detour16 pipelines.

The original algorithm is used on local continuous IDs and the explicit
entry/exit cost matrix. Only its export/certification callback is replaced.
There is no additional common-H invocation in this adapter.
"""
from __future__ import annotations

import copy
import math
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rc3_v4 import runtime
from rc3_nav_value.r1.oracle import _empty_internal, _instrument
from rc3_global_local.terminal import export as endpoint_export
from rc3_global_local.validation import validate as endpoint_validate

_LOCK = threading.RLock()
FIXED3_RECIPES = (0, 1, 7)


class EndpointBaselineError(RuntimeError):
    """A failed/fallback baseline remains an explicit, billable failure."""
    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = {} if details is None else details


def _add(counters, key, value=1):
    counters[key] = counters.get(key, 0) + value


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _plain_copy(value):
    """Detach JSON-compatible immutable containers without trusting deepcopy."""
    if isinstance(value, dict):
        return {key: _plain_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_copy(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def make_shim(problem):
    """Build the legacy algorithm view, without changing frozen input objects."""
    n, m, arms = problem.F.shape
    _require(arms == 2 and problem.C.shape == (n, m), 'Invalid local F/C shape')
    _require(np.array_equal(problem.C, problem.F.any(2)), 'F/C semantics differ')
    _require(problem.penalty == .75, 'Current objective requires stop penalty .75')
    _require(problem.D.shape == (m+1, m+1), 'Endpoint D shape mismatch')
    _require(len(problem.task_ids) == n and len(problem.candidate_ids) == m, 'ID map shape mismatch')
    parent = problem.parent
    target = (parent.F & parent.A)[np.ix_(problem.task_ids, problem.candidate_ids, np.arange(2))]
    _require(np.array_equal(problem.F, target), 'Local F must use same-arm parent F&A')
    nodes = np.asarray(problem.candidate_ids, dtype=int)+1
    _require(np.array_equal(problem.D[1:, 1:], parent.D[np.ix_(nodes, nodes)]), 'Service D differs from parent')
    _require(np.array_equal(problem.D[0, 1:], parent.D[problem.entry, nodes]), 'Outgoing depot row must mean entry')
    _require(np.array_equal(problem.D[1:, 0], parent.D[nodes, problem.exit]), 'Incoming depot column must mean exit')
    _require(problem.D[0, 0] == parent.D[problem.entry, problem.exit], 'Empty-route entry/exit cost mismatch')
    block = problem.D[1:, 1:]
    _require(np.array_equal(np.isfinite(block), np.isfinite(block.T)), 'Asymmetric service reachability')
    finite = np.isfinite(block)
    _require(np.allclose(block[finite], block.T[finite], rtol=0., atol=1e-8), '2-opt needs symmetric service distances')
    _require(np.isfinite(problem.D[0]).all() and np.isfinite(problem.D[:, 0]).all(), 'Unavailable endpoint connection')
    _require(not np.isnan(problem.D).any() and not (problem.D < 0).any(), 'Negative/NaN navigation cost')
    _require(math.isfinite(float(problem.D[0, 0])), 'Nonfinite direct endpoint cost')

    scene = _plain_copy(parent.scene)
    scene['tasks'] = [dict(_plain_copy(parent.scene['tasks'][int(i)]), id=k,
                           global_id=int(i)) for k, i in enumerate(problem.task_ids)]
    scene['candidates'] = [dict(_plain_copy(parent.scene['candidates'][int(j)]), id=k,
                                global_id=int(j)) for k, j in enumerate(problem.candidate_ids)]
    scene['start'] = (_plain_copy(parent.scene['start']) if problem.entry == 0 else
                      [parent.scene['candidates'][problem.entry-1][key] for key in ('x', 'y')])
    # Repair features retain full parent corridor extents and table/obstacle data.
    F = np.asarray(problem.F, dtype=bool).copy()
    sparse = [dict(task_id=i, left=np.flatnonzero(F[i, :, 0]).tolist(),
                   right=np.flatnonzero(F[i, :, 1]).tolist()) for i in range(n)]
    return SimpleNamespace(scene=scene, F=F, arm_dist=np.zeros(F.shape), sparse=sparse,
                           nav=dict(costs=np.asarray(problem.D, dtype=float).copy()),
                           penalty=.75, preprocessing_s=0.)


def run_endpoint(problem, method='fixed3', *, time_limit_s=None,
                 reject_fallback=True, counters=None):
    """Return ``(global_ID_plan, info)``; serialized patches are always restored.

    Preparation/global navigation is supplied by the caller and is not included
    in this function's algorithm timer. A full-request runner must count it.
    ``counters`` records all attempts, including invalid inputs and failures.
    """
    counters = {} if counters is None else counters
    _add(counters, 'baseline_attempts')
    internal = _empty_internal()
    started = time.perf_counter()
    try:
        if method not in ('fixed3', 'detour16'):
            raise ValueError('Expected fixed3 or detour16')
        shim = make_shim(problem)

        def certify_callback(_shim, route, label):
            _require(_shim is shim, 'Unexpected runtime problem')
            _add(counters, 'endpoint_export_attempts')
            canonical = [int(x) for x in route]
            plan = endpoint_export(problem, canonical, 'endpoint_'+method)
            _add(counters, 'endpoint_export_completed')
            _add(counters, 'endpoint_validation_attempts')
            checked = endpoint_validate(problem, plan)
            if not checked.get('valid_model_plan') or not checked.get('all_tasks_served'):
                raise EndpointBaselineError('Endpoint certification failed', details=checked)
            _add(counters, 'endpoint_validation_completed')
            plan['validation'] = checked
            return plan

        with _LOCK, _instrument(internal), patch.object(runtime, 'export_plan', certify_callback):
            plan, info = runtime.run(shim, 'fixed3' if method == 'fixed3' else 'detour',
                                     horizon=0 if method == 'fixed3' else 16,
                                     time_limit_s=time_limit_s)
        if reject_fallback and (info.get('fallback_used') or info.get('failures')):
            raise EndpointBaselineError('Native pipeline reported a failure/fallback; not a conforming normal baseline',
                                        details=copy.deepcopy(info))
        global_ids = plan['visit_order_candidate_ids']
        inverse = {int(j): k+1 for k, j in enumerate(problem.candidate_ids)}
        route = [0, *[inverse[int(j)] for j in global_ids], 0]
        metrics = problem.metrics(route)
        _require(abs(metrics['J']-plan['objective_length_plus_stops_m']) <= 1e-7,
                 'Returned endpoint objective differs from local route')
        _require(plan['entry_global_node'] == problem.entry and plan['exit_global_node'] == problem.exit,
                 'Returned route changed requested endpoints')
        info = dict(info, adapter_schema='rc3-endpoint-baseline-adapter-v1',
                    adapter_method=method, portfolio_recipes=list(FIXED3_RECIPES),
                    horizon_cap=0 if method == 'fixed3' else 16, local_route=route,
                    global_navigation_nodes=plan['navigation_global_nodes'],
                    source_pipeline='Frozen rc3_v4.runtime.run with original portfolio/search; endpoint export/validator callback',
                    common_H_added=False, endpoint_adapted_native=True,
                    original_closed_native_equivalence_scope='Same candidate/task domain, constraints and closed endpoint; route and objective, not a claim about historical result files',
                    internal_search=copy.deepcopy(internal),
                    adapter_elapsed_s=time.perf_counter()-started)
        _add(counters, 'baseline_completed')
        return plan, info
    except Exception:
        _add(counters, 'baseline_failed')
        raise
    finally:
        for key, value in internal['counts'].items():
            _add(counters, 'native_'+key+'_calls', value)
        _add(counters, 'adapter_wall_s', time.perf_counter()-started)


def endpoint_fixed3(problem, **kwargs):
    return run_endpoint(problem, 'fixed3', **kwargs)


def endpoint_detour16(problem, **kwargs):
    return run_endpoint(problem, 'detour16', **kwargs)


def closed_native_reference(problem, method='fixed3', *, time_limit_s=None,
                            reject_fallback=True, counters=None):
    """Original closed native pipeline on ALL parent candidates and navigation.

    Only target tasks are remapped to contiguous row IDs. No export or validator
    function is patched: a temporary call-profile observer counts their original
    entries/returns. This proxy-native comparison is deliberately unavailable
    for open endpoints or externally attested reachability-table backends.
    """
    counters = {} if counters is None else counters
    _add(counters, 'full_reference_attempts')
    started = time.perf_counter(); internal = _empty_internal()
    installed = False
    try:
        if method not in ('fixed3', 'detour16'):
            raise ValueError('Expected fixed3 or detour16')
        _require(problem.entry == problem.exit == 0, 'Original full native reference requires original depot0 at both ends')
        _require(problem.parent.mode == 'proxy', 'Original scalar proxy exporter cannot certify an external-table backend')
        make_shim(problem)  # Check source F/A and every endpoint-cost mapping first.
        parent = problem.parent
        scene = _plain_copy(parent.scene)
        scene['tasks'] = [dict(_plain_copy(parent.scene['tasks'][int(i)]), id=k,
                               global_id=int(i)) for k, i in enumerate(problem.task_ids)]
        F = (parent.F & parent.A)[problem.task_ids].copy()
        sparse = [dict(task_id=i, left=np.flatnonzero(F[i, :, 0]).tolist(),
                       right=np.flatnonzero(F[i, :, 1]).tolist()) for i in range(len(F))]
        native = SimpleNamespace(scene=scene, F=F, arm_dist=np.zeros(F.shape), sparse=sparse,
            nav=parent.base.nav, penalty=.75, preprocessing_s=0.)
        from rc3_fast.solver import export_plan as original_export
        from rc3_fast import solver
        expected_export_code = original_export.__code__
        expected_validation_code = solver.validate.__code__

        def observer(frame, event, result):
            code = frame.f_code
            label = ('full_reference_export' if code is expected_export_code else
                     'full_reference_validation' if code is expected_validation_code else None)
            if label is None:
                return
            if event == 'call':
                _add(counters, label+'_attempts')
            elif event == 'return':
                if isinstance(result, dict):
                    _add(counters, label+'_completed')
                    if label.endswith('validation') and result.get('valid_model_plan'):
                        _add(counters, label+'_passed')
                else:
                    _add(counters, label+'_aborted')

        with _LOCK:
            _require(runtime.export_plan is original_export, 'Original native exporter was changed by another caller')
            _require(sys.getprofile() is None, 'Full reference call observer cannot replace an active profiler')
            try:
                sys.setprofile(observer); installed = True
                with _instrument(internal):
                    plan, info = runtime.run(native, 'fixed3' if method == 'fixed3' else 'detour',
                                             horizon=0 if method == 'fixed3' else 16,
                                             time_limit_s=time_limit_s)
            finally:
                if installed:
                    sys.setprofile(None); installed = False
        if reject_fallback and (info.get('fallback_used') or info.get('failures')):
            raise EndpointBaselineError('Full native reference failed/fell back', details=copy.deepcopy(info))
        info = dict(info, reference_schema='rc3-closed-full-native-reference-v1',
            adapter_method=method, candidate_domain='ALL original parent candidates; target tasks only are remapped',
            task_ids_global=problem.task_ids.tolist(), candidate_count=F.shape[1],
            original_export_and_proxy_validator_unpatched=True, arm_assignment_tie_break='L-before-R via zero distance tie, both arms still constrained by effective F',
            common_H_added=False, internal_search=copy.deepcopy(internal),
            profiling_scope='Call-count observer only; original export/validator execute unchanged; overhead included in reference elapsed',
            reference_elapsed_s=time.perf_counter()-started)
        _add(counters, 'full_reference_completed')
        return plan, info
    except Exception:
        _add(counters, 'full_reference_failed')
        raise
    finally:
        if installed:
            sys.setprofile(None)
        for key, value in internal['counts'].items():
            _add(counters, 'full_reference_native_'+key+'_calls', value)
        _add(counters, 'full_reference_wall_s', time.perf_counter()-started)
