"""Synthetic-only baseline contracts; no scene/nav generator or learned model.

Manual coordinates, exact complete-graph predecessors and per-arm allow tables
provide independent route enumeration and closed-native comparison fixtures.
All native search/certification calls are exposed through TEST_ACCOUNTING.
"""
import copy
import itertools
import math
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rc3_global_local.problem import Parent, LocalProblem, signature, array_hash
from rc3_v4 import runtime
from rc3_fast import solver
from rc3_nav_value.r1.oracle import _instrument, _empty_internal
from rc3_global_local_impl_review import endpoint_baselines as adapter

TEST_ACCOUNTING = {}


def count(key, amount=1):
    TEST_ACCOUNTING[key] = TEST_ACCOUNTING.get(key, 0)+amount


def fixture(entry=0, exit=0, task_ids=None):
    # No true scene generation: hand-coded positions, A, and direct metric paths.
    xs = np.array([-3., 0., 1., 4., 8., 12.])
    m = len(xs)-1
    scene = dict(schema_version=2, nx=1, ny=1, seed=42, start=[xs[0], 0.],
        x_lines=[-5., 30.], y_lines=[-5., 30.], obstacles=[],
        tables=[dict(id=0, ix=0, iy=0, cx=20.5, cy=20.5, xmin=20., xmax=21., ymin=20., ymax=21., z=.75)],
        tasks=[dict(id=i, table_id=0, x=float(i), y=0., z=.75) for i in range(3)],
        candidates=[dict(id=j, table_id=0, x=float(xs[j+1]), y=0., theta=0., h=.4,
                         corridor_anchor=[float(xs[j+1]), 0.]) for j in range(m)],
        robot=dict(base_radius=.1, clearance=0., arm_forward_offset=0., arm_spacing=0.,
                   arm_min_reach=0., arm_max_reach=100., arm_sector_half_angle_deg=180.,
                   shoulder_height=.75, fixed_lift_height=.4))
    F = np.ones((3, m, 2), dtype=bool)
    A = np.zeros_like(F)
    A[0, [0, 1], 0] = True
    A[1, [1, 2, 3], 1] = True
    A[2, [3, 4], 1] = True
    D = np.abs(xs[:, None]-xs[None, :])
    pred = np.broadcast_to(np.arange(m+1)[:, None], (m+1, m+1)).copy()
    np.fill_diagonal(pred, -9999)
    nav = dict(costs=D, coordinates=np.column_stack([xs, np.zeros_like(xs)]).tolist(),
               service_nodes=list(range(m+1)), predecessors=pred,
               node_count=m+1, edge_count=(m+1)*m//2)
    parent = Parent(SimpleNamespace(scene=scene, nav=nav, arm_dist=np.zeros_like(F, dtype=float)), F, A)
    return LocalProblem(parent, list(range(3)) if task_ids is None else task_ids, [0], entry, exit)


def enumerate_cost(p):
    """Independent positive-gain enumeration, not production metrics/search."""
    C = p.F.any(2)
    best = math.inf
    routes = []
    if not len(C):
        return float(p.D[0, 0]), [[0, 0]]
    for length in range(1, min(len(C), C.shape[1])+1):
        for permutation in itertools.permutations(range(1, C.shape[1]+1), length):
            left = set(range(len(C)))
            for node in permutation:
                served = {t for t in left if C[t, node-1]}
                if not served:
                    break
                left -= served
            else:
                if left:
                    continue
                route = [0, *permutation, 0]
                value = sum(float(p.D[a, b]) for a, b in zip(route, route[1:]))+.75*length
                if value < best-1e-9:
                    best, routes = value, [route]
                elif abs(value-best) <= 1e-9:
                    routes.append(route)
    return best, routes


def immutable_fixture(entry=0, exit=0, task_ids=None):
    from rc3_global_local_impl_review.runtime_guards import (
        bind_navigation, current_navigation_sources, snapshot_parent, snapshot_local)
    from rc3_fast.planner import map_key
    p = fixture(entry, exit, task_ids)
    parent = p.parent
    nav = bind_navigation(parent.scene, parent.base.nav,
        source_map_key=map_key(parent.scene), source_sha256=current_navigation_sources(),
        provenance=dict(producer='hand-coded synthetic complete-graph metric fixture'))
    frozen = snapshot_parent(parent.scene, parent.F, parent.A, navigation=nav)
    return snapshot_local(frozen, p.task_ids.tolist(), p.table_ids, entry, exit)


def native_closed(p):
    """Run the original closed exporter on the same synthetic local domain."""
    if p.entry != p.exit:
        raise ValueError('Original exporter comparison requires a closed endpoint')
    shim = adapter.make_shim(p)
    original = p.parent.base.nav
    indices = [p.entry, *[int(x)+1 for x in p.candidate_ids]]
    shim.nav = dict(costs=p.D.copy(), coordinates=copy.deepcopy(original['coordinates']),
                    service_nodes=[original['service_nodes'][i] for i in indices],
                    predecessors=original['predecessors'][indices].copy())
    legacy_export = runtime.export_plan
    legacy_validate = solver.validate
    internal = _empty_internal()

    def exporting(*args, **kwargs):
        count('legacy_closed_export_attempts')
        answer = legacy_export(*args, **kwargs)
        count('legacy_closed_export_completed')
        return answer

    def validating(*args, **kwargs):
        count('legacy_closed_validation_attempts')
        answer = legacy_validate(*args, **kwargs)
        count('legacy_closed_validation_completed')
        return answer

    count('legacy_closed_attempts')
    with _instrument(internal), patch.object(runtime, 'export_plan', exporting), patch.object(solver, 'validate', validating):
        result = runtime.run(shim, 'fixed3', horizon=0)
    count('legacy_closed_completed')
    for key, value in internal['counts'].items():
        count('legacy_native_'+key+'_calls', value)
    return result


class EndpointBaselineContracts(unittest.TestCase):
    def run_view(self, p, method='fixed3', **kwargs):
        return adapter.run_endpoint(p, method, counters=TEST_ACCOUNTING, **kwargs)

    def test_fixed3_closed_matches_native_route_and_objective(self):
        p = fixture()
        got, info = self.run_view(p)
        old, oldinfo = native_closed(p)
        remapped = [int(p.candidate_ids[j]) for j in old['visit_order_candidate_ids']]
        self.assertEqual(got['visit_order_candidate_ids'], remapped)
        self.assertAlmostEqual(got['objective_length_plus_stops_m'], old['objective_length_plus_stops_m'])
        self.assertFalse(info['fallback_used']); self.assertFalse(oldinfo['fallback_used'])
        self.assertEqual(info['portfolio_recipes'], [0, 1, 7])
        self.assertEqual(info['internal_search']['counts']['recipe'], 3)

    def test_fixed3_open_matches_independent_exhaustive_cost(self):
        p = fixture(2, 5)
        got, info = self.run_view(p)
        expected, routes = enumerate_cost(p)
        self.assertAlmostEqual(expected, 12.5)
        self.assertAlmostEqual(got['objective_length_plus_stops_m'], expected)
        self.assertIn(info['local_route'], routes)
        self.assertNotEqual(got['route_polyline'][0], got['route_polyline'][-1])
        self.assertEqual(got['navigation_global_nodes'][0], 2)
        self.assertEqual(got['navigation_global_nodes'][-1], 5)
        self.assertTrue(got['validation']['endpoint_constraints_satisfied'])

    def test_detour16_open_uses_original_repair_cap(self):
        p = fixture(2, 5)
        got, info = self.run_view(p, 'detour16')
        best, _ = enumerate_cost(p)
        self.assertAlmostEqual(got['objective_length_plus_stops_m'], best)
        self.assertEqual(info['horizon_cap'], 16)
        self.assertGreater(info['dp_calls'], 0)
        self.assertLessEqual(info['dp_calls'], 16)
        self.assertFalse(info['fallback_used'])

    def test_detour16_closed_is_nonincreasing_from_fixed3(self):
        p = fixture()
        reference, _ = self.run_view(p)
        got, info = self.run_view(p, 'detour16')
        self.assertLessEqual(got['objective_length_plus_stops_m'], reference['objective_length_plus_stops_m']+1e-8)
        self.assertTrue(info['cost_nonincreasing'])

    def test_endpoint_column_is_not_artificially_symmetrized(self):
        p = fixture(2, 5)
        self.assertFalse(np.array_equal(p.D, p.D.T))
        shim = adapter.make_shim(p)
        self.assertTrue(np.array_equal(shim.nav['costs'], p.D))
        self.assertEqual(shim.nav['costs'][0, 0], 11.)

    def test_mapping_and_effective_F_are_preserved(self):
        p = fixture(2, 5, [1, 2])
        shim = adapter.make_shim(p)
        self.assertEqual([c['id'] for c in shim.scene['candidates']], list(range(len(p.candidate_ids))))
        self.assertEqual([c['global_id'] for c in shim.scene['candidates']], p.candidate_ids.tolist())
        self.assertTrue(np.array_equal(shim.F, p.F))
        got, _ = self.run_view(p)
        self.assertEqual({a['task_id'] for a in got['assignments']}, {1, 2})
        self.assertTrue(all(a['arm'] == 'R' for a in got['assignments']))
        self.assertTrue(set(got['visit_order_candidate_ids']) <= set(p.candidate_ids))

    def test_full_parent_geometry_extents_survive_shim(self):
        p = fixture(2, 5)
        shim = adapter.make_shim(p)
        self.assertEqual(shim.scene['tables'], p.parent.scene['tables'])
        self.assertEqual(shim.scene['x_lines'], p.parent.scene['x_lines'])
        self.assertEqual(shim.scene['y_lines'], p.parent.scene['y_lines'])
        self.assertEqual(shim.scene['start'], [1., 0.])

    def test_empty_open_route_uses_direct_exit_cost(self):
        p = fixture(2, 5, [])
        for method in ('fixed3', 'detour16'):
            got, info = self.run_view(p, method)
            self.assertEqual(info['local_route'], [0, 0])
            self.assertEqual(got['visit_order_candidate_ids'], [])
            self.assertEqual(got['objective_length_plus_stops_m'], 11.)
            self.assertEqual(got['navigation_global_nodes'], [2, 5])

    def test_empty_closed_matches_original_exporter(self):
        p = fixture(task_ids=[])
        got, _ = self.run_view(p)
        old, _ = native_closed(p)
        self.assertEqual(got['objective_length_plus_stops_m'], old['objective_length_plus_stops_m'])
        self.assertEqual(got['objective_length_plus_stops_m'], 0.)

    def test_service_at_entry_or_exit_zero_legs_is_valid(self):
        p = fixture(2, 4)
        got, _ = self.run_view(p)
        self.assertIn(0., got['route_leg_costs_m'])
        self.assertTrue(got['validation']['valid_model_plan'])

    def test_input_scene_F_A_D_and_functions_unchanged(self):
        p = fixture(2, 5)
        before = (signature(p.parent.scene), array_hash(p.parent.F), array_hash(p.parent.A),
                  array_hash(p.parent.D), array_hash(p.D), array_hash(p.F))
        oldexport = runtime.export_plan
        old_construct, old_two_opt = solver.construct, solver.two_opt
        self.run_view(p, 'detour16')
        after = (signature(p.parent.scene), array_hash(p.parent.F), array_hash(p.parent.A),
                 array_hash(p.parent.D), array_hash(p.D), array_hash(p.F))
        self.assertEqual(before, after)
        self.assertIs(runtime.export_plan, oldexport)
        self.assertIs(solver.construct, old_construct); self.assertIs(solver.two_opt, old_two_opt)

    def test_no_extra_common_H_is_added(self):
        p = fixture(2, 5)
        with patch('rc3_global_local.terminal.finish', side_effect=AssertionError('Common H must not run')):
            _, info = self.run_view(p)
        self.assertFalse(info['common_H_added'])

    def test_invalid_method_and_mutated_endpoint_D_fail_before_search(self):
        before = TEST_ACCOUNTING.get('native_construct_calls', 0)
        with self.assertRaises(ValueError):self.run_view(fixture(), 'not_a_baseline')
        p = fixture(2, 5)
        p.D[0, 1:] = p.D[1:, 0]
        with self.assertRaises(ValueError):self.run_view(p)
        self.assertEqual(TEST_ACCOUNTING.get('native_construct_calls', 0), before)

    def test_certification_failure_restores_runtime_callback(self):
        p = fixture(2, 5)
        original = runtime.export_plan
        with patch.object(adapter, 'endpoint_validate', return_value=dict(valid_model_plan=False, all_tasks_served=False)):
            with self.assertRaises(ValueError):self.run_view(p)
        self.assertIs(runtime.export_plan, original)

    def test_original_failure_or_fallback_is_not_silently_normalized(self):
        count('mocked_runtime_fallback_cases')
        with patch.object(runtime, 'run', return_value=({}, {'fallback_used':True, 'failures':[{'stage':'synthetic'}]})):
            with self.assertRaises(adapter.EndpointBaselineError) as cm:self.run_view(fixture())
        self.assertTrue(cm.exception.details['fallback_used'])

    def test_explicit_soft_deadline_remains_visible(self):
        got, info = self.run_view(fixture(2, 5), 'detour16', time_limit_s=0.)
        self.assertTrue(got['validation']['valid_model_plan'])
        self.assertEqual(info['stop_reason'], 'deadline')
        self.assertTrue(info['deadline_exceeded'])
        self.assertEqual(info['dp_calls'], 0)

    def test_closed_full_parent_fixed3_matches_compressed_subset(self):
        p = fixture(task_ids=[1, 2])
        self.assertLess(len(p.candidate_ids), len(p.parent.scene['candidates']))
        got, _ = self.run_view(p)
        original, info = adapter.closed_native_reference(p, 'fixed3', counters=TEST_ACCOUNTING)
        self.assertEqual(got['visit_order_candidate_ids'], original['visit_order_candidate_ids'])
        self.assertAlmostEqual(got['objective_length_plus_stops_m'], original['objective_length_plus_stops_m'])
        self.assertEqual(info['candidate_count'], len(p.parent.scene['candidates']))
        self.assertTrue(info['original_export_and_proxy_validator_unpatched'])

    def test_closed_full_parent_detour_matches_compressed_subset(self):
        p = fixture(task_ids=[1, 2])
        got, gotinfo = self.run_view(p, 'detour16')
        original, info = adapter.closed_native_reference(p, 'detour16', counters=TEST_ACCOUNTING)
        self.assertEqual(got['visit_order_candidate_ids'], original['visit_order_candidate_ids'])
        self.assertAlmostEqual(got['objective_length_plus_stops_m'], original['objective_length_plus_stops_m'])
        self.assertEqual(gotinfo['dp_calls'], info['dp_calls'])

    def test_full_reference_passes_parent_domain_and_original_exporter(self):
        p = fixture(task_ids=[1, 2]); original_run = runtime.run; old_export = runtime.export_plan
        def watching(native, *args, **kwargs):
            self.assertEqual(native.F.shape, (2, 5, 2))
            self.assertTrue(np.array_equal(native.F, (p.parent.F & p.parent.A)[[1, 2]]))
            self.assertIs(native.nav, p.parent.base.nav)
            self.assertEqual(native.scene['candidates'], p.parent.scene['candidates'])
            self.assertIs(runtime.export_plan, old_export)
            return original_run(native, *args, **kwargs)
        with patch.object(runtime, 'run', side_effect=watching):
            _, info = adapter.closed_native_reference(p, counters=TEST_ACCOUNTING)
        self.assertFalse(info['fallback_used']); self.assertIsNone(sys.getprofile())

    def test_full_reference_rejects_open_external_and_active_profiler(self):
        with self.assertRaises(ValueError):adapter.closed_native_reference(fixture(2, 5), counters=TEST_ACCOUNTING)
        p = fixture(); p.parent.mode = 'external_tables'
        with self.assertRaises(ValueError):adapter.closed_native_reference(p, counters=TEST_ACCOUNTING)
        def existing_profile(*args):pass
        sys.setprofile(existing_profile)
        try:
            with self.assertRaises(ValueError):adapter.closed_native_reference(fixture(), counters=TEST_ACCOUNTING)
            self.assertIs(sys.getprofile(), existing_profile)
        finally:sys.setprofile(None)

    def test_open_adapter_accepts_immutable_guard_snapshot(self):
        p = immutable_fixture(2, 5)
        before = (signature(p.parent.scene), array_hash(p.F), array_hash(p.D))
        shim = adapter.make_shim(p)
        self.assertIs(type(shim.scene), dict)
        self.assertIs(type(shim.scene['tasks']), list)
        self.assertIs(type(shim.scene['candidates'][0]), dict)
        shim.scene['tasks'][0]['x'] = 99.
        self.assertNotEqual(p.parent.scene['tasks'][0]['x'], 99.)
        got, _ = self.run_view(p, 'detour16')
        self.assertAlmostEqual(got['objective_length_plus_stops_m'], 12.5)
        self.assertEqual(before, (signature(p.parent.scene), array_hash(p.F), array_hash(p.D)))

    def test_closed_full_reference_accepts_immutable_guard_snapshot(self):
        p = immutable_fixture(task_ids=[1, 2])
        before = (signature(p.parent.scene), array_hash(p.F), array_hash(p.D))
        adapted, _ = self.run_view(p)
        original, info = adapter.closed_native_reference(p, counters=TEST_ACCOUNTING)
        self.assertEqual(adapted['visit_order_candidate_ids'], original['visit_order_candidate_ids'])
        self.assertAlmostEqual(adapted['objective_length_plus_stops_m'], original['objective_length_plus_stops_m'])
        self.assertTrue(info['original_export_and_proxy_validator_unpatched'])
        self.assertEqual(before, (signature(p.parent.scene), array_hash(p.F), array_hash(p.D)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
