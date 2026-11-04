"""Versioned external reachability tables, guarded planning and record validation.

This adapter does not solve IK. Audit-table checks and an optional independent
service_checker are deliberately distinct from the old spherical-sector proxy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

SCHEMA = 'rc3-external-reachability-v1'
PLAN_SCHEMA = 'rc3-external-plan-v1'
CONSTRAINTS = ('joint_limits', 'end_effector_pose', 'self_collision',
               'environment_collision', 'lift_configuration')
SCOPE = ('External-table consistency plus circular-base/AABB navigation checks; '
         'no built-in IK, arm swept-path, lift-transition or cooperative-task certification.')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _point(point):
    return isinstance(point, (list, tuple)) and len(point) == 2 and all(map(_finite, point))


def _scene_check(scene):
    _canonical(scene)  # Reject NaN / infinity even in ancillary imported fields.
    for kind in ('tasks', 'candidates', 'tables'):
        records = scene[kind]
        _require(isinstance(records, list), f'{kind} must be a list')
        _require(all(type(v.get('id')) is int for v in records)
                 and [v['id'] for v in records] == list(range(len(records))),
                 f'{kind} IDs must be contiguous integers starting at zero')
    for task in scene['tasks']:
        _require(all(_finite(task[k]) for k in ('x', 'y', 'z')), 'invalid task coordinates')
        _require(type(task['table_id']) is int and 0 <= task['table_id'] < len(scene['tables']),
                 'invalid task table ID')
        _require(task.get('service_mode') == 'independent_single_arm',
                 'cooperative/simultaneous dual-arm tasks require a joint-service schema')
    for c in scene['candidates']:
        _require(all(_finite(c[k]) for k in ('x', 'y', 'theta', 'h')) and c['h'] >= 0,
                 'candidate pose must be finite with nonnegative lift height')
        _require(_point(c['corridor_anchor']), 'invalid candidate corridor anchor')
    for box in [*scene['tables'], *scene.get('obstacles', [])]:
        _require(all(_finite(box[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
                 and box['xmin'] < box['xmax'] and box['ymin'] < box['ymax'], 'invalid obstacle')
    _require(_point(scene['start']), 'invalid depot')
    for k in ('x_lines', 'y_lines'):
        _require(isinstance(scene[k], list) and len(scene[k]) > 0
                 and all(map(_finite, scene[k])) and scene[k] == sorted(set(scene[k])),
                 f'{k} must be nonempty sorted unique finite corridor coordinates')
    for key in ('base_radius', 'clearance'):
        _require(_finite(scene['robot'][key]) and scene['robot'][key] >= 0,
                 f'{key} must be finite and nonnegative')


def _matrix(value, shape):
    raw = np.asarray(value)
    if math.prod(shape) == 0 and raw.size == 0:
        raw = raw.reshape(shape).astype(bool)
    _require(raw.shape == tuple(shape), f'F shape must be {tuple(shape)}; got {raw.shape}')
    _require(raw.dtype.kind in 'biu' and np.isin(raw, [0, 1]).all(), 'F entries must be bool or integer 0/1')
    return raw.astype(bool)


def _source_check(source):
    _require(isinstance(source, dict), 'source must be a mapping')
    for k in ('producer_id', 'generator', 'generator_version', 'robot_model_id', 'generated_utc'):
        _require(isinstance(source.get(k), str) and bool(source[k].strip()), f'source requires {k}')
    _require(source.get('evidence_level') in ('synthetic_demo', 'externally_attested'), 'invalid evidence_level')
    assets = source.get('asset_sha256')
    _require(isinstance(assets, dict) and bool(assets), 'source requires model/constraint asset hashes')
    for k, v in assets.items():
        _require(isinstance(k, str) and isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) is not None,
                 'invalid asset SHA-256')
    statuses = source.get('constraint_checks', {})
    _require(set(statuses) == set(CONSTRAINTS)
             and all(v in ('checked', 'not_checked', 'synthetic') for v in statuses.values()),
             'constraint_checks must name every required constraint with explicit status')
    if source['evidence_level'] == 'externally_attested':
        _require(all(v == 'checked' for v in statuses.values()),
                 'externally_attested tables require all constraint checks; this records a producer claim only')
        _require({'kinematics', 'joint_limits', 'collision_geometry', 'task_pose_spec', 'lift_spec'} <= set(assets),
                 'externally_attested tables require version hashes for all physical input assets')
    _require(source.get('navigation_footprint_scope') == 'conservative_circular_bound_all_candidate_heights',
             'source must explicitly assert the navigation footprint covers every candidate height')


def make_certificate(scene, F, source, *, role, witnesses=None):
    """Seal producer-supplied data; hashes bind values, not physical truth/authorship."""
    _scene_check(scene)
    _source_check(source)
    _require(role in ('planning', 'audit'), 'certificate role must be planning or audit')
    shape = [len(scene['tasks']), len(scene['candidates']), 2]
    matrix = _matrix(F, shape)
    cert = dict(schema=SCHEMA, role=role, task_ids=[t['id'] for t in scene['tasks']],
                candidate_ids=[c['id'] for c in scene['candidates']], arm_order=['L', 'R'],
                shape=shape, scene_sha256=digest(scene), F=matrix.astype(int).tolist(),
                source=copy.deepcopy(source), witnesses=copy.deepcopy(witnesses or []))
    cert['F_sha256'] = digest(dict(shape=shape, values=cert['F']))
    cert['certificate_sha256'] = digest(cert)
    return cert


def check_certificate(scene, certificate, role=None):
    """Check shape, ordering, metadata and content binding, then return bool F."""
    _scene_check(scene)
    c = certificate
    _require(c.get('schema') == SCHEMA, 'unsupported external certificate schema')
    _require(c.get('role') in ('planning', 'audit') and (role is None or c['role'] == role),
             'certificate role mismatch')
    _require(c.get('arm_order') == ['L', 'R'], 'arm order must be L,R')
    shape = [len(scene['tasks']), len(scene['candidates']), 2]
    _require(c.get('shape') == shape, 'declared F shape differs from scene')
    _require(c.get('task_ids') == [t['id'] for t in scene['tasks']]
             and c.get('candidate_ids') == [v['id'] for v in scene['candidates']], 'ID ordering mismatch')
    _require(c.get('scene_sha256') == digest(scene), 'scene binding hash mismatch')
    _source_check(c['source'])
    F = _matrix(c['F'], shape)
    _require(c.get('F_sha256') == digest(dict(shape=shape, values=F.astype(int).tolist())), 'F hash mismatch')
    body = {k: v for k, v in c.items() if k != 'certificate_sha256'}
    _require(c.get('certificate_sha256') == digest(body), 'certificate hash mismatch')
    _require(isinstance(c.get('witnesses'), list), 'witnesses must be a list')
    return F


def check_bundle(bundle):
    _require(bundle.get('schema') == SCHEMA, 'unsupported external bundle schema')
    scene, planning, audit = bundle['scene'], bundle['planning'], bundle['audit']
    F = check_certificate(scene, planning, 'planning')
    A = check_certificate(scene, audit, 'audit')
    p, a = planning['source'], audit['source']
    _require(p['producer_id'] != a['producer_id'], 'audit must identify a distinct producer/checker')
    _require(p['robot_model_id'] == a['robot_model_id'] and p['asset_sha256'] == a['asset_sha256'],
             'planning and audit must bind the same robot/constraint asset versions')
    _require(p['evidence_level'] == a['evidence_level'], 'mixed evidence levels are unsupported')
    return F, A


def save_bundle(bundle, path):
    """JSON is directly usable by MATLAB jsondecode; NPZ is a Python compact alternative."""
    check_bundle(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == '.json':
        path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    elif path.suffix.lower() == '.npz':
        metadata = copy.deepcopy(bundle)
        arrays = {role: np.asarray(metadata[role].pop('F'), dtype=np.uint8).reshape(metadata[role]['shape'])
                  for role in ('planning', 'audit')}
        np.savez_compressed(path, metadata_json=np.asarray(_canonical(metadata).decode('utf-8')),
                            planning_F=arrays['planning'], audit_F=arrays['audit'])
    else:
        raise ValueError('bundle extension must be .json or .npz')


def seal_draft(draft):
    """Seal plain JSON exported by MATLAB without requiring foreign JSON hashing."""
    scene = copy.deepcopy(draft['scene'])
    bundle = dict(schema=SCHEMA, scene=scene)
    for role in ('planning', 'audit'):
        raw = draft[role]
        bundle[role] = make_certificate(scene, raw['F'], raw['source'], role=role,
                                        witnesses=raw.get('witnesses'))
    check_bundle(bundle)
    return bundle


def load_bundle(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        bundle = json.loads(path.read_text(encoding='utf-8-sig'))
    elif path.suffix.lower() == '.npz':
        with np.load(path, allow_pickle=False) as raw:
            _require(set(raw.files) == {'metadata_json', 'planning_F', 'audit_F'}, 'unexpected NPZ fields')
            bundle = json.loads(str(raw['metadata_json'].item()))
            for role in ('planning', 'audit'):
                F = _matrix(raw[role+'_F'], bundle[role]['shape'])
                bundle[role]['F'] = F.astype(int).tolist()
    else:
        raise ValueError('bundle extension must be .json or .npz')
    check_bundle(bundle)
    return bundle


def prepare_external(bundle, penalty=.75):
    """Use the shared navigation/solver, with no proxy F or proxy validation calls."""
    from rc3_baseline.baseline import navigation
    from rc3_fast.solver import Problem
    start = time.perf_counter()
    F, _ = check_bundle(bundle)
    _require(_finite(penalty) and penalty >= 0, 'penalty must be finite and nonnegative')
    scene = copy.deepcopy(bundle['scene'])
    nav = navigation(scene)
    sparse = [dict(task_id=i, left=np.flatnonzero(F[i, :, 0]).tolist(),
                   right=np.flatnonzero(F[i, :, 1]).tolist()) for i in range(len(F))]
    # No fabricated arm distance/quality: deterministic L-before-R tie break.
    return Problem(scene, F, np.zeros(F.shape), sparse, nav, time.perf_counter()-start, penalty)


def _segment_box(a, b, box):
    """Scalar independent collision check, distinct from vectorized navigator."""
    lo, hi = 0., 1.
    for axis in (0, 1):
        d = b[axis]-a[axis]
        if abs(d) < 1e-12:
            if a[axis] < box[axis] or a[axis] > box[axis+2]:
                return False
        else:
            x, y = (box[axis]-a[axis])/d, (box[axis+2]-a[axis])/d
            lo, hi = max(lo, min(x, y)), min(hi, max(x, y))
            if lo > hi:
                return False
    return True


def validate_external(bundle, plan, service_checker=None, *, expected_penalty=.75):
    """Recompute the certificate and complete record checks; ignore stored verdicts.

    service_checker(task, candidate, arm, audit_certificate) optionally performs
    caller-owned IK/collision rechecking. Exceptions/rejections fail closed.
    A callback's identity/correctness is outside this adapter's guarantees.
    expected_penalty comes from the caller's planning request, not the plan's
    self-reported value. Supply it explicitly when using a nondefault objective.
    """
    errors = []
    try:
        _require(_finite(expected_penalty) and expected_penalty >= 0,
                 'expected_penalty must be finite and nonnegative')
        F, A = check_bundle(bundle)
        scene = bundle['scene']
        n, m = len(scene['tasks']), len(scene['candidates'])
        _require(plan.get('schema') == PLAN_SCHEMA and plan.get('status') == 'complete', 'not a complete external plan')
        _require(plan.get('reachability_certificate_sha256') == bundle['planning']['certificate_sha256']
                 and plan.get('audit_certificate_sha256') == bundle['audit']['certificate_sha256'],
                 'plan certificate association mismatch')
        order = plan['visit_order_candidate_ids']
        selected = plan['selected_candidate_ids']
        _require(all(type(j) is int and 0 <= j < m for j in order)
                 and len(order) == len(set(order)) and selected == order, 'invalid selected set/visit order')
        assignment = plan['assignments']
        ids = [item['task_id'] for item in assignment]
        _require(all(type(i) is int for i in ids) and sorted(ids) == list(range(n)),
                 'each known task must be assigned exactly once')
        _require(plan['unserved_task_ids'] == [] and plan['unserved_reasons'] == [], 'unserved bookkeeping mismatch')
        checked = 0
        for a in assignment:
            i, j, arm = a['task_id'], a['candidate_id'], a['arm']
            _require(type(j) is int and j in order and arm in ('L', 'R'), 'invalid candidate/arm assignment')
            k = 0 if arm == 'L' else 1
            _require(bool(F[i, j, k]), f'task {i}: planning F is zero')
            _require(bool(A[i, j, k]), f'task {i}: independent audit F rejects assigned service')
            if service_checker is not None:
                _require(service_checker(scene['tasks'][i], scene['candidates'][j], arm, bundle['audit']) is True,
                         f'task {i}: external service_checker rejected service')
                checked += 1
        line, legs = plan['route_polyline'], plan['route_leg_costs_m']
        _require(isinstance(line, list) and bool(line) and all(_point(p) for p in line), 'invalid route polyline')
        _require(len(legs) == len(order)+1 and all(_finite(x) and x >= 0 for x in legs), 'invalid navigation-leg costs')
        _require(math.dist(line[0], scene['start']) < 1e-8 and math.dist(line[-1], scene['start']) < 1e-8,
                 'route is not closed at the depot')
        cumulative, length = [0.], 0.
        r = scene['robot']['base_radius']+scene['robot']['clearance']
        boxes = [(v['xmin']-r, v['ymin']-r, v['xmax']+r, v['ymax']+r)
                 for v in [*scene['tables'], *scene.get('obstacles', [])]]
        point_hits = sum(any(b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3] for b in boxes) for p in line)
        collisions = point_hits
        for a, b in zip(line, line[1:]):
            length += math.dist(a, b)
            cumulative.append(length)
            collisions += sum(_segment_box(a, b, box) for box in boxes)
        _require(collisions == 0, f'{collisions} base-path obstacle collisions')
        for key in ('route_length_m', 'stop_penalty_equivalent_m', 'objective_length_plus_stops_m'):
            _require(_finite(plan[key]) and plan[key] >= 0, f'invalid {key}')
        _require(math.isclose(plan['stop_penalty_equivalent_m'], expected_penalty,
                             rel_tol=0., abs_tol=1e-12),
                 'stop penalty differs from the independent planning request')
        _require(math.isclose(length, plan['route_length_m'], rel_tol=1e-8, abs_tol=1e-7)
                 and math.isclose(length, sum(legs), rel_tol=1e-8, abs_tol=1e-7), 'path length/leg metric mismatch')
        expected = length+len(order)*expected_penalty
        _require(math.isclose(expected, plan['objective_length_plus_stops_m'], rel_tol=1e-8, abs_tol=1e-7),
                 'objective does not equal path length plus stop penalties')
        cursor, target_distance = 0, 0.
        for cost, j in zip(legs, order+[None]):
            target_distance += cost
            target = scene['start'] if j is None else [scene['candidates'][j][key] for key in ('x', 'y')]
            pos = next((k for k in range(cursor, len(line))
                        if math.isclose(cumulative[k], target_distance, abs_tol=1e-7, rel_tol=1e-8)
                        and math.dist(line[k], target) < 1e-8), None)
            _require(pos is not None, f'declared service order/leg endpoint mismatch at candidate {j}')
            cursor = pos
        stops = plan['service_stops']
        _require([v['candidate_id'] for v in stops] == order, 'service stop order mismatch')
        arrival = 0.
        for k, stop in enumerate(stops):
            j = order[k]
            arrival += legs[k]
            _require(type(stop['sequence']) is int and stop['sequence'] == k+1, 'invalid sequence number')
            _require(all(_finite(stop['pose'][key]) and abs(stop['pose'][key]-scene['candidates'][j][key]) < 1e-8
                         for key in ('x', 'y', 'theta', 'h')), 'service pose, including lift, differs from candidate')
            _require(_finite(stop['arrival_distance_m'])
                     and math.isclose(stop['arrival_distance_m'], arrival, abs_tol=1e-7, rel_tol=1e-8),
                     'invalid service arrival distance')
            actual = sorted((a['task_id'], a['arm']) for a in stop['tasks'])
            wanted = sorted((a['task_id'], a['arm']) for a in assignment if a['candidate_id'] == j)
            _require(actual == wanted, 'stop tasks differ from assignments')
        return dict(valid_model_plan=True, all_tasks_served=True, errors=[], collision_count=0,
                    independent_polyline_length_m=length, service_checker_calls=checked,
                    built_in_physical_ik_recheck=False, evidence_level=bundle['audit']['source']['evidence_level'], scope=SCOPE)
    except Exception as error:
        errors.append(str(error))
        return dict(valid_model_plan=False, all_tasks_served=False, errors=errors,
                    built_in_physical_ik_recheck=False, scope=SCOPE)


def export_external(problem, route, bundle, method='external_fixed3', service_checker=None):
    from rc3_baseline.baseline import expand_route
    from rc3_improve.environment import canonical_route
    route = canonical_route(problem, route)
    _, audit_F = check_bundle(bundle)
    left = np.ones(len(problem.F), dtype=bool)
    order, assignments = [], []
    for node in route[1:-1]:
        j = int(node)-1
        order.append(j)
        # Choose only a jointly permitted arm. Tasks rejected here remain
        # available to later stops; final validation still checks both tables
        # separately and executes the caller-owned service_checker.
        common = problem.F[:, j] & audit_F[:, j]
        for i in np.flatnonzero(left & common.any(1)):
            arm = 'L' if common[i, 0] else 'R'
            assignments.append(dict(task_id=int(i), candidate_id=j, arm=arm))
            left[i] = False
    _require(not left.any(), 'cannot export an incomplete route')
    line, legs = expand_route(route, problem.nav)
    stops, arrival = [], 0.
    for k, j in enumerate(order):
        arrival += legs[k]
        stops.append(dict(sequence=k+1, candidate_id=j,
                          pose={key: problem.scene['candidates'][j][key] for key in ('x', 'y', 'theta', 'h')},
                          arrival_distance_m=arrival,
                          tasks=[dict(task_id=a['task_id'], arm=a['arm']) for a in assignments if a['candidate_id'] == j]))
    plan = dict(schema=PLAN_SCHEMA, method=method, status='complete', selected_candidate_ids=order,
                visit_order_candidate_ids=order, assignments=assignments, service_stops=stops,
                unserved_task_ids=[], unserved_reasons=[], route_polyline=line, route_leg_costs_m=legs,
                route_length_m=sum(legs), stop_penalty_equivalent_m=problem.penalty,
                objective_length_plus_stops_m=sum(legs)+len(order)*problem.penalty,
                reachability_certificate_sha256=bundle['planning']['certificate_sha256'],
                audit_certificate_sha256=bundle['audit']['certificate_sha256'])
    plan['validation'] = validate_external(bundle, plan, service_checker,
                                           expected_penalty=problem.penalty)
    _require(plan['validation']['valid_model_plan'], '; '.join(plan['validation']['errors']))
    return plan


def plan_external(bundle, method='fixed3', horizon=8, penalty=.75, service_checker=None):
    """Build a certified anchor and optionally repair it. No built-in physical solver."""
    from rc3_fast.solver import route_objective
    from rc3_v3.environment import Env, FEATURE_NAMES, reference_candidates
    started = time.perf_counter()
    task_ids = [t.get('id') for t in bundle.get('scene', {}).get('tasks', []) if isinstance(t, dict)]
    try:
        _require(method in ('fixed3', 'detour'), 'external mode supports fixed3 or detour')
        _require(type(horizon) is int and horizon >= 0, 'horizon must be a nonnegative integer')
        problem = prepare_external(bundle, penalty)
        planning_started = time.perf_counter()
        F, A = check_bundle(bundle)
        candidates, failures = reference_candidates(problem)
        anchor = anchor_route = None
        for score, recipe, route in candidates:
            try:
                anchor = export_external(problem, route, bundle, 'external_'+method, service_checker)
                _require(abs(anchor['objective_length_plus_stops_m']-score) < 1e-7, 'abstract/export cost mismatch')
                anchor_route = route
                break
            except Exception as error:
                failures.append(dict(stage='certify_anchor', recipe=recipe, error=str(error)))
        _require(anchor is not None, f'no externally audited complete anchor: {failures}')
        returned, calls, accepted = anchor, 0, 0
        reference = anchor['objective_length_plus_stops_m']
        if method == 'detour':
            try:
                env = Env(problem, horizon, initial_route=anchor_route)
                while env.steps < horizon:
                    features, mask = env.observe()
                    if env.done or not mask.any():
                        break
                    action = int(np.where(mask, features[:, FEATURE_NAMES.index('detour_excess')], -np.inf).argmax())
                    calls += 1
                    _, _, info = env.step(action)
                    accepted += int(info['accepted'])
                if route_objective(problem, env.route) < reference-1e-8:
                    proposal = export_external(problem, env.route, bundle, 'external_detour', service_checker)
                    _require(proposal['objective_length_plus_stops_m'] <= reference+1e-8, 'cost bound violated')
                    returned = proposal
            except Exception as error:
                failures.append(dict(stage='repair_or_final_certification', error=str(error)))
        return dict(schema=PLAN_SCHEMA, status='complete', scene=problem.scene, external_bundle=bundle,
                    plans={method: returned}, diagnostics=dict(reference_objective_m=reference,
                    objective_m=returned['objective_length_plus_stops_m'], cost_nonincreasing=True,
                    dp_calls=calls, accepted=accepted, failures=failures,
                    fallback_used=returned is anchor and bool(failures),
                    planning_positive_audit_negative=int((F & ~A).sum()),
                    planning_negative_audit_positive=int((~F & A).sum())),
                    timing=dict(preprocessing_s=problem.preprocessing_s,
                                planning_s=time.perf_counter()-planning_started, total_s=time.perf_counter()-started),
                    guarantee=SCOPE)
    except Exception as error:
        return dict(schema=PLAN_SCHEMA, status='failed', code='external_input_or_certification_failure',
                    detail=str(error), plans={}, pending_task_ids=task_ids,
                    physical_infeasibility_claim=False, total_s=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--method', choices=['fixed3', 'detour'], default='fixed3')
    parser.add_argument('--budget', type=int, default=8)
    parser.add_argument('--penalty', type=float, default=.75)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-plan', type=Path, help='recheck a saved external result instead of planning')
    parser.add_argument('--seal-draft', action='store_true', help='seal MATLAB/plain-JSON draft into a JSON/NPZ bundle')
    args = parser.parse_args()
    try:
        if args.seal_draft:
            _require(args.verify_plan is None, 'seal-draft and verify-plan are mutually exclusive')
            bundle = seal_draft(json.loads(args.bundle.read_text(encoding='utf-8-sig')))
            save_bundle(bundle, args.output)
            print(json.dumps(dict(status='complete', operation='seal_only_no_physical_verification', output=str(args.output)), ensure_ascii=True))
            return 0
        bundle = load_bundle(args.bundle)
        if args.verify_plan:
            data = json.loads(args.verify_plan.read_text(encoding='utf-8-sig'))
            _require(bool(data.get('plans')), 'saved result has no plans')
            checks = {k: validate_external(bundle, v, expected_penalty=args.penalty)
                      for k, v in data['plans'].items()}
            result = dict(status='complete' if all(c['valid_model_plan'] for c in checks.values()) else 'failed', checks=checks)
        else:
            result = plan_external(bundle, args.method, args.budget, args.penalty)
    except Exception as error:
        result = dict(status='failed', detail=str(error), physical_infeasibility_claim=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], output=str(args.output)), ensure_ascii=True))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
