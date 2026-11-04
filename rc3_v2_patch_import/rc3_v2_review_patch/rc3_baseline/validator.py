"""Independent geometric and bookkeeping checks; this is not an IK validator."""
import argparse
import json
import math
from pathlib import Path


def segment_hits_rectangle(a, b, rect):
    """Closed segment versus a closed AABB, using the slab intersection test."""
    lo, hi = 0.0, 1.0
    for axis in (0, 1):
        d = b[axis] - a[axis]
        low, high = rect[axis], rect[axis + 2]
        if abs(d) < 1e-12:
            if a[axis] < low or a[axis] > high:
                return False
        else:
            t1, t2 = (low - a[axis]) / d, (high - a[axis]) / d
            lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
            if lo > hi:
                return False
    return True


def proxy_reachable(task, candidate, arm, robot):
    angle = candidate['theta']
    dx, dy = task['x'] - candidate['x'], task['y'] - candidate['y']
    xx = math.cos(angle) * dx + math.sin(angle) * dy - robot['arm_forward_offset']
    yy = -math.sin(angle) * dx + math.cos(angle) * dy
    yy -= robot['arm_spacing'] / 2 * (1 if arm == 'L' else -1)
    zz = task['z'] - robot['shoulder_height']
    r = math.sqrt(xx * xx + yy * yy + zz * zz)
    return (robot['arm_min_reach'] - 1e-10 <= r <= robot['arm_max_reach'] + 1e-10
            and abs(math.atan2(yy, xx)) <= math.radians(robot['arm_sector_half_angle_deg']) + 1e-10)


def validate(scene, plan, sparse_feasibility):
    """Fail closed on invalid records; no planner, NumPy or SciPy dependency."""
    try:
        return _validate(scene, plan, sparse_feasibility)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as error:
        return dict(valid_model_plan=False, all_tasks_served=False,
                    collision_count=None, unserved_task_ids=None,
                    independent_polyline_length_m=None,
                    errors=[f'malformed or nonfinite input: {error}'],
                    scope='record validation failed before geometric certification; no IK')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite_fields(record, keys):
    for key in keys:
        value = record[key]
        _require(not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value), f'{key} must be a finite number')


def _validate(scene, plan, sparse_feasibility):
    errors = []
    for records, key in ((scene['tasks'], 'id'), (scene['candidates'], 'id'),
                         (scene['tables'], 'id'), (sparse_feasibility, 'task_id')):
        identifiers = [record[key] for record in records]
        _require(all(type(value) is int for value in identifiers), f'{key} must be an integer')
        _require(len(set(identifiers)) == len(identifiers), f'duplicate {key} in scene or feasibility')
    for task in scene['tasks']:
        _finite_fields(task, ('x', 'y', 'z'))
    for dock in scene['candidates']:
        _finite_fields(dock, ('x', 'y', 'theta', 'h'))
    for table in [*scene['tables'], *scene.get('obstacles', [])]:
        _finite_fields(table, ('xmin', 'ymin', 'xmax', 'ymax'))
        _require(table['xmin'] < table['xmax'] and table['ymin'] < table['ymax'], 'invalid table rectangle')
    _finite_fields(scene['robot'], ('base_radius', 'clearance', 'arm_forward_offset', 'arm_spacing',
                                    'arm_min_reach', 'arm_max_reach', 'arm_sector_half_angle_deg', 'shoulder_height'))
    robot = scene['robot']
    _require(robot['base_radius'] >= 0 and robot['clearance'] >= 0 and robot['arm_spacing'] >= 0
             and 0 <= robot['arm_min_reach'] < robot['arm_max_reach']
             and 0 < robot['arm_sector_half_angle_deg'] <= 180, 'invalid robot geometry')
    for point in [scene['start'], *plan['route_polyline']]:
        _require(len(point) == 2 and all(not isinstance(v, bool) and isinstance(v, (int, float))
                                      and math.isfinite(v) for v in point), 'nonfinite or invalid 2D point')
    _finite_fields(plan, ('route_length_m',))
    _require(plan['route_length_m'] >= 0, 'route length must be nonnegative')
    _require(all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) and v >= 0
                 for v in plan['route_leg_costs_m']), 'navigation leg costs must be finite and nonnegative')
    tasks = {x['id']: x for x in scene['tasks']}
    docks = {x['id']: x for x in scene['candidates']}
    feasible = {x['task_id']: x for x in sparse_feasibility}
    assignments = plan['assignments']
    ids = [a['task_id'] for a in assignments]
    if len(set(ids)) != len(ids):
        errors.append('duplicate task assignments')
    if any(t not in tasks for t in ids):
        errors.append('unknown task assigned')
    missing = sorted(set(tasks) - set(ids))
    if missing != sorted(plan['unserved_task_ids']):
        errors.append('unserved task bookkeeping mismatch')
    if 'status' in plan and plan['status'] != ('partial' if missing else 'complete'):
        errors.append('completion status differs from task assignments')
    if 'unserved_reasons' in plan:
        reasons = plan['unserved_reasons']
        if sorted(item['task_id'] for item in reasons) != missing:
            errors.append('unserved reasons do not match missing tasks')
        if any(item['reason'] not in ('no_arm_feasible_candidate', 'no_navigable_feasible_candidate') for item in reasons):
            errors.append('unknown unserved reason')
    if set(plan['selected_candidate_ids']) != set(plan['visit_order_candidate_ids']):
        errors.append('visit order and selected set mismatch')
    if len(plan['selected_candidate_ids']) != len(set(plan['selected_candidate_ids'])):
        errors.append('candidate selected twice')
    if len(plan['visit_order_candidate_ids']) != len(set(plan['visit_order_candidate_ids'])):
        errors.append('candidate visited twice')
    unknown_docks = (set(plan['selected_candidate_ids'])
                     | set(plan['visit_order_candidate_ids'])) - set(docks)
    if unknown_docks:
        errors.append(f'unknown candidate IDs in selected set or visit order: {sorted(unknown_docks)}')
    for a in assignments:
        t, j, arm = a['task_id'], a['candidate_id'], a['arm']
        # Fail closed on invalid identifiers before dereferencing external records.
        if t not in tasks:
            continue  # The unknown task is reported above.
        if j not in docks:
            errors.append(f'task {t}: unknown candidate ID {j}')
            continue
        if arm not in ('L', 'R'):
            errors.append(f'task {t}: unknown arm {arm!r}; expected L or R')
            continue
        if j not in plan['selected_candidate_ids']:
            errors.append(f'task {t}: assigned dock is not selected')
        if t not in feasible:
            errors.append(f'task {t}: missing feasibility record')
            continue
        if j not in feasible[t]['left' if arm == 'L' else 'right']:
            errors.append(f'task {t}: F is zero')
        if not proxy_reachable(tasks[t], docks[j], arm, scene['robot']):
            errors.append(f'task {t}: independent proxy reachability failed')
    r = scene['robot']['base_radius'] + scene['robot']['clearance']
    inflated = [(t['xmin'] - r, t['ymin'] - r, t['xmax'] + r, t['ymax'] + r)
                for t in [*scene['tables'], *scene.get('obstacles', [])]]
    polyline = plan['route_polyline']
    length = 0.0
    cumulative_distances = [0.0]
    collision_count = 0
    point_collision_count = sum(any(rect[0] <= p[0] <= rect[2] and rect[1] <= p[1] <= rect[3]
                                    for rect in inflated) for p in polyline)
    if point_collision_count:
        errors.append(f'{point_collision_count} route points inside inflated obstacles')
    for a, b in zip(polyline, polyline[1:]):
        length += math.hypot(b[0] - a[0], b[1] - a[1])
        cumulative_distances.append(length)
        collision_count += sum(segment_hits_rectangle(a, b, rect) for rect in inflated)
    if collision_count:
        errors.append(f'{collision_count} segment-obstacle collisions')
    if not math.isclose(length, plan['route_length_m'], rel_tol=1e-8, abs_tol=1e-7):
        errors.append('polyline length differs from route metric')
    if not math.isclose(length, sum(plan['route_leg_costs_m']), rel_tol=1e-8, abs_tol=1e-7):
        errors.append('polyline length differs from declared navigation-leg costs')
    if 'objective_length_plus_stops_m' in plan:
        _finite_fields(plan, ('stop_penalty_equivalent_m', 'objective_length_plus_stops_m'))
        _require(plan['stop_penalty_equivalent_m'] >= 0, 'stop penalty must be nonnegative')
        expected = length + plan['stop_penalty_equivalent_m'] * len(plan['selected_candidate_ids'])
        if not math.isclose(expected, plan['objective_length_plus_stops_m'], rel_tol=1e-8, abs_tol=1e-7):
            errors.append('objective differs from route length plus stop penalties')
    if not polyline or math.dist(polyline[0], scene['start']) > 1e-8 or math.dist(polyline[-1], scene['start']) > 1e-8:
        errors.append('route is not closed at the specified start')
    for j in plan['selected_candidate_ids']:
        if j not in docks:
            continue
        point = (docks[j]['x'], docks[j]['y'])
        if not any(math.dist(point, p) < 1e-8 for p in polyline):
            errors.append(f'dock {j} is absent from the executable polyline')
    # Certify service events at the declared navigation-leg endpoints, rather
    # than accepting an incidental or repeated passage through a dock point.
    leg_costs = plan['route_leg_costs_m']
    if len(leg_costs) != len(plan['visit_order_candidate_ids']) + 1:
        errors.append('navigation leg count does not match visit order plus return to depot')
    if any(not math.isfinite(cost) or cost < 0 for cost in leg_costs):
        errors.append('navigation leg costs must be finite and nonnegative')
    else:
        endpoints = [((docks[j]['x'], docks[j]['y']), f'dock {j}')
                     if j in docks else (None, f'unknown dock {j}')
                     for j in plan['visit_order_candidate_ids']]
        endpoints.append((scene['start'], 'depot'))
        cursor, leg_end_distance = 0, 0.0
        for cost, (point, label) in zip(leg_costs, endpoints):
            leg_end_distance += cost
            position = next((k for k in range(cursor, len(polyline))
                             if math.isclose(cumulative_distances[k], leg_end_distance,
                                             rel_tol=1e-8, abs_tol=1e-7)), None)
            if position is None:
                errors.append(f'{label}: navigation leg endpoint is absent from route polyline')
                break
            if point is not None and math.dist(point, polyline[position]) > 1e-8:
                errors.append(f'{label}: declared visit order differs from navigation leg endpoint')
            cursor = position
    if scene.get('schema_version', 1) >= 2 and 'service_stops' not in plan:
        errors.append('schema v2 requires service_stops')
    if 'service_stops' in plan:
        stops = plan['service_stops']
        if [stop['candidate_id'] for stop in stops] != plan['visit_order_candidate_ids']:
            errors.append('service stops differ from visit order')
        arrival = 0.0
        for index, stop in enumerate(stops):
            j = stop['candidate_id']
            if j not in docks:
                errors.append(f'service stop uses unknown candidate {j}')
                continue
            arrival += leg_costs[index]
            _finite_fields(stop['pose'], ('x', 'y', 'theta', 'h'))
            _finite_fields(stop, ('arrival_distance_m',))
            if stop['sequence'] != index + 1:
                errors.append('service sequence numbers are not consecutive')
            if any(not math.isclose(stop['pose'][key], docks[j][key], abs_tol=1e-8, rel_tol=0)
                   for key in ('x', 'y', 'theta', 'h')):
                errors.append(f'dock {j}: service pose differs from candidate')
            if not math.isclose(stop['arrival_distance_m'], arrival, abs_tol=1e-7, rel_tol=1e-8):
                errors.append(f'dock {j}: service arrival distance is inconsistent')
            expected_tasks = sorted((a['task_id'], a['arm']) for a in assignments if a['candidate_id'] == j)
            actual_tasks = sorted((a['task_id'], a['arm']) for a in stop['tasks'])
            if actual_tasks != expected_tasks:
                errors.append(f'dock {j}: service tasks differ from assignments')
    return {'valid_model_plan': not errors, 'all_tasks_served': not missing,
            'collision_count': collision_count + point_collision_count,
            'point_collision_count': point_collision_count, 'unserved_task_ids': missing,
            'independent_polyline_length_m': length, 'errors': errors,
            'scope': 'base AABB-inflation collision and spherical-sector reachability proxy only; no IK'}


def validate_file(path):
    """Recompute checks, deliberately ignoring stored validation booleans."""
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        _require(isinstance(data['plans'], dict) and bool(data['plans']), 'no plans found')
        return [dict(file=path.name, method=method,
                     **validate(data['scene'], plan, data['feasibility']))
                for method, plan in data['plans'].items()]
    except (OSError, ValueError, KeyError, TypeError) as error:
        return [dict(file=path.name, method=None, valid_model_plan=False,
                     all_tasks_served=False, errors=[f'cannot validate file: {error}'])]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Independently validate saved scene plans; standard library only.')
    parser.add_argument('path', type=Path, help='scene_plan JSON or a directory of scene_plan_*.json files')
    parser.add_argument('--report', type=Path, help='optional JSON report destination')
    args = parser.parse_args(argv)
    files = sorted(args.path.glob('scene_plan_*.json')) if args.path.is_dir() else [args.path]
    if not files:
        parser.error('no scene_plan_*.json files found')
    checks = [check for path in files for check in validate_file(path)]
    report = dict(files=len(files), plans=len(checks),
                  all_valid=all(c['valid_model_plan'] for c in checks),
                  all_tasks_served=all(c['all_tasks_served'] for c in checks), checks=checks)
    if args.report:
        resolved_report = args.report.resolve()
        if any(resolved_report == path.resolve() for path in files):
            parser.error('--report must not overwrite an input scene plan')
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'files={len(files)} plans={len(checks)} all_valid={report["all_valid"]} '
          f'all_tasks_served={report["all_tasks_served"]}')
    for check in checks:
        if not check['valid_model_plan'] or not check['all_tasks_served']:
            print(f'{check["file"]} {check["method"]}: {check["errors"]}; '
                  f'unserved={check.get("unserved_task_ids")}')
    return 0 if report['all_valid'] and report['all_tasks_served'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
