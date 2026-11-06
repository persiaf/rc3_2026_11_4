"""Independent S0 audit of saved bytes only; never execute project kernels.

Allowed: JSON/NPZ/checkpoint tensor inspection, scalar proxy/geometry arithmetic,
saved predecessor-chain expansion and read-only Git source hash comparison.
Forbidden: scene generation, navigation preparation/search, H/planning, policy
instantiation/forward/backward, optimizer, or any original-60 content access.
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import time
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def close(a, b, tol=1e-7):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol


def reachable(task, candidate, arm, robot):
    dx, dy = task['x'] - candidate['x'], task['y'] - candidate['y']
    cosine, sine = math.cos(candidate['theta']), math.sin(candidate['theta'])
    x = cosine * dx + sine * dy - robot['arm_forward_offset']
    y = -sine * dx + cosine * dy - (1 if arm == 0 else -1) * robot['arm_spacing'] / 2
    z = task['z'] - robot['shoulder_height']
    radius = math.sqrt(x*x + y*y + z*z)
    return (robot['arm_min_reach'] <= radius <= robot['arm_max_reach'] and
            abs(math.atan2(y, x)) <= math.radians(robot['arm_sector_half_angle_deg']))


def intersects(a, b, box):
    # Independent interval intersection of a segment with a closed inflated AABB.
    lower, upper = 0., 1.
    for axis in (0, 1):
        direction = b[axis] - a[axis]
        if abs(direction) < 1e-12:
            if a[axis] < box[axis] or a[axis] > box[axis + 2]:
                return False
        else:
            t0, t1 = (box[axis] - a[axis]) / direction, (box[axis + 2] - a[axis]) / direction
            lower, upper = max(lower, min(t0, t1)), min(upper, max(t0, t1))
            if lower > upper:
                return False
    return True


def point(scene, node):
    return scene['start'] if node == 0 else [scene['candidates'][node-1][k] for k in ('x', 'y')]


def expand_saved(parent, nodes):
    """Follow existing predecessor indices only; no graph search or new nav."""
    arrays = parent['arrays']; points = []; lengths = []
    coords, service, predecessor = (arrays[k] for k in ('nav_coordinates', 'service_nodes', 'predecessors'))
    for source, target in zip(nodes, nodes[1:]):
        a, b = int(service[source]), int(service[target]); chain = [b]
        while chain[-1] != a:
            previous = int(predecessor[source, chain[-1]])
            require(0 <= previous < len(coords) and previous not in chain, 'invalid/cyclic saved predecessor chain')
            chain.append(previous)
        leg = coords[list(reversed(chain))].tolist()
        points.extend(leg if not points else leg[1:]); lengths.append(float(arrays['D'][source, target]))
    return points, lengths


def metric(route, view):
    require(len(route) >= 2 and route[0] == route[-1] == 0, 'symbolic endpoint route')
    stops = route[1:-1]; n, m = view['C'].shape
    require(all(type(x) is int and 1 <= x <= m for x in stops) and len(set(stops)) == len(stops), 'invalid/repeated local node')
    remaining = np.ones(n, dtype=bool)
    for node in stops:
        gain = remaining & view['C'][:, node-1]
        require(gain.any(), 'zero-gain service stop')
        remaining &= ~gain
    require(not remaining.any(), 'incomplete route')
    length = sum(float(view['D'][a, b]) for a, b in zip(route, route[1:]))
    return length + .75 * len(stops), length, len(stops)


def main():
    started = time.perf_counter(); phases = {}; checks = []; audit = dict(schema='S0-independent-saved-data-audit-v1')
    prior_runs = []
    if (OUT/'audit_saved.json').exists():
        previous = read(OUT/'audit_saved.json'); prior_runs = previous.get('previous_audit_runs', [])
        geometry_finished = any('63 final+63 reconstructed raw' in x for x in previous.get('completed_checks', []))
        prior_runs.append(dict(verdict=previous['verdict'], audit_elapsed_wall_s=previous['audit_elapsed_wall_s'],
            findings=previous.get('blocking_findings', []), phase_elapsed_s=previous.get('phase_elapsed_s', {}),
            final_export_geometric_acceptances=previous.get('independent_counts', {}).get('final_export_geometric_acceptances', 63 if geometry_finished else 0),
            raw_saved_path_geometric_acceptances=previous.get('independent_counts', {}).get('raw_saved_path_geometric_acceptances', 63 if geometry_finished else 0),
            note='Historical audit cost only; first implementation error was Counter(mapping) versus Counter(mapping.keys()), with no research evidence changed.' if previous['verdict'] == 'FAIL' and previous.get('blocking_findings', [{}])[0].get('message') == 'exact replay/cache key conservation' else 'Prior saved-data audit; no new research execution.'))
    audit['previous_audit_runs'] = prior_runs
    audit['new_research_calls'] = {key: 0 for key in ('scene_generations', 'navigation_preparations', 'planner_calls', 'H_calls', 'model_instantiations', 'model_forwards', 'backward_calls', 'optimizer_updates', 'original60_reads')}
    evidence = sorted(set([p for folder in ('parents', 'views', 'plans') for p in (OUT/folder).glob('*')] +
        [OUT/name for name in ('STATUS.json', 'S0_protocol.json', 'S2_S3_protocol.json', 'registration_receipt.json', 'run_started.json', 'seeds.json', 'seed_audit.json', 'unit_receipt.json', 'policy_unit_receipt.json', 'prereg_review.json', 'PREREG_REVIEW.md', 'execution_environment.json', 'episodes.jsonl', 'terminal_journal.jsonl', 'resources.json', 'gradient_probe.json', 'initial_policy.pt', 'REPORT_S0.md', 'claims.json')]))
    hashes = {str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in evidence}
    try:
        p = read(OUT/'S0_protocol.json'); future = read(OUT/'S2_S3_protocol.json'); registered = read(OUT/'registration_receipt.json')
        status = read(OUT/'STATUS.json'); resources = read(OUT/'resources.json'); stats = resources['stats']; seeds = read(OUT/'seeds.json')
        require(status['S0'] == 'complete_pass', 'S0 not complete')
        require(digest(OUT/'S0_protocol.json') == registered['S0_sha256'] == future['S0_protocol_sha256'], 'S0 protocol binding')
        require(digest(OUT/'S2_S3_protocol.json') == registered['future_sha256'], 'future protocol binding')
        for name, expected in p['source_sha256'].items():
            require(digest(ROOT/name) == expected, 'changed preregistered source: ' + name)
        for name, key in [('seeds.json', 'seeds_sha256'), ('seed_audit.json', 'seed_audit_sha256'), ('unit_receipt.json', 'unit_sha256'), ('policy_unit_receipt.json', 'policy_unit_sha256'), ('prereg_review.json', 'prereg_review_json_sha256'), ('PREREG_REVIEW.md', 'prereg_review_sha256'), ('execution_environment.json', 'environment_sha256')]:
            require(digest(OUT/name) == p[key], 'changed preregistered evidence: ' + name)
        for name, expected in p['H_source_sha256'].items():
            original = subprocess.check_output(['git', 'show', p['base'] + ':' + name], cwd=ROOT)
            require(hashlib.sha256(original).hexdigest() == expected == digest(ROOT/name), 'H source differs from frozen base')
        require(signature(p['H_source_sha256']) == p['H_sha256'], 'H identity')
        for name in ('S0_protocol.json', 'S2_S3_protocol.json'):
            original = subprocess.check_output(['git', 'show', '89af4e0:rc3_global_local/' + name], cwd=ROOT)
            require(hashlib.sha256(original).hexdigest() == digest(OUT/name), 'protocol absent/changed since pre-run commit')
        checks.append('protocols/source hashes match registration, pre-run89af4e0 and frozen-base H bytes')
        phases['source_binding_s'] = time.perf_counter()-started

        parents = {}; pair_checks = 0; f_checks = 0; nav_tick = time.perf_counter()
        for case, spec in enumerate(seeds['S0']):
            scene_path = OUT/'parents'/f'{case:02}_scene.json'; arrays_path = OUT/'parents'/f'{case:02}_arrays.npz'
            scene = read(scene_path); receipt = read(OUT/'parents'/f'{case:02}_receipt.json')
            with np.load(arrays_path, allow_pickle=False) as saved:
                arrays = {k: saved[k].copy() for k in saved.files}
            F, A, D = (arrays[k] for k in ('F', 'A', 'D')); n, m, _ = F.shape
            require(scene['nx'] == scene['ny'] == 10 and len(scene['tables']) == 100, 'parent size/100 tables')
            require(scene['seed'] == spec['map_seed'] and scene['task_seed'] == spec['task_seed'] and scene['complexity'] == spec['mode'], 'parent seed/mode')
            require(receipt['metadata'] == spec and digest(scene_path) == receipt['scene_sha256'] and digest(arrays_path) == receipt['arrays_sha256'], 'parent file receipt')
            require(F.dtype == A.dtype == bool and F.shape == A.shape == (len(scene['tasks']), len(scene['candidates']), 2), 'parent F/A shape')
            require(D.shape == (m+1, m+1) and np.array_equal(np.isfinite(D), np.isfinite(D.T)) and np.allclose(D[np.isfinite(D)], D.T[np.isfinite(D)], rtol=0, atol=1e-8), 'parent D symmetry')
            require(all([x['id'] for x in scene[k]] == list(range(len(scene[k]))) for k in ('tasks', 'tables', 'candidates')), 'original contiguous global IDs')
            for name in ('F', 'A', 'D'):
                require(array_hash(arrays[name]) == receipt[name+'_sha256'], 'parent array hash ' + name)
            identity = signature(dict(scene=scene, F=array_hash(F), A=array_hash(A), mode='proxy'))
            require(identity == receipt['identity'], 'parent identity')
            expected_A = np.ones_like(F)
            if spec['A_condition']:
                condition = spec['A_condition']; expected_A[:] = False
                for i in range(n):
                    for j in range(m):
                        for arm in (0, 1):
                            expected_A[i, j, arm] = ((i+2*j+arm) % (3 if condition == 1 else 4)) >= (1 if condition == 1 else 2)
                    feasible = np.argwhere(F[i])
                    if len(feasible):
                        expected_A[i, feasible[0, 0], feasible[0, 1]] = True
            require(np.array_equal(expected_A, A), 'synthetic same-arm allow definition')
            for i, task in enumerate(scene['tasks']):
                for j, candidate in enumerate(scene['candidates']):
                    for arm in (0, 1):
                        require(bool(F[i, j, arm]) == reachable(task, candidate, arm, scene['robot']), 'independent full proxy F mismatch')
                        f_checks += 1
            coords, service, predecessor = (arrays[k] for k in ('nav_coordinates', 'service_nodes', 'predecessors'))
            require(coords.shape == (receipt['navigation_nodes'], 2) and predecessor.shape == (m+1, len(coords)) and service.shape == (m+1,), 'saved roadmap shape')
            require(np.isfinite(coords).all() and all(math.dist(coords[int(service[node])], point(scene, node)) < 1e-8 for node in range(m+1)), 'saved service-node coordinates')
            # Validate all finite service-to-service D entries against their saved predecessor tree.
            # Memoization adds arithmetic only; it neither generates graph edges nor searches for routes.
            for source in range(m+1):
                distances = {int(service[source]): 0.}
                for target in range(m+1):
                    if not math.isfinite(D[source, target]):
                        require(int(predecessor[source, int(service[target])]) < 0, 'infinite D has saved predecessor')
                        continue
                    vertex = int(service[target]); chain = []; seen = set()
                    while vertex not in distances:
                        require(vertex not in seen, 'predecessor cycle'); seen.add(vertex)
                        previous = int(predecessor[source, vertex])
                        require(0 <= previous < len(coords), 'disconnected saved finite D')
                        chain.append((vertex, previous)); vertex = previous
                    for child, previous in reversed(chain):
                        distances[child] = distances[previous] + math.dist(coords[child], coords[previous])
                    require(close(distances[int(service[target])], D[source, target]), 'saved predecessor length differs from D')
                    pair_checks += 1
            radius = scene['robot']['base_radius'] + scene['robot']['clearance']
            boxes = [(b['xmin']-radius, b['ymin']-radius, b['xmax']+radius, b['ymax']+radius) for b in [*scene['tables'], *scene.get('obstacles', [])]]
            parents[case] = dict(scene=scene, arrays=arrays, identity=identity, boxes=boxes)
        phases['parent_scalar_and_saved_predecessor_s'] = time.perf_counter()-nav_tick
        checks.append('3 full100-table parents: full scalar proxy F/A, hashes, all finite D entries and saved predecessor-chain lengths')

        views = {}; boundary_total = 0
        for case, path in enumerate(sorted((OUT/'views').glob('*.json'))):
            desc = read(path); parent = parents[desc['parent_case']]; scene = parent['scene']; arrays = parent['arrays']; w = desc['window']
            require(case == desc['case'], 'view case ID')
            tables = [t['id'] for t in scene['tables'] if w['x0'] <= t['ix'] < w['x0']+w['width'] and w['y0'] <= t['iy'] < w['y0']+w['height']]
            tasks = [t['id'] for t in scene['tasks'] if t['table_id'] in tables]
            entry, exit = desc['entry_global_node'], desc['exit_global_node']
            if desc['position'] == 'corner':
                require(entry == exit == 0 and w['x0'] == w['y0'] == 0, 'closed corner endpoints')
            else:
                connected = [j for j in range(1, len(arrays['D'])) if math.isfinite(arrays['D'][0, j]) and math.isfinite(arrays['D'][j, 0])]
                require([entry, exit] == [connected[0], connected[-1]] and entry != exit, 'open center endpoints')
                require(w['x0'] == (10-w['width'])//2 and w['y0'] == (10-w['height'])//2, 'center window')
            effective = arrays['F'] & arrays['A']
            candidates = [j for j in range(effective.shape[1]) if effective[tasks, j, :].any() and math.isfinite(arrays['D'][entry, j+1]) and math.isfinite(arrays['D'][j+1, exit])]
            F = effective[np.ix_(tasks, candidates, [0, 1])]; C = F.any(2); nodes = np.asarray(candidates)+1
            D = np.zeros((len(nodes)+1, len(nodes)+1)); D[0, 0] = arrays['D'][entry, exit]
            D[0, 1:] = arrays['D'][entry, nodes]; D[1:, 0] = arrays['D'][nodes, exit]; D[1:, 1:] = arrays['D'][np.ix_(nodes, nodes)]
            boundary = [j for j in candidates if scene['candidates'][j]['table_id'] not in tables]; boundary_total += len(boundary)
            require(tasks == desc['task_ids_global'] and candidates == desc['candidate_ids_global'] and boundary == desc['boundary_candidate_ids_global'], 'exact independent tasks/all legal candidate/boundary extraction')
            require(len(tasks) == desc['task_count'] and len(candidates) == desc['candidate_count'] and C.any(1).all(), 'local count/full coverage')
            require(array_hash(F) == desc['effective_F_sha256'] and array_hash(C) == desc['coverage_sha256'] and array_hash(D) == desc['endpoint_D_sha256'], 'same-arm F&A, C and explicit endpoint D hashes')
            identity = signature(dict(parent=parent['identity'], tasks=tasks, candidates=candidates, entry=entry, exit=exit, window=w, penalty=.75))
            require(identity == desc['identity'] and parent['identity'] == desc['parent_identity'], 'view context identity')
            views[case] = dict(desc=desc, parent=parent, C=C, F=F, D=D, tasks=tasks, candidates=candidates, entry=entry, exit=exit)
        require(len(views) == 18, '18 windows')
        checks.append('18 exact task/candidate/effective F/C/endpoint-D slices with full parent geometry retained')

        rows = [json.loads(line) for line in (OUT/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        require(len(rows) == 63 and sum(not r['gradient_probe'] for r in rows) == 54 and sum(r['gradient_probe'] for r in rows) == 9, '54 ordinary plus9 gradient episodes')
        geometry_tick = time.perf_counter(); geometry_count = 0; raw_geometry_count = 0; zero_legs = 0; outside = 0; open_plans = 0; policy_events = 0; step_count = 0
        completes = {}; per_method = Counter()
        for episode, row in enumerate(rows):
            require(episode == row['episode'], 'episode contiguous IDs'); v = views[row['case']]; desc = v['desc']; parent = v['parent']; arrays = parent['arrays']; scene = parent['scene']
            raw_J, raw_L, raw_K = metric(row['raw_route'], v); J, L, K = metric(row['final_route'], v)
            require(close(row['raw_J'], raw_J) and close(row['J'], J) and close(row['L'], L) and row['K'] == K and J <= raw_J+1e-7, 'raw/H cost and H nonincrease')
            remaining = np.ones(len(v['tasks']), dtype=bool); current = 0; expected_rewards = []; rng = np.random.default_rng(row['seed']); logsum = 0.
            require(len(row['trace']) == raw_K == len(row['rewards']) and raw_K <= len(remaining), 'positive-gain <=N horizon')
            for k, (node, trace) in enumerate(zip(row['raw_route'][1:-1], row['trace'])):
                mask = v['C'][remaining].any(0) & np.isfinite(v['D'][current, 1:]) & np.isfinite(v['D'][1:, 0])
                require(mask[node-1], 'illegal saved action')
                served = np.flatnonzero(remaining & v['C'][:, node-1]); require(len(served)>0, 'nonpositive gain')
                require(trace['action_local_node'] == node and trace['action_global_candidate'] == v['candidates'][node-1], 'raw action mapping')
                require(trace['served_tasks_global'] == [v['tasks'][i] for i in served], 'raw ordered service')
                if row['method'] == 'random':
                    require(node == int(rng.choice(np.flatnonzero(mask)))+1, 'random exact saved-seed action replay')
                elif row['method'] == 'rule':
                    gain = v['C'][remaining].sum(0)
                    expected = min(np.flatnonzero(mask), key=lambda j: (-gain[j]/(v['D'][current, j+1]+.75), v['candidates'][j]))+1
                    require(node == expected, 'rule exact scalar replay')
                remaining[served] = False; done = not remaining.any(); correction = raw_J-J if done else 0.
                reward = -float(v['D'][current, node])-.75
                if done:
                    reward += correction-float(v['D'][node, 0])
                require(trace['remaining_tasks_global'] == [v['tasks'][i] for i in np.flatnonzero(remaining)] and trace['done'] == done, 'raw remaining/done replay')
                require(close(trace['reward'], reward) and close(trace['terminal_H_correction'], correction) and close(row['rewards'][k], reward), 'per-step reward/correction replay')
                if 'policy' in trace:
                    event = trace['policy']; policy_events += 1
                    require(row['method'] == 'neural' and event['legal_actions'] == int(mask.sum()) and event['illegal_probability_max'] == 0., 'logged masked distribution')
                    require(math.isfinite(event['chosen_logprob']) and event['chosen_logprob'] <= 1e-7 and math.isfinite(event['entropy']) and -1e-7 <= event['entropy'] <= math.log(mask.sum())+1e-6, 'finite logged policy statistics')
                    logsum += event['chosen_logprob']
                if row['gradient_probe'] and k == 0:
                    require(node == row['forced_first_local_node'] and 'policy' not in trace, 'forced first action excluded from logprob')
                expected_rewards.append(reward); current = node; step_count += 1
            require(not remaining.any() and close(sum(expected_rewards), -J) and close(row['return_'], -J), 'complete return=-JH')
            require(close(row['logprob_sum_excluding_forced'], logsum, 1e-3) if row['gradient_probe'] else row['logprob_sum_excluding_forced'] == 0., 'gradient logprob sum')
            if not row['gradient_probe']:
                per_method[row['method']] += 1
            plan = read(OUT/'plans'/f'{episode:03}_{row["method"]}.json')
            visited = [v['candidates'][node-1] for node in row['final_route'][1:-1]]; global_nodes = [v['entry'], *[j+1 for j in visited], v['exit']]
            require(plan['problem_identity'] == desc['identity'] and plan['parent_identity'] == parent['identity'] and plan['window'] == desc['window'], 'plan identity/window')
            require(plan['entry_global_node'] == v['entry'] and plan['exit_global_node'] == v['exit'] and plan['navigation_global_nodes'] == global_nodes, 'plan open/closed global node mapping')
            require(plan['visit_order_candidate_ids'] == plan['selected_candidate_ids'] == visited and plan['target_task_ids_global'] == v['tasks'], 'plan task/candidate mapping')
            left = set(v['tasks']); expected_assignments = []
            for j in visited:
                served = [i for i in sorted(left) if any(arrays['F'][i, j, arm] and arrays['A'][i, j, arm] for arm in (0, 1))]
                require(bool(served), 'final zero-gain service'); left.difference_update(served)
                for i in served:
                    arm = 'L' if arrays['F'][i, j, 0] and arrays['A'][i, j, 0] else 'R'
                    expected_assignments.append(dict(task_id=i, candidate_id=j, arm=arm))
            require(not left and plan['assignments'] == expected_assignments and plan['unserved_task_ids'] == [] and plan['status'] == 'complete', 'complete ordered per-arm assignment')
            expected_points, legs = expand_saved(parent, global_nodes)
            require(plan['route_polyline'] == expected_points and np.allclose(plan['route_leg_costs_m'], legs, rtol=0, atol=1e-8), 'exact saved predecessor export')
            require(math.dist(expected_points[0], point(scene, v['entry'])) <= 1e-8 and math.dist(expected_points[-1], point(scene, v['exit'])) <= 1e-8, 'requested polyline endpoints')
            arcs = [0.]
            for a, b in zip(expected_points, expected_points[1:]):
                require(all(math.isfinite(x) for x in [*a, *b]), 'finite polyline'); arcs.append(arcs[-1]+math.dist(a, b))
                require(not any(intersects(a, b, box) for box in parent['boxes']), 'polyline intersects any full-parent inflated obstacle')
            require(close(arcs[-1], sum(legs)) and close(arcs[-1], L) and close(plan['route_length_m'], L) and close(plan['objective_length_plus_stops_m'], J) and plan['stop_penalty_equivalent_m'] == .75, 'independent geometry length/objective')
            require(len(plan['service_stops']) == len(visited), 'stop count'); arrival = 0.
            for k, (j, stop) in enumerate(zip(visited, plan['service_stops'])):
                arrival += legs[k]
                require(stop['sequence'] == k+1 and stop['candidate_id'] == j and all(close(stop['pose'][key], scene['candidates'][j][key]) for key in ('x', 'y', 'theta', 'h')), 'docking pose and order')
                require(close(stop['arrival_distance_m'], arrival) and any(close(s, arrival) and math.dist(q, point(scene, j+1)) <= 1e-8 for s, q in zip(arcs, expected_points)), 'polyline ordered stop-coordinate arrival')
                require(stop['tasks'] == [dict(task_id=a['task_id'], arm=a['arm']) for a in expected_assignments if a['candidate_id'] == j], 'stop-level service mapping')
            require(plan['arm_joint_trajectories'] is None and plan['physical_joint_trajectory_certified'] is False, 'physical joint trajectory not certified')
            require(plan['validation']['valid_model_plan'] and plan['validation']['endpoint_constraints_satisfied'] and not plan['validation']['physical_joint_trajectory_certified'], 'saved validator scope')
            w = desc['window']; x0, x1 = scene['x_lines'][w['x0']], scene['x_lines'][w['x0']+w['width']]; y0, y1 = scene['y_lines'][w['y0']], scene['y_lines'][w['y0']+w['height']]
            outside += int(any(not (x0-1e-8 <= q[0] <= x1+1e-8 and y0-1e-8 <= q[1] <= y1+1e-8) for q in expected_points))
            open_plans += int(v['entry'] != v['exit']); zero_legs += sum(close(x, 0.) for x in legs); geometry_count += 1
            # Also validate raw paths reconstructed from saved predecessors: raw routes are
            # not exported artifacts, and incur no new terminal/H/navigation invocation.
            raw_nodes = [v['entry'], *[v['candidates'][node-1]+1 for node in row['raw_route'][1:-1]], v['exit']]
            raw_points, raw_legs = expand_saved(parent, raw_nodes)
            require(close(sum(math.dist(a, b) for a, b in zip(raw_points, raw_points[1:])), raw_L), 'raw saved geometry length')
            require(not any(intersects(a, b, box) for a, b in zip(raw_points, raw_points[1:]) for box in parent['boxes']), 'raw path full-parent geometry')
            raw_geometry_count += 1; completes[(desc['identity'], tuple(row['raw_route']))] = row
        require(per_method == Counter(random=18, rule=18, neural=18), '18 ordinary episodes per untrained control')
        phases['saved_raw_and_final_geometry_and_scalar_replay_s'] = time.perf_counter()-geometry_tick
        checks.append('63 exact saved action/reward replays; positive-gain legal masks <=N; raw/H costs, complete mappings,63 final+63 reconstructed raw full-parent geometric checks')

        journal = [json.loads(line) for line in (OUT/'terminal_journal.jsonl').read_text(encoding='utf-8').splitlines()]
        counts = Counter(r['event'] for r in journal); pending = {}; cached = set(); starts = []; hits = []
        for event in journal:
            kind = event['event']
            if kind == 'terminal_start':
                key = (event['problem'], tuple(event['raw_route'])); require(key not in cached and event['call'] not in pending, 'duplicate real terminal')
                pending[event['call']] = key; starts.append(key)
            elif kind == 'terminal_complete':
                key = pending.pop(event['call']); result = event['result']; row = completes[key]
                require(key[0] == event['problem'] and key[1] == tuple(result['raw_route']), 'terminal journal key')
                require(result['route'] == row['final_route'] and all(close(result[k], row[k]) for k in ('J', 'L', 'K', 'raw_J')), 'terminal result conservation')
                require(result['cache_hit'] is False and all(math.isfinite(result[k]) and result[k] >= 0 for k in ('H_s', 'export_s', 'validation_s')), 'terminal finite timings')
                cached.add(key)
            elif kind == 'terminal_cache_hit':
                key = (event['problem'], tuple(event['raw_route'])); require(key in cached, 'cache hit before completion/wrong problem or ordered route'); hits.append(key)
            else:
                raise AssertionError('failed/unknown terminal event ' + kind)
        require(not pending and len(cached) == 63 and Counter(starts) == Counter(hits) == Counter(completes.keys()), 'exact replay/cache key conservation')
        expected_stats = dict(parent_generation_attempts=3, parent_scene_generation=3, navigation_preparation_attempts=3, navigation_preparations=3, cropextractions=18, ordinary_episodes=54, gradient_episodes=9, episode_steps=step_count, replay_episodes=63, replay_steps=step_count, terminal_requests=126, terminal_cache_hits=63, terminal_attempts=63, terminal_completed=63, terminal_failed=0, full_H_calls=63, export_attempts=63, exports=63, validation_attempts=63, validation_completed=63, neural_encode_attempts=27, neural_encodes=27, neural_score_attempts=policy_events, neural_score_calls=policy_events, backward_calls=1, backward_completed=1, optimizer_updates=0, teacher_queries=0, S1_calls=0, S2_calls=0, S3_calls=0, original60_reads=0)
        require(stats == expected_stats and counts == Counter(terminal_start=63, terminal_complete=63, terminal_cache_hit=63), 'resources/journal/episode counts')
        require(stats['terminal_attempts'] <= p['limits']['full_terminal_attempts'] == 80 and resources['elapsed_wall_s'] < p['limits']['wall_s'] == 1800, 'registered budgets')
        require(resources['internal_search']['counts'] == dict(construct=0, two_opt=252, joint_search=63, recipe=0), 'frozen H internal accounting')
        for stage in ('S1', 'S2', 'S3'):
            require(status[stage] == 'not_started', 'future stage inadvertently started')
        require(status['optimizer_updates'] == 0 and status['original60'] == 'sealed' and status['pushed'] is False, 'run boundary')
        checks.append('126 requests=63 real terminals+63 exact replay hits; all real budgets/counters conserve, zero optimizer/S1/S2/S3/original60')

        g = read(OUT/'gradient_probe.json'); loss_terms = []
        for group in g['groups']:
            group_rows = [r for r in rows if r['case'] == group['case'] and r['gradient_probe']]
            returns = [r['return_'] for r in group_rows]; baseline = sum(returns)/3; advantages = [x-baseline for x in returns]
            require(len(group_rows) == 3 and group['forced_starts'] == [r['forced_first_local_node'] for r in group_rows], 'three declared forced gradient trajectories')
            require(all(close(a, b) for a, b in zip(returns, group['returns'])) and close(baseline, group['baseline']) and all(close(a, b) for a, b in zip(advantages, group['advantages'])), 'saved gradient return/baseline/advantage')
            loss_terms.extend(-a*r['logprob_sum_excluding_forced'] for a, r in zip(advantages, group_rows))
        require(close(sum(loss_terms)/9, g['loss'], 1e-3) and g['all_finite'] and g['nonzero'] and g['forced_first_logprob_excluded'], 'finite gradient-probe loss reconstruction')
        require(len(g['parameter_gradient_norms']) == 25 and all(math.isfinite(x) and x > 0 for x in g['parameter_gradient_norms'].values()), 'finite nonzero all25 parameter gradients')
        require(g['optimizer_updates'] == 0 and len({g['tensor_hash_before'], g['tensor_hash_after'], resources['model_hash_before'], resources['model_hash_now']}) == 1, 'unchanged recorded parameter hashes')
        # Read checkpoint tensors without creating a model or executing any inference.
        import torch
        checkpoint = torch.load(OUT/'initial_policy.pt', map_location='cpu', weights_only=True)
        h = hashlib.sha256()
        for name, tensor in sorted(checkpoint['state'].items()):
            h.update(name.encode()); h.update(tensor.detach().numpy().tobytes())
        require(h.hexdigest() == checkpoint['tensor_sha256'] == g['tensor_hash_before'] and checkpoint['optimizer_updates'] == 0 and checkpoint['S0_protocol_sha256'] == digest(OUT/'S0_protocol.json'), 'initial checkpoint tensor hash/protocol')
        core = read(OUT/'unit_receipt.json'); policy = read(OUT/'policy_unit_receipt.json')
        require(core['tests_passed'] == 38 and core['failures'] == core['errors'] == 0 and policy['tests'] == 13 and policy['passed'] and policy['optimizer_updates'] == 0, 'separate synthetic contracts')
        require(resources['synthetic_pre_registration_units'] == dict(environment=core, policy=policy), 'synthetic histories preserved separately')
        checks.append('saved9-trajectory loss,25 finite/nonzero gradient norms, checkpoint tensor hash, no parameter update; synthetic38/13 and all historical costs separate')

        require(future['status'] == 'FROZEN_DESIGN_NOT_STARTED' and future['S3']['status'] == 'NOT_STARTED', 'future design status')
        train = seeds['S2_train']; s3 = seeds['S3']; groups = defaultdict(list)
        require(len(train) == 96 and len({r['map_seed'] for r in train}) == 96 and all(len(r['task_seeds']) == 8 for r in train), '96 training parents8 task draws')
        for row in s3:
            groups[row['group']].append(row)
        training_layouts = {r['map_seed'] for r in train}
        for name, parent_count in [('seen_layout_new_tasks', 6), ('unseen_layout_new_tasks', 12)]:
            cases = groups[name]; layout_counts = Counter(r['map_seed'] for r in cases)
            require(len(layout_counts) == parent_count and len(cases) == 6*parent_count and set(layout_counts.values()) == {6}, 'parent-cluster S3 counts')
            require(all((r['map_seed'] in training_layouts) == name.startswith('seen') for r in cases), 'seen/unseen layout separation')
            for layout in layout_counts:
                require({(tuple(r['size']), r['position']) for r in cases if r['map_seed'] == layout} == {(size, position) for size in ((4, 3), (6, 6), (8, 8)) for position in ('corner', 'center')}, 'six children per parent')
        require(len(s3) == 108 and len({r['map_seed'] for r in s3}) == 18 and all(r['A_condition'] == r['case'] % 3 for r in s3), '108 children18 independent parent layouts')
        train_tasks = {task for r in train for task in r['task_seeds']}
        require(not train_tasks.intersection(r['task_seed'] for r in s3), 'new S3 task draws')
        declared_seeds = {r[key] for bank in ('S0', 'S1_teachers', 'S2_development', 'S3') for r in seeds[bank] for key in ('map_seed', 'task_seed')}
        declared_seeds.update(training_layouts); declared_seeds.update(train_tasks)
        protected_low, protected_high = seeds['original60_protected']
        require(not any(protected_low <= seed <= protected_high for seed in declared_seeds), 'declared metadata overlaps sealed60 seed range')
        require('REINFORCE' in future['S2']['objective'] and 'one fully policy-sampled including first action' in future['S2']['schedule'] and 'leave-one-out' in future['S2']['loss'] and all(x in future['S2']['controls'] for x in ('frozen supervised', 'untrained', 'continued-SL')), 'genuine future policy-gradient core and SL controls')
        require('separate execution task' in future['mandatory_pre_optimizer_binding'] and 'runner' in future['mandatory_pre_optimizer_binding'], 'future runner/baseline binding still required')
        claims = read(OUT/'claims.json'); require(claims == dict(S0_consistency_passed=True, DRL_research_goal_fixed=True, trained_DRL_performance_supported=False, independent_RL_gain_supported=False, S1_started=False, S2_started=False, S3_started=False, physical_joint_trajectory_certified=False), 'report claims bounded to C1')
        report = (OUT/'REPORT_S0.md').read_text(encoding='utf-8')
        require(all(x in report for x in ('监督只作初始化/对照', '不是将地图物理切小', '未借旧闭环验证器冒充认证', '不能以本轮梯度流检查声称强化学习已有独立增量', '108子场景冒充108独立环境', 'physical_joint_trajectory_certified=false')), 'report research/physical/statistical boundaries')
        checks.append('future genuine S2 REINFORCE/auxiliary SL controls and unexecuted runner binding; v1 metadata declares6seen+12unseen parent seed labels,108 children; actual geometric independence is NOT established by seed labels; C2 not claimed')
        require(all(digest(ROOT/name) == expected for name, expected in hashes.items()), 'audit changed source evidence bytes')
        audit.update(verdict='PASS_S0_C1_ONLY', blocking_findings=[], completed_checks=checks,
            independent_counts=dict(parent_scenes=3, tables_per_parent=100, full_proxy_F_scalar_cells=f_checks, finite_service_D_pairs_checked_against_saved_predecessors=pair_checks, views=18, nonempty_boundary_views=sum(bool(v['desc']['boundary_candidate_ids_global']) for v in views.values()), boundary_candidate_total=boundary_total, episodes=63, ordinary_by_method=dict(per_method), episode_steps=step_count, final_export_geometric_acceptances=geometry_count, raw_saved_path_geometric_acceptances=raw_geometry_count, open_final_plans=open_plans, closed_final_plans=geometry_count-open_plans, final_zero_cost_navigation_legs=zero_legs, final_paths_leaving_task_window=outside, saved_policy_statistics_checked=policy_events),
            actual_research_stats=stats, actual_research_wall_s=resources['elapsed_wall_s'], saved_terminal_journal_counts=dict(counts),
            synthetic_accounting=dict(environment_tests=38, environment_historical_runs=core['run_history'], current_policy_tests=13, all_policy_unit_runs_model_operations=policy['all_unit_runs_model_operations'], all_policy_unit_runs_elapsed_s=policy['all_unit_runs_elapsed_s']),
            checkpoint_tensor_sha256=h.hexdigest(), future_declared_parent_seed_counts=dict(seen=6, unseen=12, total=18, child_cases=108), future_geometric_independence_verified=False,
            claim_verdict=dict(C1_environment_consistency='SUPPORTED_WITH_PROXY_AND_SAVED_DATA_SCOPE', C2_DRL_performance='NOT_ESTABLISHED', physical_joint_trajectory='NOT_CERTIFIED'),
            limitations=['All actual18 views have empty boundary-candidate sets; retention of a nonempty cross-window candidate is covered by synthetic adversarial contract only.', 'Distribution/logit numerics and gradients are inspected from immutable recorded statistics and registered source/contracts; no model inference/backward is repeated.', 'All finite saved D entries agree with saved predecessor-chain lengths; no independent shortest-path search or new navigation is performed.', 'Original replay trace is not separately persisted: this audit rebuilds actions/rewards from parent arrays and checks exact replay cache keys plus registered runner assertions.', 'Real episodes contain tasks; empty-task/open direct-exit corner cases are evidenced by the38 synthetic environment contracts.', 'Complete polylines certify circular-base navigation and proxy/table service records only; no IK, arm swept path or true joint-trajectory certification.', 'S0 untrained/random/rule scores and gradient flow do not demonstrate learned policy quality, RL increment, or speed.'],
            phase_elapsed_s=phases, evidence_sha256=hashes, source_sha256=p['source_sha256'], audit_script_sha256=digest(__file__))
    except Exception as exc:
        audit.update(verdict='FAIL', blocking_findings=[dict(type=type(exc).__name__, message=str(exc))], completed_checks=checks, phase_elapsed_s=phases, evidence_sha256=hashes)
        raise
    finally:
        audit['audit_elapsed_wall_s'] = time.perf_counter()-started
        audit['all_audit_runs_elapsed_wall_s'] = audit['audit_elapsed_wall_s'] + sum(r['audit_elapsed_wall_s'] for r in prior_runs)
        audit['all_audit_runs_geometric_acceptances'] = sum(r['final_export_geometric_acceptances'] + r['raw_saved_path_geometric_acceptances'] for r in prior_runs) + audit.get('independent_counts', {}).get('final_export_geometric_acceptances', 0) + audit.get('independent_counts', {}).get('raw_saved_path_geometric_acceptances', 0)
        (OUT/'audit_saved.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: audit[k] for k in ('verdict', 'independent_counts', 'audit_elapsed_wall_s', 'new_research_calls')}, ensure_ascii=False))


def review_future_only():
    """Static v2 amendment review; retain existing geometry audit without reruns."""
    tick = time.perf_counter(); audit = read(OUT/'audit_saved.json')
    require(audit['verdict'] == 'PASS_S0_C1_ONLY', 'saved S0 audit not passed')
    p0 = read(OUT/'S0_protocol.json'); v1 = read(OUT/'S2_S3_protocol.json'); old_seeds = read(OUT/'seeds.json')
    pointer = read(OUT/'FUTURE_DESIGN_STATUS.json'); v2 = read(OUT/'S2_S3_protocol_v2.json'); seeds = read(OUT/'future_seeds_v2.json')
    require(pointer['active_protocol'] == 'S2_S3_protocol_v2.json' and pointer['active_seed_registry'] == 'future_seeds_v2.json', 'active v2 pointer')
    require(pointer['active_protocol_sha256'] == digest(OUT/pointer['active_protocol']) and pointer['active_seed_registry_sha256'] == digest(OUT/pointer['active_seed_registry']) == v2['seeds_sha256'], 'v2 pointer/protocol/registry hashes')
    require(pointer['status'] == v2['status'] == 'FROZEN_DESIGN_NOT_STARTED' and pointer['formal_optimizer_updates'] == pointer['S3_task_queries'] == 0, 'v2 unexecuted boundary')
    require(pointer['superseded_design_retained'] == 'S2_S3_protocol.json' and pointer['superseded_design_sha256'] == digest(OUT/'S2_S3_protocol.json') == v2['amendment']['original_v1_sha256'], 'v1 retained bytes')
    require(pointer['superseded_design_formal_execution_allowed'] is False and pointer['S0_registration_and_results_unchanged'] is True, 'v1 execution eligibility retired')
    require(v2['S0_protocol_sha256'] == v2['amendment']['retained_S0_protocol_sha256'] == digest(OUT/'S0_protocol.json'), 'S0 unchanged binding')
    for name, expected in p0['source_sha256'].items():
        require(digest(ROOT/name) == expected, 'registered research source changed during design amendment')
    # Bind actual frozen scientific evidence; report prose may be amended to state
    # the active v2 design and retains its earlier reviewed hash in evidence_sha256.
    for name, expected in audit['evidence_sha256'].items():
        if Path(name).name not in ('REPORT_S0.md', 'claims.json'):
            require(digest(ROOT/name) == expected, 'S0 evidence changed during v2 amendment: ' + name)
    for name in ('S0_protocol.json', 'S2_S3_protocol.json', 'seeds.json', 'registration_receipt.json'):
        original = subprocess.check_output(['git', 'show', '89af4e0:rc3_global_local/' + name], cwd=ROOT)
        require(hashlib.sha256(original).hexdigest() == digest(OUT/name), 'original preregistered bytes changed')
    require(v2['fixed_research_goal'] == v1['fixed_research_goal'] and v2['physical_joint_trajectory'] == v1['physical_joint_trajectory'] and v2['protected60'] == v1['protected60'], 'research/physical/test boundary preserved')
    gate = v2['layout_independence_gate']
    require(gate['identity_function'] == 'rc3_fast.planner.map_key', 'canonical geometry identity')
    for name, expected in gate['source_sha256'].items():
        require(digest(ROOT/name) == expected, 'geometry generator/key source binding')
    planner = (ROOT/'rc3_fast/planner.py').read_text(encoding='utf-8')
    scenes = (ROOT/'rc3_fast/scenes.py').read_text(encoding='utf-8')
    baseline = (ROOT/'rc3_baseline/baseline.py').read_text(encoding='utf-8')
    require("if mode == 'regular':" in scenes and 'scene = make_scene(nx, ny, seed)' in scenes and "('tables', 'candidates', 'start', 'x_lines', 'y_lines', 'robot')" in planner, 'static regular-layout aliasing source context')
    require("rng = np.random.default_rng(seed)" in baseline and "cx, cy = ix * sx" in baseline and "payload['obstacles'] = scene.get('obstacles', [])" in planner, 'fixed regular geometry/canonical obstacles context')
    text = gate['mandatory_before_S1_or_optimizer']
    require(all(token in text for token in ('96 unique train', '3 unique development', '6 seen keys equal', '12 unique unseen', 'disjoint train/development', 'Stop on collision', 'no silent mode/seed replacement')), 'required actual geometry uniqueness/disjointness stop gate')
    require('BEFORE any S1' in v2['mandatory_pre_optimizer_binding'] and 'Do not generate/evaluate S3 target task draws' in gate['test_tasks'] and 'not verified independent geometries' in gate['statistics'], 'pre-training gate and no S3 task query')
    train, dev, s3, teachers = (seeds[k] for k in ('S2_train', 'S2_development', 'S3', 'S1_teachers'))
    require(len(train) == 96 and len(dev) == 18 and len(s3) == 108 and len(teachers) == 256, 'all future bank cardinalities')
    train_by_parent = {r['parent']: r for r in train}
    for bank in ('S2_train', 'S2_development', 'S3'):
        require(len(seeds[bank]) == len(old_seeds[bank]), 'v2 bank cardinality changed')
        for new, old in zip(seeds[bank], old_seeds[bank]):
            require({k: v for k, v in new.items() if k != 'mode'} == {k: v for k, v in old.items() if k != 'mode'}, 'non-mode future seed specification changed')
            parent_index = new['parent'] if bank != 'S2_development' else (new['map_seed']-14000000000)
            require(new['mode'] == ('mixed', 'blocked')[parent_index % 2], 'all future parent modes alternate mixed/blocked')
    require(Counter(r['mode'] for r in train) == Counter(mixed=48, blocked=48), '48/48 training modes')
    require(teachers == old_seeds['S1_teachers'], 'bounded teacher specs preserved')
    for row in teachers:
        parent = train_by_parent[row['parent']]
        require(row['map_seed'] == parent['map_seed'] and row['task_seed'] in parent['task_seeds'], 'teacher references updated training bank')
        draw = parent['task_seeds'].index(row['task_seed'])
        require(row['A_condition'] == (row['parent']+draw) % 3, 'teacher A-condition consistent with future task draw')
    groups = defaultdict(set)
    for row in s3:
        groups[row['group']].add((row['map_seed'], row['mode']))
        require(row['A_condition'] == row['case'] % 3, 'S3 declared A-condition unchanged')
        if row['group'] == 'seen_layout_new_tasks':
            parent = train_by_parent[row['parent']]
            require((row['map_seed'], row['mode']) == (parent['map_seed'], parent['mode']), 'seen parent map+mode equal intended training parent')
    require(len(groups['seen_layout_new_tasks']) == 6 and len(groups['unseen_layout_new_tasks']) == 12 and len({(r['map_seed'], r['mode']) for r in dev}) == 3, 'intended6seen12unseen3dev parent labels')
    for key in ('policy_seeds', 'sampling_seed', 'original60_protected'):
        require(seeds[key] == old_seeds[key], 'future RNG/protected boundary unchanged')
    require(seeds['provenance']['original_sha256'] == digest(OUT/'seeds.json') and seeds['provenance']['S0_not_changed'] and seeds['provenance']['new_scenes_generated'] == 0, 'v2 registry provenance')
    old_s1 = dict(v1['S1']); new_s1 = dict(v2['S1']); old_s1.pop('instances'); new_s1.pop('instances')
    require(old_s1 == new_s1 and 'future_seeds_v2.json' in v2['S1']['instances'], 'S1 budget/method intact with new registry')
    new_s2 = dict(v2['S2']); new_s2.pop('parent_modes')
    require(new_s2 == v1['S2'], 'all S2 methods/optimizer/budgets/performance controls unchanged')
    old_s3 = dict(v1['S3']); new_s3 = dict(v2['S3']); old_s3.pop('allow_mask'); new_s3.pop('allow_mask'); new_s3.pop('parent_modes')
    require(old_s3 == new_s3 and v2['S3']['allow_mask'] == v1['S3']['allow_mask'].replace('seeds.json', 'future_seeds_v2.json'), 'all S3 gates/budgets/inference/statistics preserved')
    receipt = dict(verdict='PASS_FROZEN_V2_DESIGN_ONLY', active_protocol=pointer['active_protocol'],
        reason='V1 regular seed aliases geometry; superseded without changing registered S0. V2 uses varying modes and mandates actual canonical-key uniqueness/disjointness before S1/optimizer.',
        actual_geometry_gate_executed=False, intended_parent_labels=dict(train=96, development=3, seen=6, unseen=12),
        generator_and_key_source_sha256=gate['source_sha256'], evidence_sha256={name: digest(OUT/name) for name in ('FUTURE_DESIGN_STATUS.json', 'S2_S3_protocol_v2.json', 'future_seeds_v2.json')},
        optimizer_updates=0, S3_task_queries=0, new_scene_navigation_H_model_forward_backward_calls=0, additional_saved_geometry_acceptances=0,
        static_review_elapsed_s=time.perf_counter()-tick, audit_script_sha256=digest(__file__))
    audit['future_design_static_review'] = receipt
    audit['future_geometric_independence_verified'] = False
    audit['future_declared_parent_seed_counts'] = dict(seen=6, unseen=12, total=18, child_cases=108)
    audit.pop('future_independent_parent_counts', None)
    audit['completed_checks'][-1] = 'future v2 static metadata/source/budget review passed; 6seen+12unseen are intended labels pending mandatory canonical geometry gate; v1 regular aliasing superseded; C2 not claimed'
    audit['current_audit_script_sha256'] = digest(__file__)
    audit['report_prose_at_future_review_sha256'] = digest(OUT/'REPORT_S0.md')
    (OUT/'audit_saved.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    if sys.argv[1:] == ['--future-only']:
        review_future_only()
    else:
        main()
        if (OUT/'FUTURE_DESIGN_STATUS.json').exists():
            review_future_only()
