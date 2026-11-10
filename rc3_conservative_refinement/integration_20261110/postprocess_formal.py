"""Bounded, saved-evidence-only closeout; never imports a planner or Torch.

Register this helper's exact hash in a separate ANALYSIS_PLAN.json after the
formal run closes.  Analysis wall time is charged to the SAME formal ledger.
No request/optimizer budgets are reset and no new route is constructed.
"""
from __future__ import annotations

import time
ENTRY_START = time.perf_counter()
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from rc3_conservative_refinement.storage import Ledger, canonical, identity, read, write_new

DEFAULT_RUN = HERE / 'runs/conservative_paired_11622026_v1'
_HASH_CHECK = lambda: None


def digest(path):
    """Stream hashes under the active analysis clock, including large records."""
    result = hashlib.sha256()
    _HASH_CHECK()
    with Path(path).open('rb') as stream:
        while True:
            _HASH_CHECK()
            chunk = stream.read(1 << 20)
            if not chunk: break
            result.update(chunk)
            _HASH_CHECK()
    _HASH_CHECK()
    return result.hexdigest()


def need(value, message):
    if not value:
        raise RuntimeError(message)


def project_path(value):
    path = (ROOT / value).resolve()
    need(path.is_relative_to(ROOT), 'Analysis path escaped the project')
    return path


class CSVSink:
    """Bounded-memory table output, checking the clock on every saved row."""
    def __init__(self, path, fields, check):
        self.path, self.fields, self.check = path, fields, check
    def __enter__(self):
        self.check()
        self.stream = Path(self.path).open('x', encoding='utf-8-sig', newline='')
        self.writer = csv.DictWriter(self.stream, fieldnames=self.fields)
        self.writer.writeheader()
        self.check()
        return self
    def write(self, row):
        self.check()
        self.writer.writerow(row)
        self.check()
    def __exit__(self, *args):
        self.stream.close()
        if args[0] is None:
            self.check()


def checked_csv(path, rows, check):
    check()
    rows = list(rows)  # Only small aggregate tables use this helper.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with CSVSink(path, fields, check) as output:
        for row in rows:
            output.write(row)
    check()


def checked_json(path, value, check):
    check()
    write_new(path, value)
    check()


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), abs_tol=1e-7, rel_tol=1e-11)


def iter_events(path):
    with Path(path).open('rb') as stream:
        for raw in stream:
            yield json.loads(raw)


def chain(path, check=lambda: None):
    """Retain only small journal rows; PPO per-state event payloads are streamed."""
    events, tail, sequence = [], '0' * 64, 0
    heavy = {'ppo_opportunity', 'matched_supervised_opportunity', 'critic_warm_opportunity'}
    for event in iter_events(path):
            check()
            body = {key: value for key, value in event.items() if key != 'sha256'}
            need(event['previous'] == tail and event['sequence'] == sequence
                 and identity(body) == event['sha256'], 'Canonical ledger hash/sequence failed')
            if event['event'] not in heavy:
                events.append(event)
            tail = event['sha256']
            sequence += 1
    need(events and events[0]['event'] == 'registration', 'Missing initial ledger registration')
    return events, tail, sequence


class _SequenceCounter:
    def __init__(self, count): self.count = count
    def __len__(self): return self.count
    def append(self, value): self.count += 1


def admit_one_shot(events):
    need(not any(e['event'] == 'analysis_start' or
                 (e['event'] == 'session_start' and str(e.get('purpose','')).startswith('terminal_analysis_'))
                 for e in events),
         'This one-shot analysis was already attempted; changing output/plan cannot reset its budget or failure')


class StreamingAnalysisLedger(Ledger):
    """Analysis-only session on the ORIGINAL journal, avoiding its heavy reload.

    The exact canonical prefix, sequence, tail, registration, cumulative counts
    and closed sessions are reverified while holding the original WRITER.lock.
    emit/check/close are inherited unchanged.  Training admission is disabled.
    The extra analysis events deliberately make formal training restore fail its
    clean-boundary rule; analysis is admitted only after terminal closure.
    """
    def __init__(self, folder, binding, caps, expected_tail, expected_total, precheck):
        self.folder, self.binding, self.caps = Path(folder), dict(binding), dict(caps)
        self.path = self.folder / 'events.jsonl'
        self.lock = self.folder / 'WRITER.lock'
        self.fd = None
        try:
            self.fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self.fd, canonical(dict(pid=os.getpid(), binding=self.binding, purpose='terminal_analysis_only')))
            events, tail, total = chain(self.path, precheck)
            need(tail == expected_tail and total == expected_total, 'Journal changed between precheck and exclusive analysis lock')
            need(events[0]['binding'] == self.binding and events[0]['caps'] == self.caps,
                 'Original journal binding/caps differ')
            admit_one_shot(events)
            starts = {e['sequence'] for e in events if e['event'] == 'session_start'}
            closed = {e['open_sequence'] for e in events if e['event'] == 'session_close'}
            need(starts == closed, 'Original journal has an unclosed session')
            self.counts = Counter()
            for event in events:
                if event['event'] == 'debit': self.counts.update(event['amounts'])
            need(all(value <= self.caps[key] for key, value in self.counts.items()), 'Original cumulative resource cap exceeded')
            self.wall_charged = math.fsum(e['elapsed_s'] for e in events if e['event'] == 'session_close')
            self.events, self.tail = _SequenceCounter(total), tail
            self.started = ENTRY_START
            self.emit(dict(event='session_start', carry_wall_s=self.wall_charged,
                           pre_registration_elapsed_s=time.perf_counter() - ENTRY_START,
                           purpose='terminal_analysis_only_no_training_restore'))
            self.open_sequence = total
        except BaseException:
            self.release()
            raise
    def debit(self, *args, **kwargs): raise RuntimeError('Analysis cannot debit training/request resources')
    def request(self, *args, **kwargs): raise RuntimeError('Analysis cannot execute requests')


def validate_saved_plan(plan, metrics, raw=False):
    """Arithmetic/manifest audit of SAVED geometry and its existing certificate.

    This does not rerun obstacle collision, per-arm F&A or real IK checks.  Those
    claims remain scoped to the original saved independent-validator receipt.
    """
    prefix = 'raw_' if raw else ''
    visited = plan['visit_order_candidate_ids']
    nodes = plan['navigation_global_nodes']
    expected_nodes = [plan['entry_global_node'], *[int(j) + 1 for j in visited], plan['exit_global_node']]
    need(nodes == expected_nodes and len(visited) == len(set(visited)), 'Saved route ID/order/repetition differs')
    legs = plan['route_leg_costs_m']
    points = plan['route_polyline']
    need(len(legs) == len(nodes) - 1 and len(points) >= 1, 'Saved route leg/polyline size differs')
    need(all(math.isfinite(float(x)) and float(x) >= 0 for x in legs), 'Nonfinite/negative saved leg')
    need(all(len(point) == 2 and all(math.isfinite(float(x)) for x in point) for point in points), 'Nonfinite saved polyline')
    leg_length = math.fsum(float(x) for x in legs)
    polyline_length = math.fsum(math.dist(a, b) for a, b in zip(points, points[1:]))
    stops = len(visited)
    objective = leg_length + .75 * stops
    need(close(polyline_length, leg_length) and close(plan['route_length_m'], leg_length)
         and close(plan['objective_length_plus_stops_m'], objective), 'Saved geometry J/L differs')
    need(close(metrics[prefix + 'J'], objective) and close(metrics[prefix + 'L'], leg_length)
         and metrics[prefix + 'K'] == stops, 'Saved route does not match recorded J/L/K')
    assigned = [row['task_id'] for row in plan['assignments']]
    targets = plan['target_task_ids_global']
    need(len(assigned) == len(set(assigned)) and set(assigned) == set(targets)
         and plan['unserved_task_ids'] == [] and plan['status'] == 'complete', 'Saved task manifest is incomplete')
    certificate = plan['validation']
    need(certificate['valid_model_plan'] and certificate['all_tasks_served']
         and certificate['collision_count'] == 0 and certificate['physical_joint_trajectory_certified'] is False,
         'Saved independent-validator certificate is not a complete proxy PASS')
    need(plan['physical_joint_trajectory_certified'] is False and plan['arm_joint_trajectories'] is None,
         'Unsupported physical IK/joint claim in saved plan')
    return dict(leg_length_m=leg_length, polyline_length_m=polyline_length, K=stops,
                J=objective, tasks=len(targets), saved_validation_valid=True,
                fresh_physical_validator_rerun=False)


def validate_record_ids(record):
    problem = record.get('problem', {})
    evidence = record.get('scope_evidence', {})
    candidates = problem.get('candidate_ids_global', evidence.get('global_candidate_ids'))
    targets = problem.get('task_ids_global', evidence.get('global_task_ids'))
    need(candidates is not None and targets is not None, 'Saved global candidate/task ID manifest missing')
    for route_key, plan_key, global_key in (('raw_route', 'raw_plan', 'raw_global_route'), ('route', 'plan', 'global_route')):
        route, plan = record[route_key], record[plan_key]
        need(plan['problem_identity'] == record['input_identity'], 'Raw/H plan input identity differs')
        need(targets == plan['target_task_ids_global'], 'Saved raw/H task manifest differs')
        if problem:
            need(problem['identity'] == record['input_identity']
                 and problem['parent_identity'] == plan['parent_identity']
                 and problem['entry_global_node'] == plan['entry_global_node']
                 and problem['exit_global_node'] == plan['exit_global_node'], 'Saved problem/endpoints/parent identity differ')
        need(len(route) >= 2 and route[0] == route[-1] == 0
             and all(type(node) is int and 1 <= node <= len(candidates) for node in route[1:-1]),
             'Invalid saved local route IDs')
        mapped = [plan['entry_global_node'], *[candidates[node - 1] + 1 for node in route[1:-1]], plan['exit_global_node']]
        need(mapped == plan['navigation_global_nodes'], 'Local candidate route does not map to saved global navigation IDs')
        if global_key in record:
            need(mapped == record[global_key], 'Saved redundant global route differs')


def episode_summary(record, check):
    """Bind per-state logger rows to saved complete-episode targets and actions."""
    targets = record['targets']
    task_count = targets['task_count']
    need(type(task_count) is int and task_count >= 0 and
         task_count == len(record['raw_plan']['target_task_ids_global']) == len(record['plan']['target_task_ids_global']),
         'Saved target task_count is not the exact fixed raw/H task manifest count')
    count = max(1, task_count)
    need(targets['scale'] == count and targets['penalty'] == .75, 'Complete-return normalization or stop penalty differs')
    legs = record['raw_plan']['route_leg_costs_m']
    T = len(legs) - 1
    need(len(targets['returns']) == T and len(record['V_old']) == T
         and len(record['actions_local_indices_0based']) == T, 'Saved complete trace lengths disagree')
    h_credit = float(record['metrics']['raw_J']) - float(record['metrics']['J'])
    for t in range(T):
        check()
        length = math.fsum(legs[t:])
        expected = (-(length + .75 * (T - t)) + h_credit) / count
        need(close(length, targets['raw_remaining_L'][t]) and targets['raw_remaining_K'][t] == T - t
             and close(targets['H_gain'], h_credit) and close(targets['returns'][t], expected),
             'Saved complete G target differs from raw legs/stops and terminal H credit')
    return identity(dict(actions=record['actions_local_indices_0based'],
                         G=np.asarray(targets['returns'], dtype=np.float32).astype(float).tolist(),
                         V=np.asarray(record['V_old'], dtype=np.float32).astype(float).tolist()))


def metadata_for(request_id, record, schedule, geometry):
    parts = request_id.split(':')
    kind, method, endpoint, slot = parts[0], '', '', ''
    spec = record.get('spec')
    if kind == 'train':
        _, method, round_text, slot_text = parts
        endpoint, slot = int(round_text), int(slot_text)
        spec = schedule['training'][endpoint - 1]['specs'][slot]
    elif kind == 'warm':
        _, method, batch_text, slot_text = parts
        endpoint, slot = int(batch_text), int(slot_text)
        spec = schedule['training'][endpoint]['specs'][slot]
    elif kind == 'eval':
        _, endpoint_text, repeat, case_id, method = parts
        endpoint = int(endpoint_text)
        case = next(case for case in schedule['evaluation'] if case['id'] == case_id)
        need(spec is None or spec == case['spec'], 'Evaluation spec differs from frozen schedule')
        need(record.get('endpoint') == endpoint and record.get('repeat') == int(repeat)
             and record.get('case_id') == case_id and record.get('method') == method
             and record.get('cohort') == case['group'], 'Evaluation request ID/endpoint/repeat/case/method/cohort differs')
        spec = case['spec']
    physical = geometry[str(spec['map_seed'])]['physical_key'] if spec is not None else record.get('physical_key', '')
    if record.get('physical_key'):
        need(record['physical_key'] == physical, 'Record physical layout differs from schedule')
    if kind in ('warm','train'):
        need(record.get('condition') == method, 'Online condition differs from request ID')
    if record.get('problem'):
        window = record['problem']['window']
        need(all(window[key] == spec[key] for key in ('x0','y0','width','height')),
             'Saved task window differs from fixed schedule')
    return dict(kind=kind, method=record.get('method', method), endpoint_or_round=endpoint,
                slot=slot, spec=spec, physical_key=physical,
                cohort=record.get('cohort', 'training_own_online' if kind == 'train' else 'initial_critic_warm'))


PAIRS = [('ON','OFF'), ('ON','FrozenSL'), ('OFF','FrozenSL'), ('ON','continued_SL'),
         ('OFF','continued_SL'), ('ON','Fixed3'), ('ON','Detour16')]
COMPONENTS = ('raw_remaining_L', 'raw_remaining_K', 'terminal_H_credit', 'value')


def recompute_contrast(rows, method, control, metric, check):
    check()
    cells = defaultdict(dict)
    for row in rows:
        check()
        if row['method'] in (method, control):
            cell = cells[(row['physical_key'], row['case_id'])]
            need(row['method'] not in cell, 'Duplicate contrast case')
            cell[row['method']] = float(row['metrics'][metric])
    need(cells and all(set(cell) == {method, control} for cell in cells.values()), 'Incomplete contrast pairing')
    grouped = defaultdict(list)
    for (physical, _), cell in cells.items(): grouped[physical].append(cell)
    parents = []
    for physical, cases in sorted(grouped.items()):
        check()
        a, b = (float(np.mean([v[m] for v in cases])) for m in (method, control))
        need(b > 0, 'Positive contrast control needed')
        parents.append(dict(physical_key=physical, cases=len(cases), method_mean=a,
                            control_mean=b, relative_saving=(b-a)/b))
    gains = np.array([row['relative_saving'] for row in parents])
    rng = np.random.Generator(np.random.PCG64(11652026))
    samples = gains[rng.integers(0, len(gains), size=(20000, len(gains)))].mean(1)
    result = dict(method=method, control=control, metric=metric, physical_parents=len(gains),
                  point=float(gains.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                  parents=parents, bootstrap=20000, bootstrap_seed=11652026,
                  inference_scope='Descriptive development comparison; one training seed and few independent layouts')
    check()
    return result


def aggregate_calibration(rows, check):
    grouped = defaultdict(list)
    for row in rows:
        check()
        grouped[tuple(row[key] for key in ('endpoint', 'cohort', 'method', 'component', 'physical_key'))].append(row)
    parent_rows = []
    for keys, group in sorted(grouped.items()):
        check()
        error = np.array([float(row['error']) for row in group])
        baseline = np.array([float(row['constant_squared_error']) for row in group])
        parent_rows.append(dict(zip(('endpoint', 'cohort', 'method', 'component', 'physical_key'), keys),
                                states=len(group), problems=len({row['problem_identity'] for row in group}),
                                RMSE=float(np.sqrt(np.mean(error ** 2))), MAE=float(np.mean(np.abs(error))),
                                bias=float(np.mean(error)), constant_RMSE=float(np.sqrt(np.mean(baseline))),
                                MSE=float(np.mean(error ** 2)), constant_MSE=float(np.mean(baseline))))
    macro = []
    groups = defaultdict(list)
    for row in parent_rows:
        check()
        groups[tuple(row[key] for key in ('endpoint', 'cohort', 'method', 'component'))].append(row)
    for keys, group in sorted(groups.items()):
        check()
        macro.append(dict(zip(('endpoint', 'cohort', 'method', 'component'), keys),
                          physical_parents=len(group), mean_parent_RMSE=float(np.mean([r['RMSE'] for r in group])),
                          mean_parent_MAE=float(np.mean([r['MAE'] for r in group])),
                          mean_parent_constant_RMSE=float(np.mean([r['constant_RMSE'] for r in group])),
                          mean_parent_MSE=float(np.mean([r['MSE'] for r in group])),
                          mean_parent_constant_MSE=float(np.mean([r['constant_MSE'] for r in group])),
                          inference='Descriptive parent means; one training seed, exposed development'))
    bins = []
    groups = defaultdict(list)
    for row in rows:
        check()
        groups[tuple(row[key] for key in ('endpoint', 'cohort', 'method', 'component'))].append(row)
    for keys, group in sorted(groups.items()):
        check()
        ordered = sorted(group, key=lambda r: float(r['predicted']))
        for index, indices in enumerate(np.array_split(np.arange(len(ordered)), min(10, len(ordered)))):
            check()
            sample = [ordered[int(i)] for i in indices]
            bins.append(dict(zip(('endpoint', 'cohort', 'method', 'component'), keys),
                             prediction_quantile_bin=index, states=len(sample),
                             physical_parents=len({r['physical_key'] for r in sample}),
                             predicted_min=min(float(r['predicted']) for r in sample),
                             predicted_max=max(float(r['predicted']) for r in sample),
                             predicted_mean=float(np.mean([float(r['predicted']) for r in sample])),
                             target_mean=float(np.mean([float(r['target']) for r in sample])),
                             bias=float(np.mean([float(r['error']) for r in sample])),
                             diagnostic_scope='State-pooled descriptive bins, not independent samples'))
    return parent_rows, macro, bins


def main():
    global _HASH_CHECK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    run, plan_path = args.run.resolve(), args.plan.resolve()
    need(run == DEFAULT_RUN.resolve() and plan_path.is_relative_to(ROOT), 'Only the registered formal run is supported')
    plan = read(plan_path)
    need(plan['helper_sha256'] == digest(__file__), 'Unregistered analysis helper bytes')
    need(type(plan['record_cap']) is int and 0 < plan['record_cap'] <= 6000,
         'Analysis record cap must be separately registered, at most6000')
    need(5 < plan['wall_cap_s'] <= 300, 'Analysis wall cap must allow closing reserve and be at most300s')
    def precheck():
        need(time.perf_counter() - ENTRY_START < plan['wall_cap_s'] - 2., 'Analysis wall cap/closing reserve exhausted')
    _HASH_CHECK = precheck
    output = project_path(plan['output'])
    need(not output.exists(), 'Analysis output exists; no overwrite/replay')
    events, original_tail, original_total = chain(run / 'events.jsonl', precheck)
    admit_one_shot(events)
    starts = {e['sequence'] for e in events if e['event'] == 'session_start'}
    closed = {e['open_sequence'] for e in events if e['event'] == 'session_close'}
    need(starts == closed and not (run / 'WRITER.lock').exists(), 'Formal run is still active/unclosed')
    stage_results = [e for e in events if e['event'] == 'stage_result']
    session_closes = [e for e in events if e['event'] == 'session_close']
    need(session_closes and session_closes[-1]['status'] != 'PAUSED_CLEAN_BOUNDARY', 'Formal run has not reached final closure')
    if stage_results and stage_results[-1]['status'] != 'PAUSED_CLEAN_BOUNDARY':
        stage = stage_results[-1]
    else:
        need(session_closes[-1]['status'] == 'FAILED', 'No terminal formal stage result')
        checkpoints = [e for e in events if e['event'] == 'checkpoint']
        latest_counts = {}
        for event in events:
            if 'study_method' in event and 'counts' in event:
                latest_counts[event['study_method']] = event['counts']
        stage = dict(status='FAILED', progress=checkpoints[-1]['progress'] if checkpoints else
                     dict(round=0, evaluated=[]), trainers=latest_counts,
                     scope='Failure forensic snapshot; last closed checkpoint retained')
    protocol_path = project_path(plan.get('protocol_path', 'rc3_conservative_refinement/integration_20261110/PROTOCOL_DRAFT_R03.json'))
    protocol = read(protocol_path)
    need(digest(protocol_path) == plan['protocol_sha256'] == events[0]['binding']['protocol_sha256'], 'Analysis protocol identity differs')
    source_path = project_path(protocol['source_binding_path'])
    need(digest(source_path) == plan['source_binding_sha256'] == protocol['source_binding_sha256'], 'Analysis source binding differs')
    caps, binding = events[0]['caps'], events[0]['binding']
    need(caps == protocol['caps'] and binding['source_binding_sha256'] == plan['source_binding_sha256'], 'Formal cumulative identity/caps differ')
    charged = math.fsum(e['elapsed_s'] for e in events if e['event'] == 'session_close')
    need(charged + plan['wall_cap_s'] <= caps['wall_s'], 'Insufficient remaining formal wall budget for admitted analysis')
    output.mkdir(parents=True, exist_ok=False)
    ledger = StreamingAnalysisLedger(run, binding, caps, original_tail, original_total, precheck)
    status = 'ANALYSIS_FAILED'
    streams = ExitStack()
    def check():
        precheck()
        ledger.check()
        need(ledger.wall_charged + time.perf_counter() - ENTRY_START < ledger.caps['wall_s'] - 2.,
             'Formal wall cap/closing reserve exhausted')
    _HASH_CHECK = check
    try:
        ledger.emit(dict(event='analysis_start', plan_path=plan_path.relative_to(ROOT).as_posix(),
                         plan_sha256=digest(plan_path), helper_sha256=digest(__file__),
                         original_ledger_tail=original_tail, request_and_optimizer_debits=0))
        source = read(source_path)
        for group in ('source_sha256', 'artifacts'):
            for name, expected in source[group].items():
                check()
                need(digest(project_path(name)) == expected, 'Bound source/artifact changed: ' + name)
        approvals = {}
        for name in ('PROTOCOL_APPROVAL', 'STAGE_AUTHORIZATION'):
            path = HERE / 'authorizations' / (name + '.json')
            record = read(path)
            need(record['status'] == 'APPROVED' and record['authorized_by'] == 'user'
                 and record['protocol_sha256'] == plan['protocol_sha256']
                 and record['source_binding_sha256'] == plan['source_binding_sha256'], 'Approval binding differs')
            need(digest(project_path(record['user_instruction_path'])) == record['user_instruction_sha256'], 'Approval source bytes changed')
            approvals[name] = digest(path)
        need(approvals['PROTOCOL_APPROVAL'] == binding['protocol_approval_sha256']
             and approvals['STAGE_AUTHORIZATION'] == binding['stage_authorization_sha256']
             and read(HERE / 'authorizations/STAGE_AUTHORIZATION.json')['protocol_approval_sha256'] == approvals['PROTOCOL_APPROVAL'], 'Approval chain differs')
        schedule_path = project_path(protocol['schedule_path'])
        need(digest(schedule_path) == protocol['schedule_sha256'], 'Frozen schedule changed')
        schedule = read(schedule_path)
        references = read(ROOT / 'rc3_integrated_rppo/DATA_REFERENCES.json')
        evaluated = stage['progress']['evaluated']
        need(evaluated == sorted(set(evaluated)) and all(e in protocol['checkpoints'] for e in evaluated),
             'Malformed completed evaluation endpoints')
        request_starts = [e for e in events if e['event'] == 'request_start']
        need(len(request_starts) <= plan['record_cap'], 'Saved request record cap exceeded')
        need(len({e['request_id'] for e in request_starts}) == len(request_starts), 'Repeated paid request ID')
        done = {e['request_id']: e for e in events if e['event'] == 'request_done'}
        failed = {e['request_id']: e for e in events if e['event'] == 'request_failed'}
        need(len(done) == sum(e['event'] == 'request_done' for e in events), 'Duplicate completed request')
        need(not set(done) & set(failed), 'Failed request was silently replaced')
        expected_eval = {}
        for endpoint in evaluated:
            expected_eval[endpoint] = {f"eval:{endpoint}:{repeat}:{case['id']}:{method}"
                for repeat in range(protocol['evaluation']['repeats_by_endpoint'][str(endpoint)])
                for case in schedule['evaluation'] for method in protocol['evaluation']['methods']}
            actual_ids = {e['request_id'] for e in request_starts if e['request_id'].startswith(f'eval:{endpoint}:')}
            need(actual_ids == expected_eval[endpoint] and actual_ids <= set(done),
                 'A closed endpoint is missing/duplicating evaluated requests or contains failed slots')
            for filename in ('ALL_ROUTE_RESULTS.csv', 'FULL_REQUEST_TIMING.csv',
                             'CRITIC_CALIBRATION.csv', 'PARENT_CONTRASTS.json'):
                need((run / f'evaluation_{endpoint:04}' / filename).is_file(), 'Closed evaluation table missing: ' + filename)
        counts = Counter()
        for e in events:
            if e['event'] == 'debit':
                counts.update(e['amounts'])
        need(counts['requests'] == len(request_starts), 'Request debits differ from actual starts')
        actual_optimizer = Counter()
        for e in events:
            if e['event'] == 'optimizer_attempt':
                actual_optimizer['actor_optimizer_attempts' if e['role'] in ('actor', 'matched_SL') else 'critic_optimizer_attempts'] += 1
        for key in ('actor_optimizer_attempts', 'critic_optimizer_attempts'):
            need(counts[key] == actual_optimizer[key], 'Actual optimizer attempts differ from predebits')
        if stage['status'] != 'FAILED':
            for key in ('actor_optimizer_attempts', 'critic_optimizer_attempts'):
                need(sum(row[key] for row in stage['trainers'].values()) == counts[key],
                     'Final trainer optimizer counters differ from durable predebits')

        route_fields = ['request_id','status','kind','method','endpoint_or_round','slot','physical_key','cohort',
                        'width','height','map_seed','task_seed','input_identity','record_path','record_sha256',
                        'J','L','K','raw_J','raw_L','raw_K','raw_route','route','raw_global_nodes','global_nodes',
                        'raw_leg_sum_m','raw_polyline_m','H_leg_sum_m','H_polyline_m','tasks',
                        'raw_saved_validator_valid','H_saved_validator_valid','fresh_physical_validator_rerun',
                        'raw_semantics','H_application_scope','error']
        timing_fields = ['request_id','kind','method','endpoint_or_round','slot','physical_key','cohort','width','height',
                         'service_before_IO_s','through_record_IO_s','wrapper_and_IO_overhead_s',
                         'model_startup_s','input_preparation_s','policy_components_s']
        route_output = streams.enter_context(CSVSink(output/'ALL_RAW_H_ROUTES.csv', route_fields, check))
        timing_output = streams.enter_context(CSVSink(output/'ALL_REQUEST_TIMING.csv', timing_fields, check))
        timing_rows, eval_records, episode_hashes = [], [], {}
        route_count = 0
        warm_targets = defaultdict(lambda: defaultdict(list))
        for start in request_starts:
            check()
            request_id = start['request_id']
            completion = done.get(request_id)
            if completion is None:
                route_output.write(dict(request_id=request_id, status='FAILED' if request_id in failed else 'INCOMPLETE',
                                        error=failed.get(request_id, {}).get('error', 'Unclosed request retained')))
                route_count += 1
                continue
            path = project_path(completion['path'])
            need(digest(path) == completion['artifact_sha256'], 'Saved request bytes changed')
            record = read(path)
            meta = metadata_for(request_id, record, schedule, references['geometry'])
            raw = validate_saved_plan(record['raw_plan'], record['metrics'], raw=True)
            final = validate_saved_plan(record['plan'], record['metrics'])
            validate_record_ids(record)
            if meta['kind'] in ('train', 'warm'):
                need(record['input_identity'] == record['plan']['problem_identity'], 'Training saved input/plan identity differs')
            row = dict(request_id=request_id, status='COMPLETE', **{k: v for k, v in meta.items() if k != 'spec'},
                       width=meta['spec']['width'], height=meta['spec']['height'],
                       map_seed=meta['spec']['map_seed'], task_seed=meta['spec']['task_seed'],
                       input_identity=record['input_identity'], record_path=completion['path'],
                       record_sha256=completion['artifact_sha256'], **record['metrics'],
                       raw_route=packed(record['raw_route']), route=packed(record['route']),
                       raw_global_nodes=packed(record['raw_plan']['navigation_global_nodes']),
                       global_nodes=packed(record['plan']['navigation_global_nodes']),
                       raw_leg_sum_m=raw['leg_length_m'], raw_polyline_m=raw['polyline_length_m'],
                       H_leg_sum_m=final['leg_length_m'], H_polyline_m=final['polyline_length_m'],
                       tasks=final['tasks'], raw_saved_validator_valid=True, H_saved_validator_valid=True,
                       fresh_physical_validator_rerun=False,
                       raw_semantics=record.get('raw_semantics', 'Actor complete route before frozen common H'),
                       H_application_scope=('Native final search/H route, no extra common H' if
                           meta['method'] in ('Fixed3','Detour16') else 'One unchanged terminal common H'))
            route_output.write(row)
            route_count += 1
            timing = record.get('timing', {})
            service = timing.get('request_before_IO_s', record.get('service_time_before_IO_s'))
            need(service is not None and math.isfinite(service) and service >= 0, 'Missing real complete service timing')
            through_IO = completion['elapsed_through_record_s']
            need(through_IO + 1e-6 >= service, 'Recorded IO-inclusive time is less than service')
            timing_row = dict(request_id=request_id, **{k: v for k, v in meta.items() if k != 'spec'},
                                    width=meta['spec']['width'], height=meta['spec']['height'],
                                    service_before_IO_s=service, through_record_IO_s=through_IO,
                                    wrapper_and_IO_overhead_s=through_IO - service,
                                    model_startup_s=timing.get('model_load_s', record.get('model_startup_s', 0.)),
                                    input_preparation_s=timing.get('input_prepare_s', record.get('input_preparation_s', 0.)),
                                    policy_components_s=packed(record.get('components_s', {})))
            timing_output.write(timing_row)
            if meta['kind'] == 'eval':
                timing_rows.append(timing_row)
                if record['repeat'] == 0:
                    eval_records.append({key: record[key] for key in
                        ('endpoint','case_id','method','physical_key','cohort','input_identity','spec','metrics')}
                        | ({'targets': record['targets']} if 'targets' in record else {}))
            if 'targets' in record:
                summary_hash = episode_summary(record, check)
                if meta['kind'] in ('train', 'warm'):
                    episode_hashes[request_id] = summary_hash
            if meta['kind'] == 'warm':
                target = record['targets']
                warm_targets[meta['method']]['raw_remaining_L'].extend(target['raw_remaining_L'])
                warm_targets[meta['method']]['raw_remaining_K'].extend(target['raw_remaining_K'])
                warm_targets[meta['method']]['terminal_H_credit'].extend([target['H_gain']] * len(target['returns']))
                warm_targets[meta['method']]['value'].extend(target['returns'])
        guard_fields = ['ledger_sequence','method','event','opportunity','episode_index','accepted','episode_guard_accepted',
                        'rollback_exact','state_count','max_step_kl','max_step_probability_change','max_base_total_variation',
                        'limits','rejected_state_indices','greedy_changes_old_to_new','greedy_changes_base_to_new',
                        'actor_loss','critic_loss','actor_gradient_norm','critic_gradient_norm']
        advantage_fields = ['ledger_sequence','method','event','opportunity','episode_id','state','action_local_index_0based',
                            'task_count','complete_return_to_go','old_value','advantage','actor_update_state',
                            'recorded_advantage_verified','independence']
        guard_output = streams.enter_context(CSVSink(output/'GUARD_UPDATES.csv',guard_fields,check))
        advantage_output = streams.enter_context(CSVSink(output/'PER_STATE_ADVANTAGES.csv',advantage_fields,check))
        guard_count, advantage_count = 0, 0
        # Stop at the verified original prefix; new analysis journal rows are
        # not part of the scientific evidence and are never recursively read.
        for event in iter_events(run/'events.jsonl'):
            if event['sequence'] >= original_total: break
            check()
            kind = event['event']
            if kind not in ('ppo_opportunity', 'matched_supervised_opportunity', 'critic_warm_opportunity'):
                continue
            method = event['study_method']
            if 'guard' in event:
                for index, guard in enumerate(event['guard']):
                    need(guard['limits'] == protocol['guard'], 'Guard threshold changed during formal run')
                    need(all(math.isfinite(float(v)) for v in guard['maxima'].values()), 'Nonfinite guard maxima')
                    guard_output.write(dict(ledger_sequence=event['sequence'], method=method, event=kind,
                                       opportunity=event.get('opportunity'), episode_index=index,
                                       accepted=event['accepted'], episode_guard_accepted=guard['accepted'],
                                       rollback_exact=event['rollback_exact'], state_count=guard['state_count'],
                                       **guard['maxima'], limits=packed(guard['limits']),
                                       rejected_state_indices=packed(guard['rejected_state_indices']),
                                       greedy_changes_old_to_new=guard['greedy_changes_old_to_new'],
                                       greedy_changes_base_to_new=guard['greedy_changes_base_to_new'],
                                       actor_loss=event.get('actor_loss'), critic_loss=event.get('critic_loss'),
                                       actor_gradient_norm=event.get('actor_gradient_norm'),
                                       critic_gradient_norm=event.get('critic_gradient_norm')))
                    guard_count += 1
            for episode in event.get('episodes', []):
                if 'returns' not in episode:
                    continue
                G, V = episode['returns'], episode['old_values']
                actions = episode['actions_zero_based']
                need(len(G) == len(V) == len(actions), 'Per-state advantage/action lengths differ')
                need(episode['episode_id'] in episode_hashes and episode_hashes[episode['episode_id']] ==
                     identity(dict(actions=actions, G=G, V=V)), 'Per-state event does not match its paid saved trajectory')
                explicit = episode.get('advantages')
                need(explicit is None or len(explicit) == len(G), 'Recorded advantage vector has missing/extra states')
                for index, (g, v, action) in enumerate(zip(G, V, actions)):
                    check()
                    # The admitted trainer subtracts float32 tensors.  Recreate
                    # that arithmetic rather than imposing float64 equality.
                    a = float(np.float32(g) - np.float32(v))
                    need(all(math.isfinite(float(x)) for x in (g, v, a)), 'Nonfinite G/V/advantage')
                    if explicit is not None:
                        need(close(a, explicit[index]), 'Recorded advantage does not equal complete G minus V_old')
                    advantage_output.write(dict(ledger_sequence=event['sequence'], method=method, event=kind,
                                            opportunity=event.get('opportunity', event.get('warm_opportunity')),
                                            episode_id=episode['episode_id'], state=index, action_local_index_0based=action,
                                            task_count=episode['task_count'], complete_return_to_go=g, old_value=v,
                                            advantage=a, actor_update_state=kind == 'ppo_opportunity',
                                            recorded_advantage_verified=explicit is not None,
                                            independence='States/episodes are not independent physical parents'))
                    advantage_count += 1

        calibration, contrasts, analysis_inputs = [], [], []
        eval_map = {(r['endpoint'], r['case_id'], r['method']): r for r in eval_records if r.get('repeat', 0) == 0}
        for endpoint in evaluated:
            check()
            directory = run / f'evaluation_{endpoint:04}'
            cal_path = directory / 'CRITIC_CALIBRATION.csv'
            expected_calibration = {(endpoint, case['id'], method, state, component)
                for case in schedule['evaluation'] for method in ('ON','OFF')
                for state in range(len(eval_map[(endpoint,case['id'],method)]['targets']['returns']))
                for component in COMPONENTS}
            seen_calibration = set()
            if cal_path.is_file():
                analysis_inputs.append(dict(path=cal_path.relative_to(ROOT).as_posix(), sha256=digest(cal_path)))
                with cal_path.open(encoding='utf-8-sig', newline='') as stream:
                    for row in csv.DictReader(stream):
                        check()
                        key = (int(row['endpoint']), row['case_id'], row['method'])
                        need(key in eval_map and key[0] == endpoint, 'Calibration row has no exact saved evaluation request')
                        record = eval_map[key]
                        row['physical_key'] = record['physical_key']
                        component, state = row['component'], int(row['state'])
                        full_key = (*key, state, component)
                        need(full_key in expected_calibration and full_key not in seen_calibration,
                             'Calibration extra/duplicate component/state/method/case')
                        seen_calibration.add(full_key)
                        need(row['cohort'] == record['cohort'] and row['problem_identity'] == record['input_identity'],
                             'Calibration physical problem/cohort differs from saved request')
                        targets = record['targets']
                        expected_target = (targets['H_gain'] if component == 'terminal_H_credit' else
                                           targets['returns'][state] if component == 'value' else targets[component][state])
                        expected_constant = float(np.mean(warm_targets[row['method']][component]))
                        for name in ('predicted', 'target', 'constant', 'error', 'squared_error', 'constant_squared_error'):
                            row[name] = float(row[name])
                            need(math.isfinite(row[name]), 'Nonfinite saved calibration row')
                        need(close(row['target'], expected_target) and close(row['constant'], expected_constant),
                             'Calibration target/warm-only constant differs from saved episodes')
                        need(close(row['predicted'] - row['target'], row['error'])
                             and close(row['error'] ** 2, row['squared_error'])
                             and close((row['constant'] - row['target']) ** 2, row['constant_squared_error']),
                             'Calibration saved arithmetic differs')
                        calibration.append({key: row[key] for key in ('endpoint','cohort','method','component',
                            'physical_key','problem_identity','predicted','target','constant','error','constant_squared_error')})
            need(seen_calibration == expected_calibration, 'Closed endpoint calibration state x4component table is incomplete')
            contrast_path = directory / 'PARENT_CONTRASTS.json'
            expected_contrasts = {(endpoint, cohort, size, method, control, metric)
                for cohort in {case['group'] for case in schedule['evaluation']}
                for size in ['all'] + sorted({f"{case['spec']['width']}x{case['spec']['height']}"
                    for case in schedule['evaluation'] if case['group'] == cohort})
                for method, control in PAIRS for metric in ('J','raw_J','L','K')}
            seen_contrasts = set()
            if contrast_path.is_file():
                analysis_inputs.append(dict(path=contrast_path.relative_to(ROOT).as_posix(), sha256=digest(contrast_path)))
                for contrast in read(contrast_path):
                    check()
                    contrast_key = tuple(contrast[key] for key in ('endpoint','cohort','size','method','control','metric'))
                    need(contrast_key in expected_contrasts and contrast_key not in seen_contrasts,
                         'Extra or duplicate endpoint/cohort/size/pair/metric contrast')
                    seen_contrasts.add(contrast_key)
                    selected = [r for r in eval_records if r['endpoint'] == contrast['endpoint']
                                and r['cohort'] == contrast['cohort'] and r.get('repeat', 0) == 0
                                and (contrast['size'] == 'all' or f"{r['spec']['width']}x{r['spec']['height']}" == contrast['size'])]
                    expected = recompute_contrast(selected, contrast['method'], contrast['control'], contrast['metric'], check)
                    expected.update(endpoint=contrast['endpoint'], cohort=contrast['cohort'], size=contrast['size'])
                    need(expected == contrast, 'Full contrast/parents/CI/bootstrap seed and count differ from exact registered recomputation')
                    contrasts.append(dict(contrast, raw_metric_semantics=(
                        'Learned actor raw=before common H; native raw=its final native search/H route; not a common-H ablation'
                        if contrast['metric'] == 'raw_J' and contrast['control'] in ('Fixed3','Detour16') else
                        'Recorded metric with method-specific native processing retained')))
            need(seen_contrasts == expected_contrasts, 'Closed endpoint contrast grid is incomplete')
        parents, macro, bins = aggregate_calibration(calibration, check)
        checked_csv(output / 'CRITIC_CALIBRATION_PARENTS.csv', parents, check)
        checked_csv(output / 'CRITIC_CALIBRATION_PARENT_MACRO.csv', macro, check)
        checked_csv(output / 'CRITIC_CALIBRATION_BINS.csv', bins, check)
        checked_json(output / 'QUALITY_CONTRASTS.json', contrasts, check)

        timing_groups = defaultdict(list)
        for row in timing_rows:
            check()
            if row['kind'] == 'eval':
                timing_groups[(row['endpoint_or_round'], row['cohort'], row['method'],
                               f"{row['width']}x{row['height']}")].append(row)
        summaries = []
        for keys, rows in sorted(timing_groups.items()):
            check()
            summary = dict(zip(('endpoint', 'cohort', 'method', 'size'), keys), requests=len(rows),
                           physical_parents=len({r['physical_key'] for r in rows}),
                           closed_endpoint=keys[0] in evaluated,
                           timing_scope='All requests of the closed endpoint' if keys[0] in evaluated else
                                        'Forensic completed saved requests only; failed/incomplete requests excluded from quantiles')
            for name in ('service_before_IO_s', 'through_record_IO_s', 'model_startup_s', 'input_preparation_s'):
                check()
                values = [r[name] for r in rows]
                summary[name + '_median'] = float(np.median(values))
                summary[name + '_p95'] = float(np.quantile(values, .95))
            summaries.append(summary)
        checked_csv(output / 'TIMING_SUMMARY.csv', summaries, check)
        attempted_endpoints = {int(e['request_id'].split(':')[1]) for e in request_starts if e['request_id'].startswith('eval:')}
        actual_independent_parents = {r['physical_key'] for r in eval_records if
            r['endpoint'] in evaluated and 'independent' in r['cohort']}
        endpoint_disposition = []
        for endpoint in protocol['checkpoints']:
            check()
            attempted = {e['request_id'] for e in request_starts if e['request_id'].startswith(f'eval:{endpoint}:')}
            endpoint_disposition.append(dict(endpoint=endpoint,
                planned=len(schedule['evaluation'])*len(protocol['evaluation']['methods'])*
                        protocol['evaluation']['repeats_by_endpoint'][str(endpoint)],
                attempted=len(attempted), completed=len(attempted & set(done)), failed=len(attempted & set(failed)),
                incomplete=len(attempted-set(done)-set(failed)), closed_endpoint=endpoint in evaluated))
        checked_json(output / 'STOP_SUMMARY.json', dict(status=stage['status'], progress=stage['progress'],
                  trainers=stage['trainers'], endpoint_decisions=[e for e in events if e['event'] == 'endpoint_decision'],
                  completed_evaluation_endpoints=evaluated,
                  started_incomplete_endpoints=sorted(attempted_endpoints-set(evaluated)),
                  not_started_endpoints=sorted(set(protocol['checkpoints'])-attempted_endpoints),
                  endpoint_request_disposition=endpoint_disposition,
                  one_training_seed=protocol['seed'], design_independent_exposed_physical_parents=3,
                  actual_closed_evaluation_independent_physical_parents=len(actual_independent_parents),
                  formal_reported_input_counts=stage.get('input_counts','UNKNOWN_FAILED_STAGE_HAS_NO_COMPLETE_RECEIPT'),
                  helper_new_original60_reads=0, helper_new_S3_reads=0, final_heldout_evaluation=False,
                  interpretation='Descriptive exposed development evidence; state/window/timing repeats do not add independent parents'), check)
        audit = dict(status='PASS', source_files=len(source['source_sha256']), artifacts=len(source['artifacts']),
                     source_binding_sha256=plan['source_binding_sha256'], protocol_sha256=plan['protocol_sha256'],
                     approval_hashes=approvals, original_ledger_events=original_total, original_ledger_tail=original_tail,
                     request_starts=len(request_starts), completed=len(done), failed=len(failed),
                     incomplete=len(request_starts) - len(done) - len(failed), resource_debits=dict(counts),
                     optimizer_attempts_from_events=dict(actual_optimizer), guard_rows=guard_count,
                     advantage_rows=advantage_count, route_rows=route_count, calibration_states_components=len(calibration),
                     analysis_inputs=analysis_inputs, new_requests=0, new_optimizer_calls=0,
                     weight_loads=0, navigation_solves=0, new_routes=0,
                     route_audit_scope='Recomputed saved legs/polyline J/L/K/manifests and inspected existing validator certificates; no new physical validation or IK certification',
                     elapsed_before_report_s=time.perf_counter() - ENTRY_START)
        # Flush all streamed tables while the working timer still has the
        # reserved closing margin.  Only small aggregate tables were retained.
        check()
        streams.close()
        check()
        audit['outputs'] = []
        for name in ('ALL_RAW_H_ROUTES.csv','ALL_REQUEST_TIMING.csv','GUARD_UPDATES.csv',
                     'PER_STATE_ADVANTAGES.csv','CRITIC_CALIBRATION_PARENTS.csv',
                     'CRITIC_CALIBRATION_PARENT_MACRO.csv','CRITIC_CALIBRATION_BINS.csv',
                     'QUALITY_CONTRASTS.json','TIMING_SUMMARY.csv','STOP_SUMMARY.json'):
            check()
            path = output/name
            audit['outputs'].append(dict(path=path.relative_to(ROOT).as_posix(), sha256=digest(path), bytes=path.stat().st_size))
        audit['elapsed_before_report_s'] = time.perf_counter()-ENTRY_START
        checked_json(output / 'AUDIT.json', audit, check)
        check()
        ledger.emit(dict(event='analysis_complete', output=output.relative_to(ROOT).as_posix(),
                         audit_sha256=digest(output / 'AUDIT.json'), new_requests=0, new_optimizer_calls=0))
        check()
        status = 'ANALYSIS_COMPLETE'
        print(packed(dict(status=status, output=output.relative_to(ROOT).as_posix(),
                          completed_requests=len(done), guard_rows=guard_count, advantage_rows=advantage_count)))
        return 0
    except BaseException as error:
        write_new(output / 'FAILURE.json', dict(error_type=type(error).__name__, error=str(error),
                                                elapsed_s=time.perf_counter() - ENTRY_START))
        ledger.emit(dict(event='analysis_failed', error=repr(error), output=output.relative_to(ROOT).as_posix()))
        raise
    finally:
        try:
            # On failure preserve partial files and release their handles,
            # without replacing the original exception with a second timeout.
            streams.__exit__(*sys.exc_info())
        finally:
            ledger.close(status)


if __name__ == '__main__':
    raise SystemExit(main())
