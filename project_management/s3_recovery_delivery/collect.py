"""Read-only closeout of already saved S3 results; never invokes a planner/model.

This delivery helper is deliberately outside the frozen evaluation source binding.
Run only after the continuation process has exited, including a paid failure stop.
"""
import csv
from collections import Counter
from pathlib import Path
import time

from rc3_s3_evaluator.common import (
    ROOT, SEEDS, METHODS, H_METHODS, EXPERIMENT, read, digest, identity, label, require,
    project_file, write_new, now,
)
from rc3_s3_recovery_20261108.authority import HOME, RUN, CONT, FAILED_ID, admit
from rc3_s3_recovery_20261108.storage import read_chain


def add_counts(target, values):
    for k, v in values.items():
        require(type(v) in (int, float), 'Nonnumeric resource counter: ' + k)
        target[k] += v


def collect():
    tick = time.perf_counter()
    permit, legacy, binding = admit()
    from rc3_s3_recovery_20261108.analysis import analyze, saved_rows
    from rc3_s3_recovery_20261108.runner import Inputs
    from rc3_s3_evaluator.journal import grid
    cases = Inputs(permit).cases  # Declarations only, no geometry/task materialization.
    analysis_tick = time.perf_counter()
    analysis = analyze(saved_rows(legacy), cases)
    analysis.update(binding=binding, analysis_wall_s=time.perf_counter()-analysis_tick,
        analysis_source='Read-only saved partial results, not a successful scheduling receipt',
        actual_new_analysis_generation_inference_native_H_optimizer=0)
    out = HOME / 'results'
    require(not out.exists(), 'Closeout already exists; do not overwrite evidence')

    old_index = read(ROOT / 'rc3_s3_results_20261108/ARTIFACT_INDEX.json')
    old_checked = []
    for e in old_index['artifacts']:
        p = project_file(e['path'])
        require(p.stat().st_size == e['bytes'] and digest(p) == e['sha256'], 'Original evidence changed')
        old_checked.append(e)
    require(len(old_checked) == 180, 'Wrong original evidence count')
    weights = []
    index = read(ROOT / 'rc3_s3_evaluator/S3_MODEL_INDEX.json')
    for seed, models in index['models'].items():
        for role, entry in models.items():
            files = []
            for pathkey, shakey in [('path','checkpoint_sha256'), ('sidecar_path','sidecar_sha256'), ('commit_path','commit_sha256')]:
                p = project_file(entry[pathkey])
                require(digest(p) == entry[shakey], 'Frozen weight or provenance changed')
                files.append(dict(path=entry[pathkey], sha256=entry[shakey], bytes=p.stat().st_size))
            weights.append(dict(seed=int(seed), role=role, files=files, model_tensor_sha256=entry['model_tensor_sha256']))
    require(len(weights) == 12, 'Wrong frozen model count')

    old_resource = read(ROOT / 'rc3_s3_results_20261108/RESOURCE_ACCOUNTING.json')
    operations = Counter(old_resource['request_operation_counts'])
    component_seconds = Counter(old_resource['all_logged_operation_end_seconds_inclusive'])
    pipeline = Counter(old_resource['pipeline_counters'])
    internal = Counter(old_resource['internal_search_function_entries'])
    internal_inclusive, internal_exclusive = Counter(), Counter()
    hot_counts = Counter(old_resource['hot_prime_counts'])
    method_h = Counter()
    records = []
    byseed = {}
    all_ids = set()
    completed_time = 0.0
    result_paths = []
    route_stability = {}
    rl_pairs = {}
    endpoint_certificates = Counter()
    states = {q['id']:dict(request=q,state='UNEXECUTED') for s in SEEDS for q in grid(cases,s)}
    for rid in legacy['results']:
        states[rid]['state'] = 'COMPLETED'
        states[rid]['segment'] = 'legacy_b348'
    states[FAILED_ID].update(state='INFRASTRUCTURE_FAILED_DISPOSED', segment='legacy_b348', result_missing=True)

    def row_record(path, event, old=False):
        nonlocal completed_time
        require(digest(path) == event['artifact_sha256'], 'Result hash changed')
        row = read(path)
        q = row['request']
        require(q['id'] not in all_ids, 'Duplicate successful request')
        all_ids.add(q['id'])
        require(row['valid'] and row['plan']['validation']['valid_model_plan'], 'Uncertified result')
        expected_h = int(q['method'] in H_METHODS)
        require(row['counts'].get('common_H_invocations', 0) == expected_h, 'H method mismatch')
        method_h[q['method']] += expected_h
        key = (q['case_id'], q['seed'], q['method'])
        value = (row['raw_route_sha256'], row['route_sha256'], row['metrics'])
        require(key not in route_stability or route_stability[key] == value, 'Route/cost changed across versions')
        route_stability[key] = value
        if q['method'] in ('RL_raw', 'RL_plus_H'):
            pair = (q['case_id'], q['seed'], q['kind'], q['repeat'])
            rl_pairs.setdefault(pair, {})[q['method']] = row['raw_route_sha256']
        complete_time = event['complete_time_s']
        completed_time += complete_time
        rec = dict(request_id=q['id'], case_id=q['case_id'], seed=q['seed'], method=q['method'],
                   kind=q['kind'], repeat=q['repeat'], group=row['group'], size='x'.join(map(str,row['size'])),
                   endpoint=row['endpoint'], physical_key=row['physical_key'], **row['metrics'],
                   complete_time_s=complete_time, H_gain_J=row['H_gain_J'], valid=row['valid'],
                   persistence_revision=row.get('persistence_revision','legacy_b348'),
                   raw_route_sha256=row['raw_route_sha256'], route_sha256=row['route_sha256'],
                   artifact_path=label(path), artifact_sha256=event['artifact_sha256'])
        records.append(rec)
        result_paths.append(dict(path=label(path), sha256=event['artifact_sha256'], bytes=path.stat().st_size))
        endpoint_certificates[row['endpoint']] += 1
        add_counts(internal_inclusive, row['internal_search']['inclusive_s'])
        add_counts(internal_exclusive, row['internal_search']['exclusive_s'])
        if not old:
            add_counts(pipeline, row['pipeline_counters'])
            add_counts(internal, row['internal_search']['counts'])

    for e in legacy['results'].values():
        row_record(project_file(e['artifact_path']), e, old=True)

    for seed in SEEDS:
        folder = CONT / str(seed)
        if not folder.exists():
            byseed[str(seed)] = dict(status='NOT_STARTED',new_debits=0,new_success=0,new_failures=0,
                total_debits=0,total_success=0,historical_failures=0,original_exact_wall_s=0,
                conservative_carry_s=0,measured_continuation_wall_s=0,charged_cumulative_wall_s=0,
                remaining_wall_s=14400,wall_limit_s=14400,sessions=[],anchor_retry_receipts=0,
                anchor_retry_events=0,chain_events=0,chain_tail_sha256=None,schedule_sha256=None)
            continue
        schedule_path = folder / 'SCHEDULE_COMPLETE.json'
        schedule = read(schedule_path) if schedule_path.exists() else None
        seal = read(folder / 'SCHEDULE_SEAL.json') if schedule else None
        if schedule:
            require(digest(schedule_path) == seal['schedule_sha256'], 'Schedule seal differs')
        events, tail = read_chain(folder / 'journal.jsonl')
        head = read(folder / 'HEAD.json')
        require(not (folder / 'HEAD.pending').exists() and head['sequence'] == len(events) and head['sha256'] == tail, 'Unclosed new anchor')
        if schedule:
            require(digest(folder / 'journal.jsonl') == schedule['journal_sha256'], 'Schedule journal changed')
        starts, done, errors, last_counts = {}, {}, {}, None
        for e in events:
            if e['event'] == 'request_start':
                require(last_counts is None, 'Overlapping paid request')
                starts[e['request']['id']] = e['request']
                last_counts = {}
            elif e['event'] == 'operation_end':
                component_seconds[e['operation']] += e['elapsed_s']
                if last_counts is not None:
                    last_counts = e['counts']
            elif e['event'] == 'hot_prime':
                require(last_counts is None, 'Prime inside paid request')
                add_counts(hot_counts, e['counts'])
            elif e['event'] == 'request_complete':
                require(e['id'] in starts and e['id'] not in done and last_counts is not None, 'Bad completion')
                add_counts(operations, last_counts)
                last_counts = None
                done[e['id']] = e
                row_record(folder / 'results' / (e['id']+'.json'), e)
                states[e['id']].update(state='COMPLETED',segment=binding['recovery_source_binding_sha256'])
            elif e['event'] == 'request_error':
                require(e['id'] in starts and last_counts is not None, 'Unbilled failure')
                add_counts(operations, e['counts'])
                last_counts = None
                errors[e['id']] = e
                states[e['id']].update(state='INPUT_FEASIBILITY_FAILED',segment=binding['recovery_source_binding_sha256'],
                    result_missing=True,error=e['error'],resolved_for_continuation=False)
        require(set(starts) == set(done)|set(errors) and not (set(done)&set(errors)) and last_counts is None, 'Unknown incomplete request')
        require({p.stem for p in (folder/'results').glob('*.json')} == set(done), 'Orphan result')
        require(events[-1]['event'] == 'session_close', 'Unclosed parent journal')
        require(not (set(starts) & set(legacy['request_starts'])), 'Replayed old request')
        sessions = []
        for p in sorted((folder / 'sessions').iterdir()):
            final = read(p / 'FINAL.json')
            final_seal = read(p / 'FINAL_SEAL.json')
            require(digest(p / 'FINAL.json') == final_seal['final_sha256'], 'Final clock changed')
            sessions.append(dict(path=label(p), final_sha256=final_seal['final_sha256'], **final))
        carry = 592 if seed == SEEDS[0] else 0
        used = carry + sum(s['elapsed_s'] for s in sessions)
        require(used <= 14400, 'Cumulative stage budget exceeded')
        retries = [read(p) for p in (folder / 'io_receipts').glob('ANCHOR_RETRY_*.json')]
        inherited_debits, inherited_success, inherited_failures = (163,162,1) if seed==SEEDS[0] else (0,0,0)
        byseed[str(seed)] = dict(status='STOPPED_INPUT_FEASIBILITY_FAILURE' if errors else 'SCHEDULE_DISPOSED',
            new_debits=len(starts), new_success=len(done), new_failures=len(errors),
            total_debits=inherited_debits+len(starts), total_success=inherited_success+len(done),
            historical_failures=inherited_failures,
            original_exact_wall_s=None if seed==SEEDS[0] else 0, conservative_carry_s=carry,
            measured_continuation_wall_s=sum(s['elapsed_s'] for s in sessions), charged_cumulative_wall_s=used,
            remaining_wall_s=14400-used, wall_limit_s=14400, sessions=sessions,
            anchor_retry_receipts=len(retries), anchor_retry_events=len({r['payload']['same_event_sha256'] for r in retries}),
            chain_events=len(events), chain_tail_sha256=tail, schedule_sha256=seal['schedule_sha256'] if seal else None)
    require(len(records) == analysis['observed'], 'Wrong successful result count')
    require(all(set(p) == {'RL_raw','RL_plus_H'} and len(set(p.values())) == 1 for p in rl_pairs.values()), 'Missing/different RL raw pair')
    require(FAILED_ID not in all_ids, 'Historical failure was replaced')
    resource = dict(scope='S3_ORIGINAL_LEDGER_PLUS_PAID_CONTINUATION', experiment_id=EXPERIMENT,
        original180_unchanged=True, original_exact_first_stage_wall_s=None,
        old_requests=dict(debits=163,success=162,infrastructure_failures=1),
        actual_by_seed=byseed, total_debits=sum(s['total_debits'] for s in byseed.values()),
        new_request_debits=sum(s['new_debits'] for s in byseed.values()),
        complete_requests=len(records), failed_requests=1+sum(s['new_failures'] for s in byseed.values()),
        not_executed_requests=15876-sum(s['total_debits'] for s in byseed.values()),
        missing_request_ids=analysis['missing_ids'], request_operation_counts=dict(operations),
        all_operation_end_seconds_inclusive=dict(component_seconds),
        component_timing_rule='Nested inclusive diagnostics, never added as stage wall',
        pipeline_counters=dict(pipeline), internal_search_function_entries=dict(internal),
        internal_search_inclusive_s=dict(internal_inclusive),internal_search_exclusive_s=dict(internal_exclusive),
        hot_prime_counts=dict(hot_counts), H_calls_by_method=dict(method_h),
        complete_request_time_sum_s=completed_time,
        navigation_solves_total=operations['navigation_solves']+hot_counts['navigation_solves'],
        optimizer_updates=0,new_teacher_queries=0,original60_semantic_reads=0,
        exact_all_stage_wall_s=None,
        charged_all_stage_wall_s=sum(s['charged_cumulative_wall_s'] for s in byseed.values()),
        wall_rule='592s conservative old reservation plus all measured continuation clocks, including the closed 9.802301499992609s zero-request dispatch failure; old exact time stays null',
        analysis_wall_s=analysis['analysis_wall_s'],
        analysis_science_generation_inference_native_H_optimizer=0,
        recovery_source_binding_sha256=binding['recovery_source_binding_sha256'])
    audit = dict(status='PASS', original180_checked=old_checked, frozen_models=weights,
        successful_results_checked=len(records), method_H_definition='PASS',
        RL_raw_plus_H_pairs=len(rl_pairs), deterministic_case_seed_methods=len(route_stability),
        endpoint_certificates=dict(endpoint_certificates), failed_slot_retained=True,
        all_legacy_request_ids_excluded_from_new_starts=True, new_pending_files=0,
        original_HEAD_and_pending_preserved=True, source_binding_sha256=binding['recovery_source_binding_sha256'])
    out.mkdir()
    with (out / 'REQUEST_METRICS.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    write_new(out/'RESOURCE_ACCOUNTING.json',resource)
    write_new(out/'RECOVERY_AUDIT.json',audit)
    write_new(out/'SCIENTIFIC_ANALYSIS.json',analysis)
    write_new(out/'REQUEST_STATE_INDEX.json',dict(scope='Original full grid, no replacement or imputation',
        counts=dict(Counter(s['state'] for s in states.values())), requests=list(states.values())))
    write_new(out/'RESULT_ARTIFACT_INDEX.json',dict(scope='All successful old and continuation results',artifacts=sorted(result_paths,key=lambda x:x['path'])))
    write_new(out/'CLOSEOUT_RECEIPT.json',dict(utc=now(), analysis_sha256=digest(out/'SCIENTIFIC_ANALYSIS.json'),
        recovery_binding=binding, helper_sha256=digest(Path(__file__)),
        read_only_closeout_elapsed_s=time.perf_counter()-tick,
        actual_new_target_generation_navigation_native_H_inference_optimizer=0))
    print(dict(results=len(records), new_debits=resource['new_request_debits'], quality=analysis['independent_RL_pass'],
               deployment=analysis['deployment_pass'], output=label(out)))


if __name__ == '__main__':
    collect()
