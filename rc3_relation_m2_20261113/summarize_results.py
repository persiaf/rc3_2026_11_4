"""Summarize saved M2 results only after its supervised stages have stopped."""
import csv
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
RUN = ROOT/'rc3_relation_drl_20261113/science/relation_navcover_m2_v1_20261113'
METHODS = ('FrozenSL', 'RelationInit', 'RelationSL', 'RelationRL', 'Fixed3', 'Detour16')


def read(path):
    return json.loads(path.read_text(encoding='utf8')) if path.is_file() else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def csvout(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    summary = dict(scientific_id='relation_navcover_m2_v1_20261113', generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol_sha256=sha(ROOT/'rc3_relation_drl_20261113/M2_PROTOCOL.json'), stages={}, evaluations=[],
        model_forwards=0, optimizer_calls=0, new_planning_or_H=0)
    updates, resource_curve, full_times, failures = [], [], [], []
    total = dict(request_attempts=0, request_completed=0, request_failed=0, optimizer_attempts=0, optimizer_completed=0, optimizer_failed=0)
    for stage in ('init', 'paired'):
        folder = RUN/stage
        resource = read(folder/'RESOURCE_ACCOUNTING.json')
        complete = read(folder/'COMPLETE.json')
        if not resource:
            summary['stages'][stage] = dict(status='NOT_RUN')
            continue
        if resource['status'] == 'RUNNING':
            raise RuntimeError('Do not finalize an actively running stage')
        controls = sorted((folder/'controls').glob('attempt_*/SUPERVISOR_RECEIPT.json'))
        summary['stages'][stage] = dict(status=resource['status'], resource=resource, complete=complete,
            supervisor_receipts=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), elapsed_s=read(p)['elapsed_s'], status=read(p)['status']) for p in controls])
        for key in total:
            total[key] += resource['counts'].get(key, 0)
        counts = dict(request_attempts=0, request_completed=0, request_failed=0, optimizer_attempts=0, optimizer_completed=0, optimizer_failed=0)
        update_events = {}
        success_ids, failed_ids = set(), set()
        journal = folder/'EVENTS.jsonl'
        for line in journal.read_text(encoding='utf8').splitlines():
            event = json.loads(line)
            kind = event.get('kind')
            if kind in ('request', 'optimizer') and event['event'] in ('begin', 'complete', 'failed'):
                key = kind+'_'+dict(begin='attempts', complete='completed', failed='failed')[event['event']]
                counts[key] += 1
                if kind == 'request' and event['event'] == 'complete':
                    success_ids.add(event['id'])
                if kind == 'request' and event['event'] == 'failed':
                    failed_ids.add(event['id'])
                resource_curve.append(dict(stage=stage, elapsed_s=event['elapsed_s'], **counts,
                    request_success_excluding_failed=len(success_ids-failed_ids)))
                if kind == 'optimizer' and event['event'] == 'complete':
                    update_events[event['id']] = event
                if event['event'] == 'failed':
                    failures.append(dict(stage=stage, **event))
            if event['event'] == 'request_full_time':
                full_times.append(dict(stage=stage, request_id=event['id'], elapsed_s=event['elapsed_s'], full_request_s=event['full_request_s']))
        for path in sorted((folder/'updates').glob('*.json')):
            value = read(path)
            event = update_events.get(value['id'], {})
            stream = 'InitSL' if stage == 'init' else value['id'].split('_')[-1]
            update = int(value['id'].split('_')[1])
            updates.append(dict(stage=stage, stream=stream, update=update, loss=value['loss'], gradient_norm=value['gradient_norm'],
                weight_changed=value['weight_changed'], before=value['before'], after=value['after'],
                elapsed_s=event.get('elapsed_s'), optimizer_call_s=event.get('wall_s'), path=path.relative_to(ROOT).as_posix()))
        summary['stages'][stage]['request_event_reconciliation'] = dict(
            unique_complete_ids=len(success_ids), unique_failed_ids=len(failed_ids),
            overlap_ids=sorted(success_ids & failed_ids), exclusive_success_ids=len(success_ids-failed_ids),
            interpretation='Derived only; original journal and ledger counters preserved verbatim')
    paired = summary['stages']['paired']
    completed = paired.get('complete')
    if completed and completed['status'] == 'SCIENTIFIC_PASS':
        summary['verdict'] = 'SCIENTIFIC_PASS'
    elif completed and completed['status'] in ('NO_DEVELOPMENT_SIGNAL', 'STOP_SEVERE_DEGRADATION'):
        summary['verdict'] = 'NO_DEVELOPMENT_SIGNAL'
    else:
        summary['verdict'] = 'TECHNICAL_INCOMPLETE'
    summary['exact_terminal_status'] = completed['status'] if completed else paired['status']
    if paired['status'] == 'NOT_RUN':
        summary['exact_terminal_status'] = summary['stages']['init']['status']
    rows, table, gains = [], [], []
    for endpoint in (0, 512, 2048):
        path = RUN/'paired/evaluation'/f'endpoint_{endpoint:04}'/'RESULTS.json'
        report = read(path)
        entry = dict(endpoint=endpoint, status='NOT_RUN', processed=0, complete=False)
        if report:
            entry.update(status=report.get('gate', {}).get('status', 'INCOMPLETE'), processed=len(report['rows']), complete=report['complete'],
                         path=path.relative_to(ROOT).as_posix(), sha256=sha(path), gate=report.get('gate'))
            for row in report['rows']:
                rows.append(dict(endpoint=endpoint, **{k:v for k,v in row.items() if k not in ('metrics', 'record')},
                    **row.get('metrics', {}), record_path=row.get('record', {}).get('path'), record_sha256=row.get('record', {}).get('sha256')))
        for population in ('LOCAL66', 'FULL12'):
            for method in METHODS:
                aggregate = report.get('summary', {}).get(population, {}).get(method, {}).get('overall', {}) if report else {}
                record = dict(endpoint=endpoint, population=population, method=method, planned=66 if population == 'LOCAL66' else 12,
                    processed=aggregate.get('processed', 0), valid=aggregate.get('valid', 0), complete=aggregate.get('complete', False))
                if record['complete']:
                    record.update(aggregate['parent_equal_mean'], physical_parents=aggregate['physical_parents'],
                        time_p50_s=aggregate['time_p50_s'], time_p95_s=aggregate['time_p95_s'])
                table.append(record)
        if report and report['complete'] and 'gate' in report:
            for population, refs in report['gate']['comparisons'].items():
                for reference, groups in refs.items():
                    for group, result in groups.items():
                        gains.append(dict(endpoint=endpoint, population=population, reference=reference, group=group,
                                          complete=result['complete'], relative_gain=result['mean'], parents=len(result['parents'])))
        summary['evaluations'].append(entry)
    summary.update(total_resources=total, recorded_updates=len(updates), failures=failures,
                   elapsed_stage_wall_s=sum(s.get('resource', {}).get('elapsed_s', 0) for s in summary['stages'].values()),
                   interpretation='Exposed LOCAL66 and training-parent FULL12 kept separate; no heldout/new-parent/physical claim')
    out = HOME/'analysis'
    put(out/'SUMMARY.json', summary)
    csvout(out/'TRAINING_UPDATES.csv', updates, ['stage','stream','update','loss','gradient_norm','weight_changed','elapsed_s','optimizer_call_s','before','after','path'])
    csvout(out/'RESOURCE_CURVE.csv', resource_curve, ['stage','elapsed_s',*total,'request_success_excluding_failed'])
    csvout(out/'COMPLETE_REQUEST_TIMES.csv', full_times, ['stage','request_id','elapsed_s','full_request_s'])
    csvout(out/'TASK_RESULTS.csv', rows, ['endpoint','population','case_id','method','valid','group','physical_key','width','height','mode',
        'candidate_count','task_count','entry','exit','J','L','K','raw_J','raw_L','raw_K','full_request_s','problem_identity','record_path','record_sha256','raw_semantics','error'])
    csvout(out/'METHOD_TABLE.csv', table, ['endpoint','population','method','planned','processed','valid','complete','physical_parents',
        'J','L','K','raw_J','raw_L','raw_K','time_p50_s','time_p95_s'])
    csvout(out/'PAIRED_GAINS.csv', gains, ['endpoint','population','reference','group','complete','relative_gain','parents'])
    print(dict(verdict=summary['verdict'], terminal_status=summary['exact_terminal_status'], resources=total, evaluations=summary['evaluations']))


if __name__ == '__main__':
    main()
