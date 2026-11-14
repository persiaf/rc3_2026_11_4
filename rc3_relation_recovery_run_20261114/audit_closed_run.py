"""Audit closed recovery artifacts without restoring, planning, or training."""
import csv
import json
import math
from pathlib import Path
from rc3_relation_recovery_20261114.persistence import read, sha, load_events, reduce_events

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
PLAN = ROOT / 'rc3_relation_recovery_20261114/RECOVERY_PROTOCOL_DRAFT.json'
RUN = ROOT / read(PLAN)['canonical_recovery_folder']


def write(name, value):
    (HOME / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def main():
    supervisor = read(RUN / 'controls/attempt_0001/SUPERVISOR_RECEIPT.json')
    assert supervisor['root_exit_confirmed'] and supervisor['job_active_processes_at_end'] == 0
    outer = read(RUN / 'OUTER_ACCOUNTING.json')
    proposal = read(PLAN)
    events = load_events(RUN / 'EVENTS.v2.jsonl')
    state = reduce_events(events)
    assert state['origin']['counts']['request_attempts'] == 1910
    assert state['origin']['counts']['optimizer_attempts'] == 238
    assert state['origin']['elapsed_s'] == 2221.312
    snapshot = read(RUN / 'RESOURCE_ACCOUNTING.json')
    assert dict(state['counts']) == snapshot['counts']
    assert state['head'] == snapshot['journal_head']
    requests, updates, timing = [], [], {}
    beginnings = [e for e in events if e['event'] == 'begin']
    for event in events:
        if event['event'] == 'request_full_time':
            timing[event['id']] = event['full_request_s']
        if event['event'] == 'complete' and event['kind'] == 'request':
            path = RUN / event['record']
            assert sha(path) == event['sha256']
            item = read(path)
            assert item['request_id'] == event['id'] and item['complete']
            assert item['purpose'] == 'teacher_replay'
            assert item['teacher_validation']['valid_model_plan'] is True
            assert sha(ROOT / item['teacher_source']['path']) == item['teacher_source']['sha256']
            requests.append(dict(id=event['id'], record=event['record'], sha256=event['sha256'],
                problem_identity=item['problem_identity'], J=item['J'], actions=len(item['actions']),
                teacher_path=item['teacher_source']['path'], teacher_sha256=item['teacher_source']['sha256'],
                valid=True))
    schedule = read(ROOT / proposal['recovery_schedule']['path'])['rows']
    expected_requests = [rid for row in schedule for rid in row['request_ids']]
    expected_updates = [row['optimizer_id'] for row in schedule]
    new_request_ids = [e['id'] for e in beginnings if e['kind'] == 'request']
    new_update_ids = [e['id'] for e in beginnings if e['kind'] == 'optimizer']
    assert new_request_ids == expected_requests[:len(new_request_ids)]
    assert new_update_ids == expected_updates[:len(new_update_ids)]
    assert len(new_request_ids) == len(set(new_request_ids))
    assert len(new_update_ids) == len(set(new_update_ids))
    old_ids = set(state['origin']['ids'])
    assert not old_ids.intersection(e['id'] for e in beginnings)
    previous_model = proposal['checkpoint']['model_sha256']
    for path in sorted((RUN / 'updates').glob('init_*.json')):
        update = read(path)
        assert update['before'] == previous_model
        assert math.isfinite(update['loss']) and math.isfinite(update['gradient_norm'])
        row = schedule[len(updates)]
        assert update['id'] == row['optimizer_id']
        assert [x['task_id'] for x in update['details']] == row['tasks']
        previous_model = update['after']
        updates.append(dict(id=update['id'], loss=update['loss'], gradient_norm=update['gradient_norm'],
                            before=update['before'], after=update['after'], weight_changed=update['weight_changed']))
    complete_path = RUN / 'RECOVERY_COMPLETE.json'
    success = complete_path.exists() and supervisor['status'] == 'COMPLETE'
    checkpoint_audit = None
    if (RUN / 'STATE.json').exists():
        saved_state = read(RUN / 'STATE.json')
        ref = saved_state['checkpoint']
        assert sha(ROOT / ref['path']) == ref['sha256']
        import torch
        from rc3_conservative_refinement.runtime import tensor_digest
        from rc3_relation_drl_20261113.runtime import state_digest
        saved = torch.load(ROOT / ref['path'], map_location='cpu', weights_only=False)
        assert saved['cursor'] == saved_state['cursor'] == state['cursor']
        assert tensor_digest(saved['model']) == ref['model_sha256'] == previous_model
        assert state_digest(saved['optimizer']) == ref['optimizer_sha256']
        assert state_digest(saved['rng']) == ref['rng_sha256']
        assert all(int(s['step']) == saved['cursor'] for s in saved['optimizer']['state'].values())
        assert not any(k in saved for k in ('grad', 'grads', 'graph'))
        checkpoint_audit = dict(reference=ref, model_Adam_RNG_verified=True, Adam_steps=saved['cursor'],
            checkpoint_ids=len(saved['request_ids']), checkpoint_counts=saved['counts'])
        if success:
            receipt = read(complete_path)
            assert sha(RUN / 'OUTER_ACCOUNTING.json') == receipt['outer_accounting_sha256']
            assert sha(RUN / 'controls/attempt_0001/SUPERVISOR_RECEIPT.json') == receipt['supervisor_receipt_sha256']
            assert receipt['actual_counts'] == snapshot['counts']
            assert sha(ROOT / receipt['export']['path']) == receipt['export']['sha256']
            export = torch.load(ROOT / receipt['export']['path'], map_location='cpu', weights_only=False)
            assert tensor_digest(export['model']) == export['model_sha256'] == ref['model_sha256']
            assert export['engineering_only'] is False
            assert set(saved['request_ids']) == old_ids | {e['id'] for e in beginnings}
            assert saved['counts'] == snapshot['counts']
            assert len(requests) == len(new_request_ids) == 2192 and len(updates) == len(new_update_ids) == 274
            assert state['cursor'] == 512 and not state['pending']
            assert state['counts']['request_attempts'] == state['counts']['request_completed'] == 4102
            assert state['counts']['optimizer_attempts'] == state['counts']['optimizer_completed'] == 512
            assert state['counts']['request_failed'] == 1
            assert not any(e['event'] in ('failed', 'persistence_failed') for e in events)
            assert outer['cumulative_wall_s'] <= 14400
            checkpoint_audit['export'] = receipt['export']
            checkpoint_audit['receipt'] = dict(path=complete_path.relative_to(ROOT).as_posix(), sha256=sha(complete_path))
    for record in requests:
        record['full_request_s'] = timing.get(record['id'])
    analysis = HOME / 'analysis'
    analysis.mkdir(exist_ok=True)
    for name, rows, fields in [('TEACHER_REQUESTS.csv', requests, ['id','record','sha256','problem_identity','J','actions','teacher_path','teacher_sha256','valid','full_request_s']),
                               ('TRAINING_UPDATES.csv', updates, ['id','loss','gradient_norm','before','after','weight_changed'])]:
        with (analysis / name).open('w', encoding='utf8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader(); writer.writerows(rows)
    historical = read(ROOT / proposal['historical_binding']['path'])['files']
    for name, expected in historical.items():
        assert sha(ROOT / name) == expected, name
    binding = read(ROOT / proposal['new_source_binding']['path'])['files']
    for name, expected in binding.items():
        assert sha(ROOT / name) == expected, name
    assert not (RUN.parent / 'COMPLETE.json').exists()
    assert not (RUN.parent.parent / 'paired').exists()
    result = dict(status='PASS_CLOSED_RECOVERY_AUDIT' if success else 'TECHNICAL_INCOMPLETE',
        scientific_stage_outcome='COMPLETE_RECOVERED' if success else 'TECHNICAL_INCOMPLETE',
        origin=dict(requests=1910, optimizer=238, wall_s=2221.312),
        new_requests=len(new_request_ids), new_optimizer_attempts=len(new_update_ids), new_updates_completed=len(updates),
        raw_cumulative_counts=dict(state['counts']), cursor=state['cursor'], pending=state['pending'],
        new_failed_events=[e for e in events if e['event'] in ('failed','persistence_failed')],
        checkpoint=checkpoint_audit, new_request_records_verified=len(requests),
        exact_schedule_order_verified=True, preserved_historical_files=len(historical), bound_source_files=len(binding),
        outer_wall=outer, supervisor_status=supervisor['status'], process_tree_exited=True,
        source_model_Adam_RNG_chain_verified=True, original_TECHNICAL_INCOMPLETE_preserved=True,
        original_SL512_COMPLETE_absent=True, M2b_executed=False,
        audit_model_constructions=0, audit_model_forwards=0, audit_optimizer_calls=0, audit_teacher_replays=0)
    write('CLOSED_RUN_AUDIT.json', result)
    write('CURRENT_STAGE_STATUS.json', dict(status=result['scientific_stage_outcome'], execution_closed=True,
        original_science='TECHNICAL_INCOMPLETE', M2b_authorized=False, M2b_executed=False,
        registration_remote_verified=True, cumulative_requests=state['counts']['request_attempts'],
        cumulative_optimizer_attempts=state['counts']['optimizer_attempts'], cursor=state['cursor'],
        cumulative_outer_wall_s=outer['cumulative_wall_s'], new_requests=len(new_request_ids),
        new_optimizer_attempts=len(new_update_ids), post_run_audit=result['status']))
    print(json.dumps({k: result[k] for k in ('status','new_requests','new_optimizer_attempts','new_updates_completed','cursor')}))


if __name__ == '__main__':
    main()
