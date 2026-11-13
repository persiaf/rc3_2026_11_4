"""Audit stopped evidence without restoring or running the scientific model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
SOURCE = ROOT/'rc3_relation_drl_20261113'
RUN = SOURCE/'science/relation_navcover_m2_v1_20261113/init'


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def put(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())


def main():
    resource, state = read(RUN/'RESOURCE_ACCOUNTING.json'), read(RUN/'STATE.json')
    assert resource['status'] == 'SUPERVISOR_STOPPED' and state['cursor'] == 238
    assert not (RUN/'COMPLETE.json').exists() and not (RUN.parent/'paired').exists()
    assert not (HOME/'authorizations/M2B_AUTHORIZATION.json').exists()
    from rc3_relation_drl_20261113.common import verify_binding
    binding = verify_binding(SOURCE/'SCIENTIFIC_SOURCE_BINDING.json')
    import torch
    from rc3_conservative_refinement.runtime import tensor_digest
    from rc3_relation_drl_20261113.runtime import state_digest
    checkpoint_path = ROOT/state['checkpoint']['path']
    assert sha(checkpoint_path) == state['checkpoint']['sha256']
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert saved['cursor'] == 238 and saved['engineering_only'] is False
    assert tensor_digest(saved['model']) == state['checkpoint']['model_sha256']
    assert state_digest(saved['optimizer']) == state['checkpoint']['optimizer_sha256']
    assert state_digest(saved['rng']) == state['checkpoint']['rng_sha256']
    assert all(int(x['step']) == 238 for x in saved['optimizer']['state'].values())
    events = [json.loads(line) for line in (RUN/'EVENTS.jsonl').read_text(encoding='utf8').splitlines()]
    starts = {e['id']:e for e in events if e['event'] == 'begin'}
    completes = {e['id']:e for e in events if e['event'] == 'complete'}
    failed = {e['id']:e for e in events if e['event'] == 'failed'}
    paid_after_checkpoint = sorted(set(starts)-set(saved['request_ids']))
    assert len(paid_after_checkpoint) == 6 and all(x.startswith('init_0239_teacher_') for x in paid_after_checkpoint)
    assert len(failed) == 1
    record_count = 0
    for rid, e in completes.items():
        if e['kind'] != 'request':
            continue
        path = ROOT/e['record']
        assert sha(path) == e['sha256']
        record = read(path)
        assert record['purpose'] == 'teacher_replay' and record['complete'] is True
        assert record['teacher_validation']['valid_model_plan'] is True
        record_count += 1
    assert record_count == 1910
    schedule = read(SOURCE/'M2_DATA_SCHEDULE.json')['init']
    failed_id = next(iter(failed))
    failed_slot = int(failed_id.rsplit('_', 1)[-1])
    detail = dict(request_id=failed_id, scheduled_update=239, scheduled_slot_zero_based=failed_slot,
        task_id=schedule[238]['tasks'][failed_slot], begin=starts[failed_id], complete=completes[failed_id], failed=failed[failed_id],
        persisted_record=read(ROOT/completes[failed_id]['record']),
        interpretation='Replay/independent certificate and request JSON saved; complete event appended; resource atomic replace failed; failed event appended; no optimizer239')
    put(HOME/'FAILED_REQUEST_DETAIL.json', detail)
    result = dict(status='PASS_EVIDENCE_AUDIT_NOT_STAGE_SUCCESS', conclusion='TECHNICAL_INCOMPLETE',
        source_files_unchanged=len(binding['files']), protocol_sha256=sha(SOURCE/'M2_PROTOCOL.json'),
        checkpoint=state['checkpoint'], model_Adam_RNG_verified=True, all_Adam_steps=238,
        checkpoint_request_attempts=saved['counts']['request_attempts'], checkpoint_optimizer_attempts=saved['counts']['optimizer_attempts'],
        saved_request_record_hashes_verified=record_count, saved_teacher_validation_records_valid=record_count,
        paid_after_checkpoint=paid_after_checkpoint, failed_ids=list(failed),
        raw_complete_event_count=1910, raw_failed_event_count=1, exclusive_nonfailed_requests=1909,
        automatic_restore_admissible=False, restore_reason='Six paid request IDs are absent from the last committed checkpoint; exact closed-boundary requirement fails',
        SL512_available=False, M2b_authorization_created=False, M2b_executed=False,
        model_constructions=0, model_forwards=0, optimizer_calls=0, new_H_or_planning_calls=0)
    put(HOME/'M2A_PARTIAL_AUDIT.json', result)
    print(result)


if __name__ == '__main__':
    main()
