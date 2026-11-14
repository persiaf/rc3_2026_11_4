"""Read archived tensors/events/labels only. No Actor, Pool, optimizer or replay."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path

from .persistence import read, sha, digest

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
SOURCE = ROOT / 'rc3_relation_drl_20261113'
OLD_REPORT = ROOT / 'rc3_relation_m2_20261113'
RUN = SOURCE / 'science/relation_navcover_m2_v1_20261113/init'
STOP = '527c31c0911796d01c47723264f97ce80e31b0e0'


def write(name, value):
    (HOME / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def ref(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))


def main():
    # Validate the archived folders against the stop commit, excluding user files.
    subprocess.run(['git', 'diff', '--quiet', STOP, '--', SOURCE.name, OLD_REPORT.name], cwd=ROOT, check=True)
    original_binding = read(SOURCE / 'SCIENTIFIC_SOURCE_BINDING.json')
    files = dict(original_binding['files'])
    for name, expected in files.items():
        assert sha(ROOT / name) == expected, name
    for folder in (RUN, OLD_REPORT):
        for path in folder.rglob('*'):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ('M2_PROTOCOL.json', 'SCIENTIFIC_SOURCE_BINDING.json', 'M2_DATA_SCHEDULE.json'):
        files[(SOURCE / name).relative_to(ROOT).as_posix()] = sha(SOURCE / name)
    state, resource = read(RUN / 'STATE.json'), read(RUN / 'RESOURCE_ACCOUNTING.json')
    checkpoint = state['checkpoint']
    assert sha(ROOT / checkpoint['path']) == checkpoint['sha256']
    import torch
    from rc3_conservative_refinement.runtime import tensor_digest
    from rc3_relation_drl_20261113.runtime import state_digest
    saved = torch.load(ROOT / checkpoint['path'], map_location='cpu', weights_only=False)
    assert saved['engineering_only'] is False and state['cursor'] == saved['cursor'] == 238
    assert saved['identity'] == resource['identity'] == 'M2:relation_navcover_m2_v1_20261113:init'
    assert tensor_digest(saved['model']) == checkpoint['model_sha256'] == saved['model_sha256']
    assert state_digest(saved['optimizer']) == checkpoint['optimizer_sha256'] == saved['optimizer_sha256']
    assert state_digest(saved['rng']) == checkpoint['rng_sha256'] == saved['rng_sha256']
    assert all(int(x['step']) == 238 for x in saved['optimizer']['state'].values())
    assert saved['counts']['request_attempts'] == 1904
    assert saved['counts']['optimizer_completed'] == saved['counts']['optimizer_attempts'] == 238
    assert all('grad' not in name for name in saved)
    events = [json.loads(line) for line in (RUN / 'EVENTS.jsonl').read_text(encoding='utf8').splitlines()]
    starts = [e for e in events if e['event'] == 'begin']
    begins = {e['id']: e for e in starts}
    assert len(starts) == len(begins)
    complete = {e['id']: e for e in events if e['event'] == 'complete'}
    failed = [e for e in events if e['event'] == 'failed']
    assert len(failed) == 1 and failed[0]['id'] == 'init_0239_teacher_5'
    extra = sorted(set(begins) - set(saved['request_ids']))
    assert extra == [f'init_0239_teacher_{slot}' for slot in range(6)]
    assert set(saved['request_ids']) == {f'init_{u:04}_teacher_{s}' for u in range(1, 239) for s in range(8)} | {f'init_{u:04}' for u in range(1, 239)}
    schedule = read(SOURCE / 'M2_DATA_SCHEDULE.json')
    source_refs = {}
    count = 0
    for rid, event in complete.items():
        if event['kind'] != 'request':
            continue
        path = ROOT / event['record']
        assert sha(path) == event['sha256']
        record = read(path)
        assert record['request_id'] == rid and record['complete'] is True
        assert record['teacher_validation']['valid_model_plan'] is True
        assert record['purpose'] == 'teacher_replay'
        teacher_ref = record['teacher_source']
        source_refs[teacher_ref['path']] = teacher_ref['sha256']
        count += 1
    assert count == 1910
    for name, expected in source_refs.items():
        assert sha(ROOT / name) == expected
    assert resource['counts']['request_attempts'] == 1910
    assert resource['counts']['optimizer_attempts'] == resource['counts']['optimizer_completed'] == 238
    assert resource['elapsed_s'] == 2221.312
    remaining = schedule['init'][238:]
    assert [x['update'] for x in remaining] == list(range(239, 513))
    assert all(len(x['tasks']) == 8 for x in remaining)
    recovery_rows = []
    for row in remaining:
        recovery_rows.append(dict(update=row['update'], tasks=row['tasks'],
            optimizer_id=f'init_{row["update"]:04}',
            request_ids=[f'recovery_v1_init_{row["update"]:04}_teacher_{s}' for s in range(8)],
            historical_recompute_ids=extra if row['update'] == 239 else []))
    write('RECOVERY_SCHEDULE.json', dict(schema='M2a-exact-recovery-schedule-v1', status='DRAFT_NOT_AUTHORIZED',
        original_schedule=ref(SOURCE / 'M2_DATA_SCHEDULE.json'), original_slice_digest=digest(remaining),
        order_unchanged=True, rows=recovery_rows, new_request_count=2192, new_optimizer_count=274))
    write('HISTORICAL_BINDING.json', dict(stop_commit=STOP, files=files, read_only=True))
    result = dict(status='PASS_READ_ONLY_RECOVERY_AUDIT', archived_status='TECHNICAL_INCOMPLETE',
        stop_commit=STOP, protocol=ref(SOURCE / 'M2_PROTOCOL.json'), source_binding=ref(SOURCE / 'SCIENTIFIC_SOURCE_BINDING.json'),
        original_stage_authorization=ref(OLD_REPORT / 'authorizations/M2A_AUTHORIZATION.json'),
        state=ref(RUN / 'STATE.json'), journal=ref(RUN / 'EVENTS.jsonl'), resource=ref(RUN / 'RESOURCE_ACCOUNTING.json'),
        checkpoint=checkpoint, checkpoint_model_Adam_RNG_verified=True, all_Adam_steps=238,
        optimizer_param_groups=saved['optimizer']['param_groups'], rng_components=list(saved['rng']),
        checkpoint_counts=saved['counts'], checkpoint_id_count=len(saved['request_ids']),
        original_counts=resource['counts'], original_unique_request_debits=1910,
        committed_teacher_debits=1904, paid_after_checkpoint=extra,
        overlapping_complete_failed_id=failed[0]['id'], raw_complete_events=1910, raw_failed_events=1,
        exclusive_nonfailed_complete=1909, requests_sha_verified=count, unique_teacher_source_files_verified=len(source_refs),
        new_teacher_replays=0, teacher_validation_reexecutions=0,
        batch239_tasks=remaining[0]['tasks'], no_saved_gradients=True,
        graph_reuse_possible=False, old_checkpoint_direct_restore_allowed=False,
        restore_feasibility='FEASIBLE_WITH_EXPLICIT_NEW_RECOVERY_APPROVAL_AND_FULL_BATCH239_RECOMPUTE',
        historical_wall_s=2221.312, remaining_wall_s=12178.688, total_wall_cap_s=14400,
        old_request_remaining=2186, new_requests_needed=2192, technical_recompute_increment=6,
        revised_total_request_cap=4102, remaining_optimizer_calls=274, total_optimizer_cap=512,
        scientific_optimizer_calls=0, model_constructions=0, model_forwards=0, DRL_trajectories=0, new_parent_tests=0)
    write('SL238_RECOVERY_AUDIT.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'requests_sha_verified', 'batch239_tasks', 'remaining_wall_s', 'revised_total_request_cap')}))


if __name__ == '__main__':
    main()
