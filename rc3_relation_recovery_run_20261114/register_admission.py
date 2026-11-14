"""Read-only admission audit followed by separate explicit user authorization receipts."""
import datetime
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
BASE = 'd7b76b5e4329c1ca1aaa41f51c8d90a4237a4306'
PROPOSAL = ROOT / 'rc3_relation_recovery_20261114/RECOVERY_PROTOCOL_DRAFT.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError('Do not overwrite admission/authorization evidence: ' + str(path))
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode())


def main():
    start = time.perf_counter()
    from rc3_relation_recovery_20261114.persistence import read, sha, digest
    from rc3_relation_recovery_20261114.recovery_adapter import checked_ref
    proposal = read(PROPOSAL)
    assert sha(PROPOSAL) == 'e1f0cad93b5bf639686e6824f1a92153a0628c1791aaa7c98778ab572e76dc1c'
    assert digest(proposal) == '40a01bf510b3addf474f428708b6c8ee93cf9f53ef30f1ba444afd8f5b0b380c'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == BASE
    names = {}
    for key in ('original_source_binding', 'historical_binding', 'new_source_binding'):
        content = read(checked_ref(proposal[key]))
        for name, expected in content['files'].items():
            if name in names:
                assert names[name] == expected
            names[name] = expected
    for name, expected in names.items():
        assert sha(ROOT / name) == expected, name
    for key in ('original_protocol', 'original_stage_authorization', 'sl238_audit', 'recovery_schedule'):
        checked_ref(proposal[key])
    audit = read(checked_ref(proposal['sl238_audit']))
    resource = read(checked_ref(audit['resource']))
    state = read(checked_ref(audit['state']))
    events = [json.loads(s) for s in checked_ref(audit['journal']).read_text(encoding='utf8').splitlines()]
    begin = [e for e in events if e['event'] == 'begin']
    assert len({e['id'] for e in begin}) == len(begin) == 2148
    assert sum(e['kind'] == 'request' for e in begin) == 1910
    assert sum(e['kind'] == 'optimizer' for e in begin) == 238
    assert resource['counts'] == audit['original_counts']
    assert resource['elapsed_s'] == 2221.312 and state['cursor'] == 238
    import torch
    from rc3_conservative_refinement.runtime import tensor_digest
    from rc3_relation_drl_20261113.runtime import state_digest
    saved = torch.load(checked_ref(proposal['checkpoint']), map_location='cpu', weights_only=False)
    assert saved['cursor'] == 238 and saved['engineering_only'] is False
    assert tensor_digest(saved['model']) == proposal['checkpoint']['model_sha256']
    assert state_digest(saved['optimizer']) == proposal['checkpoint']['optimizer_sha256']
    assert state_digest(saved['rng']) == proposal['checkpoint']['rng_sha256']
    assert all(int(v['step']) == 238 for v in saved['optimizer']['state'].values())
    assert saved['counts']['request_attempts'] == 1904
    assert sorted({e['id'] for e in begin} - set(saved['request_ids'])) == proposal['historical_paid_uncommitted']
    schedule = read(checked_ref(proposal['recovery_schedule']))
    original = read(checked_ref(proposal['original_schedule']))['init'][238:]
    assert [(r['update'], r['tasks']) for r in schedule['rows']] == [(r['update'], r['tasks']) for r in original]
    assert len(schedule['rows']) == 274
    ids = [i for row in schedule['rows'] for i in row['request_ids']]
    assert len(ids) == len(set(ids)) == 2192
    assert not set(ids) & {e['id'] for e in begin}
    for row in schedule['rows']:
        assert row['optimizer_id'] == f'init_{row["update"]:04}'
        assert row['request_ids'] == [f'recovery_v1_init_{row["update"]:04}_teacher_{i}' for i in range(8)]
    assert not (ROOT / proposal['canonical_recovery_folder']).exists()
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result = dict(status='GO_EXACT_AUTHORIZED_M2A_RECOVERY', utc=stamp, approved_base_commit=BASE,
        proposal_file_sha256=sha(PROPOSAL), proposal_canonical_sha256=digest(proposal),
        unique_bound_files_verified=len(names), model_Adam_RNG_verified=True, Adam_step=238,
        checkpoint_requests=1904, historical_requests=1910, historical_optimizer_calls=238,
        historical_paid_uncommitted=proposal['historical_paid_uncommitted'],
        scheduled_new_requests=2192, scheduled_new_updates=274,
        cumulative_limits=proposal['cumulative_limits'], remaining_limits=proposal['remaining_limits'],
        stage_execution_started=False, read_only_CPU_tensor_loads=1,
        model_constructions=0, model_forwards=0, optimizer_calls=0, teacher_replays=0,
        admission_wall_s=time.perf_counter() - start)
    write(HOME / 'ADMISSION_CHECK.json', result)
    approval = dict(schema='M2a-recovery-protocol-approval-v1', status='APPROVED',
        approval_id='M2A_RECOVERY_PROTOCOL_USER_APPROVAL_20261114', utc=stamp,
        authority='Explicit user M2a SL238 controlled recovery authorization in this chat',
        approved_base_commit=BASE, scientific_id=proposal['scientific_id'], recovery_id=proposal['recovery_id'],
        proposal=dict(path=PROPOSAL.relative_to(ROOT).as_posix(), sha256=sha(PROPOSAL)),
        proposal_sha256=digest(proposal), hash_algorithm='SHA256 canonical JSON: sort_keys, compact separators, allow_nan=False',
        new_source_binding=proposal['new_source_binding'], original_stage_authorization=proposal['original_stage_authorization'],
        cumulative_limits=proposal['cumulative_limits'], remaining_limits=proposal['remaining_limits'],
        original_protocol_bytes_unchanged=True, M2b_authorized=False)
    approval_path = HOME / 'authorizations/PROTOCOL_APPROVAL.json'
    write(approval_path, approval)
    authorization = dict(schema='M2a-recovery-stage-authorization-v1', status='APPROVED', stage='M2a_recovery',
        stage_authorization_id='M2A_RECOVERY_USER_STAGE_20261114_ONCE', utc=stamp,
        authority='Explicit user authorization; one recovery attempt only',
        approved_base_commit=BASE, proposal_sha256=digest(proposal), scientific_id=proposal['scientific_id'],
        recovery_id=proposal['recovery_id'], original_stage_authorization=proposal['original_stage_authorization'],
        protocol_approval=dict(path=approval_path.relative_to(ROOT).as_posix(), sha256=sha(approval_path)),
        canonical_folder=proposal['canonical_recovery_folder'], checkpoint=proposal['checkpoint'],
        cumulative_limits=proposal['cumulative_limits'], remaining_limits=proposal['remaining_limits'],
        attempt_limit=1, recompute_batch239_in_full=True, automatic_compute_retry=False,
        M2b_authorized=False, M2b_admission_review_after_success_only=True)
    auth_path = HOME / 'authorizations/M2A_RECOVERY_AUTHORIZATION.json'
    write(auth_path, authorization)
    from rc3_relation_recovery_20261114.recovery_runner import admit
    admit(PROPOSAL, approval_path, auth_path)
    write(HOME / 'CURRENT_STAGE_STATUS.json', dict(status='AUTHORIZED_ADMITTED_NOT_STARTED',
        scientific_id=proposal['scientific_id'], recovery_id=proposal['recovery_id'],
        original_science='TECHNICAL_INCOMPLETE', proposal_bytes_unchanged=True,
        protocol_approval_sha256=sha(approval_path), stage_authorization_sha256=sha(auth_path),
        M2b_authorized=False, registration_remote_verified=False))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
