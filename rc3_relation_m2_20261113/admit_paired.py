"""Verify real SL512 completion before recording conditional M2b authorization.

Reads saved tensors on CPU; no model construction/forward, optimizer or query.
Does not launch a stage. The frozen M2 runner performs its own admission.
"""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
SOURCE = ROOT/'rc3_relation_drl_20261113'
ID = 'relation_navcover_m2_v1_20261113'
INIT = SOURCE/'science'/ID/'init'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))


def check_ref(value):
    path = ROOT/value['path']
    if sha(path) != value['sha256']:
        raise ValueError('Saved completed artifact changed: '+value['path'])
    return path


def put_new(path, value):
    if path.exists():
        raise RuntimeError('Authorization/admission already exists; never replace original stage authorization')
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())


def main():
    if (HOME/'authorizations/M2B_AUTHORIZATION.json').exists():
        raise RuntimeError('M2b authorization already registered')
    protocol = read(SOURCE/'M2_PROTOCOL.json')
    protocol_sha = sha(SOURCE/'M2_PROTOCOL.json')
    approval_path = HOME/'authorizations/PROTOCOL_APPROVAL.json'
    approval = read(approval_path)
    assert approval['protocol_sha256'] == protocol_sha and approval['scientific_id'] == ID
    complete_path = INIT/'COMPLETE.json'
    complete = read(complete_path)
    resource = read(INIT/'RESOURCE_ACCOUNTING.json')
    state = read(INIT/'STATE.json')
    assert complete['status'] == resource['status'] == 'COMPLETE'
    assert complete['scientific_id'] == ID and complete['stage'] == 'init'
    assert complete['protocol_sha256'] == protocol_sha and complete['engineering_only'] is False
    assert state['cursor'] == complete['cursor'] == 512 and state['next_action'] == 'COMPLETE'
    assert not resource['pending'] and resource['elapsed_s'] <= protocol['limits']['init']['wall_s']
    for key, count in dict(request_attempts=4096, request_completed=4096, optimizer_attempts=512, optimizer_completed=512).items():
        assert resource['counts'][key] == complete['actual_counts'][key] == count
    assert resource['counts'].get('request_failed', 0) == resource['counts'].get('optimizer_failed', 0) == 0
    controls = list((INIT/'controls').glob('attempt_*/SUPERVISOR_RECEIPT.json'))
    assert controls and all(read(p)['status'] == 'COMPLETE' for p in controls)
    assert state['checkpoint'] == complete['checkpoint']
    import torch
    from rc3_relation_drl_20261113.runtime import state_digest
    from rc3_conservative_refinement.runtime import tensor_digest
    checkpoint = torch.load(check_ref(complete['checkpoint']), map_location='cpu', weights_only=False)
    export = torch.load(check_ref(complete['export']), map_location='cpu', weights_only=False)
    assert checkpoint['identity'] == 'M2:'+ID+':init' and checkpoint['engineering_only'] is False
    assert checkpoint['cursor'] == 512 and export['identity'] == ID and export['engineering_only'] is False
    expected_model = complete['checkpoint']['model_sha256']
    assert tensor_digest(checkpoint['model']) == tensor_digest(export['model']) == export['model_sha256'] == expected_model
    assert state_digest(checkpoint['optimizer']) == checkpoint['optimizer_sha256'] == complete['checkpoint']['optimizer_sha256']
    assert state_digest(checkpoint['rng']) == checkpoint['rng_sha256'] == complete['checkpoint']['rng_sha256']
    assert checkpoint['config'] == export['config'] == read(SOURCE/'M2_RANDOM_INIT.json')['config']
    events = [json.loads(line) for line in (INIT/'EVENTS.jsonl').read_text(encoding='utf8').splitlines()]
    begins = [e['id'] for e in events if e['event'] == 'begin']
    assert len(begins) == len(set(begins)) == 4608 and set(begins) == set(checkpoint['request_ids'])
    assert not any(e['event'] == 'failed' for e in events)
    states = checkpoint['optimizer']['state']
    assert len(states) == len(checkpoint['model'])
    assert all(int(s['step']) == 512 for s in states.values())
    assert all(torch.isfinite(t).all() for s in states.values() for t in s.values() if torch.is_tensor(t))
    audit = dict(status='PASS', scientific_id=ID, init_receipt=ref(complete_path), checkpoint=complete['checkpoint'],
        export=complete['export'], model_optimizer_RNG_hashes_verified=True, all_Adam_steps=512,
        exact_closed_debit_ids=4608, resource=ref(INIT/'RESOURCE_ACCOUNTING.json'), source_protocol_sha256=protocol_sha,
        model_constructions=0, model_forwards=0, optimizer_calls=0, complete_requests=0)
    put_new(HOME/'M2A_COMPLETION_AUDIT.json', audit)
    auth = dict(schema='relation-M2-stage-authorization-v1', status='APPROVED', stage='paired',
        stage_authorization_id=ID+'_M2b_conditional_user_20261113', scientific_id=ID,
        protocol_sha256=protocol_sha, execution_source_binding_sha256=protocol['source_binding']['sha256'],
        protocol_approval=ref(approval_path), init_receipt=ref(complete_path), upstream_audit=ref(HOME/'M2A_COMPLETION_AUDIT.json'),
        user_authorization=ref(HOME/'USER_AUTHORIZATION.md'), recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authority='Explicit user M2b conditional authorization; actual complete SL512 condition is now verified',
        limits=protocol['limits']['paired'], no_additional_seed_ablation_or_test=True)
    put_new(HOME/'authorizations/M2B_AUTHORIZATION.json', auth)
    print(dict(status='GO_PAIRED', init_model_sha256=expected_model, init_receipt=auth['init_receipt'],
               paired_authorization=ref(HOME/'authorizations/M2B_AUTHORIZATION.json')))


if __name__ == '__main__':
    main()
