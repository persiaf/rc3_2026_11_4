"""Register the user's explicit M2 approval. Metadata only; no scientific call."""
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(__file__).resolve().parent
SOURCE = ROOT/'rc3_relation_drl_20261113'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def put(path, obj):
    path = Path(path)
    if path.exists():
        raise RuntimeError('Existing authorization/review evidence must not be overwritten: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2)+'\n').encode())


def ref(path):
    path = Path(path)
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))


def main():
    base = '9c878928e7f619bce25ddd00792757431ca705a1'
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip() != base:
        raise ValueError('Admission must start from the user approved exact base')
    protocol = read(SOURCE/'M2_PROTOCOL.json')
    expected = '7ca20d48177b156ef4183f8d58f8354f20198513ecad66e161ff4d0b5e41a445'
    if sha(SOURCE/'M2_PROTOCOL.json') != expected:
        raise ValueError('Frozen protocol bytes differ from reviewed base')
    binding = read(ROOT/protocol['source_binding']['path'])
    if sha(ROOT/protocol['source_binding']['path']) != protocol['source_binding']['sha256']:
        raise ValueError('Scientific source binding mismatch')
    for name, expected_sha in binding['files'].items():
        if sha(ROOT/name) != expected_sha:
            raise ValueError('Frozen source/data/model identity changed: '+name)
    identity = protocol['scientific_id']
    if identity != 'relation_navcover_m2_v1_20261113':
        raise ValueError('Wrong scientific identity')
    if (SOURCE/'science'/identity).exists():
        raise RuntimeError('Existing stage state: inspect and retain original authorization instead of registering a new run')
    utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    approval = dict(schema='relation-M2-protocol-approval-v1', status='APPROVED', scientific_id=identity,
        protocol_sha256=expected, protocol=ref(SOURCE/'M2_PROTOCOL.json'),
        execution_source_binding_sha256=protocol['source_binding']['sha256'],
        base_commit=base, recorded_utc=utc, authority='Explicit user message: 研究内容三 M2正式科学实验授权',
        user_authorization=ref(HOME/'USER_AUTHORIZATION.md'),
        scope='Exact frozen M2 only; no protocol/source/identity expansion')
    approval_path = HOME/'authorizations/PROTOCOL_APPROVAL.json'
    put(approval_path, approval)
    auth = dict(schema='relation-M2-stage-authorization-v1', status='APPROVED', stage='init',
        stage_authorization_id='relation_navcover_m2_v1_20261113_M2a_user_20261113', scientific_id=identity,
        protocol_sha256=expected, execution_source_binding_sha256=protocol['source_binding']['sha256'],
        protocol_approval=ref(approval_path), base_commit=base, recorded_utc=utc,
        authority='Explicit user M2a authorization; original receipt required for any same-stage continuation',
        user_authorization=ref(HOME/'USER_AUTHORIZATION.md'), limits=protocol['limits']['init'],
        random_initialization=protocol['initialization']['random_export'], schedule=protocol['schedule'],
        failure_action='Stop according to frozen protocol; no sample/teacher replacement or automatic retry')
    put(HOME/'authorizations/M2A_AUTHORIZATION.json', auth)
    put(HOME/'ADMISSION_CHECK.json', dict(status='GO', base_commit=base, scientific_id=identity,
        protocol_sha256=expected, scientific_bound_files=len(binding['files']), all_bound_bytes_match=True,
        previous_scientific_state_absent=True, approval_receipt=ref(approval_path),
        init_authorization=ref(HOME/'authorizations/M2A_AUTHORIZATION.json'),
        paired_authorization='CONDITIONAL_NOT_CREATED: requires actual complete valid SL512 receipt',
        model_forwards=0, optimizer_calls=0, complete_requests=0, old_active_pointers_unchanged=True))
    print(dict(status='GO', source_files=len(binding['files']), protocol_sha256=expected,
               init_authorization=ref(HOME/'authorizations/M2A_AUTHORIZATION.json')))


if __name__ == '__main__':
    main()
