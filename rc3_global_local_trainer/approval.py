"""Fail-closed admission for formal science. This module grants no approval."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / 'rc3_global_local/S2_S3_protocol_v2.json'
V3 = ROOT / 'rc3_global_local_impl_review/PROPOSED_PROTOCOL_V3.json'
REGISTRY_V2 = ROOT / 'rc3_global_local/future_seeds_v2.json'
REGISTRY_V3 = ROOT / 'rc3_global_local_impl_review/PROPOSED_SEEDS_V3.json'


class AdmissionError(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def source_binding():
    paths = sorted((ROOT / 'rc3_global_local_trainer').glob('*.py'))
    paths = [p for p in paths if not p.name.startswith('test_')]
    paths += [ROOT / p for p in (
        'rc3_global_local/policy.py', 'rc3_global_local/environment.py',
        'rc3_global_local/problem.py', 'rc3_global_local/terminal.py',
        'rc3_global_local/validation.py', 'rc3_global_local/run_s0.py',
        'rc3_global_local_impl_review/runtime_guards.py',
        'rc3_global_local_impl_review/endpoint_baselines.py',
        'rc3_constructive/baselines.py', 'rc3_constructive/problem.py',
        'rc3_fast/scenes.py', 'rc3_fast/planner.py', 'rc3_fast/solver.py',
        'rc3_baseline/baseline.py', 'rc3_baseline/validator.py',
        'rc3_v4/runtime.py', 'rc3_v3/environment.py', 'rc3_improve/repair.py')]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


@dataclass(frozen=True)
class ExecutionPermit:
    protocol_path: Path
    registry_path: Path
    protocol_sha256: str
    registry_sha256: str
    binding: dict
    stages: tuple
    approval_sha256: str
    authorization_sha256: str


def require_authority(protocol_path, registry_path, approval_path, authorization_path, stage):
    """Require two user-origin receipts plus a source-bound engineering PASS.

    These JSON records are auditable declarations, not a signature system. There
    is intentionally no CLI command that manufactures either user declaration.
    Engineering fixtures call low-level APIs under their separate registration;
    they cannot be promoted into a formal run by passing a boolean flag.
    """
    if stage not in ('teachers', 'S1', 'S2', 'all'):
        raise AdmissionError('Only teachers/S1/S2 are supported; S3 is sealed')
    pp, rp = Path(protocol_path).resolve(), Path(registry_path).resolve()
    if pp == V2.resolve() or digest(pp) == digest(V2):
        raise AdmissionError('Frozen v2 lacks a complete execution recipe. Approve an explicit new protocol/addendum first.')
    if pp != V3.resolve() or rp != REGISTRY_V3.resolve():
        raise AdmissionError('This implementation is bound to the explicit v3 proposal only; other methods need a new reviewed adapter')
    if not approval_path or not authorization_path:
        raise AdmissionError('Separate protocol approval and formal execution authorization receipts are required; v3 is not active by default')
    ap, ep = Path(approval_path).resolve(), Path(authorization_path).resolve()
    if ap == ep:
        raise AdmissionError('Protocol approval and execution authorization must be separate records')
    a, e = read(ap), read(ep)
    ph, rh, sources = digest(pp), digest(rp), source_binding()
    if a.get('status') != 'APPROVED' or a.get('approved_by') != 'user' or not a.get('authorization_text'):
        raise AdmissionError('Explicit user protocol approval is missing')
    if e.get('status') != 'AUTHORIZED' or e.get('authorized_by') != 'user' or not e.get('authorization_text'):
        raise AdmissionError('Explicit user formal execution authorization is missing')
    for receipt in (a, e):
        if receipt.get('protocol_sha256') != ph or receipt.get('registry_sha256') != rh:
            raise AdmissionError('Protocol/registry approval hash mismatch')
        if receipt.get('source_sha256') != sources:
            raise AdmissionError('Source binding mismatch; re-review changed implementation')
    if e.get('protocol_approval_sha256') != digest(ap):
        raise AdmissionError('Execution authorization does not reference this protocol approval')
    stages = tuple(e.get('stages', ()))
    required = {'teachers', 'S1', 'S2'} if stage == 'all' else {stage}
    if not required.issubset(stages) or 'S3' in stages:
        raise AdmissionError('Execution stage scope mismatch; S3 is not an allowed scope')
    vp = Path(e.get('trainer_verification_path', '')).resolve()
    if not vp.is_file() or digest(vp) != e.get('trainer_verification_sha256'):
        raise AdmissionError('A completed source-bound trainer verification receipt is required')
    verification = read(vp)
    if verification.get('status') != 'PASS' or verification.get('source_sha256') != sources:
        raise AdmissionError('Trainer acceptance is missing or stale')
    p = read(pp)
    if p.get('seeds_sha256') != rh:
        raise AdmissionError('Protocol seed hash mismatch')
    h = read(ROOT / 'rc3_global_local/S0_protocol.json')['H_sha256']
    binding = dict(protocol_sha256=ph, registry_sha256=rh, source_sha256=sources,
                   H_sha256=h, approval_sha256=digest(ap), authorization_sha256=digest(ep))
    return ExecutionPermit(pp, rp, ph, rh, binding, stages, digest(ap), digest(ep))


def output_directory(path, *, resume=False):
    target = Path(path).resolve()
    allowed = (ROOT / 'rc3_global_local_trainer/runs').resolve()
    if target == allowed or not target.is_relative_to(allowed):
        raise AdmissionError('Formal output must be a new child of rc3_global_local_trainer/runs')
    if target.exists() and not resume:
        raise AdmissionError('Output exists; no overwrite or implicit resume')
    if resume and not target.is_dir():
        raise AdmissionError('Resume output does not exist')
    return target
