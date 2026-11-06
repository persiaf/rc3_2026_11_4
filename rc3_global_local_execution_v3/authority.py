"""Separate stable scientific lineage from immutable per-stage authorization.

Imports grant no authority. Approval records are auditable user declarations,
not cryptographic signatures. The CLI does not manufacture such declarations.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import re
import platform
import sys

from rc3_global_local_trainer.approval import (
    ROOT, AdmissionError, ExecutionPermit, digest, read,
    source_binding as trainer_source_binding)
from rc3_global_local_trainer.data import write_new

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/'V3_EXECUTION_CANDIDATE.json'
REGISTRY = HERE/'REGISTRY.json'
SOURCE_BINDING = HERE/'EXECUTION_SOURCE_BINDING.json'
STAGES = ('S1-A', 'S1-B', 'S2')
SEEDS = (11622026, 11622027, 11622028)
LEGACY_STAGES = {'S1-A': 'teachers', 'S1-B': 'S1', 'S2': 'S2'}


def source_binding():
    values = trainer_source_binding()
    for path in sorted(HERE.glob('*.py')):
        if not path.name.startswith('test_'):
            values[path.relative_to(ROOT).as_posix()] = digest(path)
    return dict(sorted(values.items()))


def environment_current():
    import numpy
    import torch
    return dict(python_version=platform.python_version(),python_implementation=sys.implementation.name,
        torch_version=torch.__version__,numpy_version=numpy.__version__,device='cpu',
        torch_threads=2,threadpool_limit=1,deterministic_algorithms=True)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')).hexdigest()


def science_binding(protocol_sha256, registry_sha256, sources, h_sha256,
                    experiment_id, approval_sha256, source_manifest_sha256):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', str(experiment_id)):
        raise AdmissionError('Experiment ID must be a fixed safe identifier')
    return dict(schema='rc3-stable-scientific-lineage-v1',
        protocol_sha256=protocol_sha256, registry_sha256=registry_sha256,
        source_sha256=copy.deepcopy(sources), H_sha256=h_sha256,
        experiment_id=str(experiment_id), approval_sha256=approval_sha256,
        source_manifest_sha256=source_manifest_sha256)


@dataclass(frozen=True)
class StagePermit:
    execution: ExecutionPermit
    stage: str
    seed: object
    experiment_id: str
    authorization_path: Path
    authorization: dict
    predecessor_hashes: dict

    @property
    def binding(self):
        return self.execution.binding

    @property
    def authorization_sha256(self):
        return self.execution.authorization_sha256


def _project_file(path, label):
    p = Path(path).resolve()
    if not p.is_file() or not p.is_relative_to(ROOT):
        raise AdmissionError(label+' must be an existing file inside this project')
    return p


def require_stage_authority(protocol_path, registry_path, approval_path,
                            authorization_path, stage, seed, experiment_id):
    if stage not in STAGES:
        raise AdmissionError('Only S1-A/S1-B/S2 are supported; S3 remains sealed')
    if (stage == 'S1-A' and seed is not None) or (stage != 'S1-A' and (type(seed) is not int or seed not in SEEDS)):
        raise AdmissionError('S1-A has no optimizer seed; S1-B/S2 require one fixed declared seed')
    pp, rp = Path(protocol_path).resolve(), Path(registry_path).resolve()
    if pp != PROTOCOL.resolve() or rp != REGISTRY.resolve():
        raise AdmissionError('Only the new explicitly reviewed candidate/registry pair is supported')
    if not approval_path or not authorization_path:
        raise AdmissionError('User protocol approval and corresponding stage authorization are both required')
    ap, ep = _project_file(approval_path, 'Approval'), _project_file(authorization_path, 'Authorization')
    if ap == ep:
        raise AdmissionError('Protocol approval and stage authorization must be different immutable records')
    a, e = read(ap), read(ep)
    sources, ph, rh = source_binding(), digest(pp), digest(rp)
    h = read(ROOT/'rc3_global_local/S0_protocol.json')['H_sha256']
    manifest = read(SOURCE_BINDING)
    if (manifest.get('protocol_sha256') != ph or manifest.get('registry_sha256') != rh
            or manifest.get('source_sha256') != sources or manifest.get('H_sha256') != h):
        raise AdmissionError('Execution source manifest is absent or stale')
    hb=read(HERE/'H_BINDING.json');hs=hb['H_source_sha256']
    if (any(digest(ROOT/n)!=v for n,v in hs.items()) or canonical_hash(hs)!=h
            or manifest.get('H_source_sha256')!=hs):
        raise AdmissionError('Frozen H source bytes or canonical identity differs')
    env=environment_current()
    if manifest.get('environment')!=env or manifest.get('environment_sha256')!=canonical_hash(env):
        raise AdmissionError('Execution dependency/device/precision environment differs from its binding')
    verification_record=_project_file(manifest.get('engineering_verification_path',''),'Manifest engineering verification')
    if digest(verification_record)!=manifest.get('engineering_verification_sha256'):
        raise AdmissionError('Source manifest engineering verification bytes differ')
    manifest_verification=read(verification_record)
    if manifest_verification.get('status')!='PASS' or manifest_verification.get('source_sha256')!=sources:
        raise AdmissionError('Source manifest engineering verification is stale')
    for receipt in (a, e):
        if receipt.get('engineering_fixture') or receipt.get('non_executable_fixture'):
            raise AdmissionError('Engineering authorization fixtures cannot enter formal execution')
        if (receipt.get('protocol_sha256') != ph or receipt.get('registry_sha256') != rh
                or receipt.get('source_sha256') != sources or receipt.get('H_sha256') != h
                or receipt.get('source_manifest_sha256') != digest(SOURCE_BINDING)):
            raise AdmissionError('Protocol/registry/source/H/manifest approval binding mismatch')
    if a.get('status') != 'APPROVED' or a.get('approved_by') != 'user' or not a.get('authorization_text'):
        raise AdmissionError('Explicit user protocol approval is missing')
    if e.get('status') != 'AUTHORIZED' or e.get('authorized_by') != 'user' or not e.get('authorization_text'):
        raise AdmissionError('Explicit user stage authorization is missing')
    if e.get('protocol_approval_sha256') != digest(ap) or e.get('experiment_id') != experiment_id:
        raise AdmissionError('Stage authorization references a different approval or experiment')
    stages = e.get('stages', [])
    if not isinstance(stages, list) or stage not in stages or any(s not in STAGES for s in stages):
        raise AdmissionError('Stage authorization scope is invalid or contains sealed stages')
    seeds = e.get('optimizer_seeds', [])
    if (not isinstance(seeds, list) or len(set(seeds)) != len(seeds)
            or any(type(s) is not int or s not in SEEDS for s in seeds)
            or (seed is not None and seed not in seeds)):
        raise AdmissionError('Optimizer seed is outside the explicit authorization scope')
    verification = _project_file(e.get('admission_verification_path', ''), 'Admission verification')
    if digest(verification) != e.get('admission_verification_sha256'):
        raise AdmissionError('Admission verification bytes differ')
    v = read(verification)
    if v.get('status') != 'PASS' or v.get('source_sha256') != sources:
        raise AdmissionError('Current source-bound engineering admission PASS is required')
    if read(pp).get('seeds_sha256') != rh:
        raise AdmissionError('Protocol does not bind the current full registry')
    predecessors = e.get('predecessors', {})
    if not isinstance(predecessors, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in predecessors.items()):
        raise AdmissionError('Explicit predecessor receipt hashes are required')
    binding = science_binding(ph, rh, sources, h, experiment_id, digest(ap), digest(SOURCE_BINDING))
    execution = ExecutionPermit(pp, rp, ph, rh, binding, (LEGACY_STAGES[stage],), digest(ap), digest(ep))
    return StagePermit(execution, stage, seed, experiment_id, ep, copy.deepcopy(e), copy.deepcopy(predecessors))


def output_directory(path, experiment_id, *, existing=False):
    target = Path(path).resolve()
    allowed = (HERE/'runs').resolve()
    if target != (allowed/experiment_id).resolve():
        raise AdmissionError('Output must be the fixed experiment_id child under new execution runs/')
    if target.exists() != bool(existing):
        raise AdmissionError('Explicit existing-run choice must match directory existence')
    return target


def stage_key(stage, seed=None):
    if stage not in STAGES or (stage == 'S1-A' and seed is not None):
        raise AdmissionError('Invalid stage/seed key')
    if stage != 'S1-A' and (type(seed) is not int or seed not in SEEDS):
        raise AdmissionError('Invalid stage optimizer seed')
    return stage if seed is None else stage+'/'+str(seed)


class StageChain:
    """Append-only authorization state; no model, oracle or scene operations."""
    def __init__(self, folder, binding):
        self.folder, self.binding = Path(folder).resolve(), copy.deepcopy(binding)
        self.root = self.folder/'stage_admission'

    def path(self, key, name):
        parts = key.split('/')
        if len(parts) not in (1,2) or parts[0] not in STAGES:
            raise AdmissionError('Invalid admission key')
        if (parts[0]=='S1-A')!=(len(parts)==1) or Path(name).name!=name or name in ('.','..'):
            raise AdmissionError('Invalid stage key or record filename')
        if len(parts) == 2 and (not parts[1].isdigit() or int(parts[1]) not in SEEDS):
            raise AdmissionError('Invalid admission seed key')
        return self.root.joinpath(*parts)/name

    def required_predecessors(self, stage, seed):
        result = []
        if stage != 'S1-A': result.append('S1-A')
        if stage == 'S2': result += [stage_key('S1-B', s) for s in SEEDS]
        if seed is not None:
            result += [stage_key(stage,s) for s in SEEDS[:SEEDS.index(seed)]]
        return list(dict.fromkeys(result))

    def completed(self, key):
        f = self.path(key,'COMPLETE.json')
        if not f.is_file(): raise AdmissionError('Required predecessor is incomplete: '+key)
        value = read(f)
        if value.get('status') != 'COMPLETE' or value.get('science_binding') != self.binding:
            raise AdmissionError('Predecessor scientific identity or status differs: '+key)
        if value.get('stage_key') != key:
            raise AdmissionError('Predecessor stage identity differs')
        parts=key.split('/')
        stage,seed=parts[0],None if len(parts)==1 else int(parts[1])
        required=set(self.required_predecessors(stage,seed))
        recorded=value.get('predecessors')
        if not isinstance(recorded,dict) or set(recorded)!=required:
            raise AdmissionError('Completed stage predecessor set differs from its fixed chain')
        for parent_key,expected_hash in recorded.items():
            parent=self.path(parent_key,'COMPLETE.json')
            if not parent.is_file() or digest(parent)!=expected_hash:
                raise AdmissionError('Completed stage points to a changed predecessor receipt: '+parent_key)
        auth = self.path(key,'AUTHORIZATION.json')
        start = self.path(key,'START.json')
        if (not auth.is_file() or not start.is_file() or value.get('authorization_record_sha256') != digest(auth)
                or value.get('start_record_sha256') != digest(start)):
            raise AdmissionError('Predecessor authorization/start record differs')
        for rel, expected in value.get('artifacts', {}).items():
            p = (self.folder/rel).resolve()
            if not p.is_relative_to(self.folder) or not p.is_file() or digest(p) != expected:
                raise AdmissionError('Predecessor artifact missing or changed: '+rel)
        return value, digest(f)

    def admit(self, stage, seed, authorization_sha256, predecessors, *, resume=False):
        key = stage_key(stage, seed)
        if self.path(key,'COMPLETE.json').exists():
            raise AdmissionError('Stage already complete; a new authorization cannot repeat it')
        required = self.required_predecessors(stage, seed)
        observed = {k:self.completed(k)[1] for k in required}
        # User authorization binds existing upstream stages. Earlier seeds of
        # this same already-authorized stage are deterministic sequence gates;
        # their as-yet unknown future hashes are chained automatically at start.
        upstream={k:v for k,v in observed.items() if k.split('/')[0]!=stage}
        if predecessors != upstream:
            raise AdmissionError('Explicit authorization predecessor set/hashes differ')
        auth = self.path(key,'AUTHORIZATION.json'); start = self.path(key,'START.json')
        if auth.exists():
            saved = read(auth)
            if saved.get('science_binding') != self.binding or saved.get('authorization_sha256') != authorization_sha256:
                raise AdmissionError('Same stage cannot switch authorization or reset its scientific binding')
            if saved.get('predecessors') != observed:
                raise AdmissionError('Same-stage parent lineage changed')
            if not resume or not start.exists() or self.path(key,'FAILED.json').exists():
                raise AdmissionError('Existing/failed stage cannot silently restart; explicit safe resume required')
        else:
            if resume:
                raise AdmissionError('No admitted stage exists to resume')
            if start.exists():
                raise AdmissionError('Partial admission record; no implicit restart')
            write_new(auth, dict(schema='rc3-stage-authorization-record-v1', stage_key=key,
                science_binding=self.binding, authorization_sha256=authorization_sha256, predecessors=observed))
            write_new(start, dict(schema='rc3-stage-start-v1', stage_key=key,
                science_binding=self.binding, authorization_record_sha256=digest(auth),
                budget_reset_allowed=False, scope='Independent stage; actual cumulative resources stored at update boundaries and completion/failure.'))
        return dict(key=key, predecessors=observed, authorization_record_sha256=digest(auth), start_record_sha256=digest(start))

    def close(self, admission, artifacts, resources, *, complete, error=None):
        key = admission['key']; dest = self.path(key,'COMPLETE.json' if complete else 'FAILED.json')
        if dest.exists(): raise AdmissionError('Stage terminal record cannot be overwritten')
        checked = {}
        for path in artifacts:
            p = Path(path).resolve()
            if not p.is_relative_to(self.folder) or not p.is_file():
                raise AdmissionError('Stage artifact outside the exact run or missing')
            checked[p.relative_to(self.folder).as_posix()] = digest(p)
        if callable(resources):
            resources=resources()
        row = dict(schema='rc3-stage-completion-v1', stage_key=key,
            status='COMPLETE' if complete else 'FAILED_INCOMPLETE', science_binding=self.binding,
            predecessors=admission['predecessors'], authorization_record_sha256=admission['authorization_record_sha256'],
            start_record_sha256=admission['start_record_sha256'], artifacts=checked,
            cumulative_stage_resources=copy.deepcopy(resources), error=error,
            no_automatic_retry_or_budget_reset=True)
        write_new(dest,row)
        return row
