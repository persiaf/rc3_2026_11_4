"""Use the already reviewed classifier in the newly authorized real M1 attempt.

No Torch import, scientific computation, or repository-wide source scan occurs
here. The existing exact-byte verifier, import guard, cache and tracer remain
the implementation; this adapter binds their sources and records actual use.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import types

from rc3_r04_m1_recovery_20261111.provenance import build_manifest
from rc3_r04_m1_recovery_20261111.provenance_revision import (
    RevisedTracedProvenance,
    classify_project_source,
)


PRIOR_DIRECTORY = 'rc3_r04_m1_recovery_20261111'
PRIOR_BINDING_SHA256 = 'cfc0a4b3fc007e047843f24a08acfff84ee30fe0c3a0e9b3d040144d54887594'
CLASSIFIER_REGISTRATION_SHA256 = '14dced57aea7e5f0075278b686ee59d33951acdd6ebd03970725a05231e860c5'
CLASSIFIER_SOURCE_SHA256 = '200b75ea7db401152a924ae083e4449c8a00d32d011301f7c4b6d165900ec04f'
LOCKED_FILES = (
    'M1_DATA_AND_SOURCE_IDENTITY.json',
    'BATCH_SCHEDULE.json',
    'M1_EXECUTION_PROTOCOL.json',
)


def _metadata(path, expected):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError('Frozen classifier provenance metadata SHA mismatch: '+str(path))
    return json.loads(payload)


def register_manifest(root, package_directory, attempt_directory):
    """Register this entry's sources and verbatim locked metadata, without runs.

    The returned binding must itself be frozen before launching the supervised
    worker. Registration work/time is charged by the caller. Old source hashes
    come from their original frozen binding, not from a fresh source scan.
    """
    root = Path(root).resolve()
    package = Path(package_directory).resolve()
    attempt = Path(attempt_directory).resolve()
    if not package.is_relative_to(root) or not attempt.is_relative_to(package/'attempts'):
        raise ValueError('Attempt/source registration must remain inside the new package')
    prior = _metadata(root/PRIOR_DIRECTORY/'SOURCE_BINDING.json', PRIOR_BINDING_SHA256)
    repair = _metadata(root/PRIOR_DIRECTORY/'CLASSIFIER_REPAIR_REGISTRATION.json',
                       CLASSIFIER_REGISTRATION_SHA256)
    if repair['source_sha256']['provenance_revision.py'] != CLASSIFIER_SOURCE_SHA256:
        raise ValueError('Previously reviewed classifier identity differs')
    manifest = build_manifest(root, package, new_source_files=sorted(package.glob('*.py')))
    sources = manifest['source_sha256']
    for name, expected in prior['source_sha256'].items():
        if name.startswith(PRIOR_DIRECTORY+'/'):
            existing = sources.setdefault(name, expected)
            if existing != expected:
                raise ValueError('Conflicting frozen recovery source identity: '+name)
    classifier_path = PRIOR_DIRECTORY+'/provenance_revision.py'
    existing = sources.setdefault(classifier_path, CLASSIFIER_SOURCE_SHA256)
    if existing != CLASSIFIER_SOURCE_SHA256:
        raise ValueError('Conflicting reviewed classifier source identity')
    required_sources = (PRIOR_DIRECTORY+'/__init__.py', PRIOR_DIRECTORY+'/provenance.py', classifier_path)
    if any(name not in sources for name in required_sources):
        raise ValueError('Missing exact old provenance implementation source identity')
    artifacts = manifest['artifact_sha256']
    reasons = manifest['artifact_reasons']
    for name, expected in (
        (PRIOR_DIRECTORY+'/SOURCE_BINDING.json', PRIOR_BINDING_SHA256),
        (PRIOR_DIRECTORY+'/CLASSIFIER_REPAIR_REGISTRATION.json', CLASSIFIER_REGISTRATION_SHA256),
    ):
        if artifacts.setdefault(name, expected) != expected:
            raise ValueError('Conflicting frozen metadata identity: '+name)
        reasons.setdefault(name, []).append('frozen provenance implementation/classifier source identity')
    identities = {}
    attempt.mkdir(parents=True, exist_ok=True)
    for name in LOCKED_FILES:
        source_name = 'rc3_r04_m1_20261111/'+name
        expected = artifacts[source_name]
        payload = (root/source_name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError('Locked M1 metadata SHA mismatch: '+source_name)
        target = attempt/name
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError('Attempt metadata copy changed: '+name)
        else:
            with target.open('xb') as stream:
                stream.write(payload)
        relative = target.relative_to(root).as_posix()
        if artifacts.setdefault(relative, expected) != expected:
            raise ValueError('Conflicting locked attempt copy: '+relative)
        reasons.setdefault(relative, []).append('verbatim fixed M1 metadata, same 116 trajectories and batch schedule')
        identities[name] = expected
    manifest.update(
        schema='rc3-r04-m1-real-existing-provenance-integration-v1',
        prior_runtime_binding_sha256=PRIOR_BINDING_SHA256,
        classifier_registration_sha256=CLASSIFIER_REGISTRATION_SHA256,
        classifier_source_sha256=CLASSIFIER_SOURCE_SHA256,
        classifier_revision='ModuleSpec_SourceFileLoader_v1',
        classifier_used_for_scientific_attempt=True,
        protocol_sha256=identities['M1_EXECUTION_PROTOCOL.json'],
        data_sha256=identities['M1_DATA_AND_SOURCE_IDENTITY.json'],
        schedule_sha256=identities['BATCH_SCHEDULE.json'],
        new_source_scope=package.relative_to(root).as_posix(),
        immutable_cache_reuse='Within one worker entry only; no previous worker or experiment cache',
    )
    return manifest


class ScientificTracedProvenance(RevisedTracedProvenance):
    """Use the frozen classifier unchanged, labelling actual scientific use."""

    def emit(self, event, **fields):
        if event == 'loaded_modules_checked':
            fields['used_for_scientific_attempt'] = True
            fields['integration'] = 'existing_reviewed_classifier_real_M1'
        return super().emit(event, **fields)


def verify_real_environment(provenance, torch_module, ordinary_modules):
    """Check actual imported Torch namespaces and registered project modules.

    Called only inside the supervised worker, after real Torch import. Reading
    module instance dictionaries avoids Torch's dynamic namespace lookup hooks.
    No namespace label is resolved as a purported local Python source.
    """
    if not isinstance(torch_module, types.ModuleType):
        raise TypeError('Actual imported Torch module required')
    namespace = types.ModuleType.__getattribute__(torch_module, '__dict__')
    if namespace.get('__name__') != 'torch' or sys.modules.get('torch') is not torch_module:
        raise ValueError('Real imported Torch module identity differs')
    result = dict(status='GO', used_for_scientific_attempt=True,
                  classifier_revision='ModuleSpec_SourceFileLoader_v1',
                  virtual_namespaces=[], project_sources=[])
    for short_name in ('ops', 'classes'):
        name = 'torch.'+short_name
        module = sys.modules.get(name)
        if not isinstance(module, types.ModuleType) or module is not namespace.get(short_name):
            raise ValueError('Real Torch namespace registration differs: '+name)
        classification = classify_project_source(module, provenance.root)
        if classification['kind'] != 'skip':
            raise ValueError('Torch virtual namespace falsely classified as project source: '+name)
        result['virtual_namespaces'].append(dict(module=name, **classification))
    if not ordinary_modules:
        raise ValueError('At least one actual project source module is required')
    for name, module in ordinary_modules.items():
        classification = classify_project_source(module, provenance.root)
        if classification['kind'] != 'source':
            raise ValueError('Ordinary project module missing valid source spec/loader: '+name)
        path = classification['path']
        expected = provenance.hash_file(path)
        relative = path.relative_to(provenance.root).as_posix()
        provenance.imported.add(relative)
        result['project_sources'].append(dict(module=name, path=relative, sha256=expected,
            classification_reason=classification['reason']))
    provenance.emit('real_torch_namespace_preflight_end', **result)
    return result
