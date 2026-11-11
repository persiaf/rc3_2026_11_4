"""Attempt-scoped exact identities, lazy source verification and file timing.

This module uses only the standard library. No source-wide scan, model import,
model read, task reconstruction or numerical analysis occurs on import.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading
import time

OLD_DIRECTORY = 'rc3_r04_m1_20261111'
OLD_BINDING_SHA = '20ac28c723ad3ffb7caf2d0fe90391470dab836d12c566aae356fee4dceae0d9'
OLD_DATA_SHA = '8cb63231f748e2dbf47f3c18135ea721af589a2d0d4d4e55857e69ca582feeff'
OLD_PROTOCOL_SHA = '99e7030e0293b87cf4302f5d76f6802fceb3d2d9c0f3816de24d46cc18ed3982'


def _read_metadata(root, name, expected=None):
    """Registration-only small metadata read, never an artifact/model scan."""
    payload = (root/name).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if expected is not None and actual != expected:
        raise ValueError('Frozen registration metadata changed: '+name)
    return json.loads(payload), actual


def _literal_assignment(root, name, variable):
    source = (root/name).read_text(encoding='utf-8-sig')
    for node in ast.parse(source, filename=name).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Missing literal source identity: '+name+':'+variable)


def build_manifest(root, recovery_directory, *, new_source_files=None):
    """Create an expected identity registry without hashing historical files.

    The inherited source map is an identity catalogue, NOT an eager preflight
    loop. Only imported/read dependencies are checked during the new attempt.
    Selection is reused verbatim; no parent/trajectory reselection is performed.
    Caller must include this metadata-only registration time in its receipt.
    """
    root = Path(root).resolve(); recovery_directory = Path(recovery_directory).resolve()
    prior, prior_sha = _read_metadata(root, OLD_DIRECTORY+'/SOURCE_BINDING.json', OLD_BINDING_SHA)
    data, data_sha = _read_metadata(root, OLD_DIRECTORY+'/M1_DATA_AND_SOURCE_IDENTITY.json', OLD_DATA_SHA)
    if data['count'] != 116 or len(data['fit_physical_parents']) != 33 or len(data['holdout_physical_parents']) != 16:
        raise ValueError('Locked 116 / 33 / 16 identity differs')
    sources = dict(prior['source_sha256'])
    files = list(new_source_files) if new_source_files is not None else sorted(recovery_directory.glob('*.py'))
    for path in files:
        path = Path(path).resolve()
        if not path.is_relative_to(recovery_directory):
            raise ValueError('New executable source outside recovery directory')
        sources[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    artifacts = {}
    reasons = {}
    def admit(path, sha, reason):
        if not isinstance(path, str) or not isinstance(sha, str) or len(sha) != 64:
            raise ValueError('Exact project path / SHA256 pair required')
        p = (root/path).resolve()
        if not p.is_relative_to(root): raise ValueError('Artifact escaped project')
        previous = artifacts.setdefault(path, sha)
        if previous != sha: raise ValueError('Conflicting artifact identity: '+path)
        reasons.setdefault(path, []).append(reason)
    for filename, sha in (('SOURCE_BINDING.json', prior_sha),
                          ('M1_DATA_AND_SOURCE_IDENTITY.json', data_sha),
                          ('M1_EXECUTION_PROTOCOL.json', OLD_PROTOCOL_SHA),
                          ('BATCH_SCHEDULE.json', prior['schedule_sha256'])):
        admit(OLD_DIRECTORY+'/'+filename, sha, 'exact immutable previous M1 registration')
        clone = recovery_directory/filename
        # Copies are permitted only if their payload bytes equal the original.
        if clone.exists() and clone.name != 'SOURCE_BINDING.json':
            actual = hashlib.sha256(clone.read_bytes()).hexdigest()
            if actual != sha: raise ValueError('New attempt changed locked data/config/schedule bytes: '+filename)
            admit(clone.relative_to(root).as_posix(), sha, 'verbatim locked-registration copy')
    selected_receipts = set()
    for row in data['trajectories']:
        admit(row['request_path'], row['request_sha256'], 'selected saved complete trajectory')
        for item in row['saved_input_files'].values():
            admit(item['path'], item['sha256'], 'selected trajectory saved input dependency')
        if 'receipt' in row['saved_input_files']:
            selected_receipts.add(row['saved_input_files']['receipt']['path'])
    # Keep exact frozen loader assertions, including byte-only registry checks;
    # this does not parse any sealed original60/S3 case or generate a task.
    anchors = _literal_assignment(root, 'rc3_conservative_refinement/saved_inputs.py', 'ANCHORS')
    for path, sha in anchors.items(): admit(path, sha, 'actual saved-input loader provenance anchor')
    slpath = _literal_assignment(root, 'rc3_conservative_refinement/runtime.py', 'SL_PATH')
    slsha = _literal_assignment(root, 'rc3_conservative_refinement/runtime.py', 'SL_SHA256')
    admit(slpath, slsha, 'accurate frozen SL256 feature encoder')
    upstream = _literal_assignment(root, 'rc3_conservative_refinement/runtime.py', 'UPSTREAM')
    for path_key, sha_key in (('path', 'checkpoint_sha256'), ('sidecar_path','sidecar_sha256'),('commit_path','commit_sha256')):
        admit(upstream[path_key], upstream[sha_key], 'mandatory SL256 original upstream byte identity; not another model load')
    legacy_name = 'rc3_navigation_compat_20261106/LEGACY_TEACHER_REPLAY_MANIFEST.json'
    legacy, _ = _read_metadata(root, legacy_name, anchors[legacy_name])
    for kind in ('run_manifest','stage_complete','pool_index','source_manifest','protocol_approval'):
        admit(legacy[kind+'_path'],legacy[kind+'_sha256'],'actual LegacyTeacherReplay bootstrap anchor')
    for row in legacy['records']:
        if row['task_receipt_path'] not in selected_receipts: continue
        for kind in ('task_receipt','scene','arrays','label','navigation','navigation_receipt'):
            admit(row[kind+'_path'],row[kind+'_sha256'],'selected old-schema input exact legacy provenance hash; no teacher request')
    return dict(schema='rc3-r04-m1-recovery-lazy-dependency-identities-v1',
        inherited_binding_sha256=prior_sha, locked_data_sha256=data_sha,
        source_sha256=sources, artifact_sha256=artifacts, artifact_reasons=reasons,
        source_verification='Only actually imported/read project dependencies, exact verified source bytes compiled; catalogue is not eagerly hashed',
        cache_scope='One attempt, immutable canonical path + expected SHA256; no dynamic outputs, no cross-run cache',
        H_identity=prior['H_identity'], environment=prior['environment'],
        original_source_catalogue_count=len(prior['source_sha256']),
        eager_historical_source_hash_count=0,
        registered_trajectories=116, registered_fit_parents=33, registered_holdout_parents=16)


class TracedProvenance:
    """An attempt-local verifier; deadline checks complement an external killer.

    Immutable source bytes are memoized only AFTER a complete expected-SHA match.
    Cache hits explicitly retain that exact identity; mtime is never evidence.
    Output paths are excluded. File writes during an attempt are not supported.
    """
    def __init__(self, root, output_directory, manifest, check_deadline, run_id):
        self.root = Path(root).resolve(); self.output = Path(output_directory).resolve()
        self.run_id = str(run_id); self.check = check_deadline
        self.expected = dict(manifest['source_sha256'])
        for path, sha in manifest['artifact_sha256'].items():
            if path in self.expected and self.expected[path] != sha:
                raise ValueError('Conflicting source/artifact SHA256')
            self.expected[path] = sha
        self.trace_path = self.output/'PREFLIGHT_TRACE.jsonl'
        self.cache = {}; self.verified = {}; self.imported = set()
        self.started = time.perf_counter(); self.lock = threading.RLock()
        self.counts = dict(hash_started=0, hash_completed=0, cache_hits=0, bytes_hashed=0)
        self.last_completed_operation = None; self.active_stage = None
        self.operation_in_progress = None
        self.public_stage_stack = []
        self.original_get_code = None

    def _identity(self, path, expected=None):
        p = Path(path).resolve()
        if not p.is_relative_to(self.root): raise ValueError('Source escaped project: '+str(p))
        relative = p.relative_to(self.root).as_posix()
        admitted = self.expected.get(relative)
        if admitted is None: raise ValueError('Unregistered actual dependency: '+relative)
        if expected is not None and expected != admitted: raise ValueError('Caller supplied another source SHA: '+relative)
        return p,relative,admitted

    def emit(self, event, **fields):
        record = dict(event=event,run_id=self.run_id,monotonic_s=time.perf_counter(),
                      wall_unix_s=time.time(),elapsed_s=time.perf_counter()-self.started,
                      stage=self.active_stage)
        record.update(fields)
        with self.lock, self.trace_path.open('a',encoding='utf-8',newline='\n') as stream:
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush()
            # Per-file operation boundaries remain visible while another read
            # stalls; the independent supervisor owns crash/timeout receipts.
            os.fsync(stream.fileno())
            if event.endswith('_start'):
                self.operation_in_progress={k:v for k,v in record.items() if k not in ('run_id',)}
            if (event.endswith('_end') and record.get('status','GO')=='GO') or event.endswith('_cache_hit'):
                self.last_completed_operation=dict(record)
                self.operation_in_progress=None
            heartbeat=dict(run_id=self.run_id,pid=os.getpid(),wall_unix_s=record['wall_unix_s'],
                monotonic_s=record['monotonic_s'],stage=record.get('stage'),
                operation_in_progress=self.operation_in_progress,last_completed_operation=self.last_completed_operation,
                last_event=event)
            temp=self.output/'WORKER_HEARTBEAT.tmp'
            with temp.open('w',encoding='utf-8',newline='\n') as state:
                json.dump(heartbeat,state,ensure_ascii=False,allow_nan=False);state.write('\n');state.flush()
            temp.replace(self.output/'WORKER_HEARTBEAT.json')
        return record

    def event(self,event,**fields):
        """Public phase/hash/audit callback; never changes scientific ledgers."""
        if event=='stage_start':
            self.public_stage_stack.append((self.active_stage,time.perf_counter()))
            self.active_stage=fields.get('name',self.active_stage)
            return self.emit(event,**fields)
        if event=='stage_end':
            previous,tick=self.public_stage_stack[-1] if self.public_stage_stack else (None,time.perf_counter())
            fields.setdefault('seconds',time.perf_counter()-tick)
            result=self.emit(event,**fields)
            if self.public_stage_stack:self.public_stage_stack.pop()
            self.active_stage=previous
            return result
        return self.emit(event,**fields)

    def _verified_bytes(self, path, expected=None):
        self.check(); p,relative,sha = self._identity(path,expected)
        key = (str(p).casefold(),sha)
        if key in self.cache:
            self.counts['cache_hits'] += 1
            self.emit('hash_cache_hit',path=relative,sha256=sha,bytes=len(self.cache[key]),
                      scope='same_attempt_exact_identity_verified_bytes')
            self.check()
            return self.cache[key]
        tick=time.perf_counter(); total=0; h=hashlib.sha256(); pieces=[]
        self.counts['hash_started'] += 1
        self.emit('hash_start',path=relative,expected_sha256=sha)
        try:
            with p.open('rb') as stream:
                while True:
                    self.check()
                    block=stream.read(1<<20)
                    self.check()
                    if not block: break
                    h.update(block);pieces.append(block);total+=len(block)
            actual=h.hexdigest()
            if actual != sha: raise ValueError('Source identity differs: '+relative)
            value=b''.join(pieces)
            self.cache[key]=value; self.verified[relative]=sha
            self.counts['hash_completed']+=1;self.counts['bytes_hashed']+=total
            self.last_completed_operation=dict(operation='hash',path=relative,sha256=actual,
                                               bytes=total,seconds=time.perf_counter()-tick)
            self.emit('hash_end',**self.last_completed_operation,status='GO')
            return value
        except BaseException as exc:
            self.emit('hash_end',path=relative,bytes=total,seconds=time.perf_counter()-tick,
                      status='FAIL',error=repr(exc))
            raise

    def hash_file(self, path, expected=None):
        _,_,sha=self._identity(path,expected)
        self._verified_bytes(path,sha)
        return sha

    def read_bytes(self, path, expected=None):
        return self._verified_bytes(path,expected)

    def read_json(self, path, expected=None):
        return json.loads(self.read_bytes(path,expected))

    @contextlib.contextmanager
    def stage(self,name):
        previous=self.active_stage;self.active_stage=name;tick=time.perf_counter()
        self.check();self.emit('stage_start',name=name)
        try:
            yield
        except BaseException as exc:
            self.emit('stage_end',name=name,status='FAIL',seconds=time.perf_counter()-tick,error=repr(exc));raise
        else:
            self.last_completed_operation=dict(operation='stage',name=name,seconds=time.perf_counter()-tick)
            self.emit('stage_end',name=name,status='GO',seconds=time.perf_counter()-tick)
        finally:self.active_stage=previous

    def _project_source(self,path):
        try:p=Path(path).resolve()
        except (TypeError,OSError):return None
        if p.suffix.lower()!='.py' or not p.is_relative_to(self.root):return None
        relative=p.relative_to(self.root)
        if any(part in ('.venv','site-packages','__pycache__') for part in relative.parts):return None
        return p

    def install_import_guard(self):
        if self.original_get_code is not None: raise RuntimeError('Import guard already installed')
        original=importlib.machinery.SourceFileLoader.get_code
        self.original_get_code=original;owner=self
        def verified_code(loader,fullname):
            path=owner._project_source(loader.get_filename(fullname))
            if path is None:return original(loader,fullname)
            payload=owner.read_bytes(path)
            owner.imported.add(path.relative_to(owner.root).as_posix())
            owner.emit('project_import',module=fullname,path=path.relative_to(owner.root).as_posix(),
                       semantics='compile_exact_verified_source_bytes_no_pyc_mtime_trust')
            return compile(payload,str(path),'exec',dont_inherit=True,optimize=-1)
        importlib.machinery.SourceFileLoader.get_code=verified_code
        self.emit('import_guard_installed',scope='actual local project sources only; third-party runtime untouched')
        return self

    def check_loaded_modules(self):
        """Cover startup modules imported before the hook and dynamic imports."""
        checked=[]
        for name,module in list(sys.modules.items()):
            source=self._project_source(getattr(module,'__file__',None))
            if source is None:continue
            self.hash_file(source);relative=source.relative_to(self.root).as_posix()
            self.imported.add(relative);checked.append(dict(module=name,path=relative))
        self.emit('loaded_modules_checked',count=len(checked))
        return checked

    def restore_import_guard(self):
        if self.original_get_code is not None:
            importlib.machinery.SourceFileLoader.get_code=self.original_get_code
            self.original_get_code=None

    def summary(self):
        return dict(run_id=self.run_id,counts=dict(self.counts),verified_sha256=dict(sorted(self.verified.items())),
                    imported_sources=sorted(self.imported),last_completed_operation=self.last_completed_operation,
                    cache_scope='sameattempt exact immutable identity only; not outputs; never cross-attempt',
                    total_trace_wall_s=time.perf_counter()-self.started)
