"""Train/development-only data, frozen task semantics and reproducible schedule."""
from collections import OrderedDict
import copy
import json
import os
from pathlib import Path
import time

import numpy as np

from .approval import ROOT, ExecutionPermit, digest, read
from rc3_fast.planner import map_key
from rc3_global_local_impl_review.runtime_guards import (
    bind_navigation, current_navigation_sources, snapshot_parent, extract_snapshot)
from .navigation_compat import (LegacyTeacherReplay, load_navigation_variants,
    save_navigation_artifact, select_navigation)

SIZES = ((4, 3), (6, 6), (8, 8))


def plain(value):
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8', newline='\n') as f:
        f.write(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def journal(path, value):
    with Path(path).open('a', encoding='utf8', newline='\n') as f:
        f.write(json.dumps(plain(value), ensure_ascii=False, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def add(ledger, key, n=1):
    ledger[key] = ledger.get(key, 0) + n


class TrainingRegistry:
    """Never exposes a S3/task60 accessor, even though declaration JSON has S3 metadata."""
    def __init__(self, path):
        self.path = Path(path).resolve(); self.sha256 = digest(self.path)
        raw = read(self.path)
        self.parents = copy.deepcopy(raw['S2_train'])
        self.teachers = copy.deepcopy(raw['S1_teachers'])
        self.development = copy.deepcopy(raw['S2_development'])
        self.policy_seeds = tuple(raw['policy_seeds'])
        self.sampling_seed = raw['sampling_seed']
        if len(self.parents) != 96 or len(self.teachers) != 256 or len(self.development) != 18:
            raise ValueError('Registry must contain exactly96/256/18 declared train/teacher/development records')
        for i, p in enumerate(self.parents):
            if p['parent'] != i or len(p['task_seeds']) != 8 or p['mode'] not in ('mixed', 'blocked'):
                raise ValueError('Invalid declared training bank')
        self.train_lookup = {(p['map_seed'], t): p for p in self.parents for t in p['task_seeds']}
        self.dev_lookup = {(d['map_seed'], d['task_seed']): d for d in self.development}
        if set(self.train_lookup) & set(self.dev_lookup):
            raise ValueError('Train/development seed collision')

    def validate_spec(self, spec):
        key = (spec['map_seed'], spec['task_seed'])
        role = spec.get('role')
        if role == 'train' and key in self.train_lookup:
            expected = self.train_lookup[key]
            if spec['mode'] == expected['mode'] and spec['parent'] == expected['parent']:
                return
        if role == 'development' and key in self.dev_lookup:
            if spec['mode'] == self.dev_lookup[key]['mode']:
                return
        raise ValueError('Only declared train/development inputs are admitted; S3/old60 are sealed')

    def teacher_specs(self):
        for s in self.teachers:
            p = self.parents[s['parent']]
            value = dict(s, role='train', mode=p['mode'])
            self.validate_spec(value)
            yield value

    def development_specs(self):
        for d in self.development:
            w, h = d['size']; center = d['position'] == 'center'
            yield dict(d, role='development', parent=None, width=w, height=h,
                       x0=(10-w)//2 if center else 0, y0=(10-h)//2 if center else 0,
                       endpoints='open_first_last_connected' if center else 'closed_depot')


class TrainingSchedule:
    """Explicit v3 candidate schedule; construction does not activate its protocol."""
    def __init__(self, registry, streams, batch_size=4):
        if batch_size != 4:
            raise ValueError('Reviewed candidate macro batch is4')
        self.registry, self.streams, self.batch_size = registry, streams, batch_size

    def batch(self, update):
        if isinstance(update, bool) or not isinstance(update, int) or update < 0:
            raise ValueError('Zero-based integer update required')
        rng = self.streams.numpy('S2_data')
        result = []
        for slot in range(4):
            pi, ti = int(rng.integers(96)), int(rng.integers(8))
            p = self.registry.parents[pi]
            if update % 2 == 0:
                w = h = 10; x = y = 0; endpoint = 'closed_depot'; local_index = None
            else:
                local_index = ((update-1)//2)*4+slot
                w, h = SIZES[local_index % 3]
                x, y = int(rng.integers(11-w)), int(rng.integers(11-h))
                endpoint = 'closed_depot' if local_index % 2 == 0 else 'open_first_last_connected'
            result.append(dict(role='train', parent=pi, task_draw=ti, map_seed=p['map_seed'],
                task_seed=p['task_seeds'][ti], mode=p['mode'], A_condition=(pi+ti)%3,
                width=w, height=h, x0=x, y0=y, endpoints=endpoint,
                update=update, slot=slot, local_index=local_index))
        return result


class SceneProvider:
    """Lazy generation under a formal permit, maximum2 resident navigation maps.

    Every generated task draw and navigation artifact is persisted with hashes.
    Bad F&A/connectivity stops before training; it is never replaced or repaired.
    """
    def __init__(self, registry, folder, ledger, permit, record=None, deadline=None, read_only=False):
        if not isinstance(permit, ExecutionPermit) or permit.registry_sha256 != registry.sha256:
            raise PermissionError('Formal data generation requires a matching admitted execution permit')
        self.registry, self.folder, self.ledger = registry, Path(folder), ledger
        self.permit, self.record, self.deadline = permit, record or (lambda _: None), deadline
        self.read_only = bool(read_only)
        self.legacy_replay = LegacyTeacherReplay(permit.binding)
        if self.read_only and not self.folder.is_dir():
            raise ValueError('Read-only teacher data folder is missing')
        if not self.read_only:
            self.folder.mkdir(parents=True, exist_ok=True)
        self.nav_cache = OrderedDict()
        gate = read(ROOT / 'rc3_global_local_impl_review/GEOMETRY_GATE.json')
        if gate['status'] != 'PASS':
            raise ValueError('Actual geometry gate is not PASS')
        self.geometry = {(r['map_seed'], r['mode']): r['map_key'] for r in gate['records']
                         if r['role'] in ('train', 'development')}
        self.geometry_files = {r['map_key']: ROOT / r['file'] for r in gate['records']
                               if r['role'] in ('train', 'development')}

    def check(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline:
            raise TimeoutError('Data preparation stage wall budget reached')

    def _navigation(self, scene, expected_parent_identity=None, F=None, A=None):
        self.check(); key = map_key(scene)
        path = self.folder / 'navigation' / (key+'.npz')
        receipt = path.with_suffix('.json')
        self.legacy_replay.check_navigation(path)
        if key in self.nav_cache:
            add(self.ledger, 'navigation_memory_cache_hits')
            self.nav_cache.move_to_end(key)
            return select_navigation(self.nav_cache[key], scene, F, A, expected_parent_identity)
        if receipt.exists():
            variants = load_navigation_variants(scene, path)
            add(self.ledger, 'navigation_artifact_loads')
        else:
            if path.exists() or self.read_only:
                raise ValueError('Missing/incomplete navigation artifact; no read-only generation or free retry')
            from rc3_baseline.baseline import navigation
            add(self.ledger, 'navigation_attempts'); tick = time.perf_counter()
            self.record(dict(event='navigation_start', map_key=key))
            nav = navigation(scene)
            add(self.ledger, 'navigation_completed')
            save_navigation_artifact(scene, nav, path, time.perf_counter()-tick)
            variants = load_navigation_variants(scene, path)
            self.record(dict(event='navigation_complete', map_key=key, wall_s=time.perf_counter()-tick))
        # Cache both historical representations. A map can have two ORIGINAL
        # task identities; never let the first loaded task choose for later ones.
        self.nav_cache[key] = variants
        while len(self.nav_cache) > 2:
            self.nav_cache.popitem(last=False)
        return select_navigation(variants, scene, F, A, expected_parent_identity)

    def get(self, spec):
        self.check(); self.registry.validate_spec(spec)
        if spec['A_condition'] not in (0, 1, 2):
            raise ValueError('Unknown F&A condition')
        name = f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
        folder = self.folder / 'task_draws' / name
        receipt = folder / 'receipt.json'
        identity = {k: spec[k] for k in ('role', 'map_seed', 'task_seed', 'mode', 'A_condition')}
        add(self.ledger, 'parent_requests'); tick = time.perf_counter()
        replay = None
        if receipt.exists():
            r = read(receipt)
            replay = self.legacy_replay.check_task(folder, r)
            sources = (self.legacy_replay.manifest['science_binding']['source_sha256']
                       if replay is not None else self.permit.binding['source_sha256'])
            if (r['spec'] != identity or r['source_sha256'] != sources
                    or digest(folder/'scene.json') != r['scene_sha256'] or digest(folder/'arrays.npz') != r['arrays_sha256']):
                raise ValueError('Saved task draw does not match its original receipt')
            scene = read(folder/'scene.json')
            with np.load(folder/'arrays.npz', allow_pickle=False) as a:
                F, A = a['F'], a['A']
            add(self.ledger, 'parent_artifact_loads')
        else:
            if folder.exists() or self.read_only:
                raise ValueError('Missing/partial task artifact; no read-only or silent re-generation')
            from rc3_fast.scenes import make_complex_scene
            from rc3_constructive.problem import resample_tasks
            from rc3_fast.solver import vectorized_feasibility
            from rc3_global_local.run_s0 import make_allow
            add(self.ledger, 'task_scene_generation_attempts')
            self.record(dict(event='task_scene_start', spec=identity))
            scene = resample_tasks(make_complex_scene(10, 10, spec['map_seed'], spec['mode']), spec['task_seed'])
            add(self.ledger, 'task_scene_generations')
            add(self.ledger, 'proxy_feasibility_calls')
            F, _, _ = vectorized_feasibility(scene); A = make_allow(F, spec['A_condition'])
        expected_geometry = self.geometry.get((spec['map_seed'], spec['mode']))
        if expected_geometry is None or map_key(scene) != expected_geometry:
            raise ValueError('Task draw geometry differs from admitted actual geometry gate')
        expected_parent = r['parent_identity'] if receipt.exists() else None
        parent = snapshot_parent(scene, F, A, navigation=self._navigation(scene, expected_parent, F, A))
        if receipt.exists() and r['parent_identity'] != parent.identity:
            raise ValueError('Saved parent interpretation differs from original receipt')
        # Validate full target bank connectivity/F&A even for a small window.
        extract_snapshot(parent, 0, 0, 10, 10, 0, 0)
        if not receipt.exists():
            folder.mkdir(parents=True)
            write_new(folder/'scene.json', scene)
            with (folder/'arrays.npz').open('xb') as f:
                np.savez_compressed(f, F=F, A=A)
            write_new(receipt, dict(spec=identity, scene_sha256=digest(folder/'scene.json'),
                arrays_sha256=digest(folder/'arrays.npz'), parent_identity=parent.identity,
                map_key=expected_geometry, source_sha256=self.permit.binding['source_sha256'],
                wall_s=time.perf_counter()-tick))
        if spec['endpoints'] == 'closed_depot':
            entry = exit_ = 0
        elif spec['endpoints'] == 'open_first_last_connected':
            connected = np.flatnonzero(np.isfinite(parent.D[0, 1:]) & np.isfinite(parent.D[1:, 0])) + 1
            if len(connected) < 2:
                raise ValueError('Fewer than2 connected open endpoints; stop without replacement')
            entry, exit_ = int(connected[0]), int(connected[-1])
        else:
            raise ValueError('Unknown endpoint convention')
        view = extract_snapshot(parent, spec['x0'], spec['y0'], spec['width'], spec['height'], entry, exit_)
        if receipt.exists() and replay is not None and spec == replay['spec'] and view.identity != replay['view_identity']:
            raise ValueError('Legacy teacher view identity differs from exact replay receipt')
        add(self.ledger, 'view_extractions')
        self.record(dict(event='view', spec=spec, identity=view.identity, description=view.description(),
                         wall_s=time.perf_counter()-tick))
        self.check()
        return view

    def state_dict(self):
        return dict(registry_sha256=self.registry.sha256, nav_cache_keys=list(self.nav_cache))

    def load_state_dict(self, state):
        if state['registry_sha256'] != self.registry.sha256:
            raise ValueError('Provider resume registry mismatch')
        keys = list(state['nav_cache_keys'])
        if len(keys) > 2 or len(set(keys)) != len(keys) or any(k not in self.geometry_files for k in keys):
            raise ValueError('Invalid saved navigation LRU state')
        # Reload exactly the two saved artifacts in their original LRU order.
        # These reads/time are charged as resume overhead, never free nav calls.
        for key in keys:
            path = self.folder / 'navigation' / (key+'.npz')
            if not path.is_file() or not path.with_suffix('.json').is_file():
                raise ValueError('Saved resume navigation artifact is missing; do not regenerate')
        self.nav_cache.clear()
        for key in keys:
            scene = read(self.geometry_files[key])
            self._navigation(scene)
            add(self.ledger, 'navigation_resume_artifact_loads')
