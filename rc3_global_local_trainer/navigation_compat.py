"""Versioned navigation persistence and exact, read-only S1-A replay lineage.

This module enumerates SAVED corridor topology only. It never filters collision
edges, generates scenes, solves navigation, runs a teacher, or grants authority.
"""
import json
from pathlib import Path
import re

import numpy as np

from .approval import ROOT, digest, read
from rc3_fast.planner import map_key
from rc3_global_local_impl_review.runtime_guards import (
    bind_navigation, current_navigation_sources, navigation_hash, snapshot_parent)

SCHEMA = 'rc3-navigation-artifact-v2'
ARRAY_KEYS = {'costs', 'coordinates', 'service_nodes', 'predecessors'}
COUNTS = {'node_count', 'edge_count', 'rejected_edge_count'}
LEGACY_META = {'sha256', 'source_sha256', 'map_key', 'node_count', 'edge_count', 'wall_s'}
FULL_META = LEGACY_META | {'schema', 'rejected_edge_count', 'navigation_content_sha256'}


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(name+' must be a nonnegative integer, not bool/float')
    return value


def _project_file(name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError('Replay path must be project relative')
    p = (ROOT/name).resolve()
    if not p.is_relative_to(ROOT) or not p.is_file():
        raise ValueError('Replay file missing or outside project')
    return p


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Expected exact SHA256 identity')
    return value


def recovered_rejected_edge_count(scene, nav):
    """Count potential edges on persisted coordinates; no geometry/graph solve."""
    coords = np.asarray(nav['coordinates'])
    nodes = _integer(nav['node_count'], 'node_count')
    accepted = _integer(nav['edge_count'], 'edge_count')
    services = np.asarray(nav['service_nodes'])
    if (coords.shape != (nodes, 2) or not np.isfinite(coords).all()
            or services.shape != (len(scene['candidates'])+1,)
            or services.dtype.kind not in 'iu'
            or (services < 0).any() or (services >= nodes).any()):
        raise ValueError('Invalid persisted navigation topology')
    points = {tuple(p): i for i, p in enumerate(coords.tolist())}
    if len(points) != nodes:
        raise ValueError('Duplicate persisted navigation coordinates')
    if not np.array_equal(coords[int(services[0])], np.round(scene['start'], 10)):
        raise ValueError('Persisted depot node differs')
    edges = set()
    for axis, lines in ((0, scene['x_lines']), (1, scene['y_lines'])):
        for line in lines:
            lane = sorted((i for i,p in enumerate(coords) if abs(p[axis]-line)<1e-8),
                          key=lambda i: coords[i][1-axis])
            edges.update(zip(lane, lane[1:]))
    for j, candidate in enumerate(scene['candidates']):
        a = int(services[j+1])
        point = (round(float(candidate['x']), 10), round(float(candidate['y']), 10))
        anchor = tuple(round(float(x), 10) for x in candidate['corridor_anchor'])
        if tuple(coords[a]) != point or anchor not in points:
            raise ValueError('Persisted service/anchor node differs')
        edges.add((a, points[anchor]))
    total = len({tuple(sorted((a,b))) for a,b in edges if a != b})
    if accepted > total:
        raise ValueError('Accepted edge count exceeds saved candidate topology')
    return total-accepted


def _validated(scene, nav, complete):
    expected = ARRAY_KEYS | {'node_count', 'edge_count'} | ({'rejected_edge_count'} if complete else set())
    if set(nav) != expected:
        raise ValueError('Unknown or missing navigation identity fields')
    rejected = recovered_rejected_edge_count(scene, nav)
    if complete and _integer(nav['rejected_edge_count'], 'rejected_edge_count') != rejected:
        raise ValueError('Rejected edge count contradicts saved topology')
    pred = np.asarray(nav['predecessors'])
    costs = np.asarray(nav['costs'])
    count = len(scene['candidates'])+1
    if (pred.shape != (count, nav['node_count']) or pred.dtype.kind not in 'iu'
            or ((pred < 0) & (pred != -9999)).any() or (pred >= nav['node_count']).any()
            or costs.dtype != np.dtype('float64') or costs.shape != (count, count)):
        raise ValueError('Invalid persisted navigation arrays')
    return rejected


def _bind(scene, nav, path, sha):
    return bind_navigation(scene, nav, source_map_key=map_key(scene),
        source_sha256=current_navigation_sources(), provenance=dict(
            producer='formal_registered_navigation', artifact_sha256=sha, map_key=map_key(scene)))


def save_navigation_artifact(scene, nav, path, wall_s):
    """Save every identity field; never overwrite either half of an artifact."""
    path = Path(path); receipt = path.with_suffix('.json')
    _validated(scene, nav, True)
    if type(wall_s) not in (int, float) or not np.isfinite(wall_s) or wall_s < 0:
        raise ValueError('Invalid navigation preparation time')
    if path.exists() or receipt.exists():
        raise ValueError('Navigation artifact already exists; no overwrite/retry')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez_compressed(f, **{k:nav[k] for k in sorted(ARRAY_KEYS | COUNTS)})
    meta = dict(schema=SCHEMA, sha256=digest(path), source_sha256=current_navigation_sources(),
        map_key=map_key(scene), navigation_content_sha256=navigation_hash(nav),
        wall_s=float(wall_s), **{k:nav[k] for k in sorted(COUNTS)})
    with receipt.open('x', encoding='utf8', newline='\n') as f:
        f.write(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def load_navigation_variants(scene, path):
    """Return one new-schema binding, or both EXACT historical representations."""
    path = Path(path); meta = read(path.with_suffix('.json'))
    complete = meta.get('schema') == SCHEMA
    if set(meta) != (FULL_META if complete else LEGACY_META):
        raise ValueError('Unknown navigation schema or metadata field')
    if (meta['map_key'] != map_key(scene) or meta['source_sha256'] != current_navigation_sources()
            or digest(path) != _sha(meta['sha256'])):
        raise ValueError('Saved navigation source/content/geometry mismatch')
    if type(meta['wall_s']) not in (int, float) or not np.isfinite(meta['wall_s']) or meta['wall_s'] < 0:
        raise ValueError('Invalid navigation wall metadata')
    for key in COUNTS if complete else COUNTS-{'rejected_edge_count'}:
        _integer(meta[key], key)
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != (ARRAY_KEYS | COUNTS if complete else ARRAY_KEYS):
            raise ValueError('Unexpected navigation NPZ fields')
        nav = dict(costs=arrays['costs'], coordinates=arrays['coordinates'].tolist(),
            service_nodes=arrays['service_nodes'].tolist(), predecessors=arrays['predecessors'],
            node_count=meta['node_count'], edge_count=meta['edge_count'])
        if complete:
            for key in COUNTS:
                a = arrays[key]
                if a.shape != () or a.dtype.kind not in 'iu' or int(a) != meta[key]:
                    raise ValueError('Navigation NPZ/receipt scalar differs: '+key)
            nav['rejected_edge_count'] = meta['rejected_edge_count']
    rejected = _validated(scene, nav, complete)
    if complete:
        if navigation_hash(nav) != _sha(meta['navigation_content_sha256']):
            raise ValueError('Full navigation content identity mismatch')
        return [('complete_v2', _bind(scene, nav, path, meta['sha256']))]
    full = dict(nav, rejected_edge_count=rejected)
    return [('legacy_reduced', _bind(scene, nav, path, meta['sha256'])),
            ('legacy_recovered', _bind(scene, full, path, meta['sha256']))]


def select_navigation(variants, scene, F=None, A=None, expected_parent_identity=None):
    if expected_parent_identity is not None:
        _sha(expected_parent_identity)
        if F is None or A is None:
            raise ValueError('Saved parent matching requires original F/A')
        matches = [b for _,b in variants if
            snapshot_parent(scene, F, A, navigation=b).identity == expected_parent_identity]
    else:
        matches = [b for _,b in variants if 'rejected_edge_count' in b.nav]
    if len(matches) != 1:
        raise ValueError('Navigation identity has no unique exact matching representation')
    return matches[0]


class LegacyTeacherReplay:
    """An explicit hash-pinned bridge, never a general old-source whitelist."""
    def __init__(self, binding):
        self.manifest = None; self.tasks = {}; self.navigation = {}
        bridge = binding.get('legacy_teacher_replay')
        if bridge is None:
            return
        if not isinstance(bridge, dict) or set(bridge) != {'manifest_path', 'manifest_sha256', 'science_binding'}:
            raise ValueError('Invalid explicit legacy teacher bridge')
        path = _project_file(bridge['manifest_path'])
        if digest(path) != _sha(bridge['manifest_sha256']):
            raise ValueError('Legacy replay manifest bytes differ')
        m = read(path)
        if m.get('schema') != 'rc3-legacy-teacher-replay-manifest-v1' or m.get('science_binding') != bridge['science_binding']:
            raise ValueError('Legacy replay manifest science binding differs')
        for kind in ('run_manifest', 'stage_complete', 'pool_index', 'source_manifest', 'protocol_approval'):
            p = _project_file(m[kind+'_path'])
            if digest(p) != _sha(m[kind+'_sha256']):
                raise ValueError('Legacy replay anchor mismatch: '+kind)
        run = (ROOT/m['run']).resolve()
        if not run.is_relative_to(ROOT) or not run.is_dir():
            raise ValueError('Invalid original teacher run')
        if (read(ROOT/m['run_manifest_path'])['binding'] != m['science_binding']
                or read(ROOT/m['stage_complete_path'])['science_binding'] != m['science_binding']
                or read(ROOT/m['stage_complete_path'])['status'] != 'COMPLETE'
                or read(ROOT/m['pool_index_path'])['source_binding'] != m['science_binding']
                or read(ROOT/m['source_manifest_path'])['source_sha256'] != m['science_binding']['source_sha256']):
            raise ValueError('Legacy replay anchors have inconsistent lineage')
        if (m['source_manifest_sha256'] != m['science_binding']['source_manifest_sha256']
                or m['protocol_approval_sha256'] != m['science_binding']['approval_sha256']):
            raise ValueError('Legacy manifest/approval identity differs')
        approval = read(ROOT/m['protocol_approval_path'])
        if approval.get('status') != 'APPROVED' or approval.get('approved_by') != 'user':
            raise ValueError('Original teacher approval is absent')
        rows = m.get('records')
        if not isinstance(rows, list) or [r.get('index') for r in rows] != list(range(256)):
            raise ValueError('Legacy teacher list must be exactly ordered256')
        self.manifest = m
        for row in rows:
            for name in ('task_receipt', 'scene', 'arrays', 'label', 'navigation', 'navigation_receipt'):
                p = _project_file(row[name+'_path'])
                if not p.is_relative_to(run):
                    raise ValueError('Legacy record escapes its original run')
                _sha(row[name+'_sha256'])
            _sha(row['parent_identity']); _sha(row['view_identity'])
            folder = (ROOT/row['task_receipt_path']).parent.resolve()
            if folder in self.tasks:
                raise ValueError('Duplicate legacy task identity')
            self.tasks[folder] = row
            navpath = (ROOT/row['navigation_path']).resolve()
            pair = (row['navigation_sha256'], row['navigation_receipt_sha256'])
            if navpath in self.navigation and self.navigation[navpath] != pair:
                raise ValueError('Conflicting legacy navigation source')
            self.navigation[navpath] = pair

    def check_navigation(self, path):
        path = Path(path).resolve()
        if path in self.navigation:
            a,b = self.navigation[path]
            if digest(path) != a or digest(path.with_suffix('.json')) != b:
                raise ValueError('Frozen legacy navigation artifact/receipt changed')

    def check_task(self, folder, receipt):
        row = self.tasks.get(Path(folder).resolve())
        if row is None:
            return None
        for name in ('task_receipt', 'scene', 'arrays', 'label', 'navigation', 'navigation_receipt'):
            if digest(ROOT/row[name+'_path']) != row[name+'_sha256']:
                raise ValueError('Frozen legacy teacher source changed: '+name)
        expected_spec = {k:row['spec'][k] for k in ('role','map_seed','task_seed','mode','A_condition')}
        if (receipt['spec'] != expected_spec or receipt['parent_identity'] != row['parent_identity']
                or receipt['map_key'] != row['map_key']
                or receipt['source_sha256'] != self.manifest['science_binding']['source_sha256']
                or receipt['scene_sha256'] != row['scene_sha256']
                or receipt['arrays_sha256'] != row['arrays_sha256']):
            raise ValueError('Legacy task receipt identity/source differs')
        return row
