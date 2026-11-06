"""Immutable execution snapshots and navigation provenance for future runners.

This is an execution guard layer, not a new planner or cost model. It never
generates navigation. A sealed navigation binding prevents accidental cross-map
reuse and post-seal changes; it is not a cryptographic proof that a producer
computed shortest paths correctly. The saved-S0 loader additionally verifies
the original scene/array receipts before constructing a binding.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rc3_fast.planner import map_key
from rc3_global_local.problem import (Parent as LegacyParent,
                                     LocalProblem as LegacyLocalProblem,
                                     array_hash, signature)


ROOT = Path(__file__).resolve().parents[1]
NAVIGATION_SOURCES = ('rc3_baseline/baseline.py', 'rc3_fast/planner.py')


class FrozenDict(dict):
    def _deny(self, *args, **kwargs):
        raise TypeError('Snapshot mappings are immutable')
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _deny

    def __deepcopy__(self, memo):
        return self


class FrozenList(list):
    def _deny(self, *args, **kwargs):
        raise TypeError('Snapshot sequences are immutable')
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _deny

    def __deepcopy__(self, memo):
        return self


def immutable_array(value, dtype=None):
    """Bytes-backed arrays cannot regain write permission via setflags()."""
    a = np.ascontiguousarray(value, dtype=dtype)
    if a.dtype.hasobject:
        raise ValueError('Object arrays are not execution inputs')
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def freeze(value):
    if isinstance(value, np.ndarray):
        return immutable_array(value)
    if isinstance(value, dict):
        return FrozenDict({k: freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return FrozenList([freeze(v) for v in value])
    if isinstance(value, np.generic):
        return value.item()
    return copy.deepcopy(value)


def _hashable(value):
    if isinstance(value, np.ndarray):
        return dict(array_sha256=array_hash(value))
    if isinstance(value, dict):
        return {k: _hashable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_hashable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('Unsupported navigation metadata type: '+type(value).__name__)


def navigation_hash(nav):
    return signature(_hashable(nav))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_navigation_sources():
    return {name: file_hash(ROOT/name) for name in NAVIGATION_SOURCES}


@dataclass(frozen=True)
class BoundNavigation:
    geometry_key: str
    content_sha256: str
    source_sha256: FrozenDict
    provenance: FrozenDict
    nav: FrozenDict


def bind_navigation(scene, nav, *, source_map_key, source_sha256, provenance):
    """Seal a producer-labelled navigation result; raw nav is never a snapshot.

    The caller must supply the geometry identity recorded when the navigation
    was produced, rather than silently assigning the current scene's identity.
    Source hashes must match this checkout's frozen navigation implementation.
    """
    if not isinstance(nav, dict) or 'costs' not in nav:
        raise ValueError('A navigation producer result with costs is required')
    if source_map_key != map_key(scene):
        raise ValueError('Navigation geometry key differs from current scene')
    sources = current_navigation_sources()
    if dict(source_sha256) != sources:
        raise ValueError('Navigation producer source version is not the bound checkout')
    if not isinstance(provenance, dict) or not provenance.get('producer'):
        raise ValueError('Explicit navigation provenance is required')
    m = len(scene['candidates'])
    costs = np.asarray(nav['costs'])
    if costs.shape != (m+1, m+1) or costs.dtype.kind not in 'fiu':
        raise ValueError('Navigation costs must include depot and all original candidates')
    if np.isnan(costs).any() or np.isneginf(costs).any() or (costs[np.isfinite(costs)] < 0).any():
        raise ValueError('Navigation contains invalid or negative costs')
    stored = dict(nav)
    stored['costs'] = np.asarray(costs, dtype=np.float64)
    frozen = freeze(stored)
    return BoundNavigation(source_map_key, navigation_hash(frozen),
                           freeze(sources), freeze(provenance), frozen)


@dataclass(frozen=True)
class BaseSnapshot:
    scene: FrozenDict
    F: np.ndarray
    nav: FrozenDict
    arm_dist: object = None
    sparse: object = None
    penalty: float = .75
    preprocessing_s: float = 0.


@dataclass(frozen=True)
class ParentSnapshot:
    base: BaseSnapshot
    F: np.ndarray
    A: np.ndarray
    effective: np.ndarray
    mode: str
    bundle: object
    identity: str
    legacy_identity: str
    navigation_binding: BoundNavigation

    @property
    def scene(self): return self.base.scene

    @property
    def D(self): return self.base.nav['costs']


def _binary(value, shape, name):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in 'biu' or not np.isin(raw, [0, 1]).all():
        raise ValueError(name+' must be an explicitly binary table of the requested shape')
    return immutable_array(raw, bool)


def snapshot_parent(scene, F, A, *, navigation, mode='proxy', bundle=None):
    if not isinstance(navigation, BoundNavigation):
        raise ValueError('Bare navigation is rejected; bind its producer and geometry first')
    if navigation.geometry_key != map_key(scene):
        raise ValueError('Bound navigation belongs to another geometry')
    if navigation.content_sha256 != navigation_hash(navigation.nav):
        raise ValueError('Navigation changed after producer binding')
    if dict(navigation.source_sha256) != current_navigation_sources():
        raise ValueError('Navigation source version changed')
    if mode not in ('proxy', 'external_tables'):
        raise ValueError('Unsupported service-constraint mode')
    shape = (len(scene['tasks']), len(scene['candidates']), 2)
    f = _binary(F, shape, 'F'); a = _binary(A, shape, 'A')
    frozen_scene = freeze(scene)
    if mode == 'proxy':
        fixed = frozen_scene['robot']['fixed_lift_height']
        if any(t.get('service_mode', 'independent_single_arm') != 'independent_single_arm' for t in frozen_scene['tasks']):
            raise ValueError('Coupled tasks need an explicit joint-service model')
        if any(not np.isfinite(c['h']) or abs(c['h']-fixed) > 1e-9 for c in frozen_scene['candidates']):
            raise ValueError('Proxy snapshot requires its fixed candidate height')
    else:
        from rc3_v5.external import check_bundle
        if bundle is None:
            raise ValueError('External table mode requires its original bundle')
        bf, ba = check_bundle(bundle)
        if signature(bundle['scene']) != signature(scene) or not np.array_equal(f, bf) or not np.array_equal(a, ba):
            raise ValueError('External bundle and snapshot inputs differ')
    base = BaseSnapshot(frozen_scene, f if mode == 'proxy' else immutable_array(f & a), navigation.nav)
    # Frozen S0 constructor retains its numeric/ID/symmetry preconditions.
    legacy = LegacyParent(base, f, a, mode, freeze(bundle) if bundle is not None else None)
    effective = immutable_array(f & a)
    identity = signature(dict(schema='rc3-bound-parent-v1', legacy=legacy.identity,
                              D=array_hash(navigation.nav['costs']),
                              navigation_sha256=navigation.content_sha256,
                              source_sha256=dict(navigation.source_sha256),
                              provenance=dict(navigation.provenance)))
    return ParentSnapshot(base, f, a, effective, mode,
                          freeze(bundle) if bundle is not None else None,
                          identity, legacy.identity, navigation)


@dataclass(frozen=True)
class LocalSnapshot:
    parent: ParentSnapshot
    task_ids: np.ndarray
    table_ids: tuple
    entry: int
    exit: int
    candidate_ids: np.ndarray
    F: np.ndarray
    C: np.ndarray
    D: np.ndarray
    boundary_candidate_ids: FrozenList
    window: object
    identity: str
    legacy_identity: str
    penalty: float = .75

    def global_route(self, route): return LegacyLocalProblem.global_route(self, route)

    def metrics(self, route): return LegacyLocalProblem.metrics(self, route)

    def description(self):
        result = LegacyLocalProblem.description(self)
        result.update(legacy_identity=self.legacy_identity,
                      navigation_content_sha256=self.parent.navigation_binding.content_sha256,
                      snapshot_schema='rc3-bound-local-v1')
        return result


def snapshot_local(parent, task_ids, table_ids, entry_node=0, exit_node=0, window=None):
    if not isinstance(parent, ParentSnapshot):
        raise ValueError('Execution local problems require an immutable bound parent')
    legacy = LegacyLocalProblem(parent, task_ids, table_ids, entry_node, exit_node, window)
    f = immutable_array(legacy.F); c = immutable_array(legacy.C); d = immutable_array(legacy.D)
    identity = signature(dict(schema='rc3-bound-local-v1', legacy=legacy.identity,
                              F=array_hash(f), C=array_hash(c), D=array_hash(d),
                              kernel_sha256=file_hash(ROOT/'rc3_global_local/problem.py')))
    return LocalSnapshot(parent, immutable_array(legacy.task_ids), legacy.table_ids,
                         legacy.entry, legacy.exit, immutable_array(legacy.candidate_ids),
                         f, c, d, freeze(legacy.boundary_candidate_ids),
                         freeze(window), identity, legacy.identity)


def extract_snapshot(parent, x0, y0, width, height, entry_node=0, exit_node=0):
    from rc3_global_local.problem import integer
    x0, y0, width, height = [integer(v, n) for v, n in zip(
        (x0, y0, width, height), ('x0', 'y0', 'width', 'height'))]
    if min(x0, y0) < 0 or min(width, height) < 1 or x0+width > parent.scene['nx'] or y0+height > parent.scene['ny']:
        raise ValueError('Window outside parent grid')
    tables = [t['id'] for t in parent.scene['tables']
              if x0 <= t['ix'] < x0+width and y0 <= t['iy'] < y0+height]
    tasks = [t['id'] for t in parent.scene['tasks'] if t['table_id'] in set(tables)]
    return snapshot_local(parent, tasks, tables, entry_node, exit_node,
                          dict(x0=x0, y0=y0, width=width, height=height))


def load_saved_s0_parent(case, s0_root=None):
    """Read original S0 scene/NPZ receipts only; no preparation or navigation."""
    from rc3_global_local.problem import integer
    case = integer(case, 'case')
    if case not in (0, 1, 2):
        raise ValueError('Only the three existing S0 parent artifacts are allowed')
    s0_root = Path(s0_root) if s0_root is not None else ROOT/'rc3_global_local'
    folder = s0_root/'parents'
    scene_path = folder/f'{case:02}_scene.json'
    arrays_path = folder/f'{case:02}_arrays.npz'
    receipt_path = folder/f'{case:02}_receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if file_hash(scene_path) != receipt['scene_sha256'] or file_hash(arrays_path) != receipt['arrays_sha256']:
        raise ValueError('Saved S0 inputs differ from their frozen receipt')
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    with np.load(arrays_path, allow_pickle=False) as arrays:
        F, A, D = arrays['F'], arrays['A'], arrays['D']
        if any(array_hash(value) != receipt[key] for value, key in
               ((F, 'F_sha256'), (A, 'A_sha256'), (D, 'D_sha256'))):
            raise ValueError('Saved S0 numeric array identity mismatch')
        nav = dict(costs=D, coordinates=arrays['nav_coordinates'].tolist(),
                   service_nodes=arrays['service_nodes'].tolist(),
                   predecessors=arrays['predecessors'], node_count=receipt['navigation_nodes'],
                   edge_count=receipt['navigation_edges'])
        binding = bind_navigation(scene, nav, source_map_key=map_key(scene),
            source_sha256=current_navigation_sources(), provenance=dict(
                producer='frozen_saved_S0_arrays', case=case,
                scene_sha256=file_hash(scene_path), arrays_sha256=file_hash(arrays_path),
                original_receipt_sha256=file_hash(receipt_path)))
        parent = snapshot_parent(scene, F, A, navigation=binding)
    if parent.legacy_identity != receipt['identity']:
        raise ValueError('Saved S0 parent interpretation differs from original identity')
    return parent
