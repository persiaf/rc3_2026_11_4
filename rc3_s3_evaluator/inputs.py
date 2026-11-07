"""Whole-parent geometry and same-arm F&A, with navigation-only warm state."""
from pathlib import Path
import copy
import numpy as np

from .common import (ROOT, HERE, PROTOCOL, REGISTRY, SIZES, require, read, digest,
                     project_file, identity, exact_int, write_new)
from rc3_fast.planner import map_key
from rc3_global_local.problem import array_hash
from rc3_global_local_impl_review.runtime_guards import (
    bind_navigation, current_navigation_sources, snapshot_parent, extract_snapshot, navigation_hash)


def window_spec(row):
    w, h = row['size']
    require(type(w) is int and type(h) is int and (w, h) in SIZES and row['position'] in ('corner', 'center'), 'Invalid declared window')
    require(type(row['A_condition']) is int and row['A_condition'] in (0, 1, 2), 'Invalid A condition')
    center = row['position'] == 'center'
    endpoint = 'open_first_last_connected' if center else 'closed_depot'
    if 'endpoints' in row:
        require(row['endpoints'] == endpoint, 'Position/endpoint declaration mismatch')
    return dict(row, width=w, height=h, x0=(10-w)//2 if center else 0,
                y0=(10-h)//2 if center else 0, endpoints=endpoint)


class NavigationPool:
    """The sole shared cache contains immutable navigation, never F/A/actors/tasks."""
    def __init__(self, namespace):
        self.namespace = namespace
        self.sources = current_navigation_sources()
        self.cache = {}

    def key(self, scene):
        require(current_navigation_sources() == self.sources, 'Navigation implementation changed during run')
        return (self.namespace, map_key(scene), identity(self.sources))

    def prime(self, scene, meter):
        key = self.key(scene)
        if key not in self.cache:
            self.cache[key] = self._build(scene, meter, 'hot_navigation_prime')
        else:
            meter.add('hot_prime_existing_navigation')

    def _build(self, scene, meter, operation):
        from rc3_baseline.baseline import navigation
        with meter.span(operation):
            nav = navigation(scene)
            meter.add('navigation_solves')
            bound = bind_navigation(scene, nav, source_map_key=map_key(scene),
                source_sha256=self.sources, provenance=dict(producer='frozen_s3_navigation',
                geometry_key=map_key(scene), source_identity=identity(self.sources)))
        return bound

    def get(self, scene, kind, meter):
        require(kind in ('quality', 'cold', 'hot'), 'Unknown request/cache kind')
        if kind == 'hot':
            require(self.key(scene) in self.cache, 'Hot navigation must be explicitly primed before request timer')
            b = self.cache[self.key(scene)]
            require(b.geometry_key == map_key(scene) and b.source_sha256 == self.sources and
                    b.content_sha256 == navigation_hash(b.nav), 'Stale/corrupt warm navigation')
            meter.add('navigation_only_cache_hits')
            return b
        # Cold/quality requests neither consult nor populate warm cache.
        return self._build(scene, meter, 'cold_navigation')


class SavedDevelopmentInputs:
    def __init__(self, permit, output):
        require(permit.scope == 'engineering', 'Saved-data provider is engineering-only')
        self.cases = copy.deepcopy(permit.registration['cases'])
        self.by_id = {c['id']: c for c in self.cases}
        self.source = read(ROOT/'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json')

    def _source(self, case):
        require(self.by_id.get(case['id']) == case, 'Unregistered or modified old case')
        paths = {n: project_file(v['path']) for n, v in case['files'].items()}
        for n, path in paths.items():
            require(digest(path) == case['files'][n]['sha256'], 'Saved engineering input changed: ' + n)
        receipt, scene = read(paths['receipt']), read(paths['scene'])
        spec = case['spec']
        expected = {k: spec[k] for k in ('role', 'map_seed', 'task_seed', 'mode', 'A_condition')}
        require(receipt['spec'] == expected and receipt['source_sha256'] == self.source['source_sha256'], 'Wrong saved input producer/spec')
        require(receipt['scene_sha256'] == case['files']['scene']['sha256'] and
                receipt['arrays_sha256'] == case['files']['arrays']['sha256'] and
                receipt['parent_identity'] == case['original_parent_identity'] and
                receipt['map_key'] == case['physical_key'] == map_key(scene), 'Saved parent/receipt mismatch')
        return paths, receipt, scene

    def geometry(self, case, meter):
        with meter.span('warm_geometry_read'):
            _, _, scene = self._source(case)
        return scene

    def load(self, case, meter):
        with meter.span('input_parse_and_source_verify'):
            paths, receipt, scene = self._source(case)
            with np.load(paths['arrays'], allow_pickle=False) as a:
                require(set(a.files) == {'F', 'A'}, 'Unexpected saved reachability fields')
                F, A = a['F'].copy(), a['A'].copy()
            require(array_hash(F) == case['F_sha256'] and array_hash(A) == case['A_sha256'], 'Saved F/A differs')
            meter.add('saved_task_input_loads')
        return scene, F, A, case['spec'], dict(original_parent_identity=receipt['parent_identity'],
            input_source_sha256={n: v['sha256'] for n, v in case['files'].items()},
            input_kind='existing_development_only', navigation_costs_sha256=case['D_sha256'])


class FormalInputs:
    """Only an admitted formal permit exposes declarations or materializes S3 tasks."""
    def __init__(self, permit, output):
        require(permit.scope == 'formal', 'S3 target access requires a real formal S3 permit')
        self.output = Path(output)
        require(digest(ROOT/'rc3_global_local_execution_v3/REGISTRY.json') == REGISTRY, 'Registry changed')
        registry = read(ROOT/'rc3_global_local_execution_v3/REGISTRY.json')
        self.cases = []
        gate = read(ROOT/'rc3_global_local_impl_review/GEOMETRY_GATE.json')
        require(gate['status'] == 'PASS', 'Physical geometry gate absent')
        geometries = {(r['map_seed'], r['mode']): r for r in gate['records']}
        for row in registry['S3']:
            spec = window_spec(row)
            g = geometries[(row['map_seed'], row['mode'])]
            require(row['group'] in ('seen_layout_new_tasks', 'unseen_layout_new_tasks'), 'Unknown declared S3 group')
            group = row['group'].split('_')[0]
            require(g['role'] == ('train' if group == 'seen' else 'unseen'), 'Seen/unseen physical source differs')
            self.cases.append(dict(id=f"S3_{row['case']:03}", group=group, spec=spec,
                size=list(row['size']), endpoint=spec['endpoints'], physical_key=g['map_key'], geometry_file=g['file'],
                geometry_sha256=digest(project_file(g['file']))))
        require(len(self.cases) == 108 and len({c['id'] for c in self.cases}) == 108, 'S3 declarations must be exact108')
        self.by_id = {c['id']: c for c in self.cases}
        for group, count in (('seen', 6), ('unseen', 12)):
            subset = [c for c in self.cases if c['group'] == group]
            keys = {c['physical_key'] for c in subset}
            require(len(keys) == count, 'Physical clustering differs from approved geometry gate')
            for key in keys:
                rows = [c for c in subset if c['physical_key'] == key]
                require(len(rows) == 6 and {(tuple(c['size']), c['endpoint']) for c in rows} ==
                        {(size, ep) for size in SIZES for ep in ('closed_depot', 'open_first_last_connected')}, 'Parent child grid differs')

    def geometry(self, case, meter):
        require(self.by_id.get(case['id']) == case, 'Unknown/modified heldout declaration')
        with meter.span('warm_geometry_read'):
            path = project_file(case['geometry_file'])
            require(digest(path) == case['geometry_sha256'], 'Frozen parent geometry changed')
            scene = read(path)
            require(map_key(scene) == case['physical_key'], 'Physical parent geometry mismatch')
        return scene

    def load(self, case, meter):
        require(self.by_id.get(case['id']) == case, 'Unknown/modified heldout declaration')
        from rc3_fast.scenes import make_complex_scene
        from rc3_constructive.problem import resample_tasks
        from rc3_fast.solver import vectorized_feasibility
        from rc3_global_local.run_s0 import make_allow
        spec = case['spec']
        with meter.span('input_generation_and_preparation'):
            # Uniform per-request work: no method gets a free prepared target or feature set.
            scene = resample_tasks(make_complex_scene(10, 10, spec['map_seed'], spec['mode']), spec['task_seed'])
            meter.add('target_task_generations')
            require(map_key(scene) == case['physical_key'], 'Generated parent differs from frozen physical layout')
            F, _, _ = vectorized_feasibility(scene)
            A = make_allow(F, spec['A_condition'])
            meter.add('proxy_F_A_preparations')
            content = dict(scene=identity(scene), F=array_hash(F), A=array_hash(A), spec=spec)
            folder = self.output/'inputs'/case['id']
            receipt = folder/'RECEIPT.json'
            if receipt.exists():
                r = read(receipt)
                require(r['content'] == content and digest(folder/'scene.json') == r['scene_sha256'] and
                        digest(folder/'arrays.npz') == r['arrays_sha256'], 'Persisted S3 input changed; no replacement')
            else:
                require(not folder.exists(), 'Incomplete input artifact; no free regeneration/replacement')
                write_new(folder/'scene.json', scene)
                with (folder/'arrays.npz').open('xb') as f:
                    np.savez_compressed(f, F=F, A=A)
                write_new(receipt, dict(content=content, scene_sha256=digest(folder/'scene.json'),
                    arrays_sha256=digest(folder/'arrays.npz'), preparation='Frozen producer chain; every request regenerates equally and is charged'))
        return scene, F, A, spec, dict(input_kind='authorized_formal_S3', content=content)


def prepare_view(provider, case, pool, kind, meter):
    scene, F, A, spec, source = provider.load(case, meter)
    with meter.span('navigation_access'):
        navigation = pool.get(scene, kind, meter)
        if 'navigation_costs_sha256' in source:
            require(array_hash(navigation.nav['costs']) == source['navigation_costs_sha256'], 'Recomputed navigation differs from saved old input')
    with meter.span('whole_parent_and_window'):
        require(scene['nx'] == scene['ny'] == 10 and map_key(scene) == case['physical_key'], 'Parent was cropped or substituted')
        parent = snapshot_parent(scene, F, A, navigation=navigation)
        # Full task-bank feasibility precedes local extraction; no task/candidate repair.
        extract_snapshot(parent, 0, 0, 10, 10, 0, 0)
        if spec['endpoints'] == 'closed_depot':
            entry = exit_ = 0
        else:
            require(spec['endpoints'] == 'open_first_last_connected', 'Unknown endpoint convention')
            connected = np.flatnonzero(np.isfinite(parent.D[0, 1:]) & np.isfinite(parent.D[1:, 0])) + 1
            require(len(connected) >= 2, 'Open entry/exit unavailable; stop without replacement')
            entry, exit_ = int(connected[0]), int(connected[-1])
        view = extract_snapshot(parent, spec['x0'], spec['y0'], spec['width'], spec['height'], entry, exit_)
        source.update(evaluation_parent_identity=parent.identity, view_identity=view.identity,
                      F_sha256=array_hash(parent.F), A_sha256=array_hash(parent.A),
                      D_sha256=array_hash(parent.D), global_task_ids=view.task_ids.tolist(),
                      global_candidate_ids=view.candidate_ids.tolist(), full_parent_tables=len(scene['tables']),
                      full_parent_obstacles=len(scene.get('obstacles', [])), full_parent_geometry=map_key(scene))
    return view, source
