"""Frozen geometry adapters and a bounded, in-memory navigation cache."""
from __future__ import annotations
from collections import OrderedDict
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from rc3_fast.scenes import make_complex_scene
from rc3_fast.planner import map_key
from rc3_fast.solver import prepare, export_plan

ROOT = Path(__file__).resolve().parent
MODES = ('regular', 'mixed', 'blocked')


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def protocol():
    return json.loads((ROOT/'protocol.yaml').read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resample_tasks(layout, task_seed, exact_count=None):
    """Task generator v1: frozen table boxes/margins; no geometry resampling."""
    # Static layout dictionaries are read-only in this module and the frozen adapters.
    scene = dict(layout); rng = np.random.default_rng(task_seed)
    tables = scene['tables']; counts = np.zeros(len(tables), dtype=int)
    if exact_count is not None:
        for _ in range(exact_count):
            choices = np.flatnonzero(counts < 3); counts[int(rng.choice(choices))] += 1
    elif scene['complexity'] == 'regular':
        counts = rng.integers(0, 4, len(tables))
    else:
        probabilities = ((.45,.30,.20,.05),(.25,.25,.25,.25),(.05,.10,.30,.55))[scene['density_profile']]
        counts = rng.choice(4, size=len(tables), p=probabilities)
    tasks = []
    margin = scene['geometry']['task_margin']
    for table, count in zip(tables, counts):
        # Mixed maps scale the original margin with each table, as the frozen generator does.
        mx = margin * (table['xmax']-table['xmin']) / scene['geometry']['table_width']
        my = margin * (table['ymax']-table['ymin']) / scene['geometry']['table_depth']
        for _ in range(int(count)):
            tasks.append(dict(id=len(tasks), table_id=table['id'],
                x=float(rng.uniform(table['xmin']+mx, table['xmax']-mx)),
                y=float(rng.uniform(table['ymin']+my, table['ymax']-my)), z=table['z']))
    scene.update(tasks=tasks, task_seed=int(task_seed), task_generator='rc3_constructive.resample_tasks-v1')
    return scene


def registered_scene(split, index):
    spec = protocol()['splits'][split]
    seed = spec['seed_start'] + index
    return make_complex_scene(10, 10, seed, MODES[index % 3])


@dataclass
class Input:
    base: object
    bundle: dict | None = None
    cache_hit: bool = False
    preparation_s: float = 0.

    def __post_init__(self): self.coverage = self.base.F.any(2)

    @property
    def C(self): return self.coverage

    @property
    def D(self): return self.base.nav['costs']

    def certify(self, route, method):
        if self.bundle is not None:
            from rc3_v5.external import export_external
            return export_external(self.base, route, self.bundle, method)
        return export_plan(self.base, route, method)

    def check(self):
        scene = self.base.scene
        for records in ('tasks','candidates','tables'):
            if [x['id'] for x in scene[records]] != list(range(len(scene[records]))):
                raise ValueError('noncontiguous original IDs')
        if self.bundle is None:
            if any(abs(c['h']-scene['robot']['fixed_lift_height']) > 1e-9 for c in scene['candidates']):
                raise ValueError('variable lift unsupported in proxy mode')
            if any(t.get('service_mode','independent_single_arm') != 'independent_single_arm' for t in scene['tasks']):
                raise ValueError('coupled tasks unsupported')
        connected = np.isfinite(self.D[0,1:]) & np.isfinite(self.D[1:,0])
        if not (self.C & connected).any(1).all():
            missing = np.flatnonzero(~(self.C & connected).any(1)).tolist()
            raise ValueError(f'no depot-connected service candidate for tasks {missing}; no physical infeasibility claim')
        return self


class Cache:
    def __init__(self, size=24): self.size, self.nav = size, OrderedDict()

    def prepare(self, scene):
        started = time.perf_counter()
        if 'planning' in scene and 'audit' in scene:
            from rc3_v5.external import prepare_external
            return Input(prepare_external(scene), scene, False, time.perf_counter()-started).check()
        key = map_key(scene); hit = key in self.nav
        base = prepare(scene, nav=self.nav.get(key))
        if self.size:
            self.nav[key] = base.nav; self.nav.move_to_end(key)
            while len(self.nav) > self.size: self.nav.popitem(last=False)
        return Input(base, None, hit, time.perf_counter()-started).check()


def features(item):
    scene, cover = item.base.scene, item.C
    n, m = cover.shape
    depot = np.asarray(scene['start'], dtype=float)
    xy = np.asarray([[x['x'], x['y']] for x in scene['candidates']], float).reshape(m,2)-depot
    scale = max(1., float(np.linalg.norm(np.ptp(np.vstack([xy, np.zeros((1,2))]), axis=0))))
    tasks = np.asarray([[x['x']-depot[0], x['y']-depot[1], x['z']] for x in scene['tasks']], float).reshape(n,3)
    tf = np.column_stack([tasks[:,0]/scale, tasks[:,1]/scale, tasks[:,2],
                          item.base.F[:,:,0].sum(1)/max(m,1), item.base.F[:,:,1].sum(1)/max(m,1)]).astype('float32')
    cf = np.column_stack([xy/scale, [np.cos(x['theta']) for x in scene['candidates']],
                          [np.sin(x['theta']) for x in scene['candidates']], [x['h'] for x in scene['candidates']],
                          cover.sum(0)/max(n,1), np.isfinite(item.D[0,1:])]).astype('float32')
    return tf, cf, scale
