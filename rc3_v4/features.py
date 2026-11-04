"""Observable local F/D tensors; never call the repair oracle for features."""
from __future__ import annotations
import numpy as np
from rc3_improve.repair import _local, MAX_CANDIDATES, MAX_TASKS
from rc3_v3.environment import adaptive_windows


def local_graphs(problem, route, windows=None):
    windows = adaptive_windows(problem, route)[0] if windows is None else windows
    cover = problem.F.any(2)
    counts = cover[:, np.asarray(route[1:-1], dtype=int)-1].sum(1)
    costs, scene = problem.nav['costs'], problem.scene
    configs = np.asarray([[c[k] for k in ('x','y','theta','h')] for c in scene['candidates']])
    tasks = np.asarray([[t[k] for k in ('x','y','z')] for t in scene['tasks']])
    on_route = np.zeros(len(configs)+1,bool);on_route[np.asarray(route,dtype=int)] = True
    result = []
    for window in windows:
        start, end, required, pool, _ = _local(problem, route, window, cover, counts)
        if len(required) > MAX_TASKS or len(pool) > MAX_CANDIDATES:
            raise ValueError('unsupported graph exceeds exact repair caps')
        entrance, exit_node = route[start-1], route[end]
        segment = np.asarray(route[start-1:end+1])
        scale = max(float(costs[segment[:-1], segment[1:]].sum()), 1.)
        origin = scene['candidates'][entrance-1] if entrance else scene['start']
        # The depot is a list; candidate service configurations are mappings.
        ox, oy = (origin['x'], origin['y']) if isinstance(origin, dict) else origin[:2]
        n, m = len(required), len(pool)
        cand = np.zeros((max(m, 1), 10), np.float32)
        task = np.zeros((max(n, 1), 4), np.float32)
        reach = np.zeros((max(n, 1), max(m, 1), 2), np.float32)
        dist = np.zeros((max(m, 1), max(m, 1)), np.float32)
        cmask, tmask = np.zeros(max(m, 1), bool), np.zeros(max(n, 1), bool)
        # An empty required set is represented by one masked-task dummy candidate.
        cmask[:max(m, 1)] = True
        tmask[:n] = True
        if n and m:
            reach[:n, :m] = problem.F[required][:, pool-1]
            dist[:m, :m] = costs[np.ix_(pool, pool)]/scale
            points = configs[pool-1]
            cand[:m] = np.column_stack((costs[entrance,pool]/scale,costs[pool,exit_node]/scale,
                (points[:,0]-ox)/scale,(points[:,1]-oy)/scale,np.sin(points[:,2]),
                np.cos(points[:,2]),points[:,3],on_route[pool],
                reach[:n,:m].any(2).mean(0),reach[:n,:m,0].mean(0)))
            xyz=tasks[required]
            task[:n]=np.column_stack(((xyz[:,0]-ox)/scale,(xyz[:,1]-oy)/scale,xyz[:,2]/scale,
                                     reach[:n,:m].any(2).mean(1)))
        graph = dict(cand=cand, task=task, reach=reach, dist=dist, cmask=cmask, tmask=tmask)
        if not all(np.isfinite(v).all() for v in graph.values()):
            raise ValueError('nonfinite structured observation')
        result.append(graph)
    return result


def collate(rows):
    """Pad states/windows/candidates/tasks with masks at every level."""
    s, w = len(rows), max(len(r['features']) for r in rows)
    c = max(len(g['cmask']) for r in rows for g in r['graphs'])
    t = max(len(g['tmask']) for r in rows for g in r['graphs'])
    batch = dict(base=np.zeros((s,w,28),np.float32), mask=np.zeros((s,w),bool),
        cand=np.zeros((s,w,c,10),np.float32), task=np.zeros((s,w,t,4),np.float32),
        reach=np.zeros((s,w,t,c,2),np.float32), dist=np.zeros((s,w,c,c),np.float32),
        cmask=np.zeros((s,w,c),bool), tmask=np.zeros((s,w,t),bool))
    batch['cmask'][:,:,0] = True  # Prevent NaNs even for padded, unused windows.
    for si, row in enumerate(rows):
        nw = len(row['features'])
        if len(row['graphs']) != nw or len(row['mask']) != nw:
            raise ValueError('window/graph alignment mismatch')
        batch['base'][si,:nw], batch['mask'][si,:nw] = row['features'], row['mask']
        for wi, g in enumerate(row['graphs']):
            nc, nt = len(g['cmask']), len(g['tmask'])
            for name in ('cand','cmask'): batch[name][si,wi,:nc] = g[name]
            for name in ('task','tmask'): batch[name][si,wi,:nt] = g[name]
            batch['reach'][si,wi,:nt,:nc] = g['reach']
            batch['dist'][si,wi,:nc,:nc] = g['dist']
    return batch


def observation(env, features=None, mask=None):
    if features is None: features, mask = env.observe()
    return dict(features=features, mask=mask,
                graphs=local_graphs(env.problem, env.route, env.windows))
