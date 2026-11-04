"""Scene-only features: never read seed, teacher costs or evaluation labels."""
import numpy as np


FEATURE_NAMES = [
    'log_tables', 'aspect', 'tasks_per_table', 'empty_fraction', 'one_fraction', 'two_fraction', 'three_fraction',
    'table_width_mean', 'table_width_std', 'table_depth_mean', 'table_depth_std', 'aisle_gap', 'obstacles_per_table',
    'navigable_fraction', 'candidate_cover_mean', 'candidate_cover_std', 'candidate_cover_max',
    'task_options_mean', 'task_options_std', 'task_options_min', 'both_arms_fraction',
    'distance_mean', 'distance_std', 'distance_q25', 'distance_q75', 'detour_ratio_mean', 'detour_ratio_std',
    'task_x_mean', 'task_y_mean', 'task_x_std', 'task_y_std', 'task_xy_cov', 'row_work_std', 'column_work_std',
    'task_options_q25', 'task_options_q75', 'single_dock_table_fraction', 'coverage_density']


def scene_features(problem):
    scene = problem.scene
    tables, tasks = scene['tables'], scene['tasks']
    nt, n = len(tables), len(tasks)
    cover = problem.F.any(2)
    counts = np.bincount([t['table_id'] for t in tasks], minlength=nt)
    widths = np.asarray([t['xmax']-t['xmin'] for t in tables])
    depths = np.asarray([t['ymax']-t['ymin'] for t in tables])
    navigable = np.isfinite(problem.nav['costs'][0, 1:])
    gains = cover.sum(0)
    options = (cover & navigable).sum(1)
    ids = np.flatnonzero(navigable & (gains > 0))
    if len(ids) > 32:
        ids = ids[np.linspace(0, len(ids)-1, 32).astype(int)]
    d = problem.nav['costs'][np.ix_(ids+1, ids+1)]
    xy = np.asarray([[scene['candidates'][j]['x'], scene['candidates'][j]['y']] for j in ids]).reshape(-1, 2)
    euclidean = np.sqrt(((xy[:, None]-xy[None, :])**2).sum(-1))
    mask = np.isfinite(d) & (euclidean > 1e-7)
    length_scale = max(scene['x_lines'][-1], scene['y_lines'][-1], 1.)
    distances = d[mask]/length_scale
    detours = d[mask]/euclidean[mask]
    points = np.asarray([[t['x'], t['y']] for t in tasks]).reshape(-1, 2)/length_scale
    row_work = counts.reshape(scene['ny'], scene['nx']).sum(1)/scene['nx']
    column_work = counts.reshape(scene['ny'], scene['nx']).sum(0)/scene['ny']
    singles = []
    for table in tables:
        indices = [t['id'] for t in tasks if t['table_id'] == table['id']]
        if indices:
            singles.append((cover[indices].all(0) & navigable).any())
    mean = lambda x: float(np.mean(x)) if np.size(x) else 0.
    std = lambda x: float(np.std(x)) if np.size(x) else 0.
    quantile = lambda x, q: float(np.quantile(x, q)) if np.size(x) else 0.
    values = [np.log1p(nt), scene['nx']/scene['ny'], n/nt,
              *[mean(counts == k) for k in range(4)], mean(widths), std(widths), mean(depths), std(depths),
              scene['aisle_gap'], len(scene.get('obstacles', []))/nt, mean(navigable), mean(gains), std(gains),
              float(gains.max(initial=0)), mean(options), std(options), float(options.min()) if n else 0.,
              mean(problem.F.all(2)), mean(distances), std(distances), quantile(distances, .25), quantile(distances, .75),
              mean(detours), std(detours), mean(points[:, 0]), mean(points[:, 1]), std(points[:, 0]), std(points[:, 1]),
              mean((points[:, 0]-mean(points[:, 0]))*(points[:, 1]-mean(points[:, 1]))),
              std(row_work), std(column_work), quantile(options, .25), quantile(options, .75), mean(singles),
              mean(cover)]
    result = np.asarray(values, dtype=np.float32)
    if len(result) != len(FEATURE_NAMES) or not np.isfinite(result).all():
        raise ValueError('invalid scene features')
    return result
