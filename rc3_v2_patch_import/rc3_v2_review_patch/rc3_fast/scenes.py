"""Reproducible regular, perturbed and obstructed scenes, without task resampling."""
import numpy as np
from rc3_baseline.baseline import make_scene

MODES = ('regular', 'mixed', 'blocked')


def make_complex_scene(nx, ny, seed, mode='regular'):
    if mode not in MODES:
        raise ValueError(f'unknown scene mode {mode}')
    if mode == 'regular':
        scene = make_scene(nx, ny, seed)
        scene.update(complexity=mode, obstacles=[])
        return scene
    rng = np.random.default_rng(seed+7919)
    density = int(rng.integers(3))
    probabilities = ((.45, .30, .20, .05), (.25, .25, .25, .25), (.05, .10, .30, .55))[density]
    counts = rng.choice(4, size=nx*ny, p=probabilities).tolist()
    geometry = dict(table_width=float(rng.uniform(.85, .95)), table_depth=float(rng.uniform(.60, .75)),
                    aisle_gap=float(rng.uniform(1.15, 1.7)))
    scene = make_scene(nx, ny, seed, counts=counts, geometry=geometry)
    for table in scene['tables']:
        old_x, old_y = table['cx'], table['cy']
        sx, sy = rng.uniform(.95, 1.05, 2)
        dx, dy = rng.uniform(-.04, .04, 2)
        table['cx'], table['cy'] = old_x+dx, old_y+dy
        for low, high, center, shift, scale in (('xmin', 'xmax', old_x, dx, sx),
                                                ('ymin', 'ymax', old_y, dy, sy)):
            table[low] = center+shift+(table[low]-center)*scale
            table[high] = center+shift+(table[high]-center)*scale
        width, depth = table['xmax']-table['xmin'], table['ymax']-table['ymin']
        for task in scene['tasks']:
            if task['table_id'] == table['id']:
                task['x'] = old_x+dx+(task['x']-old_x)*sx
                task['y'] = old_y+dy+(task['y']-old_y)*sy
        for local, candidate in enumerate(scene['candidates'][table['id']*8:table['id']*8+8]):
            frac = -.25 if local % 2 == 0 else .25
            side = candidate['side']
            if side in ('bottom', 'top'):
                candidate['x'] = table['cx']+frac*width
                candidate['y'] = table['cy']+(-1 if side == 'bottom' else 1)*(depth/2+.5)
                candidate['corridor_anchor'][0] = candidate['x']
            else:
                candidate['x'] = table['cx']+(-1 if side == 'left' else 1)*(width/2+.5)
                candidate['y'] = table['cy']+frac*depth
                candidate['corridor_anchor'][1] = candidate['y']
    obstacles = []
    if mode == 'blocked':
        # Block some horizontal aisle segments, never all ways around a table.
        # Alternate rows and free vertical lanes retain an interpretable roadmap.
        slots = [(ix, iy) for iy in range(1, ny, 2) for ix in range(nx)]
        if slots:
            selected = rng.choice(len(slots), size=min(len(slots), max(1, round(nx*ny*.08))), replace=False)
            for index in selected:
                ix, iy = slots[index]
                x = (scene['x_lines'][ix]+scene['x_lines'][ix+1])/2
                y = scene['y_lines'][iy]
                obstacles.append(dict(xmin=x-.06, xmax=x+.06, ymin=y-.20, ymax=y+.20))
    scene.update(complexity=mode, density_profile=density, obstacles=obstacles,
                 layout_description='Individual table scale 0.95-1.05; position jitter +/-0.04 m; horizontal aisle blockers on alternating rows.')
    return scene
