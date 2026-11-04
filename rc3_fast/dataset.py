"""Offline recipe-cost data, with separate files for train/validation/held-out test."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from rc3_baseline import baseline
from .features import FEATURE_NAMES, scene_features
from .scenes import MODES, make_complex_scene
from .solver import RECIPES, DEFAULT_RECIPE, prepare, solve_route, route_objective, export_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-per-group', type=int, default=24)
    parser.add_argument('--val-per-group', type=int, default=6)
    parser.add_argument('--test-per-group', type=int, default=8)
    parser.add_argument('--scales', nargs='+', type=baseline.scale_argument, default=[(4,3),(6,6),(8,8),(10,10)])
    parser.add_argument('--seed', type=int, default=20261102)
    args = parser.parse_args(argv)
    if min(args.train_per_group, args.val_per_group, args.test_per_group) < 1:
        parser.error('all split sizes must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    total = (args.train_per_group+args.val_per_group+args.test_per_group)*len(args.scales)*len(MODES)
    seeds = iter(rng.choice(2**31-1, size=total, replace=False).tolist())
    manifest = dict(status='running', seed=args.seed, scales=args.scales, modes=MODES,
                    recipes=RECIPES, feature_names=FEATURE_NAMES, penalty=.75, search_passes=2,
                    splits={}, source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__).parent/'solver.py', Path(__file__).parent/'scenes.py', Path(__file__).parent/'features.py')})
    baseline.write_json(args.output/'manifest.json', manifest)
    started = time.perf_counter()
    for split, size in [('train',args.train_per_group),('validation',args.val_per_group),('test',args.test_per_group)]:
        features, targets, originals, records = [], [], [], []
        folder = args.output/'teachers'/split
        folder.mkdir(parents=True)
        for nx, ny in args.scales:
            for mode in MODES:
                for _ in range(size):
                    seed = next(seeds)
                    scene = make_complex_scene(nx,ny,seed,mode)
                    problem = prepare(scene)
                    features.append(scene_features(problem))
                    routes = [solve_route(problem, i) for i in range(len(RECIPES))]
                    scores = [route_objective(problem, route) for route in routes]
                    winner = int(np.argmin(scores))
                    teacher = export_plan(problem,routes[winner],'portfolio_teacher')
                    assert abs(teacher['objective_length_plus_stops_m']-scores[winner]) < 1e-7
                    original = route_objective(problem,solve_route(problem,DEFAULT_RECIPE,0))
                    targets.append(scores)
                    originals.append(original)
                    records.append(dict(nx=nx,ny=ny,seed=seed,mode=mode,tasks=len(problem.F),
                                        obstacles=len(scene['obstacles']),teacher_recipe=winner,
                                        preprocessing_s=problem.preprocessing_s))
                    baseline.write_json(folder/f'scene_plan_{nx}x{ny}_{mode}_seed{seed}.json',
                                        dict(scene=scene, feasibility=problem.sparse, plans={'portfolio_teacher':teacher}))
                    if len(records) % 12 == 0:
                        print(f'{split}: {len(records)}/{size*len(args.scales)*len(MODES)} scenes; elapsed={time.perf_counter()-started:.1f}s',flush=True)
        np.savez_compressed(args.output/f'{split}.npz',x=np.asarray(features),costs=np.asarray(targets),
                            original=np.asarray(originals))
        baseline.write_json(args.output/f'{split}_cases.json',records)
        manifest['splits'][split]=len(records)
        baseline.write_json(args.output/'manifest.json',manifest)
    manifest.update(status='complete',generation_s=time.perf_counter()-started)
    baseline.write_json(args.output/'manifest.json',manifest)
    print(f'COMPLETE {manifest["splits"]}; seconds={manifest["generation_s"]:.1f}',flush=True)


if __name__ == '__main__':
    main()
