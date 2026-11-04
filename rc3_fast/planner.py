"""Reusable planner with bounded static-map cache and portable selector inference."""
from collections import OrderedDict
import hashlib
import json
import time

from .features import scene_features
from .selector import Selector
from .solver import DEFAULT_RECIPE, prepare, solve_route, route_objective, export_plan


def map_key(scene):
    # Task positions/counts, seed and density are deliberately absent. All navigation
    # inputs are present, so changing an obstacle, candidate, depot or footprint misses.
    payload = {key: scene[key] for key in ('tables', 'candidates', 'start', 'x_lines', 'y_lines', 'robot')}
    payload['obstacles'] = scene.get('obstacles', [])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def choose_and_solve(problem, selector=None, top_k=1, recipe_ids=None):
    start = time.perf_counter()
    if recipe_ids is None:
        if selector is None:
            recipe_ids, features_s, model_s = [DEFAULT_RECIPE], 0., 0.
        else:
            features = scene_features(problem)
            features_s = time.perf_counter()-start
            infer_start = time.perf_counter()
            recipe_ids = selector.rank(features, top_k).tolist()
            model_s = time.perf_counter()-infer_start
    else:
        recipe_ids, features_s, model_s = list(recipe_ids), 0., 0.
    if not recipe_ids or any(not isinstance(i, int) or not 0 <= i < 13 for i in recipe_ids):
        raise ValueError('recipe_ids must be nonempty indices in 0..12')
    solve_start = time.perf_counter()
    routes = [solve_route(problem, i) for i in recipe_ids]
    scores = [route_objective(problem, route) for route in routes]
    winner = min(range(len(scores)), key=scores.__getitem__)
    return routes[winner], dict(recipe_ids=recipe_ids, winning_recipe=recipe_ids[winner],
                               feature_s=features_s, model_s=model_s,
                               search_s=time.perf_counter()-solve_start)


class Planner:
    def __init__(self, model_path=None, top_k=1, cache_size=2, policy='learned'):
        if not 1 <= top_k <= 13 or cache_size < 0:
            raise ValueError('invalid top_k or cache_size')
        if policy not in ('learned', 'fixed') or (policy == 'fixed' and model_path is None):
            raise ValueError('fixed policy requires the model metadata containing train-selected portfolios')
        self.selector = Selector(model_path) if model_path else None
        self.policy = policy
        if policy == 'fixed' and f'top{top_k}' not in self.selector.meta['fixed_portfolios']:
            raise ValueError('fixed policy supports top_k=1 or 3')
        self.top_k, self.cache_size, self._cache = top_k, cache_size, OrderedDict()

    def plan(self, scene):
        start = time.perf_counter()
        key = map_key(scene)
        cache_hit = key in self._cache
        problem = prepare(scene, nav=self._cache.get(key))
        if self.cache_size:
            self._cache[key] = problem.nav
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        recipe_ids = self.selector.meta['fixed_portfolios'][f'top{self.top_k}'] if self.policy == 'fixed' else None
        route, timing = choose_and_solve(problem, self.selector, self.top_k, recipe_ids)
        export_start = time.perf_counter()
        method = f'{self.policy}_top{self.top_k}' if self.selector else 'default_joint'
        plan = export_plan(problem, route, method)
        timing.update(preprocessing_s=problem.preprocessing_s, cache_hit=cache_hit,
                      export_and_validation_s=time.perf_counter()-export_start,
                      total_s=time.perf_counter()-start)
        return dict(scene=scene, feasibility=problem.sparse, plans={method: plan}, timing=timing,
                    model=dict(type='scene-level heuristic selector + coverage-preserving local search',
                               checkpoint=str(self.selector.path) if self.selector else None),
                    navigation=dict(nodes=problem.nav['node_count'], edges=problem.nav['edge_count'],
                                    rejected_collision_edges=problem.nav['rejected_edge_count']))
