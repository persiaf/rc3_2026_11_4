"""Budget-aware local-repair MDP with deterministic no-improvement caching."""
from __future__ import annotations

import math
import numpy as np

from rc3_fast.solver import solve_route, route_objective
from rc3_improve.environment import canonical_route
from rc3_improve.repair import (FEATURE_NAMES as BASE_FEATURE_NAMES, MAX_TASKS,
                               MAX_CANDIDATES, _local, window_features,
                               repair_window, route_is_valid)


FEATURE_SCHEMA = 'rc3-v3-budget28'
FEATURE_NAMES = list(BASE_FEATURE_NAMES) + [
    'remaining_budget_fraction', 'steps_fraction', 'current_reference_ratio',
    'cumulative_gain_fraction']
FIXED3_RECIPES = (0, 1, 7)


def reference_candidates(problem, recipes=FIXED3_RECIPES):
    """Construct cheap candidates; runtime independently certifies in cost order."""
    routes, failures = [], []
    for recipe in recipes:
        try:
            route = canonical_route(problem, solve_route(problem, recipe))
            score = route_objective(problem, route)
            if not math.isfinite(score):
                raise ValueError('nonfinite reference cost')
            routes.append((score, int(recipe), route))
        except Exception as error:
            failures.append(dict(stage='construct_reference', recipe=int(recipe), error=str(error)))
    return sorted(routes, key=lambda entry: (entry[0], entry[1])), failures


def adaptive_windows(problem, route, max_width=4):
    """Largest supported width at each service start; stride one, no duplicates."""
    if not route_is_valid(problem, route):
        raise ValueError('adaptive windows require a complete valid incumbent')
    if type(max_width) is not int or max_width < 1:
        raise ValueError('max_width must be a positive integer')
    cover = problem.F.any(2)
    nodes = np.asarray(route, dtype=np.int64)
    counts = cover[:, nodes[1:-1]-1].sum(1)
    result, signatures, seen = [], [], set()
    for start in range(1, len(route)-1):
        for width in range(min(max_width, len(route)-1-start), 0, -1):
            window = dict(start=start, end=start+width)
            _, end, required, pool, _ = _local(problem, route, window, cover, counts)
            if len(required) > MAX_TASKS or len(pool) > MAX_CANDIDATES:
                continue
            signature = (tuple(route[start-1:end+1]), tuple(required.tolist()), tuple(pool.tolist()))
            if signature not in seen:
                result.append(window); signatures.append(signature); seen.add(signature)
            break
    return result, signatures


class Env:
    def __init__(self, problem, horizon=8, initial_route=None):
        if type(horizon) is not int or horizon < 0:
            raise ValueError('horizon must be a nonnegative integer')
        if initial_route is None:
            candidates, failures = reference_candidates(problem)
            if not candidates:
                raise ValueError(f'no complete Fixed3 incumbent: {failures}')
            initial_route = candidates[0][2]
        self.problem, self.horizon = problem, horizon
        self.initial_route = canonical_route(problem, initial_route)
        self.route = list(self.initial_route)
        self.reference_cost = route_objective(problem, self.route)
        self.steps, self.attempted = 0, set()
        self.windows, self._signatures = [], []
        self._key, self._base_features = None, None
        self.features = np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        self.done = self.horizon == 0 or len(self.route) == 2

    def _ensure_structure(self):
        key = tuple(self.route)
        if key != self._key:
            self.windows, self._signatures = adaptive_windows(self.problem, self.route)
            self._key, self._base_features = key, None

    def _mask(self):
        mask = np.asarray([signature not in self.attempted for signature in self._signatures], dtype=bool)
        if self.steps >= self.horizon:
            mask[:] = False
        self.done = self.steps >= self.horizon or not mask.any()
        return mask

    def observe(self):
        self._ensure_structure()
        if self._base_features is None:
            self._base_features = window_features(self.problem, self.route, self.windows)
        mask = self._mask()
        current = route_objective(self.problem, self.route)
        ratio = current/self.reference_cost if self.reference_cost > 1e-8 else 1.
        gain = (self.reference_cost-current)/max(self.reference_cost, 1e-8)
        progress = np.asarray([max(self.horizon-self.steps, 0)/max(self.horizon, 1),
                               self.steps/max(self.horizon, 1), ratio, gain], dtype=np.float32)
        dynamic = np.broadcast_to(progress, (len(self.windows), 4))
        self.features = np.concatenate((self._base_features, dynamic), axis=1).astype(np.float32)
        return self.features.copy(), mask

    def step(self, action):
        # No call to observe/window_features here. Structural context may refresh
        # after acceptance, while expensive neural observations wait for next use.
        self._ensure_structure()
        mask = self._mask()
        if self.done or not isinstance(action, (int, np.integer)) or isinstance(action, (bool, np.bool_)) or not 0 <= action < len(mask) or not mask[action]:
            raise ValueError('invalid or masked repair action')
        action = int(action)
        before = route_objective(self.problem, self.route)
        signature = self._signatures[action]
        candidate = repair_window(self.problem, self.route, self.windows[action])
        if not route_is_valid(self.problem, candidate):
            raise ValueError('repair returned an invalid route; incumbent preserved')
        candidate = canonical_route(self.problem, candidate)
        after = route_objective(self.problem, candidate)
        if not math.isfinite(after):
            raise ValueError('repair returned a nonfinite objective; incumbent preserved')
        accepted = after < before-1e-8
        if accepted:
            self.route = list(candidate)
            # Successful contexts are never memoized; other failed contexts stay.
            self.attempted.discard(signature)
        else:
            self.attempted.add(signature)
            after = before
        self.steps += 1
        self._ensure_structure()
        self._mask()
        reward = 100*(before-after)/max(self.reference_cost, 1e-8)
        return reward, self.done, dict(accepted=accepted, before=before, after=after,
                                       cached_no_improvement=not accepted)


ImprovementEnv = Env
