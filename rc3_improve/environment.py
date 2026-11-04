"""Finite-horizon improvement MDP with refreshed windows and monotone incumbents."""
import numpy as np
from rc3_fast.solver import solve_route, route_objective
from .repair import windows, window_features, repair_window, route_is_valid


def canonical_route(problem, route):
    """Use exactly the visit semantics of export_plan before scoring a route.

    A station automatically serves every remaining covered task. Stations adding
    no new coverage are omitted, so rewards and teacher labels measure the same
    path and stop penalty that the independently checked exported plan reports.
    """
    if not route_is_valid(problem, route):
        raise ValueError('canonicalization requires a complete valid route')
    cover=problem.F.any(2)
    remaining=np.ones(len(cover),dtype=bool)
    kept=[0]
    for node in route[1:-1]:
        if np.any(remaining & cover[:,int(node)-1]):
            kept.append(int(node))
            remaining &= ~cover[:,int(node)-1]
    result=kept+[0]
    if remaining.any() or not route_is_valid(problem,result):
        raise ValueError('canonicalization would invalidate complete coverage')
    return result


class ImprovementEnv:
    def __init__(self, problem, horizon=8):
        self.problem, self.horizon = problem, horizon
        self.initial_route = canonical_route(problem,solve_route(problem, 1))
        self.route = list(self.initial_route)
        self.reference_cost = route_objective(problem,self.route)
        self.steps, self.attempted, self.done = 0, set(), False
        self._key = None

    def observe(self):
        key=tuple(self.route)
        if key != self._key:
            self.windows=windows(self.problem,self.route)
            self.features=window_features(self.problem,self.route,self.windows)
            self._signatures=[tuple(self.route[w['start']-1:w['end']+1]) for w in self.windows]
            self._key=key
        mask=np.asarray([s not in self.attempted for s in self._signatures],bool)
        if len(mask):
            mask &= self.features[:,-1]>0
        self.done=self.steps>=self.horizon or not mask.any()
        return self.features.copy(),mask

    def step(self,action):
        _,mask=self.observe()
        if self.done or not isinstance(action,(int,np.integer)) or not 0<=action<len(mask) or not mask[action]:
            raise ValueError('invalid or masked repair action')
        before=route_objective(self.problem,self.route)
        self.attempted.add(self._signatures[action])
        candidate=repair_window(self.problem,self.route,self.windows[action])
        if route_is_valid(self.problem,candidate):
            candidate=canonical_route(self.problem,candidate)
            after=route_objective(self.problem,candidate)
        else:
            after=before
        accepted=after<before-1e-8
        if accepted:
            self.route=list(candidate)
        else:
            after=before
        self.steps+=1
        self.observe()
        return 100*(before-after)/max(self.reference_cost,1e-8), self.done, dict(accepted=accepted,before=before,after=after)
