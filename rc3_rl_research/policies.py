"""Small non-learning selectors over the *same* observed research menu.

These selectors never construct candidates or query the repair solver.  The
adaptive reference changes only operator sampling weights, not acceptance,
candidate caps or the proposal implementation.
"""
from __future__ import annotations

import copy
import math
import numpy as np


def _legal(observation):
    actions, mask = observation['actions'], observation['mask']
    if len(actions) != len(mask):
        raise ValueError('action table and mask lengths differ')
    legal = [i for i, valid in enumerate(mask) if valid]
    if not legal:
        raise ValueError('observation has no permitted action')
    return actions, legal


def _kind(action):
    return action.get('kind', action.get('operator'))


class RandomPolicy:
    """Uniform over the actual legal table, including the explicit STOP."""

    name = 'random'

    def select(self, observation, rng):
        _, legal = _legal(observation)
        return int(rng.choice(legal))

    def update(self, info):
        pass

    def state_dict(self):
        return {'name': self.name}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError('random policy state is incompatible')


class DetourPolicy(RandomPolicy):
    """Select by the cheap menu feature; no all-actions repair lookahead."""

    name = 'detour'

    def select(self, observation, rng):
        actions, legal = _legal(observation)
        non_stop = [i for i in legal if _kind(actions[i]) != 'stop']
        if not non_stop:
            return legal[0]
        feature_names = observation.get('feature_names', [])
        feature_rows = observation.get('features', [])

        def score(index):
            action = actions[index]
            if 'detour_excess' in action:
                return float(action['detour_excess'])
            if 'score' in action:
                return float(action['score'])
            if feature_rows and 'detour_excess' in feature_names:
                return float(feature_rows[index][feature_names.index('detour_excess')])
            return 0.0

        return max(non_stop, key=lambda i: (score(i), -i))


class AdaptiveOperatorPolicy(RandomPolicy):
    """An ALNS-style operator-weight reference using one selected query/step.

    It is deliberately labelled an adaptive operator reference rather than a
    full ALNS implementation. All acceptance is delegated to the environment.
    Fixed scores: best improvement=4, accepted feasible move=1, rejection=0.
    """

    name = 'adaptive_operator'

    def __init__(self, reaction=0.2, minimum_weight=0.05):
        if not 0 < reaction <= 1 or not math.isfinite(reaction):
            raise ValueError('reaction must be in (0, 1]')
        if minimum_weight <= 0 or not math.isfinite(minimum_weight):
            raise ValueError('minimum_weight must be positive')
        self.reaction = float(reaction)
        self.minimum_weight = float(minimum_weight)
        self.weights = {'improve': 1.0, 'replace': 1.0, 'stop': 0.05}
        self.selections = 0

    def select(self, observation, rng):
        actions, legal = _legal(observation)
        kinds = sorted({_kind(actions[i]) for i in legal})
        weights = np.asarray([self.weights.get(kind, 1.0) for kind in kinds])
        selected_kind = kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))]
        pool = [i for i in legal if _kind(actions[i]) == selected_kind]
        self.selections += 1
        return int(rng.choice(pool))

    def update(self, info):
        action = info.get('action', {})
        kind = _kind(action)
        if kind not in ('improve', 'replace'):
            return
        score = 4.0 if info.get('best_improved') else 1.0 if info.get('accepted') else 0.0
        old = self.weights.get(kind, 1.0)
        self.weights[kind] = max(self.minimum_weight,
                                 (1.0 - self.reaction) * old + self.reaction * score)

    def state_dict(self):
        return dict(name=self.name, reaction=self.reaction,
                    minimum_weight=self.minimum_weight,
                    weights=copy.deepcopy(self.weights), selections=self.selections)

    def load_state_dict(self, state):
        if state.get('name') != self.name:
            raise ValueError('adaptive policy state is incompatible')
        candidate = AdaptiveOperatorPolicy(state['reaction'], state['minimum_weight'])
        if any(not math.isfinite(float(v)) or float(v) <= 0 for v in state['weights'].values()):
            raise ValueError('adaptive weights must be positive and finite')
        candidate.weights = {str(k): float(v) for k, v in state['weights'].items()}
        candidate.selections = int(state['selections'])
        self.__dict__.update(candidate.__dict__)


def make_policy(name):
    factories = {'random': RandomPolicy, 'detour': DetourPolicy,
                 'greedy': DetourPolicy, 'adaptive_operator': AdaptiveOperatorPolicy,
                 'alns': AdaptiveOperatorPolicy}
    if name not in factories:
        raise ValueError(f'unknown shared-menu policy: {name}')
    return factories[name]()
