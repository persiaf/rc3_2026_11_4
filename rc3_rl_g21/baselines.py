"""Causal G2.1 controls over the existing shared, masked action menu.

This module creates no environment and never evaluates a proposal.  Its only
feedback is the ``info`` of actions already committed by the caller.  In
particular, saved future-prefix returns are not accepted as policy inputs.
"""
from __future__ import annotations

import copy
import math
from numbers import Integral, Real


STATE_SCHEMA = "rc3-g21-causal-rule-v1"

# Predeclared development-only grid.  These are not fitted parameters/results.
DETOUR_EARLY_GRID = (
    {"max_proposals": 1, "patience": 1},
    {"max_proposals": 2, "patience": 1},
    {"max_proposals": 4, "patience": 1},
    {"max_proposals": 4, "patience": 2},
)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("invalid " + name)
    return value


def _number(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("invalid " + name)
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError("invalid " + name)
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError("invalid " + name)
    return value


def _menu(observation):
    actions, mask = observation["actions"], observation["mask"]
    if len(actions) != len(mask):
        raise ValueError("action table and mask lengths differ")
    legal = [i for i, permitted in enumerate(mask) if permitted]
    stops = [i for i in legal if actions[i]["kind"] == "stop"]
    if len(stops) != 1:
        raise ValueError("exactly one legal STOP is required")
    return actions, legal, stops[0]


def _detour(observation, legal, stop):
    """Same cheap feature and first-index tie break as frozen DetourPolicy."""
    actions = observation["actions"]
    candidates = [i for i in legal if i != stop]
    if not candidates:
        return stop
    names = observation.get("feature_names", [])
    rows = observation.get("features", [])

    def score(index):
        action = actions[index]
        if "detour_excess" in action:
            value = action["detour_excess"]
        elif "score" in action:
            value = action["score"]
        elif len(rows) and "detour_excess" in names:
            value = rows[index][names.index("detour_excess")]
        else:
            value = 0.0
        return _number(value, "detour feature")

    return max(candidates, key=lambda index: (score(index), -index))


class DetourEarlyPolicy:
    """Detour ranking with paid, realized net-gain stagnation and a hard cap.

    A completed proposal resets stagnation only if it changed the route,
    improved certified best, and ``quality_reward - work_penalty`` exceeds
    ``progress_tolerance``.  STOP keeps its ordinary shared-environment cost.
    No future benefit or candidate repair is computed by this selector.
    """

    name = "detour_early"

    def __init__(self, max_proposals=4, patience=1, progress_tolerance=1e-12):
        self.max_proposals = _integer(max_proposals, "proposal cap", 1)
        self.patience = _integer(patience, "patience", 1)
        self.progress_tolerance = _number(progress_tolerance, "progress tolerance", 0)
        self.proposals = self.stagnation = self.paid_work = 0
        self.total_quality = self.total_work_penalty = 0.0
        self.stopped = False

    def select(self, observation, rng):
        actions, legal, stop = _menu(observation)
        if (self.stopped or self.proposals >= self.max_proposals
                or self.stagnation >= self.patience):
            return stop
        return _detour(observation, legal, stop)

    def update(self, info):
        if self.stopped:
            raise ValueError("cannot update a stopped policy")
        kind = info["action"]["kind"]
        if kind not in ("improve", "replace", "stop"):
            raise ValueError("unknown action kind")
        if kind != "stop" and self.proposals >= self.max_proposals:
            raise ValueError("committed proposal cap already reached")
        work = _integer(info["work_units"], "paid work")
        quality = _number(info["quality_reward"], "realized quality")
        penalty = _number(info["work_penalty"], "paid work penalty", 0)
        improved = _boolean(info["best_improved"], "best improvement")
        changed = _boolean(info["route_changed"], "route change")
        # Validate before committing state; malformed feedback is not progress.
        self.paid_work += work
        self.total_quality += quality
        self.total_work_penalty += penalty
        if kind == "stop":
            self.stopped = True
        else:
            self.proposals += 1
            progress = (improved and changed
                        and quality - penalty > self.progress_tolerance)
            self.stagnation = 0 if progress else self.stagnation + 1

    def state_dict(self):
        return dict(schema=STATE_SCHEMA, name=self.name,
                    max_proposals=self.max_proposals, patience=self.patience,
                    progress_tolerance=self.progress_tolerance,
                    proposals=self.proposals, stagnation=self.stagnation,
                    paid_work=self.paid_work, total_quality=self.total_quality,
                    total_work_penalty=self.total_work_penalty, stopped=self.stopped)

    def load_state_dict(self, state):
        if state.get("schema") != STATE_SCHEMA or state.get("name") != self.name:
            raise ValueError("wrong policy state identity")
        candidate = DetourEarlyPolicy(state["max_proposals"], state["patience"],
                                      state["progress_tolerance"])
        for name in ("proposals", "stagnation", "paid_work"):
            setattr(candidate, name, _integer(state[name], name))
        if candidate.stagnation > candidate.proposals:
            raise ValueError("stagnation exceeds committed proposals")
        if candidate.proposals > candidate.max_proposals:
            raise ValueError("committed proposals exceed cap")
        candidate.total_quality = _number(state["total_quality"], "total quality")
        candidate.total_work_penalty = _number(state["total_work_penalty"], "total penalty", 0)
        candidate.stopped = _boolean(state["stopped"], "stopped")
        self.__dict__.update(candidate.__dict__)


class OneStepPolicy:
    """Allow a base selector at root, then STOP after one committed proposal.

    The base selector may choose STOP at root: STOP is never removed from its
    menu.  A learned selector can be injected through the same select/update/
    state interface; it must score already-built observations, not query the
    solver.  Frozen model weights are the caller's responsibility.
    """

    name = "one_step"

    def __init__(self, base_name="detour_memory", base_kwargs=None, selector=None):
        if selector is not None and base_kwargs:
            raise ValueError("base_kwargs cannot accompany a supplied selector")
        if selector is None:
            if base_name in ("one_step", "detour_early"):
                raise ValueError("one-step base must be an unwrapped selector")
            from rc3_rl_g2.baselines import make_policy as make_g2_policy
            selector = make_g2_policy(base_name, **(base_kwargs or {}))
        self.selector = selector
        self.proposals = 0
        self.stopped = False

    def select(self, observation, rng):
        _, legal, stop = _menu(observation)
        if self.stopped or self.proposals:
            return stop
        action = self.selector.select(observation, rng)
        if isinstance(action, bool) or not isinstance(action, Integral) or action not in legal:
            raise ValueError("base selector returned an illegal action")
        return int(action)

    def update(self, info):
        if self.stopped:
            raise ValueError("cannot update a stopped policy")
        kind = info["action"]["kind"]
        if kind not in ("improve", "replace", "stop"):
            raise ValueError("unknown action kind")
        if kind != "stop" and self.proposals:
            raise ValueError("one-step policy cannot commit a second proposal")
        self.selector.update(info)
        self.proposals += int(kind != "stop")
        self.stopped = kind == "stop"

    def state_dict(self):
        return dict(schema=STATE_SCHEMA, name=self.name, proposals=self.proposals,
                    stopped=self.stopped, selector=copy.deepcopy(self.selector.state_dict()))

    def load_state_dict(self, state):
        if state.get("schema") != STATE_SCHEMA or state.get("name") != self.name:
            raise ValueError("wrong policy state identity")
        proposals = _integer(state["proposals"], "one-step proposals")
        if proposals > 1:
            raise ValueError("one-step proposal cap exceeded")
        stopped = _boolean(state["stopped"], "stopped")
        selector = copy.deepcopy(self.selector)
        selector.load_state_dict(copy.deepcopy(state["selector"]))
        self.selector, self.proposals, self.stopped = selector, proposals, stopped


def make_policy(name, **kwargs):
    """Drop-in factory for new G2.1 evaluators; frozen G2 is left untouched."""
    if name == "detour_early":
        return DetourEarlyPolicy(**kwargs)
    if name == "one_step":
        return OneStepPolicy(**kwargs)
    from rc3_rl_g2.baselines import make_policy as make_g2_policy
    return make_g2_policy(name, **kwargs)
