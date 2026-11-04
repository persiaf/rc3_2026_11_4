"""Certified non-monotone search state, finite work budget and replay records.

This is a G1 environment, not a PPO implementation. Observations are JSON data;
the exact ordered menu used for a transition is kept in that transition.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np

from .types import SCHEMA, Action, ActionMenu, Certificate, ProposalResult, SearchConfig


def _copy(value):
    return copy.deepcopy(value)


def _menu_dict(menu):
    return asdict(menu)


def _menu_from_dict(value):
    data = _copy(value)
    data['actions'] = [Action(**a) for a in data['actions']]
    return ActionMenu(**data)


def bootstrap_value(next_value, *, terminated, truncated=False):
    """A collector cut retains V(s_next); actual task termination does not."""
    if not math.isfinite(float(next_value)):
        raise ValueError('bootstrap value must be finite')
    return 0.0 if terminated else float(next_value)


def rollout_record(observation, action_id, old_log_probability, value,
                   step_result, *, collector_truncated=False, next_value=0.0):
    """Store old action order/mask; never reconstruct them from a later state."""
    next_observation, reward, terminated, truncated, info = step_result
    if not (math.isfinite(float(old_log_probability)) and math.isfinite(float(value))):
        raise ValueError('sample log probability and value must be finite')
    if type(action_id) is not int or not 0 <= action_id < len(observation['mask']):
        raise ValueError('invalid sampled action index')
    if not observation['mask'][action_id]:
        raise ValueError('sampled action was masked')
    if info.get('sampled_observation') != observation:
        raise ValueError('rollout observation differs from the menu actually sampled')
    if info.get('action_id') != action_id or info.get('action') != observation['actions'][action_id]:
        raise ValueError('rollout action differs from the action actually executed')
    return dict(observation=_copy(observation), action_id=action_id,
                action=_copy(observation['actions'][action_id]),
                old_log_probability=float(old_log_probability), value=float(value),
                reward=float(reward), terminated=bool(terminated),
                truncated=bool(not terminated and (truncated or collector_truncated)),
                next_observation=_copy(next_observation),
                bootstrap_value=bootstrap_value(next_value, terminated=terminated),
                info=_copy(info))


class SearchEnv:
    def __init__(self, adapter, config=SearchConfig(), seed=0, *,
                 proposer=None, action_builder=None, certifier=None):
        from .action_builder import build_actions
        from .repair_proposal import propose
        self.adapter, self.config = adapter, config
        self.proposer = propose if proposer is None else proposer
        self.action_builder = build_actions if action_builder is None else action_builder
        self.certifier = adapter.certify if certifier is None else certifier
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.anchor_certificate = _copy(adapter.anchor)
        self.anchor = list(self.anchor_certificate.route)
        self.current, self.best = list(self.anchor), list(self.anchor)
        self.anchor_cost = float(self.anchor_certificate.cost)
        if not math.isfinite(self.anchor_cost) or self.anchor_cost < 0:
            raise ValueError('anchor must have a finite nonnegative certified cost')
        self.current_cost = self.best_cost = self.anchor_cost
        self.best_certificate = _copy(self.anchor_certificate)
        self.steps = self.work_used = 0
        self.history = []
        self.transitions = []
        self.events = []
        self._menu = None
        self.terminated = False
        self.stop_reason = None
        self.returned_certificate = None
        self.finalization = None
        self.total_reward = self.total_quality_reward = self.total_work_penalty = 0.0
        self.total_shaping_reward = 0.0
        self.counters = dict(menu_builds=0, menu_work=0, proposal_calls=0,
                             proposal_work=0, certification_calls=0, accepted=0,
                             best_improvements=0, work_count_incomplete=False)
        self.timings = dict(menu_s=0.0, proposal_s=0.0, certification_s=0.0,
                            final_validation_s=0.0)
        if config.horizon == 0 or config.work_budget == 0 or self.anchor_cost == 0.0:
            reason = 'horizon' if config.horizon == 0 else 'budget' if config.work_budget == 0 else 'empty_tasks'
            self._finish(reason)

    @property
    def remaining_work(self):
        return max(0, self.config.work_budget - self.work_used)

    @property
    def done(self):
        return self.terminated

    def _potential(self):
        return 100.0 * (self.best_cost - self.current_cost) / max(self.anchor_cost, 1e-8)

    def _ensure_menu(self):
        if self._menu is None and not self.terminated:
            tick = time.perf_counter()
            self._menu = self.action_builder(
                self.adapter.problem, self.current, remaining_work=self.remaining_work,
                steps_remaining=self.config.horizon - self.steps, history=tuple(self.history),
                config=self.config, problem_id=self.adapter.problem_id)
            self.timings['menu_s'] += time.perf_counter() - tick
            menu = self._menu
            n = len(menu.actions)
            if (n == 0 or len(menu.mask) != n or len(menu.features) != n or
                    len(menu.action_context_keys) != n or type(menu.work_units) is not int or
                    menu.work_units < 0):
                raise ValueError('invalid action menu contract')
            stops = [i for i, a in enumerate(menu.actions) if a.kind == 'stop']
            if len(stops) != 1 or not menu.mask[stops[0]]:
                raise ValueError('an active menu requires exactly one legal STOP')
            self.counters['menu_builds'] += 1
        return self._menu

    def observe(self):
        menu = self._ensure_menu()
        if menu is None:
            action_data, mask, features, names, context, keys = [], [], [], [], None, []
            pending = 0
        else:
            action_data = [a.to_dict() for a in menu.actions]
            mask, features, names = _copy(menu.mask), _copy(menu.features), list(menu.feature_names)
            context, keys, pending = menu.context_key, list(menu.action_context_keys), menu.work_units
        problem = self.adapter.problem
        cover = problem.F.any(2)
        selected = np.asarray(self.current[1:-1], dtype=np.int64)
        counts = (cover[:, selected-1].sum(1) if len(selected)
                  else np.zeros(cover.shape[0], dtype=np.int64))
        return dict(schema=SCHEMA, problem_id=self.adapter.problem_id,
                    actions=action_data, mask=mask, features=features, feature_names=names,
                    menu_context_key=context, action_context_keys=keys,
                    current=list(self.current), best=list(self.best), anchor=list(self.anchor),
                    current_cost=self.current_cost, best_cost=self.best_cost,
                    anchor_cost=self.anchor_cost, current_best_gap=self.current_cost-self.best_cost,
                    task_coverage_counts=counts.tolist(), task_count=int(cover.shape[0]),
                    candidate_count=int(cover.shape[1]), steps=self.steps,
                    steps_remaining=max(0, self.config.horizon-self.steps),
                    remaining_work=self.remaining_work, work_budget=self.config.work_budget,
                    work_remaining_fraction=self.remaining_work/max(self.config.work_budget, 1),
                    pending_menu_work=pending, work_after_menu=max(0, self.remaining_work-pending),
                    history=list(self.history), terminated=self.terminated,
                    stop_reason=self.stop_reason)

    def _finish(self, reason):
        """Independently verify best; return correction relative to stored best."""
        if self.terminated:
            return 0.0
        tick = time.perf_counter()
        fallback, error = False, None
        self.counters['certification_calls'] += 1
        try:
            certificate = self.certifier(list(self.best), method='rl_search_final')
            if (not math.isfinite(certificate.cost) or
                    abs(certificate.cost-self.best_cost) > 1e-7 or
                    certificate.cost > self.anchor_cost+1e-8):
                raise ValueError('final certificate cost disagrees with certified best')
        except Exception as exc:
            certificate = _copy(self.anchor_certificate)
            fallback, error = True, f'{type(exc).__name__}: {exc}'
        self.timings['final_validation_s'] += time.perf_counter() - tick
        self.returned_certificate = _copy(certificate)
        correction = 100.0 * (self.best_cost-certificate.cost) / max(self.anchor_cost, 1e-8)
        self.terminated, self.stop_reason = True, str(reason)
        self._menu = None
        self.finalization = dict(reason=reason, fallback_used=fallback, failure_reason=error,
                                 returned_cost=float(certificate.cost), best_cost=self.best_cost,
                                 quality_correction=correction,
                                 work_overrun=max(0, self.work_used-self.config.work_budget))
        return correction

    def _accumulate(self, quality, penalty, shaping):
        reward = quality - penalty + shaping
        self.total_quality_reward += quality
        self.total_work_penalty += penalty
        self.total_shaping_reward += shaping
        self.total_reward += reward
        return reward

    def step(self, action_id):
        if self.terminated:
            raise RuntimeError('cannot step a terminated search; reset or load another state')
        observation = self.observe()
        menu = self._menu
        if (not isinstance(action_id, (int, np.integer)) or isinstance(action_id, (bool, np.bool_)) or
                not 0 <= int(action_id) < len(menu.actions) or not menu.mask[int(action_id)]):
            raise ValueError('selected action is outside the saved legal menu')
        action_id = int(action_id)
        action = menu.actions[action_id]
        rng_before = _copy(self.rng.bit_generator.state)
        before_current, before_best = self.current_cost, self.best_cost
        phi_before = self._potential()
        menu_work = menu.work_units
        proposal_work, accepted, best_improved, proposal_valid = 0, False, False, False
        failure, proposal, certificate = None, None, None
        reason = None
        self.counters['menu_work'] += menu_work
        if action.kind == 'stop':
            reason = 'budget' if self.remaining_work <= menu_work else 'stop'
        elif self.remaining_work <= menu_work:
            reason, failure = 'budget', 'menu_exhausted_budget_before_proposal'
        else:
            self.counters['proposal_calls'] += 1
            self.history.append(menu.action_context_keys[action_id])
            tick = time.perf_counter()
            certification_before = self.timings['certification_s']
            try:
                proposal = self.proposer(self.adapter.problem, list(self.current), action,
                                         config=self.config)
                if (not isinstance(proposal, ProposalResult) or
                        type(proposal.work_units) is not int or proposal.work_units < 1):
                    raise ValueError('proposal must return positive actual work_units')
                proposal_work = proposal.work_units
                failure = proposal.failure_reason
                if proposal.valid and proposal.route is not None:
                    cert_tick = time.perf_counter()
                    self.counters['certification_calls'] += 1
                    try:
                        certificate = self.certifier(proposal.route, method='rl_search_accept')
                        if not math.isfinite(certificate.cost) or certificate.cost < 0:
                            raise ValueError('proposal certificate has invalid cost')
                        proposal_valid = True
                    except Exception as exc:
                        failure = f'certification_failed: {type(exc).__name__}: {exc}'
                    finally:
                        self.timings['certification_s'] += time.perf_counter() - cert_tick
                if certificate is not None and proposal_valid:
                    bound = self.best_cost + self.config.epsilon*self.anchor_cost
                    if certificate.cost <= bound+1e-8:
                        self.current, self.current_cost = list(certificate.route), float(certificate.cost)
                        accepted = True
                        self.counters['accepted'] += 1
                        if certificate.cost < self.best_cost-1e-8:
                            self.best, self.best_cost = list(certificate.route), float(certificate.cost)
                            self.best_certificate = _copy(certificate)
                            best_improved = True
                            self.counters['best_improvements'] += 1
                    else:
                        failure = 'outside_best_plus_epsilon_anchor'
            except Exception as exc:
                # Expected solver failures must be ProposalResult objects. An
                # unexpected exception has unknown internal work; never hide it.
                failure = f'proposal_exception: {type(exc).__name__}: {exc}'
                self.counters['work_count_incomplete'] = True
                reason = 'proposal_error'
            finally:
                self.timings['proposal_s'] += max(0., time.perf_counter()-tick-
                    (self.timings['certification_s']-certification_before))
        self.steps += 1
        self.work_used += menu_work+proposal_work
        self.counters['proposal_work'] += proposal_work
        self._menu = None
        if reason is None:
            reason = 'budget' if self.remaining_work == 0 else 'horizon' if self.steps >= self.config.horizon else None
        quality = 100.0*(before_best-self.best_cost)/max(self.anchor_cost, 1e-8)
        if reason is not None:
            quality += self._finish(reason)
        shaping = self.config.potential_weight*((0.0 if self.terminated else self._potential())-phi_before)
        penalty = self.config.lambda_work*(menu_work+proposal_work)/self.config.work_scale
        reward = self._accumulate(quality, penalty, shaping)
        info = dict(action_id=action_id, action=action.to_dict(),
                    sampled_observation=_copy(observation), current_cost=self.current_cost,
                    best_cost=self.best_cost, previous_current_cost=before_current,
                    previous_best_cost=before_best, proposal_valid=proposal_valid,
                    proposal_cost=None if certificate is None else float(certificate.cost),
                    accepted=accepted, best_improved=best_improved,
                    work_units=menu_work+proposal_work, menu_work_units=menu_work,
                    proposal_work_units=proposal_work, failure_reason=failure,
                    remaining_budget=self.remaining_work, quality_reward=quality,
                    work_penalty=penalty, shaping_reward=shaping,
                    proposal=None if proposal is None else asdict(proposal),
                    finalization=_copy(self.finalization),
                    work_count_incomplete=self.counters['work_count_incomplete'])
        next_observation = self.observe()
        transition = dict(observation=_copy(observation), action_id=action_id,
                          action=action.to_dict(), reward=reward, terminated=self.terminated,
                          truncated=False, next_observation=_copy(next_observation),
                          info=_copy(info), rng_before=rng_before,
                          rng_after=_copy(self.rng.bit_generator.state))
        self.transitions.append(transition)
        return next_observation, reward, self.terminated, False, info

    def terminate(self, reason='external_deadline'):
        """True runtime end, distinct from a collector's rollout boundary."""
        if self.terminated:
            return 0.0, _copy(self.finalization)
        phi_before = self._potential()
        pending_observation = self.observe() if self._menu is not None else None
        pending = self._menu.work_units if self._menu is not None else 0
        self.work_used += pending
        self.counters['menu_work'] += pending
        quality = self._finish(reason)
        shaping = -self.config.potential_weight*phi_before
        penalty = self.config.lambda_work*pending/self.config.work_scale
        reward = self._accumulate(quality, penalty, shaping)
        event = dict(event='runtime_termination', reason=reason, reward=reward,
                     quality_reward=quality, shaping_reward=shaping, work_penalty=penalty,
                     menu_work_units=pending, after_steps=self.steps,
                     observation_before_termination=pending_observation,
                     rng_state=_copy(self.rng.bit_generator.state),
                     finalization=_copy(self.finalization))
        self.events.append(event)
        return reward, event

    def finalize(self):
        if not self.terminated:
            self.terminate('explicit_finalize')
        return _copy(self.returned_certificate), _copy(self.finalization)

    def rollout_boundary(self, next_value):
        """Take a collector snapshot without changing current/best/budget/history."""
        return dict(observation=self.observe(), terminated=self.terminated,
                    truncated=not self.terminated,
                    bootstrap_value=bootstrap_value(next_value, terminated=self.terminated))

    def state_dict(self):
        return dict(schema=SCHEMA, problem_id=self.adapter.problem_id,
                    config=asdict(self.config), seed=self.seed,
                    anchor_certificate=asdict(self.anchor_certificate),
                    best_certificate=asdict(self.best_certificate),
                    returned_certificate=None if self.returned_certificate is None else asdict(self.returned_certificate),
                    anchor=list(self.anchor), current=list(self.current), best=list(self.best),
                    anchor_cost=self.anchor_cost, current_cost=self.current_cost, best_cost=self.best_cost,
                    steps=self.steps, work_used=self.work_used, history=list(self.history),
                    menu=None if self._menu is None else _menu_dict(self._menu),
                    terminated=self.terminated, stop_reason=self.stop_reason,
                    finalization=_copy(self.finalization), transitions=_copy(self.transitions),
                    events=_copy(self.events), rng_state=_copy(self.rng.bit_generator.state),
                    total_reward=self.total_reward, total_quality_reward=self.total_quality_reward,
                    total_work_penalty=self.total_work_penalty,
                    total_shaping_reward=self.total_shaping_reward,
                    counters=_copy(self.counters), timings=_copy(self.timings))

    def load_state_dict(self, state):
        state = _copy(state)
        if state.get('schema') != SCHEMA or state.get('problem_id') != self.adapter.problem_id:
            raise ValueError('saved state schema/problem identity mismatch')
        if state.get('config') != asdict(self.config):
            raise ValueError('saved state config mismatch')
        if state['anchor'] != self.anchor or abs(state['anchor_cost']-self.anchor_cost) > 1e-8:
            raise ValueError('saved state anchor differs from this certified anchor')
        checked = {}
        for key in ('anchor', 'current', 'best'):
            certificate = self.adapter.certify(state[key], method='rl_search_load_check')
            if abs(certificate.cost-state[key+'_cost']) > 1e-7 or certificate.route != state[key]:
                raise ValueError('saved route/cost failed independent verification')
            checked[key] = certificate
        if (state['best_cost'] > state['anchor_cost']+1e-8 or
                state['current_cost'] > state['best_cost']+self.config.epsilon*state['anchor_cost']+1e-8):
            raise ValueError('saved state violates best/acceptance bounds')
        if state['steps'] < 0 or state['work_used'] < 0:
            raise ValueError('saved counters must be nonnegative')
        for key in ('seed', 'anchor', 'current', 'best', 'anchor_cost', 'current_cost', 'best_cost',
                    'steps', 'work_used', 'history', 'terminated', 'stop_reason', 'finalization',
                    'transitions', 'events', 'total_reward', 'total_quality_reward',
                    'total_work_penalty', 'total_shaping_reward', 'counters', 'timings'):
            setattr(self, key, state[key])
        self.anchor_certificate = _copy(checked['anchor'])
        self.best_certificate = _copy(checked['best'])
        returned = state['returned_certificate']
        self.returned_certificate = None
        if returned is not None:
            # Preserve the original export label while rechecking its content;
            # otherwise an initially terminated empty scene changes plan.method.
            method = returned['plan'].get('method', 'rl_search_load_check')
            verified = self.adapter.certify(returned['route'], method=method)
            if abs(verified.cost-returned['cost']) > 1e-7:
                raise ValueError('saved returned certificate cost mismatch')
            if verified.plan != returned['plan']:
                raise ValueError('saved returned plan differs from independently verified plan')
            self.returned_certificate = verified
        self._menu = None if state['menu'] is None else _menu_from_dict(state['menu'])
        self.rng.bit_generator.state = state['rng_state']
        # Loading must not eagerly construct another menu or spend hidden work.
        return self.state_dict()

    def save(self, path, *, policy=None):
        payload = dict(schema=SCHEMA, adapter=self.adapter.payload(), state=self.state_dict(),
                       policy=None if policy is None else policy.state_dict())
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        return target


def load_environment(path):
    from .adapter import ResearchAdapter
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    adapter = ResearchAdapter.from_payload(payload['adapter'])
    state = payload['state']
    env = SearchEnv(adapter, SearchConfig(**state['config']), state['seed'])
    env.load_state_dict(state)
    return env, payload.get('policy')


def replay_transitions(env, transitions, *, atol=1e-8):
    """Reexecute ordered saved actions and compare menus, budgets and rewards.

    The caller supplies an environment at the saved prefix state. RNG state is
    restored to the recorded decision point, so prior stochastic selection does
    not silently change later continuation. Actual proposal queries run again.
    """
    for index, record in enumerate(transitions):
        observed = env.observe()
        if observed != record['observation']:
            raise ValueError(f'replay observation/menu mismatch at transition {index}')
        env.rng.bit_generator.state = _copy(record['rng_before'])
        next_obs, reward, terminated, truncated, info = env.step(record['action_id'])
        if abs(reward-record['reward']) > atol:
            raise ValueError(f'replay reward mismatch at transition {index}')
        if (next_obs != record['next_observation'] or terminated != record['terminated'] or
                truncated != record['truncated']):
            raise ValueError(f'replay next state mismatch at transition {index}')
        for key in ('accepted', 'proposal_valid', 'best_improved', 'work_units',
                    'remaining_budget', 'failure_reason', 'finalization'):
            if info[key] != record['info'][key]:
                raise ValueError(f'replay {key} mismatch at transition {index}')
        if env.rng.bit_generator.state != record['rng_after']:
            raise ValueError(f'replay RNG mismatch at transition {index}')
    return env
