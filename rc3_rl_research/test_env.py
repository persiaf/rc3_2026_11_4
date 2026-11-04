"""G1 transition semantics on explicit synthetic fixtures, not G0 evidence."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from .policies import AdaptiveOperatorPolicy, DetourPolicy, RandomPolicy
from .rl_env import SearchEnv, bootstrap_value, replay_transitions, rollout_record
from .types import Action, ActionMenu, Certificate, ProposalResult, SearchConfig


class FixtureAdapter:
    """Injectable certificate boundary; costs deliberately exercise semantics."""
    problem_id = 'synthetic-environment-semantics-v1'

    def __init__(self):
        self.problem = SimpleNamespace(F=np.ones((1, 6, 2), dtype=bool))
        self.costs = {1: 100., 2: 101., 3: 98., 4: 103., 5: 97., 6: 100.}
        self.anchor = self.certify([0, 1, 0])

    def certify(self, route, method='test'):
        if len(route) != 3 or route[0] != 0 or route[-1] != 0 or route[1] not in self.costs:
            raise ValueError('synthetic invalid route')
        value = self.costs[route[1]]
        return Certificate(list(route), dict(route=list(route),
                            objective_length_plus_stops_m=value,
                            validation=dict(valid_model_plan=True, all_tasks_served=True)), value)

    def payload(self):
        return dict(synthetic=True, problem_id=self.problem_id)


def fixture_menu(problem, route, *, remaining_work, steps_remaining, history,
                 config, problem_id):
    token = json.dumps([problem_id, route, remaining_work, steps_remaining, list(history)])
    actions = [Action('replace', 1, 2, route[1], 'replace'),
               Action('improve', 1, 2, None, 'improve'), Action('stop', key='stop')]
    available = remaining_work > 2 and steps_remaining > 0
    return ActionMenu(actions, [available, available, True], [[1.], [2.], [0.]],
                      ['detour_excess'], token,
                      [token+'replace', token+'improve', token+'stop'], 2)


def fixture_proposer(problem, route, action, *, config):
    # A replace transition 100 -> 101 is legal, followed by improve 101 -> 98.
    target = 2 if action.kind == 'replace' else 3
    return ProposalResult([0, target, 0], True, route[1] != target, 3,
                          counters=dict(attempts=1, candidate_checks=1, dp_states=1, dp_transitions=0))


def make_env(*, proposer=fixture_proposer, certifier=None, seed=7, **config):
    settings = dict(horizon=8, work_budget=1000, work_scale=10., lambda_work=.01)
    settings.update(config)
    return SearchEnv(FixtureAdapter(), SearchConfig(**settings), seed,
                     proposer=proposer, action_builder=fixture_menu, certifier=certifier)


class SearchEnvironmentTests(unittest.TestCase):
    def test_bounded_worse_current_then_better_best(self):
        env = make_env()
        _, reward, done, truncated, info = env.step(0)
        self.assertEqual((env.anchor_cost, env.best_cost, env.current_cost), (100., 100., 101.))
        self.assertTrue(info['accepted'])
        self.assertFalse(info['best_improved'])
        self.assertAlmostEqual(reward, -.005)
        self.assertFalse(done or truncated)
        env.step(1)
        self.assertEqual((env.current_cost, env.best_cost, env.anchor_cost), (98., 98., 100.))
        self.assertIsNot(env.current, env.best)
        self.assertIsNot(env.anchor, env.best)

    def test_acceptance_bound_relative_to_best_not_current(self):
        def proposal(problem, route, action, *, config):
            return ProposalResult([0, 4, 0], True, True, 3)
        env = make_env(proposer=proposal)
        _, _, _, _, info = env.step(0)
        self.assertFalse(info['accepted'])
        self.assertEqual(info['failure_reason'], 'outside_best_plus_epsilon_anchor')
        self.assertEqual(env.current_cost, 100.)
        self.assertEqual(env.work_used, 5)

    def test_certification_is_required_before_acceptance(self):
        adapter = FixtureAdapter()
        def deny(route, method='test'):
            if method == 'rl_search_accept':
                raise ValueError('independent collision witness')
            return adapter.certify(route, method)
        env = make_env(certifier=deny)
        _, _, _, _, info = env.step(1)
        self.assertFalse(info['accepted'])
        self.assertFalse(info['proposal_valid'])
        self.assertEqual(env.best_cost, 100.)
        self.assertEqual(env.work_used, 5)

    def test_failed_and_unchanged_actions_pay_actual_work(self):
        for proposer in (
            lambda p, r, a, **kw: ProposalResult(None, False, False, 7, 'infeasible'),
            lambda p, r, a, **kw: ProposalResult(list(r), True, False, 7)):
            env = make_env(proposer=proposer)
            _, reward, _, _, info = env.step(0)
            self.assertEqual(env.work_used, 9)
            self.assertEqual(info['quality_reward'], 0.)
            self.assertAlmostEqual(reward, -.009)

    def test_stop_terminal_has_no_bootstrap_or_fake_gain(self):
        env = make_env()
        obs = env.observe()
        result = env.step(2)
        record = rollout_record(obs, 2, -1., 8., result,
                                collector_truncated=True, next_value=99.)
        self.assertTrue(record['terminated'])
        self.assertFalse(record['truncated'])
        self.assertEqual(record['bootstrap_value'], 0.)
        self.assertEqual(env.stop_reason, 'stop')
        self.assertEqual(env.total_quality_reward, 0.)
        self.assertEqual(env.counters['proposal_calls'], 0)
        self.assertEqual(env.observe()['mask'], [])
        with self.assertRaises(RuntimeError):
            env.step(0)

    def test_budget_horizon_and_menu_exhaustion_are_true_ends(self):
        env = make_env(work_budget=4)
        _, _, done, trunc, _ = env.step(0)
        self.assertTrue(done)
        self.assertFalse(trunc)
        self.assertEqual(env.stop_reason, 'budget')
        self.assertEqual(env.finalization['work_overrun'], 1)
        horizon = make_env(horizon=1)
        self.assertTrue(horizon.step(0)[2])
        self.assertEqual(horizon.stop_reason, 'horizon')
        menu_only = make_env(work_budget=2)
        self.assertEqual(menu_only.observe()['mask'], [False, False, True])
        self.assertTrue(menu_only.step(2)[2])
        self.assertEqual(menu_only.counters['proposal_calls'], 0)
        self.assertEqual(menu_only.work_used, 2)
        self.assertEqual(menu_only.stop_reason, 'budget')

    def test_zero_budget_or_horizon_starts_finished(self):
        for settings in (dict(work_budget=0), dict(horizon=0)):
            env = make_env(**settings)
            self.assertTrue(env.terminated)
            self.assertEqual(env.observe()['actions'], [])
            self.assertEqual(env.finalize()[0].cost, 100.)

    def test_potential_and_recovery_cycle_cannot_generate_quality(self):
        def cycle(problem, route, action, *, config):
            return ProposalResult([0, 2 if route[1] == 1 else 1, 0], True, True, 3)
        env = make_env(proposer=cycle, potential_weight=.5)
        first = env.step(0)[1]
        second = env.step(0)[1]
        self.assertLess(first, 0.)
        self.assertGreater(second, 0.)  # restoring the negative potential, not new quality
        env.step(2)
        self.assertAlmostEqual(env.total_quality_reward, 0.)
        self.assertAlmostEqual(env.total_shaping_reward, 0.)
        self.assertAlmostEqual(env.total_reward, -env.total_work_penalty)

    def test_terminal_potential_is_zero_even_at_worse_current(self):
        env = make_env(potential_weight=.5)
        env.step(0)
        self.assertNotEqual(env.total_shaping_reward, 0.)
        env.step(2)
        self.assertAlmostEqual(env.total_shaping_reward, 0.)

    def test_final_verification_fallback_corrects_actual_return_reward(self):
        adapter = FixtureAdapter()
        def final_fail(route, method='test'):
            if method == 'rl_search_final':
                raise ValueError('synthetic final export failure')
            return adapter.certify(route, method)
        env = make_env(certifier=final_fail, potential_weight=.5)
        env.step(1)
        self.assertEqual(env.total_quality_reward, 2.)
        _, reward, _, _, info = env.step(2)
        self.assertLess(reward, -2.)
        self.assertTrue(info['finalization']['fallback_used'])
        self.assertEqual(env.finalize()[0].cost, 100.)
        self.assertEqual(env.best_cost, 98.)
        self.assertAlmostEqual(env.total_quality_reward, 0.)
        self.assertAlmostEqual(env.total_reward, -env.total_work_penalty)

    def test_collector_truncation_retains_state_and_bootstraps(self):
        env = make_env()
        obs = env.observe()
        result = env.step(0)
        record = rollout_record(obs, 0, -1.3, .5, result,
                                collector_truncated=True, next_value=6.25)
        state = env.state_dict()
        boundary = env.rollout_boundary(6.25)
        self.assertTrue(record['truncated'] and boundary['truncated'])
        self.assertFalse(record['terminated'] or env.terminated)
        self.assertEqual(record['bootstrap_value'], 6.25)
        self.assertEqual(state, env.state_dict())
        later = env.observe()
        self.assertNotEqual(obs['actions'], later['actions'])
        self.assertEqual(record['observation']['actions'], obs['actions'])
        with self.assertRaises(ValueError):
            rollout_record(later, 0, -1.3, .5, result)
        with self.assertRaises(ValueError):
            rollout_record(obs, 1, -1.3, .5, result)

    def test_observation_is_cached_and_cannot_mutate_environment(self):
        env = make_env()
        first = env.observe()
        self.assertEqual(first, env.observe())
        self.assertEqual(env.counters['menu_builds'], 1)
        first['mask'][0] = False
        first['current'][1] = 4
        self.assertTrue(env.observe()['mask'][0])
        self.assertEqual(env.current, [0, 1, 0])
        env.step(0)
        self.assertEqual(env.counters['menu_work'], 2)

    def test_save_load_order_history_rng_and_replay(self):
        env = make_env()
        env.step(0)
        state = json.loads(json.dumps(env.state_dict()))
        clone = make_env(seed=999)
        clone.load_state_dict(state)
        self.assertEqual(clone.observe(), env.observe())
        self.assertEqual(clone.history, env.history)
        self.assertEqual(int(clone.rng.integers(1000000)), int(env.rng.integers(1000000)))
        env.step(1)
        env.step(2)
        records = copy.deepcopy(env.transitions)
        replay = make_env()
        replay_transitions(replay, records)
        self.assertEqual(replay.best, env.best)
        self.assertEqual(replay.work_used, env.work_used)
        self.assertAlmostEqual(replay.total_reward, env.total_reward)
        bad = copy.deepcopy(state)
        bad['problem_id'] = 'other-problem'
        with self.assertRaises(ValueError):
            clone.load_state_dict(bad)

    def test_invalid_action_and_unreported_solver_exception(self):
        env = make_env()
        for action in (-1, True, 999):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.work_used, 0)
        def crash(*args, **kwargs):
            raise RuntimeError('internal error without counters')
        broken = make_env(proposer=crash)
        _, _, done, _, info = broken.step(0)
        self.assertTrue(done and info['work_count_incomplete'])
        self.assertEqual(broken.stop_reason, 'proposal_error')
        self.assertEqual(broken.finalize()[0].cost, 100.)

    def test_shared_menu_policies_do_not_query_proposer(self):
        env = make_env()
        obs = env.observe()
        for selector in (DetourPolicy(), RandomPolicy(), AdaptiveOperatorPolicy()):
            action = selector.select(obs, env.rng)
            self.assertTrue(obs['mask'][action])
        self.assertEqual(DetourPolicy().select(obs, env.rng), 1)
        self.assertEqual(env.counters['proposal_calls'], 0)
        policy = AdaptiveOperatorPolicy()
        policy.update(dict(action={'kind':'replace'}, accepted=True, best_improved=False))
        clone = AdaptiveOperatorPolicy()
        clone.load_state_dict(policy.state_dict())
        self.assertEqual(clone.state_dict(), policy.state_dict())

    def test_runtime_termination_debits_pending_menu_and_telescopes(self):
        env = make_env(potential_weight=.4)
        env.step(0)
        reward, event = env.terminate('external_deadline')
        self.assertEqual(event['menu_work_units'], 2)
        self.assertEqual(env.work_used, 7)
        self.assertTrue(env.terminated)
        self.assertAlmostEqual(env.total_shaping_reward, 0.)
        self.assertAlmostEqual(env.total_reward, -env.total_work_penalty)
        self.assertEqual(bootstrap_value(10., terminated=True), 0.)
        self.assertEqual(bootstrap_value(10., terminated=False, truncated=True), 10.)

    def test_runtime_artifact_replay_rejects_plan_reward_policy_tampering(self):
        from . import runtime_rl
        directory = Path(__file__).parent/'outputs'/'test_env_artifacts'
        directory.mkdir(parents=True, exist_ok=True)
        evidence_path = directory/'synthetic_runtime.json'
        actual_env_class = SearchEnv
        def environment(adapter, config, seed):
            return actual_env_class(adapter, config, seed, proposer=fixture_proposer,
                                    action_builder=fixture_menu)
        config = SearchConfig(horizon=2, work_budget=1000, work_scale=10., lambda_work=.01)
        with patch.object(runtime_rl, 'SearchEnv', environment), \
             patch('rc3_rl_research.adapter.ResearchAdapter.from_payload', return_value=FixtureAdapter()):
            runtime_rl.run(FixtureAdapter(), method='alns', config=config,
                           seed=71, output_path=evidence_path)
            result = runtime_rl.replay_file(evidence_path)
            self.assertTrue(result['replay_valid'] and result['exported_plan_verified']
                            and result['policy_state_verified'])
            original = json.loads(evidence_path.read_text(encoding='utf-8'))
            for kind in ('plan', 'reward', 'policy', 'selection'):
                bad = copy.deepcopy(original)
                if kind == 'plan':
                    bad['plan']['objective_length_plus_stops_m'] -= 10
                elif kind == 'reward':
                    bad['state']['total_quality_reward'] += 1
                elif kind == 'policy':
                    bad['policy']['selections'] += 1
                else:
                    bad['selections'] = []
                destination = directory/f'tampered_{kind}.json'
                destination.write_text(json.dumps(bad), encoding='utf-8')
                with self.assertRaises(ValueError):
                    runtime_rl.replay_file(destination)


if __name__ == '__main__':
    unittest.main()
