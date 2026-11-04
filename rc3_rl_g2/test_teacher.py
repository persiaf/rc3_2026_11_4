"""Teacher-target risks: evidence masks, terminal costs, and scene weighting."""
import copy
from types import SimpleNamespace
import unittest

import numpy as np

from rc3_rl_research.types import SearchConfig
from .teacher_dataset import (assign_scene_route_weights, finite_tree,
                              make_labels, queried_actions, semantic_action)


class TinyTree:
    """Synthetic semantic fixture, not evidence of planning performance."""
    def __init__(self):
        self.adapter = SimpleNamespace(problem_id='synthetic-teacher')
        self.config = SearchConfig()
        self.done = False
        self.path = []
        self.current = self.best = [0, 1, 0]
        self.current_cost = self.best_cost = 10.
        self.total_reward = 0.
        self.work_used = 0
        self.finalization = None
        self.counters = dict(menu_builds=0, certification_calls=0)
        self._pending = None

    def observe(self):
        if self._pending is None and not self.done:
            self.counters['menu_builds'] += 1
            self._pending = 2
        actions = [dict(kind='improve', start=i+1, end=i+2, forbidden_node=None)
                   for i in range(3)] + [dict(kind='stop', start=0, end=0, forbidden_node=None)]
        return dict(schema='synthetic', actions=actions, mask=[True]*4,
                    current=self.current, path=self.path,
                    pending_menu_work=0 if self.done else self._pending)

    def fork(self):
        return copy.deepcopy(self)

    def step(self, action_id):
        self.observe()
        if action_id == 3:
            reward = -.2
            self.done = True
            self.finalization = {'returned_cost': self.best_cost}
            proposal = None
        else:
            reward = ([.3, -.1, 3.][action_id] if not self.path else
                      ([.6, .1, 2.][action_id] if self.path[0] == 0 else
                       [1., .1, 2.][action_id]))
            proposal = {'valid': True}
        self.path = self.path+[action_id]
        self.current = [0, 1+len(self.path), 0]
        self.work_used += 3
        self.total_reward += reward
        self._pending = None
        self.counters['certification_calls'] += 1
        obs = self.observe()
        info = dict(proposal=proposal, proposal_work_units=1 if proposal else 0,
                    failure_reason=None, sampled_observation={})
        return obs, reward, self.done, False, info

    def terminate(self, reason):
        if self.done:
            return 0., {}
        self.total_reward -= .5
        self.work_used += self._pending or 0
        self.done = True
        self.finalization = dict(reason=reason, returned_cost=self.best_cost)
        self.counters['certification_calls'] += 1
        return -.5, dict(reward=-.5)


class TeacherLabelsTests(unittest.TestCase):
    def test_unknown_actions_are_not_negative_labels(self):
        result = make_labels([True, True, True, True], {0: .3, 3: -.2}, 3)
        self.assertEqual(result['evidence_mask'].tolist(), [True, False, False, True])
        self.assertEqual(result['ties'].tolist(), [True, False, False, False])
        # Unknown numerical zeros cannot win over evidence even when all Q < 0.
        negative = make_labels([True]*3, {0: -.7, 2: -.8}, 2)
        self.assertEqual(negative['ties'].tolist(), [True, False, False])

    def test_near_ties_and_stop_are_preserved(self):
        result = make_labels([True]*4, {0: .1, 1: .10005, 3: -.2}, 3, 1e-4)
        self.assertEqual(result['ties'].tolist(), [True, True, False, False])
        np.testing.assert_allclose(result['advantages_vs_stop'], [.3, .30005, 0., 0.])

    def test_illegal_nonfinite_and_missing_stop_evidence_rejected(self):
        for values, stop in [({1: 1., 2: 0.}, 2), ({0: float('nan'), 2: 0.}, 2), ({0: 1.}, 2)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                make_labels([True, False, True], values, stop)

    def test_branch_selection_keeps_stop_and_unknown_entries(self):
        env = TinyTree()
        self.assertEqual(queried_actions(env.observe(), 3), [0, 1, 3])
        self.assertEqual(queried_actions(env.observe(), 1), [3])
        obs = env.observe()
        obs['mask'][0] = False
        self.assertEqual(queried_actions(obs, 3), [1, 2, 3])

    def test_remaining_return_includes_depth_termination_charge(self):
        rows = []
        samples, summary = finite_tree(TinyTree(), lambda e: e.observe(),
                                      depth=2, branch_limit=3, log_query=rows.append)
        root = next(s for s in samples if not s['path'])
        np.testing.assert_allclose(root['q_values'], [.4, .4, 0., -.2], atol=1e-12)
        self.assertEqual(root['ties'].tolist(), [True, True, False, False])
        first = next(s for s in samples if s['path'] == [0])
        # Parent prefix .3 is removed: .3 + .6 - .5 -> remaining .1.
        self.assertAlmostEqual(first['q_values'][0], .1)
        self.assertEqual(summary['queries'], 9)
        self.assertEqual(summary['expanded_states'], 3)
        self.assertAlmostEqual(summary['root_best_total_reward'], .4)
        self.assertTrue(any(r['terminal_event_reward'] == -.5 for r in rows))
        self.assertEqual(summary['physical_oracle_work']['proposal_calls'], 6)

    def test_same_route_distinct_histories_keep_labels_but_share_weight(self):
        samples = [dict(scene_id='a', current_route=[0, 1, 0], state_signature='x'),
                   dict(scene_id='a', current_route=[0, 1, 0], state_signature='y'),
                   dict(scene_id='a', current_route=[0, 2, 0], state_signature='z'),
                   dict(scene_id='b', current_route=[0, 3, 0], state_signature='w')]
        assign_scene_route_weights(samples)
        np.testing.assert_allclose([s['weight'] for s in samples], [.125, .125, .25, .5])
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0]['state_signature'], 'x')
        self.assertEqual(samples[1]['state_signature'], 'y')

    def test_semantic_action_does_not_use_local_action_index(self):
        action = dict(kind='replace', start=2, end=3, forbidden_node=12, key='old-id-5')
        self.assertNotIn('key', semantic_action(action))
        self.assertEqual(semantic_action(action)['forbidden_node'], 12)


if __name__ == '__main__':
    unittest.main()
