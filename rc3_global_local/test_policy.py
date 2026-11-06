"""Independent synthetic policy contracts; no real scenes, navigation, H or fitting.

The F/A relation and Euclidean cost array below are hand-authored mathematical
fixtures. A single backward probe tests graph connectivity, not RL performance.
"""
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from rc3_global_local.problem import Parent, LocalProblem
from rc3_global_local.environment import Environment, rule_action
from rc3_global_local.policy import GlobalLocalPolicy, global_tokens, static_data, tensor_hash


COUNTS = dict(encode_attempts=0, encode_completed=0, score_attempts=0,
              score_completed=0, prefix_GRU_forward_calls=0, backward_calls=0,
              synthetic_weight_version_mutations=0)
GRADIENT_NORMS = {}


class CountedPolicy(GlobalLocalPolicy):
    def __init__(self):
        super().__init__()
        self.prefix.register_forward_hook(self._prefix_counter)

    @staticmethod
    def _prefix_counter(module, args, output):
        COUNTS['prefix_GRU_forward_calls'] += 1

    def encode(self, problem):
        COUNTS['encode_attempts'] += 1
        value = super().encode(problem)
        COUNTS['encode_completed'] += 1
        return value

    def score(self, env, encoded):
        COUNTS['score_attempts'] += 1
        value = super().score(env, encoded)
        COUNTS['score_completed'] += 1
        return value


def synthetic_parent():
    # Two candidates per logical table; six tasks have different overlaps.
    f = np.zeros((6, 8, 2), dtype=bool)
    for task, choices in enumerate(((0, 2, 6), (1, 3, 6), (2, 4, 7),
                                     (3, 5, 7), (4, 6), (5, 7))):
        for j in choices:
            f[task, j, (task+j) % 2] = True
    xy = np.asarray([[0., 0.], [1., .2], [1.3, .8], [3., .2], [3.3, .9],
                     [1., 3.], [1.4, 3.7], [3., 3.], [3.5, 3.8]])
    delta = xy[:, None, :] - xy[None, :, :]
    d = np.sqrt((delta*delta).sum(2))
    tables = [dict(id=y*2+x, ix=x, iy=y, cx=1.+2*x, cy=1.+2*y,
                   xmin=.5+2*x, xmax=1.5+2*x, ymin=.5+2*y, ymax=1.5+2*y)
              for y in range(2) for x in range(2)]
    scene = dict(nx=2, ny=2, start=xy[0].tolist(), tables=tables,
                 tasks=[dict(id=i, table_id=i % 4, x=float(.8+i*.37),
                             y=float(.6+(i % 3)*.61), z=.75) for i in range(6)],
                 candidates=[dict(id=j, table_id=j//2, x=float(xy[j+1, 0]),
                                  y=float(xy[j+1, 1]), theta=float(j*.27), h=.4)
                             for j in range(8)],
                 obstacles=[dict(xmin=6., xmax=6.1, ymin=6., ymax=6.2)])
    return Parent(SimpleNamespace(scene=scene, nav=dict(costs=d)), f, np.ones_like(f))


def identity_terminal(problem, route):
    # Independent scalar terminal; production metrics/H/geometry are not called.
    length = sum(float(problem.D[a, b]) for a, b in zip(route, route[1:]))
    k = len(route)-2
    return dict(J=length+.75*k, L=length, K=k, route=list(route))


class PolicyContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(638721)
        self.parent = synthetic_parent()
        self.problem = LocalProblem(self.parent, list(range(6)), list(range(4)), 0, 8)
        self.model = CountedPolicy()

    def env(self, prefix=(), problem=None):
        env = Environment(self.problem if problem is None else problem, identity_terminal)
        for node in prefix:
            env.step(node)
        return env

    def assert_same_scores(self, left, right, env):
        legal = torch.tensor(env.mask())
        self.assertTrue(torch.allclose(left[legal], right[legal], atol=1e-7, rtol=0))
        self.assertTrue(torch.equal(torch.isneginf(left), torch.isneginf(right)))

    def test_feature_dimensions_and_manual_values(self):
        g = global_tokens(self.parent)
        data = static_data(self.problem)
        self.assertEqual(g.shape, (4, 14))
        self.assertEqual(tuple(data['task'].shape), (6, 5))
        self.assertEqual(tuple(data['candidate'].shape), (8, 7))
        self.assertEqual(tuple(data['C'].shape), (6, 8))
        self.assertEqual(tuple(data['D'].shape), (9, 9))
        self.assertAlmostEqual(float(g[0, 0]), .1, places=7)
        self.assertAlmostEqual(float(g[0, 2]), .1, places=7)
        self.assertAlmostEqual(float(g[0, 4]), 2/3, places=7)
        self.assertAlmostEqual(float(g[0, 5]), .01, places=7)
        for value in data.values():
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_illegal_actions_have_exact_zero_probability(self):
        env = self.env([1])  # Candidate1 no longer covers any remaining task.
        with torch.no_grad():
            encoded = self.model.encode(self.problem)
            scores = self.model.score(env, encoded)
        legal = torch.tensor(env.mask())
        self.assertFalse(bool(legal[0]))
        self.assertTrue(bool(torch.isfinite(scores[legal]).all()))
        self.assertTrue(bool(torch.isneginf(scores[~legal]).all()))
        probabilities = scores.softmax(0)
        self.assertTrue(bool((probabilities[~legal] == 0).all()))
        self.assertAlmostEqual(float(probabilities.sum()), 1., places=6)

    def test_one_parameter_set_supports_different_local_sizes(self):
        small = LocalProblem(self.parent, [0, 4], [0])
        before = tensor_hash(self.model)
        with torch.no_grad():
            large_score = self.model.score(self.env(), self.model.encode(self.problem))
            small_score = self.model.score(self.env(problem=small), self.model.encode(small))
        self.assertEqual(tuple(large_score.shape), (8,))
        self.assertEqual(tuple(small_score.shape), (4,))
        self.assertEqual(before, tensor_hash(self.model))

    def test_incremental_prefix_cache_equals_full_recompute(self):
        env = self.env()
        with torch.no_grad():
            cached = self.model.encode(self.problem)
            self.model.score(env, cached)
            for action in (1, 2, 3):
                env.step(action)
                incremental = self.model.score(env, cached)
                fresh = self.model.encode(self.problem)
                complete = self.model.score(env, fresh)
                self.assert_same_scores(incremental, complete, env)
                self.assertTrue(torch.equal(cached['history_hidden'], fresh['history_hidden']))
                self.assertEqual(cached['history_prefix'], tuple(env.prefix))

    def test_same_prefix_score_does_not_repeat_GRU_updates(self):
        env = self.env([1, 2])
        with torch.no_grad():
            encoded = self.model.encode(self.problem)
            first = self.model.score(env, encoded)
            count = COUNTS['prefix_GRU_forward_calls']
            second = self.model.score(env, encoded)
        self.assertEqual(count, COUNTS['prefix_GRU_forward_calls'])
        self.assert_same_scores(first, second, env)

    def test_cache_branch_recomputes_ordered_prefix(self):
        first = self.env([1, 2])
        branched = self.env([2, 1, 3])
        with torch.no_grad():
            encoded = self.model.encode(self.problem)
            self.model.score(first, encoded)
            reused = self.model.score(branched, encoded)
            fresh = self.model.encode(self.problem)
            recomputed = self.model.score(branched, fresh)
        self.assert_same_scores(reused, recomputed, branched)
        self.assertTrue(torch.equal(encoded['history_hidden'], fresh['history_hidden']))

    def test_history_preserves_order_for_same_current_and_coverage(self):
        left = self.env([1, 2, 3])
        right = self.env([2, 1, 3])
        np.testing.assert_array_equal(left.remaining, right.remaining)
        self.assertEqual(left.current, right.current)
        with torch.no_grad():
            encoded_left = self.model.encode(self.problem)
            encoded_right = self.model.encode(self.problem)
            self.model.score(left, encoded_left)
            self.model.score(right, encoded_right)
        self.assertFalse(torch.allclose(encoded_left['history_hidden'], encoded_right['history_hidden'], atol=1e-7, rtol=0))

    def test_cross_problem_cache_rejected(self):
        small = LocalProblem(self.parent, [0, 4], [0])
        encoded = self.model.encode(self.problem)
        with self.assertRaises(ValueError):
            self.model.score(self.env(problem=small), encoded)

    def test_cross_model_cache_rejected_even_when_versions_equal(self):
        encoded = self.model.encode(self.problem)
        other = CountedPolicy()
        self.assertNotEqual(tensor_hash(self.model), tensor_hash(other))
        # PyTorch version counters describe mutations within an instance;
        # independently initialized models can have the same counters.
        self.assertEqual(encoded['parameter_versions'], tuple(p._version for p in other.parameters()))
        with self.assertRaises(ValueError):
            other.score(self.env(), encoded)

    def test_cross_weight_version_cache_rejected(self):
        encoded = self.model.encode(self.problem)
        with torch.no_grad():
            next(self.model.parameters()).add_(.001)
        COUNTS['synthetic_weight_version_mutations'] += 1
        with self.assertRaises(ValueError):
            self.model.score(self.env(), encoded)

    def test_terminal_has_no_action_distribution(self):
        env = self.env()
        encoded = self.model.encode(self.problem)
        while not env.done:
            env.step(rule_action(env))
        with self.assertRaises(ValueError):
            self.model.score(env, encoded)

    def test_empty_problem_has_no_action_distribution(self):
        empty = LocalProblem(self.parent, [], [], 1, 8)
        env = self.env(problem=empty)
        self.assertTrue(env.done)
        self.assertEqual(len(env.prefix), 0)
        with torch.no_grad():
            encoded = self.model.encode(empty)
        with self.assertRaises(ValueError):
            self.model.score(env, encoded)

    def test_structural_backward_is_finite_and_preserves_weights(self):
        env = self.env([1, 2])
        before = tensor_hash(self.model)
        encoded = self.model.encode(self.problem)
        scores = self.model.score(env, encoded)
        legal = np.flatnonzero(env.mask())
        selected = int(legal[len(legal)//2])
        loss = -torch.log_softmax(scores, 0)[selected]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        COUNTS['backward_calls'] += 1
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in self.model.parameters()))
        groups = dict(global_=('global_net.',), local=('task_in.', 'candidate_in.', 'to_task.', 'to_candidate.'),
                      prefix=('prefix.',), query=('query.',), key_navigation=('key.', 'dynamic.', 'nav.'))
        for name, prefixes in groups.items():
            norm = sum(float((p.grad.detach().double()**2).sum()) for n,p in self.model.named_parameters() if n.startswith(prefixes))**.5
            self.assertGreater(norm, 1e-10, name)
            GRADIENT_NORMS[name] = norm
        self.assertEqual(before, tensor_hash(self.model))


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PolicyContracts)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    root = Path(__file__).resolve().parent
    files = ('test_policy.py', 'policy.py', 'problem.py', 'environment.py')
    receipt = dict(schema='S0-independent-synthetic-policy-contract-v1',client_date='2026-11-06',
        author='/root/s0_policy_contract',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        passed=result.wasSuccessful(),elapsed_s=time.perf_counter()-started,
        command='python -B -m rc3_global_local.test_policy',
        test_names=unittest.defaultTestLoader.getTestCaseNames(PolicyContracts),
        source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
        model_operations=dict(COUNTS),structural_gradient_norms=GRADIENT_NORMS,
        optimizer_updates=0,real_scene_generation=0,navigation_preparations=0,planner_calls=0,H_calls=0,
        teacher_queries=0,retained60_reads=0,
        scope='Hand-authored F/A/D fixtures and identity terminal only; finite graph connectivity is not RL training or evidence of learning increment. One isolated in-memory parameter mutation tests version rejection; no checkpoint is written or altered.')
    (root/'policy_unit_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
