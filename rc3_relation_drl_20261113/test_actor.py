"""Synthetic interface tests; no saved data, planner, H, or optimizer calls.

Execution is intentionally left to the registered M0/M1 runner. These tests
are not evidence of planning performance or part of scientific training.
"""
from types import SimpleNamespace
import copy
import unittest

import numpy as np
import torch

from .actor import RelationActor, _masked_weights


def synthetic_problem(m=4, empty=False, identity="synthetic"):
    """A manually specified interface fixture, not a sampled research task."""
    n = 0 if empty else 3
    if empty:
        m = 0
    tasks = [dict(id=i, x=float(i), y=.2, z=.5, table_id=0) for i in range(n)]
    candidates = [dict(id=i, x=float(i), y=0., theta=0., h=.5, table_id=0)
                  for i in range(m)]
    tables = [dict(id=0, ix=0, iy=0, cx=1., cy=0., xmin=0., xmax=2., ymin=-1., ymax=1.)]
    scene = dict(start=[0., 0.], nx=1, ny=1, tasks=tasks, candidates=candidates,
                 tables=tables, obstacles=[])
    E = np.zeros((n, m, 2), dtype=bool)
    for i in range(n):
        E[i, i % m, i % 2] = True
        E[i, (i + 1) % m, (i + 1) % 2] = True
    D = np.abs(np.arange(m + 1)[:, None] - np.arange(m + 1)[None, :]).astype(float)
    parent = SimpleNamespace(scene=scene, D=D, identity=identity + "-parent")
    return SimpleNamespace(parent=parent, entry=0, exit=0, task_ids=np.arange(n),
                           candidate_ids=np.arange(m), F=E, C=E.any(-1), D=D,
                           identity=identity)


def synthetic_env(problem, remaining=None, prefix=()):
    remaining = np.ones(len(problem.C), dtype=bool) if remaining is None else np.array(remaining, bool)
    current = prefix[-1] if prefix else 0
    raw_L = sum(problem.D[a, b] for a, b in zip((0, *prefix), prefix))
    env = SimpleNamespace(problem=problem, remaining=remaining, current=current,
                          prefix=list(prefix), raw_L=float(raw_L), done=not remaining.any())
    env.mask = lambda: (problem.C[remaining].any(0) & np.isfinite(problem.D[current, 1:])
                        & np.isfinite(problem.D[1:, 0])) if remaining.any() else np.zeros(len(problem.candidate_ids), bool)
    return env


class RelationActorTests(unittest.TestCase):
    def actor(self):
        torch.manual_seed(170113)
        return RelationActor()

    def test_variable_candidate_count_and_exact_mask(self):
        actor = self.actor()
        for m in (4, 7):
            p = synthetic_problem(m=m)
            env = synthetic_env(p)
            logits = actor.score(env, actor.encode(p))
            mask = torch.tensor(env.mask())
            self.assertEqual(tuple(logits.shape), (m,))
            self.assertTrue(torch.isfinite(logits[mask]).all())
            self.assertTrue(torch.isneginf(logits[~mask]).all())

    def test_navigation_relation_changes_encoding(self):
        actor = self.actor()
        p = synthetic_problem()
        base = actor.encode(p)["c"].detach()
        q = copy.deepcopy(p)
        q.D[1, 3] = q.D[3, 1] = 12.5
        changed = actor.encode(q)["c"].detach()
        self.assertGreater(float((base - changed).abs().max()), 1e-7)

    def test_infinite_edges_are_masked(self):
        actor = self.actor()
        p = synthetic_problem()
        p.D[1, 3] = p.D[3, 1] = np.inf
        encoded = actor.encode(p)
        self.assertFalse(bool(encoded["navigation_edge_mask"][0, 2]))
        self.assertTrue(torch.isfinite(encoded["c"]).all())
        logits = actor.score(synthetic_env(p), encoded)
        self.assertTrue(torch.isfinite(logits[torch.isfinite(logits)]).all())

    def test_dynamic_remaining_cover_changes_common_legal_scores(self):
        actor = self.actor()
        p = synthetic_problem()
        a = synthetic_env(p)
        b = synthetic_env(p, [True, False, True])
        encoding = actor.encode(p)
        x = actor.score(a, actor.fork_cache(encoding))
        y = actor.score(b, actor.fork_cache(encoding))
        common = torch.tensor(a.mask() & b.mask())
        self.assertTrue(common.any())
        self.assertGreater(float((x[common] - y[common]).abs().max()), 1e-7)

    def test_prefix_forks_do_not_share_mutable_state(self):
        actor = self.actor()
        p = synthetic_problem()
        encoded = actor.encode(p)
        a, b = actor.fork_cache(encoded), actor.fork_cache(encoded)
        root = actor.score(synthetic_env(p), b).detach()
        actor.score(synthetic_env(p, [False, True, True], (1,)), a)
        self.assertEqual(b["history_prefix"], ())
        fresh = actor.score(synthetic_env(p), actor.fork_cache(encoded)).detach()
        torch.testing.assert_close(root, fresh, rtol=0, atol=0)

    def test_parameter_mutation_invalidates_cache(self):
        actor = self.actor()
        p = synthetic_problem()
        encoded = actor.encode(p)
        with torch.no_grad():
            next(actor.parameters()).add_(1e-5)
        with self.assertRaisesRegex(ValueError, "cache"):
            actor.score(synthetic_env(p), encoded)

    def test_restored_weights_invalidate_cache_and_reproduce_logits(self):
        actor = self.actor()
        p = synthetic_problem()
        state = copy.deepcopy(actor.state_dict())
        encoded = actor.encode(p)
        expected = actor.score(synthetic_env(p), encoded).detach()
        actor.load_state_dict(state, strict=True)
        with self.assertRaisesRegex(ValueError, "cache"):
            actor.score(synthetic_env(p), encoded)
        actual = actor.score(synthetic_env(p), actor.encode(p)).detach()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_problem_has_no_action(self):
        actor = self.actor()
        p = synthetic_problem(empty=True)
        encoded = actor.encode(p)
        self.assertEqual(tuple(encoded["c"].shape), (0, 64))
        with self.assertRaisesRegex(ValueError, "terminal"):
            actor.score(synthetic_env(p), encoded)

    def test_masked_softmax_unsupported_rows(self):
        logits = torch.tensor([[[1., 2.], [3., 4.]]])
        allowed = torch.tensor([[False, False], [True, False]])
        result = _masked_weights(logits, allowed)
        torch.testing.assert_close(result, torch.tensor([[[0., 0.], [1., 0.]]]))

    def test_differentiable_navigation_and_dynamic_relations(self):
        actor = self.actor()
        p = synthetic_problem()
        logits = actor.score(synthetic_env(p), actor.encode(p))
        loss = -torch.log_softmax(logits, 0)[0]
        loss.backward()
        names = ["blocks.0.navigation.edge_bias.weight", "remaining_attention.query.weight"]
        parameters = dict(actor.named_parameters())
        for name in names:
            grad = parameters[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0., name)


if __name__ == "__main__":
    unittest.main()
