"""Independent synthetic contract tests; no scene generation, solver, H or model.

Expected menus below are hand-constructed from coverage/finite-edge inputs.
The tests do not call the production state kernel to decide legality.
"""
from types import SimpleNamespace
import unittest

import numpy as np

from rc3_teacher_distill.candidate_menu import generate, rule_action


def item_from(coverage, xy=None, distances=None):
    c = np.asarray(coverage, dtype=bool)
    m = c.shape[1]
    if xy is None:
        xy = [(float(i), 0.0) for i in range(m)]
    d = np.ones((m + 1, m + 1), dtype=float) if distances is None else np.array(distances, dtype=float)
    np.fill_diagonal(d, 0.0)
    candidates = [dict(id=i, x=float(x), y=float(y)) for i, (x, y) in enumerate(xy)]
    scene = dict(candidates=candidates, tasks=[dict(id=i) for i in range(len(c))])
    return SimpleNamespace(C=c, D=d, base=SimpleNamespace(scene=scene, penalty=0.75))


def remaining_and_legal(item, prefix):
    """Small independent scalar legality oracle, deliberately no State import."""
    remaining = set(range(item.C.shape[0]))
    current = 0
    for node in prefix:
        gain = {t for t in remaining if item.C[t, node - 1]}
        if not gain or not np.isfinite(item.D[current, node]) or not np.isfinite(item.D[node, 0]):
            raise ValueError("invalid synthetic prefix")
        remaining -= gain
        current = node
    legal = [node for node in range(1, item.C.shape[1] + 1)
             if any(item.C[t, node - 1] for t in remaining)
             and np.isfinite(item.D[current, node]) and np.isfinite(item.D[node, 0])]
    return remaining, legal


def quota_item():
    # 1=reference; 2=largest gain; 3/4=cheap; 5/6=rare; 7/8=dispersed.
    c = np.zeros((6, 10), dtype=bool)
    c[0, 0] = True
    c[:4, 1] = True
    c[0, 2] = True
    c[1, 3] = True
    c[4, 4] = True
    c[5, 5] = True
    c[:2, 6:] = True
    d = np.full((11, 11), 10.0)
    d[0, 3:5] = 0.25
    xy = [(0, 0)] * 6 + [(10, 0), (-10, 0), (0, 5), (1, 1)]
    return item_from(c, xy, d), [1, 2, 5, 6]


class CandidateMenuTests(unittest.TestCase):
    def test_reference_is_always_first_even_if_bad_immediate_ratio(self):
        item, b = quota_item()
        self.assertEqual(generate(item, b, k=1), [1])

    def test_gain_quota(self):
        item, b = quota_item()
        self.assertEqual(generate(item, b, k=2), [1, 2])

    def test_efficiency_quota(self):
        item, b = quota_item()
        self.assertEqual(generate(item, b, k=4), [1, 2, 3, 4])

    def test_scarce_coverage_quota(self):
        item, b = quota_item()
        self.assertEqual(generate(item, b, k=6), [1, 2, 3, 4, 5, 6])

    def test_spatial_diversity_and_id_tie(self):
        item, b = quota_item()
        self.assertEqual(generate(item, b, k=8), [1, 2, 3, 4, 5, 6, 7, 8])

    def test_every_bound_including_more_than_legal_count(self):
        item, b = quota_item()
        for k in range(1, 13):
            with self.subTest(k=k):
                menu = generate(item, b, k=k)
                self.assertEqual(len(menu), min(k, 10))
                self.assertEqual(len(menu), len(set(menu)))
                self.assertEqual(menu[0], 1)
                self.assertTrue(set(menu) <= set(range(1, 11)))

    def test_small_legal_set_is_not_padded(self):
        item = item_from([[True, False, True]])
        self.assertEqual(generate(item, [3]), [3, 1])

    def test_navigation_and_positive_gain_masks(self):
        d = np.ones((6, 6))
        d[0, 2] = np.inf
        d[4, 0] = np.inf
        item = item_from([[1, 1, 0, 1, 0], [0, 0, 1, 0, 0]], distances=d)
        self.assertEqual(generate(item, [1, 3]), [1, 3])

    def test_current_edge_is_checked_after_prefix(self):
        d = np.ones((5, 5))
        d[1, 3] = np.inf
        item = item_from([[1, 0, 0, 0], [0, 1, 1, 1]], distances=d)
        menu = generate(item, [1, 2], prefix=[1])
        self.assertEqual(set(menu), {2, 4})

    def test_prefix_updates_reference_and_coverage(self):
        item, b = quota_item()
        self.assertEqual(generate(item, b, prefix=[2]), [5, 6])

    def test_complete_state_returns_empty(self):
        item, b = quota_item()
        self.assertEqual(generate(item, b, prefix=[2, 5, 6]), [])

    def test_zero_task_instance_returns_empty(self):
        item = item_from(np.zeros((0, 2), dtype=bool))
        self.assertEqual(generate(item, []), [])

    def test_reference_failure_is_not_replaced_with_rule_fallback(self):
        item = item_from([[1, 0], [0, 1]])
        with self.assertRaises((ValueError, RuntimeError)):
            generate(item, [1], prefix=[1])

    def test_reject_bad_k(self):
        item, b = quota_item()
        for k in (0, -1, True, False, 1.5, "8", None):
            with self.subTest(k=k), self.assertRaises((ValueError, TypeError)):
                generate(item, b, k=k)

    def test_reject_invalid_full_prefix(self):
        item, b = quota_item()
        for prefix in ([0], [11], [1, 1], [True], [1.5]):
            with self.subTest(prefix=prefix), self.assertRaises((ValueError, TypeError)):
                generate(item, b, prefix=prefix)

    def test_rule_uses_same_menu_and_original_id_ties(self):
        item = item_from([[1, 1, 1]])
        self.assertEqual(rule_action(item, [3, 2, 1]), 1)
        self.assertEqual(rule_action(item, [3, 2]), 2)

    def test_rule_cannot_select_best_action_outside_given_menu(self):
        item, b = quota_item()
        self.assertEqual(rule_action(item, [1, 2]), 2)
        self.assertEqual(rule_action(item, generate(item, b, k=8)), 3)

    def test_input_data_are_not_mutated(self):
        item, b = quota_item()
        c, d, ref = item.C.copy(), item.D.copy(), b.copy()
        prefix = [1]
        original_prefix = prefix.copy()
        first = generate(item, b, prefix=prefix)
        for _ in range(8):
            self.assertEqual(generate(item, b, prefix=prefix), first)
        np.testing.assert_array_equal(item.C, c)
        np.testing.assert_array_equal(item.D, d)
        self.assertEqual(b, ref)
        self.assertEqual(prefix, original_prefix)

    def test_every_synthetic_menu_choice_preserves_reference_completion(self):
        # All connected 3-task, 5-candidate examples below have a covering B.
        # Exhaust legal prefixes. The guarantee tested is existence of legal
        # gB completion, not presence of every future service point at once.
        cases = [
            ([[1, 0, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1]], [1, 2, 3]),
            ([[1, 1, 0, 1, 0], [1, 0, 1, 0, 1], [0, 1, 1, 0, 0]], [1, 2]),
        ]
        tested = 0
        for coverage, b in cases:
            item = item_from(coverage)
            prefixes = [()]
            seen = set()
            while prefixes:
                prefix = prefixes.pop()
                if prefix in seen:
                    continue
                seen.add(prefix)
                remaining, legal = remaining_and_legal(item, prefix)
                menu = generate(item, b, prefix=prefix, k=3)
                self.assertTrue(set(menu) <= set(legal))
                self.assertEqual(len(menu), min(3, len(legal)))
                if not remaining:
                    self.assertEqual(menu, [])
                    continue
                expected_ref = next(node for node in b if node in legal)
                self.assertEqual(menu[0], expected_ref)
                for action in menu:
                    continuation = list(prefix) + [action]
                    left, allowed = remaining_and_legal(item, continuation)
                    while left:
                        node = next(node for node in b if node in allowed)
                        continuation.append(node)
                        left, allowed = remaining_and_legal(item, continuation)
                    self.assertFalse(left)
                    prefixes.append(tuple(prefix) + (action,))
                    tested += 1
        self.assertGreater(tested, 20)


if __name__ == "__main__":
    unittest.main(verbosity=2)
