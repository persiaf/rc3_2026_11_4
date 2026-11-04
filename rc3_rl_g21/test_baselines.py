"""Pure dictionary controls regression: no scene, environment, prepare or step."""
import copy
import math
import unittest

from .baselines import DETOUR_EARLY_GRID, DetourEarlyPolicy, OneStepPolicy, make_policy


def menu():
    return dict(actions=[{"kind": "improve"}, {"kind": "replace"}, {"kind": "stop"}],
                mask=[True, True, True], feature_names=["detour_excess"],
                features=[[2.0], [1.0], [0.0]])


def feedback(kind="improve", quality=0.0, penalty=0.01, changed=False, improved=False):
    return dict(action={"kind": kind}, quality_reward=quality, work_penalty=penalty,
                work_units=100, route_changed=changed, best_improved=improved,
                accepted=changed)


class FakeSelector:
    """Synthetic logits/argmax stand-in, with STOP retained as a valid result."""
    name = "synthetic_selector"

    def __init__(self, action=2):
        self.action, self.calls, self.updates = action, 0, 0

    def select(self, observation, rng):
        self.calls += 1
        return self.action

    def update(self, info):
        self.updates += 1

    def state_dict(self):
        return dict(name=self.name, action=self.action, calls=self.calls, updates=self.updates)

    def load_state_dict(self, state):
        if state["name"] != self.name:
            raise ValueError("incompatible synthetic selector")
        self.action, self.calls, self.updates = state["action"], state["calls"], state["updates"]


class BaselineTests(unittest.TestCase):
    def test_detour_mask_and_tie(self):
        obs = menu()
        obs["features"][1][0] = 2.0
        self.assertEqual(make_policy("detour_early").select(obs, None), 0)
        obs["mask"][0] = False
        self.assertEqual(make_policy("detour_early").select(obs, None), 1)

    def test_empty_candidate_menu_stops(self):
        obs = menu()
        obs["mask"] = [False, False, True]
        self.assertEqual(DetourEarlyPolicy().select(obs, None), 2)

    def test_repeated_observation_does_not_consume_attempt(self):
        policy = DetourEarlyPolicy(max_proposals=1)
        before = policy.state_dict()
        for _ in range(4):
            self.assertEqual(policy.select(menu(), None), 0)
        self.assertEqual(before, policy.state_dict())

    def test_paid_zero_gain_triggers_patience(self):
        policy = DetourEarlyPolicy(patience=2)
        policy.update(feedback())
        self.assertEqual(policy.select(menu(), None), 0)
        policy.update(feedback(kind="replace"))
        self.assertEqual(policy.select(menu(), None), 2)
        self.assertEqual(policy.paid_work, 200)
        self.assertEqual(policy.total_work_penalty, 0.02)

    def test_gain_must_pay_for_already_observed_work(self):
        policy = DetourEarlyPolicy()
        policy.update(feedback(quality=0.005, changed=True, improved=True))
        self.assertEqual(policy.select(menu(), None), 2)

    def test_real_net_progress_resets_stagnation(self):
        policy = DetourEarlyPolicy(patience=2)
        policy.update(feedback())
        policy.update(feedback(quality=0.02, changed=True, improved=True))
        self.assertEqual(policy.stagnation, 0)
        self.assertEqual(policy.select(menu(), None), 0)

    def test_no_op_acceptance_is_not_progress(self):
        info = feedback(quality=0.5, improved=True)
        info["accepted"] = True
        policy = DetourEarlyPolicy()
        policy.update(info)
        self.assertEqual(policy.select(menu(), None), 2)

    def test_cap_applies_even_when_every_proposal_improves(self):
        policy = DetourEarlyPolicy(max_proposals=2)
        for _ in range(2):
            policy.update(feedback(quality=0.5, changed=True, improved=True))
        self.assertEqual(policy.select(menu(), None), 2)
        before = policy.state_dict()
        with self.assertRaises(ValueError):
            policy.update(feedback())
        self.assertEqual(policy.state_dict(), before)

    def test_future_labels_and_oracle_do_not_change_action(self):
        obs = menu()
        # These fields are intentionally unrelated to the accepted observation schema.
        obs.update(oracle_best_prefix=0, teacher_q=[-1000, 1000, 0], future_gain=9999)
        self.assertEqual(DetourEarlyPolicy().select(obs, None), 0)
        obs.update(oracle_best_prefix=16, teacher_q=[1000, -1000, 0], future_gain=-9999)
        self.assertEqual(DetourEarlyPolicy().select(obs, None), 0)

    def test_stop_cost_is_recorded_without_a_proposal(self):
        policy = DetourEarlyPolicy()
        policy.update(feedback(kind="stop"))
        self.assertEqual(policy.proposals, 0)
        self.assertEqual(policy.paid_work, 100)
        self.assertEqual(policy.select(menu(), None), 2)

    def test_detour_state_round_trip(self):
        original = DetourEarlyPolicy(max_proposals=4, patience=2)
        original.update(feedback())
        loaded = DetourEarlyPolicy()
        loaded.load_state_dict(original.state_dict())
        for item in (feedback(quality=1, changed=True, improved=True), feedback(), feedback()):
            original.update(item)
            loaded.update(item)
            self.assertEqual(original.state_dict(), loaded.state_dict())
            self.assertEqual(original.select(menu(), None), loaded.select(menu(), None))

    def test_bad_feedback_does_not_mutate_state(self):
        policy = DetourEarlyPolicy()
        for key, bad in (("work_units", -1), ("quality_reward", math.nan),
                         ("work_penalty", -1), ("route_changed", 1)):
            info = feedback()
            info[key] = bad
            before = policy.state_dict()
            with self.assertRaises(ValueError):
                policy.update(info)
            self.assertEqual(before, policy.state_dict())

    def test_bad_state_does_not_mutate_state(self):
        policy = DetourEarlyPolicy()
        for key, bad in (("proposals", 5), ("stagnation", 1), ("total_quality", math.inf),
                         ("stopped", 1), ("name", "oracle")):
            state = policy.state_dict()
            state[key] = bad
            before = policy.state_dict()
            with self.assertRaises(ValueError):
                policy.load_state_dict(state)
            self.assertEqual(before, policy.state_dict())

    def test_invalid_configs_rejected(self):
        for kwargs in ({"patience": 0}, {"max_proposals": 0}, {"max_proposals": True},
                       {"progress_tolerance": math.nan}, {"progress_tolerance": -1}):
            with self.assertRaises(ValueError):
                DetourEarlyPolicy(**kwargs)

    def test_bad_menu_and_nonfinite_feature_rejected(self):
        for mutation in (lambda obs: obs.update(mask=[True]),
                         lambda obs: obs.update(mask=[True, True, False]),
                         lambda obs: obs.update(features=[[math.nan], [1], [0]])):
            obs = menu()
            mutation(obs)
            with self.assertRaises(ValueError):
                DetourEarlyPolicy().select(obs, None)

    def test_one_step_detour_and_stop_root_controls(self):
        for base_name, expected in (("detour_memory", 0), ("stop", 2)):
            policy = make_policy("one_step", base_name=base_name)
            self.assertEqual(policy.select(menu(), None), expected)

    def test_one_step_learned_wrapper_retains_root_stop(self):
        selector = FakeSelector(2)
        policy = OneStepPolicy(selector=selector)
        self.assertEqual(policy.select(menu(), None), 2)
        policy.update(feedback(kind="stop"))
        self.assertEqual(policy.proposals, 0)
        self.assertEqual(selector.updates, 1)

    def test_one_step_does_not_infer_again_after_proposal(self):
        selector = FakeSelector(1)
        policy = OneStepPolicy(selector=selector)
        self.assertEqual(policy.select(menu(), None), 1)
        policy.update(feedback(kind="replace"))
        self.assertEqual(policy.select(menu(), None), 2)
        self.assertEqual(selector.calls, 1)
        with self.assertRaises(ValueError):
            policy.update(feedback())

    def test_one_step_mask_and_restore(self):
        policy = OneStepPolicy(selector=FakeSelector(1))
        obs = menu()
        obs["mask"][1] = False
        with self.assertRaises(ValueError):
            policy.select(obs, None)
        policy.update(feedback(kind="replace"))
        restored = OneStepPolicy(selector=FakeSelector())
        restored.load_state_dict(policy.state_dict())
        self.assertEqual(policy.state_dict(), restored.state_dict())
        self.assertEqual(restored.select(menu(), None), 2)

    def test_one_step_non_integer_action_rejected(self):
        for action in (True, 0.0, "0", 99):
            with self.assertRaises(ValueError):
                OneStepPolicy(selector=FakeSelector(action)).select(menu(), None)

    def test_one_step_invalid_load_is_atomic(self):
        policy = OneStepPolicy(selector=FakeSelector())
        before = policy.state_dict()
        bad = copy.deepcopy(before)
        bad["selector"]["name"] = "different"
        with self.assertRaises(ValueError):
            policy.load_state_dict(bad)
        self.assertEqual(policy.state_dict(), before)

    def test_grid_unique_bounded_and_predeclared(self):
        self.assertEqual(len(DETOUR_EARLY_GRID), 4)
        self.assertEqual(len({tuple(sorted(x.items())) for x in DETOUR_EARLY_GRID}), 4)
        for kwargs in DETOUR_EARLY_GRID:
            policy = DetourEarlyPolicy(**kwargs)
            self.assertLessEqual(policy.max_proposals, 4)

    def test_frozen_factory_passthrough(self):
        self.assertEqual(make_policy("stop").select(menu(), None), 2)
        with self.assertRaises(ValueError):
            make_policy("future_prefix_oracle")


if __name__ == "__main__":
    unittest.main()
