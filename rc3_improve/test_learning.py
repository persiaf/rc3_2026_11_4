import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rc3_baseline.baseline import make_scene
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare, route_objective, export_plan
from rc3_improve.environment import ImprovementEnv,canonical_route
from rc3_improve.data import teacher_gain
from rc3_improve.learning import Policy, batch_states, save_policy, load_policy
from rc3_improve.repair import FEATURE_NAMES, repair_window, route_is_valid, window_features


class ImprovementLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problem=prepare(make_complex_scene(10,10,50911,'blocked'))

    def model(self):
        torch.manual_seed(123)
        return Policy(np.zeros(len(FEATURE_NAMES)),np.ones(len(FEATURE_NAMES)),dim=32)

    def state(self):
        env=ImprovementEnv(self.problem,horizon=4)
        return env.observe()

    def test_padding_and_other_batch_sizes_do_not_change_valid_scores(self):
        features,mask=self.state()
        small=(features[:3].copy(),np.array([True,False,True]))
        model=self.model().eval()
        x,m=batch_states([small],'cpu')
        with torch.no_grad():
            logits,value=model(x,m)
        larger=(features,mask)
        xx,mm=batch_states([small,larger],'cpu')
        # Masked observations cannot enter attention keys or graph/value pooling.
        xx[0,~mm[0]]=10000.
        with torch.no_grad():
            other_logits,other_value=model(xx,mm)
        torch.testing.assert_close(logits[0,small[1]],other_logits[0,:3][small[1]],atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(value[0],other_value[0],atol=2e-6,rtol=2e-6)
        self.assertTrue(torch.isneginf(other_logits[0,~mm[0]]).all())

    def test_actor_critic_update_has_finite_nonzero_gradients(self):
        state=self.state()
        model=self.model().train()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        x,mask=batch_states([state,(state[0][:5],state[1][:5])],'cpu')
        logits,value=model(x,mask)
        distribution=torch.distributions.Categorical(logits=logits)
        actions=distribution.sample()
        advantage=torch.tensor([1.,-.5])
        loss=-(distribution.log_prob(actions)*advantage).mean()+.5*((value-advantage)**2).mean()-.01*distribution.entropy().mean()
        optimizer.zero_grad();loss.backward()
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)
        before=model.actor[-1].weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before,model.actor[-1].weight))

    def test_checkpoint_roundtrip_preserves_scores_and_action(self):
        model=self.model()
        state=self.state()
        expected=model.scores(*state)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'policy.pt'
            save_policy(path,model,stage='test',epoch=1)
            actual=load_policy(path).scores(*state)
        np.testing.assert_array_equal(expected,actual)
        self.assertEqual(int(expected.argmax()),int(actual.argmax()))

    def test_actual_sequential_transitions_are_monotone_and_refresh_windows(self):
        env=ImprovementEnv(self.problem,horizon=4)
        rewards=[];accepted=0
        while True:
            _,mask=env.observe()
            if env.done:break
            actions=np.flatnonzero(mask)
            candidates=[repair_window(self.problem,env.route,env.windows[int(i)]) for i in actions]
            costs=[route_objective(self.problem,r) for r in candidates]
            action=int(actions[int(np.argmin(costs))])
            old_route=list(env.route)
            reward,done,info=env.step(action)
            rewards.append(reward);accepted+=info['accepted']
            self.assertGreaterEqual(reward,0.)
            self.assertLessEqual(info['after'],info['before']+1e-8)
            self.assertTrue(route_is_valid(self.problem,env.route))
            exported=export_plan(self.problem,env.route,'per_step_cost_check')
            self.assertAlmostEqual(info['after'],exported['objective_length_plus_stops_m'],places=8)
            self.assertAlmostEqual(route_objective(self.problem,env.route),exported['objective_length_plus_stops_m'],places=8)
            if info['accepted']:self.assertNotEqual(env.route,old_route)
            features,new_mask=env.observe()
            np.testing.assert_array_equal(features,window_features(self.problem,env.route,env.windows))
            self.assertEqual(done,env.done)
            self.assertEqual(len(features),len(new_mask))
        self.assertGreater(accepted,0)
        self.assertEqual(env.steps,4)
        expected=100*(env.reference_cost-route_objective(self.problem,env.route))/env.reference_cost
        self.assertAlmostEqual(sum(rewards),expected,places=9)
        self.assertTrue(export_plan(self.problem,env.route,'sequential_test')['validation']['valid_model_plan'])

    def test_redundant_stop_canonicalization_matches_export_and_reward(self):
        env=ImprovementEnv(self.problem,horizon=2)
        _,mask=env.observe()
        candidates=[]
        for action in np.flatnonzero(mask):
            repaired=canonical_route(self.problem,repair_window(self.problem,env.route,env.windows[int(action)]))
            if route_objective(self.problem,repaired)<env.reference_cost-1e-7:
                candidates.append((int(action),repaired))
        self.assertTrue(candidates)
        action,improved=candidates[0]
        unused=[node for node in range(1,self.problem.F.shape[1]+1)
                if node not in improved and np.isfinite(self.problem.nav['costs'][0,node])]
        redundant=improved[:-1]+[unused[0]]+[0]
        self.assertTrue(route_is_valid(self.problem,redundant))
        exported=export_plan(self.problem,redundant,'redundant_fixture')
        actual=exported['objective_length_plus_stops_m']
        self.assertGreater(route_objective(self.problem,redundant),actual)
        self.assertEqual(canonical_route(self.problem,redundant),improved)
        self.assertEqual(canonical_route(self.problem,improved),improved)
        # A valid proposal is priced by the actual auto-service export semantics.
        with patch('rc3_improve.environment.repair_window',return_value=redundant):
            reward,done,info=env.step(action)
        self.assertTrue(info['accepted'])
        self.assertAlmostEqual(info['after'],actual,places=8)
        self.assertAlmostEqual(reward,100*(env.reference_cost-actual)/env.reference_cost,places=8)
        # The BC teacher labels must use the same route and stop-count semantics.
        reference_env=ImprovementEnv(self.problem)
        reference_env.observe()
        with patch('rc3_improve.data.repair_window',return_value=redundant):
            label=teacher_gain(self.problem,reference_env.route,reference_env.windows[action])
        self.assertAlmostEqual(label,reference_env.reference_cost-actual,places=8)
        with self.assertRaises(ValueError):canonical_route(self.problem,[0,0])

    def test_attempted_signature_is_masked_after_zero_gain(self):
        env=ImprovementEnv(self.problem,horizon=3)
        _,mask=env.observe();action=int(np.flatnonzero(mask)[0])
        signature=env._signatures[action]
        with patch('rc3_improve.environment.repair_window',side_effect=lambda p,r,w:list(r)):
            reward,done,info=env.step(action)
        self.assertEqual(reward,0.);self.assertFalse(info['accepted'])
        self.assertIn(signature,env.attempted)
        _,mask=env.observe()
        self.assertFalse(mask[action])
        with self.assertRaises(ValueError):env.step(action)
        with self.assertRaises(ValueError):env.step(-1)
        with self.assertRaises(ValueError):env.step(len(mask))

    def test_empty_and_horizon_terminal_never_query_policy(self):
        empty=prepare(make_scene(1,1,10,counts=[0]))
        env=ImprovementEnv(empty,horizon=3)
        features,mask=env.observe()
        self.assertTrue(env.done);self.assertEqual(len(features),0);self.assertFalse(mask.any())
        self.assertEqual(env.route,[0,0]);self.assertEqual(env.reference_cost,0.)
        with self.assertRaises(ValueError):env.step(0)
        x,m=batch_states([(features,mask)],'cpu')
        self.assertEqual(tuple(x.shape),(1,1,len(FEATURE_NAMES)))
        with self.assertRaises(ValueError):self.model()(x,m)
        bounded=ImprovementEnv(self.problem,horizon=1)
        _,valid=bounded.observe()
        bounded.step(int(np.flatnonzero(valid)[0]))
        self.assertTrue(bounded.done)
        with self.assertRaises(ValueError):bounded.step(0)


if __name__=='__main__':
    unittest.main()
