"""Independent checks of actual PPO, terminal GAE and replayed rollout states."""
import unittest

import numpy as np
import torch

from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare, route_objective
from rc3_improve.environment import ImprovementEnv
from rc3_improve.learning import Policy
from rc3_improve.repair import FEATURE_NAMES
from rc3_improve.train import advantages, collect, ppo_update


class PPOImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.problems=[prepare(make_complex_scene(4,3,8124,'mixed')),
                      prepare(make_complex_scene(6,6,8125,'blocked'))]

    def policy(self):
        torch.manual_seed(785)
        return Policy(np.zeros(len(FEATURE_NAMES)),np.ones(len(FEATURE_NAMES)),dim=32)

    def test_terminal_gae_matches_hand_computation_and_resets_between_episodes(self):
        episodes=[[dict(reward=1.,value=.5),dict(reward=2.,value=.25)],
                  [dict(reward=4.,value=2.)],[]]
        records=advantages(episodes,gamma=1.,lam=1.)
        np.testing.assert_allclose([r['advantage'] for r in records],[2.5,1.75,2.])
        np.testing.assert_allclose([r['return'] for r in records],[3.,2.,4.])
        records=advantages([[dict(reward=1.,value=.5),dict(reward=2.,value=.25)]],gamma=1.,lam=.95)
        np.testing.assert_allclose([r['advantage'] for r in records],[2.4125,1.75])
        np.testing.assert_allclose([r['return'] for r in records],[2.9125,2.])
        self.assertEqual(advantages([[]]),[])

    def test_collected_multistep_observations_replay_exactly(self):
        model=self.policy()
        episodes,gains=collect(model,self.problems,horizon=4)
        self.assertEqual(len(episodes),2)
        for problem,episode,gain in zip(self.problems,episodes,gains):
            env=ImprovementEnv(problem,horizon=4)
            self.assertGreater(len(episode),1)
            self.assertLessEqual(len(episode),4)
            rewards=[]
            for step in episode:
                features,mask=env.observe()
                np.testing.assert_array_equal(features,step['state'][0])
                np.testing.assert_array_equal(mask,step['state'][1])
                self.assertTrue(mask[step['action']])
                reward,done,info=env.step(step['action'])
                self.assertAlmostEqual(reward,step['reward'],places=10)
                self.assertEqual(info['accepted'],step['accepted'])
                rewards.append(reward)
            self.assertTrue(env.done)
            expected=100*(env.reference_cost-route_objective(problem,env.route))/max(env.reference_cost,1e-8)
            self.assertAlmostEqual(sum(rewards),expected,places=9)
            self.assertAlmostEqual(gain,expected,places=9)

    def test_real_ppo_update_is_finite_and_changes_actor_and_critic(self):
        model=self.policy()
        episodes,_=collect(model,self.problems,horizon=4)
        records=advantages(episodes)
        actor_before=model.actor[-1].weight.detach().clone()
        critic_before=model.value[-1].weight.detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        stats=ppo_update(model,optimizer,records,np.random.default_rng(9),epochs=2,batch_size=3)
        self.assertEqual(stats['updates'],2*int(np.ceil(len(records)/3)))
        self.assertTrue(np.isfinite(stats['loss']) and np.isfinite(stats['kl']))
        self.assertFalse(torch.equal(actor_before,model.actor[-1].weight))
        self.assertFalse(torch.equal(critic_before,model.value[-1].weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        grads=[p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        empty=ppo_update(model,optimizer,[],np.random.default_rng(9))
        self.assertEqual(empty['updates'],0)


if __name__=='__main__':
    unittest.main()
