"""Independent guards for the formal trainer; synthetic inputs, no train scenes."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .actor_critic import ActorCritic
from . import checkpoints, train_ppo


class TrainingGuardsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.protocol = {'ppo': dict(workers=8, rollout_steps=32, actions=32768,
                                     epochs=4, minibatch=64)}
        self.state = dict(iteration=8, actual_actions=2048, optimizer_updates=128,
                          parent_bc_sha256='same-init', elapsed_s=10., validation_s=0.)

    def test_saved_counters_and_log_acceptance(self):
        self.assertTrue(checkpoints.validate_training_state(
            self.state, self.protocol, 'same-init', self.state.copy()))

    def test_invalid_counters_and_init_rejected(self):
        mutations = [dict(iteration=True), dict(iteration=-1), dict(actual_actions=2049),
                     dict(actual_actions=65536, iteration=256), dict(optimizer_updates=129),
                     dict(parent_bc_sha256='other-init'), dict(elapsed_s=float('nan')),
                     dict(validation_s=-1.), dict(optimizer_updates=1.5)]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                checkpoints.validate_training_state({**self.state, **mutation}, self.protocol, 'same-init')

    def test_uncheckpointed_logged_interactions_rejected(self):
        with self.assertRaisesRegex(ValueError, 'log/checkpoint'):
            checkpoints.validate_training_state(self.state, self.protocol, 'same-init',
                                                {**self.state, 'actual_actions': 2304})

    def test_cpu_rng_roundtrip(self):
        original = checkpoints.capture_rng('cpu')
        try:
            torch.manual_seed(93)
            state = checkpoints.capture_rng('cpu')
            expected = torch.rand(25)
            checkpoints.restore_rng(state)
            self.assertTrue(torch.equal(expected, torch.rand(25)))
            self.assertIsNone(state['cuda'])
        finally:
            checkpoints.restore_rng(original)

    def test_cuda_rng_roundtrip_contract_without_allocating_gpu_context(self):
        # Exercise both CPU and CUDA branches without taking another ~1 GiB
        # context while eight production CPU workers are alive.
        cpu = torch.tensor([1, 2], dtype=torch.uint8)
        cuda = [torch.tensor([3, 4], dtype=torch.uint8)]
        with patch.object(torch, 'get_rng_state', return_value=cpu), \
             patch.object(torch.cuda, 'get_rng_state_all', return_value=cuda), \
             patch.object(torch, 'set_rng_state') as set_cpu, \
             patch.object(torch.cuda, 'set_rng_state_all') as set_cuda:
            state = checkpoints.capture_rng('cuda')
            checkpoints.restore_rng(state)
            self.assertTrue(torch.equal(set_cpu.call_args.args[0], cpu))
            self.assertTrue(torch.equal(set_cuda.call_args.args[0][0], cuda[0]))
            self.assertEqual(set_cpu.call_args.args[0].device.type, 'cpu')

    def test_checkpoint_same_initialization_and_identity_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'initial.pt'
            initial = ActorCritic(message_steps=2)
            checkpoints.save_model(path, initial, protocol_sha256='synthetic-only')
            for _ in range(3):
                child, payload = checkpoints.load_model(path)
                self.assertEqual(payload['training_device'], 'cpu')
                self.assertEqual(payload['source_sha256'], checkpoints.source_identity())
                for key, expected in initial.state_dict().items():
                    self.assertTrue(torch.equal(expected, child.state_dict()[key]))
            payload['source_sha256'] = {}
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'implementation identity'):
                checkpoints.load_model(path)

    def test_nonfinite_checkpoint_parameters_and_optimizer_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.pt'
            model = ActorCritic(kind='mlp')
            checkpoints.save_model(path, model)
            payload = torch.load(path, map_location='cpu', weights_only=False)
            for location in ('model', 'optimizer'):
                bad = copy.deepcopy(payload)
                if location == 'model':
                    next(iter(bad['model'].values())).view(-1)[0] = float('nan')
                else:
                    bad['optimizer'] = {'state': {0: {'exp_avg': torch.tensor([float('inf')])}}}
                torch.save(bad, path)
                with self.subTest(location=location), self.assertRaisesRegex(ValueError, 'nonfinite'):
                    checkpoints.load_model(path)

    def test_bc_trainer_unknown_action_has_no_negative_gradient(self):
        class ProbeActor(torch.nn.Module):
            last = None
            def __init__(self, **kwargs):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([2., 1., 1000.]))
                ProbeActor.last = self
            @property
            def device(self):
                return self.logits.device
            def forward(self, observations):
                return self.logits.unsqueeze(0).expand(len(observations), -1), torch.zeros(len(observations))

        sample = dict(observation={}, evidence_mask=[True, True, False],
                      ties=[True, False, False], weight=1., scene_id='synthetic')
        protocol = dict(training_seed=99, bc=dict(initial_updates=1, batch_size=1, learning_rate=1e-4))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'outputs/teacher_initial/teacher.pkl'
            data.parent.mkdir(parents=True)
            data.write_bytes(b'synthetic teacher fixture, never formal data')
            with patch.object(train_ppo, 'ROOT', root), \
                 patch.object(train_ppo, 'protocol', return_value=(protocol, 'synthetic')), \
                 patch('rc3_rl_g2.actor_critic.ActorCritic', ProbeActor), \
                 patch('rc3_rl_g2.teacher_dataset.load_teacher', return_value={'samples': [sample]}), \
                 patch('rc3_rl_g2.checkpoints.save_model'):
                train_ppo.bc_train('cpu', continued=False)
            self.assertEqual(ProbeActor.last.logits.grad[2].item(), 0.)
            self.assertEqual(ProbeActor.last.logits[2].item(), 1000.)
            self.assertGreater(ProbeActor.last.logits[0].item(), 2.)
            self.assertLess(ProbeActor.last.logits[1].item(), 1.)


if __name__ == '__main__':
    unittest.main()
