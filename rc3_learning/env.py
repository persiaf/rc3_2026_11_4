"""RL4CO environment: automatically service every uncovered reachable task."""
import torch
from rl4co.envs.common.base import RL4COEnvBase
from torchrl.data import Composite, Unbounded


class ReachabilityCoverRoutingEnv(RL4COEnvBase):
    name = 'reachability_cover_routing'

    def __init__(self, device='cpu', seed=0):
        super().__init__(device=device, seed=seed, allow_done_after_reset=True)
        # Candidate/task dimensions are dynamic; legality is checked in _step.
        self.observation_spec = Composite()
        self.action_spec = Unbounded(shape=(), dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    @staticmethod
    def get_action_mask(td):
        gains = (td['cover'] & td['remaining'].unsqueeze(-1)).sum(dim=1)
        rows = torch.arange(td.batch_size[0], device=td.device)
        travel = td['costs'][rows, td['current_node'], 1:]
        return (td['candidate_mask'] & (gains > 0) & torch.isfinite(travel)
                & torch.isfinite(td['costs'][:, 1:, 0]))

    def _reset(self, td, batch_size=None):
        state = td.clone(recurse=False)
        b, device = td.batch_size[0], td.device
        state.update(remaining=td['task_mask'].clone(),
                     current_node=torch.zeros(b, dtype=torch.long, device=device),
                     total_cost=torch.zeros(b, device=device),
                     service_count=torch.zeros(b, dtype=torch.long, device=device),
                     reward=torch.zeros(b, 1, device=device))
        mask = self.get_action_mask(state)
        complete = ~state['remaining'].any(dim=1)
        failed = ~complete & ~mask.any(dim=1)
        state.update(action_mask=mask, failed=failed, completed=complete,
                     done=(complete | failed).unsqueeze(-1))
        return state

    def _step(self, td):
        action = td['action'].reshape(-1)
        b, m = td['candidate_mask'].shape
        rows = torch.arange(b, device=td.device)
        active = ~td['done'].squeeze(-1)
        if ((action[active] < 0) | (action[active] >= m)).any():
            raise ValueError('active action is outside the candidate set')
        safe_action = torch.where(active, action, 0)
        if not td['action_mask'][rows[active], safe_action[active]].all():
            raise ValueError('policy selected an infeasible or zero-gain action')
        destination = safe_action + 1
        travel = td['costs'][rows, td['current_node'], destination]
        increment = torch.where(active, travel + td['stop_penalty'], 0.)
        covered = td['cover'][rows, :, safe_action]
        remaining = td['remaining'] & ~(covered & active.unsqueeze(-1))
        current = torch.where(active, destination, td['current_node'])
        td.update(remaining=remaining, current_node=current,
                  service_count=td['service_count']+active.long())
        mask = self.get_action_mask(td)
        complete = ~remaining.any(dim=1)
        failed = ~complete & ~mask.any(dim=1)
        done = complete | failed
        # Charge return exactly once, including when a partial scene exhausts actions.
        returning = active & done
        increment = increment + torch.where(returning, td['costs'][rows, current, 0], 0.)
        td.update(action_mask=mask, completed=complete, failed=failed, done=done.unsqueeze(-1),
                  total_cost=td['total_cost']+increment,
                  reward=(-increment/td['reference_cost']).unsqueeze(-1))
        return td

    def _get_reward(self, td, actions):
        if td['failed'].any() or not td['completed'].all():
            raise ValueError('cannot reward an incomplete solution as a completed tour')
        return -td['total_cost']/td['reference_cost']

    def check_solution_validity(self, td, actions):
        replay = self._reset(td)
        for action in actions.unbind(dim=1):
            replay['action'] = action
            replay = self._step(replay)
        if not replay['completed'].all() or replay['failed'].any():
            raise ValueError('solution leaves tasks incomplete')
        if not torch.allclose(replay['total_cost'], td['total_cost'], atol=2e-4, rtol=1e-5):
            raise ValueError('recorded route cost differs from replay')
