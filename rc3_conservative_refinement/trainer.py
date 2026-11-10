"""206-dimensional on-policy PPO with transactional, per-state guarded updates.

This module imports no historical trainer, scene registry, or scientific run.
The caller owns request admission, durable resource accounting, checkpoint I/O,
and authorization.  Calling ``update`` performs real optimizer updates; merely
constructing the trainer or calculating its losses does not.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import math
import random
import time

import numpy as np
import torch

from .guard import GuardLimits, check_update, restore_if_rejected
from .model import full_return_advantage, remaining_cost_loss


def tensor_digest(state):
    """The historical tensor identity convention, without importing its runner."""
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def checkpoint_equal(left, right):
    """Exact nested checkpoint comparison, including NumPy's RNG array."""
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.cpu(), right.cpu()))
    if isinstance(left, np.ndarray):
        return (isinstance(right, np.ndarray) and left.dtype == right.dtype
                and left.shape == right.shape and bool(np.array_equal(left, right)))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(checkpoint_equal(value, right[key]) for key, value in left.items()))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(checkpoint_equal(a, b) for a, b in zip(left, right)))
    return bool(left == right)


_state_equal = checkpoint_equal


@dataclass(frozen=True)
class PPOConfig:
    actor_lr: float = 3e-5
    critic_lr: float = 1e-4
    beta: float = .02
    ratio_clip: float = .1
    gradient_clip: float = 1.
    advantage_rms_epsilon: float = 1e-8
    old_logits_atol: float = 1e-6
    consecutive_rejection_limit: int = 16
    warm_update_cap: int = 32

    def __post_init__(self):
        for name in ('actor_lr', 'critic_lr', 'gradient_clip', 'advantage_rms_epsilon'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('beta', 'old_logits_atol'):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if isinstance(self.ratio_clip, bool) or not 0 < self.ratio_clip < 1:
            raise ValueError('ratio_clip must be strictly between zero and one')
        if type(self.consecutive_rejection_limit) is not int or self.consecutive_rejection_limit < 1:
            raise ValueError('consecutive_rejection_limit must be a positive integer')
        if type(self.warm_update_cap) is not int or self.warm_update_cap < 0:
            raise ValueError('warm_update_cap must be a nonnegative integer')


def masked_log_softmax(logits, mask):
    if logits.shape != mask.shape or mask.dtype != torch.bool or logits.ndim != 2:
        raise ValueError('Expected T x M logits and the identical boolean mask')
    if not bool(mask.any(-1).all()) or not bool(torch.isfinite(logits[mask]).all()):
        raise ValueError('Every state needs at least one finite legal logit')
    result = torch.log_softmax(logits.masked_fill(~mask, -torch.inf), -1)
    if not bool(torch.isfinite(result[mask]).all()):
        raise ValueError('Legal log probabilities must be finite')
    return result


def ppo_state_loss(new_logp, old_logp, advantage, base_logp, mask, actions, *,
                   beta=.02, ratio_clip=.1):
    """Per-state objective, with fixed old likelihoods and complete advantages."""
    if old_logp.requires_grad or advantage.requires_grad or base_logp.requires_grad:
        raise ValueError('Old probabilities, SL anchor and advantages must be detached')
    if (new_logp.ndim != 2 or new_logp.shape != base_logp.shape
            or new_logp.shape != mask.shape or old_logp.shape != actions.shape
            or advantage.shape != actions.shape or actions.shape != (len(new_logp),)):
        raise ValueError('PPO state shapes disagree')
    if actions.dtype != torch.long or not bool(((actions >= 0) & (actions < mask.shape[1])).all()):
        raise ValueError('Actions must be zero-based candidate indices')
    if not bool(mask.gather(1, actions[:, None]).all()):
        raise ValueError('An actual sampled action was illegal')
    selected = new_logp.gather(1, actions[:, None]).squeeze(1)
    ratio = (selected - old_logp).exp()
    if not bool(torch.isfinite(ratio).all()) or not bool(torch.isfinite(advantage).all()):
        raise ValueError('PPO ratio and advantage must be finite')
    gain = torch.minimum(ratio * advantage,
                         ratio.clamp(1 - ratio_clip, 1 + ratio_clip) * advantage)
    safe_base = base_logp.masked_fill(~mask, 0.)
    safe_new = new_logp.masked_fill(~mask, 0.)
    kl = (base_logp.exp() * mask * (safe_base - safe_new)).sum(-1)
    objective = -gain + beta * kl
    if not bool(torch.isfinite(objective).all()):
        raise ValueError('PPO loss must be finite')
    return objective, ratio, kl


def _gradient_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum())
                         for p in parameters if p.grad is not None))


def _finite_parameters(module):
    return all(bool(torch.isfinite(value).all()) for value in module.parameters())


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone(),
                cuda=[state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_initialized() else None)


def _set_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        if not torch.cuda.is_available():
            raise ValueError('Cannot restore a CUDA RNG checkpoint without CUDA')
        torch.cuda.set_rng_state_all(state['cuda'])


class ConservativePPO:
    """One opportunity = one Actor candidate step plus one independent Critic step.

    Episodes must have fresh online states from the current correction, and each
    ``episode_id`` must be unique.  PPO collection likelihoods are not replaced by
    the post-update likelihoods.  A rejected candidate still consumes its update
    opportunity and its actual Adam call.  Neither thresholds nor learning rates
    are relaxed.  The resource callback is called *before* each optimizer call;
    it must durably debit or raise before the update is attempted.
    """
    SCHEMA = 'rc3-conservative-ppo-checkpoint-v1'

    def __init__(self, actor, critic, *, config=None, limits=None, log=None, context=None):
        self.actor, self.critic = actor, critic
        self.config = config or PPOConfig()
        self.limits = limits or GuardLimits()
        if not isinstance(self.config, PPOConfig) or not isinstance(self.limits, GuardLimits):
            raise ValueError('An explicit PPOConfig and GuardLimits are required')
        if actor.correction.input_dim != 206 or critic.input_dim != 268:
            raise ValueError('This trainer only supports the new 206/268 adapter')
        actor_ids = set(map(id, actor.correction.parameters()))
        if actor_ids & set(map(id, critic.parameters())):
            raise ValueError('Actor and Critic must have disjoint parameters')
        if actor_ids & set(map(id, actor.encoder.parameters())):
            raise ValueError('Frozen base must not be part of the correction optimizer')
        actor.train()
        self.base_sha = tensor_digest(actor.encoder.state_dict())
        self.actor_opt = torch.optim.Adam(actor.correction.parameters(), lr=self.config.actor_lr,
                                         betas=(.9, .999), eps=1e-8, weight_decay=0.)
        self.critic_opt = torch.optim.Adam(critic.parameters(), lr=self.config.critic_lr,
                                          betas=(.9, .999), eps=1e-8, weight_decay=0.)
        self.log = log or (lambda row: None)
        self.context = dict(context or {})
        self.counts = dict(opportunities=0, actor_optimizer_attempts=0, actor_optimizer_calls=0,
                           actor_accepted=0, actor_rejected=0, critic_optimizer_attempts=0,
                           critic_optimizer_calls=0, critic_accepted=0, failed_opportunities=0,
                           consecutive_rejected=0, critic_warm_opportunities=0)
        self.used_episode_ids = set()
        self.failed = False
        self._busy = False
        self._assert_frozen()

    def _emit(self, row):
        self.log(dict(self.context, **row))

    def _assert_frozen(self):
        if self.actor.encoder.training or any(p.requires_grad or p.grad is not None
                                              for p in self.actor.encoder.parameters()):
            raise RuntimeError('Frozen SL encoder entered training or acquired gradients')
        if tensor_digest(self.actor.encoder.state_dict()) != self.base_sha:
            raise RuntimeError('Frozen SL encoder tensors changed')

    def _batch_logits(self, batch):
        return self.actor.correction(batch['z0'], batch['x'], batch['mask'])

    def _admit(self, episodes, *, supervised=False, critic_replay=False):
        if not episodes:
            raise ValueError('An update opportunity requires fresh complete episodes')
        devices = {p.device for p in self.actor.correction.parameters()}
        if len(devices) != 1 or devices != {p.device for p in self.critic.parameters()}:
            raise ValueError('Actor and Critic must use one shared compute device')
        device = next(iter(devices))
        expected_sha = tensor_digest(self.actor.correction.state_dict())
        admitted, identifiers = [], []
        for episode in episodes:
            identifier = episode.get('episode_id', episode.get('record', {}).get('request_id'))
            if (not isinstance(identifier, str) or not identifier
                    or (identifier in self.used_episode_ids and not critic_replay)):
                raise ValueError('A new unique episode_id is required; rollout replay is forbidden')
            identifiers.append(identifier)
            if 'actor_tensor_sha' in episode and episode['actor_tensor_sha'] != expected_sha:
                raise ValueError('This rollout came from a different correction checkpoint')
            task_count = episode['task_count']
            if type(task_count) is not int or task_count < 0:
                raise ValueError('Each episode needs its fixed initial task count')
            batch = episode['batch']
            if batch is None:
                if task_count != 0 or (not supervised and episode['targets']['returns']):
                    raise ValueError('Only a genuinely empty task can have no policy states')
                continue
            b = {key: batch[key].detach().to(device) for key in ('x', 'v', 'z0', 'mask')}
            b['keys'] = batch['keys']
            T, M = b['mask'].shape
            if T == 0 or b['x'].shape != (T, M, 206) or b['z0'].shape != (T, M) or b['v'].shape != (T, 268):
                raise ValueError('Real packed episodes must use T x M x 206 and T x 268 features')
            if len(b['keys']) != T or any(len(key) != 7 or key[4] != 'post-service-v1'
                    or key[5] is not self.actor.distances_enabled for key in b['keys']):
                raise ValueError('Episode source/condition keys do not match the 206 Actor')
            if any(len({key[i] for key in b['keys']}) != 1 for i in (0, 2, 3, 4, 5, 6)):
                raise ValueError('Episode mixes problem/base/D or preparation identities')
            actions = torch.as_tensor(episode['actions'], device=device, dtype=torch.long).detach()
            with torch.no_grad():
                immediate_logits = self._batch_logits(b).detach()
            old_logits = immediate_logits if supervised else torch.as_tensor(episode['old_logits'], device=device).detach()
            old_values = torch.zeros(T, device=device, dtype=b['v'].dtype) if supervised else torch.as_tensor(
                episode['V_old'], device=device, dtype=b['v'].dtype).detach()
            returns = torch.zeros(T, device=device, dtype=b['v'].dtype) if supervised else torch.as_tensor(
                episode['targets']['returns'], device=device, dtype=b['v'].dtype).detach()
            if (actions.shape != (T,) or old_values.shape != (T,) or returns.shape != (T,)
                    or old_logits.shape != (T, M) or b['mask'].dtype != torch.bool):
                raise ValueError('State, action, complete target and old-policy shapes disagree')
            old_lp = masked_log_softmax(old_logits, b['mask'])
            if not bool(((actions >= 0) & (actions < M)).all()) or not bool(b['mask'].gather(1, actions[:, None]).all()):
                raise ValueError('Every recorded online action must be a legal candidate index')
            if not torch.allclose(immediate_logits[b['mask']], old_logits[b['mask']],
                                  atol=self.config.old_logits_atol, rtol=0.):
                raise ValueError('Off-policy rollout: collection logits differ from current Actor')
            advantage = full_return_advantage(returns, old_values)
            if not supervised:
                with torch.no_grad():
                    # Reject malformed cost components before any Actor call.
                    prediction = self.critic(b['v'], task_count)
                    checked = remaining_cost_loss(prediction, episode['targets'], task_count)
                    if not bool(torch.isfinite(checked).all()):
                        raise ValueError('Complete Critic targets must be finite')
            admitted.append(dict(batch=b, actions=actions, old_logits=old_logits,
                                 immediate_logits=immediate_logits,
                                 old_logp=old_lp.gather(1, actions[:, None]).squeeze(1),
                                 base_logp=masked_log_softmax(b['z0'], b['mask']).detach(),
                                 V_old=old_values, G=returns, advantage=advantage,
                                 targets=None if supervised else episode['targets'], task_count=task_count,
                                 episode_id=identifier))
        if len(identifiers) != len(set(identifiers)):
            raise ValueError('A rollout cannot be duplicated within an update opportunity')
        if not admitted:
            raise ValueError('No policy states in this opportunity; no optimizer update is valid')
        return admitted, identifiers

    def _critic_step(self, episodes, cursor, role):
        """Identical fitting path for registered warm-up and online main data."""
        self.critic_opt.zero_grad(set_to_none=True)
        loss, predictions = 0., []
        for episode in episodes:
            prediction = self.critic(episode['batch']['v'], episode['task_count'])
            part = remaining_cost_loss(prediction, episode['targets'], episode['task_count']).sum() / len(episodes)
            if not bool(torch.isfinite(part)):
                raise RuntimeError('Nonfinite Critic loss')
            part.backward()
            loss += float(part.detach())
            predictions.append({key: value.detach().cpu().tolist() for key, value in prediction.items()})
        if any(parameter.grad is not None for parameter in self.actor.correction.parameters()):
            raise RuntimeError('Critic fitting contaminated Actor gradients')
        gradient = _gradient_norm(self.critic.parameters())
        if not math.isfinite(gradient):
            raise RuntimeError('Nonfinite Critic gradient')
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip,
                                       error_if_nonfinite=True)
        self.counts['critic_optimizer_attempts'] += 1
        self._emit(dict(event='optimizer_attempt', role=role, opportunity=cursor,
                        counts=copy.deepcopy(self.counts)))
        self.counts['critic_optimizer_calls'] += 1
        self.critic_opt.step()
        if not _finite_parameters(self.critic):
            raise RuntimeError('Nonfinite Critic parameters')
        self.counts['critic_accepted'] += 1
        self.critic_opt.zero_grad(set_to_none=True)
        return loss, gradient, predictions

    def update(self, episodes):
        if self.failed or self._busy:
            raise RuntimeError('Failed or in-progress trainer cannot start another update')
        if self.counts['consecutive_rejected'] >= self.config.consecutive_rejection_limit:
            raise RuntimeError('Fixed consecutive rejection stop reached; thresholds cannot be widened')
        self._assert_frozen()
        self.actor.train()
        e, identifiers = self._admit(episodes)
        self.used_episode_ids.update(identifiers)
        self.counts['opportunities'] += 1
        cursor = self.counts['opportunities']
        self._busy = True
        start = time.perf_counter()
        actor_before = copy.deepcopy(self.actor.correction.state_dict())
        actor_opt_before = copy.deepcopy(self.actor_opt.state_dict())
        critic_before = copy.deepcopy(self.critic.state_dict())
        critic_opt_before = copy.deepcopy(self.critic_opt.state_dict())
        accepted_before = self.counts['actor_accepted'], self.counts['critic_accepted']
        try:
            raw = torch.cat([episode['advantage'] for episode in e])
            rms = float(raw.double().square().mean().sqrt()) + self.config.advantage_rms_epsilon
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            actor_loss, ratios, anchors = 0., [], []
            for episode in e:
                b = episode['batch']
                lp = masked_log_softmax(self._batch_logits(b), b['mask'])
                objective, ratio, kl = ppo_state_loss(
                    lp, episode['old_logp'], (episode['advantage'] / rms).detach(),
                    episode['base_logp'], b['mask'], episode['actions'],
                    beta=self.config.beta, ratio_clip=self.config.ratio_clip)
                part = objective.sum() / len(e)  # Same per-episode sum objective in ON/OFF.
                part.backward()
                actor_loss += float(part.detach())
                ratios.append(ratio.detach())
                anchors.append(kl.detach())
            actor_gradient = _gradient_norm(self.actor.correction.parameters())
            if not math.isfinite(actor_gradient):
                raise RuntimeError('Nonfinite Actor gradient')
            torch.nn.utils.clip_grad_norm_(self.actor.correction.parameters(), self.config.gradient_clip,
                                           error_if_nonfinite=True)
            self.counts['actor_optimizer_attempts'] += 1
            self._emit(dict(event='optimizer_attempt', role='actor', opportunity=cursor,
                            counts=copy.deepcopy(self.counts)))
            self.counts['actor_optimizer_calls'] += 1
            self.actor_opt.step()
            if not _finite_parameters(self.actor.correction):
                raise RuntimeError('Nonfinite Actor parameters')
            with torch.no_grad():
                guards = [check_update(episode['batch']['z0'], episode['immediate_logits'],
                    self._batch_logits(episode['batch']), episode['batch']['mask'], self.limits)
                    for episode in e]
            accepted = all(report['accepted'] for report in guards)
            restoration = restore_if_rejected(self.actor.correction, self.actor_opt,
                actor_before, actor_opt_before, {'accepted': accepted})
            self.counts['actor_accepted' if accepted else 'actor_rejected'] += 1
            self.counts['consecutive_rejected'] = 0 if accepted else self.counts['consecutive_rejected'] + 1
            # Clear Actor gradients before fitting the disjoint Critic.  This
            # also makes cross-network gradient contamination observable.
            self.actor_opt.zero_grad(set_to_none=True)
            critic_loss, critic_gradient, predictions = self._critic_step(e, cursor, 'critic')
            self._assert_frozen()
            probabilities = torch.cat(ratios)
            report = dict(event='ppo_opportunity', opportunity=cursor, accepted=accepted,
                rollback_exact=bool(restoration.get('restored')),
                actor_loss=actor_loss, critic_loss=critic_loss,
                actor_gradient_norm=actor_gradient, critic_gradient_norm=critic_gradient,
                actor_gradient_clipped=actor_gradient > self.config.gradient_clip,
                critic_gradient_clipped=critic_gradient > self.config.gradient_clip,
                advantage_rms=rms, advantage_mean=float(raw.mean()),
                advantage_std=float(raw.std(unbiased=False)), negative_advantages=int((raw < 0).sum()),
                ratio_min=float(probabilities.min()), ratio_max=float(probabilities.max()),
                ratio_clip_fraction=float(((probabilities < 1 - self.config.ratio_clip)
                    | (probabilities > 1 + self.config.ratio_clip)).float().mean()),
                base_kl_loss_mean=float(torch.cat(anchors).mean()),
                guard_maxima={name: max(item['maxima'][name] for item in guards)
                              for name in guards[0]['maxima']},
                guard=guards, counts=copy.deepcopy(self.counts),
                stop_requested=self.counts['consecutive_rejected'] >= self.config.consecutive_rejection_limit,
                actor_tensor_sha=tensor_digest(self.actor.correction.state_dict()), base_tensor_sha=self.base_sha,
                elapsed_s=time.perf_counter() - start,
                episodes=[dict(episode_id=episode['episode_id'],
                    actions_zero_based=episode['actions'].cpu().tolist(),
                    returns=episode['G'].cpu().tolist(), old_values=episode['V_old'].cpu().tolist(),
                    advantages=episode['advantage'].cpu().tolist(),
                    critic_before_update=prediction, task_count=episode['task_count'])
                    for episode, prediction in zip(e, predictions)])
            self._emit(report)
            return report
        except Exception as error:
            # Restore both mutable networks/Adam after an exceptional partial
            # opportunity.  Actual calls, failures, and used IDs remain debited.
            self.actor.correction.load_state_dict(actor_before, strict=True)
            self.actor_opt.load_state_dict(actor_opt_before)
            self.critic.load_state_dict(critic_before, strict=True)
            self.critic_opt.load_state_dict(critic_opt_before)
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            self.counts['failed_opportunities'] += 1
            self.counts['actor_accepted'], self.counts['critic_accepted'] = accepted_before
            self.failed = True
            self._emit(dict(event='opportunity_failed', opportunity=cursor,
                            error_type=type(error).__name__, error=str(error),
                            actual_update_calls_retained=True, mutable_state_restored=True,
                            counts=copy.deepcopy(self.counts)))
            raise
        finally:
            self._busy = False

    def critic_only_update(self, episodes):
        """Registered initial Critic warm-up, without Actor updates or reuse as PPO.

        Replaying the registered zero-policy warm pool is permitted for fitting
        this evaluator only.  It never consumes main Actor opportunities, never
        replaces online collection, and is forbidden after the first Actor call.
        The external ledger must additionally enforce its approved warm budget.
        """
        if self.failed or self._busy or self.counts['actor_optimizer_calls']:
            raise RuntimeError('Critic warm-up is initial-only and requires a healthy closed boundary')
        if self.counts['critic_warm_opportunities'] >= self.config.warm_update_cap:
            raise RuntimeError('Fixed initial Critic warm-up cap reached')
        self._assert_frozen()
        self.actor.train()
        admitted, identifiers = self._admit(episodes, critic_replay=True)
        self.used_episode_ids.update(identifiers)  # Cannot later use these as on-policy PPO data.
        self.counts['critic_warm_opportunities'] += 1
        cursor = self.counts['critic_warm_opportunities']
        before = copy.deepcopy(self.critic.state_dict())
        opt_before = copy.deepcopy(self.critic_opt.state_dict())
        accepted_before = self.counts['critic_accepted']
        actor_sha = tensor_digest(self.actor.correction.state_dict())
        actor_opt_before = copy.deepcopy(self.actor_opt.state_dict())
        self._busy = True
        start = time.perf_counter()
        try:
            self.actor_opt.zero_grad(set_to_none=True)
            loss, gradient, predictions = self._critic_step(admitted, cursor, 'critic_warm')
            if (tensor_digest(self.actor.correction.state_dict()) != actor_sha
                    or not _state_equal(self.actor_opt.state_dict(), actor_opt_before)):
                raise RuntimeError('Critic warm-up mutated Actor or its Adam')
            self._assert_frozen()
            report = dict(event='critic_warm_opportunity', warm_opportunity=cursor,
                critic_loss=loss, critic_gradient_norm=gradient,
                critic_gradient_clipped=gradient > self.config.gradient_clip,
                counts=copy.deepcopy(self.counts), actor_tensor_sha=actor_sha,
                elapsed_s=time.perf_counter() - start,
                episodes=[dict(episode_id=episode['episode_id'],
                    actions_zero_based=episode['actions'].cpu().tolist(),
                    returns=episode['G'].cpu().tolist(), old_values=episode['V_old'].cpu().tolist(),
                    critic_before_update=prediction, task_count=episode['task_count'])
                    for episode, prediction in zip(admitted, predictions)])
            self._emit(report)
            return report
        except Exception as error:
            self.critic.load_state_dict(before, strict=True)
            self.critic_opt.load_state_dict(opt_before)
            self.critic_opt.zero_grad(set_to_none=True)
            self.counts['critic_accepted'] = accepted_before
            self.counts['failed_opportunities'] += 1
            self.failed = True
            self._emit(dict(event='opportunity_failed', role='critic_warm', opportunity=cursor,
                            error_type=type(error).__name__, error=str(error),
                            actual_update_calls_retained=True, mutable_state_restored=True,
                            counts=copy.deepcopy(self.counts)))
            raise
        finally:
            self._busy = False

    def supervised_update(self, episodes):
        """Matched continued-supervision control using the identical 206 Actor.

        The caller supplies frozen registered teacher actions (zero-based),
        their complete feature trajectories, initial task counts and unique
        episode IDs.  This is teacher-forced cross entropy, not PPO.  It uses
        the same correction Adam and guard, but performs no Critic update.
        An otherwise unused independent Critic is retained only so that all
        controls can share the exact checkpoint/resume machinery.
        """
        if self.failed or self._busy:
            raise RuntimeError('Failed or in-progress trainer cannot start another update')
        if self.counts['consecutive_rejected'] >= self.config.consecutive_rejection_limit:
            raise RuntimeError('Fixed consecutive rejection stop reached')
        self._assert_frozen()
        self.actor.train()
        admitted, identifiers = self._admit(episodes, supervised=True)
        self.used_episode_ids.update(identifiers)
        self.counts['opportunities'] += 1
        cursor = self.counts['opportunities']
        start = time.perf_counter()
        self._busy = True
        before = copy.deepcopy(self.actor.correction.state_dict())
        opt_before = copy.deepcopy(self.actor_opt.state_dict())
        accepted_before = self.counts['actor_accepted']
        critic_sha = tensor_digest(self.critic.state_dict())
        try:
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            loss = 0.
            for episode in admitted:
                b = episode['batch']
                lp = masked_log_softmax(self._batch_logits(b), b['mask'])
                # Route-mean CE makes every complete teacher episode one unit.
                part = -lp.gather(1, episode['actions'][:, None]).mean() / len(admitted)
                if not bool(torch.isfinite(part)):
                    raise RuntimeError('Nonfinite matched supervised loss')
                part.backward()
                loss += float(part.detach())
            gradient = _gradient_norm(self.actor.correction.parameters())
            if not math.isfinite(gradient):
                raise RuntimeError('Nonfinite matched supervised gradient')
            torch.nn.utils.clip_grad_norm_(self.actor.correction.parameters(), self.config.gradient_clip,
                                           error_if_nonfinite=True)
            self.counts['actor_optimizer_attempts'] += 1
            self._emit(dict(event='optimizer_attempt', role='matched_SL', opportunity=cursor,
                            counts=copy.deepcopy(self.counts)))
            self.counts['actor_optimizer_calls'] += 1
            self.actor_opt.step()
            if not _finite_parameters(self.actor.correction):
                raise RuntimeError('Nonfinite matched supervised parameters')
            with torch.no_grad():
                guards = [check_update(episode['batch']['z0'], episode['immediate_logits'],
                    self._batch_logits(episode['batch']), episode['batch']['mask'], self.limits)
                    for episode in admitted]
            accepted = all(item['accepted'] for item in guards)
            rollback = restore_if_rejected(self.actor.correction, self.actor_opt, before,
                                           opt_before, {'accepted': accepted})
            self.counts['actor_accepted' if accepted else 'actor_rejected'] += 1
            self.counts['consecutive_rejected'] = 0 if accepted else self.counts['consecutive_rejected'] + 1
            self.actor_opt.zero_grad(set_to_none=True)
            if tensor_digest(self.critic.state_dict()) != critic_sha or any(
                    p.grad is not None for p in self.critic.parameters()):
                raise RuntimeError('Supervised control mutated or differentiated its unused Critic')
            self._assert_frozen()
            report = dict(event='matched_supervised_opportunity', opportunity=cursor,
                accepted=accepted, rollback_exact=bool(rollback.get('restored')),
                actor_loss=loss, actor_gradient_norm=gradient,
                actor_gradient_clipped=gradient > self.config.gradient_clip,
                guard_maxima={name: max(item['maxima'][name] for item in guards)
                              for name in guards[0]['maxima']},
                guard=guards, counts=copy.deepcopy(self.counts),
                stop_requested=self.counts['consecutive_rejected'] >= self.config.consecutive_rejection_limit,
                actor_tensor_sha=tensor_digest(self.actor.correction.state_dict()),
                base_tensor_sha=self.base_sha, elapsed_s=time.perf_counter() - start,
                episodes=[dict(episode_id=episode['episode_id'],
                    teacher_actions_zero_based=episode['actions'].cpu().tolist(),
                    task_count=episode['task_count']) for episode in admitted])
            self._emit(report)
            return report
        except Exception as error:
            self.actor.correction.load_state_dict(before, strict=True)
            self.actor_opt.load_state_dict(opt_before)
            self.actor_opt.zero_grad(set_to_none=True)
            self.counts['actor_accepted'] = accepted_before
            self.counts['failed_opportunities'] += 1
            self.failed = True
            self._emit(dict(event='opportunity_failed', role='matched_SL', opportunity=cursor,
                            error_type=type(error).__name__, error=str(error),
                            actual_update_calls_retained=True, mutable_state_restored=True,
                            counts=copy.deepcopy(self.counts)))
            raise
        finally:
            self._busy = False

    def state_dict(self, metadata):
        """Capture a closed boundary; caller binds/atomically saves exact bytes.

        ``metadata`` must include the external request/query ledger, schedule
        cursor, source/protocol/H/environment/seed bindings and any generator
        outside global Python/NumPy/Torch RNG.  On restore it is compared in
        full, so the caller must reconcile its durable ledger before loading.
        """
        if self._busy:
            raise RuntimeError('Cannot checkpoint a partial optimizer opportunity')
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError('Nonempty externally bound checkpoint metadata is required')
        self._assert_frozen()
        return copy.deepcopy(dict(schema=self.SCHEMA, config=asdict(self.config),
            limits=asdict(self.limits), distances_enabled=self.actor.distances_enabled,
            correction_amplitude=self.actor.correction.max_adjustment,
            base_tensor_sha=self.base_sha, correction=self.actor.correction.state_dict(),
            critic=self.critic.state_dict(), actor_optimizer=self.actor_opt.state_dict(),
            critic_optimizer=self.critic_opt.state_dict(), counts=self.counts,
            used_episode_ids=sorted(self.used_episode_ids), failed=self.failed,
            rng=_rng_state(), metadata=metadata))

    def load_state_dict(self, payload, expected_metadata):
        if self._busy:
            raise RuntimeError('Cannot restore over a partial opportunity')
        self._assert_frozen()
        checks = dict(schema=self.SCHEMA, config=asdict(self.config), limits=asdict(self.limits),
                      distances_enabled=self.actor.distances_enabled,
                      correction_amplitude=self.actor.correction.max_adjustment,
                      base_tensor_sha=self.base_sha, metadata=expected_metadata)
        if not isinstance(expected_metadata, dict) or not expected_metadata:
            raise ValueError('Independently reconciled expected metadata is required')
        if any(not _state_equal(payload.get(key), expected) for key, expected in checks.items()):
            raise ValueError('Checkpoint protocol/source/base/condition/config or external ledger differs')
        if set(payload['counts']) != set(self.counts) or any(type(value) is not int or value < 0
                                                           for value in payload['counts'].values()):
            raise ValueError('Checkpoint resource counters are malformed')
        if any(payload['counts'][key] < value for key, value in self.counts.items()
               if key != 'consecutive_rejected'):
            raise ValueError('Checkpoint restore would rewind already consumed resources')
        if type(payload['failed']) is not bool or len(payload['used_episode_ids']) != len(set(payload['used_episode_ids'])):
            raise ValueError('Checkpoint failure or episode ledger is malformed')
        if not self.used_episode_ids.issubset(payload['used_episode_ids']):
            raise ValueError('Checkpoint would rewind the used rollout ledger')
        before = self.state_dict({'restore_safety_snapshot': True})
        try:
            self.actor.correction.load_state_dict(payload['correction'], strict=True)
            self.critic.load_state_dict(payload['critic'], strict=True)
            self.actor_opt.load_state_dict(payload['actor_optimizer'])
            self.critic_opt.load_state_dict(payload['critic_optimizer'])
            if not _finite_parameters(self.actor.correction) or not _finite_parameters(self.critic):
                raise ValueError('Checkpoint contains nonfinite parameters')
            for key in ('correction', 'critic', 'actor_optimizer', 'critic_optimizer'):
                actual = {'correction': self.actor.correction, 'critic': self.critic,
                          'actor_optimizer': self.actor_opt, 'critic_optimizer': self.critic_opt}[key].state_dict()
                if not _state_equal(actual, payload[key]):
                    raise ValueError(f'Checkpoint {key} did not restore exactly')
            _set_rng_state(payload['rng'])
            self.counts = copy.deepcopy(payload['counts'])
            self.used_episode_ids = set(payload['used_episode_ids'])
            self.failed = payload['failed']
            self.actor.train()
            self.actor_opt.zero_grad(set_to_none=True)
            self.critic_opt.zero_grad(set_to_none=True)
            self._assert_frozen()
        except Exception:
            self.actor.correction.load_state_dict(before['correction'], strict=True)
            self.critic.load_state_dict(before['critic'], strict=True)
            self.actor_opt.load_state_dict(before['actor_optimizer'])
            self.critic_opt.load_state_dict(before['critic_optimizer'])
            self.counts = before['counts']
            self.used_episode_ids = set(before['used_episode_ids'])
            self.failed = before['failed']
            _set_rng_state(before['rng'])
            raise
