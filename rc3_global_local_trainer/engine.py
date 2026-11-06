"""Mask-exact S1 CE and complete-return mixed REINFORCE batch updates.

Hyperparameters/edge-case policy must be explicitly bound by the caller. This
module implements interfaces; importing it never approves a research protocol
or runs optimization. Supervised replay never queries a teacher or full H.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from rc3_global_local.environment import Environment
from rc3_global_local.policy import tensor_hash
from rc3_global_local.problem import integer


DEFAULT_CONFIG = dict(phase='RL', batch_instances=4, gamma=1., clip_global_norm=1.,
                      mixture_weights=[1., .25, .25], entropy_weight=0.,
                      advantage_standardization=False, max_optimizer_updates=None,
                      binding={})


class InconclusiveBatch(RuntimeError):
    """A preregistered edge/failure stops, without substituting another sample."""


def make_optimizer(model, config=None):
    c = dict(lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    if config:
        for key in c:
            if key in config:
                c[key] = config[key]
    return torch.optim.Adam(model.parameters(), **c)


def fork_from_supervised(model, optimizer_config=None):
    """Exact weight clones with separate storage and fresh control optimizers."""
    result = {}
    for name in ('frozen_SL', 'RL', 'continued_SL'):
        clone = copy.deepcopy(model)
        clone.zero_grad(set_to_none=True)
        if name == 'frozen_SL':
            clone.eval()
            for parameter in clone.parameters():
                parameter.requires_grad_(False)
            optimizer = None
        else:
            clone.train()
            optimizer = make_optimizer(clone, optimizer_config)
        result[name] = dict(model=clone, optimizer=optimizer, tensor_sha256=tensor_hash(clone))
    return result


class Trainer:
    def __init__(self, model, optimizer, streams, terminal, config=None, ledger=None):
        self.model, self.optimizer, self.streams, self.terminal = model, optimizer, streams, terminal
        if optimizer is not None:
            optimizer_parameters = [id(p) for group in optimizer.param_groups for p in group['params']]
            if len(optimizer_parameters) != len(set(optimizer_parameters)) or set(optimizer_parameters) != {id(p) for p in model.parameters()}:
                raise ValueError('Optimizer must own exactly this model, without another branch or duplicate parameters')
            if any(not p.requires_grad for p in model.parameters()):
                raise ValueError('Frozen or partially frozen model cannot enter this bound all-parameter training recipe')
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.config.update(copy.deepcopy(config or {}))
        if float(self.config['gamma']) != 1. or self.config['entropy_weight'] != 0. or self.config['advantage_standardization']:
            raise ValueError('This implementation binds undiscounted full returns without entropy/advantage scaling')
        if list(self.config['mixture_weights']) != [1., .25, .25]:
            raise ValueError('Mixed objective weights require explicit versioned implementation')
        self.config['batch_instances'] = integer(self.config['batch_instances'], 'macro batch size')
        if self.config['batch_instances'] < 1:
            raise ValueError('Positive macro batch size required')
        if not math.isfinite(float(self.config['clip_global_norm'])) or float(self.config['clip_global_norm']) <= 0:
            raise ValueError('Positive finite gradient clip required')
        if self.config.get('max_optimizer_updates') is not None:
            self.config['max_optimizer_updates'] = integer(self.config['max_optimizer_updates'], 'optimizer update budget')
            if self.config['max_optimizer_updates'] < 0:
                raise ValueError('Negative optimizer budget')
        self.ledger = ledger if ledger is not None else {}
        self.cursors = {'S1': 0, 'RL': 0, 'continued_SL': 0}
        self.weight_version = 0
        self.in_update = False
        self.stopped = False
        self.last_diagnostics = None

    def _inc(self, key, amount=1):
        self.ledger[key] = self.ledger.get(key, 0) + amount

    @property
    def phase(self):
        return str(self.config['phase'])

    def _terminal(self, view, raw):
        self._inc('trainer_terminal_requests')
        try:
            result = self.terminal(view, raw)
        except Exception:
            self._inc('trainer_terminal_failures')
            raise
        self._inc('trainer_terminal_completed')
        return result

    def _encode(self, view):
        self._inc('encode_calls')
        return self.model.encode(view)

    def _score(self, env, encoded):
        self._inc('score_calls')
        mask = env.mask()
        if not bool(mask.any()):
            raise InconclusiveBatch('Nonempty target state has no legal service action')
        logits = self.model.score(env, encoded)
        legal = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
        if not torch.isfinite(logits[legal]).all() or not torch.isneginf(logits[~legal]).all():
            raise FloatingPointError('Nonfinite legal logits or leaked illegal action probability')
        return logits, legal

    def teacher_forcing_loss(self, view, actions):
        """Per-route mean masked CE; label-state replay adds zero oracle/H calls."""
        actions = [integer(x, 'teacher action') for x in actions]
        terminal_calls = [0]

        def label_terminal(p, raw):
            terminal_calls[0] += 1
            self._inc('supervised_label_terminal_replays')
            return p.metrics(raw)

        env = Environment(view, label_terminal)
        if env.done:
            if actions:
                raise ValueError('Empty target has nonempty teacher labels')
            return torch.zeros((), device=next(self.model.parameters()).device), dict(
                problem=view.identity, empty=True, decisions=0, loss=0., actions=[],
                label_replay_terminal_calls=terminal_calls[0], teacher_or_H_queries=0)
        encoded = self._encode(view)
        losses, legal_counts = [], []
        for node in actions:
            if env.done:
                raise ValueError('Teacher labels continue beyond complete coverage')
            logits, mask = self._score(env, encoded)
            if not 1 <= node <= len(mask) or not bool(mask[node - 1]):
                raise ValueError('Teacher label is illegal under current exact mask')
            losses.append(-torch.log_softmax(logits, 0)[node - 1])
            legal_counts.append(int(mask.sum()))
            env.step(node)
            self._inc('supervised_label_actions')
        if not env.done:
            raise ValueError('Teacher labels stop before full coverage')
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite supervised loss')
        return loss, dict(problem=view.identity, empty=False, decisions=len(actions),
                          loss=float(loss.detach()), actions=actions, legal_counts=legal_counts,
                          label_replay_terminal_calls=terminal_calls[0], teacher_or_H_queries=0,
                          label_route_raw_J=float(env.result['J']))

    def policy_instance_loss(self, view, update, slot):
        """Three independent rows, separate mutable prefix caches, detached LOO."""
        rows, log_sums = [], []
        weights = self.config['mixture_weights']
        device = next(self.model.parameters()).device
        for role in ('primary', 'forced_low', 'forced_high'):
            env = Environment(view, self._terminal)
            logps, entropies, legal_counts, actions, masks = [], [], [], [], []
            generator = self.streams.torch(self.phase, update, slot, role, device=str(device))
            forced = None
            if not env.done:
                legal = np.flatnonzero(env.mask()) + 1
                if not len(legal):
                    raise InconclusiveBatch('Nonempty initial state has no legal actions')
                unique_first = len(legal)
                if role != 'primary':
                    forced = int(legal[0] if role == 'forced_low' else legal[-1])
                encoded = self._encode(view)
                while not env.done:
                    mask_np = env.mask()
                    legal_counts.append(int(mask_np.sum()))
                    masks.append((np.flatnonzero(mask_np) + 1).tolist())
                    if forced is not None and not env.prefix:
                        node, lp, entropy = forced, None, None
                    else:
                        logits, mask = self._score(env, encoded)
                        logs = torch.log_softmax(logits, 0)
                        probs = torch.softmax(logits, 0)
                        chosen = torch.multinomial(probs, 1, generator=generator).squeeze(0)
                        node, lp = int(chosen) + 1, logs[chosen]
                        entropy = -(probs[mask] * logs[mask]).sum()
                    env.step(node)
                    actions.append(node)
                    logps.append(lp)
                    entropies.append(entropy)
                    self._inc('environment_actions')
                    if lp is None:
                        self._inc('forced_first_actions')
            else:
                unique_first = 0
            self._inc('policy_trajectories')
            logsum = sum((x for x in logps if x is not None), torch.zeros((), device=device))
            cost, return_ = float(env.result['J']), float(sum(env.rewards))
            if not math.isfinite(cost) or not math.isfinite(return_) or abs(return_ + cost) > 1e-7:
                raise FloatingPointError('Reward sum differs from negative complete terminal J')
            if actions and role == 'primary' and logps[0] is None:
                raise AssertionError('Primary first action probability omitted')
            if actions and role != 'primary' and logps[0] is not None:
                raise AssertionError('Forced first action incorrectly scored')
            rows.append(dict(role=role, sampling_seed=self.streams.seed(self.phase, update, slot, role),
                forced_first=forced, unique_legal_first=unique_first, actions=actions,
                legal_counts=legal_counts, masks_local_nodes=masks,
                logprobs=[None if x is None else float(x.detach()) for x in logps],
                logprob_sum=float(logsum.detach()), entropies=[None if x is None else float(x.detach()) for x in entropies],
                rewards=[float(x) for x in env.rewards], return_=return_, J=cost,
                raw_J=float(env.result.get('raw_J', cost)),
                raw_route=[0, *actions, 0], final_route=list(env.result.get('route', [0, *actions, 0])),
                terminal_cache_hit=bool(env.result.get('cache_hit', False)),
                H_s=float(env.result.get('H_s', 0.)),
                export_s=float(env.result.get('export_s', 0.)),
                validation_s=float(env.result.get('validation_s', 0.))))
            log_sums.append(logsum)
        returns = torch.tensor([r['return_'] for r in rows], dtype=torch.float64, device=device)
        baselines = ((returns.sum() - returns) / 2).detach()
        advantages = (returns - baselines).detach()
        loss = sum(-float(w) * a * logp for w, a, logp in zip(weights, advantages, log_sums)) / sum(weights)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite policy-gradient objective')
        empty = not len(view.C)
        return loss, dict(problem=view.identity, parent=view.parent.identity,
            update=int(update), slot=int(slot), empty=empty, weight_version=self.weight_version,
            rows=rows, baseline=baselines.cpu().tolist(), advantages=advantages.cpu().tolist(),
            mixture_weights=list(weights), loss=float(loss.detach()),
            gamma=1., objective='complete -JH mixed REINFORCE')

    def _batch_size(self, items):
        if len(items) != int(self.config['batch_instances']):
            raise ValueError('Actual macro batch size differs from bound configuration')

    def _backward(self, loss):
        if loss.requires_grad:
            self._inc('backward_calls')
            try:
                loss.backward()
            except Exception:
                self._inc('backward_failures')
                raise
            self._inc('backward_completed')

    def _gradient_diagnostics(self):
        norms = {name: 0. if p.grad is None else float(p.grad.detach().norm()) for name, p in self.model.named_parameters()}
        if not all(math.isfinite(v) for v in norms.values()):
            raise FloatingPointError('Nonfinite gradient')
        return norms, math.sqrt(sum(v * v for v in norms.values()))

    def batch_gradients(self, views, update, mode='accumulate'):
        """Zero-update RL objective, with either macro joint or micro accumulation."""
        views = list(views)
        self._batch_size(views)
        if mode not in ('accumulate', 'joint'):
            raise ValueError('Unknown gradient execution mode')
        self.model.zero_grad(set_to_none=True)
        rows, losses = [], []
        for slot, view in enumerate(views):
            loss, diagnostics = self.policy_instance_loss(view, update, slot)
            rows.append(diagnostics)
            if mode == 'accumulate':
                self._backward(loss / len(views))
            else:
                losses.append(loss)
        if mode == 'joint':
            self._backward(sum(losses) / len(views))
        norms, norm = self._gradient_diagnostics()
        return dict(phase=self.phase, update=int(update), mode=mode,
                    batch_instances=len(views), instances=rows,
                    loss=sum(r['loss'] for r in rows) / len(rows),
                    gradient_norms=norms, gradient_norm=norm,
                    empty_instances=sum(r['empty'] for r in rows),
                    zero_signal=all(all(a == 0. for a in r['advantages']) for r in rows))

    def _before_update(self):
        if self.in_update or self.stopped:
            raise RuntimeError('Trainer is busy or stopped after an incomplete update')
        if self.optimizer is None:
            raise RuntimeError('Frozen model cannot be optimized')
        cap = self.config.get('max_optimizer_updates')
        if cap is not None and self.cursors.get(self.phase, 0) >= int(cap):
            raise RuntimeError('Bound phase optimizer budget exhausted')
        self.in_update = True
        self._inc('optimizer_attempts')

    def _finish_update(self, diagnostics):
        norms, before = self._gradient_diagnostics()
        if not any(not r['empty'] for r in diagnostics['instances']):
            self._inc('all_empty_batch_stops')
            raise InconclusiveBatch('All-empty macro batch; no optimizer update or sample replacement')
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config['clip_global_norm']), error_if_nonfinite=True)
        _, after = self._gradient_diagnostics()
        self._inc('optimizer_step_calls')
        try:
            self.optimizer.step()
        except Exception:
            self._inc('optimizer_step_failures')
            raise
        self._inc('optimizer_updates')
        self.weight_version += 1
        self.cursors[self.phase] = self.cursors.get(self.phase, 0) + 1
        for parameter in self.model.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError('Optimizer produced nonfinite parameters; stop without fallback')
        diagnostics.update(optimizer_updated=True, gradient_norm_before_clip=before,
                           gradient_norm_after_clip=after, gradient_norms=norms,
                           weight_version_after=self.weight_version,
                           model_tensor_sha256=tensor_hash(self.model), cursor_after=self.cursors[self.phase])
        self.last_diagnostics = diagnostics
        self.in_update = False
        return diagnostics

    def s2_step(self, views, update=None):
        update = self.cursors.get(self.phase, 0) if update is None else int(update)
        if update != self.cursors.get(self.phase, 0):
            raise ValueError('Update index differs from persisted phase cursor')
        self._before_update()
        tick = time.perf_counter()
        try:
            diagnostics = self.batch_gradients(views, update, mode='accumulate')
            diagnostics['elapsed_s_before_optimizer'] = time.perf_counter() - tick
            result = self._finish_update(diagnostics)
            result['elapsed_s'] = time.perf_counter() - tick
            return result
        except Exception:
            self.stopped = True
            self.in_update = False
            self._inc('incomplete_update_failures')
            raise

    def s1_step(self, examples, update=None):
        examples = list(examples)
        self._batch_size(examples)
        update = self.cursors.get(self.phase, 0) if update is None else int(update)
        if update != self.cursors.get(self.phase, 0):
            raise ValueError('Update index differs from persisted phase cursor')
        self._before_update()
        tick = time.perf_counter()
        try:
            self.model.zero_grad(set_to_none=True)
            rows = []
            for example in examples:
                view, actions = (example['view'], example['actions']) if isinstance(example, dict) else example
                loss, diagnostics = self.teacher_forcing_loss(view, actions)
                rows.append(diagnostics)
                self._backward(loss / len(examples))
            diagnostics = dict(phase=self.phase, update=update, instances=rows,
                               batch_instances=len(examples), objective='route-mean masked teacher CE',
                               loss=sum(r['loss'] for r in rows) / len(rows),
                               empty_instances=sum(r['empty'] for r in rows), teacher_or_H_queries=0)
            result = self._finish_update(diagnostics)
            result['elapsed_s'] = time.perf_counter() - tick
            return result
        except Exception:
            self.stopped = True
            self.in_update = False
            self._inc('incomplete_update_failures')
            raise
