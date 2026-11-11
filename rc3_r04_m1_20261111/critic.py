"""R04-M1 production Critic units and provenance interfaces.

Only C0/C1 differ in output units. This module performs no top-level computation,
planner request, Actor construction, data admission, or optimizer update. The
registered caller owns all resource charges and file reads/writes. Checkpoint
restore validates exact units, provenance, cursor, budget and Adam before loading.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Mapping

import numpy as np
import torch
from torch import nn


INPUT_DIM = 268
HIDDEN_DIM = 128
SEED = 11622026
PENALTY = 0.75
VARIANTS = ('C0', 'C1')
COMPONENTS = ('raw_remaining_L', 'raw_remaining_K', 'terminal_H_credit',
              'normalized_L', 'normalized_K', 'normalized_H', 'value')
NORMALIZED_COMPONENTS = ('normalized_L', 'normalized_K', 'normalized_H')
REDUCTION = 'state_SUM_per_trajectory_then_episode_MEAN'
SEMANTICS = {'C0': 'raw_L_K_signed_H_v1', 'C1': 'per_fixed_initial_n_L_K_signed_H_v1'}
CHECKPOINT_SCHEMA = 'rc3-r04-m1-critic-training-checkpoint-v1'
EXPORT_SCHEMA = 'rc3-r04-m1-critic-calibration-export-v1'
IDENTITY_FIELDS = frozenset(('source_sha256', 'config_sha256', 'data_sha256',
                             'schedule_sha256', 'protocol_sha256', 'authorization_sha256'))
ADAM = dict(lr=0.0001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.,
            amsgrad=False, maximize=False)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _scale(count):
    if type(count) is not int or count <= 0:
        raise ValueError('Use the fixed positive initial task count, never remaining count')
    return count


def _variant(value):
    if value not in VARIANTS:
        raise ValueError('Only the registered C0/C1 output units are supported')
    return value


def _identity(value):
    if not isinstance(value, Mapping) or not IDENTITY_FIELDS.issubset(value):
        raise ValueError('Checkpoint needs source/config/data/schedule/protocol/authorization hashes')
    if any(not isinstance(k, str) or not k or not isinstance(v, str) or len(v) != 64
           or any(c not in '0123456789abcdef' for c in v) for k, v in value.items()):
        raise ValueError('Every identity entry must be an explicit lowercase SHA256')
    return dict(sorted(value.items()))


def _resources(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError('A nonempty cumulative resource snapshot is required')
    result = copy.deepcopy(dict(value))
    _canonical(result)
    if 'critic_optimizer_attempts' not in result or 'actor_optimizer_attempts' not in result:
        raise ValueError('Explicit cumulative Actor and Critic attempt counts required')
    if (type(result['critic_optimizer_attempts']) is not int
            or not 0 <= result['critic_optimizer_attempts'] <= 512
            or type(result['actor_optimizer_attempts']) is not int
            or result['actor_optimizer_attempts'] != 0):
        raise ValueError('Offline M1 permits at most512 Critic attempts and zero Actor attempts')
    return result


def tensor_digest(state):
    """Byte identity, independent of a torch.save container's serialization."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError('A named tensor state is required')
        value = tensor.detach().cpu().contiguous()
        digest.update(_canonical([name, str(value.dtype), list(value.shape)]))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def tree_digest(value):
    """Structural identity for tensors, Adam/RNG state and plain metadata."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            digest.update(b'tensor');digest.update(tensor_digest({'value': item}).encode())
        elif isinstance(item, np.ndarray):
            digest.update(b'ndarray');digest.update(str(item.dtype).encode())
            digest.update(_canonical(list(item.shape)));digest.update(item.tobytes())
        elif isinstance(item, Mapping):
            digest.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key);visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode());digest.update(str(len(item)).encode())
            for entry in item: visit(entry)
        else:
            digest.update(type(item).__name__.encode());digest.update(_canonical(item))
    visit(value)
    return digest.hexdigest()


@dataclass(frozen=True)
class ComponentPrediction:
    """Three net component tensors with an explicit interpretation.

    Raw H is terminal whole-path cost credit, including changes to an already
    travelled prefix; signed H/n is not physical remaining distance. The value
    is always negative complete RTG cost per fixed initial task count.
    """
    variant: str
    initial_task_count: int
    length: torch.Tensor
    stops: torch.Tensor
    h_credit: torch.Tensor

    def validate(self):
        _variant(self.variant);_scale(self.initial_task_count)
        values = self.length, self.stops, self.h_credit
        if any(not isinstance(v, torch.Tensor) or not v.is_floating_point()
               or v.ndim != 1 or not v.numel() for v in values):
            raise ValueError('Nonempty T-shaped floating components required')
        if any(v.shape != values[0].shape or v.dtype != values[0].dtype
               or v.device != values[0].device for v in values):
            raise ValueError('Component shape/dtype/device must match')
        if any(not bool(torch.isfinite(v).all()) for v in values):
            raise ValueError('Critic predictions must be finite')
        if bool((self.length < 0).any()) or bool((self.stops < 0).any()):
            raise ValueError('L/K are nonnegative; H is signed')
        return self

    @property
    def normalized_L(self):
        return self.length / self.initial_task_count if self.variant == 'C0' else self.length

    @property
    def normalized_K(self):
        return self.stops / self.initial_task_count if self.variant == 'C0' else self.stops

    @property
    def normalized_H(self):
        return self.h_credit / self.initial_task_count if self.variant == 'C0' else self.h_credit

    @property
    def raw_remaining_L(self):
        return self.length if self.variant == 'C0' else self.length * self.initial_task_count

    @property
    def raw_remaining_K(self):
        return self.stops if self.variant == 'C0' else self.stops * self.initial_task_count

    @property
    def terminal_H_credit(self):
        return self.h_credit if self.variant == 'C0' else self.h_credit * self.initial_task_count

    @property
    def value(self):
        composed = -(self.length + PENALTY * self.stops - self.h_credit)
        return composed / self.initial_task_count if self.variant == 'C0' else composed

    def as_dict(self):
        self.validate()
        return {name: getattr(self, name) for name in COMPONENTS}

    def as_training_dict(self):
        """Raw adapter: old component loss applies its one division by n."""
        self.validate()
        return {name: getattr(self, name) for name in COMPONENTS[:3] + ('value',)}

    def metadata(self):
        return dict(schema='rc3-r04-m1-typed-components-v1', variant=self.variant,
            output_semantics=SEMANTICS[self.variant], initial_task_count=self.initial_task_count,
            normalized_units=['distance/initial_task', 'stops/initial_task',
                              'signed_cost_credit/initial_task'],
            value_unit='negative_cost/initial_task', penalty=PENALTY,
            H_includes_full_path_prefix_revision=True, H_is_remaining_distance=False)


class MatchedCritic(nn.Module):
    """Exact 268-128-128-3 MLP, with C0/C1 unit metadata outside its tensors."""
    def __init__(self, variant):
        super().__init__()
        self.variant = _variant(variant)
        self.input_dim = INPUT_DIM
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 3))

    def forward(self, features, task_count, penalty=PENALTY):
        _scale(task_count)
        if isinstance(penalty, bool) or penalty != PENALTY:
            raise ValueError('Historical stop penalty0.75 is fixed')
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape[1] != INPUT_DIM or not len(features)
                or not features.is_floating_point() or not bool(torch.isfinite(features).all())):
            raise ValueError('Finite nonempty T x 268 floating saved features required')
        output = self.net(features.detach())
        return ComponentPrediction(self.variant, task_count,
            torch.nn.functional.softplus(output[:, 0]),
            torch.nn.functional.softplus(output[:, 1]), output[:, 2]).validate()


def make_matched_models(seed=SEED):
    if type(seed) is not int or seed != SEED:
        raise ValueError('M1 seed11622026 is fixed; no additional seeds')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        initial = MatchedCritic('C0')
    models = {name: copy.deepcopy(initial) for name in VARIANTS}
    for name, model in models.items(): model.variant = name
    return models


def episode_features(episode):
    source = episode['v'] if 'v' in episode else episode['batch']['v']
    result = torch.as_tensor(source, dtype=torch.float32).detach()
    if (result.ndim != 2 or result.shape[1] != INPUT_DIM or not len(result)
            or not bool(torch.isfinite(result).all())):
        raise ValueError('Saved finite nonempty T x 268 input required')
    if not bool(((result[:, 256] >= -1e-6) & (result[:, 256] <= 1 + 1e-6)).all()):
        raise ValueError('Observed remaining fraction outside [0,1]')
    return result


def episode_labels(episode, *, device=None, dtype=torch.float32):
    n = _scale(episode['task_count'])
    targets = episode['targets']
    if targets.get('scale', n) != n or targets.get('task_count', n) != n:
        raise ValueError('Saved labels use a different fixed initial task count')
    returns = torch.as_tensor(targets['returns'], device=device, dtype=dtype).detach()
    if returns.ndim != 1 or not len(returns) or not bool(torch.isfinite(returns).all()):
        raise ValueError('Finite nonempty complete RTG labels required')
    result = {'value': returns}
    for name in COMPONENTS[:3]:
        source = targets.get(name)
        if name == 'terminal_H_credit' and source is None and 'H_gain' in targets:
            source = [targets['H_gain']] * len(returns)
        if source is None:
            raise ValueError('Missing saved complete labels; producing H labels is forbidden')
        value = torch.as_tensor(source, device=device, dtype=dtype).detach()
        if value.shape != returns.shape or not bool(torch.isfinite(value).all()):
            raise ValueError('Complete saved component and RTG dimensions differ')
        if name != 'terminal_H_credit' and bool((value < 0).any()):
            raise ValueError('Saved L/K must be nonnegative')
        result[name] = value
    for raw, normalized in zip(COMPONENTS[:3], NORMALIZED_COMPONENTS):
        result[normalized] = result[raw] / n
    composed = -(result['normalized_L'] + PENALTY * result['normalized_K'] - result['normalized_H'])
    if not torch.allclose(composed, returns, atol=2e-5, rtol=2e-5):
        raise ValueError('Saved full RTG and component accounting differ')
    return result


def component_state_loss(prediction, episode):
    if not isinstance(prediction, ComponentPrediction):
        raise TypeError('Use explicitly typed C0/C1 predictions; arbitrary dictionaries forbidden')
    prediction.validate()
    if prediction.initial_task_count != episode['task_count']:
        raise ValueError('Prediction uses another trajectory initial task count')
    labels = episode_labels(episode, device=prediction.length.device, dtype=prediction.length.dtype)
    if prediction.length.shape != labels['value'].shape:
        raise ValueError('Complete saved state/label count differs')
    if prediction.variant == 'C0':
        # Preserve original arithmetic and unit interpretation, including order.
        errors = [(getattr(prediction, key) - labels[key]) / prediction.initial_task_count
                  for key in COMPONENTS[:3]]
    else:
        errors = [getattr(prediction, key) - labels[key] for key in NORMALIZED_COMPONENTS]
    state_loss = .5 * torch.stack([value.square() for value in errors]).sum(0)
    if not bool(torch.isfinite(state_loss).all()):
        raise RuntimeError('Nonfinite component loss')
    return state_loss


def matched_loss(model, episodes):
    """Exactly state SUM per trajectory then episode MEAN; no value loss."""
    if not isinstance(model, MatchedCritic) or not episodes:
        raise ValueError('M1 model and a nonempty admitted fitting batch required')
    losses = []
    for episode in episodes:
        if episode.get('split') != 'fit':
            raise ValueError('Only training-fold episodes may enter an optimizer loss')
        features = episode_features(episode)
        prediction = model(features, episode['task_count'])
        losses.append(component_state_loss(prediction, episode).sum())
    return torch.stack(losses).mean()


@torch.no_grad()
def predict_episode(model, episode):
    return {key: value.detach().cpu() for key, value in
            model(episode_features(episode), episode['task_count']).as_dict().items()}


def _description(model):
    if not isinstance(model, MatchedCritic):
        raise TypeError('Explicit M1 Critic required')
    return dict(variant=model.variant, output_semantics=SEMANTICS[model.variant],
        network=[INPUT_DIM, HIDDEN_DIM, HIDDEN_DIM, 3], penalty=PENALTY,
        normalization='fixed_positive_initial_task_count_provided_per_trajectory',
        value_unit='negative_cost/initial_task', reduction=REDUCTION,
        component_coefficients=[0.5, 0.5, 0.5], extra_value_loss=0,
        model_role='training_only_critic_not_actor', features_schema='frozen_saved_268_v1')


def _check_adam(optimizer, model):
    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
        raise ValueError('Exactly one independent Adam group required')
    group = optimizer.param_groups[0]
    if any(group.get(key) != value for key, value in ADAM.items()):
        raise ValueError('Adam configuration differs from the registered M1 constants')
    if [id(p) for p in group['params']] != [id(p) for p in model.parameters()]:
        raise ValueError('Adam parameters do not belong exactly to this Critic')


def _check_cursor(cursor):
    if type(cursor) is not int or not 0 <= cursor <= 256:
        raise ValueError('Completed per-configuration cursor must be in0..256')
    return cursor


def _check_optimizer_state(state, model_state, cursor):
    if not isinstance(state, dict) or set(state) != {'state', 'param_groups'}:
        raise ValueError('Malformed Adam state')
    if len(state['param_groups']) != 1:
        raise ValueError('Adam checkpoint has wrong group count')
    group = state['param_groups'][0]
    if any(group.get(key) != value for key, value in ADAM.items()):
        raise ValueError('Saved Adam settings differ from M1')
    params = list(model_state.values())
    ids = group['params']
    if len(ids) != len(params) or len(set(ids)) != len(ids):
        raise ValueError('Saved Adam parameter identity count differs')
    if cursor == 0:
        if state['state']:
            raise ValueError('Step0 cannot contain already consumed Adam moments')
        return
    if set(state['state']) != set(ids):
        raise ValueError('Every trained Critic parameter needs exact Adam state')
    for identifier, parameter in zip(ids, params):
        entry = state['state'][identifier]
        if set(entry) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Saved Adam schema differs')
        step = entry['step']
        if (not isinstance(step, torch.Tensor) or step.numel() != 1 or step.device.type != 'cpu'
                or step.dtype != torch.float32
                or not bool(torch.isfinite(step).all()) or float(step) != cursor):
            raise ValueError('Adam step differs from registered completed cursor')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = entry[key]
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or value.shape != parameter.shape
                    or value.dtype != parameter.dtype or not bool(torch.isfinite(value).all())):
                raise ValueError('Adam moment tensor shape/dtype/finiteness differs')
        if bool((entry['exp_avg_sq'] < 0).any()):
            raise ValueError('Adam second moment must be nonnegative')


def capture_rng():
    # M1 is CPU only. GPU RNG is neither initialized nor silently omitted.
    return dict(torch_cpu=torch.get_rng_state().clone(), python=random.getstate(),
                numpy=copy.deepcopy(np.random.get_state()), device='cpu')


def restore_rng(rng):
    if not isinstance(rng, dict) or set(rng) != {'torch_cpu', 'python', 'numpy', 'device'} or rng['device'] != 'cpu':
        raise ValueError('Exact CPU torch/Python/NumPy RNG state required')
    # Validate using private generators before modifying global state.
    torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    torch.set_rng_state(rng['torch_cpu']);random.setstate(rng['python']);np.random.set_state(rng['numpy'])


def make_checkpoint(model, optimizer, *, identity, cursor, resources):
    """Package only; caller charges and writes the actual checkpoint."""
    _check_adam(optimizer, model);_check_cursor(cursor)
    state = copy.deepcopy(model.state_dict());adam = copy.deepcopy(optimizer.state_dict())
    _check_optimizer_state(adam, state, cursor)
    resource = _resources(resources)
    if resource['critic_optimizer_attempts'] < cursor:
        raise ValueError('Resource journal cannot undercount consumed updates')
    value = dict(schema=CHECKPOINT_SCHEMA, model_description=_description(model),
        identity=_identity(identity), model=state, optimizer=adam, rng=capture_rng(),
        cursor=cursor, resources=resource, device='cpu',
        initial_task_count_binding='RESTORED_INPUT_INDEX.json task_count and M1_DATA_AND_SOURCE_IDENTITY.json source records')
    value['payload_sha256'] = tree_digest(value)
    return value


def restore_checkpoint(package, model, optimizer, *, expected_identity,
                       expected_cursor, expected_resources):
    """Exact restore, no optimizer attempt or RNG advancement.

    The caller must compare expected_cursor/resources to its durable journal;
    this interface cannot bless a stale package or reset an external budget.
    All validations occur before loading any model/Adam/RNG state.
    """
    fields = {'schema', 'model_description', 'identity', 'model', 'optimizer', 'rng',
              'cursor', 'resources', 'device', 'initial_task_count_binding', 'payload_sha256'}
    if not isinstance(package, dict) or set(package) != fields or package['schema'] != CHECKPOINT_SCHEMA:
        raise ValueError('Unknown/legacy/Actor checkpoint schema is not admitted')
    if tree_digest({k: v for k, v in package.items() if k != 'payload_sha256'}) != package['payload_sha256']:
        raise ValueError('Checkpoint payload integrity mismatch')
    if package['model_description'] != _description(model):
        raise ValueError('C0/C1 units, architecture, loss or role mismatch')
    if package['identity'] != _identity(expected_identity):
        raise ValueError('Checkpoint source/config/data/schedule/protocol/authorization mismatch')
    _check_cursor(expected_cursor);_check_adam(optimizer, model)
    if package['cursor'] != expected_cursor or package['resources'] != _resources(expected_resources):
        raise ValueError('Checkpoint cursor/resource differs from admitted cumulative journal')
    if package['device'] != 'cpu' or package['initial_task_count_binding'] != 'RESTORED_INPUT_INDEX.json task_count and M1_DATA_AND_SOURCE_IDENTITY.json source records':
        raise ValueError('Unknown device or initial-count provenance')
    current = model.state_dict()
    if list(package['model']) != list(current):
        raise ValueError('Checkpoint parameter names/order differ')
    for name, value in package['model'].items():
        if (not isinstance(value, torch.Tensor) or value.shape != current[name].shape
                or value.dtype != current[name].dtype or value.device.type != 'cpu'
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Checkpoint model parameter shape/dtype/finiteness differs')
    _check_optimizer_state(package['optimizer'], package['model'], expected_cursor)
    # Validate RNG without changing globals, then perform all load operations.
    rng = package['rng']
    if not isinstance(rng, dict) or set(rng) != {'torch_cpu', 'python', 'numpy', 'device'} or rng['device'] != 'cpu':
        raise ValueError('Unknown RNG schema')
    torch.Generator(device='cpu').set_state(rng['torch_cpu'])
    random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
    model.load_state_dict(package['model'], strict=True)
    optimizer.load_state_dict(package['optimizer'])
    restore_rng(rng)
    if tensor_digest(model.state_dict()) != tensor_digest(package['model']):
        raise RuntimeError('Exact Critic weight restore failed')
    if tree_digest(optimizer.state_dict()) != tree_digest(package['optimizer']):
        raise RuntimeError('Exact Adam restore failed')
    return dict(cursor=expected_cursor, resources=copy.deepcopy(package['resources']),
                output_semantics=SEMANTICS[model.variant], exact_restore=True)


def make_export(model, *, identity, cursor):
    """Calibration-only Critic package; never an Actor deployment artifact."""
    _check_cursor(cursor)
    value = dict(schema=EXPORT_SCHEMA, model_description=_description(model),
        identity=_identity(identity), cursor=cursor, model=copy.deepcopy(model.state_dict()),
        device='cpu', usage='offline_calibration_critic_only_not_actor_deployment')
    value['payload_sha256'] = tree_digest(value)
    return value


def load_export(package, *, expected_variant, expected_identity, expected_cursor):
    fields = {'schema', 'model_description', 'identity', 'cursor', 'model', 'device', 'usage', 'payload_sha256'}
    if not isinstance(package, dict) or set(package) != fields or package['schema'] != EXPORT_SCHEMA:
        raise ValueError('Only strict optimizer-free Critic calibration exports admitted')
    if tree_digest({k: v for k, v in package.items() if k != 'payload_sha256'}) != package['payload_sha256']:
        raise ValueError('Critic export integrity mismatch')
    # Preserve caller RNG: constructing the load target must not affect schedule.
    with torch.random.fork_rng(devices=[]):
        model = MatchedCritic(_variant(expected_variant))
    if package['model_description'] != _description(model):
        raise ValueError('Export C0/C1 output semantics or architecture differ')
    if package['identity'] != _identity(expected_identity) or package['cursor'] != _check_cursor(expected_cursor):
        raise ValueError('Export source/cursor differs')
    if package['device'] != 'cpu' or package['usage'] != 'offline_calibration_critic_only_not_actor_deployment':
        raise ValueError('Export cannot serve as an Actor deployment package')
    state = model.state_dict()
    if list(package['model']) != list(state):
        raise ValueError('Export model keys/order differ')
    for key, value in package['model'].items():
        if (not isinstance(value, torch.Tensor) or value.shape != state[key].shape
                or value.dtype != state[key].dtype or value.device.type != 'cpu'
                or not bool(torch.isfinite(value).all())):
            raise ValueError('Export tensor shape/dtype/finiteness differs')
    model.load_state_dict(package['model'], strict=True)
    return model.eval().requires_grad_(False)
