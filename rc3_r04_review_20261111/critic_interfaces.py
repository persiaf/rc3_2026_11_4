"""R04 candidate Critic interfaces; no optimizer, planner, data or model I/O.

The normalized C1 prediction is deliberately a typed object. Historical PPO
code expecting a raw tensor dictionary must explicitly request
``as_training_dict()`` rather than accidentally dividing normalized outputs by
the task count a second time. This module does not enable that old trainer.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import torch
from torch import nn


INPUT_DIM = 268
PENALTY = 0.75
C1_SCHEMA = 'rc3-r04-c1-per-initial-n-v1'
BASELINE_SCHEMA = 'rc3-r04-fit-only-conditional-components-v1'
COMPONENTS = ('normalized_L', 'normalized_K', 'normalized_H')
OBSERVATION_FIELDS = frozenset(('window_width', 'window_height', 'remaining_fraction'))


def initial_scale(initial_task_count):
    if type(initial_task_count) is not int or initial_task_count < 0:
        raise ValueError('Use the fixed nonnegative initial task count, not a remaining count')
    return max(1, initial_task_count)


def _finite_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + ' must be a finite real scalar')
    return float(value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class C1Prediction:
    """Normalized predictions with explicit reversible raw-unit adapter.

    L/n is distance per initial task, K/n is stops per initial task, H/n is
    signed cost credit per initial task. ``value`` is negative cost per initial
    task. Neither H credit nor value is a physical remaining-distance claim.
    """
    initial_task_count: int
    normalized_L: torch.Tensor
    normalized_K: torch.Tensor
    normalized_H: torch.Tensor

    @property
    def schema(self):
        return C1_SCHEMA

    @property
    def scale(self):
        return initial_scale(self.initial_task_count)

    def validate(self):
        values = (self.normalized_L, self.normalized_K, self.normalized_H)
        self.scale
        if any(not isinstance(x, torch.Tensor) or not x.is_floating_point() for x in values):
            raise ValueError('Normalized components must be floating tensors')
        if any(x.ndim != 1 or x.shape != values[0].shape or x.device != values[0].device
               or x.dtype != values[0].dtype for x in values):
            raise ValueError('All components must have matching T shape, dtype and device')
        if any(not bool(torch.isfinite(x).all()) for x in values):
            raise ValueError('Normalized components must be finite')
        if bool((self.normalized_L < 0).any()) or bool((self.normalized_K < 0).any()):
            raise ValueError('Predicted L/K must remain nonnegative; no silent clipping')
        if self.initial_task_count == 0 and self.normalized_L.numel():
            raise ValueError('Empty tasks have no action states; use a T=0 fixture')
        return self

    @property
    def value(self):
        return -(self.normalized_L + PENALTY * self.normalized_K - self.normalized_H)

    def as_training_dict(self):
        """Only tensor fields expected by the historical component loss.

        Raw values are reconstructed BEFORE that loss's single division by n.
        Metadata must be logged separately, not passed into a tensor loop.
        """
        self.validate()
        return dict(raw_remaining_L=self.normalized_L * self.scale,
                    raw_remaining_K=self.normalized_K * self.scale,
                    terminal_H_credit=self.normalized_H * self.scale,
                    value=self.value)

    def metadata(self):
        return dict(schema=self.schema, initial_task_count=self.initial_task_count,
                    scale=self.scale, input_dim=INPUT_DIM,
                    normalized_L_unit='distance / initial_task',
                    normalized_K_unit='stops / initial_task',
                    normalized_H_unit='signed_cost_credit / initial_task',
                    value_unit='negative_cost / initial_task',
                    penalty=PENALTY, H_is_physical_remaining_distance=False)


class C1Critic(nn.Module):
    """Untrained 268-128-128-3 candidate with normalized output semantics."""
    def __init__(self, input_dim=INPUT_DIM, hidden=128):
        super().__init__()
        if input_dim != INPUT_DIM or type(input_dim) is not int:
            raise ValueError('R04 retains exactly 268 input coordinates')
        if type(hidden) is not int or hidden <= 0:
            raise ValueError('Positive integer hidden width required')
        self.input_dim = input_dim
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))

    def forward(self, features, initial_task_count, penalty=PENALTY):
        initial_scale(initial_task_count)
        if penalty != PENALTY or isinstance(penalty, bool):
            raise ValueError('The historical 0.75 stop penalty is fixed')
        if (not isinstance(features, torch.Tensor) or features.ndim != 2
                or features.shape[1] != INPUT_DIM or not features.is_floating_point()
                or not bool(torch.isfinite(features).all())):
            raise ValueError('Finite T x 268 floating input required')
        if initial_task_count == 0 and len(features):
            raise ValueError('No policy states are valid for an empty task')
        output = self.net(features.detach())
        return C1Prediction(initial_task_count,
                            torch.nn.functional.softplus(output[:, 0]),
                            torch.nn.functional.softplus(output[:, 1]),
                            output[:, 2]).validate()


def normalized_component_loss(prediction, targets):
    """Per-state C1 component loss with no additional value-loss term.

    This intentionally leaves state/trajectory/batch reduction to a separately
    registered caller. It does not silently replace old PPO state SUM with the
    last offline study's state MEAN.
    """
    if not isinstance(prediction, C1Prediction):
        raise TypeError('Use an explicitly typed C1 prediction')
    prediction.validate()
    if targets.get('scale', prediction.scale) != prediction.scale:
        raise ValueError('Target scale differs from fixed initial n')
    if targets.get('task_count', prediction.initial_task_count) != prediction.initial_task_count:
        raise ValueError('Target initial task count differs')
    ref = prediction.normalized_L
    labels = []
    for key in ('raw_remaining_L', 'raw_remaining_K', 'terminal_H_credit'):
        source = targets.get(key)
        if key == 'terminal_H_credit' and source is None and 'H_gain' in targets:
            source = [targets['H_gain']] * ref.numel()
        if source is None:
            raise ValueError('Complete saved component labels are required')
        label = torch.as_tensor(source, device=ref.device, dtype=ref.dtype).detach()
        if label.shape != ref.shape or not bool(torch.isfinite(label).all()):
            raise ValueError('Finite complete component labels must match all states')
        if key != 'terminal_H_credit' and bool((label < 0).any()):
            raise ValueError('L/K targets must be nonnegative')
        labels.append(label / prediction.scale)
    returns = targets.get('returns')
    if returns is not None:
        expected = torch.as_tensor(returns, device=ref.device, dtype=ref.dtype).detach()
        composed = -(labels[0] + PENALTY * labels[1] - labels[2])
        if (expected.shape != ref.shape or not bool(torch.isfinite(expected).all())
                or not torch.allclose(expected, composed, atol=2e-5, rtol=2e-5)):
            raise ValueError('RTG differs from saved component accounting')
    return .5 * torch.stack([(estimate - label).square() for estimate, label in
                             zip((prediction.normalized_L, prediction.normalized_K,
                                  prediction.normalized_H), labels)]).sum(0)


@dataclass(frozen=True)
class ObservableCondition:
    window_width: int
    window_height: int
    remaining_fraction: float

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (self.window_width, self.window_height)):
            raise ValueError('Positive integer observable window dimensions required')
        fraction = _finite_scalar(self.remaining_fraction, 'remaining_fraction')
        if not -1e-6 <= fraction <= 1 + 1e-6:
            raise ValueError('Remaining fraction must be observable and in [0,1]')

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, Mapping) or set(value) != OBSERVATION_FIELDS:
            raise ValueError('Only window width/height and current remaining fraction may condition the baseline')
        return cls(**value)

    @classmethod
    def from_feature_row(cls, features):
        if (not isinstance(features, torch.Tensor) or features.ndim != 1
                or features.shape[0] != INPUT_DIM or not bool(torch.isfinite(features).all())):
            raise ValueError('Finite original 268-coordinate observation required')
        width, height = float(features[265]) * 10, float(features[266]) * 10
        if abs(width - round(width)) > 1e-4 or abs(height - round(height)) > 1e-4:
            raise ValueError('Original observable window coordinates are not integer dimensions')
        return cls(int(round(width)), int(round(height)), float(features[256]))

    def key(self):
        fraction = min(1., max(0., float(self.remaining_fraction)))
        progress = min(3, max(0, math.floor(4 * (1 - fraction))))
        return f'{self.window_width}x{self.window_height}|q{progress}'


@dataclass(frozen=True)
class BaselineTrainingRecord:
    physical_parent: str
    trajectory_id: str
    state_index: int
    split: str
    observation: ObservableCondition
    normalized_L: float
    normalized_K: float
    normalized_H: float

    def validate(self):
        if (not isinstance(self.physical_parent, str) or not self.physical_parent
                or not isinstance(self.trajectory_id, str) or not self.trajectory_id
                or type(self.state_index) is not int or self.state_index < 0):
            raise ValueError('Explicit physical-parent/trajectory/state identity required')
        if self.split != 'fit':
            raise ValueError('Validation, test and held-out labels cannot estimate baseline statistics')
        if not isinstance(self.observation, ObservableCondition):
            raise ValueError('Baseline condition must use the observable-only schema')
        values = tuple(_finite_scalar(getattr(self, k), k) for k in COMPONENTS)
        if values[0] < 0 or values[1] < 0:
            raise ValueError('Fitting normalized L/K labels must be nonnegative')
        return values


def _balanced_component_means(records):
    grouped = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[record.physical_parent][record.trajectory_id].append(record.validate())
    if not grouped:
        raise ValueError('No fitting records')
    parent_means = []
    for trajectories in grouped.values():
        trajectory_means = [tuple(math.fsum(x[c] for x in states) / len(states)
                                  for c in range(3)) for states in trajectories.values()]
        parent_means.append(tuple(math.fsum(x[c] for x in trajectory_means) / len(trajectory_means)
                                  for c in range(3)))
    means = tuple(math.fsum(x[c] for x in parent_means) / len(parent_means) for c in range(3))
    return dict(zip(COMPONENTS, means))


def _component_value(components):
    return -(components['normalized_L'] + PENALTY * components['normalized_K'] - components['normalized_H'])


class ConditionalComponentBaseline:
    """Fit-only component table. Lookup accepts observations, never targets.

    Construction estimates only label means. It is not a neural fit or an
    optimizer. Current engineering tests must use artificial records only.
    """
    def __init__(self, artifact):
        if artifact.get('schema') != BASELINE_SCHEMA:
            raise ValueError('Unknown conditional baseline schema')
        body = {k: v for k, v in artifact.items() if k != 'content_sha256'}
        if hashlib.sha256(_canonical(body)).hexdigest() != artifact.get('content_sha256'):
            raise ValueError('Baseline artifact identity differs')
        self._artifact = json.loads(json.dumps(artifact, allow_nan=False))

    @property
    def artifact(self):
        return json.loads(json.dumps(self._artifact, allow_nan=False))

    @classmethod
    def from_fit_records(cls, records, *, fit_physical_parents, source_sha256):
        records = list(records)
        parents = set(fit_physical_parents)
        if not parents or any(not isinstance(p, str) or not p for p in parents):
            raise ValueError('Predeclared nonempty physical fit-parent set required')
        if (not isinstance(source_sha256, Mapping) or not source_sha256
                or any(not isinstance(k, str) or not k or not isinstance(v, str)
                       or len(v) != 64 or any(c not in '0123456789abcdef' for c in v)
                       for k, v in source_sha256.items())):
            raise ValueError('Hash-bound fitting source identities required')
        seen, grouped, observed = set(), defaultdict(list), set()
        for record in records:
            if not isinstance(record, BaselineTrainingRecord):
                raise TypeError('Typed training records required; no arbitrary target/input dictionaries')
            record.validate()
            if record.physical_parent not in parents:
                raise ValueError('Record parent was not in the registered fitting fold')
            identity = record.physical_parent, record.trajectory_id, record.state_index
            if identity in seen:
                raise ValueError('Duplicate fitting state')
            seen.add(identity);observed.add(record.physical_parent)
            grouped[record.observation.key()].append(record)
        if observed != parents:
            raise ValueError('Fitting artifact must account for every declared physical parent')
        artifact = dict(schema=BASELINE_SCHEMA, input_dim=INPUT_DIM,
            fit_physical_parents=sorted(parents), source_sha256=dict(sorted(source_sha256.items())),
            condition_fields=sorted(OBSERVATION_FIELDS), remaining_feature_index=256,
            reduction='parent equal / trajectory equal / state mean within stratum',
            labels_are_training_targets_not_inputs=True,
            global_components=_balanced_component_means(records),
            conditional_components={key: _balanced_component_means(value) for key, value in sorted(grouped.items())},
            unknown_condition='fit-only global components', prediction_schema=C1_SCHEMA,
            penalty=PENALTY)
        artifact['content_sha256'] = hashlib.sha256(_canonical(artifact)).hexdigest()
        return cls(artifact)

    def lookup(self, observation):
        if not isinstance(observation, ObservableCondition):
            raise TypeError('Lookup accepts observable-only condition objects')
        condition = observation.key()
        source = self._artifact['conditional_components'].get(condition)
        fallback = source is None
        if fallback:
            source = self._artifact['global_components']
        return dict(**source, value=_component_value(source), condition_key=condition,
                    fallback=fallback, source_schema=BASELINE_SCHEMA,
                    baseline_content_sha256=self._artifact['content_sha256'])


def compose_conditional_prediction(baseline, residual_components, initial_task_count):
    """V=V_cond+f, with f composed from the SAME component residuals.

    This is an interface check, not a residual-network or optimizer. Domain is
    r_L >= -b_L and r_K >= -b_K. Invalid negative physical components fail rather
    than silently clipping or reporting an inconsistent extra scalar head.
    """
    if set(residual_components) != set(COMPONENTS):
        raise ValueError('Provide exactly the three normalized component residuals')
    residual = tuple(residual_components[k] for k in COMPONENTS)
    if any(not isinstance(x, torch.Tensor) or x.ndim != 1 or not x.is_floating_point()
           for x in residual):
        raise ValueError('T-shaped normalized residual tensors required')
    if any(x.shape != residual[0].shape or x.dtype != residual[0].dtype
           or x.device != residual[0].device for x in residual):
        raise ValueError('Residual shapes, dtypes and devices must match')
    base = tuple(_finite_scalar(baseline[k], k) for k in COMPONENTS)
    if base[0] < 0 or base[1] < 0:
        raise ValueError('Baseline L/K components must be nonnegative')
    baseline_value = -(base[0] + PENALTY * base[1] - base[2])
    if not math.isclose(float(baseline['value']), baseline_value, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError('Baseline value disagrees with its components')
    prediction = C1Prediction(initial_task_count, *(r + b for r, b in zip(residual, base))).validate()
    value_residual = -(residual[0] + PENALTY * residual[1] - residual[2])
    if not torch.allclose(prediction.value, value_residual + baseline_value, atol=1e-6, rtol=1e-6):
        raise ValueError('Conditional component/value identity violated')
    return prediction, value_residual
