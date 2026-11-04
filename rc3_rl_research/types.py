"""Versioned common contracts for budget-aware search, not a trained policy."""
from dataclasses import dataclass, field, asdict
from typing import Any
import math

SCHEMA='rc3-rl-g01-v1'


@dataclass(frozen=True)
class SearchConfig:
    epsilon: float = .02
    horizon: int = 16
    work_budget: int = 2_000_000
    work_scale: float = 10_000.
    lambda_work: float = .01
    potential_weight: float = 0.
    max_tasks: int = 8
    max_candidates: int = 80
    max_width: int = 4
    max_actions: int = 16  # Non-STOP menu entries; STOP is appended separately.
    physical_scale: float = 2.2
    cache_enabled: bool = False

    def __post_init__(self):
        for name in ('horizon','work_budget','max_tasks','max_candidates','max_width','max_actions'):
            value=getattr(self,name)
            if type(value) is not int or value<0:raise ValueError('invalid '+name)
        if not self.max_width or not self.max_actions:raise ValueError('positive menu dimensions required')
        for name in ('epsilon','work_scale','lambda_work','potential_weight','physical_scale'):
            value=getattr(self,name)
            if not math.isfinite(value) or value<0:raise ValueError('invalid '+name)
        if not self.work_scale or not self.physical_scale:raise ValueError('positive scales required')
        if self.cache_enabled:raise ValueError('G0/G1 uses no proposal cache; all selected queries are charged')


@dataclass(frozen=True)
class Action:
    kind: str
    start: int = 0
    end: int = 0
    forbidden_node: int | None = None
    key: str = ''

    def to_dict(self):return asdict(self)


@dataclass
class ActionMenu:
    actions: list[Action]
    mask: list[bool]
    features: list[list[float]]
    feature_names: list[str]
    context_key: str
    action_context_keys: list[str]
    work_units: int
    diagnostics: dict[str,Any] = field(default_factory=dict)


@dataclass
class ProposalResult:
    route: list[int] | None
    valid: bool  # Combinatorial check only; independent certification is still mandatory.
    changed: bool
    work_units: int
    failure_reason: str | None = None
    raw_route: list[int] | None = None
    counters: dict[str,int] = field(default_factory=dict)
    diagnostics: dict[str,Any] = field(default_factory=dict)


@dataclass
class Certificate:
    route: list[int]
    plan: dict[str,Any]
    cost: float
    diagnostics: dict[str,Any] = field(default_factory=dict)
