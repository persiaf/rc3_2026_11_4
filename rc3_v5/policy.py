"""Frozen NumPy graph-policy adapter with environment-local embedding caching.

F, D and task/candidate semantics must remain fixed for the entire Env lifetime,
as required by the original environment's own feature and route caches. Create
a new Env after changing them. ``invalidate`` only clears this adapter's cache;
it does not make the old environment suitable for online map/task mutations.
This adapter reuses the frozen v4 checkpoint and does not retrain any weights.
"""
from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
import time
import numpy as np

from rc3_v4.model import NumpyPolicy
from rc3_v4.features import local_graphs, collate
from rc3_v3.model import select_action


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


class FrozenCapturingModel(NumpyPolicy):
    """Capture graph_pool without changing the audited NumPy forward path."""
    _protected = frozenset(('w', 'metadata', 'variant', 'dim', 'stop_rule', 'stop_threshold'))

    def __init__(self, weights, metadata):
        super().__init__(weights, metadata)
        # A plain writeable=False flag can be reversed on owned arrays. Immutable
        # bytes backing prevents both writes and turning writeability back on.
        self.w = MappingProxyType({k: np.frombuffer(v.tobytes(), dtype=v.dtype).reshape(v.shape)
                                   for k, v in self.w.items()})
        self.metadata = _freeze(self.metadata)
        self._frozen = True
        self.captured_graph = None

    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False) and name in self._protected:
            raise AttributeError('frozen checkpoint: load a new policy to change weights/configuration')
        super().__setattr__(name, value)

    def linear(self, x, name):
        value = super().linear(x, name)
        if name == 'graph_pool':
            self.captured_graph = value.copy()
        return value


class CachedGraphPolicy:
    def __init__(self, model):
        if not isinstance(model, FrozenCapturingModel):
            raise TypeError('embedding cache requires a frozen NumPy model')
        self._model = model
        self.graph_s = 0.; self.inference_s = 0.; self.calls = 0; self.stops = 0
        self.cache_hits = 0; self.cache_misses = 0; self.invalidations = 0
        self.last_outputs = None
        self._env = self._key = self._binding = self._graph = None

    @property
    def model(self):
        return self._model

    @classmethod
    def load(cls, path):
        if Path(path).suffix.lower() != '.npz':
            raise ValueError('cached deployment requires the exported v4 NPZ checkpoint')
        return cls(FrozenCapturingModel.load(path))

    def inference_session(self):
        return self.model.inference_session()

    def invalidate(self):
        """Drop adapter state; map/task changes still require a NEW Env."""
        self._env = self._key = self._binding = self._graph = None
        self.last_outputs = None
        self.model.captured_graph = None
        self.invalidations += 1

    @staticmethod
    def _observation(env, features, mask):
        base, valid = np.asarray(features, np.float32), np.asarray(mask, bool)
        n = len(env.windows)
        if base.shape != (n, 28) or valid.shape != (n,) or not valid.any() or not np.isfinite(base).all():
            raise ValueError('invalid current observation')
        return base, valid

    @staticmethod
    def _static_binding(env):
        p = env.problem
        # Cheap replacement checks supplement, but do not replace, the static
        # Env contract. We deliberately do not hash full F/D on every action.
        return (id(p), id(p.F), p.F.shape, id(p.nav), id(p.nav['costs']),
                p.nav['costs'].shape, id(p.scene))

    def outputs_env(self, env, features, mask):
        tick = time.perf_counter()
        base, valid = self._observation(env, features, mask)
        binding = self._static_binding(env)
        if self._env is env and self._binding != binding:
            raise ValueError('problem arrays/scene replaced within an Env; create a new Env')
        key = (tuple(env.route), tuple((w['start'], w['end']) for w in env.windows))
        if self._env is not env or self._key != key:
            graphs = local_graphs(env.problem, env.route, env.windows)
            batch = collate([dict(features=base, mask=valid, graphs=graphs)])
            self.graph_s += time.perf_counter()-tick
            tick = time.perf_counter()
            out = self.model.outputs(batch)
            self._graph = self.model.captured_graph.copy()
            self._graph.setflags(write=False)
            self._env, self._key, self._binding = env, key, binding
            self.cache_misses += 1
        else:
            base, valid = base[None, :, :], valid[None, :]
            self.graph_s += time.perf_counter()-tick
            tick = time.perf_counter()
            m = self.model
            # Preserve v4 operation order and dtype for bitwise equivalence.
            common = m.linear((base-m.w['mean'])/m.w['std'], 'base_embed')
            h = np.maximum(common+self._graph, 0.)
            h = np.maximum(m.linear(h, 'mlp.1'), 0.)
            h = np.maximum(m.linear(h, 'mlp.3'), 0.)
            rank = np.where(valid, m.linear(h, 'rank')[..., 0], -np.inf)
            pooled = (h*valid[..., None]).sum(1)/valid.sum(1).astype(np.float32)[:, None]
            stop = m.linear(np.maximum(m.linear(pooled, 'stop.0'), 0.), 'stop.2')
            out = (np.concatenate((rank, stop), -1), m.linear(h, 'gain')[..., 0],
                   m.linear(h, 'positive')[..., 0])
            self.cache_hits += 1
        self.inference_s += time.perf_counter()-tick
        self.last_outputs = out
        return out

    def select_env(self, env, features, mask):
        out = self.outputs_env(env, features, mask)
        action = select_action(out[0][0], mask, self.model.stop_rule, self.model.stop_threshold)
        self.calls += 1; self.stops += int(action is None)
        return action


def load_policy(path):
    return CachedGraphPolicy.load(path)
