"""Exact R1 constructive states and batched frozen-network completion.

This module does not create scenes, invoke H, train a network, or substitute a
rule policy when a neural policy fails. Route nodes use1..M;0 is the depot.
"""
from __future__ import annotations

import operator
from typing import MutableMapping, Sequence

import numpy as np
import torch


DISTANCE_SCALE = 10.0
NEAR_NEIGHBORS = 16
LONG_NEIGHBORS = 4
MAX_COMPLETION_BATCH = 8


def _count(stats, key, amount=1):
    if stats is not None:
        stats[key] = int(stats.get(key, 0)) + int(amount)


def _node(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('Boolean is not an original node ID')
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError('Node IDs must be integers') from exc


def geometric_neighbors(xy):
    """Return the common Euclidean16-near+4-dispersed graph and padding mask.

    Self is excluded. Distance ties use zero-based original candidate ID. The
    four farther relations select ranks floor((2q+1)*R/8),q=0..3, from the
    distance-sorted candidates remaining after the nearest16. For R<4, all R
    relations are used once. Unused slots point to0 and have a false mask.
    No navigation cost or reachability is consulted in graph construction.
    """
    xy = np.asarray(xy, dtype=np.float64)
    m = len(xy)
    indices = np.zeros((m, NEAR_NEIGHBORS + LONG_NEIGHBORS), dtype=np.int64)
    valid = np.zeros_like(indices, dtype=bool)
    if not m:
        return indices, valid
    ids = np.arange(m)
    for i in range(m):
        delta = xy - xy[i]
        squared = (delta * delta).sum(1)
        order = np.lexsort((ids, squared))
        order = order[order != i]
        near = order[:NEAR_NEIGHBORS].tolist()
        rest = order[NEAR_NEIGHBORS:]
        if len(rest) <= LONG_NEIGHBORS:
            far = rest.tolist()
        else:
            far = [int(rest[((2*q+1)*len(rest))//8]) for q in range(LONG_NEIGHBORS)]
        chosen = near + far
        indices[i, :len(chosen)] = chosen
        valid[i, :len(chosen)] = True
    return indices, valid


def prepare_data(item, representation, device):
    """Create batch-one static tensors with fixed physical10m normalization.

    task5: relative x,y,z divided10; original left/right-arm F degrees /M.
    candidate7: relative x,y divided10; cos/sin theta; height/10; union-C
    degree /N; depot bidirectional navigation reachability. Base and Nav share
    all these features, the true current/home D and the geometric graph. Only
    relation input differs: Euclidean versus actual candidate-pair D, with its
    finite flag. Thus this ablates future pair relations, not all navigation.
    """
    aliases = {'base': 'base', 'basic': 'base', 'coverage': 'base',
               'nav': 'nav', 'navigation': 'nav', 'coverage+nav': 'nav',
               'coverage+trueD': 'nav'}
    if representation not in aliases:
        raise ValueError(f'Unknown representation: {representation}')
    mode = aliases[representation]
    cover = np.asarray(item.C, dtype=bool)
    n, m = cover.shape
    if not m:
        raise ValueError('The R1 network requires at least one candidate')
    D = np.asarray(item.D, dtype=np.float64)
    if D.shape != (m+1, m+1):
        raise ValueError('D shape must include the depot plus every candidate')
    scene = item.base.scene
    if len(scene['tasks']) != n or len(scene['candidates']) != m:
        raise ValueError('Scene original IDs do not match C')
    for records in ('tasks', 'candidates'):
        if [x['id'] for x in scene[records]] != list(range(len(scene[records]))):
            raise ValueError('Noncontiguous original IDs')
    depot = np.asarray(scene['start'], dtype=np.float64)
    xy = np.asarray([[x['x'], x['y']] for x in scene['candidates']], dtype=np.float64)
    relative_xy = xy - depot[:2]
    task_xyz = np.asarray([[x['x']-depot[0], x['y']-depot[1], x['z']]
                           for x in scene['tasks']], dtype=np.float64).reshape(n, 3)
    F = np.asarray(item.base.F, dtype=bool)
    if F.shape != (n, m, 2) or not np.array_equal(F.any(2), cover):
        raise ValueError('R1 requires the original two-arm union coverage F')
    task = np.column_stack((task_xyz / DISTANCE_SCALE,
                            F[:, :, 0].sum(1)/m, F[:, :, 1].sum(1)/m)).astype(np.float32)
    connected = np.isfinite(D[0, 1:]) & np.isfinite(D[1:, 0])
    candidate = np.column_stack((relative_xy/DISTANCE_SCALE,
        [np.cos(x['theta']) for x in scene['candidates']],
        [np.sin(x['theta']) for x in scene['candidates']],
        [x['h']/DISTANCE_SCALE for x in scene['candidates']],
        cover.sum(0)/max(n, 1), connected)).astype(np.float32)
    indices, valid = geometric_neighbors(xy)
    if mode == 'base':
        delta = xy[indices] - xy[:, None, :]
        distances = np.sqrt((delta*delta).sum(2))
    else:
        distances = D[np.arange(m)[:, None]+1, indices+1]
    finite = np.isfinite(distances) & valid
    edge = np.stack((np.where(finite, distances/DISTANCE_SCALE, 0.), finite), -1).astype(np.float32)
    arrays = dict(C=cover.astype(np.float32), D=D, task=task, candidate=candidate,
        task_valid=np.ones(n, dtype=bool), candidate_valid=np.ones(m, dtype=bool),
        neighbors=indices, neighbor_valid=valid, edgefeatures=edge)
    return {key: torch.as_tensor(value, device=device).unsqueeze(0)
            for key, value in arrays.items()}


class State:
    """Exact full-prefix kernel; rawL excludes the eventual closing-home edge."""

    def __init__(self, item, prefix: Sequence[int] = ()):
        self.item = item
        self.C = np.asarray(item.C, dtype=bool)
        self.D = np.asarray(item.D)
        if self.C.ndim != 2 or self.D.shape != (self.C.shape[1]+1,)*2:
            raise ValueError('Invalid C/D shapes')
        self.remaining = np.ones(self.C.shape[0], dtype=bool)
        self.current = 0
        self.prefix = []
        self.rawL = 0.0
        for node in prefix:
            self.step(node)

    @property
    def complete(self):
        return not bool(self.remaining.any())

    def legal_mask(self):
        return (self.C[self.remaining].any(0)
                & np.isfinite(self.D[self.current, 1:])
                & np.isfinite(self.D[1:, 0]))

    def step(self, node):
        node = _node(node)
        if not 1 <= node <= self.C.shape[1] or not self.legal_mask()[node-1]:
            raise ValueError(f'Illegal full-prefix action {node}')
        self.rawL += float(self.D[self.current, node])
        self.remaining &= ~self.C[:, node-1]
        self.current = node
        self.prefix.append(node)
        return self

    def route(self):
        if not self.complete:
            raise ValueError('An incomplete prefix is not a raw closed route')
        return [0, *self.prefix, 0]


def complete_gB(item, B, prefix=(), action=None, stats=None):
    """Exact fixed-order reference: scan original B for first legal gain>0."""
    state = State(item, prefix)
    if action is not None:
        state.step(action)
        _count(stats, 'decisions')
    while not state.complete:
        mask = state.legal_mask()
        chosen = next((int(n) for n in B if 1 <= int(n) <= len(mask) and mask[int(n)-1]), None)
        if chosen is None:
            raise ValueError('g_B cannot complete this full legal prefix')
        state.step(chosen)
        _count(stats, 'decisions')
    _count(stats, 'raw_trajectories')
    return state.route()


def prior_complete(item, prefix=(), action=None, stats=None):
    """Deterministic gain/(current distance+.75) warm-data prior."""
    state = State(item, prefix)
    if action is not None:
        state.step(action)
        _count(stats, 'decisions')
    while not state.complete:
        mask = state.legal_mask()
        ids = np.flatnonzero(mask)
        if not len(ids):
            raise ValueError('Prior cannot complete this full legal prefix')
        gain = state.C[state.remaining].sum(0)
        ratio = gain[ids] / (state.D[state.current, ids+1] + .75)
        chosen = int(ids[np.argmax(ratio)]) + 1
        state.step(chosen)
        _count(stats, 'decisions')
    _count(stats, 'raw_trajectories')
    return state.route()


class NeuralPolicy:
    """Greedy frozen-model policy with shared static and full-prefix encoding.

    Recreate this object after any weight update. decisions counts appended
    non-prefix actions, including forced query actions; score_state is only a
    menu inspection and increments score_state_calls. forward_score_rows counts
    all batch rows, including inactive rows in the final parallel tail.
    """

    def __init__(self, model, item, representation, device, stats: MutableMapping | None = None):
        self.model, self.item = model, item
        self.representation, self.device = representation, torch.device(device)
        self.stats = {} if stats is None else stats
        self.data = prepare_data(item, representation, self.device)
        self.model.eval()
        with torch.inference_mode():
            self.encoded = self.model.static_encode(self.data)
        self._versions = tuple(p._version for p in self.model.parameters())
        _count(self.stats, 'static_encodes')

    def _check_frozen(self):
        if tuple(p._version for p in self.model.parameters()) != self._versions:
            raise RuntimeError('Model changed after static caching; create a new NeuralPolicy')

    def _prefix_hidden(self, prefix):
        nodes = torch.as_tensor([prefix], dtype=torch.long, device=self.device)
        lengths = torch.as_tensor([len(prefix)], dtype=torch.long, device=self.device)
        _count(self.stats, 'prefix_encode_batches')
        return self.model.encode_prefix(self.encoded, nodes, lengths)

    def _score(self, encoded, states, hidden):
        remaining = torch.as_tensor(np.stack([s.remaining for s in states]), device=self.device)
        current = torch.as_tensor([s.current for s in states], device=self.device)
        rawL = torch.as_tensor([s.rawL for s in states], dtype=torch.float32, device=self.device)
        steps = torch.as_tensor([len(s.prefix) for s in states], device=self.device)
        scores = self.model.score(encoded, remaining, current, hidden, rawL, steps)
        _count(self.stats, 'forward_score_batches')
        _count(self.stats, 'forward_score_rows', len(states))
        return scores.detach().cpu().numpy()

    def score_state(self, prefix):
        """Return finite scores[M], exact legal mask[M]; no action is selected."""
        self._check_frozen()
        state = State(self.item, prefix)
        with torch.inference_mode():
            hidden = self._prefix_hidden(state.prefix)
            scores = self._score(self.encoded, [state], hidden)[0]
        if not np.isfinite(scores).all():
            raise FloatingPointError('Nonfinite neural action scores')
        _count(self.stats, 'score_state_calls')
        return scores, state.legal_mask()

    def complete_many(self, prefix, forced_actions):
        """Complete up to8 actions in one parallel policy batch, preserving prefix.

        A None action means start directly from the supplied prefix. All real
        forced actions are validated before any forward call. Every return is
        the complete raw [0,*prefix,*completion,0], with no postprocessing.
        """
        self._check_frozen()
        prefix = list(prefix)
        actions = list(forced_actions)
        if not 1 <= len(actions) <= MAX_COMPLETION_BATCH:
            raise ValueError('R1 forced-action batch must contain1..8 rows')
        states = [State(self.item, prefix) for _ in actions]
        before = [s.current for s in states]
        forced = [a is not None for a in actions]
        forced_nodes = [0 if a is None else _node(a) for a in actions]
        for state, action in zip(states, actions):
            if action is not None:
                state.step(action)
        b = len(states)
        encoded = self.model.expand_encoded(self.encoded, b)
        with torch.inference_mode():
            hidden = self._prefix_hidden(list(prefix)).expand(b, -1).contiguous()
            if any(forced):
                hidden = self.model.update_prefix(encoded, hidden,
                    torch.as_tensor(forced_nodes, device=self.device),
                    torch.as_tensor(before, device=self.device),
                    torch.as_tensor(forced, device=self.device))
                _count(self.stats, 'prefix_update_batches')
                _count(self.stats, 'prefix_update_rows', b)
                _count(self.stats, 'decisions', sum(forced))
            rounds = 0
            while any(not s.complete for s in states):
                rounds += 1
                if rounds > states[0].C.shape[0]:
                    raise RuntimeError('Positive-gain completion exceeded task count')
                scores = self._score(encoded, states, hidden)
                if not np.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite neural action scores')
                chosen, current, active = [], [], []
                for row, state in enumerate(states):
                    current.append(state.current)
                    is_active = not state.complete
                    active.append(is_active)
                    if not is_active:
                        chosen.append(0)
                        continue
                    mask = state.legal_mask()
                    if not mask.any():
                        raise ValueError('Neural policy has no legal positive-gain action')
                    node = int(np.argmax(np.where(mask, scores[row], -np.inf))) + 1
                    state.step(node)
                    chosen.append(node)
                hidden = self.model.update_prefix(encoded, hidden,
                    torch.as_tensor(chosen, device=self.device),
                    torch.as_tensor(current, device=self.device),
                    torch.as_tensor(active, device=self.device))
                _count(self.stats, 'prefix_update_batches')
                _count(self.stats, 'prefix_update_rows', b)
                _count(self.stats, 'decisions', sum(active))
        routes = [s.route() for s in states]
        _count(self.stats, 'complete_many_calls')
        _count(self.stats, 'raw_trajectories', b)
        return routes

    def trajectory(self, prefix=()):
        return self.complete_many(prefix, [None])[0]
