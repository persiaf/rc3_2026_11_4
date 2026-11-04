"""Matched graph-input ablation, with portable NumPy inference.

Both variants have exactly the same parameters. ``control`` zeroes all graph
values before encoding; only the 28 statistics remain informative. No gain,
oracle decision, or repair outcome is a model input. Thread limits belong to a
whole planning session, not to each action's inference call.
"""
from __future__ import annotations

from contextlib import nullcontext
import json
import os
from pathlib import Path
import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

SCHEMA = 'rc3-v4-local-graph-v1'
PHASE = 'gain_supervised'
_MASK_KEYS = {'mask', 'cmask', 'tmask'}
_BATCH_KEYS = {'base', 'mask', 'cand', 'task', 'reach', 'dist', 'cmask', 'tmask'}


def feature_names():
    from rc3_v3.environment import FEATURE_NAMES
    return list(FEATURE_NAMES)


def batch_torch(batch, device='cpu'):
    """Convert a collated NumPy batch without importing Torch at module load."""
    import torch
    if set(batch) != _BATCH_KEYS:
        raise ValueError('unexpected graph batch fields')
    return {k: torch.as_tensor(v, device=device,
                              dtype=torch.bool if k in _MASK_KEYS else torch.float32)
            for k, v in batch.items()}


def _check_config(mean, std, variant, dim):
    if variant not in ('control', 'structured') or type(dim) is not int or dim < 4:
        raise ValueError('unsupported variant or dimension')
    if np.shape(mean) != (28,) or np.shape(std) != (28,):
        raise ValueError('normalization must have 28 entries')
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(np.asarray(std) <= 0):
        raise ValueError('invalid normalization')


def make_policy(mean, std, variant='structured', dim=32):
    import torch
    from torch import nn
    from torch.nn import functional as F
    _check_config(mean, std, variant, dim)

    class DistanceBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(dim, 3*dim)
            self.out = nn.Linear(dim, dim)
            self.beta = nn.Parameter(torch.zeros(()))
            self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.ff1, self.ff2 = nn.Linear(dim, 2*dim), nn.Linear(2*dim, dim)

        def forward(self, h, dist, cmask):
            q, k, v = self.qkv(h).chunk(3, dim=-1)
            logits = q @ k.transpose(-1, -2)/np.sqrt(dim)-F.softplus(self.beta)*dist
            logits = logits.masked_fill(~cmask[..., None, :], -torch.inf)
            attended = torch.softmax(logits, dim=-1) @ v
            h = self.norm1(h+self.out(attended))
            return self.norm2(h+self.ff2(torch.relu(self.ff1(h))))

    class GraphPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.variant, self.dim = variant, dim
            self.stop_rule, self.stop_threshold = 'binary_threshold', .9
            self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
            self.task_embed = nn.Linear(4, dim)
            self.cand_embed = nn.Linear(10, dim)
            self.arm_mix = nn.Linear(2*dim, dim)
            self.blocks = nn.ModuleList([DistanceBlock(), DistanceBlock()])
            self.graph_pool = nn.Linear(2*dim, 64)
            self.base_embed = nn.Linear(28, 64)
            self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU())
            self.rank, self.gain, self.positive = (nn.Linear(64, 1) for _ in range(3))
            self.stop = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, batch):
            base, mask = batch['base'], batch['mask']
            cand, task, reach, dist = (batch[k] for k in ('cand', 'task', 'reach', 'dist'))
            cmask, tmask = batch['cmask'], batch['tmask']
            # All-pad windows need one numerical dummy key; their action mask
            # removes them from losses, action selection and STOP pooling.
            first = torch.arange(cmask.shape[-1], device=cmask.device) == 0
            cmask = cmask | ((~cmask.any(-1))[..., None] & first)
            if self.variant == 'control':
                cand, task, reach, dist = (torch.zeros_like(v) for v in (cand, task, reach, dist))
            cand = cand.masked_fill(~cmask[..., None], 0.)
            task = task.masked_fill(~tmask[..., None], 0.)
            edge_mask = tmask[..., :, None, None] & cmask[..., None, :, None]
            reach = reach.masked_fill(~edge_mask, 0.)
            task_h = torch.relu(self.task_embed(task))
            arms = []
            for arm in range(2):
                f = reach[..., arm]
                arms.append(torch.einsum('swtc,swtd->swcd', f, task_h)
                            / f.sum(-2).clamp(min=1.)[..., None])
            h = self.cand_embed(cand)+self.arm_mix(torch.cat(arms, dim=-1))
            for block in self.blocks:
                h = block(h, dist, cmask)
            mean_h = (h*cmask[..., None]).sum(-2)/cmask.sum(-1).clamp(min=1)[..., None]
            max_h = h.masked_fill(~cmask[..., None], -torch.inf).max(-2).values
            graph = self.graph_pool(torch.cat((mean_h, max_h), dim=-1))
            common = self.base_embed((base-self.mean)/self.std)
            h = self.mlp(common+graph)
            rank = self.rank(h).squeeze(-1).masked_fill(~mask, -torch.inf)
            pooled = (h*mask[..., None]).sum(1)/mask.sum(1).clamp(min=1)[:, None]
            return (torch.cat((rank, self.stop(pooled)), dim=-1),
                    self.gain(h).squeeze(-1), self.positive(h).squeeze(-1))

        def outputs(self, batch):
            self.eval()
            with torch.no_grad():
                b = batch_torch(batch, self.mean.device)
                return tuple(v.detach().cpu().numpy() for v in self(b))

        def inference_session(self):
            return nullcontext()

    return GraphPolicy()


def _metadata(model, extra):
    metadata = dict(schema=SCHEMA, phase=PHASE, feature_names=feature_names(),
                    variant=model.variant, dim=model.dim,
                    mean=model.mean.detach().cpu().tolist(),
                    std=model.std.detach().cpu().tolist(),
                    stop_rule=model.stop_rule, stop_threshold=model.stop_threshold,
                    **extra)
    if metadata['schema'] != SCHEMA or metadata['phase'] != PHASE:
        raise ValueError('metadata cannot change model schema or phase')
    return metadata


def save_policy(path, model, **metadata):
    import torch
    saved = _metadata(model, metadata)
    saved['state_dict'] = model.state_dict()
    torch.save(saved, path)


def _validate_metadata(meta):
    if (meta.get('schema') != SCHEMA or meta.get('phase') != PHASE
            or meta.get('feature_names') != feature_names()
            or meta.get('stop_rule') != 'binary_threshold'
            or meta.get('stop_threshold') != .9):
        raise ValueError('checkpoint schema, phase, features or stopping rule mismatch')
    _check_config(meta.get('mean'), meta.get('std'), meta.get('variant'), meta.get('dim'))


def load_policy(path, device='cpu'):
    if Path(path).suffix.lower() == '.npz':
        return NumpyPolicy.load(path)
    import torch
    torch.set_num_threads(1)
    saved = torch.load(path, map_location=device, weights_only=True)
    meta = {k: v for k, v in saved.items() if k != 'state_dict'}
    _validate_metadata(meta)
    model = make_policy(meta['mean'], meta['std'], meta['variant'], meta['dim']).to(device)
    model.load_state_dict(saved['state_dict'], strict=True)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('nonfinite checkpoint weights')
    if not np.array_equal(model.mean.cpu().numpy(), np.asarray(meta['mean'], np.float32)) or not np.array_equal(model.std.cpu().numpy(), np.asarray(meta['std'], np.float32)):
        raise ValueError('checkpoint normalization metadata mismatch')
    model.metadata = meta
    model.eval()
    return model


def export_numpy(torch_path, numpy_path):
    import torch
    load_policy(torch_path)
    saved = torch.load(torch_path, map_location='cpu', weights_only=True)
    arrays = {k: v.detach().cpu().numpy() for k, v in saved.pop('state_dict').items()}
    np.savez_compressed(numpy_path, metadata=json.dumps(saved), **arrays)
    NumpyPolicy.load(numpy_path)


def _expected_shapes(dim):
    shapes = {'mean': (28,), 'std': (28,)}
    def linear(name, out, inp):
        shapes[name+'.weight'], shapes[name+'.bias'] = (out, inp), (out,)
    linear('task_embed', dim, 4)
    linear('cand_embed', dim, 10)
    linear('arm_mix', dim, 2*dim)
    for i in range(2):
        p = f'blocks.{i}'
        linear(p+'.qkv', 3*dim, dim)
        linear(p+'.out', dim, dim)
        shapes[p+'.beta'] = ()
        for norm in ('norm1', 'norm2'):
            shapes[p+'.'+norm+'.weight'] = (dim,)
            shapes[p+'.'+norm+'.bias'] = (dim,)
        linear(p+'.ff1', 2*dim, dim)
        linear(p+'.ff2', dim, 2*dim)
    linear('graph_pool', 64, 2*dim)
    linear('base_embed', 64, 28)
    for layer in ('mlp.1', 'mlp.3', 'stop.0'):
        linear(layer, 64, 64)
    for head in ('rank', 'gain', 'positive', 'stop.2'):
        linear(head, 1, 64)
    return shapes


class NumpyPolicy:
    def __init__(self, weights, metadata):
        _validate_metadata(metadata)
        expected = _expected_shapes(metadata['dim'])
        if set(weights) != set(expected):
            raise ValueError('checkpoint arrays do not match the architecture')
        for name, shape in expected.items():
            value = weights[name]
            if value.shape != shape or value.dtype.kind != 'f' or not np.isfinite(value).all():
                raise ValueError(f'invalid checkpoint array {name}')
        if not np.array_equal(weights['mean'], np.asarray(metadata['mean'], np.float32)) or not np.array_equal(weights['std'], np.asarray(metadata['std'], np.float32)):
            raise ValueError('checkpoint normalization metadata mismatch')
        self.w = {k: v.astype(np.float32, copy=False) for k, v in weights.items()}
        self.metadata, self.variant, self.dim = metadata, metadata['variant'], metadata['dim']
        self.stop_rule, self.stop_threshold = metadata['stop_rule'], metadata['stop_threshold']
        # Discover already loaded NumPy/SciPy BLAS libraries once per loaded
        # policy, then use cached controller state for every planning session.
        try:
            import scipy.linalg  # noqa: F401 - load its BLAS before discovery
            from threadpoolctl import ThreadpoolController
            self._controller = ThreadpoolController()
        except ImportError:
            self._controller = None

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            return cls({k: archive[k].copy() for k in archive.files if k != 'metadata'}, meta)

    def inference_session(self):
        return self._controller.limit(limits=1, user_api='blas') if self._controller is not None else nullcontext()

    def linear(self, x, name):
        return x @ self.w[name+'.weight'].T+self.w[name+'.bias']

    def norm(self, x, name):
        centered = x-x.mean(-1, keepdims=True)
        return (centered/np.sqrt((centered*centered).mean(-1, keepdims=True)+np.float32(1e-5))
                * self.w[name+'.weight']+self.w[name+'.bias'])

    @staticmethod
    def _batch(batch):
        if set(batch) != _BATCH_KEYS:
            raise ValueError('unexpected graph batch fields')
        b = {k: np.asarray(v, dtype=bool if k in _MASK_KEYS else np.float32) for k, v in batch.items()}
        if b['base'].ndim != 3 or b['base'].shape[-1] != 28:
            raise ValueError('invalid base statistics shape')
        s, w = b['base'].shape[:2]
        if b['cand'].ndim != 4 or b['task'].ndim != 4:
            raise ValueError('invalid node features shape')
        c, t = b['cand'].shape[-2], b['task'].shape[-2]
        shapes = dict(base=(s,w,28), mask=(s,w), cand=(s,w,c,10),
                      task=(s,w,t,4), reach=(s,w,t,c,2), dist=(s,w,c,c),
                      cmask=(s,w,c), tmask=(s,w,t))
        if s < 1 or w < 1 or c < 1 or t < 1 or any(b[k].shape != sh for k, sh in shapes.items()):
            raise ValueError('inconsistent graph batch shapes')
        if not b['mask'].any(-1).all() or not all(np.isfinite(b[k]).all() for k in _BATCH_KEYS-_MASK_KEYS):
            raise ValueError('empty action set or nonfinite graph input')
        if (b['reach'] < 0).any() or (b['reach'] > 1).any() or (b['dist'] < 0).any():
            raise ValueError('invalid reachability or navigation input')
        return b

    def outputs(self, batch):
        b = self._batch(batch)
        base, mask = b['base'], b['mask']
        cand, task, reach, dist = (b[k] for k in ('cand', 'task', 'reach', 'dist'))
        cmask, tmask = b['cmask'].copy(), b['tmask']
        cmask[..., 0] |= ~cmask.any(-1)
        if self.variant == 'control':
            cand, task, reach, dist = (np.zeros_like(v) for v in (cand, task, reach, dist))
        cand = np.where(cmask[..., None], cand, 0.)
        task = np.where(tmask[..., None], task, 0.)
        reach = np.where(tmask[..., :, None, None] & cmask[..., None, :, None], reach, 0.)
        task_h = np.maximum(self.linear(task, 'task_embed'), 0.)
        arms = []
        for arm in range(2):
            f = reach[..., arm]
            arms.append(np.einsum('swtc,swtd->swcd', f, task_h, optimize=True)
                        / np.maximum(f.sum(-2), 1.)[..., None])
        h = self.linear(cand, 'cand_embed')+self.linear(np.concatenate(arms, -1), 'arm_mix')
        for i in range(2):
            p = f'blocks.{i}'
            q, k, v = np.split(self.linear(h, p+'.qkv'), 3, axis=-1)
            beta = np.logaddexp(np.float32(0.), self.w[p+'.beta'])
            logits = q @ np.swapaxes(k, -1, -2)/np.sqrt(np.float32(self.dim))-beta*dist
            logits = np.where(cmask[..., None, :], logits, -np.inf)
            probabilities = np.exp(logits-logits.max(-1, keepdims=True))
            probabilities /= probabilities.sum(-1, keepdims=True)
            h = self.norm(h+self.linear(probabilities @ v, p+'.out'), p+'.norm1')
            h = self.norm(h+self.linear(np.maximum(self.linear(h, p+'.ff1'), 0.), p+'.ff2'), p+'.norm2')
        mean_h = (h*cmask[..., None]).sum(-2)/np.maximum(cmask.sum(-1), 1).astype(np.float32)[..., None]
        max_h = np.where(cmask[..., None], h, -np.inf).max(-2)
        graph = self.linear(np.concatenate((mean_h, max_h), -1), 'graph_pool')
        common = self.linear((base-self.w['mean'])/self.w['std'], 'base_embed')
        h = np.maximum(common+graph, 0.)
        h = np.maximum(self.linear(h, 'mlp.1'), 0.)
        h = np.maximum(self.linear(h, 'mlp.3'), 0.)
        rank = np.where(mask, self.linear(h, 'rank')[..., 0], -np.inf)
        pooled = (h*mask[..., None]).sum(1)/mask.sum(1).astype(np.float32)[:, None]
        stop = self.linear(np.maximum(self.linear(pooled, 'stop.0'), 0.), 'stop.2')
        return (np.concatenate((rank, stop), -1), self.linear(h, 'gain')[..., 0],
                self.linear(h, 'positive')[..., 0])
