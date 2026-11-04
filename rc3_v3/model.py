"""Versioned gain ranking models. Torch trains; portable NPZ serves inference."""
import json
import os
from pathlib import Path
import numpy as np

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def schema():
    from .environment import FEATURE_NAMES, FEATURE_SCHEMA
    return FEATURE_NAMES, FEATURE_SCHEMA


def batch_states(states, device='cpu'):
    import torch
    names, _ = schema()
    n = max(1, max(len(x) for x, _ in states))
    x = np.zeros((len(states), n, len(names)), np.float32)
    mask = np.zeros((len(states), n), bool)
    for i, (features, valid) in enumerate(states):
        x[i, :len(features)] = features
        mask[i, :len(valid)] = valid
    return torch.as_tensor(x, device=device), torch.as_tensor(mask, device=device)


def select_action(logits, mask, stop_rule='binary_threshold', stop_threshold=.9):
    """Stopping is a scene decision, not a race against one of many windows."""
    logits, mask = np.asarray(logits), np.asarray(mask, bool)
    if logits.shape != (len(mask)+1,) or not mask.any() or not np.isfinite(logits[np.r_[mask,True]]).all():
        raise ValueError('invalid learned logits or action mask')
    if stop_rule == 'categorical_argmax':
        chosen = int(logits.argmax())
        return None if chosen == len(mask) else chosen
    if stop_rule != 'binary_threshold' or not 0 < stop_threshold < 1:
        raise ValueError('invalid stopping rule')
    if logits[-1] >= np.log(stop_threshold/(1-stop_threshold)):
        return None
    return int(np.where(mask,logits[:-1],-np.inf).argmax())


def make_policy(mean, std, architecture='attention', dim=64, stop_rule='binary_threshold', stop_threshold=.9):
    import torch
    from torch import nn

    class GainPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            if architecture not in ('attention', 'mlp'):
                raise ValueError('unsupported architecture')
            self.architecture, self.dim = architecture, dim
            self.stop_rule, self.stop_threshold = stop_rule, stop_threshold
            self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
            self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32))
            self.embed = nn.Linear(len(mean), dim)
            if architecture == 'attention':
                self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(dim, 4, dim*2, dropout=0., batch_first=True) for _ in range(2)])
            else:
                self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim), nn.ReLU())
            self.rank = nn.Linear(dim, 1)
            self.gain = nn.Linear(dim, 1)
            self.positive = nn.Linear(dim, 1)
            self.stop = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, 1))

        def forward(self, x, mask):
            if not mask.any(1).all():
                raise ValueError('empty legal action set')
            h = self.embed((x-self.mean)/self.std)
            if self.architecture == 'attention':
                for layer in self.encoder:
                    h = layer(h, src_key_padding_mask=~mask)
            else:
                h = self.mlp(h)
            rank = self.rank(h).squeeze(-1).masked_fill(~mask, -torch.inf)
            pooled = (h*mask[..., None]).sum(1)/mask.sum(1)[:, None]
            stop = self.stop(pooled)
            return torch.cat((rank, stop), -1), self.gain(h).squeeze(-1), self.positive(h).squeeze(-1)

        def outputs(self, features, mask):
            self.eval()
            with torch.no_grad():
                x, valid = batch_states([(features, mask)], self.mean.device)
                result = self(x, valid)
                return tuple(v[0].cpu().numpy() for v in result)

        def scores(self, features, mask):
            return self.outputs(features, mask)[0][:-1]

        def select(self, features, mask):
            logits = self.outputs(features, mask)[0]
            return select_action(logits,mask,self.stop_rule,self.stop_threshold)

    return GainPolicy()


def save_policy(path, model, **metadata):
    import torch
    names, version = schema()
    torch.save(dict(state_dict=model.state_dict(), mean=model.mean.cpu().tolist(), std=model.std.cpu().tolist(),
        architecture=model.architecture, dim=model.dim, feature_names=names, schema=version,
        phase='gain_supervised', stop_rule=model.stop_rule, stop_threshold=model.stop_threshold, **metadata), path)


def load_policy(path, device='cpu'):
    if Path(path).suffix.lower() == '.npz':
        return NumpyPolicy.load(path)
    import torch
    torch.set_num_threads(1)
    saved = torch.load(path, map_location=device, weights_only=True)
    names, version = schema()
    if saved['schema'] != version or saved['feature_names'] != names or saved['phase'] != 'gain_supervised':
        raise ValueError('checkpoint feature schema or training phase mismatch')
    model = make_policy(saved['mean'], saved['std'], saved['architecture'], saved['dim'],
        saved.get('stop_rule','categorical_argmax'),saved.get('stop_threshold',.9)).to(device)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model


def export_numpy(torch_path, numpy_path):
    import torch
    saved = torch.load(torch_path, map_location='cpu', weights_only=True)
    load_policy(torch_path)  # Validate schema and state_dict before portable export.
    arrays = {k: v.cpu().numpy() for k, v in saved.pop('state_dict').items()}
    np.savez_compressed(numpy_path, metadata=json.dumps(saved), **arrays)


class NumpyPolicy:
    def __init__(self, weights, metadata):
        names, version = schema()
        if metadata.get('schema') != version or metadata.get('feature_names') != names or metadata.get('phase') != 'gain_supervised':
            raise ValueError('checkpoint feature schema or training phase mismatch')
        if metadata['architecture'] not in ('attention', 'mlp') or metadata['dim'] % 4:
            raise ValueError('unsupported architecture or dimension')
        if not all(np.isfinite(v).all() for v in weights.values()) or (weights['std'] <= 0).any():
            raise ValueError('invalid checkpoint values')
        self.w, self.metadata = weights, metadata
        self.stop_rule = metadata.get('stop_rule','categorical_argmax')
        self.stop_threshold = metadata.get('stop_threshold',.9)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive['metadata']))
            return cls({k: archive[k].copy() for k in archive.files if k != 'metadata'}, meta)

    def linear(self, x, name):
        return x@self.w[name+'.weight'].T+self.w[name+'.bias']

    def norm(self, x, name):
        c = x-x.mean(-1, keepdims=True)
        return c/np.sqrt((c*c).mean(-1, keepdims=True)+np.float32(1e-5))*self.w[name+'.weight']+self.w[name+'.bias']

    def outputs(self, features, mask):
        from contextlib import nullcontext
        try:
            from threadpoolctl import threadpool_limits
            context = threadpool_limits(limits=1, user_api='blas')
        except ImportError:
            context = nullcontext()
        x, mask = np.asarray(features, np.float32), np.asarray(mask, bool)
        if x.shape != (len(mask), len(self.w['mean'])) or not mask.any() or not np.isfinite(x).all():
            raise ValueError('invalid observation')
        with context:
            h = self.linear((x-self.w['mean'])/self.w['std'], 'embed')
            if self.metadata['architecture'] == 'mlp':
                h = np.maximum(h, 0)
                h = np.maximum(self.linear(h, 'mlp.1'), 0)
                h = np.maximum(self.linear(h, 'mlp.3'), 0)
            else:
                n, dim = h.shape
                for i in range(2):
                    p = f'encoder.{i}'
                    qkv = h@self.w[p+'.self_attn.in_proj_weight'].T+self.w[p+'.self_attn.in_proj_bias']
                    q, k, v = qkv.reshape(n, 3, 4, dim//4).transpose(1, 2, 0, 3)
                    logits = (q@k.transpose(0, 2, 1))/np.sqrt(np.float32(dim//4))
                    logits[..., ~mask] = -np.inf
                    probabilities = np.exp(logits-logits.max(-1, keepdims=True))
                    probabilities /= probabilities.sum(-1, keepdims=True)
                    attended = (probabilities@v).transpose(1, 0, 2).reshape(n, dim)
                    h = self.norm(h+self.linear(attended, p+'.self_attn.out_proj'), p+'.norm1')
                    h = self.norm(h+self.linear(np.maximum(self.linear(h, p+'.linear1'), 0), p+'.linear2'), p+'.norm2')
            rank = self.linear(h, 'rank').reshape(-1)
            rank[~mask] = -np.inf
            pooled = h[mask].mean(0)
            stop = self.linear(np.maximum(self.linear(pooled, 'stop.0'), 0), 'stop.2').reshape(-1)
            return np.r_[rank, stop], self.linear(h, 'gain').reshape(-1), self.linear(h, 'positive').reshape(-1)

    def scores(self, features, mask):
        return self.outputs(features, mask)[0][:-1]

    def select(self, features, mask):
        logits = self.outputs(features, mask)[0]
        return select_action(logits,mask,self.stop_rule,self.stop_threshold)
