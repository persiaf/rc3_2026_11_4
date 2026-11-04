"""Portable inference for the frozen v2 actor; no training and no pickle loading.

Weights are a lossless array export of the uploaded PyTorch checkpoints.
Small floating-point differences between numerical backends remain possible.
"""
from contextlib import nullcontext
import json

import numpy as np

from .repair import FEATURE_NAMES

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


class NumpyPolicy:
    def __init__(self, weights, metadata):
        if metadata['feature_names'] != FEATURE_NAMES or metadata['dim'] != 64:
            raise ValueError('unsupported checkpoint feature schema or dimension')
        if not all(np.isfinite(value).all() for value in weights.values()):
            raise ValueError('nonfinite checkpoint weights')
        if np.any(weights['std'] <= 0):
            raise ValueError('nonpositive feature standard deviation')
        self.w, self.metadata = weights, metadata

    @classmethod
    def load(cls, path, expected_stage=None):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata']))
            weights = {name: archive[name].copy() for name in archive.files if name != 'metadata'}
        if expected_stage is not None and metadata.get('stage') != expected_stage:
            raise ValueError(f"requested {expected_stage}, checkpoint stage is {metadata.get('stage')}")
        return cls(weights, metadata)

    def _linear(self, x, name):
        return x@self.w[name+'.weight'].T+self.w[name+'.bias']

    def _norm(self, x, name):
        centered = x-x.mean(-1, keepdims=True)
        return centered/np.sqrt((centered*centered).mean(-1, keepdims=True)+np.float32(1e-5))*self.w[name+'.weight']+self.w[name+'.bias']

    def scores(self, features, mask):
        features, mask = np.asarray(features, dtype=np.float32), np.asarray(mask, dtype=bool)
        if features.shape != (len(mask), len(FEATURE_NAMES)) or not mask.any():
            raise ValueError('invalid observation or empty legal action set')
        if not np.isfinite(features).all():
            raise ValueError('nonfinite observation')
        context = threadpool_limits(limits=1, user_api='blas') if threadpool_limits else nullcontext()
        with context:
            h = self._linear((features-self.w['mean'])/self.w['std'], 'embed')
            n, dim = h.shape
            heads, head_dim = 4, dim//4
            for i in range(2):
                prefix = f'encoder.{i}'
                qkv = h@self.w[prefix+'.self_attn.in_proj_weight'].T+self.w[prefix+'.self_attn.in_proj_bias']
                q, k, v = qkv.reshape(n, 3, heads, head_dim).transpose(1, 2, 0, 3)
                scores = (q@k.transpose(0, 2, 1))/np.sqrt(np.float32(head_dim))
                scores[..., ~mask] = -np.inf
                scores -= scores.max(-1, keepdims=True)
                probabilities = np.exp(scores)
                probabilities /= probabilities.sum(-1, keepdims=True)
                attended = (probabilities@v).transpose(1, 0, 2).reshape(n, dim)
                h = self._norm(h+self._linear(attended, prefix+'.self_attn.out_proj'), prefix+'.norm1')
                h = self._norm(h+self._linear(np.maximum(self._linear(h, prefix+'.linear1'), 0), prefix+'.linear2'), prefix+'.norm2')
            logits = self._linear(np.maximum(self._linear(h, 'actor.0'), 0), 'actor.2').reshape(-1)
            logits[~mask] = -np.inf
            return logits
