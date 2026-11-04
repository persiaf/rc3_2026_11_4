"""Portable, single-forward scene selector. Deployment does not import PyTorch."""
import json
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES
from .solver import RECIPES


class Selector:
    def __init__(self, path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self.meta = json.loads(str(data['metadata']))
            if self.meta['feature_names'] != FEATURE_NAMES or self.meta['recipes'] != RECIPES:
                raise ValueError('model feature or recipe schema does not match this solver')
            self.mean, self.std = data['mean'].copy(), data['std'].copy()
            self.forest = None
            if self.meta.get('architecture') == 'random_forest':
                self.forest = {key:data[key].copy() for key in ('left','right','feature','threshold','value')}
            self.members = [
                [(data[f'm{k}_w{i}'].copy(), data[f'm{k}_b{i}'].copy()) for i in range(3)]
                for k in range(self.meta['members'])]

    def predict(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError('expected finite scene feature vector')
        z = (x-self.mean)/self.std
        if self.forest is not None:
            shape = z.shape[:-1]
            z = z.reshape(-1,z.shape[-1])
            tree = self.forest
            indices = np.arange(len(tree['left']))[None,:]
            nodes = np.zeros((len(z),len(tree['left'])),dtype=np.int64)
            for _ in range(self.meta['max_depth']):
                features = tree['feature'][indices,nodes]
                left = np.take_along_axis(z,features.clip(0),axis=1) <= tree['threshold'][indices,nodes]
                following = np.where(left,tree['left'][indices,nodes],tree['right'][indices,nodes])
                nodes = np.where(features>=0,following,nodes)
            return tree['value'][indices,nodes].mean(1).reshape(*shape,tree['value'].shape[-1])
        predictions = []
        for layers in self.members:
            h = z
            for i, (weight, bias) in enumerate(layers):
                h = h @ weight.T + bias
                if i < len(layers)-1:
                    h = np.maximum(h, 0)
            predictions.append(h)
        return np.mean(predictions, axis=0)

    def rank(self, x, top_k=1):
        if not 1 <= top_k <= len(RECIPES):
            raise ValueError('top_k must be in 1..13')
        predictions = self.predict(x)
        if 'action_sets' in self.meta:
            actions = self.meta['action_sets']
            ids = [i for i, a in enumerate(actions) if len(a) == top_k]
            if not ids:
                raise ValueError('this portfolio model supports top_k=1 or 3')
            selected = np.asarray([actions[i] for i in ids])
            return selected[np.argmin(predictions[..., ids], axis=-1)]
        return np.argsort(predictions, kind='stable')[..., :top_k]
