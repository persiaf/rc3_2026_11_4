"""Attach counterfactual navigation inputs to frozen Integrated-RPPO features.

This is a candidate adapter, not an entry point to the closed scientific run.
The old trainer's hardcoded 202-dimension/2-logit routines must NOT train it.
"""
import numpy as np
import torch
from torch import nn

from .lookahead import prepare_service_distances, prepared_post_service_distances
from .model import SmallCorrection, freeze_supervised_base


def augment_candidate_features(original, lookahead, distances_enabled=True):
    if type(distances_enabled) is not bool:
        raise ValueError('The paired feature condition must be explicit bool')
    extra = torch.as_tensor(lookahead['finite_features'], dtype=original.dtype,
                            device=original.device).detach().clone()
    if original.ndim != 2 or original.shape[1] != 202 or extra.shape != (len(original), 4):
        raise ValueError('Use the original ordered M x 202 features and matched M x 4 lookahead')
    if not bool(torch.isfinite(extra).all()):
        raise ValueError('Lookahead encoding must be finite with explicit unreachable flags')
    if not distances_enabled:
        # Same architecture and count/unreachability signals in BOTH conditions.
        # Only the two actual distance summaries are ablated.
        extra[:, :2] = 0.
    return torch.cat([original.detach(), extra], dim=1)


class LookaheadResidualActor(nn.Module):
    def __init__(self, frozen_feature_encoder, distances_enabled=True, max_adjustment=0.25):
        super().__init__()
        if type(distances_enabled) is not bool:
            raise ValueError('Use an explicit preregistered distance condition')
        self.encoder = freeze_supervised_base(frozen_feature_encoder)
        self.correction = SmallCorrection(input_dim=206, max_adjustment=max_adjustment)
        self.distances_enabled = distances_enabled

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(self, problem):
        with torch.no_grad():
            result = dict(self.encoder.encode(problem))
        m = len(problem.candidate_ids)
        if np.shape(problem.D) != (m + 1, m + 1):
            raise ValueError('Expected symbol endpoint 0 and the ordered service candidates')
        result['lookahead_prepared'] = prepare_service_distances(
            problem.D[1:, 1:], problem.C, exit_distances=problem.D[1:, 0])
        result['lookahead_problem_identity'] = problem.identity
        return result

    def features(self, env, encoded):
        with torch.no_grad():
            original = self.encoder.features(env, encoded)
        problem = env.problem
        if encoded.get('lookahead_problem_identity') != problem.identity or 'lookahead_prepared' not in encoded:
            raise ValueError('Prepare the exact immutable problem once with encode() before scoring')
        prepared = encoded['lookahead_prepared']
        extra = prepared_post_service_distances(prepared, env.remaining, env.mask())
        expected_mask = torch.as_tensor(extra['valid_action'], device=original['mask'].device)
        if not torch.equal(original['mask'], expected_mask):
            raise ValueError('Encoder and counterfactual feature masks/order disagree')
        result = dict(original)
        result['x'] = augment_candidate_features(original['x'], extra, self.distances_enabled)
        result['post_service_distances'] = extra
        if 'key' not in original or len(original['key']) != 4:
            raise ValueError('The original problem/prefix/base/D cache identity is required')
        result['key'] = tuple(original['key']) + ('post-service-v1', self.distances_enabled, prepared.source_identity)
        return result

    def logits(self, features):
        return self.correction(features['z0'], features['x'], features['mask'])

    def score(self, env, encoded):
        return self.logits(self.features(env, encoded))

    def actor_only_package(self, base_identity):
        result = self.correction.actor_only_package(base_identity)
        result['distances_enabled'] = self.distances_enabled
        result['feature_schema'] = 'original202 + mean_over10,max_over10,remaining_fraction,unreachable_fraction'
        result['deployment_note'] = 'Requires frozen base and this exact feature adapter; old D1 loader is incompatible'
        return result


def pack_candidate_states(states):
    """Pack one episode while refusing mixed source or feature conditions."""
    if not states:
        return None
    keys = [state['key'] for state in states]
    if any(len(key) != 7 for key in keys):
        raise ValueError('Expected the extended source/feature identity')
    if any(len({key[i] for key in keys}) != 1 for i in (0, 2, 3, 4, 5, 6)):
        raise ValueError('Mixed problem/base/D or nav-on/nav-off feature caches')
    if any(state['x'].shape[1] != 206 for state in states):
        raise ValueError('Use the new candidate schema, not the old 202-dim packer')
    return dict(x=torch.stack([s['x'] for s in states]),
                v=torch.stack([s['v'] for s in states]),
                z0=torch.stack([s['z0'] for s in states]),
                mask=torch.stack([s['mask'] for s in states]), keys=keys)
