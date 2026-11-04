"""Read-only numeric/model-contract audit of the frozen initial G2 teacher.

Does not instantiate an environment, call a repair oracle, or train a model.
Pickle is accepted only after matching the existing teacher summary hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


SCALES = np.asarray([1., 1., 1., 1., 4., 8., 80., 10., 10., 10., 1., 7., 7.,
                     8., 4., 4., 8., 1., 16., 1., 1., 16., 2., 2., 1., 1.,
                     1., 3., 3., 3.])
MODEL_ARRAY_KEYS = ('features', 'mask', 'task_features', 'candidate_features',
                    'task_action', 'candidate_action', 'edge_task',
                    'edge_candidate', 'nav_source', 'nav_target', 'nav_distance')


def digest_arrays(observation):
    """Exact ordered tensor equality; NOT a graph-isomorphism/learnability test."""
    h = hashlib.sha256()
    for key in MODEL_ARRAY_KEYS:
        x = np.asarray(observation[key])
        h.update(key.encode())
        h.update(str((x.dtype.str, x.shape)).encode())
        h.update(x.tobytes())
    return h.hexdigest()


def ranges(x):
    x = np.asarray(x)
    return dict(shape=list(x.shape), all_finite=bool(np.isfinite(x).all()),
                abs_quantiles=dict(zip(('p50', 'p95', 'p99', 'max'),
                    np.quantile(np.abs(x), [.5, .95, .99, 1.]).tolist())),
                per_column_min=x.min(axis=0).tolist(),
                per_column_max=x.max(axis=0).tolist())


def run(workspace):
    source = workspace / 'rc3_rl_g2/outputs/teacher_initial/teacher.pkl'
    summary = json.loads(source.with_name('summary.json').read_text('utf-8'))
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != summary['teacher_sha256']:
        raise ValueError('Frozen teacher does not match its recorded hash')
    data = pickle.loads(raw)
    samples = data['samples']
    roots = [s for s in samples if not s['path']]
    checks = []
    for s in samples:
        o = s['policy_observation']
        legal = np.asarray(o['mask'], dtype=bool)
        evidence = np.asarray(s['evidence_mask'], dtype=bool)
        ties = np.asarray(s['ties'], dtype=bool)
        q = np.asarray(s['q_values'])
        stop = s['stop_id']
        check = dict(scene_id=s['scene_id'], path=s['path'])
        check['valid'] = bool(
            np.array_equal(legal, s['legality_mask'])
            and evidence.shape == legal.shape == ties.shape == q.shape
            and not np.any(evidence & ~legal)
            and evidence[stop] and o['actions'][stop]['kind'] == 'stop'
            and ties.any() and not np.any(ties & ~evidence)
            and np.isfinite(q[evidence]).all()
            and np.array_equal(ties, evidence &
                (q >= q[evidence].max() - s['tie_tolerance']))
            and np.allclose(np.asarray(s['advantages_vs_stop'])[evidence],
                            q[evidence] - q[stop], atol=1e-12, rtol=0)
            and not np.any(np.asarray(o['task_action']) == stop)
            and not np.any(np.asarray(o['candidate_action']) == stop))
        checks.append(check)
    stop_x = np.stack([s['policy_observation']['features'][s['stop_id']] for s in roots])
    varying = np.flatnonzero(np.ptp(stop_x, axis=0) > 0).tolist()
    hashes = [digest_arrays(s['policy_observation']) for s in roots]
    positive = 0
    zero_or_negative = 0
    for s in roots:
        e = np.asarray(s['evidence_mask'], dtype=bool).copy()
        e[s['stop_id']] = False
        delta = float(np.asarray(s['advantages_vs_stop'])[e].max())
        positive += delta > s['tie_tolerance']
        zero_or_negative += delta <= s['tie_tolerance']
    arrays = {}
    for key in ('features', 'task_features', 'candidate_features'):
        x = np.concatenate([s['policy_observation'][key] for s in roots], axis=0)
        arrays[key + ('_after_model_scaling' if key == 'features' else '')] = ranges(
            x / SCALES if key == 'features' else x)
    files = ('rc3_rl_g2/actor_critic.py', 'rc3_rl_g2/graph_observation.py',
             'rc3_rl_g2/train_ppo.py', 'rc3_rl_g2/teacher_dataset.py',
             'rc3_rl_research/action_builder.py')
    return dict(schema='rc3-g21-readonly-model-input-audit-v1',
        scope='Existing initial teacher only; zero environment queries and zero training updates',
        teacher_sha256=digest, sample_count=len(samples), root_count=len(roots),
        label_and_empty_stop_graph_checks_passed=sum(c['valid'] for c in checks),
        failed_checks=[c for c in checks if not c['valid']],
        roots_with_known_positive_continue=positive,
        roots_without_known_positive_continue=zero_or_negative,
        unique_exact_ordered_model_input_count=len(set(hashes)),
        stop_features_unique_count=len(np.unique(stop_x, axis=0)),
        stop_features_varying_column_indices=varying,
        root_stop_raw_feature_ranges=ranges(stop_x), root_feature_ranges=arrays,
        interpretation_limits=[
            'Distinct ordered arrays do not prove graph-isomorphism separation or model learnability.',
            'Feature magnitude is not an activation saturation measurement.',
            'STOP has no graph nodes; its logit uses its own action/context features only.',
            'Unknown legal actions stay unlabelled and are excluded from BC normalization.',
            'This audit does not regenerate labels or verify physical executability.'],
        source_sha256={p: hashlib.sha256((workspace / p).read_bytes()).hexdigest() for p in files})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, default=Path(__file__).with_name('model_input_audit.json'))
    args = parser.parse_args()
    result = run(args.workspace)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')
    print(json.dumps({k: result[k] for k in ('sample_count', 'root_count',
        'label_and_empty_stop_graph_checks_passed', 'failed_checks',
        'unique_exact_ordered_model_input_count', 'stop_features_unique_count',
        'stop_features_varying_column_indices')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
