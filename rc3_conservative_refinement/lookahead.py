"""Read-only post-service navigation features from supplied exact distances.

There is no navigation solve, task sampling, teacher query or model dependency.
Rows/columns of D must already refer to candidates only (no symbolic node zero).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np


FEATURE_NAMES = (
    'post_service_task_min_distance_mean_div_10m',
    'post_service_task_min_distance_max_div_10m',
    'remaining_after_over_current_remaining',
    'unreachable_after_over_remaining_after',
)


def _distances(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f'{name} must have shape {shape}')
    if not (np.issubdtype(array.dtype, np.integer)
            or np.issubdtype(array.dtype, np.floating)):
        raise ValueError(f'{name} must contain real numeric distances, not bool/object/complex')
    if np.isnan(array).any() or (array < 0).any():
        raise ValueError(f'{name} must be nonnegative and not NaN; +inf denotes unreachable')
    with np.errstate(over='ignore', invalid='ignore'):
        converted = array.astype(np.float64, copy=False)
    if np.any(np.isfinite(array) & ~np.isfinite(converted)):
        raise ValueError(f'{name} finite values must be representable in float64')
    return converted


def _booleans(value, shape, name):
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.dtype(bool):
        raise ValueError(f'{name} must be a boolean array of shape {shape}')
    return array


@dataclass(frozen=True, slots=True)
class PreparedServiceDistances:
    """Immutable snapshots; the matrix is candidate-by-task, never a D substitute."""

    task_distances: np.ndarray
    coverage: np.ndarray
    source_identity: str
    prepared_identity: str
    candidate_count: int
    task_count: int


def _identity(tag, arrays, extra=''):
    digest = hashlib.sha256()
    digest.update(tag.encode('ascii'))
    digest.update(extra.encode('ascii'))
    for name, array in arrays:
        metadata = (name, array.dtype.str, array.shape)
        digest.update(json.dumps(metadata, separators=(',', ':')).encode('ascii'))
        digest.update(array.tobytes(order='C'))
    return 'sha256:' + digest.hexdigest()


def _immutable_copy(array):
    # A bytes-owned buffer cannot be made writable again with setflags().
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)


def prepare_service_distances(D, coverage, exit_distances=None):
    """Once per immutable source, precompute M x N nearest-provider distances.

    D has only candidates (the caller must remove symbolic endpoint node zero).
    Coverage is the caller's same-arm effective boolean table.  Every provider
    remains eligible except those with nonfinite exit distance, if supplied.
    No current action mask is applied to future service candidates.

    Source identity binds canonical float64 D/exit values and boolean coverage,
    including their shapes and whether an exit vector was supplied.  Original
    arrays can subsequently change without altering this detached snapshot.
    Callers must prepare a new source after their input semantics change.
    """
    raw_D = np.asarray(D)
    if raw_D.ndim != 2 or raw_D.shape[0] != raw_D.shape[1]:
        raise ValueError('D must be a square candidate-by-candidate distance matrix')
    candidates = raw_D.shape[0]
    distances = _distances(raw_D, (candidates, candidates), 'D')
    raw_coverage = np.asarray(coverage)
    if raw_coverage.ndim != 2 or raw_coverage.shape[1] != candidates:
        raise ValueError('coverage must have shape (task_count, candidate_count)')
    tasks = raw_coverage.shape[0]
    effective = _booleans(raw_coverage, (tasks, candidates), 'coverage')
    can_exit = np.ones(candidates, dtype=bool)
    source_arrays = [('D', distances), ('coverage', effective)]
    if exit_distances is not None:
        exit_costs = _distances(exit_distances, (candidates,), 'exit_distances')
        can_exit = np.isfinite(exit_costs)
        source_arrays.append(('exit_distances', exit_costs))
    source_identity = _identity('rc3-provider-source-v1', source_arrays,
                                'exit:none' if exit_distances is None else 'exit:provided')
    nearest = np.full((candidates, tasks), np.inf, dtype=np.float64)
    for task in range(tasks):
        providers = effective[task] & can_exit
        if np.any(providers):
            nearest[:, task] = np.min(distances[:, providers], axis=1)
    nearest = _immutable_copy(nearest)
    effective = _immutable_copy(effective)
    prepared_identity = _identity('rc3-provider-prepared-v1',
                                  [('task_distances', nearest), ('coverage', effective)], source_identity)
    return PreparedServiceDistances(nearest, effective, source_identity, prepared_identity, candidates, tasks)


def _validate_prepared(prepared):
    if not isinstance(prepared, PreparedServiceDistances):
        raise ValueError('prepared must come from prepare_service_distances')
    m, n = prepared.candidate_count, prepared.task_count
    if type(m) is not int or type(n) is not int or min(m, n) < 0:
        raise ValueError('invalid prepared source dimensions')
    if not isinstance(prepared.task_distances, np.ndarray) or not isinstance(prepared.coverage, np.ndarray):
        raise ValueError('prepared arrays are missing')
    if (prepared.task_distances.shape != (m, n) or prepared.coverage.shape != (n, m)
            or prepared.task_distances.dtype != np.dtype(np.float64)
            or prepared.coverage.dtype != np.dtype(bool)):
        raise ValueError('prepared source shape/dtype identity mismatch')
    if prepared.task_distances.flags.writeable or prepared.coverage.flags.writeable:
        raise ValueError('prepared arrays must remain read-only')
    for identity in (prepared.source_identity, prepared.prepared_identity):
        if (not isinstance(identity, str) or not identity.startswith('sha256:')
                or len(identity) != 71 or any(ch not in '0123456789abcdef' for ch in identity[7:])):
            raise ValueError('invalid prepared source identity')
    actual = _identity('rc3-provider-prepared-v1',
                       [('task_distances', prepared.task_distances), ('coverage', prepared.coverage)],
                       prepared.source_identity)
    if actual != prepared.prepared_identity:
        raise ValueError('prepared source identity or snapshot content changed')


def prepared_post_service_distances(prepared, remaining, action_mask):
    """O(MN) state aggregation and integrity check; no provider search repeats.

    Remove tasks served by j before taking mean/max of each remaining task's
    precomputed minimum distance.  Unreachable tasks remain present: raw mean
    and max stay +inf, their numeric features become zero, and unreachable
    ratio stays positive.  Finite zero encoding is not reachability evidence.
    Illegal j rows are zero placeholders with valid_action=False and
    all_remaining_reachable=False.  Remaining ratio uses current remaining
    tasks; unreachable ratio uses post-service remaining tasks.
    """
    _validate_prepared(prepared)
    candidates, tasks = prepared.candidate_count, prepared.task_count
    pending = _booleans(remaining, (tasks,), 'remaining')
    legal = _booleans(action_mask, (candidates,), 'action_mask')
    after = pending[np.newaxis, :] & ~prepared.coverage.T & legal[:, np.newaxis]
    after_count = np.count_nonzero(after, axis=1)
    finite = np.isfinite(prepared.task_distances)
    unreachable_count = np.count_nonzero(after & ~finite, axis=1)
    finite_distances = np.where(after & finite, prepared.task_distances, 0.)
    largest_finite = np.max(finite_distances, axis=1, initial=0.)
    divisors = np.where(largest_finite > 0, largest_finite, 1.)
    # Scale before summing, preventing overflow for finite near-limit distances.
    mean_finite = largest_finite * (np.sum(finite_distances / divisors[:, np.newaxis], axis=1)
                                    / np.maximum(1, after_count))
    raw_mean = np.where(unreachable_count > 0, np.inf, mean_finite)
    raw_max = np.where(unreachable_count > 0, np.inf, largest_finite)
    all_reachable = legal & (unreachable_count == 0)
    current_remaining = int(np.count_nonzero(pending))
    features = np.column_stack((
        np.where(np.isfinite(raw_mean), raw_mean, 0.) / 10.,
        np.where(np.isfinite(raw_max), raw_max, 0.) / 10.,
        after_count / max(1, current_remaining),
        unreachable_count / np.maximum(1, after_count),
    ))
    return {
        'raw_mean': raw_mean,
        'raw_max': raw_max,
        'remaining_after_count': after_count,
        'unreachable_remaining_count': unreachable_count,
        'valid_action': legal.copy(),
        'all_remaining_reachable': all_reachable,
        'finite_features': features,
        'feature_names': FEATURE_NAMES,
    }


def post_service_distances(D, coverage, remaining, action_mask, exit_distances=None):
    """Compatibility wrapper; persistent inference should prepare once instead."""
    prepared = prepare_service_distances(D, coverage, exit_distances=exit_distances)
    return prepared_post_service_distances(prepared, remaining, action_mask)
