"""Pure cost targets for complete service traces; no planner or trainer imports.

The evaluator's physical raw remaining distance/stops are reported separately
from a reward-to-go containing the signed final postprocessor correction.  The
latter is a learning target, not a certificate of physical distance remaining.
"""
from __future__ import annotations

import math
from numbers import Real


def _finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a real number, not bool')
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f'{name} must be representable as a finite float') from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return result


def remaining_cost_targets(raw_leg_lengths, final_J, task_count, penalty=0.75):
    """Return raw physical targets and complete, undiscounted learning targets.

    ``T`` service actions require exactly ``T + 1`` navigation leg lengths:
    one leg to each service stop and one final leg to the specified exit.
    This helper describes a *complete* positive-coverage trace, so ``T <= n``
    and ``n == 0`` iff ``T == 0``.  It never loads a scene, model or dataset.

    Before action ``t`` (zero based), ``raw_remaining_L[t]`` includes the
    eventual exit leg and ``raw_remaining_K[t] = T - t``.  The signed frozen-H
    correction is ``H_gain = J_raw - final_J``.  ``returns[t]`` is
    ``(-(L_t + penalty * K_t) + H_gain) / max(1, task_count)``.

    ``dense_rewards`` use original cost units; ``scaled_dense_rewards`` and
    ``returns`` use the explicitly reported task-count scale.  The last action
    receives the exit cost and the H correction.  For an empty task set there
    is no action or per-action reward; ``no_action_terminal_reward`` holds the
    complete terminal reward instead.  ``complete_return`` is always -final_J.
    """
    if not isinstance(raw_leg_lengths, (list, tuple)) or not raw_leg_lengths:
        raise ValueError('raw_leg_lengths must be a nonempty list or tuple of scalar legs')
    if type(task_count) is not int or task_count < 0:
        raise ValueError('task_count must be a nonnegative Python integer, not bool')
    legs = [_finite_nonnegative(value, f'raw_leg_lengths[{index}]')
            for index, value in enumerate(raw_leg_lengths)]
    final_cost = _finite_nonnegative(final_J, 'final_J')
    stop_penalty = _finite_nonnegative(penalty, 'penalty')
    stops = len(legs) - 1
    if stops > task_count or (task_count == 0) != (stops == 0):
        raise ValueError('a complete positive-coverage trace requires T <= task_count and n=0 iff T=0')

    try:
        scale = float(max(1, task_count))
        route_length = math.fsum(legs)
        raw_cost = math.fsum((route_length, stop_penalty * stops))
        h_gain = raw_cost - final_cost
        remaining_lengths = []
        tail = legs[-1]
        for leg in reversed(legs[:-1]):
            tail = math.fsum((leg, tail))
            remaining_lengths.append(tail)
        remaining_lengths.reverse()
        remaining_stops = list(range(stops, 0, -1))
        rewards = [-math.fsum((leg, stop_penalty)) for leg in legs[:-1]]
        if rewards:
            rewards[-1] = math.fsum((rewards[-1], -legs[-1], h_gain))
        no_action_reward = -final_cost if not stops else 0.0
        returns = [math.fsum((-length, -stop_penalty * count, h_gain)) / scale
                   for length, count in zip(remaining_lengths, remaining_stops)]
        scaled_rewards = [reward / scale for reward in rewards]
        complete_return = math.fsum((*rewards, no_action_reward))
    except (OverflowError, ValueError) as error:
        raise ValueError('cost aggregation exceeds the supported finite-float range') from error

    finite_outputs = [scale, route_length, raw_cost, h_gain, complete_return,
                      *remaining_lengths, *rewards, *returns, *scaled_rewards]
    if not all(math.isfinite(value) for value in finite_outputs):
        raise ValueError('derived cost targets must remain finite')
    if not math.isclose(complete_return, -final_cost, rel_tol=1e-12, abs_tol=1e-10):
        raise ValueError('numerical cancellation prevents complete reward conservation')
    for index, target in enumerate(returns):
        reward_to_go = math.fsum(rewards[index:]) / scale
        if not math.isclose(target, reward_to_go, rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError('numerical cancellation prevents reward-to-go conservation')

    return {
        'schema': 'rc3-conservative-long-horizon-cost-targets-v1',
        'scope': 'Complete trace arithmetic only; no training or physical certification',
        'raw_leg_lengths': legs,
        'raw_route_length': route_length,
        'raw_stop_count': stops,
        'task_count': task_count,
        'penalty': stop_penalty,
        'scale': scale,
        'J_raw': raw_cost,
        'final_J': final_cost,
        'H_gain': h_gain,
        'raw_remaining_L': remaining_lengths,
        'raw_remaining_K': remaining_stops,
        'returns': returns,
        'return_semantics': 'Undiscounted reward-to-go with signed terminal H correction; not physical remaining distance',
        'dense_rewards': rewards,
        'scaled_dense_rewards': scaled_rewards,
        'no_action_terminal_reward': no_action_reward,
        'scaled_no_action_terminal_reward': no_action_reward / scale,
        'complete_return': complete_return,
        'scaled_complete_return': complete_return / scale,
    }
