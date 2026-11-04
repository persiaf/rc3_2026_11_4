"""Shared-menu G1 search runtime. No neural network is trained or loaded."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from .policies import make_policy
from .rl_env import SearchEnv, load_environment, replay_transitions
from .types import SCHEMA, SearchConfig


def run(adapter, method='greedy', config=SearchConfig(), seed=0, *, policy=None,
        time_limit_s=None, output_path=None, started_at=None, preparation_s=0.0):
    """Return a certified plan and evidence, allowing bounded worse current.

    For full scene-to-plan timing use ``run_scene``; ``run`` receives an already
    prepared adapter and labels that scope explicitly. A deadline is soft:
    selected atomic repairs and mandatory final validation may cross it.
    """
    if time_limit_s is not None and (not math.isfinite(time_limit_s) or time_limit_s < 0):
        raise ValueError('time_limit_s must be finite and nonnegative')
    started = time.perf_counter() if started_at is None else started_at
    deadline = math.inf if time_limit_s is None else started+time_limit_s
    selector = make_policy(method) if policy is None else policy
    env = SearchEnv(adapter, config, seed)
    initial_state = env.state_dict()
    selection_s, failures, selections = 0.0, [], []
    policy_initial_state = selector.state_dict() if hasattr(selector, 'state_dict') else None
    while not env.terminated:
        if time.perf_counter() >= deadline:
            env.terminate('external_deadline')
            break
        try:
            observation = env.observe()
            if time.perf_counter() >= deadline:
                env.terminate('external_deadline')
                break
            tick = time.perf_counter()
            rng_before = copy.deepcopy(env.rng.bit_generator.state)
            action = selector.select(observation, env.rng)
            selection_s += time.perf_counter()-tick
            selections.append(dict(observation=copy.deepcopy(observation), action_id=action,
                                   rng_before=rng_before,
                                   rng_after=copy.deepcopy(env.rng.bit_generator.state),
                                   transition_index=len(env.transitions), executed=False))
            if time.perf_counter() >= deadline:
                env.terminate('external_deadline')
                break
            _, _, _, _, info = env.step(action)
            selections[-1]['executed'] = True
            if hasattr(selector, 'update'):
                selector.update(info)
        except Exception as error:
            failures.append(f'{type(error).__name__}: {error}')
            env.terminate('runtime_error')
            break
    certificate, finalization = env.finalize()
    elapsed = time.perf_counter()-started
    policy_state = selector.state_dict() if hasattr(selector, 'state_dict') else None
    diagnostics = dict(schema=SCHEMA, status='complete', method=method,
                       algorithm_scope='G1 shared-menu non-learning reference; not trained PPO',
                       problem_id=adapter.problem_id, config=asdict(config), seed=seed,
                       anchor_cost=env.anchor_cost, current_cost=env.current_cost,
                       best_cost=env.best_cost, returned_cost=certificate.cost,
                       cost_nonincreasing=certificate.cost <= env.anchor_cost+1e-8,
                       finalization=finalization, failures=failures,
                       stop_reason=env.stop_reason, steps=env.steps, work_used=env.work_used,
                       remaining_work=env.remaining_work, counters=env.counters,
                       total_reward=env.total_reward, quality_reward=env.total_quality_reward,
                       work_penalty=env.total_work_penalty, shaping_reward=env.total_shaping_reward,
                       timing=dict(**env.timings, selection_s=selection_s,
                                   preparation_s=preparation_s, complete_s=elapsed,
                                   scope='scene_to_plan' if started_at is not None else 'prepared_adapter_to_plan'),
                       time_limit_s=time_limit_s,
                       time_overrun_s=max(0.0, elapsed-time_limit_s) if time_limit_s is not None else 0.0,
                       policy_initial_state=policy_initial_state, policy_state=policy_state,
                       guarantee_scope='Certified under supplied static discrete proxy/common constraints; no physical IK or hard real-time guarantee.')
    evidence = dict(schema=SCHEMA, adapter=adapter.payload(), initial_state=initial_state,
                    state=env.state_dict(), policy=policy_state, diagnostics=diagnostics,
                    plan=certificate.plan, selections=selections)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return certificate.plan, diagnostics


def run_scene(scene_or_bundle, *, external=False, penalty=.75, **kwargs):
    from .adapter import ResearchAdapter
    started = time.perf_counter()
    adapter = (ResearchAdapter.from_external(scene_or_bundle, penalty=penalty) if external
               else ResearchAdapter.from_scene(scene_or_bundle, penalty=penalty))
    return run(adapter, started_at=started, preparation_s=time.perf_counter()-started, **kwargs)


def replay_file(path):
    from .adapter import ResearchAdapter
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    state = payload['state']
    adapter = ResearchAdapter.from_payload(payload['adapter'])
    env = SearchEnv(adapter, SearchConfig(**state['config']), state['seed'])
    if 'initial_state' in payload:
        env.load_state_dict(payload['initial_state'])
    replay_transitions(env, state['transitions'])
    for event in state['events']:
        if event['event'] == 'runtime_termination':
            pending = event.get('observation_before_termination')
            if pending is not None and env.observe() != pending:
                raise ValueError('runtime termination pending menu differs from saved observation')
            if pending is None and event.get('menu_work_units', 0) > 0:
                # Early G1 artifacts recorded the pending charge but not its
                # observation. Rebuild from the saved prefix and verify charge.
                if env.observe()['pending_menu_work'] != event['menu_work_units']:
                    raise ValueError('runtime termination pending menu charge mismatch')
            env.rng.bit_generator.state = event['rng_state']
            reward, actual = env.terminate(event['reason'])
            if abs(reward-event['reward']) > 1e-8 or actual['finalization'] != event['finalization']:
                raise ValueError('runtime finalization replay mismatch')
    if (env.current != state['current'] or env.best != state['best'] or
            env.work_used != state['work_used'] or abs(env.total_reward-state['total_reward']) > 1e-8 or
            env.history != state['history'] or env.terminated != state['terminated']):
        raise ValueError('replayed final state differs from saved state')
    for key in ('total_quality_reward', 'total_work_penalty', 'total_shaping_reward'):
        if abs(getattr(env, key)-state[key]) > 1e-8:
            raise ValueError(f'replayed {key} differs from saved state')
    if env.rng.bit_generator.state != state['rng_state']:
        raise ValueError('replayed final RNG differs from saved state')
    if env.finalization != state['finalization']:
        raise ValueError('replayed finalization differs from saved state')
    returned = env.returned_certificate
    if returned is not None:
        saved = state['returned_certificate']
        if saved is None or (saved['route'] != returned.route or saved['cost'] != returned.cost or
                             saved['plan'] != returned.plan):
            raise ValueError('saved returned certificate differs from replayed certificate')
        if 'plan' in payload and payload['plan'] != returned.plan:
            raise ValueError('saved exported plan differs from replayed plan')
    policy_verified = False
    if 'selections' in payload and payload.get('policy') is not None:
        import numpy as np
        policy = make_policy(payload['diagnostics']['method'])
        policy.load_state_dict(payload['diagnostics']['policy_initial_state'])
        rng = np.random.default_rng()
        expected_rng = payload['initial_state']['rng_state']
        executed_indices = []
        for selection in payload['selections']:
            idx = selection['transition_index']
            if selection['rng_before'] != expected_rng:
                raise ValueError('policy RNG chain is discontinuous')
            if selection['executed']:
                executed_indices.append(idx)
                if selection['observation'] != state['transitions'][idx]['observation']:
                    raise ValueError('policy selection did not use the recorded transition observation')
            rng.bit_generator.state = selection['rng_before']
            action = policy.select(selection['observation'], rng)
            if action != selection['action_id'] or rng.bit_generator.state != selection['rng_after']:
                raise ValueError('policy selection/RNG replay mismatch')
            expected_rng = selection['rng_after']
            if selection['executed']:
                if action != state['transitions'][idx]['action_id']:
                    raise ValueError('policy selected action differs from executed action')
                if selection['rng_after'] != state['transitions'][idx]['rng_before']:
                    raise ValueError('policy and environment RNG records disagree')
                policy.update(state['transitions'][idx]['info'])
        if executed_indices != list(range(len(state['transitions']))):
            raise ValueError('policy trace does not cover each executed transition once')
        if expected_rng != state['rng_state']:
            raise ValueError('policy trace final RNG differs from saved state')
        if policy.state_dict() != payload['policy']:
            raise ValueError('saved policy state differs from replayed policy state')
        policy_verified = True
    return dict(schema=SCHEMA, replay_valid=True, transitions=len(state['transitions']),
                work_used=env.work_used, total_reward=env.total_reward,
                finalization=env.finalization, exported_plan_verified=returned is not None,
                policy_state_verified=policy_verified)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--method', choices=['greedy', 'random', 'alns'], default='greedy')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--horizon', type=int, default=16)
    parser.add_argument('--work-budget', type=int, default=2_000_000)
    parser.add_argument('--epsilon', type=float, default=.02)
    parser.add_argument('--time-limit-ms', type=float)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'outputs'/'runtime.json')
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay_file(args.replay), ensure_ascii=False))
        return
    if args.scene is None:
        parser.error('--scene is required except when --replay is supplied')
    source = json.loads(args.scene.read_text(encoding='utf-8-sig'))
    if not args.external and ('planning' in source and 'audit' in source):
        parser.error('external constraints require --external; do not discard F/A')
    if not args.external:
        source = source.get('scene', source)
    _, diagnostics = run_scene(source, external=args.external, method=args.method,
                              seed=args.seed, config=SearchConfig(epsilon=args.epsilon,
                              horizon=args.horizon, work_budget=args.work_budget),
                              time_limit_s=None if args.time_limit_ms is None else args.time_limit_ms/1000,
                              output_path=args.output)
    print(json.dumps(dict(status=diagnostics['status'], method=args.method,
                         returned_cost=diagnostics['returned_cost'],
                         work_used=diagnostics['work_used'], output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
