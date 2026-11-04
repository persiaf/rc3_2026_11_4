"""G2 semantics with G21-only persistence and explicit committed-step ledgers."""
from collections import OrderedDict
import copy
import json
import sys
from pathlib import Path
import time
import traceback
sys.dont_write_bytecode=True

import numpy as np


ROOT = Path(__file__).resolve().parent


def root_stop_reference(pending_menu_work, config):
    """Exact root STOP reward under frozen zero-potential G2 proxy semantics.

    Uses the already built menu and certified anchor, with no counterfactual
    environment step. An actual first-action STOP is checked against this value.
    """
    if config['potential_weight'] != 0 or config['work_scale'] <= 0:
        raise ValueError('Analytic STOP reference requires frozen zero-potential semantics')
    if pending_menu_work < 0:
        raise ValueError('Negative menu work')
    return -float(config['lambda_work']) * pending_menu_work / config['work_scale']


def worker_main(connection, cases, config, seed, worker_id, run_dir, cache_limit=1):
    from threadpoolctl import threadpool_limits
    from rc3_rl_research.types import SearchConfig
    from rc3_rl_g2.env_g2 import SearchEnvG2
    from rc3_rl_g2.graph_observation import build_policy_observation
    from rc3_rl_g2.prepared_cache import prepare_case
    threadpool_limits(1)
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Worker run directory must stay in G21 outputs')
    cache_dir = ROOT / 'outputs/prepared_train'
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = (run_dir / f'worker_{worker_id:02d}_committed.jsonl').open('x', encoding='utf-8')
    rng = np.random.default_rng(seed)
    adapters = OrderedDict()
    order = []; position = 0; episode = 0; committed = 0
    env = None; active_case = None; episode_reward = 0.; episode_start = 0.
    prepare_s = 0.; feature_s = 0.; stop_reference = None; root_menu_work = None

    def observe():
        nonlocal feature_s
        tick = time.perf_counter()
        result = build_policy_observation(env)
        feature_s += time.perf_counter() - tick
        return result

    def reset():
        nonlocal env, active_case, order, position, episode, episode_reward
        nonlocal episode_start, prepare_s, feature_s, stop_reference, root_menu_work
        if position >= len(order):
            order = rng.permutation(len(cases)).tolist(); position = 0
        active_case = cases[order[position]]; position += 1; episode += 1
        episode_start = time.perf_counter()
        key = json.dumps(active_case, sort_keys=True)
        if key in adapters:
            adapters.move_to_end(key); adapter = adapters[key]
        else:
            adapter, _ = prepare_case(active_case, cache_dir)
            adapters[key] = adapter
            while len(adapters) > cache_limit:
                adapters.popitem(last=False)
        prepare_s = time.perf_counter() - episode_start
        env = SearchEnvG2(adapter, SearchConfig(**config), seed + episode, record_transitions=False)
        if env.done:
            raise ValueError('Registered training case has no actor decisions')
        episode_reward = 0.; feature_s = 0.
        observation = observe()
        root_menu_work = int(env._menu.work_units)
        stop_reference = root_stop_reference(root_menu_work, config)
        return observation

    try:
        while True:
            command, data = connection.recv()
            if command == 'reset':
                response = reset()
            elif command == 'step':
                proposals_before = env.counters['proposal_calls']
                observation_key = env._menu.context_key
                _, reward, terminated, truncated, info = env.step(int(data))
                committed += 1; episode_reward += reward
                proposal_delta = env.counters['proposal_calls'] - proposals_before
                record = dict(worker=worker_id, worker_committed=committed, episode=episode,
                    case=active_case, problem_id=env.adapter.problem_id,
                    context_key=observation_key, episode_step=env.steps, action_id=int(data),
                    action=info['action'], reward=float(reward), terminated=bool(terminated),
                    proposal_calls=int(proposal_delta), work=int(info['work_units']),
                    root_stop_reference=stop_reference, accepted=bool(info['accepted']),
                    route_changed=bool(info['route_changed']), best_improved=bool(info['best_improved']),
                    work_count_incomplete=bool(env.counters.get('work_count_incomplete')))
                ledger.write(json.dumps(record, allow_nan=False) + '\n'); ledger.flush()
                if record['work_count_incomplete']:
                    raise RuntimeError('Incomplete proposal work; committed step logged but excluded from PPO')
                if env.steps == 1 and info['action']['kind'] == 'stop' and abs(reward-stop_reference) > 1e-7:
                    raise AssertionError('Analytical root STOP reference disagrees with actual STOP')
                summary = None
                if terminated:
                    certificate, finalization = env.finalize()
                    if abs(episode_reward-env.total_reward) > 1e-7:
                        raise AssertionError('Episode reward ledger mismatch')
                    summary = dict(worker=worker_id, case=active_case, episode=episode,
                        return_value=env.total_reward, root_stop_return=stop_reference,
                        return_minus_root_stop=env.total_reward-stop_reference,
                        stop_reference_source='Analytical frozen root-menu cost, no extra environment action',
                        root_menu_work=root_menu_work, J=certificate.cost, anchor_J=env.anchor_cost,
                        work=env.work_used, steps=env.steps,
                        actual_proposal_calls=env.counters['proposal_calls'],
                        has_at_least_two_proposals=env.counters['proposal_calls'] >= 2,
                        quality_reward=env.total_quality_reward, work_penalty=env.total_work_penalty,
                        fallback=finalization['fallback_used'], stop_reason=env.stop_reason,
                        episode_wall_s_including_training_waits=time.perf_counter()-episode_start, preparation_s=prepare_s,
                        feature_s=feature_s, timings=copy.deepcopy(env.timings),
                        counters=copy.deepcopy(env.counters),
                        validation=certificate.plan['validation']['valid_model_plan'])
                    nextobs = reset()
                else:
                    nextobs = observe()
                response = dict(observation=nextobs, reward=float(reward), terminated=bool(terminated),
                    summary=summary, action_kind=info['action']['kind'],
                    route_changed=info['route_changed'], proposal_calls=proposal_delta,
                    transition=record)
            elif command == 'state':
                response = dict(cases=cases, config=config, seed=seed, worker_id=worker_id,
                    order=order, position=position, rng_state=copy.deepcopy(rng.bit_generator.state),
                    episode=episode, committed=committed, episode_reward=episode_reward,
                    active_case=active_case, root_stop_reference=stop_reference,
                    env_state=env.state_dict())
            elif command == 'close':
                connection.send(('ok', None)); break
            else:
                raise ValueError('Unsupported pilot worker command')
            connection.send(('ok', response))
    except EOFError:
        pass
    except Exception:
        failure = dict(worker=worker_id, committed=committed, case=active_case,
                       traceback=traceback.format_exc(), scope='Concurrent worker steps may be in flight; inspect all worker ledgers; no automatic resume')
        (run_dir / f'worker_{worker_id:02d}_failure.json').write_text(json.dumps(failure), 'utf-8')
        try:
            connection.send(('error', failure['traceback']))
        except Exception:
            pass
    finally:
        ledger.close(); connection.close()
