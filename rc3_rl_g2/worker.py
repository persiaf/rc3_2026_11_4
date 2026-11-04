"""CPU environment workers. No Torch import or model copies in these processes."""
from collections import OrderedDict
from dataclasses import asdict
import copy
import time
import traceback
import json
from pathlib import Path
import numpy as np
from rc3_fast.scenes import make_complex_scene
from rc3_rl_research.types import SearchConfig
from .adapter_g2 import ResearchAdapterG2
from .env_g2 import SearchEnvG2
from .graph_observation import build_policy_observation
from .prepared_cache import prepare_case,validate_worker_snapshot


def worker_main(connection,cases,config,seed,cache_limit=1):
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    rng=np.random.default_rng(seed);cache=OrderedDict();order=[];position=0
    env=None;episode=0;episode_reward=0.;episode_start=0.;prepare_s=0.;feature_s=0.
    preparations=0;cache_reuses=0;active_case=None;cache_before={};prepared_stats={}

    def adapter_for(case):
        nonlocal preparations,cache_reuses,prepared_stats
        key=int(case['seed'])
        if key in cache:
            cache_reuses+=1;cache.move_to_end(key);prepared_stats=dict(memory_hit=True);return cache[key],True
        adapter,prepared_stats=prepare_case(case,Path(__file__).parent/'outputs/prepared_train')
        preparations+=int(not prepared_stats['cache_hit']);cache[key]=adapter
        while len(cache)>cache_limit:cache.popitem(last=False)
        return adapter,False

    def observe_policy():
        nonlocal feature_s
        tick=time.perf_counter();obs=build_policy_observation(env);feature_s+=time.perf_counter()-tick
        return obs

    def reset():
        nonlocal env,order,position,episode,episode_reward,episode_start,prepare_s,feature_s,active_case,cache_before
        if position>=len(order):order=rng.permutation(len(cases)).tolist();position=0
        active_case=cases[order[position]];position+=1;episode+=1
        episode_start=time.perf_counter();adapter,_=adapter_for(active_case)
        prepare_s=time.perf_counter()-episode_start
        env=SearchEnvG2(adapter,SearchConfig(**config),seed+episode,record_transitions=False)
        if env.done:raise ValueError('registered training instance has no actor decisions')
        episode_reward=0.;feature_s=0.;cache_before=copy.deepcopy(adapter.cache_stats)
        return observe_policy()

    try:
        while True:
            command,data=connection.recv()
            if command=='reset':response=reset()
            elif command=='step':
                _,reward,terminated,truncated,info=env.step(int(data));episode_reward+=reward
                if info.get('work_count_incomplete') or env.counters.get('work_count_incomplete'):
                    failure=Path(__file__).parent/'outputs/worker_failures';failure.mkdir(exist_ok=True)
                    (failure/f'{seed}_{time.time_ns()}.json').write_text(json.dumps(dict(case=active_case,
                        problem_id=env.adapter.problem_id,steps=env.steps,reason=info.get('failure_reason'),
                        scope='Unknown internal proposal work: not admitted to PPO rollout; concurrent batch may contain in-flight work'),allow_nan=False),encoding='utf-8')
                    raise RuntimeError('proposal work count is incomplete; aborting learning from this transition')
                summary=None
                if terminated:
                    certificate,finalization=env.finalize()
                    if abs(episode_reward-env.total_reward)>1e-7:raise AssertionError('actor episode reward differs from environment ledger')
                    summary=dict(case=active_case,episode=episode,return_value=env.total_reward,
                        quality_reward=env.total_quality_reward,work_penalty=env.total_work_penalty,
                        J=certificate.cost,anchor_J=env.anchor_cost,work=env.work_used,steps=env.steps,
                        fallback=finalization['fallback_used'],overrun=finalization['work_overrun'],
                        stop_reason=env.stop_reason,complete_s=time.perf_counter()-episode_start,
                        preparation_s=prepare_s,feature_s=feature_s,timings=copy.deepcopy(env.timings),
                        counters=copy.deepcopy(env.counters),cache_before=cache_before,
                        cache_after=copy.deepcopy(env.adapter.cache_stats),
                        total_preparations=preparations,total_cache_reuses=cache_reuses,prepared_stats=copy.deepcopy(prepared_stats),
                        validation=certificate.plan['validation']['valid_model_plan'])
                    nextobs=reset()
                else:nextobs=observe_policy()
                response=dict(observation=nextobs,reward=float(reward),terminated=bool(terminated),
                    summary=summary,action_kind=info['action']['kind'],accepted=info['accepted'],
                    route_changed=info['route_changed'],best_improved=info['best_improved'],
                    work=info['work_units'],failure_reason=info['failure_reason'])
            elif command=='state':
                response=dict(cases=cases,config=config,seed=seed,order=order,position=position,
                    rng_state=copy.deepcopy(rng.bit_generator.state),episode=episode,episode_reward=episode_reward,
                    active_case=active_case,env_state=env.state_dict(),prepare_s=prepare_s,feature_s=feature_s,
                    elapsed_episode_s=time.perf_counter()-episode_start,preparations=preparations,cache_reuses=cache_reuses)
            elif command=='load':
                data=validate_worker_snapshot(data,cases,config,seed)
                adapter,_=adapter_for(data['active_case'])
                restored=SearchEnvG2(adapter,SearchConfig(**config),seed,record_transitions=False)
                restored.load_state_dict(data['env_state'])
                env=restored;active_case=data['active_case'];episode=data['episode'];episode_reward=data['episode_reward']
                order=data['order'];position=data['position'];rng.bit_generator.state=data['rng_state']
                prepare_s=data['prepare_s'];feature_s=data['feature_s'];episode_start=time.perf_counter()-data['elapsed_episode_s']
                preparations=data['preparations'];cache_reuses=data['cache_reuses'];cache_before=copy.deepcopy(adapter.cache_stats)
                response=observe_policy()
            elif command=='close':connection.send(('ok',None));break
            else:raise ValueError('unknown worker command '+command)
            connection.send(('ok',response))
    except EOFError:pass
    except Exception:
        try:connection.send(('error',traceback.format_exc()))
        except Exception:pass
    finally:connection.close()
