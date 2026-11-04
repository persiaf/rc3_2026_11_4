"""Single old development case probe; correctness check, not a speed benchmark."""
import copy
import hashlib
import json
from pathlib import Path
import time
from threadpoolctl import threadpool_limits
from rc3_rl_g2.adapter_g2 import ResearchAdapterG2
from rc3_rl_g2.env_g2 import SearchEnvG2
from rc3_rl_g2.baselines import make_policy
from rc3_rl_research.types import SearchConfig


def execute(scene,method,cached):
    start=time.perf_counter();adapter=ResearchAdapterG2.from_scene(scene)
    preparation=time.perf_counter()-start
    adapter.certificate_cache_enabled=cached
    adapter.cache_stats={key:0. if key.endswith('_s') else 0 for key in adapter.cache_stats}
    env=SearchEnvG2(adapter,SearchConfig(),record_transitions=False,seed=0)
    policy=make_policy(method);actions=[];tick=time.perf_counter()
    while not env.done:
        obs=env.observe();a=policy.select(obs,env.rng)
        actions.append(obs['actions'][a])
        _,_,_,_,info=env.step(a);policy.update(info)
    search=time.perf_counter()-tick;cert,final=env.finalize()
    return dict(method=method,cache_enabled=cached,preparation_s=preparation,search_s=search,
        total_s=time.perf_counter()-start,steps=env.steps,proposal_calls=env.counters['proposal_calls'],
        work_used=env.work_used,total_reward=env.total_reward,J=cert.cost,route=cert.route,
        actions=actions,timings=env.timings,cache_stats=adapter.cache_stats,finalization=final,
        input_unchanged=adapter.input_identity()==adapter.problem_id)


def main():
    root=Path(__file__).resolve().parents[2]
    scene=json.loads((root/'rc3_rl_research/outputs/g0/case_01/scene.json').read_text(encoding='utf-8'))
    with threadpool_limits(limits=1):
        rows=[execute(scene,method,cached) for method in ('detour_memory','ils','stop') for cached in (False,True)]
    equal=[]
    for i in range(0,len(rows),2):
        left,right=rows[i:i+2]
        equal.append(dict(method=left['method'],equivalent=all(left[k]==right[k] for k in
            ('steps','proposal_calls','work_used','total_reward','J','route','actions','finalization'))))
    source_hashes={name:hashlib.sha256((root/'rc3_rl_g2'/name).read_bytes()).hexdigest()
                   for name in ('env_g2.py','action_memory.py','adapter_g2.py','baselines.py')}
    report=dict(scope='One previously inspected G01 case_01 used only for implementation/cache equivalence and exploratory throughput; not G2 validation, new scenes, or statistical deployment-speed evidence.',
                source_hashes=source_hashes,
                same_machine_serial=True,threadpool_limit=1,rows=rows,checks=equal,
                all_passed=all(x['equivalent'] for x in equal))
    out=root/'rc3_rl_g2/outputs/environment_probe.json';out.parent.mkdir(exist_ok=True,parents=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(all_passed=report['all_passed'],timing=[dict(method=r['method'],cache=r['cache_enabled'],steps=r['steps'],total_s=r['total_s'],cache_hits=r['cache_stats']['hits']) for r in rows])))


if __name__=='__main__':main()
