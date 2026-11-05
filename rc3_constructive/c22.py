"""C2.2 matched warm-start raw versus complete-post objective experiment."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import csv
import gzip
import json
import math
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from rc3_fast import solver
from rc3_fast.scenes import make_complex_scene
from rc3_baseline.validator import validate
from . import baselines
from .problem import ROOT, Cache, resample_tasks, write, digest
from .env import batch, unpack_routes, Env
from .model import Policy

OUT = ROOT/'c22'
BASE = '44aee8f508d6c39a1d5f44d631e8e519da4f8c21'
WARM = ROOT/'outputs/checkpoints/update0128.pt'
MODES = ('regular', 'mixed', 'blocked')


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def append(p, value):
    with Path(p).open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, allow_nan=False)+'\n'); f.flush()


def settings():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False


def load(device):
    m=Policy().to(device)
    m.load_state_dict(torch.load(WARM,map_location=device,weights_only=False)['state'])
    return m


def sync(device):
    if device == 'cuda': torch.cuda.synchronize()


def sources():
    paths=[Path(__file__), ROOT/'model.py',ROOT/'env.py',ROOT/'problem.py',ROOT/'baselines.py',
           ROOT/'protocol.yaml',ROOT/'outputs/training_summary.json']
    paths+=list((ROOT/'outputs/checkpoints').glob('*.pt'))
    paths+=[ROOT.parent/p for p in ('rc3_fast/solver.py','rc3_fast/scenes.py','rc3_fast/planner.py',
        'rc3_baseline/baseline.py','rc3_baseline/validator.py','rc3_v4/runtime.py')]
    return {p.relative_to(ROOT.parent).as_posix():digest(p) for p in paths}


def check():
    p=read(OUT/'protocol.json')
    assert all(digest(ROOT.parent/n)==h for n,h in p['source_sha256'].items())
    assert digest(OUT/'instances.jsonl') == p['instances_sha256']
    assert digest(OUT/'mechanism_seeds.json') == p['mechanism_seeds_sha256']
    return p


def register():
    if (OUT/'protocol.json').exists(): raise FileExistsError('Already registered')
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
    OUT.mkdir(exist_ok=True)
    instances=[dict(ordinal=i,map_seed=7100000000+i%24,task_seed=7800000000+i,
                    mode=MODES[(i%24)%3]) for i in range(4096)]
    with (OUT/'instances.jsonl').open('x',encoding='utf-8') as f:
        for r in instances:f.write(json.dumps(r)+'\n')
    seeds=[dict(case=i,map_seed=7900000000+i,task_seed=8000000000+i,mode=MODES[i%3]) for i in range(48)]
    write(OUT/'mechanism_seeds.json',seeds)
    write(OUT/'protocol.json',dict(schema='rc3-c22-objective-consistency-v1',date='2026-11-05',base_commit=BASE,
        primary_claim='Final-cost training improves final greedy cost versus matched raw-cost training',
        supporting_claim='B improves the common warm-start model, rather than merely avoiding A deterioration',
        warm_start='outputs/checkpoints/update0128.pt',optimizer='Fresh independent Adam for both; lr=0.0001, default betas/eps/weight_decay, clip=1.0; summed logp / constant 100',
        rollout_baseline='Original update0128 fixed, no replacement; one shared greedy calculation per instance, raw and final costs cached for both uses',
        updates_per_arm=128,batch_size=32,distinct_training_instances=4096,policy_samples_total=8192,
        sampling_seed=11422026,sampling_rule='Independent of task RNG: fork torch RNG per arm/update, seed=sampling_seed+update; same per-update seed for A/B; no stochastic dropout; instance flow prewritten',
        A_cost='Raw closed-route J',B_cost='J of complete H(raw), both sample and baseline',
        H='normalize(joint_search(copy(H0(raw)), C, D, passes=2)); H0=original baselines.postprocess, including both normalize/60-pass two_opt rounds and final normalize',
        training_H_calls=8192,shared_baseline_greedy_routes=4096,baseline_trajectory_uses=8192,
        endpoint='Exactly update128 per arm; no checkpoint search, intermediate scene evaluation, rollout updates or hyperparameter search',
        training_distribution='Original 24-map 10x10 training bank, new explicit task seeds; maps reused, 4096 distinct task/map pairs shared by arms',
        evaluation_methods=['warm','A','B','prior','fixed3','detour16'],evaluation_cases=48,evaluation_plans=288,
        evaluation_rule='New map/task seeds; modes 16 each. Evaluate fixed endpoints once on CPU. All four constructors use same complete H; strong baselines retain full pipelines. Raw numeric audit, final export geometry certificate, separate saved-plan scalar validation.',
        reserved_60_test_queries=0,C3_started=False,default_planner='Fixed3/Detour',
        calibration=dict(count=32,map_seed_start=8100000000,task_seed_start=8200000000,
            iterations=2,greedy_forward_trajectories=192,greedy_backward_trajectories=128,H_calls=128,
            sampled_trajectories=0,optimizer_updates=0,
            rule='Before optimizer lock min(7200,max(1800,ceil(128*(max_pair_probe_s+all_32_scene_preparation_s)*2+600))) seconds for entire paired training stage; check between paired updates; abort both on invalid/OOM/nonfinite; no quality stopping'),
        mechanism_screen=dict(clear_signal='mean(g)>=0.5 and lower95(g)>0 and lower95(h)>0 and all feasible',
            unsupported='upper95(g)<0.5 or upper95(h)<0 or feasibility failure',
            inconclusive='All other outcomes; stop this round without more updates/seeds',
            g='100*(J_A-J_B)/J_A',h='100*(J_warm-J_B)/J_warm',
            uncertainty='paired scene bootstrap 10000, fixed seed114, two-sided95; one matched training seed; two criteria jointly required, no post-hoc endpoint selection'),
        deployment_gates_unchanged='Detour16 mean paired relative degradation one-sided95 upper<=0.5%; complete median time ratio<=0.7 and p95<=1.0; all feasible. NOT independently tested here.',
        scope='Mechanism screening, not C3, deployment validation, cross-seed stability, cross-scale or true IK; after use 48 scenes become known development data. State for theory includes entire route prefix, network summary unchanged.',
        instances_sha256=digest(OUT/'instances.jsonl'),mechanism_seeds_sha256=digest(OUT/'mechanism_seeds.json'),source_sha256=sources()))
    print('Registered: A/B 128x32; 48 new mechanism scenes; no retained test queries',flush=True)


def full_H(item, raw, counter):
    counter['H_started']+=1;tick=time.perf_counter()
    h0=baselines.postprocess(item,raw)
    final=baselines.normalize(item,solver.joint_search(list(h0),item.C,item.D,passes=2))
    counter['H_s']+=time.perf_counter()-tick;counter['H_completed']+=1
    return final


def counts(): return dict(H_started=0,H_completed=0,H_s=0.)


def metric(item,route, raw=False):
    assert route[0]==route[-1]==0
    nodes=np.asarray(route[1:-1],dtype=int)
    assert len(nodes)==len(set(nodes.tolist())) and ((nodes>=1)&(nodes<=item.C.shape[1])).all()
    assert item.C[:,nodes-1].any(1).all()
    if raw:
        env=Env(item);env.replay((nodes-1).tolist());assert env.route==route
    L=float(sum(item.D[a,b] for a,b in zip(route,route[1:])))
    assert np.isfinite(L)
    return dict(J=L+.75*len(nodes),L=L,K=len(nodes))


def calibrate():
    p=check();target=OUT/'wall_protocol.json'
    if target.exists() or (OUT/'calibration_started.json').exists():raise FileExistsError('Calibration already started; no implicit replay')
    write(OUT/'calibration_started.json',dict(protocol_sha256=digest(OUT/'protocol.json')))
    c=counts();device='cuda' if torch.cuda.is_available() else 'cpu';m=load(device);cache=Cache(32)
    spec=p['calibration'];tick=time.perf_counter()
    items=[cache.prepare(resample_tasks(make_complex_scene(10,10,spec['map_seed_start']+i,MODES[i%3]),spec['task_seed_start']+i)) for i in range(32)]
    prep=time.perf_counter()-tick;data=batch(items,device);durations=[]
    for _ in range(2):
        sync(device);tick=time.perf_counter()
        with torch.no_grad():
            _,_,actions,lengths=m(data)
        baseline=unpack_routes(actions,lengths)
        for item,route in zip(items,baseline):full_H(item,route,c)
        for objective in ('A','B'):
            m.zero_grad(set_to_none=True);raw,logp,actions,lengths=m(data)
            routes=unpack_routes(actions,lengths)
            final=[metric(item,full_H(item,route,c))['J'] for item,route in zip(items,routes)] if objective=='B' else raw.detach().cpu().tolist()
            (torch.tensor(final,device=device).float()/100*logp).mean().backward()
            assert all(x.grad is None or torch.isfinite(x.grad).all() for x in m.parameters())
        sync(device);durations.append(time.perf_counter()-tick)
    cap=int(min(7200,max(1800,math.ceil(128*(max(durations)+prep)*2+600))))
    assert c['H_completed']==128
    write(target,dict(protocol_sha256=digest(OUT/'protocol.json'),device=device,pair_probe_s=durations,
        preparation_32_s=prep,paired_training_wall_limit_s=cap,calibration_counters=c,
        greedy_forward_trajectories=192,greedy_backward_trajectories=128,optimizer_updates=0,
        sampled_trajectories=0,scope='Greedy throughput probes only; gradients discarded, no optimizer created or step; fixed predeclared formula, no quality inspection',
        hardware=dict(cpu=platform.processor(),gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
            python=platform.python_version(),torch=torch.__version__,numpy=np.__version__)))
    print(json.dumps(read(target)),flush=True)


def save_checkpoint(path,model,opt,update):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema='rc3-constructive-policy-v1',state={k:v.detach().cpu() for k,v in model.state_dict().items()},
        optimizer=opt.state_dict(),c22_update=update,warm_checkpoint_sha256=digest(WARM),protocol_sha256=digest(OUT/'protocol.json')),path)


def train():
    p=check();wall=read(OUT/'wall_protocol.json')
    assert wall['protocol_sha256']==digest(OUT/'protocol.json')
    if (OUT/'training.jsonl').exists() or (OUT/'training_started.json').exists():raise FileExistsError('No implicit training replay/resume')
    write(OUT/'training_started.json',dict(wall_protocol_sha256=digest(OUT/'wall_protocol.json')))
    device=wall['device'];baseline=load(device).eval();models={k:load(device) for k in ('A','B')}
    opts={k:torch.optim.Adam(m.parameters(),lr=.0001) for k,m in models.items()}
    assert all(len(o.state)==0 for o in opts.values())
    initial={k:v.detach().cpu().clone() for k,v in baseline.state_dict().items()}
    assert all(torch.equal(initial[k],m.state_dict()[k].cpu()) for m in models.values() for k in initial)
    maps=[make_complex_scene(10,10,7100000000+i,MODES[i%3]) for i in range(24)]
    records=[json.loads(s) for s in (OUT/'instances.jsonl').read_text().splitlines()]
    cache=Cache(24);c=counts();steps={k:0 for k in models};arm_s={k:0. for k in models}
    baseline_s=preparation_s=0.;started=time.perf_counter()
    with gzip.open(OUT/'training_routes.jsonl.gz','wt',encoding='utf-8') as replay:
        for update in range(1,129):
            if time.perf_counter()-started>=wall['paired_training_wall_limit_s']:break
            tick=time.perf_counter();ss=records[(update-1)*32:update*32]
            items=[cache.prepare(resample_tasks(maps[r['ordinal']%24],r['task_seed'])) for r in ss]
            prep=time.perf_counter()-tick;preparation_s+=prep;data=batch(items,device)
            sync(device);tick=time.perf_counter()
            with torch.no_grad():br,_,ba,bl=baseline(data)
            baseline_routes=unpack_routes(ba,bl);baseline_final=[];base_H_before=c['H_s']
            for item,r,route,value in zip(items,ss,baseline_routes,br.cpu().tolist()):
                assert abs(metric(item,route)['J']-value)<1e-6
                final=full_H(item,route,c);v=metric(item,final)['J'];baseline_final.append(v)
                replay.write(json.dumps(dict(role='baseline',update=update,**r,raw_route=route,raw_J=value,final_route=final,final_J=v))+'\n')
            base_time=time.perf_counter()-tick;baseline_s+=base_time
            losses={};order=('A','B') if update%2 else ('B','A')
            for arm in order:
                sync(device);tick=time.perf_counter();m=models[arm];m.train();before=c['H_s']
                devices=[torch.cuda.current_device()] if device=='cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(p['sampling_seed']+update)
                    raw,logp,actions,lengths=m(data,sample=True)
                routes=unpack_routes(actions,lengths);costs=raw.detach().cpu().tolist();finals=[]
                for item,r,route,value in zip(items,ss,routes,costs):
                    assert abs(metric(item,route)['J']-value)<1e-6
                    final=full_H(item,route,c) if arm=='B' else None
                    fv=metric(item,final)['J'] if final is not None else None
                    if fv is not None:assert fv<=value+1e-7
                    finals.append(fv)
                    replay.write(json.dumps(dict(role=arm,update=update,**r,raw_route=route,raw_J=value,final_route=final,final_J=fv))+'\n')
                objective=torch.tensor(finals,device=device,dtype=torch.float64) if arm=='B' else raw
                base_cost=torch.tensor(baseline_final,device=device,dtype=torch.float64) if arm=='B' else br
                advantage=(objective-base_cost).detach().float()/100.
                loss=(advantage*logp).mean();assert torch.isfinite(loss)
                opts[arm].zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(m.parameters(),1.);assert torch.isfinite(grad)
                opts[arm].step();steps[arm]+=1;sync(device);duration=time.perf_counter()-tick;arm_s[arm]+=duration
                losses[arm]=dict(loss=float(loss.item()),gradient_norm=float(grad.item()),mean_raw_J=float(raw.mean().item()),
                    mean_objective_J=float(objective.mean().item()),mean_baseline_objective_J=float(base_cost.mean().item()),
                    mean_logp=float(logp.mean().item()),optimization_s=duration,sample_H_s=c['H_s']-before,
                    prior_weight=float(torch.nn.functional.softplus(m.prior_weight).item()))
            replay.flush()
            append(OUT/'training.jsonl',dict(update=update,arms=losses,shared_preparation_s=prep,
                shared_baseline_s=base_time,shared_baseline_H_s=c['H_s']-base_H_before-losses['B']['sample_H_s'],
                H_started=c['H_started'],H_completed=c['H_completed'],elapsed_s=time.perf_counter()-started))
            write(OUT/'training_progress.json',dict(updates=steps,counters=c,elapsed_s=time.perf_counter()-started))
            if update%16==0:print(json.dumps(dict(update=update,A=losses['A']['mean_objective_J'],B=losses['B']['mean_objective_J'],elapsed_s=time.perf_counter()-started)),flush=True)
    assert steps['A']==steps['B']
    assert all(torch.equal(initial[k],v.cpu()) for k,v in baseline.state_dict().items())
    for arm,m in models.items():save_checkpoint(OUT/'checkpoints'/f'{arm}_update{steps[arm]:04}.pt',m,opts[arm],steps[arm])
    check();n=steps['A']*32
    write(OUT/'training_summary.json',dict(updates=steps,policy_sampled_trajectories=2*n,distinct_training_instances=n,
        baseline_actual_greedy_trajectories=n,baseline_trajectory_uses=2*n,baseline_forward_batches=steps['A'],
        complete_H_counters=c,training_elapsed_s=time.perf_counter()-started,arm_optimization_s=arm_s,
        shared_baseline_s=baseline_s,shared_preparation_s=preparation_s,wall_limit_s=wall['paired_training_wall_limit_s'],
        fixed_endpoint_reached=steps['A']==128,baseline_weights_unchanged=True,fresh_Adam_both=True,
        changed_parameter_tensors={a:sum(not torch.equal(initial[k],v.cpu()) for k,v in m.state_dict().items()) for a,m in models.items()},
        checkpoint_sha256={a:digest(OUT/'checkpoints'/f'{a}_update{steps[a]:04}.pt') for a in models},
        trajectories_sha256=digest(OUT/'training_routes.jsonl.gz'),source_sha256=sources(),
        scope='Matched warm start, not exact C2 continuation; one shared training stream; no endpoint selection. Shared baseline compute split out, not duplicated or silently omitted.'))
    print(json.dumps(read(OUT/'training_summary.json')),flush=True)


def evaluate():
    check();summary=read(OUT/'training_summary.json');assert summary['fixed_endpoint_reached']
    if (OUT/'evaluation_started.json').exists():raise FileExistsError('No implicit evaluation rerun')
    write(OUT/'evaluation_started.json',dict(training_summary_sha256=digest(OUT/'training_summary.json')))
    models={'warm':load('cpu').eval()}
    for arm in ('A','B'):
        path=OUT/'checkpoints'/f'{arm}_update0128.pt';assert digest(path)==summary['checkpoint_sha256'][arm]
        models[arm]=Policy().eval();models[arm].load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state'])
    c=counts();cache=Cache(1)
    for seed in read(OUT/'mechanism_seeds.json'):
        case=seed['case'];scene=resample_tasks(make_complex_scene(10,10,seed['map_seed'],seed['mode']),seed['task_seed'])
        write(OUT/'scenes'/f'case{case:02}.json',scene);item=cache.prepare(scene)
        names=['warm','A','B','prior','fixed3','detour16'];names=names[case%6:]+names[:case%6]
        for name in names:
            if name in ('fixed3','detour16'):
                plan,info,_=baselines.reference(item,name)
                route=[0]+[j+1 for j in plan['visit_order_candidate_ids']]+[0]
                raw=route;rawJ=metric(item,route)['J'];fallback=bool(info['fallback_used'])
            else:
                if name=='prior':raw=baselines.prior(item)
                else:
                    with torch.inference_mode():cost,_,actions,lengths=models[name](batch([item],'cpu'))
                    raw=unpack_routes(actions,lengths)[0];assert abs(metric(item,raw)['J']-cost.item())<1e-6
                rawJ=metric(item,raw,raw=True)['J'];route=full_H(item,raw,c)
                plan=item.certify(route,'c22_'+name);fallback=False
            final=metric(item,route);assert final['J']<=rawJ+1e-7
            assert plan['validation']['valid_model_plan']
            assert [0]+[j+1 for j in plan['visit_order_candidate_ids']]+[0]==route
            assert abs(plan['objective_length_plus_stops_m']-final['J'])<1e-7
            path=OUT/'plans'/f'case{case:02}_{name}.json';write(path,plan)
            append(OUT/'evaluation.jsonl',dict(**seed,method=name,raw_route=raw,raw_J=rawJ,route=route,**final,
                valid=True,fallback=fallback,plan_path=path.relative_to(ROOT.parent).as_posix(),plan_sha256=digest(path)))
        print(json.dumps(dict(mechanism_cases=case+1,plans=(case+1)*6)),flush=True)
    assert c['H_completed']==192
    write(OUT/'evaluation_resources.json',dict(final_plans=288,independent_cases=48,neural_greedy_trajectories=144,
        prior_constructions=48,Fixed3_complete_calls=48,Detour16_complete_calls=48,complete_H_counters=c,
        export_geometry_certificates=288,complete_pipeline_timing_measurements=0,retained_test_queries=0,
        scope='One fixed-endpoint CPU quality evaluation; strong-rule internal searches retained, no standardized timing or independent deployment gate claim'))


def bootstrap(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(114)
    means=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),upper_one_sided95=float(np.quantile(means,.95)),
        positive=int((a>1e-7).sum()),zero=int((abs(a)<=1e-7).sum()),negative=int((a< -1e-7).sum()),values=a.tolist())


def analyze():
    check();rows=[json.loads(s) for s in (OUT/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows)==288 and len({(r['case'],r['method']) for r in rows})==288
    by={(r['case'],r['method']):r for r in rows}
    summary={m:{k:float(np.mean([by[i,m][k] for i in range(48)])) for k in ('J','L','K','raw_J')}
             for m in ('warm','A','B','prior','fixed3','detour16')}
    g=bootstrap([100*(by[i,'A']['J']-by[i,'B']['J'])/by[i,'A']['J'] for i in range(48)])
    h=bootstrap([100*(by[i,'warm']['J']-by[i,'B']['J'])/by[i,'warm']['J'] for i in range(48)])
    valid=all(r['valid'] for r in rows)
    verdict='clear_mechanism_signal' if g['mean']>=.5 and g['ci95'][0]>0 and h['ci95'][0]>0 and valid else (
        'current_budget_does_not_support_continuation' if g['ci95'][1]<.5 or h['ci95'][1]<0 or not valid else 'inconclusive_stop_this_round')
    comparisons={}
    for comparator in ('warm','A','prior','fixed3','detour16'):
        comparisons[comparator]=bootstrap([100*(by[i,'B']['J']-by[i,comparator]['J'])/by[i,comparator]['J'] for i in range(48)])
    write(OUT/'analysis.json',dict(summary=summary,g=g,h=h,mechanism_screen=verdict,all_valid=valid,
        B_degradation_vs=comparisons,endpoint_fixed=128,scope='Pre-registered single matched-training-seed mechanism screen; paired scene bootstrap, no train-seed stability or held-out deployment claim'))
    with (OUT/'per_scene.csv').open('w',encoding='utf-8-sig',newline='') as f:
        flat=[]
        for i in range(48):
            r=dict(case=i,mode=by[i,'warm']['mode'],map_seed=by[i,'warm']['map_seed'],task_seed=by[i,'warm']['task_seed'],g_percent=g['values'][i],h_percent=h['values'][i])
            for m in summary:
                for k in ('J','L','K','raw_J'):r[m+'_'+k]=by[i,m][k]
            flat.append(r)
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(dict(summary=summary,g={k:v for k,v in g.items() if k!='values'},h={k:v for k,v in h.items() if k!='values'},verdict=verdict)),flush=True)


def verify():
    check()
    if (OUT/'verification.json').exists():raise FileExistsError('Verification already done; no duplicate certificates')
    cache=Cache(24);maps=[make_complex_scene(10,10,7100000000+i,MODES[i%3]) for i in range(24)]
    counts_by_role={k:0 for k in ('baseline','A','B')};steps=0;last_update=0;items={};paired={};tick=time.perf_counter()
    with gzip.open(OUT/'training_routes.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);o=r['ordinal'];role=r['role']
            if r['update']!=last_update:items={};last_update=r['update']
            if o not in items:items[o]=cache.prepare(resample_tasks(maps[o%24],7800000000+o))
            item=items[o];assert r['task_seed']==7800000000+o and r['map_seed']==7100000000+o%24
            assert r['update']==o//32+1
            raw=metric(item,r['raw_route'],raw=True);assert abs(raw['J']-r['raw_J'])<1e-6
            steps+=raw['K'];counts_by_role[role]+=1
            paired.setdefault(o,{})[role]=r['task_seed']
            if r['final_route'] is not None:
                final=metric(item,r['final_route']);assert abs(final['J']-r['final_J'])<1e-7 and final['J']<=raw['J']+1e-7
    assert counts_by_role==dict(baseline=4096,A=4096,B=4096)
    assert len(paired)==4096 and all(set(v)=={'baseline','A','B'} and len(set(v.values()))==1 for v in paired.values())
    certificates=[];rows=[json.loads(s) for s in (OUT/'evaluation.jsonl').read_text().splitlines()]
    for case in range(48):
        scene=read(OUT/'scenes'/f'case{case:02}.json');item=cache.prepare(scene)
        for r in [x for x in rows if x['case']==case]:
            path=ROOT.parent/r['plan_path'];assert digest(path)==r['plan_sha256'];plan=read(path)
            actual=metric(item,r['route'])
            for key in ('J','L','K'):assert abs(actual[key]-r[key])<1e-7
            cert=validate(scene,plan,item.base.sparse);assert cert['valid_model_plan'],cert
            assert [0]+[j+1 for j in plan['visit_order_candidate_ids']]+[0]==r['route']
            certificates.append(dict(path=r['plan_path'],sha256=r['plan_sha256'],valid=True))
    check()
    write(OUT/'verification.json',dict(all_passed=True,training_numeric_replays=counts_by_role,service_transitions=steps,
        matched_distinct_instances=4096,separate_geometry_certificates=288,export_geometry_certificates=288,
        total_geometry_certificates=576,certificates=certificates,new_neural_inferences=0,new_H_calls=0,
        new_optimizer_updates=0,retained_test_queries=0,elapsed_s=time.perf_counter()-tick,
        scope='All training/baseline raw routes replayed for legal gains, coverage and closed-tour rewards; saved final routes checked without rerunning H; scalar geometry on 288 saved mechanism plans, not all training trajectories; no IK'))
    print('PASS: 12288 training/baseline numeric replays; 288 saved-plan scalar certificates; no new H calls',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','calibrate','train','evaluate','analyze','verify']);args=parser.parse_args()
    settings()
    with threadpool_limits(limits=1):globals()[args.phase]()
