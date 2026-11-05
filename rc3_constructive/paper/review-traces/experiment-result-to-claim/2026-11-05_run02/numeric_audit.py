"""Independent saved-record audit. No project imports or research execution."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
OUT = ROOT/'rc3_constructive/c22'
TRACE = Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def close(a,b,tol=1e-6):assert abs(a-b)<=tol,(a,b)
def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def boot(a):
    a=np.array(a,float);rng=np.random.default_rng(114)
    m=a[rng.integers(len(a),size=(10000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),ci95=np.quantile(m,[.025,.975]).tolist(),upper_one_sided95=float(np.quantile(m,.95)),
                positive=int((a>1e-7).sum()),zero=int((abs(a)<=1e-7).sum()),negative=int((a< -1e-7).sum()),values=a.tolist())

def training_audit():
    p=read(OUT/'protocol.json');s=read(OUT/'training_summary.json');wall=read(OUT/'wall_protocol.json')
    assert p['base_commit']=='44aee8f508d6c39a1d5f44d631e8e519da4f8c21'
    for name,h in p['source_sha256'].items():assert sha(ROOT/name)==h,name
    assert s['source_sha256']==p['source_sha256']
    assert sha(OUT/'instances.jsonl')==p['instances_sha256']
    assert sha(OUT/'mechanism_seeds.json')==p['mechanism_seeds_sha256']
    assert wall['protocol_sha256']==sha(OUT/'protocol.json')
    assert read(OUT/'training_started.json')['wall_protocol_sha256']==sha(OUT/'wall_protocol.json')
    assert read(OUT/'calibration_started.json')['protocol_sha256']==sha(OUT/'protocol.json')
    cap=min(7200,max(1800,math.ceil(128*(max(wall['pair_probe_s'])+wall['preparation_32_s'])*2+600)))
    assert cap==wall['paired_training_wall_limit_s']==s['wall_limit_s']==2401
    assert wall['optimizer_updates']==wall['sampled_trajectories']==0
    assert wall['greedy_forward_trajectories']==192 and wall['greedy_backward_trajectories']==128
    assert wall['calibration_counters']['H_started']==wall['calibration_counters']['H_completed']==128
    instances=lines(OUT/'instances.jsonl');seeds=read(OUT/'mechanism_seeds.json')
    assert len(instances)==4096 and len(seeds)==48
    for i,r in enumerate(instances):
        assert r['ordinal']==i and r['task_seed']==7800000000+i and r['map_seed']==7100000000+i%24
        assert r['mode']==('regular','mixed','blocked')[(i%24)%3]
    for i,r in enumerate(seeds):
        assert r['case']==i and r['map_seed']==7900000000+i and r['task_seed']==8000000000+i
        assert r['mode']==('regular','mixed','blocked')[i%3]
    hist=lines(OUT/'training.jsonl');assert [r['update'] for r in hist]==list(range(1,129))
    assert sha(OUT/'training_routes.jsonl.gz')==s['trajectories_sha256']
    by={};role_counts=dict(A=0,B=0,baseline=0);role_steps=dict(A=0,B=0,baseline=0)
    with gzip.open(OUT/'training_routes.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);o=r['ordinal'];role=r['role'];assert 0<=o<4096
            assert r['update']==o//32+1
            for key in ('ordinal','map_seed','task_seed','mode'):assert r[key]==instances[o][key]
            assert (o,role) not in by;by[o,role]=r;role_counts[role]+=1
            raw=r['raw_route'];assert raw[0]==raw[-1]==0 and len(set(raw[1:-1]))==len(raw)-2
            assert all(type(v) is int and v>0 for v in raw[1:-1])
            role_steps[role]+=len(raw)-2
            if role in ('baseline','B'):
                final=r['final_route'];assert final[0]==final[-1]==0 and len(set(final[1:-1]))==len(final)-2
                assert math.isfinite(r['final_J']) and r['final_J']<=r['raw_J']+1e-7
            else:assert r['final_route'] is None and r['final_J'] is None
    assert role_counts==dict(A=4096,B=4096,baseline=4096)
    assert all((i,role) in by for i in range(4096) for role in role_counts)
    for i in range(32):
        assert by[i,'A']['raw_route']==by[i,'B']['raw_route'];close(by[i,'A']['raw_J'],by[i,'B']['raw_J'])
    for u,row in enumerate(hist,1):
        assert row['H_started']==row['H_completed']==u*64
        for role in ('A','B'):
            raw=[by[i,role]['raw_J'] for i in range((u-1)*32,u*32)]
            objective=raw if role=='A' else [by[i,role]['final_J'] for i in range((u-1)*32,u*32)]
            baseline=[by[i,'baseline']['raw_J' if role=='A' else 'final_J'] for i in range((u-1)*32,u*32)]
            arm=row['arms'][role]
            close(np.mean(raw),arm['mean_raw_J']);close(np.mean(objective),arm['mean_objective_J']);close(np.mean(baseline),arm['mean_baseline_objective_J'])
            assert math.isfinite(arm['loss']) and math.isfinite(arm['mean_logp']) and arm['mean_logp']<0
            assert math.isfinite(arm['gradient_norm']) and arm['gradient_norm']>0
        close(row['arms']['A']['sample_H_s'],0)
    close(hist[0]['arms']['A']['mean_logp'],hist[0]['arms']['B']['mean_logp'])
    assert s['updates']==dict(A=128,B=128) and s['fixed_endpoint_reached']
    assert s['policy_sampled_trajectories']==8192 and s['distinct_training_instances']==4096
    assert s['baseline_actual_greedy_trajectories']==4096 and s['baseline_trajectory_uses']==8192 and s['baseline_forward_batches']==128
    assert s['complete_H_counters']['H_started']==s['complete_H_counters']['H_completed']==8192
    close(sum(r['shared_baseline_H_s']+r['arms']['B']['sample_H_s'] for r in hist),s['complete_H_counters']['H_s'])
    for role in ('A','B'):close(sum(r['arms'][role]['optimization_s'] for r in hist),s['arm_optimization_s'][role])
    close(sum(r['shared_baseline_s'] for r in hist),s['shared_baseline_s'])
    close(sum(r['shared_preparation_s'] for r in hist),s['shared_preparation_s'])
    assert s['training_elapsed_s']<s['wall_limit_s']
    warm=torch.load(ROOT/'rc3_constructive/outputs/checkpoints/update0128.pt',map_location='cpu',weights_only=False)
    changed={};checkpoint_info={}
    for role in ('A','B'):
        path=OUT/'checkpoints'/f'{role}_update0128.pt';assert sha(path)==s['checkpoint_sha256'][role]
        cp=torch.load(path,map_location='cpu',weights_only=False)
        assert cp['c22_update']==128 and cp['protocol_sha256']==sha(OUT/'protocol.json')
        assert cp['warm_checkpoint_sha256']==sha(ROOT/'rc3_constructive/outputs/checkpoints/update0128.pt')
        changed[role]=sum(not torch.equal(v,cp['state'][k]) for k,v in warm['state'].items())
        assert changed[role]==s['changed_parameter_tensors'][role]==26
        st=cp['optimizer'];assert len(st['state'])==26
        assert all(int(v['step'].item())==128 for v in st['state'].values())
        assert len(st['param_groups'])==1 and st['param_groups'][0]['lr']==.0001
        assert st['param_groups'][0]['betas']==(.9,.999) and st['param_groups'][0]['eps']==1e-8
        checkpoint_info[role]=dict(parameter_tensors_changed=changed[role],optimizer_state_tensors=26,optimizer_steps=128)
    assert s['fresh_Adam_both'] and s['baseline_weights_unchanged']
    return dict(all_passed=True,source_hashes=len(p['source_sha256']),calibration_H=128,training_H=8192,
        training_routes=role_counts,service_transitions=role_steps,distinct_training_instances=4096,policy_samples=8192,
        baseline_actual=4096,baseline_uses=8192,training_updates=256,wall_limit_s=cap,training_elapsed_s=s['training_elapsed_s'],
        checkpoint_info=checkpoint_info,
        loss_formula_audit='Source checked: detached cost difference times summed logp /100, positive sign for cost minimization. All 256 per-batch raw/objective/baseline means match stored trajectories. Exact trajectory-weighted loss cannot be reconstructed because only mean_logp is retained; mean products cannot replace mean of products. No exact-loss reconstruction claimed.',
        fixed_baseline_audit='Same warm source checkpoint, source holds baseline fixed and asserts unchanged tensors at finish; one shared stored baseline route per instance, two uses. No new neural execution.',
        research_calls_added=dict(neural=0,planner=0,H=0,optimizer=0,geometry=0,test_generation=0))

def main(training_only=False):
    train=training_audit()
    if training_only:
        write(TRACE/'training_audit.json',train);print('Training saved-record audit PASS');return
    p=read(OUT/'protocol.json');a=read(OUT/'analysis.json');v=read(OUT/'verification.json');rows=lines(OUT/'evaluation.jsonl');resources=read(OUT/'resources.json')
    assert len(rows)==288 and len({(r['case'],r['method']) for r in rows})==288
    by={(r['case'],r['method']):r for r in rows};summary={};hash_count=0
    methods=['warm','A','B','prior','fixed3','detour16']
    for r in rows:
        seed=read(OUT/'mechanism_seeds.json')[r['case']]
        for key in ('case','mode','map_seed','task_seed'):assert r[key]==seed[key]
        path=ROOT/r['plan_path'];assert sha(path)==r['plan_sha256'];hash_count+=1;plan=read(path)
        assert r['route']==[0]+[i+1 for i in plan['visit_order_candidate_ids']]+[0]
        close(plan['objective_length_plus_stops_m'],r['J']);close(plan['route_length_m'],r['L'])
        assert r['K']==len(plan['service_stops']);close(r['L']+.75*r['K'],r['J'])
        close(sum(plan['route_leg_costs_m']),r['L']);assert r['valid'] and plan['validation']['valid_model_plan']
    for m in methods:
        summary[m]={k:float(np.mean([by[i,m][k] for i in range(48)])) for k in ('J','L','K','raw_J')}
        for key,val in summary[m].items():close(val,a['summary'][m][key])
    g=boot([100*(by[i,'A']['J']-by[i,'B']['J'])/by[i,'A']['J'] for i in range(48)])
    h=boot([100*(by[i,'warm']['J']-by[i,'B']['J'])/by[i,'warm']['J'] for i in range(48)])
    for name,val in [('g',g),('h',h)]:
        for key in val:
            if isinstance(val[key],list):assert np.allclose(val[key],a[name][key],rtol=0,atol=1e-10)
            else:close(val[key],a[name][key],tol=1e-10)
    verdict='clear_mechanism_signal' if g['mean']>=.5 and g['ci95'][0]>0 and h['ci95'][0]>0 else ('current_budget_does_not_support_continuation' if g['ci95'][1]<.5 or h['ci95'][1]<0 else 'inconclusive_stop_this_round')
    assert verdict==a['mechanism_screen']
    comparisons={}
    for m in ('warm','A','prior','fixed3','detour16'):
        z=boot([100*(by[i,'B']['J']-by[i,m]['J'])/by[i,m]['J'] for i in range(48)]);comparisons[m]=z
        for key in z:
            if isinstance(z[key],list):assert np.allclose(z[key],a['B_degradation_vs'][m][key],rtol=0,atol=1e-10)
            else:close(z[key],a['B_degradation_vs'][m][key],tol=1e-10)
    assert v['all_passed'] and v['training_numeric_replays']==dict(baseline=4096,A=4096,B=4096)
    assert v['service_transitions']==sum(train['service_transitions'].values())==1172353
    assert v['separate_geometry_certificates']==288 and v['total_geometry_certificates']==576
    assert v['new_neural_inferences']==v['new_H_calls']==v['new_optimizer_updates']==v['retained_test_queries']==0
    ev=read(OUT/'evaluation_resources.json');assert ev['final_plans']==288 and ev['complete_H_counters']['H_completed']==192
    assert resources['optimizer_updates']==dict(A=128,B=128,total=256)
    assert resources['training_policy_sampled_trajectories']==8192 and resources['distinct_training_instances']==4096
    assert resources['baseline_training_actual_greedy_trajectories']==4096 and resources['baseline_training_trajectory_uses']==8192
    assert resources['total_neural_input_trajectories']==8192+4096+192+144==12624
    assert resources['complete_H_calls']==dict(calibration=128,training=8192,evaluation=192,total=8512)
    assert resources['complete_H_training_breakdown']==dict(B_samples=4096,shared_fixed_baseline=4096,A_samples=0)
    assert resources['geometric_certificates']==dict(export=288,separate_saved_plan_checks=288,total=576)
    assert resources['training_raw_numeric_replays']==12288 and resources['verification_service_transitions']==1172353
    assert resources['retained_test_queries']==resources['complete_pipeline_timing_measurements']==resources['review_new_research_calls']==0
    hist=lines(OUT/'training.jsonl')
    for role in ('A','B'):
        diag=resources['raw_diagnostics'][role]
        for label,rr in [('first16',hist[:16]),('last16',hist[-16:])]:
            close(diag[label+'_mean_raw_J'],np.mean([r['arms'][role]['mean_raw_J'] for r in rr]))
            close(diag[label+'_objective_J'],np.mean([r['arms'][role]['mean_objective_J'] for r in rr]))
    report=(ROOT/'rc3_constructive/REPORT_C22.md').read_text(encoding='utf-8')
    assert '无定论，本轮停止' in report and '1.5491%' in report and '-0.9084%' in report
    write(TRACE/'numeric_audit.json',dict(all_passed=True,training=train,summary=summary,g=g,h=h,screen=verdict,
        comparisons= comparisons,matched_plan_hashes=hash_count,existing_geometry_certificates=576,
        evaluation=ev,resources=resources,new_research_calls=0,
        scope='Saved arithmetic, hashes, checkpoint data inspection only; no model creation/forward, postprocessing, geometry certificates, scene generation or training.'))
    print(json.dumps(dict(audit='PASS',g=g['mean'],g_ci=g['ci95'],h=h['mean'],h_ci=h['ci95'],screen=verdict)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training-only',action='store_true');args=p.parse_args();main(args.training_only)
