"""Independent saved-record closeout audit; no project or policy imports."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def same(a,b):
    assert np.allclose(a,b,rtol=0,atol=1e-9),(a,b)


def statistics(values):
    a=np.array(values,dtype=float)
    prng=np.random.default_rng(114)
    samples=np.mean(a[prng.integers(0,len(a),size=(10000,len(a)))],axis=1)
    return dict(mean=float(np.mean(a)),ci95=np.percentile(samples,[2.5,97.5]).tolist(),
        upper_one_sided95=float(np.percentile(samples,95)),positive=int(np.sum(a>1e-7)),
        zero=int(np.sum(abs(a)<=1e-7)),negative=int(np.sum(a< -1e-7)),scene_values=a.tolist())


def main():
    saved=read(ROOT/'closeout/analysis.json')
    rows=[json.loads(line) for line in (ROOT/'c22/evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
    expected_methods={'warm','A','B','prior','fixed3','detour16'}
    assert len(rows)==288
    by={(r['case'],r['method']):r for r in rows}
    assert set(by)=={(i,m) for i in range(48) for m in expected_methods}
    seeds=read(ROOT/'c22/mechanism_seeds.json')
    with (ROOT/'c22/per_scene.csv').open(encoding='utf-8-sig',newline='') as f:
        csv_by={int(r['case']):r for r in csv.DictReader(f)}
    assert set(csv_by)==set(range(48))
    for i in range(48):
        for m in expected_methods:
            r=by[i,m]
            assert (r['map_seed'],r['task_seed'])==(seeds[i]['map_seed'],seeds[i]['task_seed'])
            for field in ('J','L','K','raw_J'):same(r[field],float(csv_by[i][m+'_'+field]))
            same(r['J'],r['L']+.75*r['K'])
    comparisons={}
    for ours,base in [('B','A'),('B','warm'),('A','warm'),('warm','prior'),('B','prior')]:
        name=ours+'_vs_'+base
        values=[100*(by[i,base]['J']-by[i,ours]['J'])/by[i,base]['J'] for i in range(48)]
        comparisons[name]=statistics(values)
        for key,value in comparisons[name].items():same(value,saved['improvements'][name][key])
    gaps={}
    for ours in ('warm','B'):
        for base in ('fixed3','detour16'):
            name=ours+'_vs_'+base
            gaps[name]=statistics([100*(by[i,ours]['J']-by[i,base]['J'])/by[i,base]['J'] for i in range(48)])
            for key,value in gaps[name].items():same(value,saved['degradation'][name][key])
    cost_diff=np.array([by[i,'B']['J']-by[i,'warm']['J'] for i in range(48)])
    relative=np.array(comparisons['B_vs_warm']['scene_values'])
    loo=np.array([(relative.sum()-relative[i])/47 for i in range(48)])
    positive=cost_diff[cost_diff>0];negative=cost_diff[cost_diff<0]
    balance=dict(wins=int(np.sum(cost_diff< -1e-7)),losses=int(np.sum(cost_diff>1e-7)),
        saved_J_sum=float(-negative.sum()),extra_J_sum=float(positive.sum()),
        saved_J_mean_on_wins=float(-negative.mean()),extra_J_mean_on_losses=float(positive.mean()),
        net_extra_J=float(cost_diff.sum()),mean_extra_J=float(cost_diff.mean()),
        leave_one_out_relative_improvement_range=[float(loo.min()),float(loo.max())],
        leave_one_out_all_negative=bool(np.max(loo)<0),leave_one_out_values=loo.tolist())
    for key,value in balance.items():same(value,saved['B_vs_warm_balance'][key])
    decomposition={k:float(np.mean([by[i,'B'][k]-by[i,'warm'][k] for i in range(48)])) for k in ('L','K','J')}
    decomposition['penalty']=decomposition['K']*.75
    same(decomposition['L']+decomposition['penalty'],decomposition['J'])
    for k,value in decomposition.items():same(value,saved['B_vs_warm_decomposition'][k])
    logs=[json.loads(s) for s in (ROOT/'c22/training.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['update'] for r in logs]==list(range(1,129))
    train_gap=np.mean([r['arms']['B']['mean_objective_J']-r['arms']['B']['mean_baseline_objective_J'] for r in logs])
    clips={m:sum(r['arms'][m]['gradient_norm']>1 for r in logs) for m in ('A','B')}
    same(train_gap,saved['training_diagnostics']['B_sample_minus_fixed_baseline_mean_final_J'])
    assert clips==saved['training_diagnostics']['batches_over_clip']
    paths=[]
    for name,sha in saved['input_sha256'].items():
        p=ROOT.parent/name
        actual=hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual==sha,(name,actual,sha)
        paths.append(name)
    original=read(ROOT/'c22/analysis.json')
    assert original['mechanism_screen']==saved['historical_screen_unchanged']=='inconclusive_stop_this_round'
    counts=saved['counts'];assert all(v==0 for v in counts.values())
    status=read(ROOT/'closeout/STATUS.json')
    assert status['status']=='closed' and status['C22_screen']==original['mechanism_screen']
    assert all(x==0 for x in status['new_research_calls'].values())
    refs=[status['research_reference'],*status['archived_endpoints'].values()]
    for ref in refs:assert hashlib.sha256((ROOT.parent/ref['path']).read_bytes()).hexdigest()==ref['sha256']
    assert all(not ref['continue_training'] for ref in status['archived_endpoints'].values())
    assert status['future_navigation_information_gate']['status']=='proposed_not_started'
    assert status['future_navigation_information_gate']['abstract_witness_status']=='not_searched'
    assert status['future_navigation_information_gate']['generator_realization_status']=='not_attempted'
    docs=['REPORT_CLOSEOUT.md','NAVIGATION_INFORMATION_GATE.md','closeout/STATUS.json']
    result=dict(all_passed=True,method='Independent NumPy and stdlib saved-record audit; no project imports',
        scene_count=48,rows_checked=288,input_hashes_checked=len(paths),improvements=comparisons,
        degradation=gaps,balance=balance,decomposition=decomposition,training_gap=float(train_gap),clip_batches=clips,
        inherited_original_screen=original['mechanism_screen'],new_research_calls=0,
        status_reference_hashes_matched=3,
        scope_documents_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in docs},
        scope_review='PASS by independent source/document reading: current scoring consumed inputs versus whole D; all complete-H optimal-action sets including ties; positive certified separation; abstract versus generator evidence separated; future gate explicitly not executed.',
        limitations=['Single matched training seed; conditional scene-bootstrap intervals',
            'No causal root attribution from clipping or sample-baseline mean gaps',
            'No per-trajectory logp/KL/action margin reconstruction',
            'No deletion of difficult scenes; leave-one-out is descriptive sensitivity only'])
    (HERE/'numeric_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(all_passed=True,scene_count=48,input_hashes_checked=len(paths),new_research_calls=0)))


if __name__=='__main__':main()
