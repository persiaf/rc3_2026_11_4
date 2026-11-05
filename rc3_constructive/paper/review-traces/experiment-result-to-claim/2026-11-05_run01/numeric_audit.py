"""Independent saved-record arithmetic audit; imports no project planning code."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
TRACE = Path(__file__).resolve().parent
OUT = ROOT/'c21'
REPO = ROOT.parent

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def close(a,b):
    assert abs(a-b)<1e-9, (a,b)

def stat(x):
    a=np.asarray(x,float)
    index=np.random.default_rng(114).integers(len(a),size=(10000,len(a)))
    return dict(mean=float(a.mean()),ci95=np.quantile(a[index].mean(1),[.025,.975]).tolist(),
        positive_cases=int((a>1e-7).sum()),negative_cases=int((a < -1e-7).sum()),zero_cases=int((abs(a)<=1e-7).sum()),scene_values=a.tolist())

def check_stat(actual,x):
    expected=stat(x)
    for key in ('mean','positive_cases','negative_cases','zero_cases'):close(actual[key],expected[key])
    assert np.allclose(actual['ci95'],expected['ci95'],rtol=0,atol=1e-9)
    assert np.allclose(actual['scene_values'],expected['scene_values'],rtol=0,atol=1e-9)
    return expected

def main():
    rows=[json.loads(x) for x in (OUT/'rows.jsonl').read_text(encoding='utf-8').splitlines()]
    ledger=[json.loads(x) for x in (OUT/'call_ledger.jsonl').read_text(encoding='utf-8').splitlines()]
    p=read(OUT/'protocol.json'); a=read(OUT/'analysis.json'); resource=read(OUT/'resources.json');v=read(OUT/'verification.json')
    assert len(rows)==72 and len(ledger)==144
    by={(r['case'],r['method']):r for r in rows};assert len(by)==72
    old=read(ROOT/'outputs/final_evaluation.json')
    originals={(r['case'],r['method']):r for r in old if r['rep']==0 and r['cache_mode']=='cold'}
    for name,expected in p['input_sha256'].items():assert sha(REPO/name)==expected, name
    for case in range(24):
        assert {r['method'] for r in rows if r['case']==case}=={'prior','initial','model'}
        source_order=('prior','initial','model');source_order=source_order[case%3:]+source_order[:case%3]
        assert [r['method'] for r in rows if r['case']==case]==list(source_order)
    for k,r in enumerate(rows,1):
        assert r['call']==k
        start,end=ledger[2*(k-1):2*k]
        assert start['status']=='started' and end['status']=='completed'
        for log in (start,end):assert (log['call'],log['case'],log['method'])==(k,r['case'],r['method'])
        close(end['J'],r['H1']['J'])
        original=originals[r['case'],r['method']]
        assert r['H0']['route']==original['route'] and r['raw_route']==original['raw_route']
        close(r['raw_J'],original['raw_J'])
        for key in ('J','L','K'):
            close(r['H0'][key],original[key]);close(r['fixed3'][key],originals[r['case'],'fixed3'][key])
        for h in ('H0','H1','fixed3'):
            q=r[h];assert q['route'][0]==q['route'][-1]==0
            assert q['candidate_ids']==sorted(j-1 for j in q['route'][1:-1])
            assert q['K']==len(q['candidate_ids'])==len(set(q['candidate_ids']))
            close(q['J'],q['L']+.75*q['K'])
        close(r['delta_H_J'],r['H0']['J']-r['H1']['J'])
        close(r['delta_H_L'],r['H0']['L']-r['H1']['L'])
        close(r['delta_H_penalty'],.75*(r['H0']['K']-r['H1']['K']))
        close(r['delta_H_J'],r['delta_H_L']+r['delta_H_penalty'])
        assert r['delta_H_J']>0
        assert r['removed_candidate_ids']==sorted(set(r['H0']['candidate_ids'])-set(r['H1']['candidate_ids']))
        assert r['added_candidate_ids']==sorted(set(r['H1']['candidate_ids'])-set(r['H0']['candidate_ids']))
        planpath=REPO/r['plan_path'];assert sha(planpath)==r['plan_sha256']
        plan=read(planpath);assert plan['validation']['valid_model_plan'] and r['valid']
        assert plan['visit_order_candidate_ids']==[j-1 for j in r['H1']['route'][1:-1]]
        close(plan['objective_length_plus_stops_m'],r['H1']['J'])
        close(plan['route_length_m'],r['H1']['L'])
        assert len(plan['service_stops'])==r['H1']['K']
    summaries={}
    for method in ('prior','initial','model'):
        rr=[by[c,method] for c in range(24)]; saved=a['summary'][method]
        for h in ('H0','H1'):
            for key in ('J','L','K'):close(saved[h][key],np.mean([r[h][key] for r in rr]))
        deltas={
            'delta_H_J':[r['H0']['J']-r['H1']['J'] for r in rr],
            'delta_H_L':[r['H0']['L']-r['H1']['L'] for r in rr],
            'delta_H_penalty':[.75*(r['H0']['K']-r['H1']['K']) for r in rr],
            'delta_H_relative_percent':[100*(r['H0']['J']-r['H1']['J'])/r['H0']['J'] for r in rr],
            'gap_Fixed3_J':[r['H1']['J']-r['fixed3']['J'] for r in rr],
            'gap_Fixed3_relative_percent':[100*(r['H1']['J']-r['fixed3']['J'])/r['fixed3']['J'] for r in rr]}
        for name,x in deltas.items():check_stat(saved[name],x)
        times=[1000*r['H1_added_s'] for r in rr]
        for name,val in dict(median=np.median(times),p95=np.quantile(times,.95),min=min(times),max=max(times)).items():close(saved['H1_added_ms'][name],val)
        for field,source in [('mean_removed_candidate_ids','removed_candidate_ids'),('mean_added_candidate_ids','added_candidate_ids')]:close(saved[field],np.mean([len(r[source]) for r in rr]))
        summaries[method]={k:saved[k] for k in ('H0','H1','delta_H_J','gap_Fixed3_relative_percent')}
    learning={}
    for comparator in ('prior','initial'):
        for h in ('H0','H1'):
            label=f'model_vs_{comparator}_{h}'
            diff=[by[c,comparator][h]['J']-by[c,'model'][h]['J'] for c in range(24)]
            rel=[100*diff[c]/by[c,comparator][h]['J'] for c in range(24)]
            learning[label]={
                'absolute_J':check_stat(a['learning_increment'][label]['absolute_J'],diff),
                'relative_percent':check_stat(a['learning_increment'][label]['relative_percent'],rel)}
    for h in ('H0','H1'):
        for other in ('prior','initial','fixed3'):
            values=[]
            for c in range(24):
                aa=set(by[c,'model'][h]['candidate_ids'])
                bb=set(by[c,'model']['fixed3']['candidate_ids'] if other=='fixed3' else by[c,other][h]['candidate_ids'])
                values.append(len(aa&bb)/len(aa|bb))
            check_stat(a['candidate_jaccard'][f'model_vs_{other}_{h}'],values)
    for mode in ('regular','mixed','blocked'):
        rr=[r for r in rows if r['method']=='model' and r['scene_mode']==mode]
        assert len(rr)==a['model_by_scene_mode'][mode]['cases']==8
        close(a['model_by_scene_mode'][mode]['mean_delta_H_J'],np.mean([r['delta_H_J'] for r in rr]))
        close(a['model_by_scene_mode'][mode]['mean_gap_fixed3_percent'],np.mean([100*(r['H1']['J']/r['fixed3']['J']-1) for r in rr]))
    assert resource['new_bounded_H1_calls']==72 and resource['geometry_certificates_total']==144
    assert resource['export_geometry_certificates']==resource['separate_saved_plan_geometry_checks']==72
    assert v['all_passed'] and len(v['certificates'])==v['separate_geometry_checks']==72
    for cert in v['certificates']:assert cert['valid'] and sha(REPO/cert['file'])==cert['sha256']
    assert all(resource[k]==0 for k in ('new_H0_calls','new_neural_routes','neural_forward_calls','optimizer_updates','reserved_test_scenes_queried','full_pipeline_measurements'))
    assert resource['protocol_sha256']==sha(OUT/'protocol.json')
    rr=[by[c,'model'] for c in range(24)]
    gap0=float(np.mean([r['H0']['J']-r['fixed3']['J'] for r in rr]));gap1=float(np.mean([r['H1']['J']-r['fixed3']['J'] for r in rr]))
    result=dict(all_passed=True,method='Independent implementation with NumPy, stdlib and saved files only; no project module imports',
        row_count=72,ledger_starts=72,ledger_completions=72,matched_input_hashes=len(p['input_sha256']),
        matched_plan_hashes=72,existing_geometry_certificates_observed=144,new_geometry_certificates=0,
        new_planner_calls=0,new_neural_calls=0,new_optimizer_updates=0,new_test_scene_queries=0,
        mean_absolute_gap_before=gap0,mean_absolute_gap_after=gap1,fraction_of_mean_absolute_gap_removed=(gap0-gap1)/gap0,
        summary=summaries,learning_increment=learning,
        limitations='No rerun of H1 or geometry. Validity is checked through saved certificates, hashes, numeric plan fields, and code inspection; temporal preregistration is evidenced by executor sequence and protocol hash, not an external timestamp service.')
    (TRACE/'numeric_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('summary','learning_increment')},ensure_ascii=False))

if __name__=='__main__':main()
