"""Read-only arithmetic and byte audit for constructive branch closeout.

No project planner, environment, policy, torch, or scene-generator imports.
Writes new closeout artifacts only; never rewrites C012/C21/C22 evidence.
"""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import time
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'closeout'
BASE='5d852186974ab21274443b8bebc5e276164564df'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def stats(values):
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(114)
    means=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),
        upper_one_sided95=float(np.quantile(means,.95)),positive=int((a>1e-7).sum()),
        zero=int((abs(a)<=1e-7).sum()),negative=int((a< -1e-7).sum()),scene_values=a.tolist())


def main():
    OUT.mkdir(exist_ok=True)
    rows=[json.loads(s) for s in (ROOT/'c22/evaluation.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==288 and len({(r['case'],r['method']) for r in rows})==288
    by={(r['case'],r['method']):r for r in rows}
    seeds=read(ROOT/'c22/mechanism_seeds.json')
    with (ROOT/'c22/per_scene.csv').open(encoding='utf-8-sig',newline='') as f:csv_rows=list(csv.DictReader(f))
    assert len(csv_rows)==48
    for i in range(48):
        for m in ('warm','A','B','prior','fixed3','detour16'):
            r=by[i,m]
            assert r['map_seed']==seeds[i]['map_seed'] and r['task_seed']==seeds[i]['task_seed']
            for k in ('J','L','K','raw_J'):assert abs(r[k]-float(csv_rows[i][m+'_'+k]))<1e-7
            assert abs(r['J']-r['L']-.75*r['K'])<1e-7
    improvements={}
    for ours,base in (('B','A'),('B','warm'),('A','warm'),('warm','prior'),('B','prior')):
        improvements[ours+'_vs_'+base]=stats([100*(by[i,base]['J']-by[i,ours]['J'])/by[i,base]['J'] for i in range(48)])
    gaps={}
    for ours in ('warm','B'):
        for ref in ('fixed3','detour16'):
            gaps[ours+'_vs_'+ref]=stats([100*(by[i,ours]['J']-by[i,ref]['J'])/by[i,ref]['J'] for i in range(48)])
    delta=np.array([by[i,'B']['J']-by[i,'warm']['J'] for i in range(48)])
    h=np.asarray(improvements['B_vs_warm']['scene_values'])
    leave_one_out=[float(np.delete(h,i).mean()) for i in range(48)]
    decomposition={k:float(np.mean([by[i,'B'][k]-by[i,'warm'][k] for i in range(48)])) for k in ('J','L','K')}
    decomposition['penalty']=.75*decomposition['K']
    assert abs(decomposition['J']-decomposition['L']-decomposition['penalty'])<1e-7
    balance=dict(wins=int((delta< -1e-7).sum()),losses=int((delta>1e-7).sum()),
        saved_J_sum=float(-delta[delta<0].sum()),extra_J_sum=float(delta[delta>0].sum()),
        saved_J_mean_on_wins=float(-delta[delta<0].mean()),extra_J_mean_on_losses=float(delta[delta>0].mean()),
        net_extra_J=float(delta.sum()),mean_extra_J=float(delta.mean()),
        leave_one_out_relative_improvement_range=[min(leave_one_out),max(leave_one_out)],
        leave_one_out_all_negative=bool(max(leave_one_out)<0),leave_one_out_values=leave_one_out,
        scope='Sensitivity calculation only. No scene removed, no re-selection, no new bootstrap unit.')
    logs=[json.loads(s) for s in (ROOT/'c22/training.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(logs)==128
    training=dict(B_sample_minus_fixed_baseline_mean_final_J=float(np.mean([r['arms']['B']['mean_objective_J']-r['arms']['B']['mean_baseline_objective_J'] for r in logs])),
        clip_threshold=1.,batches_over_clip={a:sum(r['arms'][a]['gradient_norm']>1. for r in logs) for a in ('A','B')},
        missing_diagnostics=['per-trajectory log probabilities','same-state policy KL','action ranking margins'],
        interpretation='Clipping frequency is not evidence of gradient explosion; sample-vs-baseline gap does not identify cause. No gradient or policy recomputation.')
    old=read(ROOT/'c22/analysis.json')
    for name,key in (('B_vs_A','g'),('B_vs_warm','h')):
        assert abs(improvements[name]['mean']-old[key]['mean'])<1e-10
        assert np.allclose(improvements[name]['ci95'],old[key]['ci95'],atol=1e-10)
    assert old['mechanism_screen']=='inconclusive_stop_this_round'
    paths=[ROOT/'c22/evaluation.jsonl',ROOT/'c22/per_scene.csv',ROOT/'c22/training.jsonl',ROOT/'c22/analysis.json',
        ROOT/'model.py',ROOT/'problem.py',ROOT/'baselines.py',ROOT/'plan.py',ROOT/'c22.py',Path(__file__)]
    paths+=list((ROOT/'outputs/checkpoints').glob('*.pt'))+list((ROOT/'c22/checkpoints').glob('*.pt'))
    write('analysis.json',dict(schema='rc3-constructive-closeout-statistics-v1',base_commit=BASE,
        statistical_unit='48 C22 scenes, single matched training seed; paired bootstrap10000 seed114',
        improvements=improvements,degradation=gaps,B_vs_warm_balance=balance,B_vs_warm_decomposition=decomposition,
        training_diagnostics=training,input_sha256={p.relative_to(ROOT.parent).as_posix():digest(p) for p in paths},
        historical_screen_unchanged=old['mechanism_screen'],counts=dict(new_training_updates=0,new_policy_samples=0,
            new_neural_forwards=0,new_H_calls=0,new_planner_calls=0,new_geometry_certificates=0,
            new_scene_generation=0,reserved_test_queries=0),
        scope='Arithmetic on saved records and file hashing only; no training, planning, geometry validation or new information-gap search.'))
    target=OUT/'frozen_integrity.json'
    if not target.exists():
        baseline_raw=subprocess.check_output(['git','show',BASE+':project_management/GIT_MANIFEST.json'])
        manifest=json.loads(baseline_raw);tick=time.perf_counter();fail=[]
        allowed={'PROJECT_VERSION.json','README.md','rc3_constructive/README.md','rc3_constructive/findings.md',
            'rc3_constructive/EXPERIMENT_RESULT_TO_CLAIM.md','rc3_constructive/paper/EXPERIMENT_TRACKER.md','rc3_constructive/paper/MANIFEST.md'}
        changed=[]
        for name,e in manifest['files'].items():
            p=ROOT.parent/name
            if not p.is_file() or p.stat().st_size!=e['bytes'] or digest(p)!=e['sha256']:
                (changed if name in allowed else fail).append(name)
        assert not fail,fail
        write('frozen_integrity.json',dict(base_commit=BASE,baseline_manifest_sha256=hashlib.sha256(baseline_raw).hexdigest(),
            original_files_checked=len(manifest['files']),original_bytes_checked=manifest['total_bytes'],
            original_research_payloads_unchanged=True,status_documentation_changes=changed,unexpected_changes=fail,
            elapsed_s=time.perf_counter()-tick,scope='Original manifest payloads checked against base Git commit. Manifest self excluded by original contract. Status docs may change for closeout; no original code, weight, result, plan or training-log changes.'))
    print(json.dumps(dict(B_vs_A=improvements['B_vs_A']['mean'],B_vs_warm=improvements['B_vs_warm']['mean'],
        A_vs_warm=improvements['A_vs_warm']['mean'],balance=balance,training=training),ensure_ascii=False))


if __name__=='__main__':main()
