"""Independent saved-plan/reward/resource checks and scene-cluster decisions."""
import gzip,json
from collections import defaultdict
import numpy as np
from rc3_baseline.validator import validate
from rc3_fast.solver import vectorized_feasibility
from rc3_fast.scenes import make_complex_scene
from .core import ROOT,WS,read,write,sha

def main():
    p=read(ROOT/'outputs/protocol.json');rows=read(ROOT/'outputs/test_results.json');by={}
    for c in p['splits']['test']:
        scene=make_complex_scene(10,10,c['seed'],c['mode']);_,_,sparse=vectorized_feasibility(scene);by[c['case_id']]=(scene,sparse)
    validations=[]
    for row in rows:
        path=ROOT/f"outputs/plans/{row['case_id']}_{row['rep']}_{row['method']}.json";plan=read(path)
        check=validate(*((by[row['case_id']][0],plan,by[row['case_id']][1])))
        if not check['valid_model_plan'] or abs(row['J']-row['L']-.75*row['K'])>1e-7:raise ValueError('Saved plan validation/cost failed')
        validations.append(dict(file=path.name,sha256=sha(path),valid=True))
    summary={}
    for method in sorted({r['method'] for r in rows}):
        own=[r for r in rows if r['method']==method];gains=[]
        for r in own:
            anchor=next(x['J'] for x in rows if x['case_id']==r['case_id'] and x['rep']==r['rep'] and x['method']=='original_fixed3')
            if r['J']>anchor+1e-7:raise ValueError('Returned plan worsens anchor')
            gains.append(100*(anchor-r['J'])/anchor)
        summary[method]=dict(n=len(own),mean_J=float(np.mean([r['J'] for r in own])),mean_gain_percent=float(np.mean(gains)),
            median_complete_ms=float(np.median([r['complete_s'] for r in own])*1000),p95_ms=float(np.quantile([r['complete_s'] for r in own],.95)*1000),
            overruns=sum(r['overrun'] for r in own),calls=sum(r['calls'] for r in own),fallbacks=sum(r['fallback'] for r in own))
    comparisons={};rng=np.random.default_rng(2026110423)
    for comp in p['main_gate']['comparators']:
        per_case=[]
        for case_id in range(6):
            paired=[]
            for seed in p['seeds']:
                for rep in range(2):
                    multi=next(r for r in rows if r['case_id']==case_id and r['rep']==rep and r['method']==f'{seed}_multi_policy_iteration')
                    method=comp if comp in ('same_interface_rule','original_detour16') else f'{seed}_{comp}'
                    base=next(r for r in rows if r['case_id']==case_id and r['rep']==rep and r['method']==method)
                    paired.append(100*(base['J']-multi['J'])/base['J'])
            per_case.append(float(np.mean(paired)))
        values=np.asarray(per_case);bootstrap=values[rng.integers(0,6,(10000,6))].mean(1);lo,hi=np.quantile(bootstrap,[.025,.975])
        comparisons[comp]=dict(scene_values=per_case,mean_gain_percent=float(values.mean()),ci95=[float(lo),float(hi)],
            passed=bool(values.mean()>=.3 and lo>0),statistical_unit='six scenes; seeds/repetitions averaged within each scene')
    overrun=sum(r['overrun'] for r in rows)
    passed=all(v['passed'] for v in comparisons.values()) and overrun==0
    with (ROOT/'outputs/resource.jsonl').open() as f:events=[json.loads(line) for line in f]
    counted={k:sum(e['n'] for e in events if e['kind']==k) for k in p['limits']}
    resource=read(ROOT/'outputs/resources.json')
    if counted!=resource['counts'] or any(counted[k]>p['limits'][k] for k in counted):raise ValueError('Resource accounting mismatch')
    from rc3_rl_g22.register import frozen_snapshot
    frozen=frozen_snapshot();before=read(ROOT/'outputs/freeze_before.json')
    if {k:before[k] for k in frozen}!=frozen:raise ValueError('Old G2/G21 files changed')
    if sha(WS/'rc3_rl_g22_delivery.zip')!=before['g22_archive_sha']:raise ValueError('Old G22 archive changed')
    for name,identity in before['g22_manifest']['files'].items():
        if sha(WS/name)!=identity['sha256']:raise ValueError('Old G22 payload changed')
    write(ROOT/'outputs/freeze_after.json',dict(all_passed=True,old_versions_unchanged=True))
    write(ROOT/'outputs/analysis.json',dict(all_plans_valid=True,plans=len(rows),summary=summary,comparisons=comparisons,
        overruns=overrun,primary_passed=passed,resource_counts=counted,limits=p['limits'],
        decision='Limited positive mechanism evidence; separate cross-scale protocol needed' if passed else 'STOP local-window-choice+STOP as the DRL core; no more same-direction rounds or loss variants',
        caveat='This bounded small-data experiment does not prove every possible local-search RL algorithm ineffective'))
    write(ROOT/'outputs/plan_validation.json',validations);print(dict(plans=len(rows),primary_passed=passed,counts=counted,comparisons=comparisons))

if __name__=='__main__':main()
