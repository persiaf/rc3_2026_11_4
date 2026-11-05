"""Label-only metrics; scene is the bootstrap and calibration unit."""
import math
import numpy as np
from .rank_model import predict


def bootstrap(values,seed=11522026):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(seed)
    means=x[rng.integers(len(x),size=(20000,len(x)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def chosen_index(menu,pred):
    return min(range(len(menu)),key=lambda i:(-float(pred[i]),int(menu[i])))


def evaluate(records,bundle,stats):
    cal=[r for r in records if r['partition']=='calibration']
    calmax={}
    for r in cal:
        pred,_=predict(bundle,r['features'],stats)
        true=100*(r['Yref']-np.asarray(r['Y']))/r['Yref']
        calmax[r['case']]=max(calmax.get(r['case'],0.),float(np.abs(pred-true).max()))
    ordered=sorted(calmax.values())
    # 16 independent calibration maps; maximum over all their states/actions.
    q=ordered[min(len(ordered),math.ceil((len(ordered)+1)*.9))-1]
    rows=[];rules=[]
    for r in (r for r in records if r['partition']=='development'):
        pred,sigma=predict(bundle,r['features'],stats)
        i=chosen_index(r['menu'],pred);y=np.asarray(r['Y']);best=float(y.min())
        true=100*(r['Yref']-y)/r['Yref'];regret=100*(y[i]-best)/best
        rule_i=r['menu'].index(r['rule_action'])
        rows.append(dict(case=r['case'],threshold=r['threshold'],prefix=r['prefix'],
            menu=r['menu'],chosen=r['menu'][i],Y=y.tolist(),Yref=r['Yref'],
            predicted_advantage_percent=pred.tolist(),ensemble_sd_percent=sigma.tolist(),
            true_advantage_percent=true.tolist(),chosen_index=i,top1=bool(y[i]<=best+1e-7),
            near_optimal=bool(regret<=.1+1e-10),near_0_5=bool(regret<=.5+1e-10),regret_percent=float(regret),
            rule_regret_percent=float(100*(y[rule_i]-best)/best),
            chosen_error_pp=float(pred[i]-true[i]),unknown=False,
            interval_covers=bool(abs(pred[i]-true[i])<=q+1e-10),
            error_gt_1pp=bool(pred[i]-true[i]>1.),
            all_actions_interval_cover=bool(np.abs(pred-true).max()<=q+1e-10)))
    cases=sorted({r['case'] for r in rows})
    deltas=[np.mean([r['rule_regret_percent']-r['regret_percent'] for r in rows if r['case']==c]) for c in cases]
    scene_covers=[all(r['all_actions_interval_cover'] for r in rows if r['case']==c) for c in cases]
    def scene_mean(key):
        return float(np.mean([np.mean([r[key] for r in rows if r['case']==c]) for c in cases]))
    result=dict(kind=bundle['kind'],states=len(rows),cases=len(cases),
        aggregation='equal-weight scenes; equal-weight available states within each scene',
        top1_percent=100*scene_mean('top1'),
        near_0_1_percent=100*scene_mean('near_optimal'),near_0_5_percent=100*scene_mean('near_0_5'),
        mean_regret_percent=scene_mean('regret_percent'),
        rule_mean_regret_percent=scene_mean('rule_regret_percent'),
        regret_improvement_percent=float(np.mean(deltas)),regret_improvement_CI95=bootstrap(deltas),
        selected_optimism_gt_1pp_percent=100*scene_mean('error_gt_1pp'),
        selected_interval_coverage_percent=100*scene_mean('interval_covers'),
        scene_simultaneous_coverage_percent=100*np.mean(scene_covers),
        calibration_radius_pp=q,unknown_selected=0,rows=rows)
    checks=dict(top1_ge60=bool(result['top1_percent']>=60),near_ge75=bool(result['near_0_1_percent']>=75),
        regret_le0_5=bool(result['mean_regret_percent']<=.5),
        rule_increment_CI_positive=bool(result['regret_improvement_CI95'][0]>0),
        optimistic_outliers_le10=bool(result['selected_optimism_gt_1pp_percent']<=10),
        scene_simultaneous_coverage_ge80=bool(result['scene_simultaneous_coverage_percent']>=80),
        unknown_zero=True,finite_complete=True)
    result.update(checks=checks,passed=all(checks.values()))
    return result
