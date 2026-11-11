"""Prespecified saved-state calibration, paired physical-parent inference and G3."""
from collections import defaultdict
import csv
import gzip
import math
import numpy as np
from rc3_critic_diagnostics_20261111 import critic as historical
from . import critic

COMPONENTS = historical.COMPONENTS
AGGREGATIONS = ('state_pooled', 'parent_equal')

def evaluate(model, episodes, endpoint, references):
    rows = []
    for episode in episodes:
        prediction = critic.predict_episode(model, episode)
        labels = historical.episode_labels(episode, dtype=__import__('torch').float64)
        count = len(labels['value'])
        if episode['split']=='validation' and episode['parent_id'] in references['parent_ids']:
            raise ValueError('Validation parent leaked into fitted references')
        for t in range(count):
            key = historical.condition_key(episode, t, count)
            reference = references['conditional_constants'].get(key, references['global_constant'])
            for component in COMPONENTS:
                y, p = float(labels[component][t]), float(prediction[component][t])
                if not math.isfinite(y) or not math.isfinite(p): raise ValueError('Nonfinite evaluation')
                rows.append(dict(configuration=model.variant, endpoint=endpoint, split=episode['split'],
                    parent_id=episode['parent_id'], trajectory_id=episode['trajectory_id'],
                    window=episode['window_key'], source_stage=episode['stage'], state_index=t,
                    task_count=episode['task_count'], component=component, target=y, prediction=p,
                    remaining_fraction=float(episode['v'][t,256]),
                    global_constant=references['global_constant']['values'][component],
                    conditional_constant=reference['values'][component],
                    conditional_fallback=key not in references['conditional_constants']))
    return rows

def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        for window, stage in (('ALL','ALL'), (row['window'],'ALL'), ('ALL',row['source_stage']), (row['window'],row['source_stage'])):
            key = (row['configuration'],row['endpoint'],row['split'],window,stage,row['component'])
            grouped[key].append(row)
    result=[]
    for key, group in sorted(grouped.items()):
        parents=defaultdict(list)
        for row in group:
            y,p,g,c=map(float,(row['target'],row['prediction'],row['global_constant'],row['conditional_constant']))
            parents[row['parent_id']].append([(p-y)**2,abs(p-y),p-y,y,p,(g-y)**2,(c-y)**2,float(row['conditional_fallback'])])
        means=np.asarray([np.mean(v,axis=0) for _,v in sorted(parents.items())],dtype=np.float64)
        counts=np.asarray([len(v) for _,v in sorted(parents.items())])
        for agg in AGGREGATIONS:
            weights=counts if agg=='state_pooled' else np.ones(len(counts))
            m=np.average(means,axis=0,weights=weights)
            pb=float(np.average(means[:,2]**2,weights=weights))
            record=dict(zip(('configuration','endpoint','split','window','source_stage','component'),key))
            record.update(aggregation=agg,n_states=len(group),n_parents=len(parents),
                n_trajectories=len({r['trajectory_id'] for r in group}),
                mse=float(m[0]),mae=float(m[1]),bias=float(m[2]),mean_target=float(m[3]),mean_prediction=float(m[4]),
                global_mse=float(m[5]),conditional_mse=float(m[6]),
                ratio_global=float(m[0]/m[5]) if m[5]>0 else None,
                ratio_conditional=float(m[0]/m[6]) if m[6]>0 else None,
                bias_squared=float(m[2]**2),mean_parent_bias_squared=pb,
                within_parent_centered_error=max(0.,float(m[0])-pb),conditional_fallback_fraction=float(m[7]))
            result.append(record)
    return result

def paired_intervals(rows, endpoint):
    parents=defaultdict(lambda:defaultdict(lambda:[0.,0.,0]))
    for row in rows:
        if row['split']!='validation' or row['component']!='value': continue
        p=parents[row['parent_id']][row['configuration']]
        p[0]+=(row['prediction']-row['target'])**2
        p[1]+=(row['conditional_constant']-row['target'])**2
        p[2]+=1
    names=sorted(parents)
    if not names: return dict(endpoint=endpoint,status='INSUFFICIENT',parent_ids=[])
    a=np.asarray([parents[p]['C1'] for p in names]); b=np.asarray([parents[p]['C0'] for p in names])
    if not np.array_equal(a[:,2],b[:,2]): raise ValueError('Unpaired evaluation states')
    draws=np.random.Generator(np.random.PCG64(11622026)).integers(0,len(names),size=(20000,len(names)))
    intervals=[]
    for agg in AGGREGATIONS:
        for reference, denominator in (('conditional_constant',a[:,1]),('C0',b[:,0])):
            numerator=a[:,0]
            if agg=='parent_equal': numerator=numerator/a[:,2]; denominator=denominator/a[:,2]
            den=denominator[draws].sum(1)
            if np.any(den<=0): raise ValueError('Zero reference MSE in paired bootstrap')
            ratios=numerator[draws].sum(1)/den
            low,high=np.quantile(ratios,[.025,.975],method='linear')
            intervals.append(dict(aggregation=agg,reference=reference,point_ratio=float(numerator.sum()/denominator.sum()),
                ci95_low=float(low),ci95_high=float(high)))
    return dict(endpoint=endpoint,status='GO',unit='physical_parent',draws=20000,seed=11622026,
                generator='PCG64',quantile='linear',parent_ids=names,intervals=intervals)

def gate(metrics, intervals, endpoint, selection):
    def find(name,comp,agg,window='ALL'):
        match=[r for r in metrics if r['configuration']==name and r['endpoint']==endpoint and r['split']=='validation'
               and r['component']==comp and r['aggregation']==agg and r['window']==window and r['source_stage']=='ALL']
        return match[0] if len(match)==1 else None
    checks=[]
    def add(name,passed,**evidence): checks.append(dict(check=name,status='GO' if passed else 'FAIL',**evidence))
    for agg in AGGREGATIONS:
        c1,c0=find('C1','value',agg),find('C0','value',agg)
        if c1 is None or c0 is None:
            checks.append(dict(check='value_'+agg,status='INSUFFICIENT'));continue
        ci=next(x for x in intervals['intervals'] if x['aggregation']==agg and x['reference']=='conditional_constant')
        add('value_vs_condition_'+agg,c1['ratio_conditional']<=.95,actual=c1['ratio_conditional'],maximum=.95)
        add('paired_upper_'+agg,ci['ci95_high']<1.,actual=ci['ci95_high'],strict_upper=1.,interval=ci)
        add('value_vs_C0_'+agg,c1['mse']<c0['mse'],C1_mse=c1['mse'],C0_mse=c0['mse'],ratio=c1['mse']/c0['mse'])
        for comp in ('normalized_L','normalized_K','normalized_H'):
            a,b=find('C1',comp,agg),find('C0',comp,agg)
            ratio=a['mse']/b['mse'] if b and b['mse']>0 else None
            add(comp+'_vs_C0_'+agg,ratio is not None and ratio<=1.10,actual=ratio,maximum=1.10,C1_mse=a['mse'],C0_mse=b['mse'])
        for window in ('10x10','4x3','6x6','8x8'):
            row=find('C1','value',agg,window)
            if row is None or row['n_parents']<3:
                checks.append(dict(check='size_'+window+'_'+agg,status='INSUFFICIENT',parents=0 if row is None else row['n_parents'],required=3))
            else: add('size_'+window+'_'+agg,row['ratio_conditional']<=1.,parents=row['n_parents'],actual=row['ratio_conditional'],maximum=1.)
    status='INSUFFICIENT' if any(c['status']=='INSUFFICIENT' for c in checks) else ('GO' if all(c['status']=='GO' for c in checks) else 'FAIL')
    ratios=[find('C1','value',agg)['ratio_conditional'] for agg in AGGREGATIONS]
    mid_stop=endpoint==128 and all(r>1.25 for r in ratios)
    if mid_stop: status='FAIL'
    return dict(schema='rc3-r04-m1-g3-decision-v1',status=status,endpoint=endpoint,checks=checks,
        midpoint_stop_triggered=mid_stop,midpoint_rule='C1 value/conditional >1.25 in both aggregations at128 stops paired C0/C1 block',
        current_ratios=ratios,endpoint256='NOT_RUN_REGISTERED_MIDPOINT_STOP' if mid_stop else ('COMPLETE' if endpoint==256 else 'PENDING'),
        G2='NOT_PASSED_REAL_ACTOR_ADAM_TRANSACTION_NOT_EXECUTED',M2='NOT_AUTHORIZED_NOT_RUN',M3='NOT_AUTHORIZED_NOT_RUN',
        next_step='G2_FINAL_ACCEPTANCE_SEPARATE_AUTHORIZATION' if status=='GO' and endpoint==256 else 'STOP_THIS_C1_CONFIGURATION',
        interpretation='Offline calibration only. MSE is not route benefit. Failure does not falsify relative Actor H2.')

def write_csv(path, rows):
    if not rows: raise ValueError('Empty evidence table')
    with open(path,'x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
