"""V3 read-only training/label audit; only a numeric NumPy pickle allowlist.

No Torch import, parameter fitting or test-label construction. Model scores use
the shipped portable NPZ. Validation state enumeration is diagnostic extra work,
not deployment timing. The old/new trained weights remain untouched.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import pickle
import pickletools
from pathlib import Path
import sys
import time

import numpy as np

SOURCE=Path.cwd()
OUT=Path.cwd()/'v3_training_audit'


def configure_source(source_root):
    """Locate an extracted original v3 project without platform-specific paths."""
    global SOURCE, NumpyPolicy, Env, FEATURE_NAMES, prepare, route_objective, TeacherLabels
    SOURCE=Path(source_root).expanduser().resolve()
    if not (SOURCE/'rc3_v3').is_dir():
        raise ValueError('--source-root must contain the rc3_v3 directory')
    sys.path.insert(0,str(SOURCE))
    from rc3_v3.model import NumpyPolicy
    from rc3_v3.environment import Env, FEATURE_NAMES
    from rc3_fast.solver import prepare, route_objective
    from rc3_v3.data import TeacherLabels


def numeric_dtype(dtype):
    dtype=np.dtype(dtype)
    if dtype.hasobject or dtype.kind not in 'biufc':
        raise ValueError(f'unsupported dtype {dtype}')
    return dtype


def safe_array(buffer,dtype,shape,order):
    dtype=numeric_dtype(dtype)
    if order not in ('C','F') or any(type(n) is not int or n<0 for n in shape):
        raise ValueError('invalid numeric array layout')
    return np.frombuffer(buffer,dtype=dtype).reshape(shape,order=order)


def safe_scalar(dtype,raw):
    dtype=numeric_dtype(dtype)
    if len(raw)!=dtype.itemsize: raise ValueError('invalid scalar byte count')
    return np.frombuffer(raw,dtype=dtype,count=1)[0]


class NumericReader(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','dtype'):np.dtype,('collections','Counter'):collections.Counter,
            ('numpy._core.numeric','_frombuffer'):safe_array,
            ('numpy.core.numeric','_frombuffer'):safe_array,
            ('numpy._core.multiarray','scalar'):safe_scalar,
            ('numpy.core.multiarray','scalar'):safe_scalar}
        if (module,name) not in allowed: raise ValueError(f'Forbidden global: {module}.{name}')
        return allowed[module,name]


def safe_load_data(path):
    with path.open('rb') as handle:
        value=NumericReader(handle).load()
        if handle.read(1): raise ValueError('Trailing pickle bytes')
    return value


def q(values):
    values=np.asarray(values,dtype=float)
    return np.quantile(values,[0,.25,.5,.75,.9,.99,1]).tolist() if len(values) else []


def inspect_predictions(policy,states):
    rows=[]
    detour=FEATURE_NAMES.index('detour_excess')
    for row in states:
        features,mask,gains=row['features'],row['mask'],row['gains']
        logits,pred,binary=policy.outputs(features,mask)
        legal=np.flatnonzero(mask)
        selections={'rank':int(np.where(mask,logits[:-1],-np.inf).argmax()),
            'gain':int(np.where(mask,pred,-np.inf).argmax()),
            'positive':int(np.where(mask,binary,-np.inf).argmax()),
            'detour':int(np.where(mask,features[:,detour],-np.inf).argmax())}
        if policy.stop_rule=='categorical_argmax':
            active=logits[np.r_[mask,True]]
            prob=np.exp(active-active.max());prob/=prob.sum()
            stop_probability=float(prob[-1])
        else:
            stop_probability=float(1/(1+np.exp(-float(logits[-1]))))
        stop=policy.select(features,mask) is None
        oracle=float(gains[mask].max())
        random_gain=float(gains[mask].mean())
        summary=dict(scene_id=int(row['scene_id']),step=int(row['step']),teacher=row.get('teacher','learned'),
            oracle_gain_pct=oracle,random_expected_gain_pct=random_gain,
            positive_actions=int((gains[mask]>1e-7).sum()),legal_actions=int(mask.sum()),
            stop_probability=stop_probability,stopped=stop,true_stop=oracle<=1e-7,
            rank_stop_gain_pct=0. if stop else float(gains[selections['rank']]),
            **{method+'_gain_pct':float(gains[index]) for method,index in selections.items()},
            **{method+'_index':index for method,index in selections.items()})
        rows.append(summary)
    return rows


def summarize_prediction_rows(rows):
    oracle=np.array([r['oracle_gain_pct'] for r in rows]);positive=oracle>1e-7
    summaries={}
    for name in ['rank','rank_stop','gain','positive','detour','random_expected']:
        chosen=np.array([r[name+'_gain_pct'] for r in rows])
        summaries[name]=dict(mean_selected_gain_pct=float(chosen.mean()),
            positive_selection_rate_on_improvable=float((chosen[positive]>1e-7).mean()) if positive.any() else None,
            mean_per_state_fraction_of_oracle=float((chosen[positive]/oracle[positive]).mean()) if positive.any() else None,
            total_gain_fraction_of_oracle=float(chosen.sum()/oracle.sum()) if oracle.sum() else None,
            exact_best_on_improvable=int((np.abs(chosen[positive]-oracle[positive])<1e-7).sum()))
        if name=='random_expected':
            summaries[name]['positive_selection_rate_on_improvable']=float(np.mean([
                r['positive_actions']/r['legal_actions'] for r in rows if r['oracle_gain_pct']>1e-7])) if positive.any() else None
            summaries[name]['exact_best_on_improvable']=None
    stops=np.array([r['stopped'] for r in rows]);truth=~positive
    return dict(states=len(rows),improvable=int(positive.sum()),all_zero=int(truth.sum()),
        oracle_mean_gain_pct=float(oracle.mean()),methods=summaries,
        stop=dict(predicted=int(stops.sum()),true_positive=int((stops&truth).sum()),
            false_positive=int((stops&~truth).sum()),false_negative=int((~stops&truth).sum()),
            true_negative=int((~stops&~truth).sum()),
            probabilities_all=q([r['stop_probability'] for r in rows]),
            probabilities_zero=q([r['stop_probability'] for r in rows if r['true_stop']]),
            probabilities_improvable=q([r['stop_probability'] for r in rows if not r['true_stop']])) )


def dataset_summary(data,policy):
    rows=data['states'];flat=np.concatenate([r['features'][r['mask']] for r in rows])
    gains=np.concatenate([r['gains'][r['mask']] for r in rows]);positive=gains>1e-7
    budget_column=FEATURE_NAMES.index('remaining_budget_fraction')
    summary=dict(keys=list(data),splits={key:len(v) for key,v in data['scenes'].items()},
        stored_states=len(rows),initial_step_zero_states=sum(r['step']==0 for r in rows),
        noninitial_step_states=sum(r['step']>0 for r in rows),
        distinct_routes=len({(r['scene_id'],tuple(r['route'])) for r in rows}),
        states_with_improved_route=sum(r['current_cost']<r['reference_cost']-1e-7 for r in rows),
        route_changed_from_initial=sum(list(r['route'])!=list(data['initial_routes']['train'][r['scene_id']]) for r in rows),
        legal_window_labels=len(gains),positive_window_labels=int(positive.sum()),zero_window_labels=int((~positive).sum()),
        positive_gain_quantiles=q(gains[positive]),all_gain_quantiles=q(gains),
        negative_label_count=int((gains<0).sum()),
        positive_states=sum((r['gains'][r['mask']]>1e-7).any() for r in rows),
        step_counts=dict(collections.Counter(r['step'] for r in rows)),
        teacher_counts=dict(collections.Counter(r['teacher'] for r in rows)),
        all_zero_by_step=dict(collections.Counter(r['step'] for r in rows if not (r['gains'][r['mask']]>1e-7).any())),
        all_zero_by_teacher=dict(collections.Counter(r['teacher'] for r in rows if not (r['gains'][r['mask']]>1e-7).any())),
        all_zero_source_scene_count=len({r['scene_id'] for r in rows if not (r['gains'][r['mask']]>1e-7).any()}),
        per_scene_state_counts=dict(collections.Counter(collections.Counter(r['scene_id'] for r in rows).values())),
        feature_mean_max_abs_error=float(np.max(np.abs(flat.mean(0)-policy.w['mean']))),
        feature_std_max_abs_error=float(np.max(np.abs(flat.std(0).clip(.05)-policy.w['std']))),
        gain_scale_reconstructed=float(np.quantile(gains[positive],.9)),
        positive_loss_weight_reconstructed=float((~positive).sum()/positive.sum()),
        budget_feature_consistent=all(np.allclose(r['features'][:,budget_column],(8-r['step'])/8) for r in rows),
        initial_source_seed_overlap=set(),
    )
    summary.pop('initial_source_seed_overlap')
    for field in ['positive_states']: summary[field]=int(summary[field])
    return summary


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path.cwd(),
        help='Extracted original project root containing rc3_v3, rc3_fast and rc3_baseline.')
    parser.add_argument('--output-dir',type=Path,default=Path.cwd()/'v3_training_audit',
        help='Directory for newly generated audit JSON files; original source is not modified.')
    args=parser.parse_args()
    configure_source(args.source_root)
    OUT=args.output_dir.expanduser().resolve()
    OUT.mkdir(parents=True,exist_ok=True);root=SOURCE/'rc3_v3';started=time.perf_counter()
    path=root/'data_v3_20261102/data.pkl';data=safe_load_data(path)
    selected=NumpyPolicy.load(root/'deployment/policy.npz')
    summary=dataset_summary(data,selected)
    cases={split:json.loads((root/'data_v3_20261102'/f'{split}_cases.json').read_text()) for split in ['train','validation','test']}
    seeds={split:{r['seed'] for r in rows} for split,rows in cases.items()}
    summary['splits_overlap']={f'{a}/{b}':list(seeds[a]&seeds[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    summary['cache_scene_seed_matches_manifest']={split:[s['seed'] for s in data['scenes'][split]]==[r['seed'] for r in cases[split]] for split in ['train','validation']}
    summary['data_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    summary['model']={'metadata':selected.metadata,'file_bytes':(root/'deployment/policy.npz').stat().st_size,
        'parameters':sum(v.size for k,v in selected.w.items() if k not in ['mean','std'])}
    print('DATA',summary,flush=True)
    training={}
    for run in ['run_v3_20261102','run_v3_stopfix_20261102']:
        training[run]={}
        for path in sorted((root/run).glob('*_seed*.npz')):
            policy=NumpyPolicy.load(path)
            predictions=inspect_predictions(policy,data['states'])
            training[run][path.stem]=summarize_prediction_rows(predictions)
            if path.name=='mlp_seed29.npz' and 'stopfix' in run:
                (OUT/'selected_training_predictions.json').write_text(json.dumps(predictions,indent=2))
            print('TRAIN_PRED',run,path.stem,training[run][path.stem]['methods']['rank'],training[run][path.stem]['stop'],flush=True)
    (OUT/'dataset_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (OUT/'training_prediction_diagnostics.json').write_text(json.dumps(training,ensure_ascii=False,indent=2))

    # Full labels at actual deployed model states, validation only. Other heads
    # and geometric rule are queried on these same states, not granted own traces.
    validation_states=[];validation_rollouts=[]
    for index,(scene,route) in enumerate(zip(data['scenes']['validation'],data['initial_routes']['validation'])):
        problem=prepare(scene);env=Env(problem,8,initial_route=route)
        stats=collections.defaultdict(int);teacher=TeacherLabels(problem,env.reference_cost,stats,FEATURE_NAMES)
        trace=[]
        while True:
            features,mask=env.observe()
            if env.done:break
            gains=teacher.evaluate(env,features,mask)
            validation_states.append(dict(features=features,mask=mask,gains=gains,scene_id=index,step=env.steps,teacher='deployed_mlp'))
            action=selected.select(features,mask)
            if action is None:
                trace.append(dict(step=env.steps,action=None,stop=True));break
            reward,done,info=env.step(action)
            trace.append(dict(step=env.steps-1,action=action,reward=reward,accepted=info['accepted']))
        validation_rollouts.append(dict(scene_id=index,seed=scene['seed'],mode=cases['validation'][index]['mode'],
            calls=env.steps,gain_pct=100*(env.reference_cost-route_objective(problem,env.route))/env.reference_cost,
            label_calls=int(stats['teacher_repair_calls']),trace=trace))
        print('VALIDATION',index+1,'gain',validation_rollouts[-1]['gain_pct'],'elapsed',round(time.perf_counter()-started,2),flush=True)
    val_predictions=inspect_predictions(selected,validation_states)
    val_summary=summarize_prediction_rows(val_predictions)
    val_summary['rollouts']=validation_rollouts
    val_summary['selected_validation_gain_pct']=float(np.mean([r['gain_pct'] for r in validation_rollouts]))
    val_summary['stored_selected_validation_gain_pct']=selected.metadata['val_gain_pct']
    val_summary['label_repair_calls']=sum(r['label_calls'] for r in validation_rollouts)
    (OUT/'validation_prediction_diagnostics.json').write_text(json.dumps(val_summary,ensure_ascii=False,indent=2))
    (OUT/'selected_validation_predictions.json').write_text(json.dumps(val_predictions,ensure_ascii=False,indent=2))
    print('VALIDATION_SUMMARY',{k:v for k,v in val_summary.items() if k!='rollouts'},flush=True)
    print('DONE',time.perf_counter()-started,flush=True)


if __name__=='__main__':main()
