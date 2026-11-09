"""Analyze only saved fixed GL-v5 outcomes; no scientific callbacks."""
import csv
import json
import math
import time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from rc3_gl_v5_20261109.common import ROOT,HOME,EXPERIMENT,read,write_new,digest,require,label


def csv_new(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def main():
    start=time.perf_counter();folder=HOME/'runs/SCIENCE';final=read(folder/'PHASE_FINAL.json')
    require(final['status']=='COMPLETE','No complete science outcome')
    done=read(folder/'G128/COMPLETE.json');require(done['updates']==128,'Not fixed G128')
    r=read(HOME/'INPUT_REGISTRY.json');parents=r['new_development_parents'];mode={p['physical_key']:p['mode'] for p in parents}
    updates=[read(folder/'G128/updates'/f'update_{u:04}.json') for u in range(1,129)]
    training=[];samples=[];gradients=[]
    for u,d in enumerate(updates,1):
        training.append(dict(row_type='G_training_update',update=u,study_method='G128',loss=d['loss'],
            gradient_norm_before_clip=d['gradient_norm_before_clip'],gradient_norm_after_clip=d['gradient_norm_after_clip'],
            clipped=d['gradient_norm_before_clip']>1.,Adam_step_l2=d['Adam_step_l2'],Adam_relative_step_l2=d['Adam_relative_step_l2'],
            elapsed_s=d['elapsed_s'],tensor_sha256=d['model_tensor_sha256']))
        for i,(spec,instance,g) in enumerate(zip(d['specs'],d['instances'],d['instance_gradient_norms'])):
            reference=instance['reference'];size=f"{spec['width']}x{spec['height']}"
            gradients.append(dict(update=u,slot=i,size=size,mode=spec['mode'],gradient_batch_scaled_l2=g['batch_scaled_l2']))
            for row,adv in zip(instance['rows'],instance['advantages']):
                require(abs(adv-(reference['J']-row['J']))<1e-8,'Wrong saved fixed reference advantage')
                samples.append(dict(update=u,slot=i,size=size,mode=spec['mode'],view_identity=instance['problem'],role=row['role'],
                    sample_J=row['J'],sample_raw_J=row['raw_J'],sample_L=row['L'],sample_K=row['K'],
                    reference_J=reference['J'],reference_raw_J=reference['raw_J'],reference_request=reference['request'],
                    sample_minus_reference_J=row['J']-reference['J'],advantage=adv,logprob_sum=row['logprob_sum'],
                    actions=len(row['actions']),H_gain_J=row['raw_J']-row['J'],sampling_seed=row['sampling_seed']))
    events=[json.loads(s) for s in (folder/'DEV432/events.jsonl').read_text(encoding='utf8').splitlines()]
    evals=[];sumcounts=Counter();internal=Counter()
    for event in events:
        if event['event']!='request_complete':continue
        p=ROOT/event['artifact_path'];require(digest(p)==event['artifact_sha256'],'Result bytes changed')
        x=read(p);role=x['study_method'];q=x['request'];require(role==q['study_method'],'Role mismatch')
        metrics=x['metrics'];evals.append(dict(row_type='fixed_endpoint_evaluation',case_id=x['case_id'],physical_key=x['physical_key'],
            mode=mode[x['physical_key']],size='x'.join(map(str,x['size'])),endpoint=x['endpoint'],study_method=role,
            checkpoint_role=x['checkpoint_role'],actor_file=x['actor_file'],actor_sha256=x['actor_sha256'],tensor_sha256=x['tensor_sha256'],
            training_run=x['training_run'],fixed_update=x['fixed_update'],**metrics,H_gain_J=x['H_gain_J'],
            complete_time_s=event['complete_time_s'],raw_route_sha256=x['raw_route_sha256'],route_sha256=x['route_sha256'],
            artifact_path=event['artifact_path'],artifact_sha256=event['artifact_sha256']))
        sumcounts.update(event['counts']);internal.update(x['internal_search']['counts'])
    require(len(evals)==432 and len(samples)==1536,'Incomplete grid')
    grouped=defaultdict(list)
    for x in evals:grouped[(x['study_method'],x['physical_key'])].append(x)
    keys=sorted(mode);rng=np.random.Generator(np.random.PCG64(11642026))
    strata=[[i for i,k in enumerate(keys) if mode[k]==m] for m in ('mixed','blocked')]
    # Preserve registered 6/6 composition in every true-parent bootstrap draw.
    draws=np.concatenate([rng.choice(indices,size=(20000,6),replace=True) for indices in strata],axis=1)
    comparisons=[]
    for metric in ('raw_J','J'):
        for control in ('SL256','B128','C128','Fixed3','Detour16'):
            gains=[]
            for k in keys:
                require(len(grouped[('G128',k)])==len(grouped[(control,k)])==6,'Not six tasks per physical parent')
                g=np.mean([v[metric] for v in grouped[('G128',k)]]);c=np.mean([v[metric] for v in grouped[(control,k)]])
                gains.append(float((c-g)/c))
            gains=np.array(gains);interval=np.quantile(gains[draws].mean(1),[.025,.975],method='linear')
            comparisons.append(dict(metric=metric,control=control,mean=float(gains.mean()),lower95=float(interval[0]),upper95=float(interval[1]),
                parent_keys=keys,parent_gains=gains.tolist(),positive=bool(gains.mean()>0 and interval[0]>0)))
    candidate=all(c['positive'] for c in comparisons if c['metric']=='J' and c['control'] in ('SL256','B128','C128'))
    summaries=[]
    for dim,values in [('ALL',['ALL']),('size',['4x3','6x6','8x8']),('mode',['mixed','blocked'])]:
        for value in values:
            for role in ('SL256','B128','C128','G128','Fixed3','Detour16'):
                xs=[x for x in evals if x['study_method']==role and (dim=='ALL' or x[dim]==value)]
                summaries.append(dict(group=dim,value=value,study_method=role,n=len(xs),
                    **{k:float(np.mean([x[k] for x in xs])) for k in ('J','L','K','raw_J','raw_L','raw_K','H_gain_J')}))
    grad_summary=[]
    for size in ('10x10','4x3','6x6','8x8'):
        xs=[g['gradient_batch_scaled_l2'] for g in gradients if g['size']==size]
        grad_summary.append(dict(size=size,n=len(xs),mean=float(np.mean(xs)),median=float(np.median(xs)),p95=float(np.quantile(xs,.95)),max=float(max(xs))))
    ref=done['reference_state'];refcounts=Counter();refseconds=Counter()
    for op in ref['operations']:refcounts.update(op['counts']);refseconds.update(op['seconds'])
    advantage=np.array([s['advantage'] for s in samples]);old=ROOT/'rc3_gl_v4_20261109/runs/SCIENCE/B128'
    oldds=[read(old/'updates'/f'update_{u:04}.json') for u in range(1,129)]
    training_by_size=[]
    for size in ('10x10','4x3','6x6','8x8'):
        ss=[s for s in samples if s['size']==size]
        training_by_size.append(dict(size=size,n=len(ss),mean_sample_J=float(np.mean([x['sample_J'] for x in ss])),
            mean_reference_J=float(np.mean([x['reference_J'] for x in ss])),mean_advantage=float(np.mean([x['advantage'] for x in ss])),
            std_advantage=float(np.std([x['advantage'] for x in ss])),negative_advantages=sum(x['advantage']<0 for x in ss),
            mean_H_gain_J=float(np.mean([x['H_gain_J'] for x in ss]))))
    summary=dict(candidate_screen=candidate,comparisons=comparisons,method_summary=summaries,gradient_by_size=grad_summary,training_by_size=training_by_size,
        advantage=dict(mean=float(advantage.mean()),std=float(advantage.std()),min=float(advantage.min()),max=float(advantage.max()),
            positive=int((advantage>0).sum()),negative=int((advantage<0).sum()),zero=int((advantage==0).sum()),
            quantiles={str(q):float(np.quantile(advantage,q)) for q in (0,.05,.25,.5,.75,.95,1)}),
        training=dict(clipped_updates=sum(d['gradient_norm_before_clip']>1 for d in updates),
            zero_gradient_updates=sum(d['gradient_norm_before_clip']==0 for d in updates),
            G_update_inclusive_s=sum(d['elapsed_s'] for d in updates),old_B_update_inclusive_s=sum(d['elapsed_s'] for d in oldds),
            timing_scope='Recorded internal update clocks, not paired speed experiment; no data preparation/checkpoint charges omitted from outer wall',
            Adam_step_l2_mean=float(np.mean([d['Adam_step_l2'] for d in updates])),
            mean_sample_J=float(np.mean([s['sample_J'] for s in samples])),mean_reference_J=float(np.mean([s['reference_J'] for s in samples])),
            extra_reference_requests_vs_B=512,extra_terminal_requests_fraction=512/1536))
    resource=dict(experiment_id=EXPERIMENT,phase=final,sample_requests=done['sample_requests'],reference_requests=done['reference_requests'],
        evaluation_requests=len(evals),total_requests=done['sample_requests']+done['reference_requests']+len(evals),optimizer_updates=128,
        sample_training_ledger=done['ledger'],reference_ledger=ref['ledger'],reference_counts=dict(refcounts),reference_component_seconds=dict(refseconds),
        evaluation_operation_counts=dict(sumcounts),evaluation_internal_search=dict(internal),
        engineering=read(HOME/'ENGINEERING_VERIFICATION.json'),preparation_events=[json.loads(s) for s in (folder/'preparation/events.jsonl').read_text(encoding='utf8').splitlines()],
        original60_and_new_S3_semantic_access=0,old_A_B_C_retrained=False,deployment_grid_rerun=False,
        old_missing_slots_refilled=False,new_teacher_queries=0,nav_solves=0,
        timing_note='Inclusive nested component clocks must not be blindly summed. Outer shared7200 contains registration, tests, all reference/training/evaluation/preparation/checkpoints and IO.')
    (HOME/'results').mkdir(exist_ok=True)
    csv_new(HOME/'TRAINING_CURVES.csv',training);csv_new(HOME/'SAMPLE_REFERENCE_COMPARISON.csv',samples)
    csv_new(HOME/'GRADIENTS_BY_INSTANCE.csv',gradients);csv_new(HOME/'TRAINING_AND_ROUTE_RESULTS.csv',training+evals)
    csv_new(HOME/'results/ENDPOINT_EVALUATION.csv',evals);csv_new(HOME/'results/METHOD_SUMMARY.csv',summaries)
    write_new(HOME/'results/SUMMARY.json',summary);write_new(HOME/'RESOURCE_ACCOUNTING.json',resource)
    write_new(HOME/'results/ANALYSIS_CLOSURE.json',dict(new_scientific_calls=0,wall_s=time.perf_counter()-start,source_sha256=digest(Path(__file__))))
    print(dict(updates=128,requests=resource['total_requests'],evaluation=len(evals),candidate=candidate,comparisons=[{k:v for k,v in c.items() if k not in ('parent_keys','parent_gains')} for c in comparisons]))


if __name__=='__main__':main()
