"""Saved-data equivalence, timing and matched fixed-endpoint comparisons only."""
import csv
import collections
import time
import numpy as np
from .common import *


def completed(folder):
    events,tail=read_chain(folder/'events.jsonl');head=read(folder/'HEAD.json')
    require(not (folder/'HEAD.pending').exists() and head['sha256']==tail and head['sequence']==len(events),'Unclosed event chain')
    starts={};rows=[]
    for e in events:
        if e['event']=='request_start':
            q=e['request'];require(q['id'] not in starts,'Duplicate predebit');starts[q['id']]=q
        if e['event']=='request_error':raise ValueError('Failed scientific request retained')
        if e['event']=='request_complete':
            q=starts[e['id']];p=ROOT/e['artifact_path'];require(digest(p)==e['artifact_sha256'],'Result hash differs')
            r=read(p);require(r['request']==q and r['valid'] and r['plan']['validation']['valid_model_plan'],'Unbilled/invalid result')
            rows.append((r,e))
    require(len(starts)==len(rows) and len({r['request']['id'] for r,e in rows})==len(rows),'Incomplete/duplicate request set')
    require({p.stem for p in (folder/'results').glob('*.json')}==set(starts),'Orphan result')
    return rows,events


def csv_new(path,rows):
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def verify_deployment():
    begin=time.perf_counter();expected=read(HOME/'DEPLOYMENT_GRID.json')['requests'];seen=[];stable={};pairs={};encodes=0;counts=collections.Counter()
    for mode in ('D0','D1','D2'):
        rows,events=completed(HOME/'runs/DEPLOYMENT'/mode)
        require([r['request'] for r,e in rows]==[q for q in expected if q['mode']==mode],'Deployment grid/order differs')
        for r,e in rows:
            q=r['request'];seen.append(q['id']);k=(q['case_id'],q['method'])
            val=(r['raw_route'],r['route'],r['metrics'],r['source']['view_identity'],r['source']['D_sha256'],r['plan']['assignments'])
            require(k not in stable or stable[k]==val,'Deployment lifecycle changed route, J/L/K, source or assignments')
            stable[k]=val
            if q['method'] in ('RL_raw','RL_plus_H'):
                key=(q['case_id'],q['mode'],q['kind'],q['repeat']);pairs.setdefault(key,{})[q['method']]=r['raw_route']
            require(e['counts'].get('common_H_invocations',0)==int(q['method'] in ('untrained','frozen_supervised','continued_SL','RL_plus_H')),'Wrong H usage')
            counts.update(e['counts']);encodes+=e['counts'].get('fresh_encodings',0)
    require(len(seen)==756 and len(set(seen))==756 and encodes==540,'Missing request or task encoding reuse')
    require(all(len(p)==2 and p['RL_raw']==p['RL_plus_H'] for p in pairs.values()),'RL raw/H mismatch')
    for roles in read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')['models'].values():
        for e in roles.values():
            for pk,hk in (('path','checkpoint_sha256'),('sidecar_path','sidecar_sha256'),('commit_path','commit_sha256')):
                require(digest(ROOT/e[pk])==e[hk],'Original training checkpoint/source changed')
    write_new(HOME/'runs/DEPLOYMENT/EQUIVALENCE.json',dict(status='PASS',requests=756,case_method_combinations=len(stable),
        RL_raw_H_pairs=len(pairs),fresh_encodings=encodes,exact_raw_final_routes_costs_and_assignments=True,
        original_model_source_files_unchanged=36,exported_parameter_exactness='actors/ACTOR_INDEX.json and per-actor .pt.json',
        request_operation_counts=dict(counts),wall_s=time.perf_counter()-begin,new_scientific_calls=0))


def analyze():
    began=time.perf_counter();out=HOME/'results';require(not out.exists(),'Do not overwrite analysis');out.mkdir()
    deployment_rows=[];timings=[];groups=collections.defaultdict(list);phase_resources={}
    for phase in ('DEPLOYMENT','A128','B128','DEV54'):
        final=read(HOME/'runs'/phase/'PHASE_FINAL.json');require(final['status']=='COMPLETE','Incomplete phase');phase_resources[phase]=final
    for mode in ('D0','D1','D2'):
        rr,events=completed(HOME/'runs/DEPLOYMENT'/mode)
        for r,e in rr:
            q=r['request'];rec=dict(case_id=q['case_id'],method=q['method'],mode=mode,navigation=q['kind'],repeat=q['repeat'],
                **r['metrics'],complete_time_s=e['complete_time_s'],raw_route_sha256=r['raw_route_sha256'],route_sha256=r['route_sha256'],
                artifact_path=e['artifact_path'],artifact_sha256=e['artifact_sha256'])
            deployment_rows.append(rec);groups[q['method'],mode,q['kind']].append(e['complete_time_s'])
    for (method,mode,cache),values in groups.items():
        timings.append(dict(method=method,mode=mode,navigation=cache,n=len(values),median_s=float(np.median(values)),p95_s=float(np.quantile(values,.95))))
    lookup={(r['method'],r['mode'],r['navigation']):r for r in timings};speed=[]
    for method in METHODS:
        for cache in ('cold','hot'):
            d0=lookup[method,'D0',cache]
            for mode in ('D1','D2'):
                m=lookup[method,mode,cache]
                speed.append(dict(method=method,navigation=cache,mode=mode,D0_median_s=d0['median_s'],new_median_s=m['median_s'],
                    median_speedup=d0['median_s']/m['median_s'],median_reduction=1-m['median_s']/d0['median_s'],
                    D0_p95_s=d0['p95_s'],new_p95_s=m['p95_s'],p95_speedup=d0['p95_s']/m['p95_s']))
    csv_new(out/'DEPLOYMENT_REQUESTS.csv',deployment_rows);csv_new(out/'DEPLOYMENT_TIMING.csv',timings);csv_new(out/'DEPLOYMENT_SPEEDUP.csv',speed)
    devrows,events=completed(HOME/'runs/DEV54');require(len(devrows)==54,'Incomplete fixed DEV grid')
    flat=[];costs={};parents={};strata={}
    for r,e in devrows:
        role=r['study_method'];cid=r['case_id'];parents[cid]=r['physical_key'];strata[cid]=(r['size'],r['endpoint'])
        costs[cid,role]=r['metrics']
        flat.append(dict(case_id=cid,physical_key=r['physical_key'],size='x'.join(map(str,r['size'])),endpoint=r['endpoint'],model=role,
            **r['metrics'],H_gain_J=r['H_gain_J'],complete_time_s=e['complete_time_s'],raw_route_sha256=r['raw_route_sha256'],route_sha256=r['route_sha256'],
            actor_sha256=r['weight']['checkpoint_sha256'],artifact_path=e['artifact_path'],artifact_sha256=e['artifact_sha256']))
    csv_new(out/'DEV54_RESULTS.csv',flat)
    unique=sorted(set(parents.values()));require(len(unique)==3,'Wrong development cluster count')
    from rc3_s3_evaluator.statistics import bootstrap,gains
    comparisons=[];percase=[]
    for metric in ('raw_J','J'):
        for control in ('A128','SL256'):
            actor=[];baseline=[]
            for parent in unique:
                ids=sorted(cid for cid,p in parents.items() if p==parent);require(len(ids)==6,'Missing DEV parent children')
                actor.append(np.mean([costs[c,'B128'][metric] for c in ids]));baseline.append(np.mean([costs[c,control][metric] for c in ids]))
            stats=bootstrap(gains(np.array(baseline),np.array(actor)))
            comparisons.append(dict(actor='B128',control=control,metric=metric,**stats,parent_gains=gains(np.array(baseline),np.array(actor)).tolist(),
                exploratory_positive=bool(stats['mean']>0 and stats['lower95']>0)))
            for cid in sorted(parents):
                percase.append(dict(case_id=cid,size='x'.join(map(str,strata[cid][0])),endpoint=strata[cid][1],control=control,metric=metric,
                    baseline=costs[cid,control][metric],B128=costs[cid,'B128'][metric],gain=(costs[cid,control][metric]-costs[cid,'B128'][metric])/costs[cid,control][metric]))
    csv_new(out/'DEV_MATCHED_PER_CASE.csv',percase)
    training=[];updates={};resources={}
    for arm in ('A128','B128'):
        complete=read(HOME/'runs'/arm/'COMPLETE.json');require(complete['updates']==128 and complete['terminal_requests']==1536,'Training count mismatch')
        rows=[read(HOME/'runs'/arm/'updates'/f'update_{u:04}.json') for u in range(1,129)];updates[arm]=rows
        for d in rows:
            trajectories=[r for instance in d['instances'] for r in instance['rows']]
            require(len(trajectories)==12 and all(np.isfinite(r['J']) for r in trajectories),'Invalid rollout count/return')
            if arm=='B128':require(all(r['forced_first'] is None and all(v is not None for v in r['logprobs']) for r in trajectories),'B forced/omitted first action')
            training.append(dict(arm=arm,update=d['cursor_after'],loss=d['loss'],gradient_norm=d['gradient_norm_before_clip'],
                gradient_norm_after_clip=d['gradient_norm_after_clip'],clipped=d['gradient_norm_before_clip']>1,zero_signal=d['zero_signal'],
                mean_raw_J=float(np.mean([r['raw_J'] for r in trajectories])),mean_H_J=float(np.mean([r['J'] for r in trajectories])),
                tensor_sha256=d['model_tensor_sha256'],gradient_weight_path=d['full_gradient_and_weight_artifact']['path']))
        resources[arm]=dict(optimizer_updates=128,terminal_requests=1536,ledger=complete['ledger'],tensor_sha256=complete['tensor_sha256'])
    require(all(a['specs']==b['specs'] and a['view_identities']==b['view_identities'] for a,b in zip(updates['A128'],updates['B128'])),'A/B inputs diverged')
    require(read(HOME/'runs/A128/WEIGHT_BRANCH.json')['initial_tensor_sha256']==read(HOME/'runs/B128/WEIGHT_BRANCH.json')['initial_tensor_sha256'],'Different warm starts')
    csv_new(out/'TRAINING_CURVES.csv',training)
    passed=all(c['exploratory_positive'] for c in comparisons if c['metric']=='J')
    write_new(out/'MECHANISM_COMPARISONS.json',dict(comparisons=comparisons,candidate_screen='GO_CANDIDATE_ONLY' if passed else 'REVISE',
        physical_parent_clusters=3,optimizer_seeds=1,cases=18,no_S3_confirmation=True,no_convergence_claim=True,
        B_changes_forcing_and_weighting_jointly=True,identical_512_input_views=True))
    write_new(out/'RESOURCE_ACCOUNTING.json',dict(experiment_id=EXPERIMENT,phase_resources=phase_resources,
        deployment_complete_requests=756,deployment_optimizer_updates=0,mechanism=resources,
        mechanism_terminal_requests_training=3072,mechanism_terminal_requests_development=54,mechanism_total=3126,
        scientific_optimizer_updates=256,new_teacher_queries=0,old60_reads=0,old_S3_slots_refilled=0,old_budgets_reused=False,
        deployment_operations=read(HOME/'runs/DEPLOYMENT/EQUIVALENCE.json')['request_operation_counts'],
        analysis_wall_s=time.perf_counter()-began,analysis_scientific_calls=0))
    write_new(out/'SUMMARY.json',dict(deployment_equivalence=read(HOME/'runs/DEPLOYMENT/EQUIVALENCE.json'),
        deployment_startup={m:read(HOME/'runs/DEPLOYMENT'/m/'STARTUP.json') for m in ('D0','D1','D2')},
        speedup=speed,mechanism_comparisons=comparisons,candidate_screen='GO_CANDIDATE_ONLY' if passed else 'REVISE',
        original_S3_conclusions_changed=False))
    print(dict(deployment_requests=756,mechanism_updates=256,mechanism_terminal=3126,candidate=passed))


if __name__=='__main__':analyze()
