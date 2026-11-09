"""Saved-data reporting only: never materialize a policy or call an oracle."""
import csv
from collections import Counter,defaultdict
from pathlib import Path
import time
import numpy as np
from rc3_gl_v4_20261109.common import ROOT,HOME,SEED,EXPERIMENT,read,write_new,digest,identity,require,label,read_chain,now


def csv_new(path,rows):
    require(bool(rows),'Empty CSV')
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def chain(folder):
    events,tail=read_chain(folder/'events.jsonl');head=read(folder/'HEAD.json')
    require(not (folder/'HEAD.pending').exists() and head['sha256']==tail and head['sequence']==len(events),'Unclosed/tampered journal')
    return events,tail


def completed(folder):
    events,tail=chain(folder);starts={};rows=[];errors=[]
    for e in events:
        if e['event']=='request_start':
            q=e['request'];require(q['id'] not in starts,'Duplicate predebit');starts[q['id']]=q
        if e['event']=='request_error':errors.append(e)
        if e['event']=='request_complete':
            p=ROOT/e['artifact_path'];require(digest(p)==e['artifact_sha256'],'Saved result changed')
            r=read(p);require(r['request']==starts[e['id']] and r['valid'] and r['plan']['validation']['valid_model_plan'],'Wrong/invalid result')
            rows.append((r,e))
    require(len({r['request']['id'] for r,e in rows})==len(rows),'Duplicate result')
    require({p.stem for p in (folder/'results').glob('*.json')}=={r['request']['id'] for r,e in rows},'Orphan result')
    return rows,events,errors


def analyze():
    began=time.perf_counter();out=HOME/'results';require(not out.exists(),'Do not overwrite analysis')
    deployment,dep_events,dep_errors=completed(HOME/'runs/DEPLOYMENT')
    require(not dep_errors and len(deployment)==396,'Deployment not complete')
    expected=read(HOME/'DEPLOYMENT_GRID.json')['requests'];require([r['request'] for r,e in deployment]==expected,'Deployment order changed')
    flat=[];groups=defaultdict(list);stable={};dep_counts=Counter();dep_seconds=Counter();dep_prime=Counter()
    dep_pipeline=Counter();dep_internal={k:Counter() for k in ('counts','inclusive_s','exclusive_s')}
    for r,e in deployment:
        q=r['request'];k=(q['case_id'],q['role']);v=(r['raw_route'],r['route'],r['metrics'],r['trace'],r['plan']['assignments'])
        require(k not in stable or stable[k]==v,'Deployment equivalence failed');stable[k]=v
        rec=dict(row_type='deployment_request',case_id=q['case_id'],role=q['role'],mode=q['mode'],navigation=q['kind'],repeat=q['repeat'],
            **r['metrics'],complete_time_s=e['complete_time_s'],tensor_sha256=r['weight']['model_tensor_sha256'],
            raw_route_sha256=r['raw_route_sha256'],route_sha256=r['route_sha256'],actual_mask_logits_trace_sha256=identity(r['trace']),
            artifact_path=e['artifact_path'],artifact_sha256=e['artifact_sha256'])
        flat.append(rec);groups[q['role'],q['mode'],q['kind']].append(e['complete_time_s']);dep_counts.update(e['counts']);dep_seconds.update(e['seconds'])
        dep_pipeline.update(r['pipeline_counters'])
        for k in dep_internal:dep_internal[k].update(r['internal_search'][k])
    for e in dep_events:
        if e['event']=='hot_prime':dep_prime.update(e['counts'])
    timing=[]
    for (role,mode,cache),ts in sorted(groups.items()):
        timing.append(dict(role=role,mode=mode,navigation=cache,n=len(ts),median_s=float(np.median(ts)),p95_s=float(np.quantile(ts,.95))))
    lookup={(x['role'],x['mode'],x['navigation']):x for x in timing};speed=[]
    paired={(r['case_id'],r['role'],r['navigation'],r['repeat'],r['mode']):r for r in flat}
    for role in ('SL256','RL512'):
        for nav in ('cold','hot'):
            for mode in ('D1','D2'):
                base=lookup[role,'D0',nav];target=lookup[role,mode,nav]
                pp=[(r['complete_time_s'],paired[r['case_id'],role,nav,r['repeat'],mode]['complete_time_s']) for r in flat if r['role']==role and r['mode']=='D0' and r['navigation']==nav]
                speed.append(dict(role=role,navigation=nav,mode=mode,paired_requests=len(pp),D0_median_s=base['median_s'],new_median_s=target['median_s'],
                    median_reduction=1-target['median_s']/base['median_s'],median_speedup=base['median_s']/target['median_s'],
                    paired_fraction_faster=float(np.mean([b<a for a,b in pp])),paired_median_reduction=float(np.median([1-b/a for a,b in pp])),
                    D0_p95_s=base['p95_s'],new_p95_s=target['p95_s'],p95_speedup=base['p95_s']/target['p95_s']))
    endpoints,ev_events,ev_errors=completed(HOME/'runs/SCIENCE/DEV216')
    require(not ev_errors and len(endpoints)==216,'Fixed DEV grid incomplete')
    bank=read(HOME/'runs/SCIENCE/NEW_TASK_BANK_FROZEN.json')['cases'];declared={c['id']:c for c in bank}
    costs={};route_rows=[];ev_counts=Counter();ev_seconds=Counter();ev_pipeline=Counter();ev_internal={k:Counter() for k in ('counts','inclusive_s','exclusive_s')}
    for r,e in endpoints:
        q=r['request'];role=q['study_method'];c=declared[q['case_id']]
        require(r['source']['view_identity']==c['view_identity'],'DEV input changed')
        require((q['case_id'],role) not in costs,'Duplicate endpoint');costs[q['case_id'],role]=r['metrics']
        require(e['counts'].get('common_H_invocations',0)==int(role not in ('Fixed3','Detour16')),'Wrong H usage')
        route_rows.append(dict(row_type='fixed_endpoint_evaluation',case_id=q['case_id'],physical_key=r['physical_key'],size='x'.join(map(str,r['size'])),
            endpoint=r['endpoint'],model=role,**r['metrics'],H_gain_J=r['H_gain_J'],complete_time_s=e['complete_time_s'],
            tensor_sha256=r['weight']['model_tensor_sha256'] if r['weight'] else None,
            raw_route_sha256=r['raw_route_sha256'],route_sha256=r['route_sha256'],artifact_path=e['artifact_path'],artifact_sha256=e['artifact_sha256']))
        ev_counts.update(e['counts']);ev_seconds.update(e['seconds'])
        ev_pipeline.update(r['pipeline_counters'])
        for k in ev_internal:ev_internal[k].update(r['internal_search'][k])
    require(len(costs)==216,'DEV coverage differs')
    from rc3_s3_evaluator.statistics import bootstrap,gains
    parents=sorted({c['physical_key'] for c in bank});require(len(parents)==3,'Wrong physical cluster count')
    comparisons=[];percase=[];summaries=[]
    for metric in ('raw_J','J'):
        for control in ('A128','SL256','C128','Fixed3','Detour16'):
            actor=[];base=[]
            for p in parents:
                ids=[c['id'] for c in bank if c['physical_key']==p];require(len(ids)==12,'Wrong child count per parent')
                actor.append(np.mean([costs[c,'B128'][metric] for c in ids]));base.append(np.mean([costs[c,control][metric] for c in ids]))
            vals=gains(np.asarray(base),np.asarray(actor));stats=bootstrap(vals)
            comparisons.append(dict(actor='B128',control=control,metric=metric,**stats,parent_gains=vals.tolist(),
                exploratory_positive=bool(stats['mean']>0 and stats['lower95']>0)))
            for c in bank:
                cid=c['id'];b=costs[cid,control][metric];v=costs[cid,'B128'][metric]
                percase.append(dict(case_id=cid,size='x'.join(map(str,c['size'])),endpoint=c['endpoint'],control=control,metric=metric,
                    control_cost=b,B128=v,gain=(b-v)/b))
    for role in ('SL256','A128','B128','C128','Fixed3','Detour16'):
        rr=[r for r in route_rows if r['model']==role]
        for size in ('ALL','4x3','6x6','8x8'):
            ss=rr if size=='ALL' else [r for r in rr if r['size']==size]
            summaries.append(dict(model=role,size=size,n=len(ss),**{k:float(np.mean([r[k] for r in ss])) for k in ('J','L','K','raw_J','raw_L','raw_K','H_gain_J')}))
    training=[];arm_resources={};ab={};all_operations=Counter()
    for arm in ('A128','B128','C128'):
        folder=HOME/'runs/SCIENCE'/arm;c=read(folder/'COMPLETE.json');require(c['updates']==128,'Missing fixed endpoint')
        events,_=chain(folder);starts=[e for e in events if e['event']=='update_start'];ends=[e for e in events if e['event']=='update_complete']
        require(len(starts)==len(ends)==128,'Optimizer predebits/closures differ')
        terminal_starts=[e for e in events if e['event']=='terminal_request_predebit'];terminal_done=[e for e in events if e['event']=='terminal_request_complete']
        require(len(terminal_starts)==len(terminal_done)==(0 if arm=='C128' else 1536),'RL terminal predebit count differs')
        rows=[]
        for u in range(1,129):
            p=folder/'updates'/f'update_{u:04}.json';d=read(p);rows.append(d)
            require(digest(p)==ends[u-1]['update_artifact_sha256'] and d['optimizer_updated'] and d['cursor_after']==u,'Wrong update source/cursor')
            rec=dict(row_type='training_update',model=arm,update=u,loss=d['loss'],gradient_norm=d['gradient_norm_before_clip'],
                gradient_norm_after_clip=d['gradient_norm_after_clip'],clipped=d['gradient_norm_before_clip']>1.,tensor_sha256=d['model_tensor_sha256'],
                artifact_path=label(p),artifact_sha256=digest(p),full_gradient_weight_artifact=d['full_gradient_and_weight_artifact']['path'])
            if arm!='C128':
                ts=[r for x in d['instances'] for r in x['rows']];require(len(ts)==12,'Wrong trajectories per update')
                for inst in d['instances']:
                    require(inst['mixture_weights']==[1.,.25,.25],'Mixture weights changed')
                    ret=np.array([r['return_'] for r in inst['rows']]);advantages=ret-(ret.sum()-ret)/2
                    require(np.allclose(advantages,inst['advantages'],rtol=0,atol=1e-10),'LOO mismatch')
                    expected_loss=sum(-w*a*r['logprob_sum'] for w,a,r in zip([1.,.25,.25],advantages,inst['rows']))/1.5
                    require(abs(expected_loss-inst['loss'])<1e-6,'Weighted policy loss mismatch')
                    if arm=='B128':require(all(r['forced_first'] is None and (not r['actions'] or r['logprobs'][0] is not None) for r in inst['rows']),'B omitted/forced first action')
                rec.update(J=float(np.mean([r['J'] for r in ts])),raw_J=float(np.mean([r['raw_J'] for r in ts])),
                    L=float(np.mean([r['L'] for r in ts])),K=float(np.mean([r['K'] for r in ts])),
                    raw_L=float(np.mean([r['raw_L'] for r in ts])),raw_K=float(np.mean([r['raw_K'] for r in ts])),
                    zero_signal=d['zero_signal'],zero_advantage_rows=sum(a==0 for inst in d['instances'] for a in inst['advantages']))
            else:require(all(i['teacher_or_H_queries']==0 for i in d['instances']),'C queried oracle')
            training.append(rec)
        if arm!='C128':ab[arm]=rows
        arm_resources[arm]=dict(updates=128,terminal_predebits=len(terminal_starts),terminal_success=len(terminal_done),
            fresh_Adam=read(folder/'WEIGHT_BRANCH.json')['fresh_Adam'],initial_tensor=read(folder/'WEIGHT_BRANCH.json')['initial_tensor_sha256'],
            final_tensor=c['tensor_sha256'],ledger=c['ledger'],clipped_updates=sum(d['gradient_norm_before_clip']>1 for d in rows),
            zero_gradient_updates=sum(d['gradient_norm_before_clip']==0 for d in rows))
    require(all(a['specs']==b['specs'] and a['view_identities']==b['view_identities'] for a,b in zip(ab['A128'],ab['B128'])),'A/B data schedule diverged')
    require(len({v['initial_tensor'] for v in arm_resources.values()})==1,'Warm starts differ')
    passed=all(c['exploratory_positive'] for c in comparisons if c['metric']=='J' and c['control'] in ('A128','SL256','C128'))
    mechanism=dict(comparisons=comparisons,candidate_screen='GO_SINGLE_SEED_CANDIDATE_ONLY' if passed else 'REVISE',
        cases=36,physical_parent_clusters=3,optimizer_seeds=1,new_task_draws=6,
        isolated_change='Two forced initial actions become natural sampled/scored actions; original1/.25/.25 weights retained',
        no_final_generalization_or_deployment_claim=True,all_512_A_B_views_identical=True,untrained_layouts_claim=False)
    phase={p:read(HOME/'runs'/p/'PHASE_FINAL.json') for p in ('DEPLOYMENT','SCIENCE')}
    require(all(v['status']=='COMPLETE' and v['wall_s']<v['cap_s'] for v in phase.values()),'Incomplete/over-budget phase')
    prep_events,_=chain(HOME/'runs/SCIENCE/preparation');prep_counts=Counter()
    for e in prep_events:
        if e['event']=='new_DEV_input_frozen':prep_counts.update(e['counts'])
    resource=dict(experiment_id=EXPERIMENT,phase_resources=phase,deployment_requests=396,deployment_optimizer_updates=0,
        deployment_operation_counts=dict(dep_counts),deployment_navigation_prime_counts=dict(dep_prime),deployment_component_seconds=dict(dep_seconds),
        deployment_pipeline_counters=dict(dep_pipeline),deployment_internal_search={k:dict(v) for k,v in dep_internal.items()},
        science=dict(arms=arm_resources,RL_terminal_requests=3072,fixed_endpoint_evaluations=216,total_top_level_requests=3288,
            total_optimizer_updates=384,new_task_draws=6,new_FA_generations=6,physical_layouts=3,
            preparation_recorded_counts=dict(prep_counts),evaluation_operation_counts=dict(ev_counts),evaluation_component_seconds=dict(ev_seconds)),
        evaluation_pipeline_counters=dict(ev_pipeline),evaluation_internal_search={k:dict(v) for k,v in ev_internal.items()},
        scientific_terminal_cache_counting='All3072 requests predebited incl hits; actual H/internal search/IO in each original BudgetedTerminal ledger',
        original60_semantic_reads=0,new_S3_task_reads=0,old_S3_slots_refilled=0,old_budget_reused=False,new_teacher_queries=0,
        engineering=read(HOME/'ENGINEERING_VERIFICATION.json'),registration=read(HOME/'REGISTRATION_ACCOUNTING.json'),
        analysis_wall_s=time.perf_counter()-began,new_analysis_scientific_calls=0)
    out.mkdir()
    csv_new(HOME/'DEPLOYMENT_BENCHMARK.csv',flat)
    csv_new(out/'DEPLOYMENT_TIMING.csv',timing);csv_new(out/'DEPLOYMENT_SPEEDUP.csv',speed)
    csv_new(out/'ENDPOINT_EVALUATION.csv',route_rows);csv_new(out/'MATCHED_PER_CASE.csv',percase);csv_new(out/'METHOD_SUMMARY.csv',summaries)
    csv_new(out/'TRAINING_CURVES.csv',training);csv_new(HOME/'TRAINING_AND_ROUTE_RESULTS.csv',training+route_rows)
    write_new(out/'MECHANISM_COMPARISONS.json',mechanism)
    write_new(HOME/'RESOURCE_ACCOUNTING.json',resource)
    write_new(out/'SUMMARY.json',dict(deployment_equivalence=read(HOME/'runs/DEPLOYMENT/EQUIVALENCE.json'),
        deployment_startup=read(HOME/'runs/DEPLOYMENT/STARTUP.json'),speedups=speed,comparisons=comparisons,candidate_screen=passed,
        method_summary=summaries,original_S3_conclusions_changed=False))
    write_new(out/'READ_ONLY_ANALYSIS_CLOSURE.json',dict(wall_s=time.perf_counter()-began,source_sha256=digest(Path(__file__)),new_scientific_calls=0))
    print(dict(deployment=396,updates=384,RL_terminals=3072,evaluation=216,candidate=passed))


if __name__=='__main__':analyze()
