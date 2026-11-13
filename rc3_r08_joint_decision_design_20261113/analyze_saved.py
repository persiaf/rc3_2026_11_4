"""Analyze ONLY already-recorded routes. No historical planning module imported."""
import csv,json,time,sys
from collections import defaultdict
import numpy as np
from .adapters import ROOT,HOME,SavedEvidence,recorded_route

def dump(name,data):
    (HOME/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def table(name,rows):
    with (HOME/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def mean(v):return float(np.mean(v)) if v else None
def compare(a,b):
    aa,bb=a['visits'],b['visits'];sa,sb=set(aa),set(bb);common=sa&sb
    u=[j for j in aa if j in common];v=[j for j in bb if j in common];rank={j:k for k,j in enumerate(v)}
    inv=sum(rank[u[i]]>rank[u[j]] for i in range(len(u)) for j in range(i+1,len(u)));den=len(u)*(len(u)-1)//2
    changed=[i for i in a['assign'] if a['assign'][i][0]!=b['assign'][i][0]]
    arms=[i for i in a['assign'] if a['assign'][i][1]!=b['assign'][i][1]]
    return dict(delta_J=b['J']-a['J'],delta_L=b['L']-a['L'],delta_K=b['K']-a['K'],relative_gain=(a['J']-b['J'])/a['J'],
        source_K=a['K'],target_K=b['K'],source_L=a['L'],target_L=b['L'],source_J=a['J'],target_J=b['J'],
        common_configurations=len(common),source_only_configurations=len(sa-sb),target_only_configurations=len(sb-sa),
        set_jaccard=len(common)/len(sa|sb) if sa|sb else 1.,same_set=sa==sb,same_order=aa==bb,
        common_order_inversions=inv,common_order_pair_count=den,common_order_disagreement=inv/den if den else 0.,
        common_subsequence_order_equal=u==v,assignment_config_changed_fraction=len(changed)/len(a['assign']) if a['assign'] else 0.,
        assignment_arm_changed_fraction=len(arms)/len(a['assign']) if a['assign'] else 0.,
        source_redundant_coverage_edges=a['redundant_coverage_edges'],target_redundant_coverage_edges=b['redundant_coverage_edges'],
        source_noncritical_configs=a['noncritical_configs'],target_noncritical_configs=b['noncritical_configs'],
        source_endpoint_length=a['endpoint_length'],target_endpoint_length=b['endpoint_length'],
        source_top3_legs=a['top3_legs'],target_top3_legs=b['top3_legs'],
        common_undirected_edges=len(a['edges']&b['edges']),
        exact_component_identity_error=abs((b['J']-a['J'])-(b['L']-a['L'])-.75*(b['K']-a['K'])))

def describe(plan,p,eid):
    r=recorded_route(plan,eid);index={j:k for k,j in enumerate(p.candidate_ids)}
    if r.problem_id!=p.problem_id or plan['parent_identity']!=p.parent_id:raise ValueError('Strict plan identity')
    if (plan['entry_global_node'],plan['exit_global_node'])!=(p.entry,p.exit):raise ValueError('Strict endpoint identity')
    if tuple(plan['target_task_ids_global'])!=p.task_ids:raise ValueError('Strict task identity')
    cols=[index[j] for j in r.visit_ids];covered=p.cover[:,cols];counts=covered.sum(1)
    if not (counts>0).all():raise ValueError('Saved route incomplete')
    nodes=[0,*[c+1 for c in cols],0];legs=np.asarray([p.D[a,b] for a,b in zip(nodes,nodes[1:])]);oldlegs=plan['route_leg_costs_m']
    if not np.allclose(legs,oldlegs,rtol=0,atol=1e-7) or abs(float(legs.sum())-r.L)>1e-7:raise ValueError('Stored navigation lengths inconsistent')
    assignment={x['task_id']:(x['candidate_id'],x['arm']) for x in plan['assignments']}
    if len(assignment)!=len(plan['assignments']) or set(assignment)!=set(p.task_ids):raise ValueError('Assignment source incomplete')
    left=np.ones(len(p.task_ids),bool);critical={};service={};taskindex={i:k for k,i in enumerate(p.task_ids)}
    for j,col in zip(r.visit_ids,cols):
        served=np.flatnonzero(left&p.cover[:,col]);expected={p.task_ids[i] for i in served}
        actual={i for i,(c,arm) in assignment.items() if c==j}
        if not expected or actual!=expected:raise ValueError('Saved ordered greedy service differs')
        service[str(j)]=[dict(task_id=i,arm=assignment[i][1]) for i in sorted(actual)]
        for i in actual:
            arm=0 if assignment[i][1]=='L' else 1
            if assignment[i][1] not in ('L','R') or not p.legal_service[taskindex[i],col,arm]:raise ValueError('Saved same-arm violation')
        critical[str(j)]=[p.task_ids[i] for i in np.flatnonzero(p.cover[:,col]&(counts==1))];left[served]=False
    return dict(J=r.J,L=r.L,K=r.K,visits=list(r.visit_ids),assign=assignment,
        redundant_coverage_edges=int(counts.sum()-len(counts)),multiply_covered_tasks=int((counts>1).sum()),
        noncritical_configs=sum(not x for x in critical.values()),critical_tasks=critical,stop_service=service,
        endpoint_length=float(legs[0]+legs[-1]) if len(legs)>1 else float(legs[0]),top3_legs=float(sum(sorted(legs,reverse=True)[:3])),
        edges={tuple(sorted((a,b))) for a,b in zip(['entry',*map(str,r.visit_ids)], [*map(str,r.visit_ids),'exit'])},
        unique_xy=len({p.poses[index[j]][:2] for j in r.visit_ids}))

def main():
    start=time.perf_counter();e=SavedEvidence();rows=[];details=[];route_rows=[];route_details={};paired_records=0
    def route(record,p,key,method):
        out={}
        for phase,plan in [('raw',record['raw_plan']),('H',record['plan'])]:
            d=describe(plan,p,key+':'+phase);out[phase]=d
            route_rows.append(dict(route_id=key,phase=phase,method=method,problem_id=p.problem_id,physical_key=p.physical_key,
                size=f"{record['spec']['width']}x{record['spec']['height']}",J=d['J'],L=d['L'],K=d['K'],unique_xy=d['unique_xy'],
                redundant_coverage_edges=d['redundant_coverage_edges'],multiply_covered_tasks=d['multiply_covered_tasks'],
                noncritical_configs=d['noncritical_configs'],endpoint_length=d['endpoint_length'],top3_legs=d['top3_legs']))
        route_details[key]=dict(problem_id=p.problem_id,raw=dict(visits=out['raw']['visits'],service=out['raw']['stop_service'],critical_tasks=out['raw']['critical_tasks']),
            H=dict(visits=out['H']['visits'],service=out['H']['stop_service'],critical_tasks=out['H']['critical_tasks']))
        return out
    def pair(cohort,tid,p,a,b,source,target,phase,sa,sb):
        nonlocal paired_records
        d=compare(a,b)
        if d['exact_component_identity_error']>1e-7:raise ValueError('Component identity')
        row=dict(cohort=cohort,task_id=tid,problem_id=p.problem_id,physical_key=p.physical_key,size=sa,phase=phase,source_method=source,target_method=target,**d)
        rows.append(row);common=set(a['visits'])&set(b['visits']);paired_records+=1
        details.append(dict(row_index=len(rows)-1,problem_id=p.problem_id,source_evidence=sb[0],target_evidence=sb[1],
            source_only_configurations=sorted(set(a['visits'])-set(b['visits'])),target_only_configurations=sorted(set(b['visits'])-set(a['visits'])),
            common_source_order=[j for j in a['visits'] if j in common],common_target_order=[j for j in b['visits'] if j in common],
            assignment_changed_task_ids=[i for i in a['assign'] if a['assign'][i][0]!=b['assign'][i][0]],
            arm_changed_task_ids=[i for i in a['assign'] if a['assign'][i][1]!=b['assign'][i][1]]))

    d5='rc3_d5_revision_20261112/'
    full=e.json(d5+'FULL_TEACHER_MANIFEST.json')['records'];local=e.json(d5+'ORIGINAL_LOCAL_TEACHERS.json')['records'];refs=e.json(d5+'REFERENCE_MANIFEST.json')['records']
    meta={**{tid:dict(id=tid,**r) for tid,r in full.items()},**{r['id']:r for r in local}}
    r07ids={x['id'] for x in e.json('rc3_r07_20261113/TASKS.json')['records']}
    train_values={}
    for tid in sorted(meta):
        rr=refs[tid];ref=e.json(rr['path'],rr['sha256']);p=e.snapshot(ref);m=meta[tid]
        if m['physical_key']!=p.physical_key:raise ValueError('Training parent mismatch')
        if tid in full:
            lab=e.json(m['path'],m['sha256']);teacher=lab;sp=lab['problem_identity'];teacher_source=m['path']
        else:
            lab=e.json(m['source']['path'],m['source']['sha256']);sp=lab['view_identity'];teacher=dict(raw_plan=lab['native_plan'],plan=lab['H_result']['plan'],spec=ref['spec']);teacher_source=m['source']['path']
        if sp!=p.problem_id:raise ValueError('Strict teacher/reference pairing')
        a=route(ref,p,rr['path'],'FrozenSL');b=route(teacher,p,teacher_source,'StrongTeacher');size=f"{ref['spec']['width']}x{ref['spec']['height']}"
        train_values[tid]=(p,a,b,size,rr['path'],teacher_source)
        for phase in ('raw','H'):pair('training352',tid,p,a[phase],b[phase],'FrozenSL','StrongTeacher',phase,size,(rr['path'],teacher_source))
        for method,data,path in [('FrozenSL',a,rr['path']),('StrongTeacher',b,teacher_source)]:pair('training352_H_effect',tid,p,data['raw'],data['H'],method,method,'raw_to_H',size,(path+':raw',path+':H'))
        if tid in r07ids:pair('R07_fixed48',tid,p,a['H'],b['H'],'FrozenSL','StrongTeacher','H',size,(rr['path'],teacher_source))
    # Fixed main endpoints only; keep experiments separate. No D5 online-history scan.
    for cohort,base,files in [('D5_fixed4096',d5+'evaluations/development/',{
        'FrozenSL':'baseline_FrozenSL.json','D5-PG':'2026111101_D5-PG_4096.json','D5-SL-U':'2026111101_D5-SL-U_4096.json'}),
        ('D6_fixed256','rc3_d6_20261113/evaluations/development/',{'FrozenSL':'FrozenSL_256.json','D5-PG-A':'A_256.json','D6-PG':'B_256.json','D6-SL':'SL_256.json'})]:
        methods={}
        for method,name in files.items():
            manifest=e.json(base+name);mm={}
            for row in manifest['rows']:
                record=e.archived(row['record']);p=e.snapshot(record)
                if row['physical_key']!=p.physical_key:raise ValueError('DEV physical identity')
                key=base+name+'#'+row['case_id'];out=route(record,p,key,method);mm[row['case_id']]=(p,out,row,key)
                pair(cohort+'_H_effect',row['case_id'],p,out['raw'],out['H'],method,method,'raw_to_H',f"{row['width']}x{row['height']}",(key+':raw',key+':H'))
            methods[method]=mm
        comparisons=[('FrozenSL',m) for m in methods if m!='FrozenSL']
        comparisons+=[('D5-SL-U','D5-PG')] if cohort.startswith('D5_') else [('D6-SL','D6-PG'),('D5-PG-A','D6-PG')]
        for source,target in comparisons:
            if set(methods[source])!=set(methods[target]):raise ValueError('Incomplete matched evaluation cohort')
            for tid in sorted(methods[source]):
                p,a,row,sa=methods[source][tid];pp,b,_,sb=methods[target][tid]
                if p.problem_id!=pp.problem_id:raise ValueError('DEV exact problem pairing')
                for phase in ('raw','H'):pair(cohort,tid,p,a[phase],b[phase],source,target,phase,f"{row['width']}x{row['height']}",(sa,sb))
    table('ROUTE_GAP_RESULTS.csv',rows);table('RECORDED_ROUTE_METRICS.csv',route_rows)
    dump('ROUTE_STRUCTURE_EVIDENCE.json',dict(pairs=details,routes=route_details))
    # Physical-parent means; sizes are reported separately, never treated as extra layouts.
    group=defaultdict(list)
    for r in rows:
        for size in (r['size'],'ALL'):group[(r['cohort'],r['source_method'],r['target_method'],r['phase'],size)].append(r)
    summaries=[];parent_rows=[]
    fields=['relative_gain','delta_J','delta_L','delta_K','source_K','target_K','set_jaccard','same_set','same_order','common_order_disagreement',
        'assignment_config_changed_fraction','assignment_arm_changed_fraction','source_noncritical_configs','target_noncritical_configs','source_redundant_coverage_edges','target_redundant_coverage_edges']
    for key,rr in sorted(group.items()):
        pp=defaultdict(list)
        for r in rr:pp[r['physical_key']].append(r)
        parent=[]
        for pid,rs in sorted(pp.items()):
            q=dict(cohort=key[0],source_method=key[1],target_method=key[2],phase=key[3],size=key[4],physical_key=pid,tasks=len(rs),**{f:mean([r[f] for r in rs]) for f in fields})
            # Separate prior-stage gate convention from mean of per-task relative gains.
            q['gain_of_parent_mean_J']=(mean([r['source_J'] for r in rs])-mean([r['target_J'] for r in rs]))/mean([r['source_J'] for r in rs])
            parent.append(q);parent_rows.append(q)
        s=dict(cohort=key[0],source_method=key[1],target_method=key[2],phase=key[3],size=key[4],tasks=len(rr),parents=len(pp),
            **{f:mean([r[f] for r in parent]) for f in fields},gain_of_parent_mean_J=mean([r['gain_of_parent_mean_J'] for r in parent]),
            task_equal_gain=mean([r['relative_gain'] for r in rr]),target_better=sum(r['delta_J']<-1e-7 for r in rr),tied=sum(abs(r['delta_J'])<=1e-7 for r in rr),target_worse=sum(r['delta_J']>1e-7 for r in rr))
        summaries.append(s)
    table('PARENT_STRUCTURAL_RESULTS.csv',parent_rows);table('ROUTE_GAP_SUMMARY.csv',summaries)
    dump('ANALYSIS_SUMMARY.json',dict(training_pairs=352,training_physical_parents=len({p.physical_key for p,*_ in train_values.values()}),
        R07_subset_tasks=48,total_comparison_rows=len(rows),stored_route_phases=len(route_rows),read_archive_members=e.archives,
        snapshots=len(e.input_cache),source_files=len(e.reads),source_bytes_hash_checked=sum(v['bytes'] for v in e.reads.values()),
        analysis_wall_s=time.perf_counter()-start,maximum_component_identity_error=max(x['exact_component_identity_error'] for x in rows),
        no_new_planning_navigation_native_H_teacher_scene_or_Actor_Critic_calls=True,
        historical_runtime_modules_imported=[m for m in sys.modules if m.startswith(('torch','scipy','rc3_global_local.','rc3_fast.','rc3_v4.'))],
        caveat='Observed structural differences and algebraic component accounting, not independent causal effects of set/assignment/order. D5/D6 endpoint10x10 inference absent; only saved training FrozenSL/teacher full10x10 gap available.',summaries=summaries))
    dump('ANALYSIS_INPUTS.json',dict(files=e.reads,archive_member_scope='Only exact registered D5/D6 exposed-development offsets; no whole old journal scan',archive_members=e.archives))
    print(dict(status='COMPLETE',comparison_rows=len(rows),sources=len(e.reads),archive_members=e.archives,wall_s=time.perf_counter()-start),flush=True)

if __name__=='__main__':main()
