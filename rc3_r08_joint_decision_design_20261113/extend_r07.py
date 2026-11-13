"""Read all 144 existing R07 pairs, without repeating a rollout or H."""
import csv,time,json
from collections import defaultdict
from .adapters import HOME,SavedEvidence
from .analyze_saved import table,dump,describe,compare,mean

def parse(value):
    if value in ('True','False'):return value=='True'
    try:return float(value)
    except ValueError:return value

def main():
    tick=time.perf_counter();e=SavedEvidence();previous=json.loads((HOME/'ANALYSIS_INPUTS.json').read_text(encoding='utf-8'))
    # Same-task immutable inputs were already hash-verified in this R08-A run.
    # Reuse that verification cache; do not repeat the 583MB source audit.
    e.reads.update(previous['files'])
    rows=[{k:parse(v) for k,v in row.items()} for row in csv.DictReader((HOME/'ROUTE_GAP_RESULTS.csv').open(encoding='utf-8'))]
    if any(r['cohort']=='R07_counterfactual144' for r in rows):raise ValueError('Already extended; no duplicate append')
    route_rows=[dict(r) for r in csv.DictReader((HOME/'RECORDED_ROUTE_METRICS.csv').open(encoding='utf-8'))]
    evidence=json.loads((HOME/'ROUTE_STRUCTURE_EVIDENCE.json').read_text(encoding='utf-8'));tasks=e.json('rc3_r07_20261113/TASKS.json')['records'];specs={r['id']:r['spec'] for r in tasks}
    refs=e.json('rc3_d5_revision_20261112/REFERENCE_MANIFEST.json')['records'];states=0
    for f in sorted((HOME.parent/'rc3_r07_20261113/pairs').glob('*.json')):
        pair=e.json(f.relative_to(HOME.parent).as_posix());s=pair['state'];rr=refs[s['task_id']];ref=e.json(rr['path'],rr['sha256']);p=e.snapshot(ref);data={}
        for branch in ('baseline','alternative'):
            record=e.archived(pair[branch]['record']);key='R07:'+s['state_id']+':'+branch
            if record['problem_identity']!=p.problem_id or record['physical_key']!=p.physical_key or record['state_fingerprint']!=s['fingerprint'] or record['prefix_local']!=s['prefix_local']:raise ValueError('R07 same-prefix identity')
            data[branch]={};detail={}
            for phase,name in [('raw','raw_plan'),('H','plan')]:
                d=describe(record[name],p,key+':'+phase);data[branch][phase]=d
                route_rows.append(dict(route_id=key,phase=phase,method='R07_'+branch,problem_id=p.problem_id,physical_key=p.physical_key,size=s['size'],J=d['J'],L=d['L'],K=d['K'],unique_xy=d['unique_xy'],redundant_coverage_edges=d['redundant_coverage_edges'],multiply_covered_tasks=d['multiply_covered_tasks'],noncritical_configs=d['noncritical_configs'],endpoint_length=d['endpoint_length'],top3_legs=d['top3_legs']))
                detail[phase]=dict(visits=d['visits'],service=d['stop_service'],critical_tasks=d['critical_tasks'])
            evidence['routes'][key]=dict(problem_id=p.problem_id,**detail)
        comparisons=[('R07_counterfactual144','R07_baseline','R07_alternative',phase,data['baseline'][phase],data['alternative'][phase]) for phase in ('raw','H')]
        comparisons += [('R07_counterfactual144_H_effect','R07_'+branch,'R07_'+branch,'raw_to_H',data[branch]['raw'],data[branch]['H']) for branch in ('baseline','alternative')]
        for cohort,source,target,phase,a,b in comparisons:
            d=compare(a,b);rows.append(dict(cohort=cohort,task_id=s['state_id'],problem_id=p.problem_id,physical_key=p.physical_key,size=s['size'],phase=phase,source_method=source,target_method=target,**d))
            common=set(a['visits'])&set(b['visits']);evidence['pairs'].append(dict(row_index=len(rows)-1,problem_id=p.problem_id,source_evidence=pair['baseline']['record'],target_evidence=pair['alternative']['record'],phase=phase,source_only_configurations=sorted(set(a['visits'])-set(b['visits'])),target_only_configurations=sorted(set(b['visits'])-set(a['visits'])),common_source_order=[j for j in a['visits'] if j in common],common_target_order=[j for j in b['visits'] if j in common],assignment_changed_task_ids=[i for i in a['assign'] if a['assign'][i][0]!=b['assign'][i][0]],arm_changed_task_ids=[i for i in a['assign'] if a['assign'][i][1]!=b['assign'][i][1]]))
        states+=1
        if states%24==0:print(dict(saved_R07_states=states),flush=True)
    assert states==144 and len(rows)==3550
    canonical_references(evidence,rows)
    table('ROUTE_GAP_RESULTS.csv',rows);table('RECORDED_ROUTE_METRICS.csv',route_rows);dump('ROUTE_STRUCTURE_EVIDENCE.json',evidence)
    groups=defaultdict(list);summaries=[];parents=[]
    fields=['relative_gain','delta_J','delta_L','delta_K','source_K','target_K','set_jaccard','same_set','same_order','common_order_disagreement','assignment_config_changed_fraction','assignment_arm_changed_fraction','source_noncritical_configs','target_noncritical_configs','source_redundant_coverage_edges','target_redundant_coverage_edges']
    for r in rows:
        for size in (r['size'],'ALL'):groups[(r['cohort'],r['source_method'],r['target_method'],r['phase'],size)].append(r)
    for key,rr in sorted(groups.items()):
        pp=defaultdict(list)
        for r in rr:pp[r['physical_key']].append(r)
        per=[]
        for pid,rs in sorted(pp.items()):
            q=dict(cohort=key[0],source_method=key[1],target_method=key[2],phase=key[3],size=key[4],physical_key=pid,tasks=len(rs),**{f:mean([r[f] for r in rs]) for f in fields})
            q['gain_of_parent_mean_J']=(mean([r['source_J'] for r in rs])-mean([r['target_J'] for r in rs]))/mean([r['source_J'] for r in rs]);per.append(q);parents.append(q)
        summaries.append(dict(cohort=key[0],source_method=key[1],target_method=key[2],phase=key[3],size=key[4],tasks=len(rr),parents=len(pp),**{f:mean([r[f] for r in per]) for f in fields},gain_of_parent_mean_J=mean([r['gain_of_parent_mean_J'] for r in per]),task_equal_gain=mean([r['relative_gain'] for r in rr]),target_better=sum(r['delta_J']<-1e-7 for r in rr),tied=sum(abs(r['delta_J'])<=1e-7 for r in rr),target_worse=sum(r['delta_J']>1e-7 for r in rr)))
    table('PARENT_STRUCTURAL_RESULTS.csv',parents);table('ROUTE_GAP_SUMMARY.csv',summaries)
    summary=json.loads((HOME/'ANALYSIS_SUMMARY.json').read_text(encoding='utf-8'));summary.update(total_comparison_rows=len(rows),stored_route_phases=len(route_rows),read_archive_members=previous['archive_members']+e.archives,source_files=len(e.reads),summaries=summaries,maximum_component_identity_error=max(x['exact_component_identity_error'] for x in rows))
    summary['R07_extension']=dict(states=states,physical_parents=48,new_archive_reads=e.archives,wall_s=time.perf_counter()-tick,same_task_immutable_hash_cache_reused=True,new_scientific_calls=0)
    summary['first_pass_analysis_wall_s']=summary['analysis_wall_s'];summary['analysis_wall_s']+=summary['R07_extension']['wall_s']
    summary['first_pass_source_bytes_hash_checked']=summary['source_bytes_hash_checked'];summary['source_bytes_hash_checked']=sum(v['bytes'] for v in e.reads.values())
    dump('ANALYSIS_SUMMARY.json',summary);dump('ANALYSIS_INPUTS.json',dict(files=e.reads,archive_member_scope=previous['archive_member_scope']+'; plus144 prior R07 pairs x2 exact stored members',archive_members=previous['archive_members']+e.archives))
    print(dict(status='COMPLETE',comparison_rows=len(rows),R07_states=states,wall_s=time.perf_counter()-tick),flush=True)

def canonical_references(evidence,rows):
    for d in evidence['pairs']:
        row=rows[d['row_index']];phase=row['phase'];sid=row['task_id']
        if row['cohort'].startswith('R07_counterfactual144'):
            if phase=='raw_to_H':
                src=dst='R07:'+sid+':'+row['source_method'].removeprefix('R07_')
            else:src='R07:'+sid+':baseline';dst='R07:'+sid+':alternative'
            pair=json.loads((HOME.parent/'rc3_r07_20261113/pairs'/(sid.replace(':','_')+'.json')).read_text(encoding='utf-8'))
            d['source_archive']=pair['baseline' if src.endswith(':baseline') else 'alternative']['record']
            d['target_archive']=pair['baseline' if dst.endswith(':baseline') else 'alternative']['record']
            d['source_evidence']=src;d['target_evidence']=dst
        else:
            src=d['source_evidence'];dst=d['target_evidence']
            if phase=='raw_to_H':src=src.removesuffix(':raw');dst=dst.removesuffix(':H')
        sp,tp=('raw','H') if phase=='raw_to_H' else (phase,phase)
        if src not in evidence['routes'] or dst not in evidence['routes']:raise ValueError('Unresolvable recorded route index')
        d.update(source_route_key=src,target_route_key=dst,source_phase=sp,target_phase=tp)
    evidence['canonical_reference_schema']='source_route_key/source_phase and target_route_key/target_phase; archive references separate'

def repair_metadata():
    tick=time.perf_counter();rows=[{k:parse(v) for k,v in r.items()} for r in csv.DictReader((HOME/'ROUTE_GAP_RESULTS.csv').open(encoding='utf-8'))]
    ev=json.loads((HOME/'ROUTE_STRUCTURE_EVIDENCE.json').read_text(encoding='utf-8'));canonical_references(ev,rows);dump('ROUTE_STRUCTURE_EVIDENCE.json',ev)
    inp=json.loads((HOME/'ANALYSIS_INPUTS.json').read_text(encoding='utf-8'));removed=inp['files'].pop('rc3_r08_joint_decision_design_20261113/ANALYSIS_INPUTS.json',None);dump('ANALYSIS_INPUTS.json',inp)
    s=json.loads((HOME/'ANALYSIS_SUMMARY.json').read_text(encoding='utf-8'))
    s['first_pass_analysis_wall_s']=s['analysis_wall_s'];s['analysis_wall_s']=s['first_pass_analysis_wall_s']+s['R07_extension']['wall_s']
    s['first_pass_source_bytes_hash_checked']=s['source_bytes_hash_checked'];s['source_bytes_hash_checked']=sum(v['bytes'] for v in inp['files'].values());s['source_files']=len(inp['files'])
    s['source_hash_scope']='Unique opened immutable source files; reused same-task verified hash cache not hashed/charged again. Archive members separately checked by SHA; derived own index excluded.'
    s['metadata_correction']=dict(canonical_route_phase_links=len(ev['pairs']),R07_same_branch_H_effect_repaired=True,derived_self_index_removed=removed is not None,complete_request_or_archive_reads=0,wall_s=time.perf_counter()-tick)
    dump('ANALYSIS_SUMMARY.json',s);print(dict(status='METADATA_REPAIRED',pairs=len(rows),new_scientific_calls=0,wall_s=time.perf_counter()-tick),flush=True)

if __name__=='__main__':
    import sys
    repair_metadata() if '--metadata-only' in sys.argv else main()
