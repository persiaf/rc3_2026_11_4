"""Present completed saved results only; no model, optimizer or planning calls."""
import csv,collections,statistics,time
from pathlib import Path
from .common import HOME,ROOT,read,put,unarchive,sha

def write_csv(path,rows):
    if path.exists():raise FileExistsError('Do not overwrite a delivered results table')
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,columns);writer.writeheader();writer.writerows(rows)

def main():
    start=time.perf_counter();status=read(HOME/'STATUS.json')
    if status.get('active_job'):raise RuntimeError('Scientific job is active; do not scan mutable outputs')
    folder=HOME/'tables';folder.mkdir(exist_ok=False);reports={};requests=[];aggregates=[]
    for file in sorted((HOME/'evaluations').glob('*/*.json')):
        report=read(file)
        if not report.get('complete'):continue
        reports[(report['tier'],report['seed'],report['method'],report['endpoint'])]=report
        for row in report['rows']:
            out={k:row.get(k) for k in ('case_id','physical_key','group','mode','width','height','method','seed','endpoint','valid','full_request_s','actor_tensor_sha256','raw_semantics','error')}
            out.update(tier=report['tier'],source=file.relative_to(ROOT).as_posix())
            out.update({k:None if row['metrics'] is None else row['metrics'][k] for k in ('raw_J','raw_L','raw_K','J','L','K')});requests.append(out)
        for group,s in report['summary'].items():
            for aggregation in ('request_mean','parent_equal_mean'):
                out=dict(tier=report['tier'],method=report['method'],seed=report['seed'],endpoint=report['endpoint'],group=group,aggregation=aggregation,
                    planned=s['planned'],valid=s['completed_valid'],success_rate=s['success_rate'],physical_parents=s['physical_parents'],
                    actual_time_p50_s=s['time_p50_s'],actual_time_p95_s=s['time_p95_s'])
                out.update(s[aggregation]);aggregates.append(out)
    write_csv(folder/'COMPLETE_TASK_RESULTS.csv',requests);write_csv(folder/'AGGREGATE_RESULTS.csv',aggregates)
    paired=[];changes=[];records={}
    def record(row):
        key=(row['record']['path'],row['record']['offset'])
        if key not in records:records[key]=unarchive(row['record'])
        return records[key]
    for (tier,seed,method,endpoint),candidate in reports.items():
        if method not in ('D5-PG','Full-PG'):continue
        controls=[r for key,r in reports.items() if key[0]==tier and (key[2]=='FrozenSL' or key[1]==seed and (key[2]=='D5-SL-U' and key[3]==endpoint or key[2]=='D5-SL-C' and endpoint==4096))]
        for control in controls:
            indexed={r['case_id']:r for r in control['rows']}
            if {r['case_id'] for r in candidate['rows']}!=set(indexed):continue
            groups=collections.defaultdict(list)
            for row in candidate['rows']:
                other=indexed[row['case_id']];groups[row['physical_key']].append((row,other))
                if not row['valid'] or not other['valid']:continue
                a,z=record(row),record(other);out=dict(tier=tier,seed=seed,method=method,endpoint=endpoint,control=control['method'],control_endpoint=control['endpoint'],case_id=row['case_id'],physical_key=row['physical_key'])
                for prefix,key in [('raw','raw_plan'),('H','plan')]:
                    aa=a[key]['visit_order_candidate_ids'];bb=z[key]['visit_order_candidate_ids']
                    out[prefix+'_stop_set_changed']=set(aa)!=set(bb);out[prefix+'_visit_order_changed']=aa!=bb
                    out[prefix+'_same_set_order_changed']=set(aa)==set(bb) and aa!=bb
                changes.append(out)
            for parent,rr in groups.items():
                complete=all(a['valid'] and b['valid'] for a,b in rr)
                for metric in ('raw_J','J'):
                    c=statistics.mean(a['metrics'][metric] for a,b in rr) if complete else None
                    b=statistics.mean(b['metrics'][metric] for a,b in rr) if complete else None
                    paired.append(dict(tier=tier,seed=seed,method=method,endpoint=endpoint,control=control['method'],control_endpoint=control['endpoint'],metric=metric,physical_key=parent,
                        group=rr[0][0]['group'],requests=len(rr),complete=complete,candidate_parent_mean=c,control_parent_mean=b,
                        relative_gain=None if not complete else (b-c)/b))
    write_csv(folder/'PARENT_PAIRED_RESULTS.csv',paired);write_csv(folder/'STOP_SET_ORDER_CHANGES.csv',changes)
    model_index=[]
    for file in sorted((HOME/'streams').glob('*/*/step*.actor.json')):
        entry=read(file)
        if sha(ROOT/entry['path'])!=entry['sha256']:raise ValueError('Model bytes differ')
        model_index.append(dict(entry,seed=int(file.parent.parent.name),fixed_endpoint=int(file.stem.split('.')[0].replace('step',''))))
    put(HOME/'MODEL_INDEX.json',dict(status='ACTUAL_EXPORTED_MODELS',models=model_index,P1_quality_gate=status.get('P1_gate'),
        production_default_promoted=False,failed_candidate_requires_new_authorization_for_reuse=True,
        interpretation='Fixed exports retained for evidence. A completed development gate is required for any conditional deployment; diagnostic/engineering weights excluded.'))
    put(HOME/'SAVED_TABLES_RECEIPT.json',dict(status='COMPLETE',source_sha256=sha(Path(__file__)),wall_s=time.perf_counter()-start,
        actual_complete_evaluation_grids=len(reports),actual_saved_request_rows=len(requests),parent_comparison_rows=len(paired),route_comparison_rows=len(changes),
        new_forward=0,new_optimizer=0,new_H_native_navigation_teacher=0,
        timing_scope='Actual campaign request duration includes research record compression/archive write but excludes subsequent pointer/event/resource-journal tail. Input cache states differ in development; no deployment speed claim from these rows.'))
    print(dict(status='COMPLETE',saved_request_rows=len(requests),models=len(model_index),wall_s=time.perf_counter()-start))

if __name__=='__main__':main()
