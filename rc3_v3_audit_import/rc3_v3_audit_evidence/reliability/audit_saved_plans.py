"""Read-only independent validation of every repeated plan in the v3 delivery.

Source files are never modified. Reference bounds use the independently checked
saved Fixed3 plan in each scene, not the claim in learned-plan diagnostics.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source_root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(source))
    from rc3_baseline.validator import validate
    manifest=json.loads((source/'rc3_v3/results/manifest.json').read_text())
    metrics=json.loads((source/'rc3_v3/results/metrics.json').read_text())
    files=sorted((source/'rc3_v3/results/plans').glob('scene_plan_*.json'))
    started=time.perf_counter();checks=[];scene_reports=[];lookup={};errors=[]
    for i,path in enumerate(files):
        data=json.loads(path.read_text());scene=data['scene'];plans=data.get('plans',{})
        current=[];refs=[]
        for label,plan in plans.items():
            check=validate(scene,plan,data['feasibility'])
            independent_cost=(check['independent_polyline_length_m']+
                              plan['stop_penalty_equivalent_m']*len(plan['selected_candidate_ids'])) if check['valid_model_plan'] else None
            config=label.rsplit('_rep',1)[0]
            key=(scene['seed'],scene['nx'],scene['ny'],scene.get('complexity','custom'),config)
            lookup.setdefault(key,[]).append(independent_cost)
            row=dict(file=path.name,label=label,seed=scene['seed'],tables=len(scene['tables']),
                     independent_objective_m=independent_cost,**check)
            current.append(row)
            if config=='fixed3' and check['valid_model_plan'] and check['all_tasks_served']:
                refs.append(independent_cost)
            if not check['valid_model_plan'] or not check['all_tasks_served']:
                errors.append(f'Invalid or incomplete: {path.name} {label}')
        reference=min(refs) if refs else None
        if len(refs)!=manifest['repeats'] or (refs and max(refs)-min(refs)>1e-7):
            errors.append(f'Missing or differing independently checked Fixed3 reference: {path.name}')
        for row in current:
            row['reference_m']=reference
            row['nonworsening_to_checked_saved_fixed3']=(reference is not None and row['independent_objective_m'] is not None
                                                      and row['independent_objective_m']<=reference+1e-7)
            if not row['nonworsening_to_checked_saved_fixed3']:
                errors.append(f'Fixed3 bound failed: {path.name} {row["label"]}')
        checks.extend(current)
        failed_runs=[r for r in data.get('runs',[]) if r.get('status')!='complete'] if isinstance(data.get('runs'),list) else []
        scene_reports.append(dict(file=path.name,seed=scene['seed'],tables=len(scene['tables']),
                                  status=data.get('status'),plans=len(plans),tasks=len(scene['tasks']),
                                  tasks_in_original_order_preserved=data.get('task_ids')==[t['id'] for t in scene['tasks']],
                                  checked_saved_fixed3=reference,failed_runs=failed_runs))
        if (i+1)%5==0 or i+1==len(files):
            progress=dict(scenes=i+1,total_scenes=len(files),plans=len(checks),elapsed_s=time.perf_counter()-started,
                          failures=len(errors))
            (out/'progress.json').write_text(json.dumps(progress,indent=2))
            print(json.dumps(progress),flush=True)
    mean_checks=[]
    for row in metrics:
        key=(row['seed'],row['nx'],row['ny'],row['mode'],row['config'])
        values=lookup.get(key,[])
        good=(row['status']=='complete' and len(values)==manifest['repeats'] and all(v is not None for v in values)
              and abs(statistics.mean(values)-row['objective_m'])<=1e-7)
        mean_checks.append(dict(seed=row['seed'],config=row['config'],status=row['status'],saved_repeats=len(values),
                                independent_repeat_mean_matches=good))
        if row['status']=='complete' and not good:
            errors.append(f'Metric mean not reproduced: {key}')
    if len(files)!=manifest['cases']:errors.append('Scene denominator mismatch')
    if len(checks)!=manifest['saved_repeated_plans']:errors.append('Repeated plan denominator mismatch')
    if len(metrics)!=manifest['method_budget_cases']:errors.append('Method-budget-case denominator mismatch')
    if not all(s['tasks_in_original_order_preserved'] for s in scene_reports):errors.append('Task ID denominator mismatch')
    summary=dict(all_passed=not errors,errors=errors,scenes=len(files),repeated_plans=len(checks),
                 valid_complete=sum(c['valid_model_plan'] and c['all_tasks_served'] for c in checks),
                 checked_bounds=sum(c['nonworsening_to_checked_saved_fixed3'] for c in checks),
                 method_budget_cases=len(metrics),failed_metric_cases=sum(m['status']!='complete' for m in metrics),
                 reproduced_metric_means=sum(m['independent_repeat_mean_matches'] for m in mean_checks),
                 scenes_per_scale=dict(Counter(s['tables'] for s in scene_reports)),elapsed_s=time.perf_counter()-started,
                 scope='Every saved plan independently checked with unchanged standard-library geometric/bookkeeping validator. No optimizer rerun; reference bound against independently checked saved Fixed3 plans. No physical IK/trajectory or hardware claim.',
                 validator_sha256=hashlib.sha256((source/'rc3_baseline/validator.py').read_bytes()).hexdigest())
    (out/'all_plans.json').write_text(json.dumps(dict(summary=summary,checks=checks,scene_reports=scene_reports,mean_checks=mean_checks),indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
