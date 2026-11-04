"""Sample replay of the frozen NPZ policy without importing Torch."""
import argparse
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.source_root.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(root))
    import numpy as np
    from rc3_fast.solver import prepare
    from rc3_baseline.validator import validate
    from rc3_v3.runtime import run
    from rc3_v3.plan import Planner
    from rc3_v3.model import load_policy
    from rc3_improve.evaluate import safe_json
    checkpoint=root/'rc3_v3/deployment/policy.npz';policy=load_policy(checkpoint)
    files=sorted((root/'rc3_v3/results/plans').glob('scene_plan_*.json'))
    selected={};results=[]
    for file in files:
        data=json.loads(file.read_text());n=len(data['scene']['tables'])
        if n in (12,100,196,400) and n not in selected:selected[n]=(file,data)
    for n,(file,data) in selected.items():
        p=prepare(data['scene'])
        for method,label in [('fixed3','fixed3_rep0'),('learned','learned_calls_8_rep0')]:
            plan,info=run(p,method,policy=policy if method=='learned' else None,horizon=8)
            saved=data['plans'][label];check=validate(data['scene'],plan,p.sparse)
            row=dict(tables=n,file=file.name,method=method,
                     saved_objective_m=saved['objective_length_plus_stops_m'],objective_m=plan['objective_length_plus_stops_m'],
                     cost_matches=abs(saved['objective_length_plus_stops_m']-plan['objective_length_plus_stops_m'])<=1e-7,
                     service_sequence_matches=saved['visit_order_candidate_ids']==plan['visit_order_candidate_ids'],
                     assignments_match=saved['assignments']==plan['assignments'],
                     polyline_matches=saved['route_polyline']==plan['route_polyline'],
                     valid_model_plan=check['valid_model_plan'],all_tasks_served=check['all_tasks_served'],
                     dp_calls=info['dp_calls'],stop_reason=info['stop_reason'],
                     reference_bound=info['objective_m']<=info['reference_objective_m']+1e-8,
                     timing_scope='Replay correctness check only, not a hardware comparison.')
            results.append(row);print(json.dumps(row),flush=True)
    scene=copy.deepcopy(selected[100][1]['scene'])
    planner=Planner(checkpoint,'learned',horizon=8)
    cold=planner.plan(copy.deepcopy(scene));warm=planner.plan(copy.deepcopy(scene))
    reload=Planner(checkpoint,'learned',horizon=8).plan(copy.deepcopy(scene))
    portable=dict(complete=all(x['status']=='complete' for x in (cold,warm,reload)),
                  full_plan_equal=cold.get('plans')==warm.get('plans')==reload.get('plans'),
                  cache_hit_sequence=[x['timing']['cache_hit'] for x in (cold,warm,reload)])
    failures=[]
    for case in ('unreachable_task','blocked_depot','variable_lift','invalid_task_ids'):
        bad=copy.deepcopy(scene)
        if case=='unreachable_task':bad['tasks'][0]['z']=99.
        elif case=='blocked_depot':
            x,y=bad['start'];bad['obstacles'].append(dict(xmin=x-.1,xmax=x+.1,ymin=y-.1,ymax=y+.1))
        elif case=='variable_lift':bad['candidates'][0]['h']+=.1
        else:bad['tasks'][0]['id']=99999
        ids=[t['id'] for t in bad['tasks']];result=planner.plan(bad)
        failures.append(dict(case=case,passed=result['status']=='failed' and not result['plans'] and
                             result['task_ids']==ids and result['pending_task_ids']==ids,
                             code=result.get('code'),tasks_preserved=len(result.get('pending_task_ids',[]))))
    empty=copy.deepcopy(scene);empty['tasks']=[];zero=planner.plan(empty)
    portable['empty_zero_success']=zero['status']=='complete' and zero['diagnostics']['objective_m']==0
    default=Planner().plan(copy.deepcopy(scene))
    portable['default_fixed3']=default['status']=='complete' and set(default['plans'])=={'fixed3'} and default['diagnostics']['dp_calls']==0
    # The phase/schema guard must also apply to the portable file, not only .pt.
    with np.load(checkpoint,allow_pickle=False) as archive:
        arrays={k:archive[k].copy() for k in archive.files}
    metadata=json.loads(str(arrays['metadata']));wrong=dict(metadata,phase='ppo')
    with tempfile.TemporaryDirectory() as folder:
        invalid=Path(folder)/'wrong_phase.npz'
        np.savez_compressed(invalid,**(arrays|{'metadata':np.array(json.dumps(wrong))}))
        try:
            load_policy(invalid);portable['wrong_phase_rejected']=False
        except ValueError:portable['wrong_phase_rejected']=True
    portable['torch_not_imported']='torch' not in sys.modules
    summary=dict(replays=results,portable=portable,failure_probes=failures,
                 architecture=metadata['architecture'],phase=metadata['phase'],schema=metadata['schema'],
                 stop_rule=metadata.get('stop_rule'),stop_threshold=metadata.get('stop_threshold'),
                 all_passed=all(all(r[k] for k in ('cost_matches','service_sequence_matches','assignments_match','polyline_matches','valid_model_plan','all_tasks_served','reference_bound')) for r in results)
                 and portable['complete'] and portable['full_plan_equal'] and portable['empty_zero_success'] and portable['default_fixed3'] and portable['wrong_phase_rejected'] and portable['torch_not_imported'] and all(f['passed'] for f in failures))
    (out/'replay_summary.json').write_text(json.dumps(summary,indent=2))
    (out/'scene_plan_replayed_100.json').write_text(json.dumps(safe_json(cold),indent=2))
    print(json.dumps(dict(all_passed=summary['all_passed'],portable=portable,failure_probes=failures)),flush=True)


if __name__=='__main__':main()
