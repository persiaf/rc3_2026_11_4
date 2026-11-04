"""Replay stored sampled routes; certify saved outputs without neural inference."""
from __future__ import annotations
import gzip
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import vectorized_feasibility,route_objective
from rc3_baseline.validator import validate
from .problem import ROOT,Cache,protocol,write,digest,resample_tasks,registered_scene
from .env import Env


def main(finish_only=False):
    out=ROOT/'outputs';p=protocol();summary=json.loads((out/'training_summary.json').read_text(encoding='utf-8'))
    if (out/'verification.json').exists():raise FileExistsError('Saved-route audit already exists; do not silently repeat')
    registration=json.loads((out/'registration.json').read_text(encoding='utf-8'))
    assert all(digest(ROOT.parent/name)==sha for name,sha in registration['source_sha256'].items())
    source=json.loads((out/'execution_sources.json').read_text(encoding='utf-8'))
    assert all(digest(ROOT/name)==sha for name,sha in source['source_sha256'].items())
    cache=Cache(24);config=p['training'];maps={config['map_seed_start']+i:make_complex_scene(10,10,config['map_seed_start']+i,('regular','mixed','blocked')[i%3]) for i in range(24)}
    episodes=steps=0;examples=[];tick=time.perf_counter()
    recovery=None
    if finish_only:
        recovery=json.loads((out/'replay_recovery.json').read_text(encoding='utf-8'))
        assert digest(ROOT/recovery['original_log_path'])==recovery['original_log_sha256']
        assert digest(out/'sampled_routes.jsonl.gz')==recovery['sampled_routes_sha256']==summary['stored_sampled_routes_sha256']
        episodes=recovery['completed_replay_episodes'];steps=recovery['completed_replay_transitions'];examples=recovery['replay_examples']
        assert episodes==summary['sampled_trajectories']
        print(json.dumps({'explicit_completed_replay_recovery':True,'episodes':episodes,'no_new_numeric_replays':True}),flush=True)
    else:
        with gzip.open(out/'sampled_routes.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                record=json.loads(line);ordinal=episodes
                assert record['task_seed']==config['task_seed_start']+ordinal
                assert record['map_seed']==config['map_seed_start']+ordinal%24
                assert record['update']==ordinal//32+1
                scene=resample_tasks(maps[record['map_seed']],record['task_seed']);item=cache.prepare(scene)
                env=Env(item);trace=env.replay([j-1 for j in record['route'][1:-1]])
                assert env.route==record['route'] and abs(-sum(env.rewards)-record['J'])<1e-6
                assert len(trace)<=len(scene['tasks'])
                if record['update'] in (1,128,summary['updates']) and ordinal%32==0:
                    plan=item.certify(env.route,'sampled_replay')
                    file=out/'replay_examples'/f"update{record['update']:04}.json"
                    write(file,dict(scene=scene,seed_record=record,trace=trace,plan=plan))
                    examples.append(str(file.relative_to(ROOT)))
                episodes+=1;steps+=len(trace)
                if episodes%1024==0:print(json.dumps({'replayed_sampled_episodes':episodes,'steps':steps}),flush=True)
    assert episodes==summary['sampled_trajectories']==summary['updates']*32
    # Independently execute the existing scalar geometry/assignment validator on every saved plan.
    validations=[];sparse_by_case={}
    for directory in ('plans','final_plans'):
        for path in sorted((out/directory).glob('*.json')):
            case=int(path.name.split('_')[1]);scene=registered_scene('development',case)
            if case not in sparse_by_case:
                sparse_by_case[case]=vectorized_feasibility(scene)[2]
            plan=json.loads(path.read_text(encoding='utf-8'));certificate=validate(scene,plan,sparse_by_case[case])
            assert certificate['valid_model_plan'],(path,certificate)
            validations.append(dict(file=str(path.relative_to(ROOT)),sha256=digest(path),valid=True))
    small=json.loads((out/'small_exact.json').read_text(encoding='utf-8'))
    for row in small:
        path=out/'small_plans'/f"{row['case']:02}.json";plan=json.loads(path.read_text(encoding='utf-8'))
        certificate=validate(row['scene'],plan,vectorized_feasibility(row['scene'])[2]);assert certificate['valid_model_plan']
        validations.append(dict(file=str(path.relative_to(ROOT)),sha256=digest(path),valid=True))
        assert row['full_exact']['objective']<=min(x['J'] for x in row['baselines'].values())+1e-7
    # Certify both selected-set routes independently, without re-decoding or reselection.
    selected=json.loads((out/'selection'/f"update{summary['selected_update']:04}.json").read_text(encoding='utf-8'))
    for record in selected:
        scene=registered_scene('selection',record['case']);item=cache.prepare(scene)
        for kind,key in [('raw','raw_route'),('post','route')]:
            plan=item.certify(record[key],'selected_'+kind)
            expected=record['raw_J'] if kind=='raw' else record['post_J']
            assert abs(plan['objective_length_plus_stops_m']-expected)<1e-7
            path=out/'selected_plans'/f"case{record['case']:02}_{kind}.json";write(path,plan)
            validations.append(dict(file=str(path.relative_to(ROOT)),sha256=digest(path),valid=True))
    cli=out/'cli_example.json'
    if cli.exists():
        result=json.loads(cli.read_text(encoding='utf-8'));certificate=validate(result['scene'],result['plans']['constructive'],result['feasibility'])
        assert certificate['valid_model_plan'];validations.append(dict(file=str(cli.relative_to(ROOT)),sha256=digest(cli),valid=True))
    if finish_only:
        for name in examples:
            example=json.loads((ROOT/name).read_text(encoding='utf-8'))
            assert validate(example['scene'],example['plan'],vectorized_feasibility(example['scene'])[2])['valid_model_plan']
    write(out/'verification.json',dict(all_passed=True,sampled_route_replays=episodes,replayed_service_transitions=steps,
        saved_plan_geometry_checks=len(validations),saved_plans=validations,replay_examples=examples,
        sampled_example_geometry_certificates=len(examples),verification_geometry_certificates_total=len(validations)+len(examples),
        recovered_completed_replay=finish_only,
        initial_attempt_additional_geometry_certificates=recovery['initial_attempt_geometry_certificates'] if recovery else 0,
        recovery_scope='First attempt completed all numeric replays and543 geometry certificates before CLI UTF8 decoding error. Explicit completion reran geometry checks without repeating numeric replays.' if finish_only else None,
        new_neural_inferences=0,new_optimizer_updates=0,verification_elapsed_s=time.perf_counter()-tick,
        scope='All sampled route rewards/coverage replayed from seeds and frozen navigation; all listed saved plans independently geometry/assignment checked. Full geometry certification is not claimed for every sampled trajectory. No true IK.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--finish-only',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):main(args.finish_only)
