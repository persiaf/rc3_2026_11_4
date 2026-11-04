"""Recheck portable inference, cache, deployment modes and all saved teacher plans."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from rc3_baseline.baseline import write_json, write_csv, schedule_rows, SCHEDULE_COLUMNS
from rc3_baseline.validator import validate_file
from .planner import Planner
from .scenes import make_complex_scene


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('rc3_fast'))
    args=parser.parse_args(argv)
    root=args.root
    out=root/'demo_final';out.mkdir(exist_ok=True)
    scene=make_complex_scene(10,10,20261102,'blocked')
    records=[]
    for model,policy,k in [('selector.npz','learned',1),('selector.npz','learned',3),
                           ('selector.npz','fixed',1),('selector.npz','fixed',3),
                           ('selector_neural.npz','learned',3)]:
        planner=Planner(root/'deployment'/model,k,policy=policy)
        result=planner.plan(scene)
        warm=planner.plan(scene)
        assert not result['timing']['cache_hit'] and warm['timing']['cache_hit']
        name=next(iter(result['plans']))
        assert result['plans'][name]==warm['plans'][name]
        path=out/f'scene_plan_{Path(model).stem}_{policy}{k}.json'
        write_json(path,result)
        write_csv(path.with_suffix('.csv'),list(schedule_rows(result)),SCHEDULE_COLUMNS)
        checks=validate_file(path)
        assert all(c['valid_model_plan'] and c['all_tasks_served'] for c in checks)
        records.append(dict(model=model,policy=policy,top_k=k,valid=True,cache_equal=True,
                            objective_m=result['plans'][name]['objective_length_plus_stops_m'],
                            cold_s=result['timing']['total_s'],warm_s=warm['timing']['total_s']))
    assert 'torch' not in sys.modules and 'sklearn' not in sys.modules
    selected=root/'deployment'/'selection.json'
    frozen=json.loads(selected.read_text(encoding='utf-8'))
    assert hashlib.sha256((root/'deployment'/'selector.npz').read_bytes()).hexdigest()==frozen['model_sha256']
    started=time.perf_counter()
    teacher_files=sorted((root/'data_extended_20261102'/'teachers').glob('*/*.json'))
    failures=[]
    for path in teacher_files:
        checks=validate_file(path)
        failures += [dict(path=str(path),check=c) for c in checks if not c['valid_model_plan'] or not c['all_tasks_served']]
    report=dict(portable_deployment=records,no_torch_or_sklearn_imported=True,frozen_model_unchanged=True,
                teacher_files=len(teacher_files),teacher_validation_s=time.perf_counter()-started,
                teacher_failures=failures,all_valid=not failures)
    write_json(root/'deployment_check.json',report)
    if failures:
        raise AssertionError(failures[:1])
    print(json.dumps(dict(deployment_cases=len(records),teacher_files=len(teacher_files),all_valid=True,
                          frozen_model_unchanged=True,no_torch_or_sklearn_imported=True)),flush=True)


if __name__=='__main__':
    main()
