"""Generate one independently verified executable abstract service plan."""
import argparse
import json
from pathlib import Path

from rc3_baseline import baseline
from .planner import Planner
from .scenes import MODES, make_complex_scene


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--scene', type=Path, help='existing scene JSON or scene_plan JSON')
    parser.add_argument('--scale', type=baseline.scale_argument, default=(10,10))
    parser.add_argument('--mode', choices=MODES, default='blocked')
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--top-k', type=int, default=1)
    parser.add_argument('--policy', choices=('learned','fixed'), default='learned')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.scene:
        value = json.loads(args.scene.read_text(encoding='utf-8'))
        scene = value.get('scene',value)
    else:
        scene = make_complex_scene(*args.scale,args.seed,args.mode)
    planner = Planner(args.model,args.top_k,policy=args.policy)
    result = planner.plan(scene)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_json(args.output,result)
    method, plan = next(iter(result['plans'].items()))
    baseline.write_csv(args.output.with_suffix('.csv'),list(baseline.schedule_rows(result)),baseline.SCHEDULE_COLUMNS)
    print(json.dumps(dict(method=method,tasks=len(scene['tasks']),stops=len(plan['service_stops']),
                          objective_m=plan['objective_length_plus_stops_m'],
                          route_length_m=plan['route_length_m'],timing=result['timing'],
                          valid=plan['validation']['valid_model_plan'],output=str(args.output)),ensure_ascii=False))


if __name__ == '__main__':
    main()
