"""New adapter-only correctness sample; no training or original timing replacement."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest

import numpy as np
import scipy

from reliable import plan_reliably
from rc3_fast.scenes import make_complex_scene
from rc3_baseline.validator import validate_file
from test_reliable import ReliabilityTests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('validation_results'))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be a new or empty directory; existing evidence is not overwritten')
    args.output.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReliabilityTests)
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    if not tests.wasSuccessful():
        raise SystemExit('Targeted adapter tests failed.')
    scales = [(4, 3), (6, 6), (8, 8), (10, 10)]
    cases = [(nx, ny, mode) for nx, ny in scales for mode in ('regular', 'mixed', 'blocked')]
    cases += [(nx, ny, mode) for nx, ny in scales for mode in ('mixed', 'blocked')]
    rows, checks = [], []
    began = time.perf_counter()
    for index, (nx, ny, scene_mode) in enumerate(cases):
        seed = 3700000000 + 1009 * index
        generation_started = time.perf_counter()
        scene = make_complex_scene(nx, ny, seed, scene_mode)
        generation_s = time.perf_counter() - generation_started
        fixed1_j = None
        for mode in ('fast', 'quality'):
            result = plan_reliably(scene, mode=mode)
            assert result['status'] == 'complete', result
            certificate = result['certificate']
            plan = result['plans']['reliable']
            if mode == 'fast':
                fixed1_j = certificate['returned_objective_m']
            assert certificate['returned_objective_m'] <= fixed1_j
            path = args.output / f'scene_plan_{nx}x{ny}_{scene_mode}_{seed}_{mode}.json'
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            # Independent disk reread, deliberately ignoring serialized validation booleans.
            for check in validate_file(path):
                assert check['valid_model_plan'] and check['all_tasks_served'], check
                checks.append(check)
            rows.append(dict(nx=nx, ny=ny, scene_mode=scene_mode, seed=seed, mode=mode,
                             tasks=len(scene['tasks']), stops=len(plan['selected_candidate_ids']),
                             route_length_m=plan['route_length_m'], objective_m=certificate['returned_objective_m'],
                             reference='fixed1_recipe_1', reference_objective_m=certificate['reference_objective_m'],
                             attempts=certificate['attempts'], candidate_monotone=certificate['cost_not_worse_than_reference'],
                             scene_generation_s=generation_s, preprocessing_s=result['timing']['preprocessing_s'],
                             adapter_total_s=result['timing']['total_s'],
                             full_case_s=generation_s+result['timing']['total_s']))
        print(f'{nx}x{ny} {scene_mode} seed={seed}: fixed1={fixed1_j:.6f}, '
              f'quality={certificate["returned_objective_m"]:.6f}; both validated', flush=True)
    with (args.output/'metrics_new_cpu.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    cpu = platform.processor()
    if not cpu and Path('/proc/cpuinfo').exists():
        cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                    if line.startswith('model name')), 'unknown')
    summary = []
    for nx, ny in scales:
        for mode in ('fast', 'quality'):
            group = [r for r in rows if (r['nx'], r['ny'], r['mode']) == (nx, ny, mode)]
            summary.append(dict(scale=f'{nx}x{ny}', mode=mode, instances=len(group),
                                mean_objective_m=float(np.mean([r['objective_m'] for r in group])),
                                median_adapter_total_ms=float(np.median([r['adapter_total_s'] for r in group])*1000)))
    report = dict(scope='New deterministic adapter correctness sample, not a retraining run or statistically powered performance comparison.',
                  scene_sampling='Five new seeds per size: regular 1, mixed 2, blocked 2. All four sizes covered.',
                  scenes=20, saved_plans=40, targeted_test_methods=tests.testsRun,
                  targeted_tests_passed=tests.wasSuccessful(), all_disk_revalidated=True,
                  all_quality_objectives_no_worse_than_fixed1=True,
                  no_original_metrics_modified=True,
                  cpu=cpu, platform=platform.platform(), python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                  adapter_source_sha256=hashlib.sha256(Path('reliable.py').read_bytes()).hexdigest(),
                  elapsed_new_sample_s=time.perf_counter()-began,
                  timing_scope='New CPU measurements, one run each. Adapter includes preprocessing, all candidate '
                  'searches/exports and validation, excluding imports and JSON I/O. No navigation cache used.',
                  summary=summary, disk_checks=checks)
    (args.output/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('disk_checks','python','summary')},ensure_ascii=False))


if __name__ == '__main__':
    main()
