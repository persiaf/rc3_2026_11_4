"""Reproduce external v5 regression evidence from the project root.

python -B -m rc3_v5.external_results.reproduce
Only synthetic fixtures are used; this is not a physical robot experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from rc3_v4 import external as frozen
from rc3_v5 import external
from rc3_v5.external_example import make_demo, reseal
from rc3_v5.test_external import (ExternalTests, CommonArmTests,
                                  CommonCoverageTests, later_stop_counterexample)
from rc3_improve.environment import canonical_route


ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def export_record(module, problem, route, bundle):
    try:
        return dict(status='complete', plan=module.export_external(problem, route, bundle))
    except Exception as error:
        return dict(status='failed', error=str(error), physical_infeasibility_claim=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    originals = [ROOT/'rc3_v4'/name for name in
                 ('external.py', 'test_external.py', 'external_example.py', 'EXTERNAL_SCHEMA.md')]
    before_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in originals}
    review_file = Path(__file__).resolve().parent/'assignment_only_review.py'
    spec = importlib.util.spec_from_file_location('rc3_v5_uploaded_assignment_only_review', review_file)
    review = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(review)
    # No assignment to sys.modules or rc3_v4 attributes: old and new coexist.

    classes = (ExternalTests, CommonArmTests, CommonCoverageTests)
    suites = [unittest.defaultTestLoader.loadTestsFromTestCase(cls) for cls in classes]
    counts = [suite.countTestCases() for suite in suites]
    stream = io.StringIO()
    test_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.TestSuite(suites))
    (out/'tests.log').write_text(stream.getvalue(), encoding='utf-8')

    bundle = make_demo()
    F, A = external.check_bundle(bundle)
    permitted = F[0, :, 0].copy()
    F[0, :, 0] = F[0, :, 1] = permitted
    A[0] = False
    A[0, :, 1] = permitted
    case = reseal(bundle, F, A)
    arm_results = dict(original=frozen.plan_external(case), assignment_only_patch=review.plan_external(case),
                       v5=external.plan_external(case))
    write_json(out/'common_arm_counterexample.json', dict(
        description='Planning permits L/R but audit permits only R for task 0.',
        evidence_level='synthetic_demo', bundle=case, results=arm_results))

    later_case, route = later_stop_counterexample()
    legacy_problem = frozen.prepare_external(later_case)
    export_results = dict(original=export_record(frozen, legacy_problem, route, later_case),
                          assignment_only_patch=export_record(review, legacy_problem, route, later_case),
                          v5=export_record(external, legacy_problem, route, later_case))
    later_results = {name: {method: module.plan_external(later_case, method)
                           for method in ('fixed3', 'detour')}
                     for name, module in (('original', frozen), ('assignment_only_patch', review), ('v5', external))}
    write_json(out/'canonicalization_counterexample.json', dict(
        description='F-only pruning drops the sole A-permitted later stop for task 0.',
        evidence_level='synthetic_demo', bundle=later_case, input_route=route,
        f_only_canonical_route=canonical_route(legacy_problem, route),
        common_canonical_route=canonical_route(external.prepare_external(later_case), route),
        direct_export=export_results, planning= later_results))

    demo = out/'demo'
    commands = [
        (['-m', 'rc3_v5.external_example', '--output', str(demo)], 0),
        (['-m', 'rc3_v5.external', str(demo/'external_demo.json'), '--verify-plan',
          str(demo/'detour_plan.json'), '--output', str(out/'cli_verify_json.json')], 0),
        (['-m', 'rc3_v5.external', str(demo/'external_demo.npz'), '--method', 'detour',
          '--penalty', '1.5', '--output', str(out/'cli_npz_custom_penalty.json')], 0),
        (['-m', 'rc3_v5.external', str(demo/'external_demo.npz'), '--verify-plan',
          str(out/'cli_npz_custom_penalty.json'), '--penalty', '1.5',
          '--output', str(out/'cli_verify_npz_custom_penalty.json')], 0),
        (['-m', 'rc3_v5.external', str(demo/'external_demo.npz'), '--verify-plan',
          str(out/'cli_npz_custom_penalty.json'), '--output', str(out/'cli_verify_wrong_penalty.json')], 2),
    ]
    cli = []
    for argv, expected in commands:
        proc = subprocess.run([sys.executable, '-B', *argv], cwd=ROOT, capture_output=True,
                              text=True, encoding='utf-8', errors='replace', check=False)
        cli.append(dict(argv=[sys.executable, '-B', *argv], returncode=proc.returncode,
                        expected_returncode=expected, passed=proc.returncode == expected,
                        stdout=proc.stdout, stderr=proc.stderr))
    write_json(out/'cli_checks.json', cli)
    after_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in originals}
    status_expectations = (
        arm_results['original']['status'] == 'failed'
        and arm_results['assignment_only_patch']['status'] == 'complete'
        and arm_results['v5']['status'] == 'complete'
        and export_results['assignment_only_patch']['status'] == 'failed'
        and export_results['v5']['status'] == 'complete'
        and all(r['status'] == 'complete' for r in later_results['v5'].values())
        and all(r['status'] == 'failed' for r in later_results['assignment_only_patch'].values()))
    summary = dict(
        all_passed=test_result.wasSuccessful() and all(c['passed'] for c in cli)
                   and before_hashes == after_hashes and status_expectations,
        tests_run=test_result.testsRun, original_external_tests=counts[0],
        imported_common_arm_regressions=counts[1], v5_common_coverage_and_guard_regressions=counts[2],
        test_denominator='Unittest test methods; subTest parameter combinations are not counted separately.',
        failures=len(test_result.failures), errors=len(test_result.errors), skipped=len(test_result.skipped),
        cli_checks=len(cli), cli_checks_passed=sum(c['passed'] for c in cli),
        source_hashes_before=before_hashes, source_hashes_after=after_hashes,
        inspected_v4_source_files_unchanged=before_hashes == after_hashes,
        uploaded_assignment_only_sha256=sha256(review_file),
        v5_external_sha256=sha256(ROOT/'rc3_v5/external.py'),
        v5_tests_sha256=sha256(ROOT/'rc3_v5/test_external.py'),
        frozen_module_not_replaced=sys.modules['rc3_v4.external'] is frozen,
        torch_imported_in_this_process='torch' in sys.modules,
        common_arm_status={k: v['status'] for k, v in arm_results.items()},
        canonicalization_direct_export_status={k: v['status'] for k, v in export_results.items()},
        canonicalization_v5_objective=export_results['v5']['plan']['objective_length_plus_stops_m'],
        evidence_level='synthetic_demo',
        scope='Both tables participate in decisions. No unseen audit-label accuracy or physical IK claim.')
    write_json(out/'SUMMARY.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
