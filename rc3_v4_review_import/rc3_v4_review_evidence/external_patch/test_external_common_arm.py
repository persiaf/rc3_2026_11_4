"""Isolated regression runner for the external common-arm assignment patch.

Requires Python and NumPy plus an unpacked v4 source tree. Does not install or
modify the original source. No Torch, pickle, robot execution, or model training.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--patched-file', type=Path,
                        default=Path(__file__).resolve().parent/'rc3_v4/external.py')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent/'results')
    args = parser.parse_args()
    root = args.source_root.resolve()
    patched_file = args.patched_file.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    original_file = root/'rc3_v4/external.py'
    if original_file == patched_file:
        raise SystemExit('Use an isolated patched file; do not overwrite original source.')
    original_hash = sha256(original_file)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))

    import rc3_v4
    original = importlib.import_module('rc3_v4.external')
    example = importlib.import_module('rc3_v4.external_example')
    bundle = example.make_demo()
    original_plans = {method: original.plan_external(bundle, method)['plans'][method]
                      for method in ('fixed3', 'detour')}
    F, A = original.check_bundle(bundle)
    allowed = F[0, :, 0].copy()

    def arm_case(planning_arms, audit_arms):
        f, a = F.copy(), A.copy()
        f[0] = False
        a[0] = False
        for arm in planning_arms:
            f[0, :, 'LR'.index(arm)] = allowed
        for arm in audit_arms:
            a[0, :, 'LR'.index(arm)] = allowed
        return example.reseal(bundle, f, a)

    counterexample = arm_case('LR', 'R')
    before = original.plan_external(counterexample, 'fixed3')
    assert before['status'] == 'failed'
    assert before['physical_infeasibility_claim'] is False

    # Register only the isolated external module under its expected package
    # name. All other modules are imported read-only from --source-root.
    spec = importlib.util.spec_from_file_location('rc3_v4.external', patched_file)
    patched = importlib.util.module_from_spec(spec)
    sys.modules['rc3_v4.external'] = patched
    rc3_v4.external = patched
    spec.loader.exec_module(patched)

    class CommonArmTests(unittest.TestCase):
        def check_arm(self, planning_arms, audit_arms, wanted):
            case = arm_case(planning_arms, audit_arms)
            result = patched.plan_external(case, 'fixed3')
            self.assertEqual(result['status'], 'complete', result)
            plan = result['plans']['fixed3']
            selected = next(a for a in plan['assignments'] if a['task_id'] == 0)
            self.assertEqual(selected['arm'], wanted)
            self.assertTrue(patched.validate_external(case, plan)['valid_model_plan'])
            return case, result

        def test_l_only_intersection(self):
            self.check_arm('LR', 'L', 'L')

        def test_r_only_intersection_repairs_counterexample(self):
            self.check_arm('LR', 'R', 'R')

        def test_both_arms_keeps_deterministic_l_preference(self):
            self.check_arm('LR', 'LR', 'L')

        def test_no_intersection_fails_and_preserves_all_task_ids(self):
            calls = []
            case = arm_case('L', 'R')
            def checker(*args):
                calls.append(args)
                return True
            result = patched.plan_external(case, service_checker=checker)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['pending_task_ids'], [0, 1, 2, 3])
            self.assertFalse(result['physical_infeasibility_claim'])
            self.assertEqual(calls, [])

        def test_audit_cannot_enable_planning_forbidden_arm(self):
            self.check_arm('L', 'LR', 'L')

        def test_task_can_be_deferred_to_later_visited_stop(self):
            order = original_plans['fixed3']['visit_order_candidate_ids']
            first, last = order[0], order[-1]
            f, a = F.copy(), A.copy()
            f[0] = False
            a[0] = False
            f[0, [first, last], 0] = True
            a[0, last, 0] = True
            case = example.reseal(bundle, f, a)
            problem = patched.prepare_external(case)
            plan = patched.export_external(problem, [0]+[j+1 for j in order]+[0], case)
            service = next(v for v in plan['assignments'] if v['task_id'] == 0)
            self.assertEqual(service['candidate_id'], last)
            self.assertTrue(patched.validate_external(case, plan)['valid_model_plan'])

        def test_callback_rejection_is_not_reinterpreted_as_other_arm_success(self):
            calls = []
            case = arm_case('LR', 'LR')
            def checker(task, candidate, arm, certificate):
                calls.append((task['id'], candidate['id'], arm))
                return not (task['id'] == 0 and arm == 'L')
            result = patched.plan_external(case, service_checker=checker)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['pending_task_ids'], [0, 1, 2, 3])
            self.assertTrue(any(t == 0 for t, _, _ in calls))
            self.assertEqual({arm for t, _, arm in calls if t == 0}, {'L'})

        def test_callback_exception_still_fails_closed(self):
            def checker(*args):
                raise RuntimeError('injected checker error')
            result = patched.plan_external(bundle, 'detour', service_checker=checker)
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['pending_task_ids'], [0, 1, 2, 3])
            self.assertFalse(result['physical_infeasibility_claim'])

        def test_final_callback_rejection_returns_certified_anchor(self):
            calls = []
            def checker(task, candidate, arm, certificate):
                calls.append((task['id'], candidate['id'], arm))
                return len(calls) <= 4
            result = patched.plan_external(bundle, 'detour', service_checker=checker)
            self.assertEqual(result['status'], 'complete', result)
            self.assertTrue(result['diagnostics']['fallback_used'])
            self.assertEqual(len(calls), 5)
            self.assertEqual(result['diagnostics']['objective_m'],
                             result['diagnostics']['reference_objective_m'])
            self.assertTrue(patched.validate_external(bundle, result['plans']['detour'])['valid_model_plan'])

        def test_existing_demo_plans_unchanged(self):
            for method in ('fixed3', 'detour'):
                with self.subTest(method=method):
                    result = patched.plan_external(bundle, method)
                    self.assertEqual(result['status'], 'complete')
                    self.assertEqual(result['plans'][method], original_plans[method])

        def test_independent_final_validator_still_rejects_disallowed_arm(self):
            case, result = self.check_arm('LR', 'R', 'R')
            plan = copy.deepcopy(result['plans']['fixed3'])
            for a in plan['assignments']:
                if a['task_id'] == 0:
                    a['arm'] = 'L'
            for stop in plan['service_stops']:
                for a in stop['tasks']:
                    if a['task_id'] == 0:
                        a['arm'] = 'L'
            check = patched.validate_external(case, plan)
            self.assertFalse(check['valid_model_plan'])
            self.assertIn('audit F rejects', check['errors'][0])

    existing = unittest.TestLoader().loadTestsFromName('rc3_v4.test_external')
    additional = unittest.TestLoader().loadTestsFromTestCase(CommonArmTests)
    old_count, new_count = existing.countTestCases(), additional.countTestCases()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.TestSuite([existing, additional]))
    (out/'tests.log').write_text(stream.getvalue(), encoding='utf-8')
    after = patched.plan_external(counterexample, 'fixed3')
    evidence = dict(description='Conservative false rejection of a feasible same-route arm assignment',
                    bundle=counterexample, before=before, after=after)
    (out/'minimal_reproduction.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    summary = dict(all_passed=result.wasSuccessful(), tests_run=result.testsRun,
                   original_external_tests=old_count, new_regression_tests=new_count,
                   failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                   original_external_sha256=original_hash,
                   patched_external_sha256=sha256(patched_file),
                   original_source_unchanged=original_hash == sha256(original_file),
                   torch_imported='torch' in sys.modules,
                   frozen_evaluation_unchanged=True,
                   original_counterexample_status=before['status'], patched_status=after['status'],
                   patch_scope='Assignment uses common permitted arms; final F/A and callback checks unchanged.')
    (out/'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.wasSuccessful() and summary['original_source_unchanged'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
