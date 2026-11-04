"""Reproduce the isolated v5.1 input-consistency patch without replacing v5 imports."""
from __future__ import annotations
import argparse
import contextlib
import difflib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

from rc3_v5 import robot_input as original
from rc3_v5.test_robot_input import RobotInputTests as OriginalTests
from rc3_v5_1 import robot_input as patched
from rc3_v5_1.test_robot_input import RobotInputTests, exchange_grippers


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf-8')


def counterexample(cls):
    case=cls();case.setUp()
    try:
        baseline=case.check()
        exchange_grippers(case)
        swapped=case.check()
        return dict(baseline=baseline,swapped=swapped)
    finally:
        case.doCleanups()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    originals=['rc3_v5/robot_input.py','rc3_v5/test_robot_input.py']
    source_before={p:digest(root/p) for p in originals}
    before=counterexample(OriginalTests);after=counterexample(RobotInputTests)
    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(RobotInputTests))
    (out/'tests.log').write_text(stream.getvalue(),encoding='utf-8')
    assets=root/'rc3_v5/robot_assets'
    old_template=original.preflight(assets/'manifest.json',assets/'runtime_config.template.json')
    new_template=patched.preflight(assets/'manifest.json',assets/'runtime_config.template.json')
    write(out/'preflight_report.json',new_template)
    commands=[('help',0,[sys.executable,'-B','-m','rc3_v5_1.robot_input','--help']),
        ('inventory',0,[sys.executable,'-B','-m','rc3_v5_1.robot_input','inventory',
        '--manifest',str(assets/'manifest.json'),'--output',str(out/'asset_inventory.json')]),
        ('empty_template',2,[sys.executable,'-B','-m','rc3_v5_1.robot_input','check',
        '--manifest',str(assets/'manifest.json'),'--config',str(assets/'runtime_config.template.json'),
        '--output',str(out/'cli_preflight_report.json')])]
    cli=[]
    for name,expected,command in commands:
        proc=subprocess.run(command,cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace')
        help_ok=(name!='help' or ('python -m rc3_v5_1.robot_input --help' in proc.stdout
                                 and 'python -m rc3_v5.robot_input --help' not in proc.stdout))
        cli.append(dict(name=name,command=command,returncode=proc.returncode,expected_returncode=expected,
                        stdout=proc.stdout,stderr=proc.stderr,passed=proc.returncode==expected and help_ok))
    write(out/'cli_checks.json',cli)
    uploaded=root/'rc3_v5_acceptance_import/rc3_v5_acceptance_evidence/robot_patch/rc3_v5/robot_input.py'
    uploaded_bytes=uploaded.read_bytes();maintenance=(root/'rc3_v5_1/robot_input.py').read_bytes()
    docline=b'Run ``python -m rc3_v5.robot_input --help``'
    expected=uploaded_bytes.replace(docline,b'Run ``python -m rc3_v5_1.robot_input --help``')
    docstring_only=uploaded_bytes.count(docline)==1 and maintenance==expected
    (out/'upload_to_maintenance.patch').write_text(''.join(difflib.unified_diff(
        uploaded_bytes.decode('utf-8').splitlines(keepends=True),maintenance.decode('utf-8').splitlines(keepends=True),
        fromfile='uploaded/robot_input.py',tofile='rc3_v5_1/robot_input.py')),encoding='utf-8')
    source_after={p:digest(root/p) for p in originals}
    summary=dict(tests_run=result.testsRun,original_tests=15,new_regressions=4,
        failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        unit_tests_passed=result.wasSuccessful(),cli_checks=3,cli_passed=all(c['passed'] for c in cli),
        original_counterexample_status=before['swapped']['status'],
        patched_counterexample_status=after['swapped']['status'],
        baseline_inputs_complete=(before['baseline']['status']==after['baseline']['status']=='inputs_complete'),
        original_template_preserved=old_template==new_template,
        template_status=new_template['status'],template_blockers=len(new_template['blocked_inputs']),
        original_source_sha256=source_before,original_source_after_sha256=source_after,
        original_source_unchanged=source_before==source_after,
        uploaded_patch_sha256=digest(uploaded),maintenance_source_sha256=digest(root/'rc3_v5_1/robot_input.py'),
        only_additional_change_is_cli_docstring=docstring_only,functional_patch_unchanged=docstring_only,
        test_sha256=digest(root/'rc3_v5_1/test_robot_input.py'),
        direct_module_imports=True,sys_modules_replacement_used=False,
        torch_imported='torch' in sys.modules,training_performed=False,timing_benchmark_performed=False,
        real_calibration_or_ik_performed=False,
        scope='Synthetic configuration consistency only; no actual calibration, IK, collision computation or planner changes.')
    summary['all_passed']=(result.wasSuccessful() and result.testsRun==19 and summary['cli_passed']
        and summary['original_source_unchanged'] and summary['original_template_preserved'] and docstring_only
        and summary['baseline_inputs_complete'] and before['swapped']['status']=='inputs_complete'
        and after['swapped']['status']=='blocked_inputs' and new_template['status']=='blocked_inputs'
        and not new_template['physical_ik_verified'] and not new_template['collision_verified'])
    write(out/'counterexample_before_after.json',dict(before=before,after=after))
    write(out/'SUMMARY.json',summary)
    print(json.dumps(summary,ensure_ascii=True));return 0 if summary['all_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
