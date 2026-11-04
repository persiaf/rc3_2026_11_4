"""CPU-only integration checks and unit tests for the v5 maintenance release."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from rc3_baseline.validator import validate
from rc3_fast.scenes import make_complex_scene
from .plan import Planner
from .external_example import make_demo
from .external import plan_external, validate_external
from .robot_input import preflight


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=True,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('rc3_v5/verification'))
    parser.add_argument('--skip-unit',action='store_true')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parent.parent
    report={'python':sys.version,'checks':[]}
    if not args.skip_unit:
        suite=unittest.defaultTestLoader.discover(str(root/'rc3_v5'),pattern='test_*.py',top_level_dir=str(root))
        stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (out/'unittest.log').write_text(stream.getvalue(),encoding='utf-8')
        report['tests']={'run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
                         'skipped':len(result.skipped),'all_passed':result.wasSuccessful()}
    scene=make_complex_scene(3,3,771903,'blocked')
    for method in ('fixed3','detour','legacy','learned'):
        planner=Planner(method=method,horizon=8)
        first=planner.plan(scene);second=planner.plan(scene)
        assert first['status']=='complete',first
        assert second['status']=='complete',second
        a,b=first['plans'][method],second['plans'][method]
        certificate=validate(scene,a,first['feasibility'])
        assert certificate['valid_model_plan'] and certificate['all_tasks_served']
        assert a==b and not first['timing']['cache_hit'] and second['timing']['cache_hit']
        assert first['diagnostics']['objective_m']<=first['diagnostics']['reference_objective_m']+1e-8
        write(out/(method+'.json'),first)
        report['checks'].append({'check':method+'_cold_warm','passed':True,
            'objective':first['diagnostics']['objective_m'],'independent_validation':certificate})
    bundle=make_demo();write(out/'external_input.json',bundle)
    for method in ('fixed3','detour'):
        result=plan_external(bundle,method=method)
        assert result['status']=='complete',result
        # The external entry's final checker is rerun, using both tables.
        certificate=validate_external(bundle,result['plans'][method])
        assert certificate['valid_model_plan'],certificate
        write(out/('external_'+method+'.json'),result)
        report['checks'].append({'check':'external_'+method,'passed':True})
    # Test the actual CLI, including CSV export and wrong-mode rejection.
    cp=subprocess.run([sys.executable,'-B','-m','rc3_v5.plan','--method','learned',
        '--scale','3x3','--seed','771903','--output',str(out/'cli.json')],capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr+cp.stdout
    assert (out/'cli.csv').is_file()
    rejected=subprocess.run([sys.executable,'-B','-m','rc3_v5.plan','--scene',str(out/'external_input.json'),
        '--output',str(out/'should_not_exist.json')],capture_output=True,text=True)
    assert rejected.returncode==2 and 'proxy mode must not discard F/A' in rejected.stderr
    report['checks'].append({'check':'proxy_cli_export_and_external_guard','passed':True})
    robot=preflight(root/'rc3_v5/robot_assets/manifest.json',root/'rc3_v5/robot_assets/runtime_config.template.json')
    assert robot['status']=='blocked_inputs' and not robot['ready_for_service_generation']
    assert not robot['physical_ik_verified'] and not robot['physical_infeasibility_claimed']
    report['checks'].append({'check':'unconfigured_robot_explicitly_blocked','passed':True,
        'blocker_count':len(robot['blocked_inputs'])})
    report['torch_imported']='torch' in sys.modules
    assert not report['torch_imported']
    report['all_passed']=all(r['passed'] for r in report['checks']) and report.get('tests',{}).get('all_passed',True)
    write(out/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}));return 0 if report['all_passed'] else 2


if __name__=='__main__':raise SystemExit(main())
