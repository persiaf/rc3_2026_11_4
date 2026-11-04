"""Isolated preflight chain regression; synthetic fixtures only, no real IK."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

sys.dont_write_bytecode=True

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exchange_grippers(case):
    case.config['joints']['lg']['role']='right_gripper'
    case.config['joints']['rg']['role']='left_gripper'
    case.config['grippers']['left']['joint']='rg'
    case.config['grippers']['right']['joint']='lg'
    case.config['park']['left']['positions']={'left':0.2,'rg':0.2}
    case.config['park']['right']['positions']={'right':0.2,'lg':0.2}

def run_case(cls):
    case=cls(); case.setUp()
    try:
        baseline=case.check()
        exchange_grippers(case)
        swapped=case.check()
        return {'baseline':baseline,'swapped':swapped}
    finally:
        case.doCleanups()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args()
    src=args.source_root.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    if out.is_relative_to(src):raise ValueError('keep outputs separate from frozen source')
    original_path=src/'rc3_v5/robot_input.py'
    original_hash=digest(original_path)
    sys.path.insert(0,str(src))
    import rc3_v5.robot_input as original
    import rc3_v5.test_robot_input as original_tests
    before=run_case(original_tests.RobotInputTests)
    patched_path=Path(__file__).resolve().parent/'rc3_v5/robot_input.py'
    spec=importlib.util.spec_from_file_location('rc3_v5.robot_input',patched_path)
    patched=importlib.util.module_from_spec(spec)
    sys.modules['rc3_v5.robot_input']=patched
    spec.loader.exec_module(patched)
    original_tests=importlib.reload(original_tests)
    after=run_case(original_tests.RobotInputTests)

    class ChainRegressionTests(original_tests.RobotInputTests):
        def update_geometry_identity(self,tree):
            # Changes only the synthetic temporary fixture. Every revision is
            # rebound to a fresh asset identity and the same synthetic evidence.
            self.urdf.write_text(ET.tostring(tree,encoding='unicode'))
            identity={k:v for k,v in self.config['identity'].items() if k!='urdf_sha256'}
            manifest=patched.make_manifest(self.urdf,self.root,{'Test':'robot'},identity)
            self.write(self.manifest,manifest)
            self.config['identity']=manifest['identity']
            self.config['sources']['fixture']['identity']=manifest['identity']

        def test_swapped_gripper_roles_block_both_sides(self):
            exchange_grippers(self)
            result=self.check()
            fields={x['field'] for x in result['blocked_inputs']}
            self.assertEqual(result['status'],'blocked_inputs')
            self.assertIn('grippers.left.chain',fields)
            self.assertIn('grippers.right.chain',fields)

        def test_declared_left_gripper_physically_on_right_arm_blocks(self):
            tree=ET.parse(self.urdf).getroot()
            tree.find("joint[@name='lg']/parent").set('link','right_palm')
            self.update_geometry_identity(tree)
            result=self.check()
            self.assertTrue(any(x['field']=='grippers.left.chain' for x in result['blocked_inputs']))
            self.assertFalse(result['physical_ik_verified'])

        def test_gripper_upstream_of_own_arm_blocks(self):
            tree=ET.parse(self.urdf).getroot()
            tree.find("joint[@name='lg']/parent").set('link','mast')
            self.update_geometry_identity(tree)
            result=self.check()
            self.assertTrue(any(x['field']=='grippers.left.chain' for x in result['blocked_inputs']))

        def test_valid_fixed_tool_connector_is_allowed(self):
            tree=ET.parse(self.urdf).getroot()
            mount=ET.SubElement(tree,'link',name='left_tool_mount')
            geometry=ET.SubElement(ET.SubElement(mount,'collision'),'geometry')
            ET.SubElement(geometry,'box',size='0.1 0.1 0.1')
            joint=ET.SubElement(tree,'joint',name='left_tool_fixed',type='fixed')
            ET.SubElement(joint,'parent',link='left_palm')
            ET.SubElement(joint,'child',link='left_tool_mount')
            tree.find("joint[@name='lg']/parent").set('link','left_tool_mount')
            self.update_geometry_identity(tree)
            result=self.check()
            self.assertEqual(result['status'],'inputs_complete')
            self.assertEqual(result['blocked_inputs'],[])
            self.assertFalse(result['physical_ik_verified'])
            self.assertFalse(result['collision_verified'])

    original_suite=unittest.defaultTestLoader.loadTestsFromTestCase(original_tests.RobotInputTests)
    new_names=[n for n in ChainRegressionTests.__dict__ if n.startswith('test_')]
    regressions=unittest.TestSuite(ChainRegressionTests(n) for n in sorted(new_names))
    log=io.StringIO()
    with contextlib.redirect_stdout(log):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.TestSuite([original_suite,regressions]))
    (out/'tests.log').write_text(log.getvalue())
    assets=src/'rc3_v5/robot_assets'
    original_template=original.preflight(assets/'manifest.json',assets/'runtime_config.template.json')
    patched_template=patched.preflight(assets/'manifest.json',assets/'runtime_config.template.json')
    summary={'tests_run':result.testsRun,'original_tests':15,'new_regressions':len(new_names),
             'all_passed':result.wasSuccessful(),'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
             'original_counterexample_status':before['swapped']['status'],'patched_counterexample_status':after['swapped']['status'],
             'original_template_preserved':original_template==patched_template,'template_blockers':len(patched_template['blocked_inputs']),
             'original_source_sha256':original_hash,'patched_source_sha256':digest(patched_path),
             'original_source_unchanged':original_hash==digest(original_path),
             'torch_imported':'torch' in sys.modules,'real_calibration_or_ik_performed':False,
             'scope':'Only synthetic input consistency regressions. No real values filled, IK, collision computation, training or planner changes.'}
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    (out/'counterexample_before_after.json').write_text(json.dumps({'before':before,'after':after},ensure_ascii=False,indent=2))
    assert result.wasSuccessful() and result.testsRun==19
    assert before['swapped']['status']=='inputs_complete' and after['swapped']['status']=='blocked_inputs'
    assert original_template==patched_template and summary['original_source_unchanged']
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
