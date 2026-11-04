"""Local asset and configuration-input audit; no calibration, IK or network."""
from __future__ import annotations
import argparse
import collections
import contextlib
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import unittest
import xml.etree.ElementTree as ET

import numpy as np

sys.dont_write_bytecode = True

def sha(data):
    return hashlib.sha256(data).hexdigest()

def blob(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    source = args.source_root.resolve(); out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if out.is_relative_to(source):
        raise ValueError('audit outputs must not modify source tree')
    sys.path.insert(0, str(source))
    from rc3_v5 import robot_input as r
    from rc3_v5.test_robot_input import RobotInputTests
    assets = source/'rc3_v5/robot_assets'
    manifest = json.loads((assets/'manifest.json').read_text())
    recorded_comparison = json.loads((assets/'source_comparison.json').read_text())
    saved_listings = json.loads((assets/'upstream_listings.json').read_text())
    supplementary = json.loads((assets/'supplementary_fetch.json').read_text())
    mesh_listings = [saved_listings[0], json.loads(supplementary['official_meshes']['content'])]
    urdf_listings = json.loads((assets/'upstream_urdf_listings.json').read_text())
    meshes = []
    dtype = np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    assert dtype.itemsize == 50
    for record in manifest['assets']:
        path = (assets/record['path']).resolve()
        assert path.is_relative_to(assets.resolve())
        raw = path.read_bytes()
        count = struct.unpack('<I', raw[80:84])[0]
        triangles = np.frombuffer(raw, dtype=dtype, offset=84)
        refs = []
        for listing in mesh_listings:
            item = next(x for x in listing if x['name'] == path.name)
            refs.append({'path':item['path'],'size_matches':len(raw)==item['size'],'blob_matches':blob(raw)==item['sha']})
        result = {
            'uri':record['uri'], 'path':record['path'], 'bytes':len(raw),
            'sha256_matches':sha(raw)==record['sha256'], 'byte_count_matches':len(raw)==record['bytes'],
            'git_blob_matches':blob(raw)==record['git_blob_sha1'],
            'binary_stl_triangles':count, 'binary_stl_size_valid':len(raw)==84+50*count,
            'all_normals_finite':bool(np.isfinite(triangles['normal']).all()),
            'all_vertices_finite':bool(np.isfinite(triangles['vertices']).all()),
            'coordinate_min':triangles['vertices'].min(axis=(0,1)).astype(float).tolist(),
            'coordinate_max':triangles['vertices'].max(axis=(0,1)).astype(float).tolist(),
            'saved_fixed_commit_listings':refs,
        }
        assert all(result[k] for k in ('sha256_matches','byte_count_matches','git_blob_matches','binary_stl_size_valid','all_normals_finite','all_vertices_finite'))
        assert all(x['size_matches'] and x['blob_matches'] for x in refs)
        meshes.append(result)

    paths = [assets/manifest['urdf']['path'], assets/'official_reference/Aloha/urdf/Aloha.urdf']
    urdfs = []
    raw_urdfs = []
    for ix,path in enumerate(paths):
        raw=path.read_bytes(); raw_urdfs.append(raw)
        tree=ET.fromstring(raw)
        joints=tree.findall('joint'); links=tree.findall('link')
        bounded=[j for j in joints if j.get('type') in ('revolute','prismatic')]
        zero=[j.get('name') for j in bounded if j.find('limit') is not None and all(float(j.find('limit').get(k,'nan'))==0.0 for k in ('lower','upper','velocity','effort'))]
        references={m.get('filename') for m in tree.findall('.//geometry/mesh')}
        listing=next(x for x in urdf_listings[ix] if x['name']=='Aloha.urdf')
        expected=recorded_comparison['simulation' if ix==0 else 'official_reference']
        result={
            'path':path.relative_to(source).as_posix(),'bytes':len(raw), 'sha256':sha(raw),'git_blob_sha1':blob(raw),
            'matches_comparison_record':len(raw)==expected['bytes'] and sha(raw)==expected['urdf_sha256'] and blob(raw)==expected['urdf_git_blob_sha1'],
            'matches_saved_listing':len(raw)==listing['size'] and blob(raw)==listing['sha'],
            'links':len(links),'joints':len(joints),'joint_types':dict(collections.Counter(j.get('type') for j in joints)),
            'bounded_joints':len(bounded),'zero_lower_upper_velocity_effort':zero,'mesh_uris':len(references),
            'all_mesh_uris_in_manifest':references=={x['uri'] for x in manifest['assets']},
            'all_links_have_collision_geometry':all(link.find('collision/geometry') is not None for link in links)
        }
        assert result['matches_comparison_record'] and result['matches_saved_listing'] and result['all_mesh_uris_in_manifest']
        urdfs.append(result)
    equal_normalized=raw_urdfs[0].replace(b'\r\n',b'\n')==raw_urdfs[1].replace(b'\r\n',b'\n')
    assert equal_normalized

    stdout=io.StringIO()
    with contextlib.redirect_stdout(stdout):
        inventory_exit=r.main(['inventory','--manifest',str(assets/'manifest.json'),'--output',str(out/'asset_inventory.json')])
        preflight_exit=r.main(['check','--manifest',str(assets/'manifest.json'),'--config',str(assets/'runtime_config.template.json'),'--output',str(out/'preflight_report.json')])
    (out/'cli.log').write_text(stdout.getvalue())
    inventory=json.loads((out/'asset_inventory.json').read_text())
    preflight=json.loads((out/'preflight_report.json').read_text())
    original_preflight=json.loads((assets/'preflight_report.json').read_text())
    assert inventory_exit==0 and inventory['status']=='assets_verified'
    assert preflight_exit==2 and preflight['status']=='blocked_inputs'
    assert len(preflight['blocked_inputs'])==135
    assert preflight==original_preflight

    stream=io.StringIO()
    with contextlib.redirect_stdout(stream):
        tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RobotInputTests))
    (out/'unit_tests.log').write_text(stream.getvalue())
    assert tests.wasSuccessful() and tests.testsRun==15

    # Reuse the existing synthetic fixture only. This is a configuration-chain
    # consistency counterexample, not calibration of either Aloha arm.
    case=RobotInputTests(); case.setUp()
    try:
        baseline=case.check()
        case.config['joints']['lg']['role']='right_gripper'
        case.config['joints']['rg']['role']='left_gripper'
        case.config['grippers']['left']['joint']='rg'
        case.config['grippers']['right']['joint']='lg'
        case.config['park']['left']['positions']={'left':0.2,'rg':0.2}
        case.config['park']['right']['positions']={'right':0.2,'lg':0.2}
        crossed=case.check()
        counterexample={'scope':'Existing synthetic validator fixture only; no calibration or IK',
                        'baseline_status':baseline['status'],'actual_joint_parents':{'lg':'left_palm','rg':'right_palm'},
                        'mutated_roles':{'lg':'right_gripper','rg':'left_gripper'},'result':crossed}
        (out/'gripper_chain_counterexample.json').write_text(json.dumps(counterexample,ensure_ascii=False,indent=2))
    finally:
        case.doCleanups()
    summary={
        'asset_checks_passed':True,'mesh_count':len(meshes),'mesh_bytes':sum(x['bytes'] for x in meshes),
        'stl_format_and_finite_geometry_checked':True,'urdfs':urdfs,'urdf_equal_after_crlf_normalization':equal_normalized,
        'inventory_exit_code':inventory_exit,'inventory_status':inventory['status'],
        'template_preflight_exit_code':preflight_exit,'template_status':preflight['status'],
        'template_blockers':len(preflight['blocked_inputs']),'template_report_matches_saved':preflight==original_preflight,
        'test_count':tests.testsRun,'tests_passed':tests.wasSuccessful(),'test_failures':len(tests.failures),'test_errors':len(tests.errors),
        'cross_arm_gripper_assignment_false_acceptance':crossed['status']=='inputs_complete',
        'physical_ik_verified':False,'collision_verified':False,'robot_calibration_performed':False,
        'network_source_retrieval_performed':False,
        'source_file_hashes':{n:sha((source/'rc3_v5'/n).read_bytes()) for n in ('robot_input.py','test_robot_input.py','robot_assets/runtime_config.template.json')},
        'scope':'Local files compared to manifest and saved fixed-commit listings; binary geometry integrity is not robot calibration, collision validity or execution certification. All outputs outside original source.'
    }
    (out/'mesh_checks.json').write_text(json.dumps(meshes,ensure_ascii=False,indent=2))
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
