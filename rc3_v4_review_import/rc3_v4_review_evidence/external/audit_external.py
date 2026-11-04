"""Read-only v4 external/certification audit; writes only to --output-dir."""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    source=args.source_root.resolve();out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(source));sys.dont_write_bytecode=True
    from rc3_v4.external import (load_bundle,save_bundle,check_bundle,make_certificate,
                                plan_external,validate_external,CONSTRAINTS)
    from rc3_v4.external_example import make_demo,reseal
    from rc3_v4.plan import Planner
    from rc3_fast.scenes import make_complex_scene
    from rc3_v4.audit import audit_manifest
    start=time.perf_counter()
    watched=[source/'rc3_v4'/n for n in ('external.py','runtime.py','audit.py','plan.py','verify.py')]
    before={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}

    tests=unittest.TestLoader().loadTestsFromNames([
        'rc3_v4.test_external','rc3_v4.test_audit','rc3_v4.test_runtime'])
    stream=io.StringIO()
    unit=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
    (out/'unit_tests.log').write_text(stream.getvalue(),encoding='utf-8')
    assert unit.wasSuccessful()

    bundle=make_demo();roundtrips={}
    for suffix in ('json','npz'):
        file=out/f'external_demo.{suffix}'
        save_bundle(bundle,file)
        roundtrips[suffix]=load_bundle(file)==bundle
        assert roundtrips[suffix]
    F,A=check_bundle(bundle)
    results={}
    with (patch('rc3_fast.solver.vectorized_feasibility',side_effect=AssertionError('proxy forbidden')),
          patch('rc3_baseline.validator.proxy_reachable',side_effect=AssertionError('proxy forbidden'))):
        for method in ('fixed3','detour'):
            result=plan_external(bundle,method)
            assert result['status']=='complete'
            plan=result['plans'][method]
            check=validate_external(bundle,plan)
            assert check['valid_model_plan']
            assert result['diagnostics']['cost_nonincreasing']
            (out/f'replayed_external_{method}.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            results[method]=dict(objective=plan['objective_length_plus_stops_m'],
                route=plan['visit_order_candidate_ids'],arms=sorted({a['arm'] for a in plan['assignments']}),
                heights=sorted({s['pose']['h'] for s in plan['service_stops']}),check=check)

    anchor=plan_external(bundle,'fixed3')['plans']['fixed3']
    counted=[]
    def record_callback(task,candidate,arm,cert):
        counted.append((task['id'],candidate['id'],arm));return True
    check=validate_external(bundle,anchor,record_callback)
    assert check['service_checker_calls']==len(bundle['scene']['tasks'])==len(counted)

    denied=plan_external(bundle,'detour',service_checker=lambda *unused:False)
    assert denied['status']=='failed' and denied['pending_task_ids']==[0,1,2,3]
    def raise_callback(*unused):raise RuntimeError('injected checker failure')
    raised=plan_external(bundle,'detour',service_checker=raise_callback)
    assert raised['status']=='failed' and raised['pending_task_ids']==[0,1,2,3]
    final_calls=[]
    def reject_final(task,candidate,arm,cert):
        final_calls.append(task['id']);return len(final_calls)<=4
    final_rejected=plan_external(bundle,'detour',service_checker=reject_final)
    assert final_rejected['status']=='complete' and final_rejected['diagnostics']['fallback_used']
    assert final_rejected['diagnostics']['objective_m']==final_rejected['diagnostics']['reference_objective_m']
    assert validate_external(bundle,final_rejected['plans']['detour'])['valid_model_plan']
    (out/'final_callback_rejection.json').write_text(json.dumps(final_rejected,indent=2),encoding='utf-8')

    # Content binding is not provenance or physical evidence. This is an intended
    # trust boundary explicitly documented by the upload, not a hash bypass.
    relabelled=copy.deepcopy(bundle)
    for role in ('planning','audit'):
        src=relabelled[role]['source']
        src['evidence_level']='externally_attested'
        src['constraint_checks']={name:'checked' for name in CONSTRAINTS}
        src['asset_sha256']={name:hashlib.sha256(('DECLARATION-ONLY-'+name).encode()).hexdigest()
                            for name in ('kinematics','joint_limits','collision_geometry','task_pose_spec','lift_spec')}
        relabelled[role]=make_certificate(relabelled['scene'],relabelled[role]['F'],src,role=role)
    declaration=plan_external(relabelled)
    assert declaration['status']=='complete'
    assert declaration['plans']['fixed3']['validation']['built_in_physical_ik_recheck'] is False

    # A feasible common-table service can be missed by deterministic L-first
    # export. Preserve an independently accepted witness on the exact same route.
    altF=F.copy();altA=A.copy()
    allowed=F[0,:,0].copy()
    altF[0,:,1]=allowed
    altA[0,:,0]=False;altA[0,:,1]=allowed
    alternate=reseal(bundle,altF,altA)
    incomplete=plan_external(alternate,'fixed3')
    assert incomplete['status']=='failed' and incomplete['physical_infeasibility_claim'] is False
    witness=copy.deepcopy(anchor)
    witness['reachability_certificate_sha256']=alternate['planning']['certificate_sha256']
    witness['audit_certificate_sha256']=alternate['audit']['certificate_sha256']
    for a in witness['assignments']:
        if a['task_id']==0:a['arm']='R'
    for stop in witness['service_stops']:
        for a in stop['tasks']:
            if a['task_id']==0:a['arm']='R'
    witness_check=validate_external(alternate,witness)
    assert witness_check['valid_model_plan']
    alternate_record=dict(scope='Incomplete L-first export, not an invalid accepted plan or physical counterexample',
        bundle=alternate,planner_result=incomplete,valid_same_route_witness=witness,witness_check=witness_check)
    (out/'alternate_arm_completeness.json').write_text(json.dumps(alternate_record,indent=2),encoding='utf-8')

    smoke=[]
    scene=make_complex_scene(4,3,730920,'blocked')
    for method in ('fixed3','detour','learned','legacy'):
        planner=Planner(method=method)
        cold=planner.plan(scene);warm=planner.plan(scene)
        assert cold['status']==warm['status']=='complete'
        assert cold['plans']==warm['plans'] and warm['timing']['cache_hit']
        assert cold['diagnostics']['cost_nonincreasing']
        smoke.append(dict(method=method,objective=cold['diagnostics']['objective_m'],
                          cold_cache_hit=cold['timing']['cache_hit'],warm_cache_hit=warm['timing']['cache_hit']))
    missing=copy.deepcopy(scene)
    for t in missing['tasks']:t['z']=9.
    bad=Planner().plan(missing)
    assert bad['status']=='failed' and bad['pending_task_ids']==[t['id'] for t in missing['tasks']]
    assert 'torch' not in sys.modules

    package=audit_manifest(source/'PACKAGE_MANIFEST.json',source)
    assert package['all_passed']
    (out/'package_integrity.json').write_text(json.dumps(package,indent=2),encoding='utf-8')
    after={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    assert before==after
    summary=dict(all_checks_passed=True,unit_tests_run=unit.testsRun,unit_failures=len(unit.failures),
        unit_errors=len(unit.errors),unit_skipped=len(unit.skipped),torch_imported=False,
        roundtrips=roundtrips,external_methods=results,callback=dict(counted_calls=len(counted),
        false_rejected=True,exception_rejected=True,final_rejection_returned_anchor=True,
        final_trial_calls=len(final_calls)),proxy_smoke=smoke,invalid_proxy_task_ids_preserved=True,
        metadata_trust_boundary=dict(relabelled_declaration_accepted=True,built_in_physical_ik_recheck=False,
        interpretation='Expected: self-declared asset hashes and status labels bind records but do not independently validate physics or provenance.'),
        alternate_arm_completeness=dict(planner_failed=True,valid_same_route_witness=True,
        interpretation='F declares both arms, audit A permits R only. Export picks L and rejects all anchors, despite an independently accepted R assignment.'),
        package_files_checked=package['checked_files'],package_files_matched=package['matched_files'],
        original_source_unchanged=before==after,source_sha256=after,elapsed_s=time.perf_counter()-start,
        scope='No robot operation, no IK computation, no retraining, no full Torch suite; original upload files are read-only.')
    (out/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
