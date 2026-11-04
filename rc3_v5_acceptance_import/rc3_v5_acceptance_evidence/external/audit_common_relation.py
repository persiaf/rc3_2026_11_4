"""Trace effective external coverage through construction, repair and export.

Only synthetic tables are used. This is not robot calibration or an IK run.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    args=ap.parse_args(); source=args.source_root.resolve(); out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(source))
    import numpy as np
    from rc3_v5 import external
    from rc3_v5.test_external import later_stop_counterexample
    from rc3_v5.external_example import make_demo,reseal
    import rc3_v3.environment as envmod
    import rc3_fast.solver as solver
    import rc3_improve.environment as oldenv

    paths=['rc3_v5/external.py','rc3_v5/test_external.py','rc3_v4/external.py',
           'rc3_v3/environment.py','rc3_fast/solver.py','rc3_improve/repair.py',
           'rc3_improve/environment.py']
    before={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in paths}
    cases={'canonicalization':later_stop_counterexample()[0]}
    b=make_demo(); f,a=external.check_bundle(b); allowed=f[0,:,0].copy()
    f[0,:,0]=f[0,:,1]=allowed; a[0]=False; a[0,:,1]=allowed
    cases['common_arm']=reseal(b,f,a)
    records=[]
    for case_name,bundle in cases.items():
        unchanged=copy.deepcopy(bundle); f,a=external.check_bundle(bundle); common=f&a
        calls=[]
        def wrap(label, fn):
            def checked(problem,*a,**kw):
                assert np.array_equal(problem.F,common), label
                calls.append(label)
                return fn(problem,*a,**kw)
            return checked
        for method in ('fixed3','detour'):
            calls.clear(); callback_calls=[]
            def checker(task,candidate,arm,certificate):
                i,j,k=task['id'],candidate['id'],'LR'.index(arm)
                assert f[i,j,k] and a[i,j,k]
                assert certificate==bundle['audit']
                callback_calls.append([i,j,arm]); return True
            with (patch.object(solver,'construct',wrap('construct',solver.construct)),
                  patch.object(envmod,'reference_candidates',wrap('reference',envmod.reference_candidates)),
                  patch.object(envmod,'canonical_route',wrap('environment_canonical',envmod.canonical_route)),
                  patch.object(envmod,'repair_window',wrap('repair',envmod.repair_window)),
                  patch.object(oldenv,'canonical_route',wrap('export_canonical',oldenv.canonical_route))):
                result=external.plan_external(bundle,method,service_checker=checker)
            assert result['status']=='complete',result
            plan=result['plans'][method]; valid=external.validate_external(bundle,plan)
            assert valid['valid_model_plan'] and valid['all_tasks_served']
            assert bundle==unchanged
            for label in ('reference','construct','environment_canonical','export_canonical'):
                assert label in calls
            if method=='detour':assert 'repair' in calls
            records.append(dict(case=case_name,method=method,common_relation_checked_at=sorted(set(calls)),
                phase_call_counts={label:calls.count(label) for label in sorted(set(calls))},
                independent_validator_passed=True,bundle_unchanged=True,callback_calls=callback_calls,
                objective_m=plan['objective_length_plus_stops_m'],visit_order=plan['visit_order_candidate_ids'],
                all_assignment_common=True,dp_calls=result['diagnostics']['dp_calls']))
    after={p:hashlib.sha256((source/p).read_bytes()).hexdigest() for p in paths}
    report=dict(all_passed=True,records=records,source_hashes_before=before,
        source_hashes_after=after,source_unchanged=before==after,torch_imported='torch' in sys.modules,
        scope='Synthetic call-chain checks only; no physical assets loaded, no IK/calibration, no retraining.')
    (out/'common_relation_trace.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_hashes_before','source_hashes_after')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
