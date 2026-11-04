"""Generate synthetic interoperability fixtures; these are not robot experiments."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from rc3_baseline.baseline import make_scene
from rc3_fast.solver import vectorized_feasibility
from .external import (SCHEMA, CONSTRAINTS, make_certificate, save_bundle,
                       plan_external, validate_external)


def source(producer):
    return dict(producer_id=producer, generator='explicit-synthetic-height-arm-table',
                generator_version='1', robot_model_id='DEMO-NO-PHYSICAL-ROBOT',
                generated_utc='2026-11-03T00:00:00Z', evidence_level='synthetic_demo',
                asset_sha256={'synthetic_spec': hashlib.sha256(b'rc3-v4-demo-not-IK').hexdigest()},
                constraint_checks={k: 'synthetic' for k in CONSTRAINTS},
                navigation_footprint_scope='conservative_circular_bound_all_candidate_heights')


def make_demo():
    scene = make_scene(2, 1, 870001, counts=[2, 2])
    original, _, _ = vectorized_feasibility(scene)
    candidates = copy.deepcopy(scene['candidates'])
    scene['candidates'] = []
    for c in candidates:
        for h in (.4, .7):
            v = copy.deepcopy(c)
            v.update(id=len(scene['candidates']), h=h)
            scene['candidates'].append(v)
    F = np.zeros((4, len(scene['candidates']), 2), dtype=bool)
    for i in range(4):
        # Explicit artificial requirement: even tasks at low height by L,
        # odd tasks at high height by R. Geometry only picks plausible docks.
        arm, height_index = i % 2, i % 2
        allowed = np.flatnonzero(original[i, :, arm])
        if not len(allowed):
            raise RuntimeError('synthetic fixture has no geometrically plausible dock')
        F[i, 2*allowed+height_index, arm] = True
        scene['tasks'][i]['service_mode'] = 'independent_single_arm'
        scene['tasks'][i]['synthetic_height_requirement_m'] = (.4, .7)[height_index]
    planning = make_certificate(scene, F, source('synthetic-planning-producer'), role='planning')
    audit = make_certificate(scene, F, source('synthetic-audit-producer'), role='audit')
    return dict(schema=SCHEMA, scene=scene, planning=planning, audit=audit)


def reseal(bundle, F=None, A=None):
    b = copy.deepcopy(bundle)
    for role, values in (('planning', F), ('audit', A)):
        if values is None:
            values = b[role]['F']
        b[role] = make_certificate(b['scene'], values, b[role]['source'], role=role)
    return b


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    bundle = make_demo()
    save_bundle(bundle, output/'external_demo.json')
    save_bundle(bundle, output/'external_demo.npz')
    records = {}
    for method in ('fixed3', 'detour'):
        result = plan_external(bundle, method=method)
        (output/f'{method}_plan.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        if result['status'] != 'complete':
            raise AssertionError(result)
        plan = result['plans'][method]
        check = validate_external(bundle, plan)
        records[method] = dict(status=result['status'], objective=plan['objective_length_plus_stops_m'],
                               heights=sorted(set(v['pose']['h'] for v in plan['service_stops'])),
                               arms=sorted(set(a['arm'] for a in plan['assignments'])),
                               independent_validation=check)
    F = np.asarray(bundle['planning']['F'], dtype=bool)
    low = F.copy()
    for j, c in enumerate(bundle['scene']['candidates']):
        if c['h'] != .4:
            low[:, j] = False
    only_left = F.copy()
    only_left[:, :, 1] = False
    reject = F.copy()
    reject[1] = False
    for name, b in [('fixed_height_only', reseal(bundle, low, low)),
                    ('left_arm_only', reseal(bundle, only_left, only_left)),
                    ('audit_disagrees', reseal(bundle, F, reject))]:
        result = plan_external(b)
        records[name] = result
        if result['status'] != 'failed':
            raise AssertionError(name+' should reject this synthetic case')
    proxy, _, _ = vectorized_feasibility(bundle['scene'])
    records['proxy_vs_synthetic_table'] = dict(
        proxy_positive_external_negative=int((proxy & ~F).sum()),
        proxy_negative_external_positive=int((~proxy & F).sum()),
        note='Synthetic assumptions only; this is not measured IK false-positive/negative accuracy.')
    records['scope'] = ('Interoperability/guard tests on a deliberately specified table; '
                        'no actual robot model, no measured lift/dual-arm benefit.')
    (output/'diagnostics.json').write_text(json.dumps(records, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v.get('status') for k, v in records.items() if isinstance(v, dict)}))
    return records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'examples')
    run(parser.parse_args().output)
