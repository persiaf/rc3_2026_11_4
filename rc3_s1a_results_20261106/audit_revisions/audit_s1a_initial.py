"""Read-only independent audit of saved S1-A inputs, plans, and paid ledgers.

No scene/roadmap creation, solver, native search, H, model, or optimizer calls.
Run only after the scientific process has stopped. Output stays in this new
audit directory; the approved runtime/source-binding directory is unchanged.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import contextlib
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import traceback
from unittest.mock import patch

import numpy as np

from rc3_fast.planner import map_key
from rc3_global_local.problem import array_hash
from rc3_global_local.validation import validate
from rc3_global_local_impl_review.runtime_guards import (
    bind_navigation, snapshot_parent, extract_snapshot, current_navigation_sources)

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUNTIME = ROOT/'rc3_global_local_execution_v3'
RUN = RUNTIME/'runs/rc3_v3_global_local_20261106_01'
EXPECTED = dict(protocol='0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb',
    source_manifest='435b1c8be95d6ec4f14f4aa87913a6eb3e937e3684eaadfabbad160b57ffe968',
    registry='75d110bef7e9954e9ace14198d262a940af6aacb17abad71cf83771ace8b363a',
    H='5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14')


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8', newline='\n')


def jsonl(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line]


class Auditor:
    def __init__(self, run):
        self.run = run.resolve()
        if self.run != RUN.resolve(): raise ValueError('Only the authorized scientific run may be audited')
        self.counts = {}
        self.digests = {}
        self.nav_cache = OrderedDict()
        self.rows = []
        self.paid_key_results = {}
        self.forbidden = []

    def inc(self, key, n=1): self.counts[key] = self.counts.get(key, 0)+n

    def need(self, value, message):
        self.inc('assertions')
        if not value: raise AssertionError(message)

    def digest(self, path):
        path = Path(path).resolve()
        if path not in self.digests:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4*1024*1024), b''): h.update(block)
            self.digests[path] = h.hexdigest()
            self.inc('files_hashed'); self.inc('bytes_hashed', path.stat().st_size)
        return self.digests[path]

    @contextlib.contextmanager
    def no_scientific_calls(self):
        import rc3_baseline.baseline as baseline
        import rc3_fast.scenes as scenes
        import rc3_constructive.problem as problem
        import rc3_constructive.baselines as constructive
        import rc3_global_local.terminal as terminal
        from rc3_v4 import runtime
        def forbidden(name):
            def fail(*args, **kwargs):
                self.forbidden.append(name)
                raise AssertionError('Independent audit attempted scientific computation: '+name)
            return fail
        with contextlib.ExitStack() as stack:
            for obj, name in [(baseline, 'navigation'), (scenes, 'make_complex_scene'),
                (problem, 'resample_tasks'), (constructive, 'postprocess'),
                (terminal, 'finish'), (runtime, 'run')]:
                stack.enter_context(patch.object(obj, name, forbidden(name)))
            yield

    def bindings(self):
        manifest = read(RUNTIME/'EXECUTION_SOURCE_BINDING.json')
        self.need(self.digest(RUNTIME/'V3_EXECUTION_CANDIDATE.json') == EXPECTED['protocol'], 'Protocol bytes differ')
        self.need(self.digest(RUNTIME/'EXECUTION_SOURCE_BINDING.json') == EXPECTED['source_manifest'], 'Source manifest bytes differ')
        self.need(self.digest(RUNTIME/'REGISTRY.json') == EXPECTED['registry'], 'Registry bytes differ')
        self.need(manifest['H_sha256'] == EXPECTED['H'], 'H identity differs')
        for name, expected in manifest['source_sha256'].items():
            self.need(self.digest(ROOT/name) == expected, 'Runtime source changed: '+name)
        self.need(signature(manifest['H_source_sha256']) == EXPECTED['H'], 'H canonical identity differs')
        run_manifest = read(self.run/'RUN_MANIFEST.json')
        binding = run_manifest['binding']
        self.binding = binding
        for key, expected in [('protocol_sha256', EXPECTED['protocol']), ('registry_sha256', EXPECTED['registry']),
            ('source_manifest_sha256', EXPECTED['source_manifest']), ('H_sha256', EXPECTED['H'])]:
            self.need(binding[key] == expected, 'Run science binding differs: '+key)
        self.need(binding['source_sha256'] == manifest['source_sha256'], 'Run sources differ from approved map')
        self.need(binding['experiment_id'] == self.run.name, 'Wrong experiment ID')
        self.need(not binding.get('engineering_fixture'), 'Engineering run cannot be audited as science')
        approvals = []
        authorization_folder = RUNTIME/'authorizations'
        if authorization_folder.exists():
            for f in authorization_folder.rglob('*.json'):
                d = read(f)
                if isinstance(d, dict) and d.get('status') == 'APPROVED' and d.get('approved_by') == 'user':
                    if self.digest(f) == binding['approval_sha256']: approvals.append(f)
        self.need(len(approvals) == 1, 'Exact separate user protocol approval is absent or ambiguous')
        approval = read(approvals[0]); self.need(bool(approval.get('authorization_text')), 'Approval user source missing')
        for key in ['protocol_sha256', 'registry_sha256', 'source_sha256', 'H_sha256', 'source_manifest_sha256']:
            self.need(approval[key] == binding[key], 'Approval does not bind '+key)
        declaration = read(self.run/'stage_admission/S1-A/AUTHORIZATION_SOURCE.json')['declaration']
        self.need(declaration.get('status') == 'AUTHORIZED' and declaration.get('authorized_by') == 'user', 'S1-A user authorization missing')
        self.need(declaration['stages'] == ['S1-A'] and declaration['optimizer_seeds'] == [], 'Authorization exceeded S1-A scope')
        self.need(declaration['experiment_id'] == self.run.name and declaration['predecessors'] == {}, 'Authorization run/parent differs')
        self.need(declaration['protocol_approval_sha256'] == binding['approval_sha256'], 'Stage approval lineage differs')
        self.need(bool(declaration.get('authorization_text')), 'Stage user source missing')
        for key in ['protocol_sha256', 'registry_sha256', 'source_sha256', 'H_sha256', 'source_manifest_sha256']:
            self.need(declaration[key] == binding[key], 'Stage does not bind '+key)
        registry = read(RUNTIME/'REGISTRY.json')
        self.specs = [dict(s, role='train', mode=registry['S2_train'][s['parent']]['mode']) for s in registry['S1_teachers']]
        self.need(len(self.specs) == 256, 'Teacher declaration count differs')
        gate = read(ROOT/'rc3_global_local_impl_review/GEOMETRY_GATE.json')
        self.map_keys = {(r['map_seed'], r['mode']): r['map_key'] for r in gate['records'] if r['role'] == 'train'}
        return dict(approval_path=approvals[0].relative_to(ROOT).as_posix(), approval_sha256=binding['approval_sha256'],
            stage_authorization_user_source=True, scientific_identity=binding)

    def navigation(self, scene, key):
        if key in self.nav_cache:
            self.nav_cache.move_to_end(key)
            return self.nav_cache[key]
        path = self.run/'data/navigation'/(key+'.npz')
        meta = read(path.with_suffix('.json'))
        self.need(meta['map_key'] == key and self.digest(path) == meta['sha256'], 'Navigation artifact identity differs')
        self.need(meta['source_sha256'] == current_navigation_sources(), 'Navigation source differs')
        with np.load(path, allow_pickle=False) as a:
            nav = dict(costs=a['costs'], coordinates=a['coordinates'].tolist(), service_nodes=a['service_nodes'].tolist(),
                predecessors=a['predecessors'], node_count=meta['node_count'], edge_count=meta['edge_count'])
        sealed = bind_navigation(scene, nav, source_map_key=key, source_sha256=meta['source_sha256'],
            provenance=dict(producer='formal_registered_navigation', artifact_sha256=self.digest(path), map_key=key))
        self.inc('saved_navigation_array_loads')
        self.nav_cache[key] = sealed
        while len(self.nav_cache) > 2: self.nav_cache.popitem(last=False)
        return sealed

    def view(self, spec):
        folder = self.run/'data/task_draws'/f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
        meta = read(folder/'receipt.json')
        expected = {k: spec[k] for k in ('role', 'map_seed', 'task_seed', 'mode', 'A_condition')}
        self.need(meta['spec'] == expected and meta['source_sha256'] == self.binding['source_sha256'], 'Saved task draw provenance differs')
        self.need(self.digest(folder/'scene.json') == meta['scene_sha256'], 'Task scene bytes differ')
        self.need(self.digest(folder/'arrays.npz') == meta['arrays_sha256'], 'F/A bytes differ')
        scene = read(folder/'scene.json')
        with np.load(folder/'arrays.npz', allow_pickle=False) as a:
            F, A = a['F'], a['A']
        self.inc('saved_task_array_loads')
        self.need(F.shape == A.shape == (len(scene['tasks']), len(scene['candidates']), 2), 'F/A shapes differ')
        self.need(F.dtype.kind in 'biu' and A.dtype.kind in 'biu' and np.isin(F, [0,1]).all() and np.isin(A, [0,1]).all(), 'F/A is not binary')
        c = spec['A_condition']
        if c == 0: expected_A = np.ones_like(F, dtype=bool)
        else:
            i, j, arm = np.indices(F.shape)
            expected_A = ((i+2*j+arm) % (3 if c == 1 else 4)) >= (1 if c == 1 else 2)
            for t in range(len(F)):
                ids = np.argwhere(F[t])
                if len(ids): expected_A[t, ids[0,0], ids[0,1]] = True
        self.need(np.array_equal(A, expected_A), 'Allow condition differs from fixed declaration')
        key = map_key(scene)
        self.need(key == meta['map_key'] == self.map_keys[(spec['map_seed'], spec['mode'])], 'Actual task geometry differs')
        parent = snapshot_parent(scene, F, A, navigation=self.navigation(scene, key))
        self.need(parent.identity == meta['parent_identity'], 'Immutable parent identity differs')
        connected = np.flatnonzero(np.isfinite(parent.D[0,1:]) & np.isfinite(parent.D[1:,0]))+1
        self.need(parent.effective[:, connected-1, :].any((1,2)).all(), 'Full parent task draw is not reachable/coverable')
        if spec['endpoints'] == 'closed_depot': entry = exit_ = 0
        else:
            self.need(spec['endpoints'] == 'open_first_last_connected' and len(connected) >= 2, 'Invalid endpoints')
            entry, exit_ = int(connected[0]), int(connected[-1])
        view = extract_snapshot(parent, spec['x0'], spec['y0'], spec['width'], spec['height'], entry, exit_)
        tables = {t['id'] for t in scene['tables'] if spec['x0'] <= t['ix'] < spec['x0']+spec['width'] and spec['y0'] <= t['iy'] < spec['y0']+spec['height']}
        tasks = [t['id'] for t in scene['tasks'] if t['table_id'] in tables]
        candidates = [j for j in range(len(scene['candidates'])) if parent.effective[tasks,j,:].any()
            and np.isfinite(parent.D[entry,j+1]) and np.isfinite(parent.D[j+1,exit_])]
        self.need(view.task_ids.tolist() == tasks and view.candidate_ids.tolist() == candidates, 'Window extraction removed or added IDs')
        self.inc('saved_views_reconstructed')
        return view

    def check_plan(self, view, plan, label):
        cert = validate(view, plan)
        self.need(cert['valid_model_plan'] and cert['all_tasks_served'], label+' independent scalar/geometry validation failed: '+str(cert['errors']))
        self.need(plan['validation']['valid_model_plan'] and not plan['physical_joint_trajectory_certified'], 'Saved certification differs')
        self.inc('plans_revalidated'); self.inc('assignments_revalidated', len(plan['assignments']))
        self.inc('polyline_segments_revalidated', max(1,len(plan['route_polyline'])-1))
        self.inc('segment_box_pairs_revalidated', max(1,len(plan['route_polyline'])-1)*(len(view.parent.scene['tables'])+len(view.parent.scene.get('obstacles',[]))))
        inverse = {int(g): j+1 for j,g in enumerate(view.candidate_ids)}
        route = [0, *[inverse[g] for g in plan['visit_order_candidate_ids']], 0]
        metrics = view.metrics(route)
        self.need(abs(metrics['J']-plan['objective_length_plus_stops_m']) <= 1e-7, 'Saved objective differs')
        return route, metrics

    def label(self, path, expected_index):
        row = read(path); self.need(row['index'] == expected_index and row['spec'] == self.specs[expected_index], 'Teacher row replaced/reordered')
        self.need(row['registry_sha256'] == EXPECTED['registry'], 'Label registry differs')
        view = self.view(row['spec']); self.need(view.identity == row['view_identity'], 'Teacher view identity differs')
        native, info, hr = row['native_plan'], row['native_info'], row['H_result']
        self.need(info.get('adapter_method') == 'detour16' and info.get('horizon_cap') == 16, 'Not the declared Detour16 teacher')
        self.need(info.get('common_H_added') is False and info.get('fallback_used') is False and not info.get('failures'), 'Hidden H/fallback/native failure')
        self.need(info.get('portfolio_recipes') == [0,1,7] and 0 <= info['dp_calls'] <= 16, 'Native portfolio/budget differs')
        raw, nm = self.check_plan(view, native, 'native')
        final, hm = self.check_plan(view, hr['plan'], 'H')
        self.need(hr['raw_route'] == raw and hr['route'] == final and row['actions'] == final[1:-1], 'Native/H/label route chain differs')
        for k in ('J','L','K'):
            self.need(abs(float(hr[k])-hm[k]) <= 1e-7, 'H '+k+' differs')
            self.need(abs(float(hr['raw_'+k])-nm[k]) <= 1e-7, 'Native '+k+' differs')
        self.need(abs(row['teacher_J']-hm['J']) <= 1e-7 and hm['J'] <= nm['J']+1e-7, 'Teacher label terminal cost differs/worsens')
        key = (view.identity, tuple(raw))
        self.paid_key_results[key] = (hm['J'], tuple(final))
        self.rows.append(dict(index=expected_index, spec=row['spec'], task_count=len(view.task_ids), candidate_count=len(view.candidate_ids),
            native_J=nm['J'], native_L=nm['L'], native_K=nm['K'], teacher_J=hm['J'], teacher_L=hm['L'], teacher_K=hm['K'],
            native_DP_calls=info['dp_calls'], H_cache_hit=bool(hr['cache_hit']), native_wall_s=info['adapter_elapsed_s'],
            H_s=hr['H_s'], export_s=hr['export_s'], validation_s=hr['validation_s'], valid=True,
            internal_native_search=info['internal_search']['counts'], label_sha256=self.digest(path)))

    def artifacts_and_ledgers(self, complete, labels):
        pool = self.run/'teachers256'
        events = jsonl(pool/'events.jsonl')
        counts = {k: sum(e['event'] == k for e in events) for k in set(e['event'] for e in events)}
        h_files = sorted((pool/'H_plans').glob('teacher_H_*.json')) if (pool/'H_plans').exists() else []
        for f in h_files:
            record = read(f); result = record['result']
            key = (record['problem']['identity'], tuple(result['raw_route']))
            if key in self.paid_key_results:
                j,route = self.paid_key_results[key]
                self.need(abs(j-result['J']) <= 1e-7 and tuple(result['route']) == route, 'Paid H artifact differs from matching label')
            self.digest(f); self.inc('paid_H_artifacts_checked')
        terminal_path = self.run/'stage_admission/S1-A'/('COMPLETE.json' if complete else 'FAILED.json')
        terminal = read(terminal_path)
        self.need(terminal['science_binding'] == self.binding, 'Terminal stage binding differs')
        self.need(terminal['status'] == ('COMPLETE' if complete else 'FAILED_INCOMPLETE'), 'Wrong final stage state')
        self.need(terminal['predecessors'] == {}, 'S1-A has unexpected parent')
        for name,expected in [('AUTHORIZATION.json',terminal['authorization_record_sha256']),('START.json',terminal['start_record_sha256'])]:
            self.need(self.digest(terminal_path.parent/name) == expected, 'Authorization/start bytes changed')
        for name,expected in terminal.get('artifacts',{}).items():
            f=(self.run/name).resolve();self.need(f.is_relative_to(self.run) and self.digest(f)==expected, 'Stage artifact differs: '+name)
        resources = terminal['cumulative_stage_resources']
        ctx = resources.get('shared_teachers')
        if ctx is None:
            self.need(not labels and resources.get('science_calls') == 0, 'Missing charged stage context')
            return dict(stage=terminal, event_counts=counts, resources=resources, native_attempts=0,H_attempts=0,top_level_attempts=0)
        ledger = ctx['ledger']; s = ledger.get('terminal',{}).get('teacher_H',{})
        native_attempts=ledger.get('native_teacher_attempts',0);h_attempts=s.get('terminal_attempts',0)
        self.need(native_attempts <= 256 and h_attempts <= 256 and native_attempts+h_attempts <= 512, 'Paid query cap exceeded')
        self.need(ctx['wall_cap_s'] == 1800 and ctx['elapsed_wall_s'] >= 0, 'Wrong scientific wall cap')
        self.need(s.get('terminal_cache_hits',0)+h_attempts == s.get('terminal_requests',0), 'H cache/request accounting does not conserve')
        self.need(s.get('terminal_completed',0)+s.get('terminal_failed',0) <= h_attempts, 'H paid attempts do not conserve')
        self.need(counts.get('teacher_native_start',0) == native_attempts, 'Native attempt journal differs')
        self.need(counts.get('terminal_start',0) == h_attempts, 'H attempt journal differs')
        self.need(counts.get('terminal_complete',0) == s.get('terminal_completed',0), 'H completion journal differs')
        self.need(counts.get('terminal_error',0) == s.get('terminal_failed',0), 'H failure journal differs')
        self.need(len(h_files) <= h_attempts, 'More H artifacts than charged calls')
        self.need(not any('optimizer' in k and ledger[k] for k in ledger), 'Unexpected optimizer work')
        if complete:
            self.need(ctx['elapsed_wall_s'] <= 1800, 'Completed stage exceeds wall cap')
            self.need(native_attempts == ledger['native_teacher_completed'] == labels == 256, 'Native completion count differs')
            self.need(s['terminal_requests'] == 256 and s['terminal_failed'] == 0 and s['terminal_completed'] == h_attempts, 'H final counters differ')
            self.need(len(h_files) == h_attempts, 'Missing paid H artifacts')
            self.need(s['export_attempts'] == s['exports'] == s['validation_attempts'] == s['validation_completed'] == h_attempts, 'H export/validation accounting differs')
            self.need(sum(r['H_cache_hit'] for r in self.rows) == s['terminal_cache_hits'], 'Label cache flags differ')
            native_internal=ledger['teacher_native_internal']
            for k in ('construct','two_opt','joint_search','recipe'):
                self.need(sum(r['internal_native_search'][k] for r in self.rows) == native_internal['native_'+k+'_calls'], 'Native internal '+k+' accounting differs')
            self.need(ledger['parent_requests'] == ledger['view_extractions'] == 256, 'Teacher input/view accounting differs')
            self.need(ledger['task_scene_generations']+ledger.get('parent_artifact_loads',0) == 256, 'Parent input accounting differs')
            self.need(ledger['navigation_completed']+ledger.get('navigation_artifact_loads',0)+ledger.get('navigation_memory_cache_hits',0) == 256, 'Navigation accounting differs')
            index=read(pool/'POOL_INDEX.json')
            self.need(index['ledger']==ledger, 'Pool/final resource ledger differs')
        return dict(stage=terminal,event_counts=counts,resources=resources,native_attempts=native_attempts,H_attempts=h_attempts,
            top_level_attempts=native_attempts+h_attempts,internal_native=ledger.get('teacher_native_internal'),
            internal_H=ledger.get('internal_H_search'),native_DP_calls=sum(r['native_DP_calls'] for r in self.rows))

    def run_audit(self):
        approvals=self.bindings()
        stage=self.run/'stage_admission/S1-A'
        complete=(stage/'COMPLETE.json').exists()
        self.need(complete or (stage/'FAILED.json').exists(), 'Scientific run has not stopped; audit must wait')
        self.need(not (complete and (stage/'FAILED.json').exists()), 'Both complete and failed terminal markers exist')
        pool=self.run/'teachers256';paths=sorted(pool.glob('label_*.json')) if pool.exists() else []
        if complete:
            index=read(pool/'POOL_INDEX.json')
            self.need(index['count']==256 and index['registry_sha256']==EXPECTED['registry'] and index['source_binding']==self.binding, 'Wrong complete teacher index')
            self.need([r['index'] for r in index['labels']]==list(range(256)) and len(paths)==256, 'Missing/reordered complete labels')
            for i,r in enumerate(index['labels']):
                self.need(r['file']==f'label_{i:03}.json' and self.digest(pool/r['file'])==r['sha256'], 'Teacher index bytes differ')
        with self.no_scientific_calls():
            for i,f in enumerate(paths):
                self.need(f.name==f'label_{i:03}.json','Partial labels not an exact prefix')
                self.label(f,i)
        accounting=self.artifacts_and_ledgers(complete,len(paths))
        return dict(complete=complete,approvals=approvals,accounting=accounting)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default=str(RUN))
    parser.add_argument('--output',default=str(HERE/'INDEPENDENT_AUDIT.json'))
    args=parser.parse_args()
    output=Path(args.output).resolve()
    if not output.is_relative_to(HERE): raise ValueError('Audit output outside owned directory')
    started=time.perf_counter();cpu=time.process_time();audit=Auditor(Path(args.run))
    result=dict(schema='rc3-s1a-independent-saved-teacher-audit-v1',reviewer='/root/s1a_teacher_independent_audit',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=RUN.relative_to(ROOT).as_posix(),
        expected_versions=EXPECTED,status='RUNNING',error=None)
    try:
        evidence=audit.run_audit();complete=evidence['complete']
        result.update(status='PASS_COMPLETE_TEACHER_POOL' if complete else 'PASS_PARTIAL_EVIDENCE_ONLY_S1A_INCOMPLETE',
            scientific_stage_complete=complete,S1_B_technical_decision='GO_PENDING_SEPARATE_USER_AUTHORIZATION' if complete else 'REVISE',evidence=evidence)
    except Exception as exc:
        result.update(status='AUDIT_FAILED',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()),S1_B_technical_decision='REVISE')
    result.update(counts=audit.counts,teacher_rows=audit.rows,forbidden_operations=audit.forbidden,
        independent_scientific_calls=dict(native=0,H=0,scene_generation=0,navigation_generation=0,model=0,optimizer=0,S3_queries=0,old60_semantic_reads=0),
        wall_s=time.perf_counter()-started,cpu_s=time.process_time()-cpu,script_sha256=audit.digest(Path(__file__)),
        validation_scope='All saved selected service assignments checked same-arm F&A and independent scalar proxy, ordered coverage/poses, navigation endpoints/cost/polyline/inflated-AABB collisions and full J/L/K; raw input tables hashed and fixed A reconstructed. No exhaustive all-pair F regeneration, IK, swept arm or joint trajectory certificate.',
        timing_scope='Only read-only audit body after imports; separate from charged scientific generation and not a deployment measurement.')
    if audit.rows:
        result['quality_descriptive']=dict(cases=len(audit.rows),native_J_mean=statistics.mean(r['native_J'] for r in audit.rows),
            teacher_J_mean=statistics.mean(r['teacher_J'] for r in audit.rows),teacher_L_mean=statistics.mean(r['teacher_L'] for r in audit.rows),
            teacher_K_mean=statistics.mean(r['teacher_K'] for r in audit.rows),feasible_cases=sum(r['valid'] for r in audit.rows),
            H_strict_improvement_cases=sum(r['teacher_J']<r['native_J']-1e-7 for r in audit.rows),
            interpretation='Teacher outputs on preregistered training inputs only; no neural/DRL performance or deployment claim.')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        old=read(output)
        result['audit_run_history']=old.get('audit_run_history',[])+[{k:old.get(k) for k in ('status','wall_s','cpu_s','script_sha256')}]
    write(output,result)
    print(json.dumps({k:result.get(k) for k in ('status','scientific_stage_complete','S1_B_technical_decision','counts','wall_s','error')},ensure_ascii=True))
    if result['status']=='AUDIT_FAILED':raise SystemExit(1)


if __name__=='__main__':main()
