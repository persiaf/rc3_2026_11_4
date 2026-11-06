"""Zero-query production SceneProvider/TeacherPool replay of saved S1-A data.

This is a source-bound engineering verifier, not a formal science permit or a
training entry point. It never invokes the authority factory or creates user
approval/authorization. Saved scientific bytes are checked before and after.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
from unittest.mock import patch

import numpy as np

from rc3_global_local_trainer.approval import ExecutionPermit, digest, read
from rc3_global_local_trainer.data import SceneProvider, TrainingRegistry
from rc3_global_local_trainer.runner import TeacherPool, StageClock
from rc3_global_local.environment import Environment
from rc3_global_local.problem import array_hash, signature
from rc3_global_local.validation import validate
from rc3_s1a_results_20261106.audit_s1a import Auditor

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUN = ROOT/'rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01'
BASE = '0b715b1eedfd3107a1e5188c25c60fe7fde113f1'


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False)+'\n', encoding='utf8', newline='\n')


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


class ReplayVerification:
    def __init__(self, output):
        self.output = output
        self.counts = {}
        self.rows = []
        self.forbidden = {}
        self.production_ledger = {}
        self.assertions = 0
        self.events = []

    def need(self, condition, message):
        self.assertions += 1
        if not condition: raise AssertionError(message)

    @contextlib.contextmanager
    def zero_scientific_calls(self):
        import rc3_baseline.baseline as baseline
        import rc3_fast.scenes as scenes
        import rc3_fast.solver as solver
        import rc3_constructive.problem as constructive_problem
        import rc3_constructive.baselines as constructive
        import rc3_global_local.terminal as terminal
        import rc3_global_local_impl_review.endpoint_baselines as endpoints
        import rc3_global_local.policy as policy
        import rc3_global_local_trainer.runner as runner
        import rc3_global_local_trainer.engine as engine
        from rc3_v4 import runtime
        import torch
        entries = [(baseline, 'navigation'), (scenes, 'make_complex_scene'),
            (solver, 'vectorized_feasibility'), (constructive_problem, 'resample_tasks'),
            (constructive, 'postprocess'), (terminal, 'finish'),
            (endpoints, 'run_endpoint'), (runner, 'run_endpoint'),
            (runner, 'finish'), (runtime, 'run'),
            (policy.GlobalLocalPolicy, '__init__'), (torch.nn.Module, '__init__'),
            (engine, 'make_optimizer'), (torch.optim.Optimizer, '__init__'),
            (torch.optim.Adam, 'step')]
        with contextlib.ExitStack() as stack:
            for obj, name in entries:
                key = getattr(obj, '__name__', type(obj).__name__)+'.'+name
                self.forbidden[key] = 0
                def fail(*args, _key=key, **kwargs):
                    self.forbidden[_key] += 1
                    raise AssertionError('Forbidden scientific call during saved-data replay: '+_key)
                stack.enter_context(patch.object(obj, name, fail))
            yield

    def frozen_inputs(self):
        original = json.loads(subprocess.check_output(['git', 'show', BASE+':project_management/GIT_MANIFEST.json'], cwd=ROOT))
        prefix = RUN.relative_to(ROOT).as_posix()+'/'
        expected = {p: v for p,v in original['files'].items() if p.startswith(prefix)}
        actual = {p.relative_to(ROOT).as_posix() for p in RUN.rglob('*') if p.is_file()}
        self.need(set(expected) == actual, 'Original S1-A artifact file set changed')
        failures = []
        for path, meta in expected.items():
            f = ROOT/path
            if f.stat().st_size != meta['bytes'] or hash_file(f) != meta['sha256']:
                failures.append(path)
        self.need(not failures, 'S1-A original bytes changed: '+str(failures))
        return dict(files=len(expected), bytes=sum(v['bytes'] for v in expected.values()),
                    all_hashes_passed=True, base_commit=BASE)

    def permit(self, source_manifest):
        from rc3_global_local_execution_v3.authority import source_binding
        origin = read(RUN/'RUN_MANIFEST.json')['binding']
        binding = copy.deepcopy(origin)
        binding.update(source_sha256=source_binding(), engineering_fixture=True)
        if source_manifest:
            source_manifest = Path(source_manifest).resolve()
            current = read(source_manifest)
            self.need(current['source_sha256'] == binding['source_sha256'], 'Updated source binding is stale')
            binding['source_manifest_sha256'] = digest(source_manifest)
        # Explicit original upstream source, not a bypass of source checks.
        manifest = HERE/'LEGACY_TEACHER_REPLAY_MANIFEST.json'
        binding['legacy_teacher_replay'] = dict(science_binding=origin,
            manifest_path=manifest.relative_to(ROOT).as_posix(), manifest_sha256=digest(manifest))
        permit = ExecutionPermit(ROOT/'rc3_global_local_execution_v3/V3_EXECUTION_CANDIDATE.json',
            ROOT/'rc3_global_local_execution_v3/REGISTRY.json', origin['protocol_sha256'],
            origin['registry_sha256'], binding, (), 'ENGINEERING_NO_SCIENCE_APPROVAL',
            'ENGINEERING_NO_STAGE_AUTHORIZATION')
        return permit

    def reference(self):
        a = Auditor(RUN)
        a.binding = read(RUN/'RUN_MANIFEST.json')['binding']
        gate = read(ROOT/'rc3_global_local_impl_review/GEOMETRY_GATE.json')
        a.map_keys = {(r['map_seed'], r['mode']): r['map_key'] for r in gate['records'] if r['role']=='train'}
        return a

    def compare_view(self, actual, expected):
        self.need(actual.parent.identity == expected.parent.identity, 'Independent parent identity mismatch')
        self.need(actual.identity == expected.identity, 'Independent local identity mismatch')
        self.need(signature(actual.parent.scene) == signature(expected.parent.scene), 'Independent scene mismatch')
        for obj, ref, names in [(actual.parent, expected.parent, ('F','A','effective','D')),
                (actual, expected, ('task_ids','candidate_ids','F','C','D'))]:
            for name in names:
                self.need(np.array_equal(getattr(obj,name), getattr(ref,name)), 'Array mismatch: '+name)
        self.need(actual.parent.navigation_binding.content_sha256 == expected.parent.navigation_binding.content_sha256,
            'Navigation content identity mismatch')
        self.need(actual.description() == expected.description(), 'View description/source/endpoint mismatch')
        self.need(actual.parent.navigation_binding.source_sha256 == expected.parent.navigation_binding.source_sha256,
                  'Navigation producer source mismatch')

    def label_states(self, view, row):
        actions = row['actions']; hr = row['H_result']
        self.need(actions == hr['route'][1:-1], 'Supervision actions differ from frozen H result')
        metrics = view.metrics([0, *actions, 0])
        self.need(abs(metrics['J']-row['teacher_J']) <= 1e-7, 'Supervision terminal J mismatch')
        for key in ('J','L','K'):
            self.need(abs(metrics[key]-hr[key]) <= 1e-7, 'Frozen final '+key+' mismatch')
        env = Environment(view, lambda p, raw: p.metrics(raw))
        independent_remaining = np.ones(len(view.C), dtype=bool)
        state_hashes = []
        for node in actions:
            self.need(not env.done, 'Labels continue past full coverage')
            expected_mask = view.C[independent_remaining].any(0) & np.isfinite(view.D[env.current,1:]) & np.isfinite(view.D[1:,0])
            mask = env.mask()
            self.need(np.array_equal(mask, expected_mask), 'Production legal mask differs from independent coverage/navigation mask')
            self.need(type(node) is int and 1 <= node <= len(mask) and bool(mask[node-1]), 'Illegal teacher action')
            served = view.task_ids[np.flatnonzero(independent_remaining & view.C[:,node-1])].tolist()
            state_hashes.append(signature(dict(remaining=view.task_ids[independent_remaining].tolist(),
                legal_local_nodes=(np.flatnonzero(mask)+1).tolist(), action=node, served_tasks=served)))
            event = env.step(node)
            self.need(event['served_tasks_global'] == served and bool(served), 'Supervision served-task transition differs')
            independent_remaining &= ~view.C[:,node-1]
            self.need(np.array_equal(env.remaining, independent_remaining), 'Remaining task transition mismatch')
        self.need(env.done and not independent_remaining.any(), 'Teacher route does not cover all tasks')
        self.need(abs(sum(env.rewards)+metrics['J']) <= 1e-7, 'No-H label replay reward sum differs from raw final route cost')
        return metrics, dict(decisions=len(actions), state_trace_sha256=signature(state_hashes), all_legal=True,
                             coverage_complete=True, added_teacher_H_queries=0)

    def run(self, source_manifest=None):
        started = time.perf_counter()
        before = self.frozen_inputs()
        permit = self.permit(source_manifest)
        registry = TrainingRegistry(permit.registry_path)
        provider = SceneProvider(registry, RUN/'data', self.production_ledger, permit, record=self.events.append, read_only=True)
        reference = self.reference()
        pool = TeacherPool(RUN/'teachers256', registry, provider, self.production_ledger, StageClock(1800), permit.binding['H_sha256'])
        declared = list(registry.teacher_specs())
        index = read(RUN/'teachers256/POOL_INDEX.json')
        with self.zero_scientific_calls():
            pool.validate()
            for i, spec in enumerate(declared):
                label_path = RUN/'teachers256'/f'label_{i:03}.json'
                row = read(label_path)
                self.need(row['spec'] == spec and row['index'] == i, 'Registry teacher order mismatch')
                self.need(digest(label_path) == index['labels'][i]['sha256'], 'Label bytes differ from original index')
                # Production path used by real S1 training, not a JSON-only surrogate.
                example = pool.example(i)
                view = example['view']
                expected = reference.view(spec)
                self.compare_view(view, expected)
                task_folder = RUN/'data/task_draws'/f"{spec['map_seed']}_{spec['task_seed']}_A{spec['A_condition']}"
                receipt = read(task_folder/'receipt.json')
                self.need(view.parent.identity == receipt['parent_identity'], 'Original parent receipt mismatch')
                self.need(view.identity == row['view_identity'], 'Original teacher view identity mismatch')
                self.need(example['actions'] == row['actions'] and example['teacher_J'] == row['teacher_J'], 'Production labels differ')
                metrics, label_check = self.label_states(view, row)
                plan_checks = []
                for name, plan in [('native',row['native_plan']),('H',row['H_result']['plan'])]:
                    certificate = validate(view, plan)
                    self.need(certificate['valid_model_plan'] and certificate['all_tasks_served'], name+' saved plan fails scalar validation')
                    inverse = {int(g): j+1 for j,g in enumerate(view.candidate_ids)}
                    route = [0, *[inverse[g] for g in plan['visit_order_candidate_ids']], 0]
                    m = view.metrics(route)
                    self.need(abs(m['J']-plan['objective_length_plus_stops_m']) <= 1e-7, name+' cost changed')
                    plan_checks.append(dict(name=name, valid=True, J=m['J'], L=m['L'], K=m['K']))
                nav = view.parent.navigation_binding
                self.rows.append(dict(index=i, spec=spec, original_parent_identity=receipt['parent_identity'],
                    loaded_parent_identity=view.parent.identity, original_view_identity=row['view_identity'],
                    loaded_view_identity=view.identity, scene_sha256=digest(task_folder/'scene.json'),
                    task_arrays_sha256=digest(task_folder/'arrays.npz'), F_sha256=array_hash(view.parent.F),
                    A_sha256=array_hash(view.parent.A), parent_D_sha256=array_hash(view.parent.D),
                    local_F_sha256=array_hash(view.F), local_C_sha256=array_hash(view.C), local_D_sha256=array_hash(view.D),
                    candidate_ids_sha256=array_hash(view.candidate_ids), task_ids_sha256=array_hash(view.task_ids),
                    navigation_content_sha256=nav.content_sha256, navigation_artifact_sha256=nav.provenance['artifact_sha256'],
                    navigation_source_sha256=dict(nav.source_sha256), original_task_source_sha256=receipt['source_sha256'],
                    navigation_schema_variant='full_with_recovered_rejected_edge_count' if 'rejected_edge_count' in nav.nav else 'legacy_reduced_original_identity',
                    rejected_edge_count=nav.nav.get('rejected_edge_count'), label_sha256=digest(label_path),
                    actions=row['actions'], label_state_checks=label_check, teacher_J=metrics['J'], teacher_L=metrics['L'],
                    teacher_K=metrics['K'], saved_plan_checks=plan_checks, passed=True))
                if (i+1)%32 == 0: print(json.dumps(dict(replayed=i+1,total=256)), flush=True)
        after = self.frozen_inputs()
        self.need(before == after, 'S1-A evidence differs before/after replay')
        self.need(len(self.rows) == 256, 'Not all teacher instances replayed')
        self.need(not any(self.forbidden.values()), 'Forbidden scientific function executed')
        for key in ('task_scene_generation_attempts','task_scene_generations','proxy_feasibility_calls',
                'navigation_attempts','navigation_completed','native_teacher_attempts','optimizer_updates'):
            self.need(self.production_ledger.get(key,0) == 0, 'Scientific generation counter nonzero: '+key)
        full = sum(r['navigation_schema_variant'].startswith('full_') for r in self.rows)
        self.need(full == 96 and len(self.rows)-full == 160, 'Original96/160 navigation identity split changed')
        result = dict(schema='rc3-production-teacher-replay-verification-v1', status='PASS',
            created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), verifier_sha256=digest(Path(__file__)),
            production_source_sha256=permit.binding['source_sha256'], binding_scope='Engineering fixture; stages=(); no user approval or scientific execution authority created',
            source_manifest_sha256=digest(source_manifest) if source_manifest else None,
            scientific_run=RUN.relative_to(ROOT).as_posix(), original_source_binding_sha256=read(RUN/'RUN_MANIFEST.json')['binding']['source_manifest_sha256'],
            teachers_replayed=256, original_full_schema_restored=full, original_reduced_schema_preserved=256-full,
            saved_plans_revalidated=512, assertions=self.assertions, production_ledger=self.production_ledger,
            forbidden_scientific_call_counters=self.forbidden, source_generation_navigation_native_H_model_optimizer=0,
            original_artifacts_before=before, original_artifacts_after=after, original_bytes_preserved=True,
            independent_reference_counts=reference.counts, wall_s=time.perf_counter()-started, teacher_rows=self.rows)
        write(self.output,result)
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-binding')
    parser.add_argument('--output',type=Path,default=HERE/'TEACHER_REPLAY_VERIFICATION.json')
    args=parser.parse_args(); verifier=ReplayVerification(args.output); started=time.perf_counter()
    try:
        result=verifier.run(args.source_binding)
        print(json.dumps({k:result[k] for k in ('status','teachers_replayed','saved_plans_revalidated','wall_s')},ensure_ascii=False))
    except Exception as exc:
        write(args.output.with_name(args.output.stem+'_FAILED.json'),dict(status='FAIL',error=type(exc).__name__,
            message=str(exc),traceback=traceback.format_exc(),wall_s=time.perf_counter()-started,
            assertions=verifier.assertions,completed_teachers=len(verifier.rows),forbidden_calls=verifier.forbidden,
            production_ledger=verifier.production_ledger,teacher_rows=verifier.rows))
        raise


if __name__=='__main__': main()
