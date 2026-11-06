"""Saved-array navigation schema and strict-rejection regression; zero queries."""
from __future__ import annotations

import contextlib
import copy
import datetime
import json
from pathlib import Path
import shutil
import tempfile
import time
import traceback
from unittest.mock import patch

import numpy as np

from rc3_global_local_trainer.approval import digest, read
from rc3_global_local_impl_review.runtime_guards import snapshot_parent
from .verify_teacher_replay import ROOT, HERE, RUN, ReplayVerification, write


def run_tests():
    from rc3_global_local_trainer.navigation_compat import (
        load_navigation_variants, save_navigation_artifact, select_navigation, LegacyTeacherReplay)
    started=time.perf_counter(); checker=ReplayVerification(HERE/'UNIT_UNUSED.json')
    rows=[]; temporary_bytes=0
    def check(name, fn, *, reject=False):
        before=time.perf_counter()
        try:
            result=fn()
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            if not reject: raise
            rows.append(dict(name=name,status='PASS_REJECTED',error=type(exc).__name__,message=str(exc),wall_s=time.perf_counter()-before))
        else:
            if reject: raise AssertionError('Negative fixture was accepted: '+name)
            rows.append(dict(name=name,status='PASS',wall_s=time.perf_counter()-before))
            return result

    ref=checker.reference(); specs=list(__import__('rc3_global_local_trainer.data',fromlist=['TrainingRegistry']).TrainingRegistry(
        ROOT/'rc3_global_local_execution_v3/REGISTRY.json').teacher_specs())
    with checker.zero_scientific_calls():
        source=ref.view(specs[0]); scene=source.parent.scene
        nav=source.parent.navigation_binding.nav
        original_receipt=read(RUN/'data/task_draws'/f"{specs[0]['map_seed']}_{specs[0]['task_seed']}_A0"/'receipt.json')
        key=original_receipt['map_key']; original_path=RUN/'data/navigation'/(key+'.npz')
        tmp_parent=HERE/'unit_temp'; tmp_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='navigation_schema_',dir=tmp_parent) as directory:
            folder=Path(directory).resolve(); checker.need(folder.is_relative_to(HERE.resolve()),'Temporary fixture path escapes task folder')
            new=folder/'new'/(key+'.npz'); new.parent.mkdir()
            check('new_schema_complete_save',lambda:save_navigation_artifact(scene,nav,new,0.0))
            check('new_schema_boolean_wall_rejected_before_save',lambda:save_navigation_artifact(scene,nav,folder/'bad_bool_wall.npz',True),reject=True)
            checker.need(not (folder/'bad_bool_wall.npz').exists(),'Invalid wall metadata produced a partial navigation artifact')
            new_meta=read(new.with_suffix('.json'))
            checker.need(new_meta['schema']=='rc3-navigation-artifact-v2','New navigation schema version missing')
            with np.load(new,allow_pickle=False) as a:
                checker.need(set(a.files)=={'costs','coordinates','service_nodes','predecessors','node_count','edge_count','rejected_edge_count'},'New schema omits/extends identity fields')
            variants=check('new_schema_load',lambda:load_navigation_variants(scene,new))
            checker.need(len(variants)==1 and variants[0][1].content_sha256==source.parent.navigation_binding.content_sha256,'New schema content differs from original full navigation')
            parent=snapshot_parent(scene,source.parent.F,source.parent.A,navigation=variants[0][1])
            selected=check('new_schema_select_exact_parent',lambda:select_navigation(variants,scene,source.parent.F,source.parent.A,parent.identity))
            checker.need(selected.content_sha256==variants[0][1].content_sha256,'Selected full content mismatch')

            old=folder/'old'/(key+'.npz'); old.parent.mkdir(); shutil.copyfile(original_path,old); shutil.copyfile(original_path.with_suffix('.json'),old.with_suffix('.json'))
            variants_old=check('old_schema_two_identity_variants',lambda:load_navigation_variants(scene,old))
            checker.need(len(variants_old)==2,'Legacy navigation must retain both exact historical variants')
            full=check('old_schema_restore_original96_parent',lambda:select_navigation(variants_old,scene,source.parent.F,source.parent.A,source.parent.identity))
            checker.need('rejected_edge_count' in full.nav,'Original full identity not restored')
            reduced=[v for _,v in variants_old if 'rejected_edge_count' not in v.nav][0]
            reduced_parent=snapshot_parent(scene,source.parent.F,source.parent.A,navigation=reduced)
            selected=check('old_schema_preserve_reduced160_identity',lambda:select_navigation(variants_old,scene,source.parent.F,source.parent.A,reduced_parent.identity))
            checker.need('rejected_edge_count' not in selected.nav,'Reduced identity was silently upgraded')
            check('old_schema_wrong_parent_identity',lambda:select_navigation(variants_old,scene,source.parent.F,source.parent.A,'0'*64),reject=True)
            check('duplicate_matching_parent_identity_conflict',lambda:select_navigation([variants_old[0],variants_old[0]],scene,source.parent.F,source.parent.A,
                snapshot_parent(scene,source.parent.F,source.parent.A,navigation=variants_old[0][1]).identity),reject=True)

            def corrupt_meta(name, mutate):
                d=folder/name; d.mkdir(); target=d/(key+'.npz')
                shutil.copyfile(new,target); meta=copy.deepcopy(new_meta); mutate(meta); write(target.with_suffix('.json'),meta)
                return target
            p=corrupt_meta('wrong_map',lambda m:m.update(map_key='0'*64))
            check('new_schema_wrong_geometry_receipt',lambda:load_navigation_variants(scene,p),reject=True)
            p=corrupt_meta('wrong_source',lambda m:m['source_sha256'].update({'rc3_baseline/baseline.py':'0'*64}))
            check('new_schema_wrong_navigation_source',lambda:load_navigation_variants(scene,p),reject=True)
            p=corrupt_meta('wrong_hash',lambda m:m.update(sha256='0'*64))
            check('new_schema_tampered_file_hash',lambda:load_navigation_variants(scene,p),reject=True)
            p=corrupt_meta('wrong_content',lambda m:m.update(navigation_content_sha256='0'*64))
            check('new_schema_wrong_content_identity',lambda:load_navigation_variants(scene,p),reject=True)
            p=corrupt_meta('bad_schema',lambda m:m.update(schema='rc3-navigation-artifact-v999'))
            check('unknown_schema_refused',lambda:load_navigation_variants(scene,p),reject=True)

            def corrupt_array(name, mutate):
                d=folder/name; d.mkdir(); target=d/(key+'.npz')
                with np.load(new,allow_pickle=False) as a: values={k:a[k] for k in a.files}
                mutate(values)
                with target.open('xb') as f: np.savez_compressed(f,**values)
                meta=copy.deepcopy(new_meta); meta['sha256']=digest(target); write(target.with_suffix('.json'),meta)
                return target
            p=corrupt_array('missing_rejected',lambda a:a.pop('rejected_edge_count'))
            check('new_schema_missing_rejected_field',lambda:load_navigation_variants(scene,p),reject=True)
            p=corrupt_array('illegal_rejected',lambda a:a.update(rejected_edge_count=np.asarray(-1,dtype=np.int64)))
            check('new_schema_negative_rejected_field',lambda:load_navigation_variants(scene,p),reject=True)
            p=corrupt_array('bool_rejected',lambda a:a.update(rejected_edge_count=np.asarray(True)))
            check('new_schema_boolean_rejected_field',lambda:load_navigation_variants(scene,p),reject=True)
            p=corrupt_array('extra_field',lambda a:a.update(unknown_field=np.asarray(1)))
            check('new_schema_unknown_identity_field',lambda:load_navigation_variants(scene,p),reject=True)
            d=folder/'truncated'; d.mkdir(); p=d/(key+'.npz'); p.write_bytes(b'not a valid zip file'); meta=copy.deepcopy(new_meta); meta['sha256']=digest(p); write(p.with_suffix('.json'),meta)
            check('new_schema_invalid_npz_bytes',lambda:load_navigation_variants(scene,p),reject=True)
            binding=checker.permit(None).binding
            check('legacy_manifest_exact_original_anchors',lambda:LegacyTeacherReplay(binding))
            bad_binding=copy.deepcopy(binding); bad_binding['legacy_teacher_replay']['manifest_sha256']='0'*64
            check('legacy_manifest_bad_pinned_hash',lambda:LegacyTeacherReplay(bad_binding),reject=True)
            bad_binding=copy.deepcopy(binding); bad_binding['legacy_teacher_replay']['science_binding']['source_sha256']['rc3_global_local_trainer/data.py']='0'*64
            check('legacy_manifest_wrong_original_source_binding',lambda:LegacyTeacherReplay(bad_binding),reject=True)
            original_manifest=read(HERE/'LEGACY_TEACHER_REPLAY_MANIFEST.json')
            def altered_manifest(name,mutate):
                target=folder/(name+'.json'); value=copy.deepcopy(original_manifest); mutate(value); write(target,value)
                b=copy.deepcopy(binding); b['legacy_teacher_replay']['manifest_path']=target.relative_to(ROOT).as_posix(); b['legacy_teacher_replay']['manifest_sha256']=digest(target)
                return b
            b=altered_manifest('bad_stage_anchor',lambda m:m.update(stage_complete_sha256='0'*64))
            check('legacy_manifest_wrong_complete_anchor',lambda:LegacyTeacherReplay(b),reject=True)
            b=altered_manifest('duplicate_task',lambda m:m['records'][1].update(task_receipt_path=m['records'][0]['task_receipt_path']))
            check('legacy_manifest_duplicate_task_identity',lambda:LegacyTeacherReplay(b),reject=True)
            b=altered_manifest('bad_parent_identity',lambda m:m['records'][0].update(parent_identity='0'*64))
            task_folder=RUN/'data/task_draws'/f"{specs[0]['map_seed']}_{specs[0]['task_seed']}_A0"
            check('legacy_manifest_parent_conflicts_original_receipt',lambda:LegacyTeacherReplay(b).check_task(task_folder,original_receipt),reject=True)
            b=altered_manifest('bad_label_hash',lambda m:m['records'][0].update(label_sha256='0'*64))
            check('legacy_manifest_label_hash_conflicts_saved_file',lambda:LegacyTeacherReplay(b).check_task(task_folder,original_receipt),reject=True)
            wrong_receipt=copy.deepcopy(original_receipt); wrong_receipt['source_sha256']['rc3_global_local_trainer/data.py']='0'*64
            check('legacy_task_wrong_source_receipt',lambda:LegacyTeacherReplay(binding).check_task(task_folder,wrong_receipt),reject=True)
            from rc3_global_local_trainer.data import TrainingRegistry, SceneProvider
            from rc3_global_local_trainer.runner import TeacherPool, StageClock
            import rc3_global_local_trainer.runner as runner
            permit=checker.permit(None); registry=TrainingRegistry(permit.registry_path); ledger={}
            provider=SceneProvider(registry,RUN/'data',ledger,permit,read_only=True)
            pool=TeacherPool(RUN/'teachers256',registry,provider,ledger,StageClock(1800),permit.binding['H_sha256'])
            pool.validate()
            original_digest=runner.digest
            def changed_index_digest(path):
                return '0'*64 if Path(path).resolve()==(RUN/'teachers256/POOL_INDEX.json').resolve() else original_digest(path)
            with patch.object(runner,'digest',changed_index_digest), patch.object(provider,'get',side_effect=AssertionError('Provider called before rejecting changed pool index')):
                check('production_example_rechecks_index_before_loading',lambda:pool.example(0),reject=True)
            temporary_bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        # Only per-test isolated copies were removed; original files never opened for writing.
        if not any(tmp_parent.iterdir()): tmp_parent.rmdir()
    checker.need(not any(checker.forbidden.values()),'Scientific entry executed during codec tests')
    result=dict(schema='rc3-navigation-compatibility-negative-tests-v1',status='PASS',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_script_sha256=digest(Path(__file__)),
        tests=len(rows),assertions=checker.assertions,results=rows,wall_s=time.perf_counter()-started,
        forbidden_scientific_call_counters=checker.forbidden,scientific_generations_and_queries=0,
        fixture_scope='Copies/re-serialization of saved arrays only, not generated navigation or tasks. Bad fixtures isolated in checked task-local temp directory, removed after tests.',
        temporary_fixture_bytes=temporary_bytes,original_artifacts_written=0)
    write(HERE/'NAVIGATION_COMPATIBILITY_TESTS.json',result); return result


if __name__=='__main__':
    started=time.perf_counter()
    try:
        result=run_tests(); print(json.dumps({k:result[k] for k in ('status','tests','wall_s')}))
    except Exception as exc:
        write(HERE/'NAVIGATION_COMPATIBILITY_TESTS_FAILED.json',dict(status='FAIL',error=type(exc).__name__,
            message=str(exc),traceback=traceback.format_exc(),wall_s=time.perf_counter()-started))
        raise
