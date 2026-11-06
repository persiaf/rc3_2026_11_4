"""Zero-science production stage/source bridge checks; grants no authority."""
from contextlib import ExitStack
import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

from rc3_global_local_execution_v3 import authority as auth
from rc3_global_local_execution_v3 import runner as adapter
from rc3_global_local_trainer import runner as oldrunner
from rc3_global_local_trainer.approval import ExecutionPermit, teacher_source_binding
from rc3_global_local_trainer.data import TrainingRegistry

ROOT = auth.ROOT
HERE = ROOT/'rc3_navigation_compat_20261106'
COUNT = 0
ROWS = []
FORBIDDEN = {}


def check(value, label):
    global COUNT
    COUNT += 1
    if not value:
        raise AssertionError(label)


def rejects(fn, label):
    global COUNT
    COUNT += 1
    try:
        fn()
    except (auth.AdmissionError, ValueError, RuntimeError, FileNotFoundError) as exc:
        ROWS.append(dict(case=label, result='EXPECTED_REJECTION', error=type(exc).__name__, message=str(exc)))
        return
    raise AssertionError('Accepted invalid bridge request: '+label)


def banned(label):
    def fail(*args, **kwargs):
        FORBIDDEN[label] = FORBIDDEN.get(label, 0)+1
        raise AssertionError('Forbidden science operation: '+label)
    return fail


def run(output_path=None):
    import torch
    from rc3_global_local import policy, terminal
    from rc3_fast import scenes
    from rc3_constructive import problem
    from rc3_baseline import baseline

    target = Path(output_path or HERE/'STAGE_BRIDGE_VERIFICATION.json').resolve()
    if not target.is_relative_to(HERE) or target.exists():
        raise RuntimeError('New verification output must be inside this study and cannot overwrite history')
    started = time.perf_counter()
    mp = HERE/'LEGACY_TEACHER_REPLAY_MANIFEST.json'
    m = auth.read(mp)
    origin = m['science_binding']
    replay = dict(manifest_path=mp.relative_to(ROOT).as_posix(),
                  manifest_sha256=auth.digest(mp), science_binding=copy.deepcopy(origin))
    current = auth.science_binding(origin['protocol_sha256'], origin['registry_sha256'],
        auth.source_binding(), origin['H_sha256'], origin['experiment_id'],
        'ENGINEERING_NO_USER_APPROVAL', 'ENGINEERING_NO_APPROVED_SOURCE_MANIFEST')
    current['engineering_fixture'] = True
    current['legacy_teacher_replay'] = replay
    folder = ROOT/m['run']
    counters = dict(model_initializations=0, optimizer_updates=0,
        new_scene_generation=0, navigation_solves=0, native_Detour16_calls=0, H_calls=0)
    out = dict(schema='rc3-source-stage-bridge-regression-v1', status='RUNNING',
        engineering_fixture_only=True, grants_no_authorization=True,
        source_sha256=copy.deepcopy(current['source_sha256']),
        checker_sha256=auth.digest(Path(__file__)),
        legacy_manifest_sha256=auth.digest(mp), scientific_calls=counters)
    with ExitStack() as stack:
        for obj, name, label in [
            (policy.GlobalLocalPolicy, '__init__', 'model_initializations'),
            (policy.GlobalLocalPolicy, 'encode', 'model_forward'),
            (policy.GlobalLocalPolicy, 'score', 'model_forward'),
            (torch.optim.Adam, 'step', 'optimizer_updates'),
            (torch.autograd, 'backward', 'backward'),
            (oldrunner, 'run_endpoint', 'native_Detour16_calls'),
            (oldrunner, 'finish', 'H_calls'),
            (terminal, 'finish', 'H_calls'),
            (scenes, 'make_complex_scene', 'new_scene_generation'),
            (problem, 'resample_tasks', 'new_scene_generation'),
            (baseline, 'navigation', 'navigation_solves')]:
            stack.enter_context(patch.object(obj, name, banned(label)))
        check(auth.validate_legacy_teacher_replay(replay, current) == replay,
              'Exact historical teacher provenance must validate')
        check(teacher_source_binding(current) == origin, 'Old teacher source is explicit')
        check(current['source_sha256'] != origin['source_sha256'], 'Current executable source remains distinct')
        old_complete, old_hash = auth.StageChain(folder, origin).completed('S1-A')
        new_complete, new_hash = auth.StageChain(folder, current).completed('S1-A')
        check(old_hash == new_hash == m['stage_complete_sha256'], 'Completed teacher stage retains original hash')
        check(old_complete == new_complete, 'Historical stage record remains unchanged')
        check(new_complete['science_binding'] == origin, 'Historical record not rebound to new code')
        artifacts = adapter.teacher_artifacts(folder, current)
        check(len(artifacts) > 256, 'Production stage gate checks complete saved pool artifacts')
        provider = SimpleNamespace(permit=SimpleNamespace(binding=current))
        registry = TrainingRegistry(auth.REGISTRY)
        pool = oldrunner.TeacherPool(folder/'teachers256', registry, provider, {},
                                    oldrunner.StageClock(120), origin['H_sha256'])
        check(pool.validate()['count'] == 256, 'Production supervised pool accepts exact upstream teacher binding')

        for key in ('protocol_sha256', 'registry_sha256', 'H_sha256', 'experiment_id'):
            wrong = copy.deepcopy(current)
            wrong[key] = 'ENGINEERING_WRONG'
            rejects(lambda wrong=wrong:auth.validate_legacy_teacher_replay(replay, wrong),
                    'Wrong current '+key)
        wrong = copy.deepcopy(replay); wrong['manifest_sha256'] = '0'*64
        rejects(lambda:auth.validate_legacy_teacher_replay(wrong, current), 'Tampered compatibility manifest')
        wrong = copy.deepcopy(replay); wrong['science_binding']['approval_sha256'] = '0'*64
        rejects(lambda:auth.validate_legacy_teacher_replay(wrong, current), 'Wrong historical user approval')
        wrong = copy.deepcopy(current); wrong['legacy_teacher_replay']['science_binding']['source_sha256'] = {}
        bad_provider = SimpleNamespace(permit=SimpleNamespace(binding=wrong))
        bad_pool = oldrunner.TeacherPool(folder/'teachers256', registry, bad_provider, {},
                                        oldrunner.StageClock(120), origin['H_sha256'])
        rejects(bad_pool.validate, 'Supervised pool refuses substituted historical source')
        rejects(lambda:auth.StageChain(folder, dict(current, legacy_teacher_replay=None)).completed('S1-A'),
                'No blanket scientific identity substitution')
        rejects(lambda:auth.StageChain(folder, current).admit('S1-A',None,'ENGINEERING',{}),
                'Completed S1-A cannot be repeated')
        for seed in (999, True):
            rejects(lambda seed=seed:auth.require_stage_authority(auth.PROTOCOL,auth.REGISTRY,
                None,None,'S1-B',seed,origin['experiment_id']), 'Wrong fixed optimizer seed '+str(seed))
        rejects(lambda:auth.require_stage_authority(auth.PROTOCOL,auth.REGISTRY,None,None,
            'S1-B',auth.SEEDS[0],origin['experiment_id']), 'Missing new source approval and S1-B authorization')
        rejects(lambda:auth.require_stage_authority(auth.PROTOCOL,auth.REGISTRY,
            ROOT/m['protocol_approval_path'],
            ROOT/'rc3_global_local_execution_v3/authorizations/20261106_S1_A/S1_A_AUTHORIZATION.json',
            'S1-B',auth.SEEDS[0],origin['experiment_id']), 'Old S1-A receipts cannot authorize new S1-B code')
        check(adapter.main(['--stage','S1-B','--seed',str(auth.SEEDS[0]),
            '--experiment-id',origin['experiment_id'],'--output',str(folder),
            '--existing-run','--source-binding',str(HERE/'SOURCE_BINDING_UPDATED.json')]) == 2,
            'Production CLI rejects missing authority before models or writes')

        # Lower-level constructor fixtures prove that the immutable teacher run
        # manifest is preserved while new checkpoints use a new source identity.
        # This is not a StagePermit and carries no executable stage scope.
        with tempfile.TemporaryDirectory(prefix='bridge_bytes_', dir=HERE) as temp:
            f = Path(temp)
            check(f.resolve().is_relative_to(HERE.resolve()), 'Byte fixture cleanup is limited to this study')
            original = (folder/'RUN_MANIFEST.json').read_bytes()
            (f/'RUN_MANIFEST.json').write_bytes(original)
            permit = ExecutionPermit(auth.PROTOCOL,auth.REGISTRY,origin['protocol_sha256'],
                origin['registry_sha256'],current,(), 'ENGINEERING_NO_APPROVAL','ENGINEERING_NO_AUTHORIZATION')
            created = adapter.StageTrainingRun(permit, f)
            check((f/'RUN_MANIFEST.json').read_bytes() == original, 'Historical run manifest stays byte-identical')
            check(created.permit.binding == current, 'New scientific checkpoint config keeps new current source')
            check(created._config('S1',auth.SEEDS[0])['binding']['source_sha256'] == current['source_sha256'],
                  'Checkpoint source differs from teacher artifact source')
            rejects(lambda:created.s1(auth.SEEDS[0]), 'Engineering constructor has no S1 permission')
            # Reopening is stable; it does not create fresh stage resources.
            reopened = adapter.StageTrainingRun(permit, f)
            check(reopened.permit.binding == current, 'Same source admission can reopen stably')
            check(len(list(f.glob('SOURCE_REVISION_*.json'))) == 1, 'Exactly one immutable source-revision record')
        check(not FORBIDDEN, 'No model, optimizer, teacher, H, scene or navigation operation')
    out.update(status='PASS', assertions=COUNT, expected_rejections=ROWS,
        forbidden_operation_attempts=FORBIDDEN, wall_s=time.perf_counter()-started,
        scope='Existing saved teacher provenance and isolated non-executable byte fixtures only; no scientific run or new authorization.')
    target.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(dict(status=out['status'],assertions=COUNT,expected_rejections=len(ROWS),
                         forbidden_operation_attempts=FORBIDDEN,wall_s=out['wall_s'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    run(parser.parse_args().output)
