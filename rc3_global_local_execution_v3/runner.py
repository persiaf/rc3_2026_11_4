"""Stage-specific authorizations over one stable scientific experiment lineage."""
import argparse
import copy
import math
from pathlib import Path
import signal
import time

import torch

from rc3_global_local_trainer.approval import AdmissionError, digest, read
from rc3_global_local_trainer.data import SceneProvider, TrainingRegistry, write_new
from rc3_global_local_trainer.runner import TrainingRun, StageClock, TeacherPool
from .authority import (ROOT, PROTOCOL, REGISTRY, SEEDS, StagePermit, StageChain,
    require_stage_authority, output_directory, stage_key)


class StageTrainingRun(TrainingRun):
    """Reuse frozen math; only admission overhead and cumulative clocks adapt."""
    def __init__(self, permit, folder, admission_overhead_s=0.):
        super().__init__(permit, folder)
        self.admission_overhead_s = float(admission_overhead_s)
        self._initial_offset_applied = False

    def _components(self, phase, seed, limit_s, clock=None, record_context=True):
        clock = clock or StageClock(limit_s)
        if not self._initial_offset_applied:
            clock.before_s += self.admission_overhead_s
            self._initial_offset_applied = True
            clock.check()
        return super()._components(phase,seed,limit_s,clock,record_context)

    def _resume(self, checkpoint, trainer, provider, clock, folder):
        super()._resume(checkpoint, trainer, provider, clock, folder)
        # The frozen loader restores prior stage time plus current loader time.
        # Preserve this invocation's earlier authorization/parent verification IO.
        clock.before_s += self.admission_overhead_s
        clock.check()

    def build_teachers(self):
        if 'teachers' not in self.permit.stages:
            raise AdmissionError('Teacher generation is not in this stage authorization')
        clock=StageClock(self.protocol['S1']['wall_s']['shared_teacher_stage'])
        clock.before_s += self.admission_overhead_s; clock.check(); ledger={}
        self.contexts['shared_teachers']=dict(clock=clock,ledger=ledger)
        provider=SceneProvider(self.registry,self.folder/'data',ledger,self.permit,
            record=lambda r:self._teacher_event(r))
        provider.check=clock.check
        pool=TeacherPool(self.folder/'teachers256',self.registry,provider,ledger,clock,self.permit.binding['H_sha256'])
        pool.build()
        return pool

    def _teacher_event(self,row):
        from rc3_global_local_trainer.data import journal
        journal(self.folder/'teacher_data_events.jsonl',row)


def checkpoint_artifacts(folder, seed, phase, cursor, binding):
    """Hash/role binding precheck; actual tensor validation remains frozen loader."""
    folder=Path(folder).resolve(); base=folder/f'seed_{seed}'/phase
    path=base/('untrained.pt' if cursor==0 else f'update_{cursor:04}.pt')
    side=path.with_suffix('.pt.json'); commit=path.with_suffix('.pt.commit.json')
    for f in (path,side,commit):
        if not f.is_file() or not f.resolve().is_relative_to(folder):
            raise AdmissionError('Required fixed endpoint checkpoint is absent from the exact formal run')
    r,c=read(side),read(commit)
    expected=dict(binding,optimizer_seed=seed,run=folder.relative_to(ROOT).as_posix(),phase=phase)
    if (r.get('schema')!='rc3-trainer-checkpoint-integrity-v1' or r.get('phase')!=phase
            or r.get('binding')!=expected or r.get('cursors',{}).get(phase)!=cursor
            or r.get('weight_version')!=cursor or r.get('sha256')!=digest(path)
            or r.get('bytes')!=path.stat().st_size):
        raise AdmissionError('Checkpoint protocol/seed/run/phase/weight lineage differs')
    if (c.get('schema')!='rc3-checkpoint-boundary-commit-v1' or c.get('phase')!=phase
            or c.get('checkpoint_sha256')!=digest(path) or c.get('sidecar_sha256')!=digest(side)
            or c.get('cursors')!=r['cursors']):
        raise AdmissionError('Checkpoint does not have a matching closed update boundary')
    return [path,side,commit]


def teacher_artifacts(folder,binding):
    folder=Path(folder).resolve();pool=folder/'teachers256';index=pool/'POOL_INDEX.json'
    if not index.is_file(): raise AdmissionError('Shared teacher pool is incomplete')
    r=read(index)
    labels=r.get('labels',[])
    if (r.get('schema')!='rc3-shared-teacher-pool-v1' or r.get('count')!=256
            or r.get('registry_sha256')!=binding['registry_sha256']
            or r.get('source_binding')!=binding or len(labels)!=256
            or [x.get('index') for x in labels]!=list(range(256))):
        raise AdmissionError('Teacher pool scientific identity or complete count differs')
    specs=list(TrainingRegistry(REGISTRY).teacher_specs())
    artifacts=[index]
    for i,row in enumerate(labels):
        if row.get('file')!=f'label_{i:03}.json': raise AdmissionError('Teacher index/file is invalid')
        f=pool/row['file']
        if not f.is_file() or digest(f)!=row.get('sha256'): raise AdmissionError('Teacher label bytes changed')
        label=read(f)
        if (label.get('index')!=i or label.get('registry_sha256')!=binding['registry_sha256']
                or label.get('spec')!=specs[i]):
            raise AdmissionError('Teacher label identity differs')
        native=label.get('native_plan',{});hr=label.get('H_result',{});hp=hr.get('plan',{})
        identity=label.get('view_identity');actions=label.get('actions');route=hr.get('route')
        if (not isinstance(identity,str) or not identity or native.get('problem_identity')!=identity
                or hp.get('problem_identity')!=identity
                or native.get('validation',{}).get('valid_model_plan') is not True
                or hp.get('validation',{}).get('valid_model_plan') is not True
                or native.get('status')!='complete' or hp.get('status')!='complete'
                or native.get('target_task_ids_global')!=hp.get('target_task_ids_global')
                or not isinstance(native.get('target_task_ids_global'),list)
                or not isinstance(actions,list) or any(type(n) is not int or n<1 for n in actions)
                or len(set(actions))!=len(actions) or route!=[0,*actions,0]):
            raise AdmissionError('Teacher complete routes, tasks, endpoint identity or validation is missing')
        values=[label.get('teacher_J'),hr.get('J'),hp.get('objective_length_plus_stops_m'),
                hr.get('raw_J'),native.get('objective_length_plus_stops_m')]
        if (any(type(x) not in (int,float) or not math.isfinite(x) for x in values)
                or abs(values[0]-values[1])>1e-7 or abs(values[0]-values[2])>1e-7
                or abs(values[3]-values[4])>1e-7 or values[0]>values[4]+1e-7):
            raise AdmissionError('Teacher native/H terminal objective records differ or are nonfinite')
        artifacts.append(f)
    ledger=r.get('ledger',{})
    terminal=ledger.get('terminal',{}).get('teacher_H',{})
    if (ledger.get('native_teacher_attempts')!=256 or ledger.get('native_teacher_completed')!=256
            or terminal.get('terminal_requests')!=256
            or terminal.get('terminal_attempts',257)>256 or terminal.get('terminal_failed',1)!=0
            or terminal.get('terminal_completed')!=terminal.get('terminal_attempts')
            or ledger['native_teacher_attempts']+terminal['terminal_attempts']>512):
        raise AdmissionError('Teacher pool paid-query counters are incomplete or exceed the fixed cap')
    for f in pool.rglob('*'):
        if f.is_file() and f not in artifacts: artifacts.append(f)
    return artifacts


class StageRun:
    def __init__(self, permit, folder, *, run_factory=StageTrainingRun):
        if not isinstance(permit,StagePermit): raise AdmissionError('A validated per-stage permit is required')
        self.permit,self.folder=permit,Path(folder).resolve()
        if self.folder!=(Path(__file__).resolve().parent/'runs'/permit.experiment_id).resolve():
            raise AdmissionError('Experiment run directory name differs from its stable identity')
        self.chain=StageChain(self.folder,permit.binding)
        self.run_factory=run_factory

    def _validate_upstream(self):
        stage,seed=self.permit.stage,self.permit.seed
        if stage!='S1-A': teacher_artifacts(self.folder,self.permit.binding)
        if stage=='S2':
            for s in SEEDS:
                checkpoint_artifacts(self.folder,s,'S1',256,self.permit.binding)
                final=self.folder/f'seed_{s}/S1/FINAL.json'
                if not final.is_file(): raise AdmissionError('All three fixed S1 endpoints must be complete before S2')
                r=read(final)
                side=read(self.folder/f'seed_{s}/S1/update_0256.pt.json')
                if (r.get('status')!='COMPLETE_FIXED_ENDPOINT' or r.get('optimizer_updates')!=256
                        or r.get('checkpoint')!='update_0256.pt' or r.get('tensor_sha256')!=side['model_tensor_sha256']):
                    raise AdmissionError('S1 final receipt does not match its fixed checkpoint')

    def _artifacts(self):
        stage,seed=self.permit.stage,self.permit.seed
        if stage=='S1-A': return teacher_artifacts(self.folder,self.permit.binding)
        if stage=='S1-B':
            values=checkpoint_artifacts(self.folder,seed,'S1',256,self.permit.binding)
            values += checkpoint_artifacts(self.folder,seed,'S1',0,self.permit.binding)
            values += [self.folder/f'seed_{seed}/S1/FINAL.json']
            final=read(values[-1]);side=read(values[1])
            if (final.get('status')!='COMPLETE_FIXED_ENDPOINT' or final.get('optimizer_updates')!=256
                    or final.get('tensor_sha256')!=side['model_tensor_sha256']):
                raise AdmissionError('S1 completion receipt differs from its fixed endpoint')
            return values
        values=checkpoint_artifacts(self.folder,seed,'RL',512,self.permit.binding)
        values+=checkpoint_artifacts(self.folder,seed,'continued_SL',512,self.permit.binding)
        fork=self.folder/f'seed_{seed}/S2/FORK_RECEIPT.json';final=self.folder/f'seed_{seed}/S2/FINAL.json'
        if not fork.is_file() or not final.is_file(): raise AdmissionError('S2 fixed endpoint/fork record is incomplete')
        f=read(fork);sl=self.folder/f'seed_{seed}/S1/update_0256.pt'
        slmeta=read(sl.with_suffix('.pt.json'))
        if f.get('source_S1_checkpoint_sha256')!=digest(sl) or f.get('common_tensor_sha256')!=slmeta['model_tensor_sha256']:
            raise AdmissionError('S2 forks do not originate from this exact approved S1 endpoint')
        if read(final).get('status')!='COMPLETE_FIXED_ENDPOINTS': raise AdmissionError('S2 final status is incomplete')
        return values+[fork,final]

    def _resources(self,run):
        result={}
        for key,v in run.contexts.items():
            v['clock'].check()
            result[key]=dict(ledger=copy.deepcopy(v['ledger']),elapsed_wall_s=v['clock'].elapsed_s,
                wall_cap_s=v['clock'].limit_s)
        return result

    def execute(self,*,resume=None,invocation_started=None):
        started=time.perf_counter() if invocation_started is None else invocation_started
        if self.permit.stage=='S1-A' and resume:
            raise AdmissionError('Partial teacher generation has no implicit resume/re-query')
        self._validate_upstream()
        admission=self.chain.admit(self.permit.stage,self.permit.seed,
            self.permit.authorization_sha256,self.permit.predecessor_hashes,resume=bool(resume))
        # Admit before any models, data preparation or charged query. Existing
        # failed/completed/switch-authorization stages cannot obtain a fresh run.
        run=None;previous=None
        try:
            run=self.run_factory(self.permit.execution,self.folder,time.perf_counter()-started)
            event=self.chain.path(admission['key'],'AUTHORIZATION_SOURCE.json')
            source=dict(authorization_sha256=self.permit.authorization_sha256,
                declaration=copy.deepcopy(self.permit.authorization))
            if event.exists():
                if read(event)!=source: raise AdmissionError('Authorization source record differs')
            else: write_new(event,source)
            previous=signal.signal(signal.SIGINT,lambda *_:setattr(run,'stop_requested',True))
            if hasattr(run,'admission_overhead_s'):
                run.admission_overhead_s=time.perf_counter()-started
            if self.permit.stage=='S1-A':
                if resume: raise AdmissionError('Partial teachers stop; no implicit query/wall-budget restart')
                run.build_teachers();complete=True
            elif self.permit.stage=='S1-B': complete=run.s1(self.permit.seed,resume)
            else: complete=run.s2(self.permit.seed,resume)
            resource_file=run.resources('COMPLETE' if complete else 'INTERRUPTED_SAFE_UPDATE_BOUNDARY')
            if complete:
                artifacts=self._artifacts()+[self.folder/resource_file]
                # Hashing and final lineage validation use the still-live stage
                # clock. The tiny terminal record cannot include its own write.
                row=self.chain.close(admission,artifacts,lambda:self._resources(run),complete=True)
                return dict(status='COMPLETE',stage_key=admission['key'],record=row)
            resources=self._resources(run)
            dest=self.chain.path(admission['key'],'INTERRUPTED_'+str(time.time_ns())+'.json')
            write_new(dest,dict(status='INTERRUPTED_SAFE_UPDATE_BOUNDARY',resources=resources,
                authorization_sha256=self.permit.authorization_sha256,
                resume_requires_same_authorization=True,no_budget_reset=True))
            return dict(status='INTERRUPTED_SAFE_UPDATE_BOUNDARY',stage_key=admission['key'])
        except Exception as exc:
            if run is None:
                self.chain.close(admission,[],dict(admission_wall_s=time.perf_counter()-started,science_calls=0),
                    complete=False,error=dict(type=type(exc).__name__,message=str(exc)))
                raise
            resource_file=run.resources('FAILED_INCOMPLETE',dict(type=type(exc).__name__,message=str(exc)))
            # Even an exhausted clock/failing source must retain partial spent
            # counters. Do not re-check a failed clock while recording failure.
            resources={k:dict(ledger=copy.deepcopy(v['ledger']),elapsed_wall_s=v['clock'].elapsed_s,
                wall_cap_s=v['clock'].limit_s) for k,v in run.contexts.items()}
            self.chain.close(admission,[self.folder/resource_file],resources,complete=False,
                error=dict(type=type(exc).__name__,message=str(exc)))
            raise
        finally:
            if previous is not None: signal.signal(signal.SIGINT,previous)


def main(argv=None):
    invocation_started=time.perf_counter()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('S1-A','S1-B','S2'))
    parser.add_argument('--seed',type=int);parser.add_argument('--experiment-id',required=True)
    parser.add_argument('--protocol',default=str(PROTOCOL));parser.add_argument('--registry',default=str(REGISTRY))
    parser.add_argument('--protocol-approval');parser.add_argument('--stage-authorization')
    parser.add_argument('--output',required=True);parser.add_argument('--existing-run',action='store_true')
    parser.add_argument('--resume')
    args=parser.parse_args(argv)
    try:
        permit=require_stage_authority(args.protocol,args.registry,args.protocol_approval,
            args.stage_authorization,args.stage,args.seed,args.experiment_id)
        folder=output_directory(args.output,args.experiment_id,existing=args.existing_run)
        if args.resume and not args.existing_run: raise AdmissionError('Resume must name an explicit existing run')
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            result=StageRun(permit,folder).execute(resume=args.resume,invocation_started=invocation_started)
        import json
        print(json.dumps(result,ensure_ascii=False))
        return 0
    except Exception as exc:
        import json
        print(json.dumps(dict(status='REJECTED_OR_INCOMPLETE',error=type(exc).__name__,message=str(exc),formal_success=False),ensure_ascii=False))
        return 2


if __name__=='__main__': raise SystemExit(main())
