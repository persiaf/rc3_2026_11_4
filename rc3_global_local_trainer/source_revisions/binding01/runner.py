"""Explicitly admitted S1/S2 orchestration. No import-time work or rule fallback."""
import argparse
from collections import OrderedDict
import copy
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from rc3_global_local.environment import Environment
from rc3_global_local.policy import GlobalLocalPolicy, tensor_hash
from rc3_global_local.problem import signature
from rc3_global_local.terminal import finish
from rc3_global_local_impl_review.endpoint_baselines import run_endpoint
from rc3_nav_value.r1.oracle import _empty_internal, _instrument
from .approval import (ROOT, V2, REGISTRY_V2, AdmissionError, digest, read,
                       require_authority, output_directory)
from .data import (TrainingRegistry, TrainingSchedule, SceneProvider, plain,
                   write_new, journal, add)
from .engine import Trainer, make_optimizer, fork_from_supervised
from .rng import NamedStreams
from .checkpoint import save_checkpoint, load_checkpoint


class StageClock:
    def __init__(self, limit_s):
        self.limit_s = float(limit_s); self.before_s = 0.; self.started = time.perf_counter()
        if not np.isfinite(self.limit_s) or self.limit_s <= 0:
            raise ValueError('Positive finite stage time cap required')

    @property
    def elapsed_s(self):
        return self.before_s + time.perf_counter() - self.started

    def check(self):
        if self.elapsed_s >= self.limit_s:
            raise TimeoutError('Cumulative stage wall cap reached; no automatic budget extension')

    def state_dict(self):
        return dict(limit_s=self.limit_s, elapsed_s=self.elapsed_s)

    def load_state_dict(self, state):
        if float(state['limit_s']) != self.limit_s or not np.isfinite(state['elapsed_s']) or state['elapsed_s'] < 0:
            raise ValueError('Stage clock restore budget/value mismatch')
        # The current invocation has already incurred loading/checking overhead.
        self.before_s = float(state['elapsed_s']); self.check()


class BudgetedTerminal:
    """Frozen full-H rewards, explicit per-role attempts and bounded persisted LRU."""
    KEYS = ('terminal_requests', 'terminal_cache_hits', 'terminal_attempts', 'terminal_completed',
            'terminal_failed', 'full_H_calls', 'export_attempts', 'exports',
            'validation_attempts', 'validation_completed')

    def __init__(self, ledger, folder, h_identity, limits, clock, record=None, cache_size=128):
        self.ledger, self.folder, self.h_identity = ledger, Path(folder), h_identity
        self.limits = {str(k): int(v) for k, v in limits.items()}
        if not self.limits or any(v < 0 for v in self.limits.values()) or not 0 <= cache_size <= 128:
            raise ValueError('Invalid terminal budget/cache cap')
        self.clock, self.record, self.cache_size = clock, record or (lambda _: None), int(cache_size)
        self.role = next(iter(self.limits)); self.cache = OrderedDict()
        self.stats = {r: {k: 0 for k in self.KEYS} for r in self.limits}
        self.internal = {r: _empty_internal() for r in self.limits}
        self.ledger['terminal'] = self.stats; self.ledger['internal_H_search'] = self.internal
        self.folder.mkdir(parents=True, exist_ok=True)

    def __call__(self, view, raw):
        if self.role not in self.limits:
            raise ValueError('Unregistered terminal role')
        s = self.stats[self.role]; s['terminal_requests'] += 1; self.clock.check()
        key = (view.identity, signature(self.h_identity), tuple(raw))
        if key in self.cache:
            s['terminal_cache_hits'] += 1; self.cache.move_to_end(key)
            self.record(dict(event='terminal_cache_hit', role=self.role, problem=view.identity, raw_route=raw))
            result = copy.deepcopy(self.cache[key]); result.update(cache_hit=True, H_s=0., export_s=0., validation_s=0.)
            return result
        if s['terminal_attempts'] >= self.limits[self.role]:
            raise RuntimeError('Registered terminal role budget exhausted')
        s['terminal_attempts'] += 1; ordinal = s['terminal_attempts']; tick = time.perf_counter()
        self.record(dict(event='terminal_start', role=self.role, ordinal=ordinal, problem=view.identity, raw_route=raw))
        try:
            with _instrument(self.internal[self.role]):
                result = plain(finish(view, raw, s))
            result['cache_hit'] = False
            write_new(self.folder/f'{self.role}_{ordinal:06}.json', dict(problem=view.description(), result=result))
            self.clock.check(); s['terminal_completed'] += 1
            self.record(dict(event='terminal_complete', role=self.role, ordinal=ordinal,
                             J=result['J'], L=result['L'], K=result['K'], wall_s=time.perf_counter()-tick))
        except Exception as exc:
            s['terminal_failed'] += 1
            self.record(dict(event='terminal_error', role=self.role, ordinal=ordinal,
                             error=type(exc).__name__, message=str(exc), wall_s=time.perf_counter()-tick))
            raise
        if self.cache_size:
            self.cache[key] = copy.deepcopy(result)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return result

    def state_dict(self):
        return dict(schema='rc3-trainer-terminal-state-v1', h_identity=plain(self.h_identity),
            limits=self.limits, cache_size=self.cache_size, role=self.role,
            stats=copy.deepcopy(self.stats), internal=copy.deepcopy(self.internal),
            cache=[dict(problem=k[0], H=k[1], raw=list(k[2]), result=copy.deepcopy(v)) for k, v in self.cache.items()])

    def validate_state_dict(self, state):
        if (state.get('schema') != 'rc3-trainer-terminal-state-v1' or state['h_identity'] != plain(self.h_identity)
                or state['limits'] != self.limits or state['cache_size'] != self.cache_size
                or state['role'] not in self.limits or set(state['stats']) != set(self.limits)):
            raise ValueError('Terminal source, role or budget state mismatch')
        if len(state['cache']) > self.cache_size:
            raise ValueError('Terminal cache exceeds registered bounded size')
        keys = set()
        for role, s in state['stats'].items():
            if set(s) != set(self.KEYS) or any(type(v) is not int or v < 0 for v in s.values()):
                raise ValueError('Invalid terminal counters')
            if s['terminal_attempts'] > self.limits[role] or s['terminal_completed'] > s['terminal_attempts']:
                raise ValueError('Terminal counters exceed budget')
        for row in state['cache']:
            key = (row['problem'], row['H'], tuple(row['raw']))
            if row['H'] != signature(self.h_identity) or key in keys or not np.isfinite(row['result']['J']):
                raise ValueError('Terminal cache identity/result mismatch')
            keys.add(key)

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.role = state['role']; self.stats = copy.deepcopy(state['stats'])
        self.internal = copy.deepcopy(state['internal'])
        self.cache = OrderedDict(((r['problem'], r['H'], tuple(r['raw'])), copy.deepcopy(r['result'])) for r in state['cache'])
        self.ledger['terminal'] = self.stats; self.ledger['internal_H_search'] = self.internal


class TeacherPool:
    """One shared paid label pool, no teacher calls during supervised updates."""
    def __init__(self, folder, registry, provider, ledger, clock, h_identity):
        self.folder, self.registry, self.provider = Path(folder), registry, provider
        self.ledger, self.clock = ledger, clock
        self.folder.mkdir(parents=True, exist_ok=True)
        self.h_identity = h_identity
        self.terminal = None

    def build(self):
        index = self.folder/'POOL_INDEX.json'
        if index.exists():
            self.validate(); return
        if any(self.folder.glob('label_*.json')):
            raise RuntimeError('Partial teacher pool: stop, do not silently re-query')
        self.terminal = BudgetedTerminal(self.ledger, self.folder/'H_plans', self.h_identity,
            {'teacher_H': 256}, self.clock, lambda r: journal(self.folder/'events.jsonl', r))
        labels = []
        for i, spec in enumerate(self.registry.teacher_specs()):
            self.clock.check(); view = self.provider.get(spec)
            add(self.ledger, 'native_teacher_attempts')
            if self.ledger['native_teacher_attempts'] > 256:
                raise RuntimeError('Teacher native budget exhausted')
            journal(self.folder/'events.jsonl', dict(event='teacher_native_start', index=i, problem=view.identity))
            plan, info = run_endpoint(view, 'detour16', counters=self.ledger.setdefault('teacher_native_internal', {}))
            add(self.ledger, 'native_teacher_completed')
            inverse = {int(g): j+1 for j, g in enumerate(view.candidate_ids)}
            raw = [0, *[inverse[int(g)] for g in plan['visit_order_candidate_ids']], 0]
            result = self.terminal(view, raw)
            label = dict(index=i, spec=spec, view_identity=view.identity,
                         actions=result['route'][1:-1], teacher_J=result['J'],
                         native_plan=plan, native_info=info, H_result=result,
                         registry_sha256=self.registry.sha256)
            path = self.folder/f'label_{i:03}.json'; write_new(path, label)
            labels.append(dict(index=i, file=path.name, sha256=digest(path)))
            self.clock.check()
        if self.ledger['native_teacher_attempts'] + self.terminal.stats['teacher_H']['terminal_attempts'] > 512:
            raise RuntimeError('Shared teacher total query cap exceeded')
        write_new(index, dict(schema='rc3-shared-teacher-pool-v1', registry_sha256=self.registry.sha256,
            count=256, labels=labels, ledger=self.ledger, elapsed_wall_s=self.clock.elapsed_s,
            source_binding=self.provider.permit.binding))

    def validate(self):
        index = read(self.folder/'POOL_INDEX.json')
        if (index['count'] != 256 or len(index['labels']) != 256
                or [r['index'] for r in index['labels']] != list(range(256))
                or any(r['file'] != f'label_{r["index"]:03}.json' for r in index['labels'])
                or index['registry_sha256'] != self.registry.sha256
                or index['source_binding'] != self.provider.permit.binding):
            raise ValueError('Teacher pool count/source/registry mismatch')
        for r in index['labels']:
            if digest(self.folder/r['file']) != r['sha256']:
                raise ValueError('Teacher label bytes changed')
        return index

    def example(self, index):
        if isinstance(index, bool) or not 0 <= int(index) < 256:
            raise ValueError('Teacher label index outside fixed pool')
        path = self.folder/f'label_{int(index):03}.json'
        frozen_index = read(self.folder/'POOL_INDEX.json')
        if digest(path) != frozen_index['labels'][int(index)]['sha256']:
            raise ValueError('Teacher label changed after pool validation')
        row = read(path)
        if row['index'] != int(index) or row['registry_sha256'] != self.registry.sha256:
            raise ValueError('Teacher label registry/index mismatch')
        view = self.provider.get(row['spec'])
        if view.identity != row['view_identity']:
            raise ValueError('Teacher labels cannot cross problem versions')
        metrics = view.metrics([0, *row['actions'], 0])
        if abs(metrics['J']-row['teacher_J']) > 1e-7:
            raise ValueError('Teacher route cost differs from frozen label')
        add(self.ledger, 'teacher_label_reads')
        return dict(view=view, actions=row['actions'], teacher_J=row['teacher_J'])


class TrainingRun:
    def __init__(self, permit, folder):
        self.permit, self.folder = permit, Path(folder)
        self.protocol = read(permit.protocol_path); self.registry = TrainingRegistry(permit.registry_path)
        self.stop_requested = False
        self.contexts = {}
        manifest = self.folder/'RUN_MANIFEST.json'
        if manifest.exists():
            if read(manifest)['binding'] != permit.binding:
                raise ValueError('Existing run belongs to another protocol/source/authorization')
        else:
            self.folder.mkdir(parents=True, exist_ok=True)
            write_new(manifest, dict(binding=permit.binding, status='ADMITTED_NOT_YET_EXECUTED',
                stages=permit.stages, target_role_whitelist=['train', 'development'], S3_allowed=False,
                checkpoint_every_updates=32, checkpoint_interruption='Graceful SIGINT saved at next completed update boundary; dirty crash logs reject implicit replay'))

    def _config(self, phase, seed):
        return dict(phase=phase, batch_instances=4, gamma=1., clip_global_norm=1.,
            mixture_weights=[1., .25, .25], entropy_weight=0., advantage_standardization=False,
            max_optimizer_updates=256 if phase == 'S1' else 512,
            binding=dict(self.permit.binding, optimizer_seed=seed, run=self.folder.relative_to(ROOT).as_posix(), phase=phase))

    def _components(self, phase, seed, limit_s, clock=None, record_context=True):
        folder = self.folder/f'seed_{seed}'/phase; folder.mkdir(parents=True, exist_ok=True)
        clock = clock or StageClock(limit_s); ledger = {}
        provider = SceneProvider(self.registry, self.folder/'data', ledger, self.permit,
            record=lambda r: journal(folder/'events.jsonl', r))
        # check is a method rather than a fixed absolute deadline, so restored
        # cumulative time is honored by data generation as well as every update.
        provider.check = clock.check
        terminal = BudgetedTerminal(ledger, folder/'plans', self.permit.binding['H_sha256'],
            {'RL': 6144, 'development': 2048}, clock, lambda r: journal(folder/'events.jsonl', r))
        if record_context:
            self.contexts[f'{seed}/{phase}'] = dict(clock=clock, ledger=ledger)
        return folder, clock, ledger, provider, terminal

    def build_teachers(self):
        if 'teachers' not in self.permit.stages:
            raise AdmissionError('Teacher generation is not authorized')
        clock = StageClock(self.protocol['S1']['wall_s']['shared_teacher_stage']); ledger = {}
        self.contexts['shared_teachers'] = dict(clock=clock, ledger=ledger)
        provider = SceneProvider(self.registry, self.folder/'data', ledger, self.permit,
            record=lambda r: journal(self.folder/'teacher_data_events.jsonl', r))
        provider.check = clock.check
        pool = TeacherPool(self.folder/'teachers256', self.registry, provider, ledger, clock, self.permit.binding['H_sha256'])
        pool.build(); return pool

    def _pool(self, provider, ledger, clock):
        pool = TeacherPool(self.folder/'teachers256', self.registry, provider, ledger, clock, self.permit.binding['H_sha256'])
        pool.validate(); return pool

    def _save(self, trainer, provider, clock, folder, name=None):
        cursor = trainer.cursors[trainer.phase]
        files = [folder/'updates.jsonl', folder/'events.jsonl']
        logs = {p.name: digest(p) if p.exists() else None for p in files}
        checkpoint = folder/(name or f'update_{cursor:04}.pt')
        snapshot_clock = clock.state_dict()
        receipt = save_checkpoint(checkpoint, trainer,
            extra=dict(provider=provider.state_dict(), clock=snapshot_clock,
                       journal_sha256=logs, checkpoint_stage=trainer.phase))
        # The checkpoint's embedded clock necessarily precedes its own large
        # serialization. An immutable closure record charges that IO on resume.
        hashes = dict(checkpoint_sha256=digest(checkpoint),
                      sidecar_sha256=digest(checkpoint.with_suffix(checkpoint.suffix+'.json')))
        write_new(checkpoint.with_suffix(checkpoint.suffix+'.commit.json'), dict(
            schema='rc3-checkpoint-boundary-commit-v1', **hashes, phase=trainer.phase,
            cursors=trainer.cursors, snapshot_elapsed_s=snapshot_clock['elapsed_s'],
            stage_elapsed_after_checkpoint_IO_s=clock.elapsed_s,
            journal_sha256=logs, terminal_cache_entries=len(getattr(trainer.terminal, 'cache', {})),
            scope='Includes checkpoint, integrity sidecar and their hash IO. Final tiny closure-record serialization is included by subsequent live stage clock, but is not self-timed in this record.'))
        clock.check()
        return receipt

    def _resume(self, checkpoint, trainer, provider, clock, folder):
        cp = Path(checkpoint).resolve()
        if not cp.is_relative_to(folder.resolve()):
            raise ValueError('Resume checkpoint must be inside this exact seed/phase folder')
        receipt = read(cp.with_suffix(cp.suffix+'.json'))
        if digest(cp) != receipt['sha256']:
            raise ValueError('Resume checkpoint hash mismatch')
        committed = read(cp.with_suffix(cp.suffix+'.commit.json'))
        if (committed['schema'] != 'rc3-checkpoint-boundary-commit-v1'
                or committed['checkpoint_sha256'] != digest(cp)
                or committed['sidecar_sha256'] != digest(cp.with_suffix(cp.suffix+'.json'))
                or committed['phase'] != trainer.phase or committed['cursors'] != receipt['cursors']):
            raise ValueError('Checkpoint boundary closure does not match checkpoint/sidecar/phase')
        # Inspect immutable metadata before load mutates any model/Adam/RNG.
        saved = torch.load(cp, map_location='cpu', weights_only=False)
        extra = saved['extra']
        post_IO = committed['stage_elapsed_after_checkpoint_IO_s']
        if (not np.isfinite(post_IO) or post_IO < extra['clock']['elapsed_s']
                or committed['snapshot_elapsed_s'] != extra['clock']['elapsed_s']
                or committed['journal_sha256'] != extra['journal_sha256']):
            raise ValueError('Invalid checkpoint closure timing/log binding')
        for name, expected in extra['journal_sha256'].items():
            path = folder/name; actual = digest(path) if path.exists() else None
            if actual != expected:
                raise RuntimeError('Work exists after this checkpoint; refuse uncharged/duplicate crash replay')
        extra = load_checkpoint(cp, trainer, self._config(trainer.phase, trainer.streams.optimizer_seed)['binding'])
        saved_clock = dict(extra['clock'], elapsed_s=post_IO)
        clock.load_state_dict(saved_clock); provider.load_state_dict(extra['provider'])
        trainer.ledger['resume_count'] = trainer.ledger.get('resume_count', 0)+1
        trainer.ledger['resume_loading_overhead_in_stage_wall'] = True

    def _loop(self, trainer, provider, clock, folder, pool=None):
        cap = trainer.config['max_optimizer_updates']
        schedule = TrainingSchedule(self.registry, trainer.streams)
        while trainer.cursors[trainer.phase] < cap:
            clock.check(); u = trainer.cursors[trainer.phase]
            if trainer.phase == 'RL':
                specs = schedule.batch(u)
                views = [provider.get(s) for s in specs]
                trainer.terminal.role = 'RL'; diag = trainer.s2_step(views)
                diag['sampling_specs'] = specs
            else:
                stream = 'S1_data' if trainer.phase == 'S1' else 'continuedSL_data'
                indices = trainer.streams.numpy(stream).integers(0, 256, 4).tolist()
                examples = [pool.example(i) for i in indices]
                diag = trainer.s1_step(examples); diag['teacher_indices'] = indices
            journal(folder/'updates.jsonl', dict(diag, stage_elapsed_s=clock.elapsed_s))
            cursor = trainer.cursors[trainer.phase]
            if trainer.phase == 'RL' and cursor in (256, 512):
                self.development(trainer.model, provider, trainer.terminal, folder, cursor)
            clock.check()
            if cursor % 32 == 0 or cursor == cap or self.stop_requested:
                self._save(trainer, provider, clock, folder)
            if self.stop_requested:
                write_new(folder/f'INTERRUPTED_{cursor:04}.json', dict(status='INTERRUPTED_SAFE_UPDATE_BOUNDARY',
                    cursor=cursor, ledger=trainer.ledger, elapsed_wall_s=clock.elapsed_s))
                return False
        return True

    def development(self, model, provider, terminal, folder, update):
        terminal.role = 'development'; was_training = model.training; model.eval()
        rows = []
        try:
            with torch.no_grad():
                for spec in self.registry.development_specs():
                    view = provider.get(spec); env = Environment(view, terminal)
                    encoded = model.encode(view); add(provider.ledger, 'development_encode_calls')
                    while not env.done:
                        scores = model.score(env, encoded); add(provider.ledger, 'development_score_calls')
                        env.step(int(torch.argmax(scores))+1)
                    rows.append(dict(case=spec['case'], problem=view.identity,
                        J=env.result['J'], L=env.result['L'], K=env.result['K'], feasible=True))
            write_new(folder/f'development_{update:04}.json', dict(update=update, rows=rows,
                purpose='Feasibility and resources only, no checkpoint selection', ledger=provider.ledger))
        finally:
            terminal.role = 'RL'; model.train(was_training)

    def s1(self, seed, resume=None):
        if 'S1' not in self.permit.stages or seed not in self.registry.policy_seeds:
            raise AdmissionError('Seed/S1 stage is not authorized')
        folder, clock, ledger, provider, terminal = self._components('S1', seed, self.protocol['S1']['wall_s']['supervised_per_seed'])
        pool = self._pool(provider, ledger, clock)
        torch.manual_seed(seed); model = GlobalLocalPolicy(); add(ledger, 'model_initializations')
        trainer = Trainer(model, make_optimizer(model, self.protocol['S1']['optimizer']),
            NamedStreams(self.permit.protocol_sha256, seed), terminal, self._config('S1', seed), ledger)
        if resume:
            self._resume(resume, trainer, provider, clock, folder)
        else:
            if (folder/'updates.jsonl').exists():
                raise RuntimeError('Existing S1 progress requires an explicit matching checkpoint')
            self._save(trainer, provider, clock, folder, 'untrained.pt')
        completed = self._loop(trainer, provider, clock, folder, pool)
        if completed:
            write_new(folder/'FINAL.json', dict(status='COMPLETE_FIXED_ENDPOINT', optimizer_updates=256,
                tensor_sha256=tensor_hash(model), checkpoint='update_0256.pt', ledger=ledger, wall_s=clock.elapsed_s))
        return completed

    def s2(self, seed, resume=None):
        shared_clock = StageClock(7200)
        if 'S2' not in self.permit.stages or seed not in self.registry.policy_seeds:
            raise AdmissionError('Seed/S2 stage is not authorized')
        # Load the exact source S1 endpoint using its own strict phase config.
        s1path = self.folder/f'seed_{seed}/S1/update_0256.pt'
        torch.manual_seed(seed); source = GlobalLocalPolicy()
        source_trainer = Trainer(source, make_optimizer(source, self.protocol['S1']['optimizer']),
            NamedStreams(self.permit.protocol_sha256, seed), lambda p, r: None,
            self._config('S1', seed), {})
        # S1 uses no real terminal, but its checkpoint contains an empty budgeted
        # terminal state. Provide the same adapter for strict restoration.
        _, source_clock, source_ledger, _, source_terminal = self._components('S1', seed,
            self.protocol['S1']['wall_s']['supervised_per_seed'], clock=shared_clock, record_context=False)
        source_trainer.terminal = source_terminal; source_trainer.ledger = source_ledger
        load_checkpoint(s1path, source_trainer, self._config('S1', seed)['binding'])
        if source_trainer.cursors['S1'] != 256:
            raise ValueError('S2 requires the complete fixed S1 endpoint, not a selected earlier model')
        branches = fork_from_supervised(source, self.protocol['S2']['optimizer_config'])
        start_hash = tensor_hash(source)
        if any(tensor_hash(b['model']) != start_hash for b in branches.values()):
            raise ValueError('Weight forks differ from common S1 endpoint')
        stage = self.folder/f'seed_{seed}/S2'; stage.mkdir(parents=True, exist_ok=True)
        fork_file = stage/'FORK_RECEIPT.json'
        fork = dict(source_S1_checkpoint_sha256=digest(s1path), common_tensor_sha256=start_hash,
                    branches={k: dict(tensor_sha256=tensor_hash(v['model']), optimizer_state_entries=0 if v['optimizer'] is None else len(v['optimizer'].state)) for k, v in branches.items()},
                    frozen_untrained_checkpoint='../S1/untrained.pt', frozen_SL=True, source_S1_optimizer_inherited=False)
        if fork_file.exists():
            if read(fork_file) != fork:
                raise ValueError('Existing weight lineage receipt differs')
        else:
            write_new(fork_file, fork)
        folder, clock, ledger, provider, terminal = self._components('RL', seed, 7200, clock=shared_clock)
        pool = self._pool(provider, ledger, clock)
        rl = Trainer(branches['RL']['model'], branches['RL']['optimizer'], NamedStreams(self.permit.protocol_sha256, seed),
                     terminal, self._config('RL', seed), ledger)
        resume_phase = read(Path(resume).with_suffix(Path(resume).suffix+'.json'))['phase'] if resume else None
        if resume_phase == 'RL':
            self._resume(resume, rl, provider, clock, folder)
        elif resume_phase not in (None, 'continued_SL'):
            raise ValueError('S2 resume phase must be RL or continued_SL')
        elif resume_phase is None:
            if (folder/'updates.jsonl').exists():
                raise RuntimeError('Existing RL progress requires explicit resume; no repeat optimization')
            self._save(rl, provider, clock, folder, 'start_from_SL.pt')
        if resume_phase != 'continued_SL' and not self._loop(rl, provider, clock, folder):
            return False
        cont_folder = self.folder/f'seed_{seed}/continued_SL'; cont_folder.mkdir(parents=True, exist_ok=True)
        # Restore-time provider operations must already use the continued-SL log.
        provider.record = lambda r: journal(cont_folder/'events.jsonl', r)
        terminal.record = lambda r: journal(cont_folder/'events.jsonl', r)
        continued = Trainer(branches['continued_SL']['model'], branches['continued_SL']['optimizer'],
            NamedStreams(self.permit.protocol_sha256, seed), terminal, self._config('continued_SL', seed), ledger)
        if resume_phase == 'continued_SL':
            self._resume(resume, continued, provider, clock, cont_folder)
        else:
            if (cont_folder/'updates.jsonl').exists():
                raise RuntimeError('Existing continued-SL progress requires explicit resume')
            self._save(continued, provider, clock, cont_folder, 'start_from_SL.pt')
        # Retarget pure logging to this phase, preserving the same cumulative
        # S2 clock, provider artifacts, rewards and aggregate budget counters.
        provider.record = lambda r: journal(cont_folder/'events.jsonl', r)
        terminal.record = lambda r: journal(cont_folder/'events.jsonl', r)
        if not self._loop(continued, provider, clock, cont_folder, pool):
            return False
        write_new(stage/'FINAL.json', dict(status='COMPLETE_FIXED_ENDPOINTS',
            RL_checkpoint='../RL/update_0512.pt', continued_SL_checkpoint='../continued_SL/update_0512.pt',
            frozen_SL_tensor_sha256=start_hash, shared_S2_wall_s=clock.elapsed_s, ledger=ledger,
            no_S3_queries=True, no_online_rule_fallback=True))
        return True

    def resources(self, status, error=None):
        """Persist even failed/partial attempts, never turn them into zero work."""
        name = 'RESOURCE_' + str(time.time_ns()) + '.json'
        contexts = {k: dict(ledger=v['ledger'], elapsed_wall_s=v['clock'].elapsed_s,
                           wall_cap_s=v['clock'].limit_s) for k, v in self.contexts.items()}
        write_new(self.folder/name, dict(status=status, error=error, contexts=contexts,
            note='Cumulative per-stage contexts; shared clocks/ledgers are not additive. Resume loading overhead is charged. No S3 or old60 semantic access.',
            formal_protocol_sha256=self.permit.protocol_sha256, S3_target_queries=0,
            original60_semantic_reads=0))
        return name


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('teachers', 'S1', 'S2', 'all'), required=True)
    parser.add_argument('--protocol', default=str(V2)); parser.add_argument('--registry', default=str(REGISTRY_V2))
    parser.add_argument('--protocol-approval'); parser.add_argument('--execution-authorization')
    parser.add_argument('--output', required=True); parser.add_argument('--seed', type=int)
    parser.add_argument('--resume'); parser.add_argument('--existing-run', action='store_true')
    args = parser.parse_args(argv)
    run = None
    try:
        if args.resume and (args.stage not in ('S1', 'S2') or args.seed is None):
            raise AdmissionError('Resume names one explicit --stage S1/S2 and --seed; never replay an entire all-seed workflow')
        permit = require_authority(args.protocol, args.registry, args.protocol_approval, args.execution_authorization, args.stage)
        folder = output_directory(args.output, resume=bool(args.resume or args.existing_run))
        run = TrainingRun(permit, folder)
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        def stop_at_boundary(signum, frame):
            run.stop_requested = True
        previous = signal.signal(signal.SIGINT, stop_at_boundary)
        try:
            if args.stage in ('teachers', 'all'):
                run.build_teachers()
            seeds = [args.seed] if args.seed is not None else list(run.registry.policy_seeds)
            for seed in seeds:
                if args.stage in ('S1', 'all') and not run.s1(seed, args.resume):
                    break
                if args.stage in ('S2', 'all') and not run.s2(seed, args.resume):
                    break
        finally:
            signal.signal(signal.SIGINT, previous)
        run.resources('INTERRUPTED_SAFE_UPDATE_BOUNDARY' if run.stop_requested else 'REQUESTED_STAGES_COMPLETE')
    except Exception as exc:
        if run is not None:
            run.resources('INCOMPLETE_STOPPED', dict(type=type(exc).__name__, message=str(exc)))
        print(json.dumps(dict(status='REJECTED_OR_INCOMPLETE', error=type(exc).__name__, message=str(exc), formal_success=False), ensure_ascii=False))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
