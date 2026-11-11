"""Durable cumulative accounting across registration and one execution session."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

HOME = Path(__file__).resolve().parent
ROOT = HOME.parent
CAPS = dict(wall_s=1800, critic_optimizer_attempts=512, actor_optimizer_attempts=0,
            saved_trajectories=128, new_scenes=0, navigation_requests=0,
            teacher_requests=0, native_requests=0, H_requests=0,
            original60_reads=0, oldS3_reads=0)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

class Budget:
    def __init__(self, phase, started):
        self.started = started
        self.path = HOME/'RESOURCE_ACCOUNTING.json'
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
            if self.data['session_open']:
                raise RuntimeError('Unclosed prior session: do not reset consumed budget; explicit recovery receipt required')
            if self.data['status'] not in ('METADATA_REGISTERED',):
                raise RuntimeError('This exact stage has already been consumed or failed')
        else:
            if phase != 'metadata': raise RuntimeError('Metadata registration must run first')
            self.data = dict(schema='rc3-r04-m1-resource-v1', caps=CAPS, counts={k:0 for k in CAPS if k!='wall_s'},
                wall_s=0., sessions=[], sequence=0, event_sha256='0'*64,
                scientific_clock='Cumulative active registration + engineering + saved-input reads/restoration + fit/evaluation/save/recheck process wall; code authoring and Git transport are separate maintenance, never reset prior consumption',
                read_counter_scope='Instrumented entry paths only; nested immutable navigation/legacy provenance helpers may perform additional reads/hashes. Their entire elapsed cost remains inside cumulative wall. These counters are not a complete filesystem syscall census.',
                failures=[])
            retained = HOME/'PRE_REGISTRATION_FAILURE.json'
            if retained.exists():
                failure = json.loads(retained.read_text(encoding='utf-8-sig'))
                self.data['wall_s'] = failure['conservative_charged_wall_s']
                self.data['failures'].append(failure)
                self.data['sessions'].append(dict(phase='metadata_initialization_failure',status='FAILED',
                    wall_s=failure['conservative_charged_wall_s']))
        self.base = self.data['wall_s']; self.phase = phase
        self.data.update(session_open=True, status='RUNNING', active_phase=phase,
                         opened_utc=datetime.now(timezone.utc).isoformat())
        self.emit('session_start')

    def elapsed(self): return self.base + time.perf_counter() - self.started

    def persist(self):
        self.data['wall_s'] = self.elapsed()
        temp = self.path.with_suffix('.tmp')
        with temp.open('w',encoding='utf-8',newline='\n') as f:
            f.write(json.dumps(self.data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
            f.flush();os.fsync(f.fileno())
        temp.replace(self.path)

    def check(self):
        if self.elapsed() > CAPS['wall_s']: raise RuntimeError('Cumulative 1800-second cap reached')

    def emit(self, event, **info):
        self.data['sequence'] += 1
        record = dict(sequence=self.data['sequence'], event=event, phase=self.phase,
                      cumulative_wall_s=self.elapsed(), previous_sha256=self.data['event_sha256'], **info)
        record['sha256'] = hashlib.sha256(canonical(record)).hexdigest()
        with (HOME/'EVENTS.jsonl').open('a', encoding='utf-8', newline='\n') as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n'); f.flush();os.fsync(f.fileno())
        self.data['event_sha256'] = record['sha256']; self.persist()

    def charge(self, kind, amount=1, **info):
        self.check()
        if kind == 'timing':
            name = info['category']; times = self.data.setdefault('component_seconds', {})
            times[name] = times.get(name, 0.) + info['seconds']; return
        current = self.data['counts'].get(kind, 0)
        if kind in CAPS and current + amount > CAPS[kind]:
            raise RuntimeError('Resource limit would be exceeded: '+kind)
        self.data['counts'][kind] = current + amount
        if kind in ('critic_optimizer_attempts','saved_trajectories'):
            self.emit('predebit', kind=kind, amount=amount, **info)

    def close(self, status, error=None):
        if error is not None: self.data['failures'].append(dict(phase=self.phase, error=str(error)))
        self.data['sessions'].append(dict(phase=self.phase, wall_s=time.perf_counter()-self.started, status=status))
        self.data.update(status=status, session_open=False)
        self.emit('session_close', status=status, error=str(error) if error else None)
