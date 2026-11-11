"""Durable pre-debits and a non-replayable cumulative offline experiment clock."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')
        stream.flush()
        os.fsync(stream.fileno())


class Budget:
    def __init__(self, folder, registration, *, started):
        self.folder = Path(folder)
        self.path = self.folder / 'events.jsonl'
        self.resource = self.folder / 'RESOURCE_ACCOUNTING.json'
        if self.path.exists() or self.resource.exists():
            raise RuntimeError('This one-shot study already consumed resources; no reset/replay')
        self.started = started
        self.caps = registration['caps']
        self.counts = {key: 0 for key in self.caps if key != 'wall_s'}
        self.timings = {}
        self.tail = '0' * 64
        self.sequence = 0
        self.registration_sha256 = digest(self.folder / 'PREREGISTRATION.json')
        self.emit(event='start', utc=datetime.now(timezone.utc).isoformat(),
                  registration_sha256=self.registration_sha256, caps=self.caps)

    def emit(self, **row):
        body = dict(row, previous=self.tail, sequence=self.sequence)
        self.tail = hashlib.sha256(canonical(body)).hexdigest()
        body['sha256'] = self.tail
        with self.path.open('ab') as stream:
            stream.write(canonical(body) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence += 1

    def check(self):
        if time.perf_counter() - self.started >= self.caps['wall_s']:
            raise TimeoutError('The single cumulative 3600-second science budget is exhausted')

    def charge(self, kind, amount=1, **context):
        self.check()
        if kind == 'timing':
            seconds = context.pop('seconds')
            category = context.pop('category')
            if seconds < 0:
                raise ValueError('Negative component time')
            self.timings[category] = self.timings.get(category, 0.) + seconds
            return
        if kind not in self.caps or type(amount) is not int or amount < 0:
            raise ValueError('Unregistered counter: ' + str(kind))
        if self.counts[kind] + amount > self.caps[kind]:
            self.emit(event='cap_refusal', kind=kind, amount=amount, context=context)
            raise RuntimeError('Registered cap exceeded: ' + kind)
        self.emit(event='debit', kind=kind, amount=amount, context=context)
        self.counts[kind] += amount

    @contextmanager
    def timed(self, category):
        self.check()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.charge('timing', 0, category=category, seconds=time.perf_counter()-start)

    def close(self, status, error=None):
        elapsed = time.perf_counter() - self.started
        self.emit(event='close', status=status, error=error, wall_s=elapsed,
                  counts=self.counts, completed_component_wall_s=self.timings)
        write_new(self.resource, dict(schema='rc3-offline-critic-rank-resource-v1',
            status=status, registration_sha256=self.registration_sha256,
            caps=self.caps, counts=self.counts, wall_s=elapsed,
            completed_component_wall_s=self.timings, journal_tail=self.tail,
            events=self.sequence, error=error, failures_retained=True,
            component_times_can_be_nested=True,
            clock_scope='Entry before scientific imports through all registered computation and artifact writes; source reads, failed attempts included. Git and code preparation separate.',
            original_R03_ledger_unchanged=True, no_joint_PPO_authorized=True))
