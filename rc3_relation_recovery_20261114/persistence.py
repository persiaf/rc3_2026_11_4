"""Single-writer, journal-first persistence; computation is NEVER retried here.

Complete JSONL events are authoritative. JSON snapshots are derived views.
An uncertain journal append stops; a torn tail is rejected without truncation.
This module deliberately contains no training, sampling or recovery launcher.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import time
import uuid
from collections import Counter
from pathlib import Path

RETRY_DELAYS = (0.02, 0.04, 0.08, 0.16, 0.32)


class PersistenceError(RuntimeError):
    def __init__(self, phase, cause):
        super().__init__(f'{phase}: {cause!r}')
        self.phase, self.cause = phase, cause


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def replace_retry(source, destination, *, replace=os.replace, sleep=time.sleep, observe=None):
    """At most 6 attempts / 0.62 s sleep, and only transient-access error codes.

    Source bytes are written and fsynced ONCE by the caller. Never delete the
    destination. A persistent denial leaves source and destination for diagnosis.
    """
    for attempt in range(1, len(RETRY_DELAYS) + 2):
        try:
            replace(source, destination)
            if observe:
                observe(dict(attempt=attempt, success=True))
            return attempt
        except OSError as exc:
            retryable = (getattr(exc, 'winerror', None) in (5, 32, 33)
                         or (os.name == 'nt' and exc.errno == errno.EACCES))
            if observe:
                observe(dict(attempt=attempt, success=False, errno=exc.errno,
                             winerror=getattr(exc, 'winerror', None)))
            if not retryable or attempt > len(RETRY_DELAYS):
                raise PersistenceError('atomic_replace', exc) from exc
            sleep(RETRY_DELAYS[attempt - 1])


def put(path, value, *, replace=os.replace, sleep=time.sleep, observe=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.' + uuid.uuid4().hex + '.pending')
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()
    try:
        with pending.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PersistenceError('write_and_fsync', exc) from exc
    return replace_retry(pending, path, replace=replace, sleep=sleep, observe=observe)


def append(path, value):
    # Never retry an append: a failed fsync may still have persisted its bytes.
    try:
        with Path(path).open('ab') as handle:
            data = canonical(value) + b'\n'
            if handle.write(data) != len(data):
                raise OSError('short journal append')
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise PersistenceError('journal_append_uncertain', exc) from exc


class WriterLock:
    """OS byte-range lock, released on process exit; stale files are harmless."""
    def __init__(self, path):
        self.handle = Path(path).open('a+b')
        self.handle.seek(0, 2)
        if not self.handle.tell():
            self.handle.write(b'0')
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError('A writer already owns this stage') from None

    def close(self):
        if not self.handle.closed:
            self.handle.close()


def reduce_events(events, initial=None):
    """Pure strict replay, also used when a snapshot is stale or missing."""
    result = dict(counts=Counter(), ids=set(), pending={}, terminal={},
                  status='CREATED', cursor=0, elapsed_s=0., event_ids={},
                  seq=0, head='0' * 64, identity=None, limits=None, origin=None)
    if initial is not None:
        result = dict(initial, counts=Counter(initial['counts']), ids=set(initial['ids']),
                      pending=dict(initial['pending']), terminal=dict(initial['terminal']),
                      event_ids=dict(initial['event_ids']))
    for event in events:
        body = {k: v for k, v in event.items() if k != 'event_sha256'}
        if (event['seq'] != result['seq'] + 1 or event['prev_sha256'] != result['head']
                or event['event_sha256'] != digest(body) or event['event_id'] in result['event_ids']):
            raise ValueError('Journal sequence/hash/duplicate event mismatch')
        name = event['event']
        if name == 'genesis':
            if result['seq']:
                raise ValueError('Repeated genesis')
            result.update(identity=event['identity'], limits=event['limits'], origin=event['origin'])
            origin = event['origin']
            result['counts'].update(origin.get('counts', {}))
            result['ids'].update(origin.get('ids', []))
            result['cursor'] = origin.get('cursor', 0)
            result['status'] = 'RUNNING'
        elif result['identity'] is None:
            raise ValueError('Missing genesis')
        elif name == 'begin':
            rid, kind = event['id'], event['kind']
            if rid in result['ids'] or kind not in ('request', 'optimizer'):
                raise ValueError('Duplicate debit or unknown kind')
            result['ids'].add(rid)
            result['pending'][rid] = kind
            result['counts'][kind + '_attempts'] += 1
        elif name in ('complete', 'failed'):
            rid, kind = event['id'], event['kind']
            if result['pending'].get(rid) != kind or rid in result['terminal']:
                raise ValueError('Terminal event without unique debit')
            result['pending'].pop(rid)
            result['terminal'][rid] = name
            result['counts'][kind + ('_completed' if name == 'complete' else '_failed')] += 1
            if name == 'failed':
                result['status'] = 'TECHNICAL_FAILURE'
        elif name == 'checkpoint_commit':
            cursor = event['cursor']
            if (result['pending'] or cursor != result['cursor'] + 1
                    or cursor != result['counts']['optimizer_completed']):
                raise ValueError('Invalid optimizer/checkpoint cursor')
            result['cursor'] = cursor
        elif name in ('persistence_failed', 'close'):
            result['status'] = event['status']
        elif name != 'request_full_time':
            raise ValueError('Unknown journal event: ' + name)
        # Instrumentation counts are supplemental, never allowed to replace debits.
        for key, value in event.get('aux_counts', {}).items():
            if key.endswith(('_attempts', '_completed', '_failed')):
                raise ValueError('Auxiliary counts cannot override transactional counts')
            result['counts'][key] = value
        result['elapsed_s'] = max(result['elapsed_s'], event['elapsed_s'])
        result.update(seq=event['seq'], head=event['event_sha256'])
        result['event_ids'][event['event_id']] = event
    return result


def load_events(path):
    data = Path(path).read_bytes() if Path(path).exists() else b''
    if data and not data.endswith(b'\n'):
        raise ValueError('Torn journal tail retained; explicit disposition required')
    return [json.loads(line) for line in data.splitlines()]


class Ledger:
    def __init__(self, folder, limits, identity, *, origin=None, writer=put, appender=append):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / 'RESOURCE_ACCOUNTING.json'
        self.journal = self.folder / 'EVENTS.v2.jsonl'
        self.writer, self.appender = writer, appender
        self.lock = WriterLock(self.folder / 'WRITER.lock')
        self.stopped = False
        self.start = time.perf_counter()
        self.start_wall = time.time()
        try:
            self.events = load_events(self.journal)
            self.state = reduce_events(self.events)
            latest_checkpoint = None
            for event in self.events:
                if event['event'] == 'complete' and event['kind'] == 'request':
                    record = (self.folder / event['record']).resolve()
                    if (not record.is_relative_to(self.folder.resolve())
                            or sha(record) != event['sha256']):
                        raise ValueError('Completed request artifact missing or changed')
                if event['event'] == 'checkpoint_commit':
                    latest_checkpoint = event
            if latest_checkpoint:
                record = (self.folder / latest_checkpoint['checkpoint']).resolve()
                if not record.is_relative_to(self.folder.resolve()) or sha(record) != latest_checkpoint['sha256']:
                    raise ValueError('Last committed checkpoint artifact missing or changed')
                if (self.state['counts']['optimizer_completed'] == self.state['cursor']
                        and sha(self.folder / 'STATE.json') != latest_checkpoint['state_sha256']):
                    raise ValueError('STATE does not match the committed checkpoint event')
            self.limits, self.identity = limits, identity
            self.prior_wall = max(self.state['elapsed_s'], (origin or {}).get('elapsed_s', 0.))
            self._refresh()
            if not self.events:
                self._event('genesis', 'genesis', identity=identity, limits=limits, origin=origin or {})
            elif (self.state['identity'] != identity or self.state['limits'] != limits
                  or (origin is not None and self.state['origin'] != origin)):
                raise ValueError('Cannot reset origin/identity/limits')
            self.save()
        except BaseException:
            self.lock.close()
            raise

    def _refresh(self):
        self.counts = Counter(self.state['counts'])
        self.pending, self.ids = dict(self.state['pending']), set(self.state['ids'])
        self.status, self.cursor = self.state['status'], self.state['cursor']

    def elapsed(self):
        return self.prior_wall + max(time.perf_counter() - self.start, time.time() - self.start_wall)

    def check(self):
        if self.stopped or self.status != 'RUNNING':
            raise RuntimeError('Ledger stopped; no additional computation permitted')
        if self.elapsed() >= self.limits['wall_s']:
            raise TimeoutError('Cumulative stage wall limit')

    def _event(self, name, event_id, **fields):
        old = self.state['event_ids'].get(event_id)
        if old is not None:
            if old['event'] != name or any(old.get(k) != v for k, v in fields.items()):
                raise ValueError('Idempotency key reused for different event')
            return old
        aux = {k: v for k, v in self.counts.items()
               if not k.endswith(('_attempts', '_completed', '_failed'))}
        event = dict(seq=self.state['seq'] + 1, prev_sha256=self.state['head'],
                     event=name, event_id=event_id, elapsed_s=self.elapsed(), aux_counts=aux, **fields)
        event['event_sha256'] = digest(event)
        next_state = reduce_events([event], initial=self.state)
        try:
            self.appender(self.journal, event)
        except BaseException:
            self.stopped = True
            raise
        self.events.append(event)
        self.state = next_state
        self._refresh()
        return event

    def snapshot(self):
        return dict(schema='relation-resource-v2', identity=self.identity, limits=self.limits,
                    counts=dict(self.counts), pending=self.pending, status=self.status,
                    cursor=self.cursor, elapsed_s=self.elapsed(), journal_sequence=self.state['seq'],
                    journal_head=self.state['head'], scientific_updates=self.counts['optimizer_completed'],
                    origin=self.state['origin'])

    def save(self):
        try:
            self.writer(self.path, self.snapshot())
        except BaseException:
            self.stopped = True
            raise

    def begin(self, kind, rid, **fields):
        tick = time.perf_counter()
        self.check()
        if rid in self.ids:
            raise ValueError('Attempt ID already consumed: ' + rid)
        if self.pending:
            raise RuntimeError('An unresolved transaction prevents new computation')
        if self.counts['optimizer_completed'] != self.cursor:
            raise RuntimeError('A completed optimizer lacks a committed checkpoint boundary')
        if self.counts[kind + '_attempts'] >= self.limits[kind + '_attempts']:
            raise RuntimeError('Cumulative attempt limit')
        self._event('begin', 'begin:' + rid, kind=kind, id=rid, **fields)
        self.save()
        return tick

    def complete(self, kind, rid, tick, **fields):
        duplicate = self.state['terminal'].get(rid) == 'complete'
        if duplicate:
            old = self.state['event_ids']['complete:' + rid]
            if old['kind'] != kind or any(old.get(k) != v for k, v in fields.items()):
                raise ValueError('Completion identity changed')
        if kind == 'request':
            if 'record' not in fields or 'sha256' not in fields:
                raise ValueError('Completion requires a request record and its hash')
            record = (self.folder / fields['record']).resolve()
            if (not record.is_relative_to(self.folder.resolve()) or not record.is_file()
                    or sha(record) != fields['sha256']):
                raise ValueError('Completion requires exact persisted request bytes')
        if duplicate:
            # Explicit republish validates the same artifact and appends nothing.
            self.save()
            return
        self._event('complete', 'complete:' + rid, kind=kind, id=rid,
                    wall_s=time.perf_counter() - tick, **fields)
        self.save()

    def fail(self, kind, rid, exc):
        # Frozen runtime's outer except can call this after complete(). Do not
        # reinterpret a committed completion as failed computation.
        if isinstance(exc, PersistenceError) and exc.phase == 'journal_append_uncertain':
            self.stopped = True
            return
        if 'persistence_failed:' + rid in self.state['event_ids']:
            self.stopped = True
            return
        if self.state['terminal'].get(rid) == 'complete':
            self._event('persistence_failed', 'persistence_failed:' + rid, id=rid,
                        status='TECHNICAL_INCOMPLETE', error=repr(exc))
        elif isinstance(exc, PersistenceError):
            # An uncertain append must be re-read by a new owner, never appended over.
            self.stopped = True
            return
        elif self.state['terminal'].get(rid) == 'failed':
            return
        else:
            self._event('failed', 'failed:' + rid, kind=kind, id=rid, error=repr(exc))
        self.stopped = True
        self.save()

    def event(self, event, **fields):
        if event != 'request_full_time':
            raise ValueError('Use a typed transaction method')
        self._event(event, event + ':' + fields['id'], **fields)

    def record_request(self, rid, value, tick):
        path = self.folder / 'requests' / (rid + '.json')
        self.writer(path, value)
        self.complete('request', rid, tick, record=str(path.relative_to(self.folder)), sha256=sha(path))
        return path

    def reconcile_record(self, rid, expected_sha256):
        """Explicit persistence-only reconciliation; no callable work is accepted."""
        path = self.folder / 'requests' / (rid + '.json')
        if self.pending.get(rid) != 'request' or sha(path) != expected_sha256:
            raise ValueError('No exact pending saved record')
        if read(path).get('request_id') != rid:
            raise ValueError('Saved record belongs to another request')
        self.complete('request', rid, time.perf_counter(), record=str(path.relative_to(self.folder)),
                      sha256=expected_sha256, reconciliation='PERSISTENCE_ONLY_NOT_RECOMPUTATION')

    def commit_checkpoint(self, cursor, checkpoint_path, state_value):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.resolve().is_relative_to(self.folder.resolve()):
            raise ValueError('Checkpoint outside stage')
        checkpoint_sha = sha(checkpoint_path)
        if state_value.get('cursor') != cursor or state_value.get('checkpoint_sha256') != checkpoint_sha:
            raise ValueError('State/checkpoint identity mismatch')
        if (self.pending or cursor != self.cursor + 1
                or cursor != self.counts['optimizer_completed']):
            raise ValueError('No completed optimizer boundary for this cursor')
        self.writer(self.folder / 'STATE.json', state_value)
        self._event('checkpoint_commit', f'checkpoint:{cursor}', cursor=cursor,
                    checkpoint=str(checkpoint_path.relative_to(self.folder)), sha256=checkpoint_sha,
                    state_sha256=sha(self.folder / 'STATE.json'))
        self.save()

    def publish(self):
        """Publish unique immutable status files; observers never open active ledger."""
        folder = self.folder / 'published'
        path = folder / f'status_{self.state["seq"]:08d}_{uuid.uuid4().hex}.json'
        self.writer(path, self.snapshot())
        return path

    def close(self, status):
        try:
            self._event('close', 'close:' + status, status=status)
            self.save()
        finally:
            self.lock.close()

    def release(self):
        """Release ownership after a failure without rewriting evidence."""
        self.lock.close()


def observe_published(folder):
    # Restrict reads to immutable publications. No active file fallback.
    paths = sorted((Path(folder) / 'published').glob('status_*.json'))
    if not paths:
        return dict(status='NO_PUBLISHED_STATUS')
    path = paths[-1]
    return dict(publication=path.name, snapshot=read(path))
