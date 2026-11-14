"""Bounded synthetic Windows fault injection. Never reads scientific data."""
from __future__ import annotations
import io
import json
import multiprocessing as mp
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path

from . import persistence as p
from .runtime_boundary import persist_request, commit_update, require_recovery_authorization

LIMITS = dict(request_attempts=100, optimizer_attempts=10, wall_s=60)
EVIDENCE = []
TEMP_ROOT = Path(__file__).resolve().parent / '.engineering_tmp'
TRANSACTIONS = 0


def denied():
    exc = PermissionError(13, 'Injected Windows access denial')
    exc.winerror = 5
    return exc


def hold_reader(path, ready, release, done, connection):
    with Path(path).open('r', encoding='utf8') as handle:
        content = handle.read()
        connection.send(dict(pid=os.getpid(), method='Path.open(r, encoding=utf8)', bytes=len(content)))
        ready.set()
        release.wait(5)
    done.set()


class MeteredLedger(p.Ledger):
    def begin(self, *args, **kwargs):
        global TRANSACTIONS
        if TRANSACTIONS >= int(os.environ.get('RC3_ENGINEERING_TRANSACTION_CAP', '2000')):
            raise RuntimeError('Engineering synthetic transaction cap')
        TRANSACTIONS += 1
        return super().begin(*args, **kwargs)


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        TEMP_ROOT.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='case_', dir=TEMP_ROOT)
        self.root = Path(self.temp.name)
        self.ledgers = []
        self.evidence = dict(case=self.id().rsplit('.', 1)[-1])
        self.start = time.perf_counter()

    def tearDown(self):
        for ledger in self.ledgers:
            ledger.release()
        self.evidence['wall_s'] = time.perf_counter() - self.start
        EVIDENCE.append(self.evidence)
        # TemporaryDirectory deletes only its freshly created, resolved child.
        assert self.root.resolve().is_relative_to(TEMP_ROOT.resolve())
        self.temp.cleanup()

    def ledger(self, **kwargs):
        result = MeteredLedger(self.root, kwargs.pop('limits', LIMITS), 'ENGINEERING:synthetic', **kwargs)
        self.ledgers.append(result)
        return result

    def reopen(self, ledger, **kwargs):
        ledger.release()
        return self.ledger(**kwargs)

    def request(self, ledger, rid='r1'):
        tick = ledger.begin('request', rid)
        ledger.record_request(rid, dict(request_id=rid, result='synthetic'), tick)
        return tick

    def test_01_plain_reader_reproduces_windows_denial(self):
        self.assertEqual(os.name, 'nt')
        path = self.root / 'resource.json'
        p.put(path, {'n': 0})
        ctx = mp.get_context('spawn')
        ready, release, done = ctx.Event(), ctx.Event(), ctx.Event()
        recv, send = ctx.Pipe(duplex=False)
        child = ctx.Process(target=hold_reader, args=(str(path), ready, release, done, send))
        child.start()
        try:
            self.assertTrue(ready.wait(5))
            self.evidence['known_reader'] = recv.recv()
            pending = self.root / 'old.pending'
            pending.write_text('{"n":1}', encoding='utf8')
            with self.assertRaises(OSError) as err:
                os.replace(pending, path)
            self.evidence['observed_winerror'] = err.exception.winerror
            self.assertIn(err.exception.winerror, (5, 32, 33))
            self.assertEqual(p.read(path), {'n': 0})
            release.set()
            self.assertTrue(done.wait(5))
            os.replace(pending, path)
            self.assertEqual(p.read(path), {'n': 1})
        finally:
            release.set()
            child.join(5)
            if child.is_alive():
                child.terminate()
                child.join()
            recv.close(); send.close()

    def test_02_live_reader_release_allows_bounded_retry(self):
        path = self.root / 'resource.json'
        p.put(path, {'n': 0})
        ctx = mp.get_context('spawn')
        ready, release, done = ctx.Event(), ctx.Event(), ctx.Event()
        recv, send = ctx.Pipe(duplex=False)
        child = ctx.Process(target=hold_reader, args=(str(path), ready, release, done, send))
        child.start()
        attempts = []
        try:
            self.assertTrue(ready.wait(5))
            self.evidence['known_reader'] = recv.recv()
            def observe(row):
                attempts.append(row)
                if not row['success']:
                    release.set()
            p.put(path, {'n': 1}, observe=observe)
            self.assertTrue(done.wait(5))
            self.assertGreater(len(attempts), 1)
            self.assertLessEqual(len(attempts), 6)
            self.evidence['replace_attempts'] = attempts
        finally:
            release.set(); child.join(5)
            if child.is_alive():
                child.terminate(); child.join()
            recv.close(); send.close()

    def test_03_transient_injection_retries_only_replace(self):
        ledger = self.ledger()
        tick = ledger.begin('request', 'r1')
        attempts, sleeps, denials = [], [], []
        def replace(a, b):
            if Path(b).name == 'RESOURCE_ACCOUNTING.json' and len(denials) < 2:
                attempts.append(False)
                denials.append(1)
                raise denied()
            attempts.append(True)
            os.replace(a, b)
        ledger.writer = lambda a, b: p.put(a, b, replace=replace, sleep=sleeps.append)
        persist_request(ledger, 'r1', dict(request_id='r1', result='only one computation'), tick)
        events = p.load_events(ledger.journal)
        self.assertEqual(sum(e['event'] == 'complete' for e in events), 1)
        self.assertEqual(ledger.counts['request_attempts'], 1)
        self.assertEqual(ledger.counts['request_completed'], 1)
        self.assertEqual(sleeps, [0.02, 0.04])
        self.evidence.update(replace_calls=len(attempts), sleeps=sleeps, computed_requests=1)

    def test_04_persistent_denial_leaves_destination(self):
        path = self.root / 'resource.json'
        p.put(path, dict(old=True))
        before = path.read_bytes()
        attempts, sleeps = [], []
        def replace(a, b):
            attempts.append(1)
            raise denied()
        with self.assertRaises(p.PersistenceError):
            p.put(path, dict(old=False), replace=replace, sleep=sleeps.append)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(len(attempts), 6)
        self.assertAlmostEqual(sum(sleeps), .62)
        self.assertEqual(len(list(self.root.glob('*.pending'))), 1)
        self.evidence.update(attempts=len(attempts), retry_sleep_s=sum(sleeps), destination_preserved=True)

    def test_05_real_readonly_destination_stops(self):
        path = self.root / 'resource.json'
        p.put(path, {'old': True})
        os.chmod(path, stat.S_IREAD)
        attempts = []
        try:
            with self.assertRaises(p.PersistenceError):
                p.put(path, {'old': False}, sleep=lambda _: None, observe=attempts.append)
            self.assertEqual(p.read(path), {'old': True})
            self.assertEqual(len(attempts), 6)
            self.evidence['attempts'] = attempts
        finally:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)

    def test_06_complete_then_snapshot_failure_rebuilds_once(self):
        ledger = self.ledger()
        tick = ledger.begin('request', 'r1')
        def writer(path, value):
            if Path(path).name == 'RESOURCE_ACCOUNTING.json':
                raise p.PersistenceError('atomic_replace', denied())
            p.put(path, value)
        ledger.writer = writer
        with self.assertRaises(p.PersistenceError):
            persist_request(ledger, 'r1', dict(request_id='r1'), tick)
        events = p.load_events(ledger.journal)
        self.assertEqual(sum(e['event'] == 'complete' for e in events), 1)
        self.assertEqual(sum(e['event'] == 'failed' for e in events), 0)
        self.assertEqual(sum(e['event'] == 'persistence_failed' for e in events), 1)
        with self.assertRaises(RuntimeError):
            ledger.begin('request', 'r2')
        rebuilt = self.reopen(ledger)
        self.assertEqual(rebuilt.counts['request_attempts'], 1)
        self.assertEqual(rebuilt.counts['request_completed'], 1)
        self.assertEqual(rebuilt.cursor, 0)
        self.assertEqual(rebuilt.status, 'TECHNICAL_INCOMPLETE')
        self.evidence.update(requests=1, completed=1, failed=0, persistence_failed=1, cursor=0)

    def test_07_record_before_completion_needs_explicit_reconcile(self):
        ledger = self.ledger()
        ledger.begin('request', 'r1')
        path = self.root / 'requests/r1.json'
        p.put(path, dict(request_id='r1', result=7))
        rebuilt = self.reopen(ledger)
        self.assertEqual(rebuilt.pending, {'r1': 'request'})
        with self.assertRaises(RuntimeError):
            rebuilt.begin('request', 'r2')
        rebuilt.reconcile_record('r1', p.sha(path))
        self.assertEqual(rebuilt.counts['request_attempts'], 1)
        self.assertEqual(rebuilt.counts['request_completed'], 1)
        self.assertEqual(rebuilt.cursor, 0)

    def test_08_duplicate_completion_is_idempotent(self):
        ledger = self.ledger()
        tick = self.request(ledger)
        complete = ledger.state['event_ids']['complete:r1']
        ledger.complete('request', 'r1', tick, record=complete['record'], sha256=complete['sha256'])
        self.assertEqual(ledger.counts['request_completed'], 1)
        self.assertEqual(sum(e['event'] == 'complete' for e in ledger.events), 1)

    def test_09_duplicate_debit_and_conflicting_completion_rejected(self):
        ledger = self.ledger()
        tick = self.request(ledger)
        with self.assertRaises(ValueError):
            ledger.begin('request', 'r1')
        with self.assertRaises((ValueError, KeyError)):
            ledger.complete('request', 'r1', tick, record='changed')
        self.assertEqual(ledger.counts['request_attempts'], 1)

    def test_10_uncertain_append_is_never_retried(self):
        ledger = self.ledger()
        calls = []
        def append_then_fail(path, event):
            calls.append(event['event'])
            p.append(path, event)
            raise p.PersistenceError('journal_append_uncertain', OSError('injected fsync uncertainty'))
        ledger.appender = append_then_fail
        with self.assertRaises(p.PersistenceError):
            ledger.begin('request', 'r1')
        self.assertEqual(calls, ['begin'])
        rebuilt = self.reopen(ledger)
        self.assertEqual(rebuilt.counts['request_attempts'], 1)
        self.assertEqual(rebuilt.pending, {'r1': 'request'})

    def test_11_torn_journal_refuses_without_truncating(self):
        ledger = self.ledger()
        ledger.release()
        with ledger.journal.open('ab') as handle:
            handle.write(b'{"incomplete":')
        before = ledger.journal.read_bytes()
        with self.assertRaises(ValueError):
            self.ledger()
        self.assertEqual(ledger.journal.read_bytes(), before)

    def test_12_hash_tampering_is_rejected(self):
        ledger = self.ledger()
        ledger.release()
        data = ledger.journal.read_bytes().replace(b'ENGINEERING:synthetic', b'ENGINEERING:tampered')
        ledger.journal.write_bytes(data)
        with self.assertRaises(ValueError):
            self.ledger()

    def test_13_second_writer_is_rejected(self):
        ledger = self.ledger()
        with self.assertRaises(RuntimeError):
            self.ledger()
        ledger.release()
        self.ledger()

    def test_14_immutable_monitor_handle_never_blocks_active_ledger(self):
        ledger = self.ledger()
        publication = ledger.publish()
        with publication.open('r', encoding='utf8') as observer:
            for i in range(40):
                self.request(ledger, f'r{i}')
                ledger.publish()
            self.assertEqual(json.load(observer)['counts'].get('request_attempts', 0), 0)
        latest = p.observe_published(self.root)
        self.assertEqual(latest['snapshot']['counts']['request_attempts'], 40)
        self.evidence.update(held_publication=publication.name, concurrent_transactions=40)

    def test_15_monitor_has_no_active_fallback(self):
        (self.root / 'RESOURCE_ACCOUNTING.json').write_text('invalid JSON', encoding='utf8')
        self.assertEqual(p.observe_published(self.root)['status'], 'NO_PUBLISHED_STATUS')

    def test_16_optimizer_complete_cannot_advance_cursor_alone(self):
        ledger = self.ledger()
        tick = ledger.begin('optimizer', 'u1')
        ledger.complete('optimizer', 'u1', tick)
        self.assertEqual(ledger.cursor, 0)
        with self.assertRaises(RuntimeError):
            ledger.begin('request', 'r1')
        checkpoint = self.root / 'synthetic_checkpoint.bin'
        checkpoint.write_bytes(b'not a torch model')
        commit_update(ledger, 1, checkpoint, dict(cursor=1, checkpoint_sha256=p.sha(checkpoint)))
        self.assertEqual(ledger.cursor, 1)
        rebuilt = self.reopen(ledger)
        self.assertEqual(rebuilt.cursor, 1)

    def test_17_state_replace_failure_does_not_advance_cursor(self):
        ledger = self.ledger()
        tick = ledger.begin('optimizer', 'u1')
        ledger.complete('optimizer', 'u1', tick)
        checkpoint = self.root / 'synthetic_checkpoint.bin'
        checkpoint.write_bytes(b'synthetic')
        def writer(path, value):
            if Path(path).name == 'STATE.json':
                raise p.PersistenceError('atomic_replace', denied())
            p.put(path, value)
        ledger.writer = writer
        with self.assertRaises(p.PersistenceError):
            commit_update(ledger, 1, checkpoint, dict(cursor=1, checkpoint_sha256=p.sha(checkpoint)))
        self.assertEqual(ledger.cursor, 0)
        rebuilt = self.reopen(ledger)
        self.assertEqual(rebuilt.cursor, 0)
        self.assertEqual(rebuilt.counts['optimizer_attempts'], 1)

    def test_18_optimizer_failed_cannot_commit_checkpoint(self):
        ledger = self.ledger()
        ledger.begin('optimizer', 'u1')
        ledger.fail('optimizer', 'u1', ValueError('synthetic computation failure'))
        checkpoint = self.root / 'synthetic_checkpoint.bin'
        checkpoint.write_bytes(b'synthetic')
        with self.assertRaises(ValueError):
            ledger.commit_checkpoint(1, checkpoint, dict(cursor=1, checkpoint_sha256=p.sha(checkpoint)))
        self.assertEqual(ledger.cursor, 0)

    def test_19_origin_budget_is_carried_and_not_reset(self):
        origin = dict(counts=dict(request_attempts=1910, request_completed=1910, request_failed=1,
                                  optimizer_attempts=238, optimizer_completed=238),
                      cursor=238, ids=['synthetic_historical_id'], elapsed_s=2221.312)
        limits = dict(request_attempts=4102, optimizer_attempts=512, wall_s=14400)
        ledger = self.ledger(origin=origin, limits=limits)
        self.request(ledger, 'recovery_0239_teacher_0')
        self.assertEqual(ledger.counts['request_attempts'], 1911)
        self.assertGreaterEqual(ledger.elapsed(), 2221.312)
        ledger.release()
        with self.assertRaises(ValueError):
            self.ledger(origin={}, limits=limits)
        self.evidence.update(historical=1910, new=1, cumulative=1911)

    def test_20_new_budget_enforced(self):
        ledger = self.ledger(limits=dict(request_attempts=1, optimizer_attempts=1, wall_s=60))
        self.request(ledger)
        with self.assertRaises(RuntimeError):
            ledger.begin('request', 'r2')
        self.assertEqual(ledger.counts['request_attempts'], 1)

    def test_21_nonaccess_error_not_retried(self):
        calls = []
        def no_space(a, b):
            calls.append(1)
            raise OSError(28, 'No space')
        with self.assertRaises(p.PersistenceError):
            p.put(self.root / 'x.json', {}, replace=no_space, sleep=lambda _: self.fail('must not sleep'))
        self.assertEqual(len(calls), 1)

    def test_22_no_recovery_permission_from_old_or_draft(self):
        proposal = dict(status='DRAFT_NOT_APPROVED', original_stage_authorization='old')
        for approval, authorization in [(None, None), ({'status': 'APPROVED'}, {'status': 'APPROVED', 'stage': 'init'})]:
            with self.assertRaises(PermissionError):
                require_recovery_authorization(proposal, approval, authorization)

    def test_23_reconcile_rejects_wrong_record_identity(self):
        ledger = self.ledger()
        ledger.begin('request', 'r1')
        path = self.root / 'requests/r1.json'
        p.put(path, dict(request_id='other'))
        with self.assertRaises(ValueError):
            ledger.reconcile_record('r1', p.sha(path))
        self.assertEqual(ledger.pending, {'r1': 'request'})

    def test_24_durable_state_before_journal_does_not_advance_cursor(self):
        ledger = self.ledger()
        tick = ledger.begin('optimizer', 'u1')
        ledger.complete('optimizer', 'u1', tick)
        checkpoint = self.root / 'checkpoint_1.bin'
        checkpoint.write_bytes(b'synthetic')
        state = dict(cursor=1, checkpoint_sha256=p.sha(checkpoint))
        p.put(self.root / 'STATE.json', state)
        rebuilt = self.reopen(ledger)
        self.assertEqual(rebuilt.cursor, 0)
        with self.assertRaises(RuntimeError):
            rebuilt.begin('optimizer', 'u2')
        # Explicitly finish persistence using the same already-written bytes.
        commit_update(rebuilt, 1, checkpoint, state)
        self.assertEqual(rebuilt.cursor, 1)
        self.assertEqual(rebuilt.counts['optimizer_attempts'], 1)

    def test_25_checkpoint_event_before_snapshot_reconstructs_cursor(self):
        ledger = self.ledger()
        tick = ledger.begin('optimizer', 'u1')
        ledger.complete('optimizer', 'u1', tick)
        checkpoint = self.root / 'checkpoint_1.bin'
        checkpoint.write_bytes(b'synthetic')
        def writer(path, value):
            if Path(path).name == 'RESOURCE_ACCOUNTING.json':
                raise p.PersistenceError('atomic_replace', denied())
            p.put(path, value)
        ledger.writer = writer
        with self.assertRaises(p.PersistenceError):
            commit_update(ledger, 1, checkpoint, dict(cursor=1, checkpoint_sha256=p.sha(checkpoint)))
        rebuilt = self.reopen(ledger)
        self.assertEqual(rebuilt.cursor, 1)
        self.assertEqual(rebuilt.counts['optimizer_attempts'], 1)

    def test_26_completed_record_corruption_stops_reconstruction(self):
        ledger = self.ledger()
        self.request(ledger)
        ledger.release()
        (self.root / 'requests/r1.json').write_text('{"corrupt":true}', encoding='utf8')
        with self.assertRaises(ValueError):
            self.ledger()

    def test_27_uncertain_complete_append_reloads_without_fail_append(self):
        ledger = self.ledger()
        tick = ledger.begin('request', 'r1')
        def appender(path, event):
            p.append(path, event)
            raise p.PersistenceError('journal_append_uncertain', OSError('after fsync'))
        ledger.appender = appender
        with self.assertRaises(p.PersistenceError):
            persist_request(ledger, 'r1', dict(request_id='r1'), tick)
        rebuilt = self.reopen(ledger)
        self.assertEqual(rebuilt.counts['request_completed'], 1)
        self.assertEqual(rebuilt.counts['request_failed'], 0)
        self.assertEqual(sum(e['event'] == 'complete' for e in rebuilt.events), 1)

    def test_28_versioned_runtime_outer_except_is_idempotent(self):
        from .runtime import request_record
        ledger = self.ledger()
        tick = ledger.begin('request', 'r1')
        def writer(path, value):
            if Path(path).name == 'RESOURCE_ACCOUNTING.json':
                raise p.PersistenceError('atomic_replace', denied())
            p.put(path, value)
        ledger.writer = writer
        with self.assertRaises(p.PersistenceError) as caught:
            request_record(ledger, 'r1', dict(request_id='r1'), tick)
        ledger.fail('request', 'r1', caught.exception)
        self.assertEqual(sum(e['event'] == 'persistence_failed' for e in ledger.events), 1)
        self.assertEqual(ledger.counts['request_failed'], 0)
        self.assertEqual(ledger.counts['request_attempts'], 1)

    def test_29_pending_publication_is_never_observed(self):
        ledger = self.ledger()
        ledger.publish()
        (self.root / 'published/status_99999999.json.pending').write_text('partial', encoding='utf8')
        self.assertEqual(p.observe_published(self.root)['snapshot']['cursor'], 0)

    def test_30_state_tamper_after_committed_checkpoint_is_rejected(self):
        ledger = self.ledger()
        tick = ledger.begin('optimizer', 'u1')
        ledger.complete('optimizer', 'u1', tick)
        checkpoint = self.root / 'checkpoint_1.bin'
        checkpoint.write_bytes(b'synthetic')
        commit_update(ledger, 1, checkpoint, dict(cursor=1, checkpoint_sha256=p.sha(checkpoint)))
        ledger.release()
        p.put(self.root / 'STATE.json', dict(cursor=9))
        with self.assertRaises(ValueError):
            self.ledger()

    def test_31_restoration_guard_precedes_any_model_access(self):
        from .recovery_adapter import restore_sl238
        with self.assertRaises(PermissionError):
            restore_sl238(None, None, None, {'original_stage_authorization': 'old'}, None, None)

    def test_32_checkpoint_bytes_tamper_does_not_advance_after_restart(self):
        ledger = self.ledger()
        tick = ledger.begin('optimizer', 'u1')
        ledger.complete('optimizer', 'u1', tick)
        checkpoint = self.root / 'checkpoint_1.bin'
        checkpoint.write_bytes(b'synthetic')
        commit_update(ledger, 1, checkpoint, dict(cursor=1, checkpoint_sha256=p.sha(checkpoint)))
        ledger.release()
        checkpoint.write_bytes(b'changed')
        with self.assertRaises(ValueError):
            self.ledger()

    def test_33_recovery_runner_refuses_without_new_approvals(self):
        from .recovery_runner import admit, CANONICAL
        before = CANONICAL.exists()
        with self.assertRaises(PermissionError):
            admit(self.root / 'proposal.json', self.root / 'absent_approval.json', self.root / 'absent_authorization.json')
        self.assertEqual(CANONICAL.exists(), before)
        self.assertFalse(before)

    def test_34_scientific_loss_and_optimizer_remain_identical(self):
        import ast
        folder = Path(__file__).resolve().parent
        original = (folder.parent / 'rc3_relation_drl_20261113/runtime.py').read_text(encoding='utf8')
        revised = (folder / 'runtime.py').read_text(encoding='utf8')
        normalized = revised.replace("f'recovery_v1_{rid}_teacher_{i}'", "f'{rid}_teacher_{i}'")
        def bodies(text):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        old, new = bodies(original), bodies(normalized)
        for name in ('optimizer', 'apply_update', 'sl_update', 'replay', 'setup', 'state_digest', 'set_rng'):
            self.assertEqual(old[name], new[name], name)

    def test_35_two_inactive_checkpoint_slots_survive_prior_overwrite(self):
        ledger = self.ledger()
        for cursor in range(1, 4):
            tick = ledger.begin('optimizer', f'u{cursor}')
            ledger.complete('optimizer', f'u{cursor}', tick)
            path = self.root / f'slot{cursor % 2}.bin'
            path.write_bytes(f'synthetic {cursor}'.encode())
            commit_update(ledger, cursor, path, dict(cursor=cursor, checkpoint_sha256=p.sha(path)))
        rebuilt = self.reopen(ledger)
        self.assertEqual(rebuilt.cursor, 3)
        self.assertEqual(rebuilt.counts['optimizer_completed'], 3)

    def test_36_computation_failure_remains_a_single_charged_failure(self):
        ledger = self.ledger()
        ledger.begin('request', 'r1')
        ledger.fail('request', 'r1', ValueError('synthetic computation error'))
        ledger.fail('request', 'r1', ValueError('same outer handler'))
        self.assertEqual(ledger.counts['request_attempts'], 1)
        self.assertEqual(ledger.counts['request_failed'], 1)
        self.assertEqual(ledger.counts['request_completed'], 0)
        self.assertEqual(ledger.cursor, 0)


def run_suite():
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PersistenceTests)
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return dict(tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                success=result.wasSuccessful(), transcript=stream.getvalue(), evidence=EVIDENCE,
                synthetic_begin_calls=TRANSACTIONS)
