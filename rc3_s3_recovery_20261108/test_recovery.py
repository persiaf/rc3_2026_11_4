"""Synthetic persistence/state/statistics tests, plus read-only legacy evidence."""
import copy
import contextlib
import json
import errno
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from rc3_s3_evaluator.common import read,write_new,digest,identity,SEEDS
from .storage import Journal,Lease,WriterBusy,commit_anchor,verify_supervisor_lease
from .authority import inspect_legacy,baseline_for,FAILED_ID,initialize,validate_activation
from .runner import finish_session,clock_used,run_seed
from .analysis import analyze,analyze_saved


def qs():return [dict(id='q'+str(i),seed=SEEDS[0],case_id='fixture',kind='cold',method='fixture',repeat=i) for i in range(4)]
def base():return dict(debits=1,success=0,infra_failure=1,terminal_ids=['q0'],failure_ids=['q0'],wall_carry_s=592,old_exact_wall_s=None)
def fake(q):return dict(request=q,valid=True,plan=dict(validation=dict(valid_model_plan=True)))


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack=contextlib.ExitStack();cls.observed={}
        names=['rc3_fast.scenes.make_complex_scene','rc3_constructive.problem.resample_tasks',
            'rc3_fast.solver.vectorized_feasibility','rc3_baseline.baseline.navigation',
            'rc3_global_local.policy.GlobalLocalPolicy.__init__','rc3_s3_evaluator.pipelines.evaluate',
            'rc3_global_local_impl_review.endpoint_baselines.run_endpoint','rc3_global_local.terminal.finish',
            'torch.optim.Optimizer.__init__']
        def forbidden(name):
            def call(*a,**kw):
                cls.observed[name]+=1;raise AssertionError('Zero-science regression forbids '+name)
            return call
        for name in names:
            cls.observed[name]=0;cls.stack.enter_context(patch(name,side_effect=forbidden(name)))

    @classmethod
    def tearDownClass(cls):
        cls.stack.close();print('OBSERVED_ZERO_SCIENCE_COUNTERS='+json.dumps(cls.observed,sort_keys=True))
        if any(cls.observed.values()):raise AssertionError('Scientific function attempted in engineering regression')

    def test_01_transient_replace_same_event(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'j';actual=os.replace;calls=[]
            def replace(a,b):
                calls.append(1)
                if len(calls)<3:raise PermissionError(errno.EACCES,'injected sharing denial')
                return actual(a,b)
            with Journal(p,{},qs(),base(),retry_delay=0) as j:
                with patch('rc3_s3_recovery_20261108.storage.os.replace',side_effect=replace):j.append(dict(event='operation_start',operation='fixture'))
                self.assertEqual(len(calls),3);self.assertEqual(len(j.events),1);self.assertEqual(len(j.path.read_bytes().splitlines()),1)
                self.assertFalse(j.pending.exists());self.assertEqual(read(j.head)['sequence'],1)
            with Journal(p,{},qs(),base()) as j:self.assertEqual(len(j.events),1)

    def test_02_persistent_denial_retains_fault(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'j'
            with Journal(p,{},qs(),base(),retry_attempts=3,retry_delay=0) as j:
                with patch('rc3_s3_recovery_20261108.storage.os.replace',side_effect=PermissionError(errno.EACCES,'persistent')) as replace:
                    with self.assertRaises(PermissionError):j.append(dict(event='operation_start',operation='fixture'))
                self.assertEqual(replace.call_count,3);self.assertFalse(j.healthy)
                self.assertEqual(len(j.events),1);self.assertTrue(j.pending.exists());saved=digest(j.pending)
                self.assertTrue(list((p/'faults').glob('*.json')))
                with self.assertRaises(ValueError):j.append(dict(event='operation_start',operation='free retry'))
            with self.assertRaises(ValueError):Journal(p,{},qs(),base())
            self.assertEqual(digest(p/'HEAD.pending'),saved)

    def test_03_single_writer_and_pending_identity(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'j'
            with Journal(p,{},qs(),base()) as j:
                with self.assertRaises(WriterBusy):Journal(p,{},qs(),base())
                j.lease.assert_owner()
            with Journal(p,{},qs(),base()) as j:j.lease.assert_owner()
            pending=p/'synthetic.pending';anchor=dict(sequence=1,sha256='a',journal_identity='j',writer_identity='w')
            write_new(pending,dict(anchor,writer_identity='foreign'))
            with self.assertRaises(ValueError):commit_anchor(pending,p/'synthetic.head',anchor,None,delay=0)

    def test_04_chain_and_orphan_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'j'
            with Journal(p,{},qs(),base()) as j:j.append(dict(event='operation_start',operation='fixture'))
            raw=(p/'journal.jsonl').read_bytes();(p/'journal.jsonl').write_bytes(raw.replace(b'fixture',b'tamper!'))
            with self.assertRaises(ValueError):Journal(p,{},qs(),base())
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'j'
            with Journal(p,{},qs(),base()):pass
            write_new(p/'results'/'q1.json',fake(qs()[1]))
            with self.assertRaises(ValueError):Journal(p,{},qs(),base())

    def test_05_disposed_failure_not_replayed_and_states(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'j'
            with Journal(p,{},qs(),base()) as j:
                self.assertEqual(j.request_state('q0'),'INFRASTRUCTURE_FAILED_DISPOSED')
                self.assertEqual(j.request_state('q1'),'UNEXECUTED');j.clean()
                with self.assertRaises(ValueError):j.debit(qs()[0])
                j.debit(qs()[1]);self.assertEqual(j.request_state('q1'),'RUNNING')
                write_new(p/'results'/'q1.json',fake(qs()[1]));j.complete(qs()[1],.1)
                self.assertEqual(j.request_state('q1'),'COMPLETED');j.clean()
            with Journal(p,{},qs(),base()) as j:
                self.assertEqual(len(j.starts),1);j.clean()
                with self.assertRaises(ValueError):j.debit(qs()[1])
            reset=base();reset['wall_carry_s']=0
            with self.assertRaises(ValueError):Journal(p,{},qs(),reset)

    def test_06_new_failure_still_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'j'
            with Journal(p,{},qs(),base()) as j:
                j.debit(qs()[1]);j.append(dict(event='request_error',id='q1',error='fixture'))
            with Journal(p,{},qs(),base()) as j:
                self.assertEqual(j.request_state('q1'),'INFRASTRUCTURE_FAILED')
                with self.assertRaises(ValueError):j.clean()

    def test_07_parent_close_failure_has_external_clock(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'j';token='session'
            write_new(p/'sessions'/token/'OPEN.json',dict(token=token,carried_wall_s=592))
            def broken(elapsed):raise ValueError('injected journal constructor/closure failure')
            final=finish_session(p,token,time.perf_counter()-.1,'FAILED','original worker failure',broken)
            self.assertEqual(final['status'],'FAILED');self.assertGreater(final['elapsed_s'],.1)
            self.assertTrue((p/'sessions'/token/'FINAL_SEAL.json').exists())
            with self.assertRaises(ValueError):clock_used(p,[dict(event='session_open',token=token)],base())

    def test_08_exact_clock_carry_and_missing_clock_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'j';token='session';op=p/'sessions'/token/'OPEN.json'
            write_new(op,dict(token=token,carried_wall_s=592))
            final=finish_session(p,token,time.perf_counter()-.1,'CLOSED_BOUNDARY',None,lambda elapsed:None)
            events=[dict(event='session_open',token=token)]
            self.assertAlmostEqual(clock_used(p,events,base()),592+final['elapsed_s'])
            with self.assertRaises(ValueError):clock_used(p,[],base())
            (p/'sessions'/token/'FINAL.json').unlink()
            with self.assertRaises(FileNotFoundError):clock_used(p,events,base())

    def test_09_legacy_exact_idempotent_read_only(self):
        a=inspect_legacy();b=inspect_legacy()
        self.assertEqual(identity(a),identity(b));self.assertEqual(a['first_seed_debits'],163)
        self.assertEqual(a['first_seed_success'],162);self.assertEqual(set(a['errors']),{FAILED_ID})
        self.assertEqual(baseline_for(a,SEEDS[0])['wall_carry_s'],592)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad_index.json';write_new(p,dict(artifacts=[]))
            with self.assertRaises(ValueError):inspect_legacy(index_path=p)

    def test_10_full_statistics_matches_frozen_reference(self):
        from rc3_s3_evaluator.test_evaluator import fixture_rows
        from rc3_s3_evaluator.statistics import analyze as old
        cases,rows=fixture_rows();ref=old(rows,cases);new=analyze(rows,cases)
        self.assertEqual(new['independent_RL_pass'],ref['independent_RL_pass']);self.assertEqual(new['deployment_pass'],ref['deployment_pass'])
        for a,b in zip(new['quality_cells'],ref['quality_cells']):self.assertEqual(a['controls'],b['controls'])
        for a,b in zip(new['deployment_cells'],ref['deployment_cells']):
            for k in ('degradation','median_ratio','p95_ratio','Fixed3_dominates','passed','timing'):self.assertEqual(a[k],b[k])

    def test_11_missing_timing_not_filled_quality_independent(self):
        from rc3_s3_evaluator.test_evaluator import fixture_rows
        cases,rows=fixture_rows()
        bad=next(r for r in rows if r['group']=='seen' and r['size']==[6,6] and r['request']['method']=='Fixed3_endpoint_adapter' and r['request']['kind']=='cold' and r['request']['repeat']==1)
        reduced=[r for r in rows if r is not bad];out=analyze(reduced,cases)
        self.assertEqual(out['missing_ids'],[bad['request']['id']]);self.assertTrue(out['independent_RL_pass'])
        self.assertEqual(out['quality_status'],'EVALUABLE');self.assertIsNone(out['deployment_pass']);self.assertIsNone(out['combined_pass'])
        affected=[c for c in out['deployment_cells'] if c['status']=='INCOMPLETE'];self.assertEqual(len(affected),1)
        self.assertIsNone(affected[0]['passed']);self.assertEqual(affected[0]['missing_ids'],[bad['request']['id']])
        self.assertEqual(affected[0]['timing']['Fixed3_endpoint_adapter']['n'],107)
        qmissing=next(r for r in reduced if r['request']['kind']=='quality')
        out=analyze([r for r in reduced if r is not qmissing],cases)
        self.assertIsNone(out['independent_RL_pass']);self.assertEqual(len([c for c in out['quality_cells'] if c['status']=='INCOMPLETE']),1)

    def test_12_activation_is_idempotent_not_directory_reset(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cont=root/'continuations'/'persistence_r1'
            with patch('rc3_s3_recovery_20261108.authority.RUN',root),patch('rc3_s3_recovery_20261108.authority.CONT',cont):
                initialize({'fixture':True},{'old':'immutable'});before=digest(root/'PERSISTENCE_RECOVERY_ACTIVATION.json')
                initialize({'fixture':True},{'old':'immutable'});self.assertEqual(digest(root/'PERSISTENCE_RECOVERY_ACTIVATION.json'),before)
                cont.rename(cont.with_name('renamed'))
                with self.assertRaises(ValueError):validate_activation(root)
                with self.assertRaises(ValueError):initialize({'fixture':True},{'old':'immutable'})
                self.assertFalse(cont.exists())

    def test_13_session_open_failure_gets_per_seed_final_clock(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cont=root/'continuation';seed=SEEDS[0]
            def factory(*a,**kw):return Journal(*a,**kw,retry_attempts=2,retry_delay=0)
            with patch('rc3_s3_recovery_20261108.runner.CONT',cont),patch('rc3_s3_recovery_20261108.runner.HOME',root/'home'), \
                 patch('rc3_s3_recovery_20261108.runner.Inputs',return_value=SimpleNamespace(cases=[])), \
                 patch('rc3_s3_recovery_20261108.runner.grid',return_value=qs()), \
                 patch('rc3_s3_recovery_20261108.runner.baseline_for',return_value=base()), \
                 patch('rc3_s3_recovery_20261108.runner.Journal',side_effect=factory), \
                 patch('rc3_s3_recovery_20261108.storage.os.replace',side_effect=PermissionError(errno.EACCES,'session_open denial')), \
                 patch('rc3_s3_recovery_20261108.runner.subprocess.Popen') as launch:
                with self.assertRaises(ValueError):run_seed(seed,time.perf_counter(),None,{},dict(fixture=True))
                launch.assert_not_called()
            finals=list((cont/str(seed)/'sessions').glob('*/FINAL.json'))
            self.assertEqual(len(finals),1);self.assertEqual(read(finals[0])['status'],'FAILED')
            self.assertGreater(read(finals[0])['elapsed_s'],0);self.assertTrue((cont/str(seed)/'HEAD.pending').exists())

    def test_14_normal_event_bytes_and_quality_only(self):
        from rc3_s3_evaluator.journal import Journal as OldJournal
        from rc3_s3_evaluator.test_evaluator import fixture_rows
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);empty=base();empty.update(debits=0,success=0,infra_failure=0,terminal_ids=[],failure_ids=[],wall_carry_s=0)
            old=OldJournal(p/'old',{},qs(),4)
            with Journal(p/'new',{},qs(),empty) as new:
                for e in [dict(event='operation_start',operation='fixture'),dict(event='operation_end',operation='fixture',elapsed_s=.01,counts={})]:
                    old.append(e);new.append(e)
                self.assertEqual(old.path.read_bytes(),new.path.read_bytes())
        cases,rows=fixture_rows();out=analyze([r for r in rows if r['request']['kind']=='quality'],cases)
        self.assertTrue(out['independent_RL_pass']);self.assertIsNone(out['deployment_pass']);self.assertEqual(len(out['deployment_cells']),12)

    def test_15_completed_analysis_is_idempotent_bound_to_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);binding=dict(fixture=True)
            write_new(p/'SCIENTIFIC_ANALYSIS.json',dict(binding=binding,input_result_identity=identity([]),missing_ids=[FAILED_ID]))
            before=digest(p/'SCIENTIFIC_ANALYSIS.json')
            with patch('rc3_s3_recovery_20261108.analysis.CONT',p),patch('rc3_s3_recovery_20261108.analysis.admit',return_value=(None,{},binding)), \
                 patch('rc3_s3_recovery_20261108.analysis.saved_rows',return_value=[]):
                self.assertEqual(analyze_saved(),analyze_saved())
            self.assertEqual(digest(p/'SCIENTIFIC_ANALYSIS.json'),before)
            with patch('rc3_s3_recovery_20261108.analysis.CONT',p),patch('rc3_s3_recovery_20261108.analysis.admit',return_value=(None,{},binding)), \
                 patch('rc3_s3_recovery_20261108.analysis.saved_rows',return_value=[dict(changed=True)]):
                with self.assertRaises(ValueError):analyze_saved()

    def test_16_actual_venv_launcher_uses_held_supervisor_lease(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'SUPERVISOR.lease';scope='synthetic-supervisor'
            code='import sys,json,os;from rc3_s3_recovery_20261108.storage import verify_supervisor_lease; o=verify_supervisor_lease(sys.argv[1],int(sys.argv[2]),sys.argv[3]); print(json.dumps(dict(owner=o,ppid=os.getppid())))'
            with Lease(path,scope):
                child=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(path),str(os.getpid()),scope],capture_output=True,timeout=20)
                self.assertEqual(child.returncode,0,child.stderr.decode('utf8'))
                self.assertEqual(json.loads(child.stdout)['owner']['pid'],os.getpid())
                with self.assertRaises(ValueError):verify_supervisor_lease(path,os.getpid()+1,scope)
            child=subprocess.run([sys.executable,'-B','-X','utf8','-c',code,str(path),str(os.getpid()),scope],capture_output=True,timeout=20)
            self.assertNotEqual(child.returncode,0)
            self.assertIn('Supervisor lease is not held',child.stderr.decode('utf8'))

    def test_17_acknowledged_zero_request_clock_is_not_refunded(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);token='zero-request'
            write_new(p/'sessions'/token/'OPEN.json',dict(token=token,carried_wall_s=592))
            final=finish_session(p,token,time.perf_counter()-.1,'FAILED','dispatch-only',lambda t:None)
            rec={token:dict(final_sha256=digest(p/'sessions'/token/'FINAL.json'),charged_elapsed_s=final['elapsed_s'])}
            events=[dict(event='session_open',token=token),dict(event='session_close',token=token)]
            with patch('rc3_s3_recovery_20261108.runner.disposed_dispatch_sessions',return_value=rec):
                self.assertAlmostEqual(clock_used(p,events,base()),592+final['elapsed_s'])
                events.insert(1,dict(event='request_start',request=qs()[1]))
                with self.assertRaises(ValueError):clock_used(p,events,base())


if __name__=='__main__':unittest.main(verbosity=2)
