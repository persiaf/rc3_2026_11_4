"""Pure analysis and durable resource-boundary tests; no planning/optimizer."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .analysis import parent_contrast
from .storage import ROOT, Ledger, read


class ExecutionBoundaryTests(unittest.TestCase):
    def test_parent_is_independent_unit_not_window(self):
        rows=[]
        for parent,cases,a,b in [('A',8,9.,10.),('B',1,6.,10.)]:
            for i in range(cases):
                for method,cost in [('ON',a),('OFF',b)]:
                    rows.append(dict(physical_key=parent,case_id=f'{parent}{i}',method=method,metrics={'J':cost}))
        result=parent_contrast(rows,'ON','OFF',bootstrap=100)
        self.assertEqual(result['physical_parents'],2)
        self.assertAlmostEqual(result['point'],.25)

    def test_incomplete_pairs_and_duplicate_repeats_fail_closed(self):
        row=dict(physical_key='A',case_id='x',method='ON',metrics={'J':1.})
        with self.assertRaises(ValueError):
            parent_contrast([row],'ON','OFF',bootstrap=10)
        with self.assertRaises(ValueError):
            parent_contrast([row,row],'ON','OFF',bootstrap=10)

    def test_budget_is_debited_and_not_reset_on_clean_resume(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)/'ledger'
            ledger=Ledger(folder,{'source':'synthetic'},{'wall_s':60,'requests':1})
            ledger.debit({'requests':1},request_id='one')
            ledger.close('clean')
            ledger=Ledger(folder,{'source':'synthetic'},{'wall_s':60,'requests':1},resume=True)
            try:
                self.assertEqual(ledger.counts['requests'],1)
                with self.assertRaises(RuntimeError):
                    ledger.debit({'requests':1},request_id='two')
            finally:
                ledger.close('clean')

    def test_caps_and_source_cannot_change_on_resume(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)/'ledger'
            ledger=Ledger(folder,{'source':'synthetic'},{'wall_s':60,'requests':1})
            ledger.close('clean')
            with self.assertRaises(RuntimeError):
                Ledger(folder,{'source':'synthetic'},{'wall_s':60,'requests':2},resume=True)
            with self.assertRaises(RuntimeError):
                Ledger(folder,{'source':'other'}, {'wall_s':60,'requests':1},resume=True)

    def test_artifact_hash_cannot_overwrite_journal_hash(self):
        with TemporaryDirectory() as tmp:
            ledger=Ledger(Path(tmp)/'ledger',{'source':'synthetic'},{'wall_s':60,'requests':1})
            try:
                with self.assertRaises(ValueError):
                    ledger.emit({'event':'artifact','sha256':'a'*64})
                ledger.emit({'event':'artifact','artifact_sha256':'a'*64})
            finally:
                ledger.close('clean')

    def test_interrupted_clock_cannot_restart(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)/'ledger'
            ledger=Ledger(folder,{'source':'synthetic'},{'wall_s':60,'requests':1})
            ledger.release()  # Simulated interruption: no session_close receipt.
            with self.assertRaises(RuntimeError):
                Ledger(folder,{'source':'synthetic'},{'wall_s':60,'requests':1},resume=True)

    def test_real_runner_checkpoint_reconciles_full_durable_cursor(self):
        from .runner import checkpoint, restore
        class FakeTrainer:
            def state_dict(self, metadata):
                return {'metadata': metadata, 'tensor': [1,2,3]}
            def load_state_dict(self, payload, expected_metadata):
                if payload['metadata'] != expected_metadata:
                    raise ValueError('Metadata differs')
                self.loaded = payload['tensor']
        with TemporaryDirectory(dir=ROOT/'rc3_conservative_refinement') as tmp:
            folder=Path(tmp)/'ledger'
            caps={'wall_s':60,'requests':2}
            ledger=Ledger(folder,{'source':'synthetic'},caps)
            ledger.debit({'requests':1},request_id='one')
            checkpoint(ledger,{'ON':FakeTrainer()},{'round':1},{'ON':{'mean':2.}})
            ledger.close('clean')
            ledger=Ledger(folder,{'source':'synthetic'},caps,resume=True)
            restored=FakeTrainer()
            try:
                progress,constants=restore(ledger,{'ON':restored})
                self.assertEqual(progress,{'round':1})
                self.assertEqual(restored.loaded,[1,2,3])
                self.assertEqual(ledger.counts['requests'],1)
                self.assertEqual(constants,{'ON':{'mean':2.}})
            finally:
                ledger.close('clean')

    def test_real_runner_refuses_dirty_request_after_checkpoint(self):
        from .runner import checkpoint, restore
        class FakeTrainer:
            def state_dict(self, metadata):
                return {'metadata':metadata}
        with TemporaryDirectory(dir=ROOT/'rc3_conservative_refinement') as tmp:
            folder=Path(tmp)/'ledger';caps={'wall_s':60,'requests':2}
            ledger=Ledger(folder,{'source':'synthetic'},caps)
            checkpoint(ledger,{'ON':FakeTrainer()},{'round':0},{})
            ledger.debit({'requests':1},request_id='dirty')
            ledger.close('failed')
            ledger=Ledger(folder,{'source':'synthetic'},caps,resume=True)
            try:
                with self.assertRaises(ValueError):
                    restore(ledger,{'ON':FakeTrainer()})
            finally:
                ledger.close('blocked')


if __name__=='__main__':
    unittest.main()
