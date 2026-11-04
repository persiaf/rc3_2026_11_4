"""Gate/selection integration checks with JSON fixtures and mocks; zero env calls."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import evaluate_g21 as evaluation


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


class EvaluationGuards(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)
        self.protocol_path=self.root/'protocol.json'
        protocol=dict(schema='rc3-g21-offline-gated-v1',toy=dict(updates=512),
            sampling=dict(updates=2048),gain=dict(updates=2048),
            opportunity_gate=dict(initial_positive_full_known_min_fraction=.7,
                extra_positive_full_known_min_fraction=.5,
                initial_clear_stop_full_stop_min_fraction=.75,initial_root_full_unknown_max_fraction=.2))
        write(self.protocol_path,protocol)
        self.digest=evaluation._sha(self.protocol_path)
        self.protocol_path.with_suffix('.sha256').write_text(self.digest)
        self.toy_path=self.root/'toy16/receipt.json'
        write(self.toy_path,dict(completed=True,updates=512,protocol_sha256=self.digest,
            gate=dict(passed=True,composition_valid=True,evidence_near_best=16,positive_continue=8)))
        self.checkpoint=self.root/'sampling_root_half/model.pt'
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(b'synthetic checkpoint; never loaded')
        def group(metric,mean):return dict(n=10,metrics={metric:dict(n=10,mean=mean)})
        self.groups=dict(positive_roots=group('full_known_positive_continue',.8),
            clear_stop_roots=group('full_greedy_stop',.9),roots=group('full_greedy_unknown',.1))
        self.receipt_path=self.checkpoint.parent/'receipt.json'
        write(self.receipt_path,dict(completed=True,stage='sampling',updates=2048,
            protocol_sha256=self.digest,checkpoint_sha256=evaluation._sha(self.checkpoint),final_diagnostic=self.groups))
        self.final_path=self.checkpoint.parent/'diagnostic_2048.json'
        write(self.final_path,dict(groups=self.groups))
        self.extra_path=self.checkpoint.parent/'extra_existing_teacher_diagnostic.json'
        write(self.extra_path,dict(groups={'positive_roots':group('full_known_positive_continue',.6)}))
        self.gate_path=self.root/'opportunity_gate.json'
        self.gate=dict(passed=True,protocol_sha256=self.digest,checkpoint_path=str(self.checkpoint),
            checkpoint_sha256=evaluation._sha(self.checkpoint),receipt_sha256=evaluation._sha(self.receipt_path),
            final_diagnostic_sha256=evaluation._sha(self.final_path),extra_diagnostic_sha256=evaluation._sha(self.extra_path))
        write(self.gate_path,self.gate)

    def tearDown(self):
        self.temporary.cleanup()

    def require(self):
        return evaluation.require_opportunity_gate(self.protocol_path,self.gate_path)

    def test_complete_offline_receipts_allow_gate_read(self):
        authorized=self.require()
        self.assertEqual(authorized['metrics']['initial_positive'],.8)

    def test_missing_root_authorization_fails_before_env_or_model_load(self):
        self.gate_path.unlink()
        with patch.object(evaluation,'SearchEnvG2') as env,patch.object(evaluation,'prepare') as prepare:
            with self.assertRaises(FileNotFoundError):
                evaluation.evaluate_methods([{}],{'stop':None},self.root/'unused',
                    protocol_path=self.protocol_path,gate_path=self.gate_path)
            env.assert_not_called();prepare.assert_not_called()
        self.assertFalse((self.root/'unused').exists())

    def test_false_root_authorization_rejected(self):
        self.gate['passed']=False;write(self.gate_path,self.gate)
        with self.assertRaises(ValueError):self.require()

    def test_intermediate_toy_update_rejected(self):
        toy=evaluation._read(self.toy_path);toy['updates']=256;write(self.toy_path,toy)
        with self.assertRaises(ValueError):self.require()

    def test_toy_boolean_without_count_evidence_rejected(self):
        toy=evaluation._read(self.toy_path);toy['gate']['positive_continue']=6;write(self.toy_path,toy)
        with self.assertRaises(ValueError):self.require()

    def test_toy_checkpoint_is_not_deployment(self):
        self.gate['checkpoint_path']=str(self.toy_path);write(self.gate_path,self.gate)
        with self.assertRaisesRegex(ValueError,'toy checkpoint'):self.require()

    def test_changed_checkpoint_rejected(self):
        self.checkpoint.write_bytes(b'changed')
        with self.assertRaises(ValueError):self.require()

    def test_failing_full_action_opportunity_rejected(self):
        receipt=evaluation._read(self.receipt_path)
        receipt['final_diagnostic']['positive_roots']['metrics']['full_known_positive_continue']['mean']=.6
        write(self.receipt_path,receipt);write(self.final_path,dict(groups=receipt['final_diagnostic']))
        # Even a new attestation cannot override the actual numerical threshold.
        self.gate['receipt_sha256']=evaluation._sha(self.receipt_path)
        self.gate['final_diagnostic_sha256']=evaluation._sha(self.final_path)
        write(self.gate_path,self.gate)
        with self.assertRaisesRegex(ValueError,'thresholds failed'):self.require()

    def test_final_diagnostic_mismatch_rejected(self):
        write(self.final_path,dict(groups={}))
        with self.assertRaises(ValueError):self.require()

    def test_actual_gate_diagnostic_hash_key_detects_tampering(self):
        self.gate['diagnostic_sha256']=self.gate.pop('final_diagnostic_sha256')
        write(self.gate_path,self.gate)
        self.assertEqual(self.require()['metrics']['initial_positive'],.8)
        self.gate['diagnostic_sha256']='0'*64;write(self.gate_path,self.gate)
        with self.assertRaisesRegex(ValueError,'diagnostic changed'):self.require()

    def test_diagnostic_hash_aliases_must_agree(self):
        self.gate['diagnostic_sha256']='0'*64;write(self.gate_path,self.gate)
        with self.assertRaisesRegex(ValueError,'aliases disagree'):self.require()

    def test_outside_g21_cache_rejected_without_prepare(self):
        with patch('rc3_rl_g2.prepared_cache.prepare_case') as prepare:
            with self.assertRaises(ValueError):
                evaluation._prepare_input({},evaluation.ROOT.parent/'rc3_rl_g2/outputs/cache','cold')
            prepare.assert_not_called()

    def test_one_step_control_retains_root_choice_and_caps_inference(self):
        obs=dict(actions=[dict(kind='improve'),dict(kind='stop')],mask=[True,True])
        self.assertIsNone(evaluation._one_step_stop(obs,0,True))
        self.assertIsNone(evaluation._one_step_stop(obs,3,False))
        self.assertEqual(evaluation._one_step_stop(obs,1,True),1)

    def test_rule_grid_ties_use_work_then_cap_then_patience(self):
        rows=[dict(mean_return=1.,mean_work=10,settings=dict(max_proposals=4,patience=2)),
              dict(mean_return=1.+5e-13,mean_work=11,settings=dict(max_proposals=1,patience=1)),
              dict(mean_return=1.,mean_work=10,settings=dict(max_proposals=2,patience=1))]
        self.assertEqual(evaluation._winner(rows)['settings'],dict(max_proposals=2,patience=1))


if __name__=='__main__':unittest.main()
