"""Fresh-plan timing, complete rewards, and preregistered rule selection."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import torch

from rc3_rl_research.types import SearchConfig
from .actor_critic import ActorCritic
from .evaluate_g2 import evaluate_methods, select_ils, _run_original


def read_evidence(row):
    with gzip.open(row['evidence_path'],'rt',encoding='utf-8') as handle:return json.load(handle)


class SlowModel(ActorCritic):
    def act(self,observations,deterministic=False):
        time.sleep(.25)
        return super().act(observations,deterministic)


class EvaluationContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(77)
        base=Path('rc3_rl_g2/outputs/evaluation_tests');base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base)
        self.output=Path(self.temp.name)
        self.case=dict(case_id=0,nx=2,ny=2,seed=2026110331,mode='regular')
        self.config=SearchConfig(horizon=3)

    def tearDown(self):self.temp.cleanup()

    def test_original_fixed3_and_menu_stop_have_distinct_work_contracts(self):
        rows=evaluate_methods([self.case],{'original_fixed3':None,'environment_stop':None},self.output,config=self.config)
        fixed,stop=rows
        self.assertTrue(all(r['status']=='complete' and r['valid'] for r in rows))
        self.assertAlmostEqual(fixed['J'],stop['J'])
        self.assertIsNone(fixed['return']);self.assertIsNone(fixed['work_used'])
        self.assertGreater(stop['work_used'],0);self.assertLess(stop['return'],0)
        self.assertEqual(fixed['proposal_calls'],0);self.assertTrue(stop['stop_at_root'])
        self.assertEqual(read_evidence(fixed)['wrapper']['preparation_deducted_from_deadline'],True)

    def test_shared_methods_return_reward_and_output_consistent_evidence(self):
        methods={m:None for m in ('detour_memory','ils','random','stop')}
        rows=evaluate_methods([self.case],methods,self.output,config=self.config)
        for row in rows:
            self.assertEqual(row['status'],'complete',row)
            expected=100*(row['anchor_J']-row['J'])/max(row['anchor_J'],1e-8)-.01*row['work_used']/10000
            self.assertAlmostEqual(row['return'],expected)
            self.assertLessEqual(row['J'],row['anchor_J']+1e-8)
            self.assertLessEqual(row['proposal_calls'],row['steps'])
            evidence=read_evidence(row)
            ledger=sum(t['reward'] for t in evidence['state']['transitions'])+sum(e['reward'] for e in evidence['state']['events'])
            self.assertAlmostEqual(ledger,row['return'])
            self.assertAlmostEqual(sum(x['reward'] for x in evidence['actor_records']),row['return'])
            timing=row['timing'];components=sum(v for k,v in timing.items() if k not in ('complete_s','residual_s'))
            self.assertAlmostEqual(components+timing['residual_s'],row['complete_s'])
            self.assertGreaterEqual(timing['residual_s'],-1e-6)

    def test_zero_full_deadline_keeps_anchor_without_free_actor_sample(self):
        rows=evaluate_methods([self.case],{'original_detour':None,'detour_memory':None},self.output,
                              config=self.config,time_limit_s=0.)
        for row in rows:
            self.assertEqual(row['status'],'complete');self.assertEqual(row['proposal_calls'],0)
            self.assertAlmostEqual(row['J'],row['anchor_J'])
            self.assertGreater(row['time_overrun_s'],0.)
        shared=rows[1];self.assertEqual(shared['steps'],0);self.assertEqual(shared['work_used'],0)
        self.assertTrue(shared['metrics_only_zero_action']);self.assertIsNone(shared['actor_reward_sum'])
        old=read_evidence(rows[0]);self.assertEqual(old['wrapper']['runtime_time_limit_s'],0.)

    def test_cpu_evaluation_does_not_mutate_training_model(self):
        model=ActorCritic(kind='graph',message_steps=2).train()
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        rows=evaluate_methods([self.case],{'learned':model},self.output,config=self.config)
        self.assertEqual(rows[0]['status'],'complete')
        self.assertTrue(model.training)
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))
        evidence=read_evidence(rows[0])
        self.assertTrue(evidence['policy_samples'])
        self.assertTrue(all(s['executed'] for s in evidence['policy_samples']))
        self.assertIsNone(rows[0]['cold_model_load_s'])

    def test_deadline_after_selection_charges_menu_but_no_fabricated_action(self):
        model=SlowModel(kind='mlp')
        rows=evaluate_methods([self.case],{'slow':model},self.output,config=self.config,time_limit_s=.2)
        row=rows[0];self.assertEqual(row['status'],'complete');self.assertEqual(row['steps'],0)
        self.assertGreater(row['work_used'],0)
        evidence=read_evidence(row)
        self.assertEqual(len(evidence['policy_samples']),1)
        self.assertFalse(evidence['policy_samples'][0]['executed'])
        self.assertEqual(evidence['actor_records'],[])
        self.assertAlmostEqual(row['return'],sum(e['reward'] for e in evidence['state']['events']))
        self.assertTrue(row['metrics_only_zero_action'])

    def test_rule_grid_registered_before_running_and_selected_without_validation(self):
        path=self.output/'protocol.json'
        protocol=dict(splits=dict(rule_dev=[self.case]),config=asdict(self.config))
        path.write_text(json.dumps(protocol),encoding='utf-8')
        root=self.output/'selection'
        def fake(cases,methods,output,**kwargs):
            self.assertTrue((root/'selection_protocol.json').exists())
            s=kwargs['policy_kwargs']['ils']
            return [dict(status='complete',total_reward=float(s['stagnation_limit']),work_used=s['max_perturbations'])]
        with patch('rc3_rl_g2.evaluate_g2.evaluate_methods',side_effect=fake) as evaluator:
            selected=select_ils(path,root)
        self.assertEqual(evaluator.call_count,4)
        self.assertEqual(selected['selected'],dict(stagnation_limit=3,max_perturbations=1))
        self.assertFalse(selected['validation_used']);self.assertTrue(selected['frozen'])

    def test_invalid_budget_or_completed_destination_rejected(self):
        with self.assertRaises(ValueError):evaluate_methods([self.case],{'stop':None},self.output,time_limit_s=float('nan'))
        evaluate_methods([self.case],{'stop':None},self.output,config=self.config)
        with self.assertRaises(ValueError):evaluate_methods([self.case],{'stop':None},self.output,config=self.config)


from dataclasses import asdict
if __name__=='__main__':unittest.main()
