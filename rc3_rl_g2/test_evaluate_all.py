"""Synthetic checks for final evaluation reporting; no actual scene evaluation."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .tools.evaluate_all import paired_statistics, summarize_methods, _export_csv
from .tools import evaluate_all


def row(case, method, J, reward, **extra):
    return dict(case_id=str(case), seed=100+case, mode='regular', method=method,
                status='complete', valid=True, J=J, total_reward=reward,
                work_used=100, complete_s=.2, fallback=False, time_overrun_s=0.,
                work_overrun=0, **extra)


class FinalEvaluationStatisticsTests(unittest.TestCase):
    def test_paired_signs_and_wins(self):
        a = [row(0,'a',8.,3.), row(1,'a',10.,2.), row(2,'a',12.,1.)]
        b = [row(0,'b',10.,2.), row(1,'b',10.,2.), row(2,'b',10.,2.)]
        result = paired_statistics(a,'a',b,'b',bootstrap=100,seed=42)
        for field in ('J','total_reward'):
            item = result['metrics'][field]
            self.assertEqual((item['wins'],item['ties'],item['losses']), (1,1,1))
            self.assertAlmostEqual(item['a_improvement_mean'],0.)
        self.assertEqual(result, paired_statistics(a,'a',list(reversed(b)),'b',bootstrap=100,seed=42))

    def test_missing_failures_and_noncomparable_rewards_are_explicit(self):
        a = [row(0,'a',8.,3.),row(1,'a',9.,2.),row(2,'a',9.,1.)]
        b = [row(0,'b',10.,None),row(1,'b',10.,None)]
        a[1]['status']='complete_with_runtime_error'
        result = paired_statistics(a,'a',b,'b',bootstrap=50)
        self.assertEqual(result['unmatched_a'],1)
        self.assertEqual(result['metrics']['J']['n'],1)
        self.assertEqual(result['metrics']['J']['excluded'],1)
        self.assertEqual(result['metrics']['total_reward']['n'],0)

    def test_duplicate_case_method_rejected(self):
        item = row(0,'a',1.,0.)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            paired_statistics([item,item],'a',[row(0,'b',1.,0.)],'b')

    def test_statistics_include_failures_and_overruns(self):
        good = row(0,'a',1.,0.)
        over = row(1,'a',2.,0.)
        over.update(complete_s=.6,time_overrun_s=.2,work_overrun=100, fallback=True)
        failed = dict(case_id='2',seed=102,mode='regular',method='a',status='failed',valid=False)
        result = summarize_methods([good,over,failed])[0]
        self.assertEqual(result['attempted'],3)
        self.assertEqual(result['complete'],2)
        self.assertEqual(result['failed_or_runtime_error'],1)
        self.assertEqual(result['invalid_plan_count'],1)
        self.assertEqual(result['time_overrun_count'],1)
        self.assertEqual(result['work_overrun_count'],1)
        self.assertEqual(result['fallback_count'],1)
        self.assertAlmostEqual(result['metrics']['complete_s']['median'],.4)
        self.assertAlmostEqual(result['metrics']['complete_s']['p95'],.58)

    def test_original_none_reward_and_work_not_fabricated(self):
        item=row(0,'original_detour',1.,None)
        item['work_used']=None
        result=summarize_methods([item])[0]
        self.assertEqual(result['metrics']['total_reward'],{'n':0})
        self.assertEqual(result['metrics']['work_used'],{'n':0})

    def test_csv_preserves_nested_raw_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rows.csv'
            item=row(0,'a',1.,0.,timing={'menu_s':.01})
            _export_csv(path,[item])
            with path.open(encoding='utf-8-sig',newline='') as handle:
                data=list(csv.DictReader(handle))
            self.assertEqual(data[0]['timing'],'{"menu_s":0.01}')

    def test_summary_pipeline_keeps_stage_latency_out_of_matched_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'outputs/final_evaluation'
            cases=[dict(case_id=i,seed=100+i,mode='regular') for i in range(2)]
            protocol=dict(training_seed=9,config={},splits={'validation':cases})
            evaluate_all._write(root/'outputs/protocol.json',protocol)
            controls=[row(i,method,10.,0.) for i in range(2)
                      for method in ('bc_frozen','bc_continued_512','bc_continued_1024','bc_continued_2048')]
            final=[row(i,method,9. if method=='ppo_final' else 10.,.1 if method=='ppo_final' else 0.)
                   for i in range(2) for method in ('ppo_final','bc_frozen','bc_continued_2048')]
            for name,rows in [('controls_fixed_work',controls),('final_fixed_work',final),('final_soft_time_0400ms',final)]:
                evaluate_all._write(output/name/'results.json',rows)
            for actions,_ in evaluate_all.PAIRS:
                stage=root/f'outputs/validation/ppo_{actions}'
                evaluate_all._write(stage/'results.json',[row(i,f'ppo_{actions}',9.,.1) for i in range(2)])
                evaluate_all._write(stage/'request.json',dict(cases=cases,config={},time_limit_s=None))
            with patch.object(evaluate_all,'ROOT',root):
                summary=evaluate_all.build_summary(output,protocol,bootstrap=100)
            self.assertEqual(len(summary['checkpoint_stage_comparisons']),3)
            for stage in summary['checkpoint_stage_comparisons']:
                self.assertEqual(len(stage['comparisons']),3)
                for comparison in stage['comparisons']:
                    self.assertNotIn('complete_s',comparison['metrics'])
                self.assertIn('regular',stage['comparisons_by_mode'])
            self.assertIn('complete_s',summary['datasets']['final_fixed_work']['ppo_vs_methods'][0]['metrics'])
            self.assertTrue((output/'rows.csv').exists())


if __name__=='__main__':
    unittest.main()
