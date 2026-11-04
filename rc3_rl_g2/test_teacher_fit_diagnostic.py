"""Array-only teacher fit checks; never imports Torch or queries environments."""
import unittest
import numpy as np
from .tools.teacher_fit_diagnostic import sample_metrics,summarize


class TeacherFitDiagnosticTests(unittest.TestCase):
    def sample(self):
        return dict(legality_mask=[True,True,True,False],evidence_mask=[True,True,False,False],
                    ties=[True,False,False,False],q_values=[2.,1.,0.,0.],stop_id=1)

    def test_unknown_greedy_is_unlabelled_not_zero_gap(self):
        result=sample_metrics([2.,1.,1000.,-np.inf],self.sample())
        self.assertTrue(result['full_greedy_unknown'])
        self.assertFalse(result['full_legal_greedy_observed_near_best_hit'])
        self.assertTrue(result['evidence_greedy_near_best_hit'])
        self.assertIsNone(result['observed_full_greedy_Q_gap'])
        self.assertEqual(result['unknown_probability_mass'],1.)
        self.assertAlmostEqual(result['evidence_normalized_tie_cross_entropy'],np.log1p(np.exp(-1.)))

    def test_stop_and_full_legal_probability(self):
        result=sample_metrics([1.,2.,0.,-np.inf,-np.inf],self.sample())
        self.assertTrue(result['full_greedy_stop'])
        self.assertTrue(result['evidence_greedy_stop'])
        self.assertAlmostEqual(result['stop_probability'],np.exp(2)/(np.exp(2)+np.exp(1)+1))
        self.assertEqual(result['observed_full_greedy_Q_gap'],1.)
        self.assertAlmostEqual(result['unknown_probability_mass']+result['evidence_probability_mass'],1.)

    def test_illegal_action_must_be_masked(self):
        with self.assertRaisesRegex(ValueError,'mask differs'):
            sample_metrics([1.,2.,0.,3.],self.sample())

    def test_groups_do_not_report_unknown_as_known_quality(self):
        a=dict(model='model',split='initial',scene_id='a',teacher_depth=2,is_root=True,analysis_weight=2/3,
               **sample_metrics([2.,1.,1000.,-np.inf],self.sample()))
        b=dict(model='model',split='extra',scene_id='b',teacher_depth=1,is_root=False,analysis_weight=1/3,
               **sample_metrics([3.,1.,0.,-np.inf],self.sample()))
        result=summarize([a,b])
        both=next(g for g in result['groups'] if g['split']=='combined' and g['level']=='all')
        self.assertEqual(both['metrics']['observed_full_greedy_Q_gap']['n'],1)
        self.assertEqual(both['metrics']['observed_full_greedy_Q_gap']['unknown'],1)
        self.assertAlmostEqual(both['metrics']['full_greedy_unknown']['sample_mean'],.5)
        self.assertAlmostEqual(both['metrics']['full_greedy_unknown']['teacher_weighted_mean'],2/3)


if __name__=='__main__':unittest.main()
