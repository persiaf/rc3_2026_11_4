"""Pure-data opportunity-gate boundaries; no Torch or environment imports."""
import copy
import math
import unittest

from rc3_rl_g21.gates import evaluate, validate_rows


CRITERIA = dict(initial_positive_full_known_min_fraction=.7,
                extra_positive_full_known_min_fraction=.5,
                initial_clear_stop_full_stop_min_fraction=.75,
                initial_root_full_unknown_max_fraction=.2)


def row(index, *, positive=True, selected='continue', root=True):
    unknown = selected == 'unknown'
    stop = selected == 'stop'
    gain = None if unknown else 0. if stop else .25 if positive else -.1
    return dict(scene_id=str(index), sample_index=index, is_root=root,
        teacher_depth=2 if root else 1, delta=.25 if positive else -.1,
        known_positive=positive, clear_stop=not positive,
        full_greedy_unknown=unknown, full_greedy_stop=stop,
        full_known_positive_continue=not unknown and not stop and positive,
        full_selected_gain=gain,
        observed_full_greedy_Q_gap=None if unknown else .25 if positive and stop else .1 if not positive and not stop else 0.)


def diagnostics():
    initial=[row(i,selected='continue' if i<7 else 'stop') for i in range(10)]
    initial += [row(10+i,positive=False,selected='stop' if i<3 else 'continue') for i in range(4)]
    extra=[row(100+i,selected='continue' if i<2 else 'stop') for i in range(4)]
    return dict(rows=initial), dict(rows=extra)


class OpportunityGateTests(unittest.TestCase):
    def test_inclusive_fraction_boundaries(self):
        initial,extra=diagnostics()
        result=evaluate(initial,extra,CRITERIA)
        self.assertTrue(result['passed'])
        self.assertEqual(result['metrics']['initial_positive_full_known'],dict(count=7,n=10,fraction=.7))
        self.assertEqual(result['metrics']['extra_positive_full_known']['fraction'],.5)
        self.assertEqual(result['metrics']['initial_clear_stop_full_stop']['fraction'],.75)

    def test_below_each_positive_minimum_fails(self):
        for which in ('initial','extra'):
            initial,extra=diagnostics()
            target=initial if which=='initial' else extra
            target['rows'][0]=row(target['rows'][0]['sample_index'],selected='stop')
            self.assertFalse(evaluate(initial,extra,CRITERIA)['passed'])

    def test_below_stop_minimum_fails(self):
        initial,extra=diagnostics();initial['rows'][10]=row(10,positive=False,selected='continue')
        self.assertFalse(evaluate(initial,extra,CRITERIA)['passed'])

    def test_unknown_is_not_known_positive_or_zero_regret(self):
        initial,extra=diagnostics();initial['rows'][0]=row(0,selected='unknown')
        result=evaluate(initial,extra,CRITERIA)
        self.assertEqual(result['metrics']['initial_positive_full_known']['count'],6)
        self.assertEqual(result['metrics']['initial_root_full_unknown']['count'],1)
        self.assertIsNone(initial['rows'][0]['observed_full_greedy_Q_gap'])
        self.assertFalse(result['passed'])

    def test_unknown_max_inclusive_and_excess_fails(self):
        initial=dict(rows=[row(i,selected='continue' if i<8 else 'unknown') for i in range(10)])
        initial['rows'] += [row(10+i,positive=False,selected='stop' if i<6 else 'unknown') for i in range(8)]
        extra=dict(rows=[row(100)])
        criteria=dict(CRITERIA,initial_root_full_unknown_max_fraction=4/18)
        self.assertTrue(evaluate(initial,extra,criteria)['passed'])
        criteria['initial_root_full_unknown_max_fraction']=4/18-1e-6
        self.assertFalse(evaluate(initial,extra,criteria)['passed'])

    def test_successors_are_excluded(self):
        initial,extra=diagnostics();before=evaluate(initial,extra,CRITERIA)
        initial['rows'].append(row(999,selected='unknown',root=False))
        self.assertEqual(before,evaluate(initial,extra,CRITERIA))

    def test_empty_denominators_fail(self):
        initial,extra=diagnostics()
        result=evaluate(initial,dict(rows=[]),CRITERIA)
        self.assertFalse(result['passed'])
        self.assertIsNone(result['metrics']['extra_positive_full_known']['fraction'])
        self.assertFalse(evaluate(dict(rows=[]),extra,CRITERIA)['passed'])

    def test_unknown_fabricated_quality_rejected(self):
        for key,value in [('full_known_positive_continue',True),('full_selected_gain',0.),
                          ('observed_full_greedy_Q_gap',0.),('full_greedy_stop',True)]:
            bad=row(0,selected='unknown');bad[key]=value
            with self.assertRaises(ValueError):validate_rows(dict(rows=[bad]))

    def test_delta_labels_and_nonfinite_values_rejected(self):
        for key,value in [('delta',math.nan),('delta',math.inf),('known_positive',False),
                          ('teacher_depth',1),('full_selected_gain',None)]:
            bad=row(0);bad[key]=value
            with self.assertRaises(ValueError):validate_rows(dict(rows=[bad]))

    def test_duplicate_rows_rejected(self):
        item=row(0)
        with self.assertRaises(ValueError):validate_rows(dict(rows=[item,copy.deepcopy(item)]))

    def test_invalid_fraction_threshold_rejected(self):
        initial,extra=diagnostics()
        for value in (math.nan,math.inf,-.1,1.1):
            criteria=dict(CRITERIA,initial_root_full_unknown_max_fraction=value)
            with self.assertRaises(ValueError):evaluate(initial,extra,criteria)

    def test_known_positive_uses_queried_gain_not_continue_alone(self):
        bad=row(0,positive=False,selected='continue')
        self.assertFalse(bad['full_known_positive_continue'])
        validate_rows(dict(rows=[bad]))


if __name__=='__main__':unittest.main()
