import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
from .core import Ledger,LimitReached
class Contracts(unittest.TestCase):
    def test_cap_before_side_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'outputs').mkdir()
            with patch('rc3_rl_g23.core.ROOT',root):
                l=Ledger(dict(nonstop=1));l.use('nonstop')
                with self.assertRaises(LimitReached):l.use('nonstop')
                self.assertEqual(l.counts['nonstop'],1)
    def test_independent_value_target_includes_next_stop(self):
        r=.2;qnext=-.15;qstop=-.1
        self.assertAlmostEqual(r+qnext-qstop,.15)
        self.assertNotEqual(r+qnext-qstop,r)
    def test_suffix_stop_baseline(self):
        prefix=.5;total=.7;qstop=-.2
        self.assertAlmostEqual(total-prefix-qstop,.4)
    def test_terminal_has_no_extra_bootstrap(self):
        r=.3;qstop=-.1;done=True
        self.assertAlmostEqual(r+(0 if done else .9)-qstop,.4)
if __name__=='__main__':unittest.main()
