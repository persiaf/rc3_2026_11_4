"""Pure schedule, admission and evaluation-statistic checks; zero model calls."""
import copy,unittest
from .common import HOME,read
from .m2 import authorize
from .evaluate import METHODS,gate,summarize

def rows():
    cases=read(HOME/'M2_EVALUATION_CASES.json');out=[]
    for population in ('LOCAL66','FULL12'):
        for c in cases[population]:
            for method in METHODS:
                j=98. if method=='RelationRL' else 100.
                out.append(dict(case_id=c['id'],population=population,method=method,valid=True,physical_key=c['physical_key'],
                    group=c.get('group','exposed_training_parent_full'),width=c['spec']['width'],height=c['spec']['height'],full_request_s=1.,
                    metrics=dict(J=j,L=j-.75,K=1,raw_J=j+1,raw_L=j+.25,raw_K=1)))
    return out

class ProtocolTests(unittest.TestCase):
    def test_no_authorization_rejects_before_ledger(self):
        with self.assertRaises(PermissionError):authorize(HOME/'M2_PROTOCOL.json',HOME/'NO_AUTHORIZATION_EXISTS.json','init')
        self.assertFalse((HOME/'science').exists())
    def test_exact_schedule_counts_and_training_only(self):
        s=read(HOME/'M2_DATA_SCHEDULE.json')
        self.assertEqual(len(s['init']),512);self.assertEqual(len(s['paired']),2048)
        for key,batch in [('init',8),('paired',4)]:
            for i,row in enumerate(s[key]):
                self.assertEqual(row['update'],i+1);self.assertEqual(len(row['tasks']),batch)
                self.assertEqual(sum(t.startswith('F') for t in row['tasks']),batch//2)
                self.assertTrue(all(t in s['training_metadata'] for t in row['tasks']))
    def test_independent_full_group_can_fail(self):
        r=rows();self.assertTrue(gate(dict(rows=r,endpoint=2048,complete=True))['passed'])
        for row in r:
            if row['population']=='FULL12' and row['method']=='RelationRL':row['metrics']['J']=101.
        self.assertFalse(gate(dict(rows=r,endpoint=2048,complete=True))['passed'])
    def test_incomplete_grid_never_passes(self):
        r=rows()[:-1];g=gate(dict(rows=r,endpoint=2048,complete=False))
        self.assertEqual(g['status'],'INSUFFICIENT')
        partial=summarize(r)['FULL12']['Detour16']['overall']
        self.assertFalse(partial['complete']);self.assertNotIn('parent_equal_mean',partial)
    def test_512_full_severe_degradation_stops(self):
        r=rows()
        for row in r:
            if row['population']=='FULL12' and row['method']=='RelationRL':row['metrics']['J']=104.
        self.assertEqual(gate(dict(rows=r,endpoint=512,complete=True))['status'],'STOP_SEVERE_DEGRADATION')
    def test_population_counts(self):
        s=summarize(rows());self.assertEqual(s['LOCAL66']['RelationRL']['overall']['planned'],66)
        self.assertEqual(s['FULL12']['RelationRL']['overall']['planned'],12)
        self.assertEqual(s['LOCAL66']['RelationRL']['SEL']['planned'],48)
        self.assertEqual(s['LOCAL66']['RelationRL']['DEV']['planned'],18)

if __name__=='__main__':unittest.main()
