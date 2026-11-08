"""Pure/synthetic tests; no S3 target provider, real model or native search calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from .common import SEEDS, SIZES, METHODS, Meter, identity, write_new, read, project_file, HERE
from .inputs import NavigationPool, FormalInputs, window_spec
from .journal import Journal, grid, run_root
from .provenance import Permit, admit_formal, ModelCatalog, verified_geometry
from .statistics import analyze, bootstrap, gains, quality_pass, dominated
from .pipelines import verify_determinism
from rc3_global_local.validation import validate
from rc3_global_local_impl_review.runtime_guards import snapshot_parent, bind_navigation, current_navigation_sources, extract_snapshot


def fixture_cases():
    return [dict(id=f'{g}{p}_{w}_{ep}',physical_key=f'{g}_geometry_{p}',group=g,size=[w,h],endpoint=ep)
        for g,n in [('seen',6),('unseen',12)] for p in range(n) for w,h in SIZES
        for ep in ('closed_depot','open_first_last_connected')]


def fixture_rows():
    cases=fixture_cases(); byid={c['id']:c for c in cases}; rows=[]
    for s in SEEDS:
        for q in grid(cases,s):
            c=byid[q['case_id']];m=q['method']
            J=90. if m in ('RL_raw','RL_plus_H') else 100.
            timing=.5 if m=='RL_plus_H' else 1.
            rows.append(dict(c,request=q,case_id=c['id'],valid=True,complete_time_s=timing,
                raw_route_sha256='raw',route_sha256='final',metrics=dict(J=J,L=J-.75,K=1.,raw_J=J,raw_L=J-.75,raw_K=1.)))
    return cases, rows


def scalar_fixture():
    # Hand-authored one-task geometry, no task generator or navigation solver.
    scene=dict(nx=10,ny=10,start=[0.,0.],tables=[],obstacles=[],tasks=[dict(id=0)],
        candidates=[dict(id=0,x=1.,y=0.,theta=0.,h=1.)],robot=dict(base_radius=.1,clearance=.1))
    parent=SimpleNamespace(scene=scene,mode='proxy',identity='parent',F=np.array([[[1,0]]],bool),
        A=np.array([[[1,1]]],bool),D=np.array([[0.,1.],[1.,0.]]))
    p=SimpleNamespace(parent=parent,identity='view',entry=0,exit=0,candidate_ids=np.array([0]),task_ids=np.array([0]))
    plan=dict(problem_identity='view',parent_identity='parent',entry_global_node=0,exit_global_node=0,
        visit_order_candidate_ids=[0],navigation_global_nodes=[0,1,0],target_task_ids_global=[0],
        assignments=[dict(task_id=0,candidate_id=0,arm='L')],service_stops=[dict(sequence=1,candidate_id=0,
        pose=dict(x=1.,y=0.,theta=0.,h=1.),tasks=[dict(task_id=0,arm='L')],arrival_distance_m=1.)],
        route_leg_costs_m=[1.,1.],route_polyline=[[0.,0.],[1.,0.],[0.,0.]],route_length_m=2.,
        stop_penalty_equivalent_m=.75,objective_length_plus_stops_m=2.75,unserved_task_ids=[],status='complete',
        arm_joint_trajectories=None,physical_joint_trajectory_certified=False)
    return p,plan


class EvaluatorTests(unittest.TestCase):
    def test_01_grid_and_order(self):
        cases=fixture_cases()
        self.assertEqual(len(cases),108)
        self.assertEqual(len(grid(cases,SEEDS[0])),5292)
        self.assertEqual(sum(len(grid(cases,s)) for s in SEEDS),15876)
        self.assertEqual(grid(cases,SEEDS[0]),grid(cases,SEEDS[0]))
        self.assertNotEqual(grid(cases,SEEDS[0])[0]['id'],grid(cases,SEEDS[1])[0]['id'])
        self.assertEqual(len(grid(cases[:6],SEEDS[0],True)),126)

    def test_02_hand_aggregation_and_intervals(self):
        # Endpoint means first, seed means second, parent ratios last.
        self.assertAlmostEqual(float(gains(np.mean([50.,150.]),np.mean([60.,120.]))),.1)
        self.assertAlmostEqual(float(gains(np.mean([100.,200.,300.]),np.mean([95.,180.,255.]))),.11666666666666667)
        b=bootstrap([.1,.3],np.array([[0,0],[0,1],[1,0],[1,1]]))
        self.assertAlmostEqual(b['mean'],.2);self.assertAlmostEqual(b['lower95'],.1075)
        self.assertAlmostEqual(b['upper95'],.2925);self.assertAlmostEqual(b['upper_one_sided95'],.285)
        self.assertFalse(quality_pass(dict(mean=.01,lower95=.001),[.01,0.,.01]))
        self.assertTrue(dominated((1,1,1),(1,2,1))); self.assertFalse(dominated((1,1,1),(1,1,1)))
        for v in ([float('nan')],[],[float('inf')]):
            with self.assertRaises(ValueError):bootstrap(v)
        with self.assertRaises(ValueError):gains([0],[1])

    def test_03_complete_synthetic_decisions(self):
        cases,rows=fixture_rows(); result=analyze(rows,cases)
        self.assertEqual(len(result['quality_cells']),6);self.assertEqual(len(result['deployment_cells']),12)
        self.assertTrue(result['combined_pass'])
        # Fail just one physical size group; pooled success must not rescue it.
        altered=copy.deepcopy(rows)
        for r in altered:
            if r['group']=='seen' and r['size']==[4,3] and r['request']['method'] in ('RL_raw','RL_plus_H'):
                r['metrics'].update(J=101.,raw_J=101.)
        bad=analyze(altered,cases)
        self.assertFalse(bad['independent_RL_pass']);self.assertFalse(bad['combined_pass'])
        for invalid in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError):analyze(invalid,cases)
        changed=copy.deepcopy(rows);changed[0]['physical_key']='wrong'
        with self.assertRaises(ValueError):analyze(changed,cases)
        changed=copy.deepcopy(rows);changed[0]['complete_time_s']=float('nan')
        with self.assertRaises(ValueError):analyze(changed,cases)

    def test_04_journal_debit_resume_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'run';reqs=grid(fixture_cases()[:1],SEEDS[0],True)
            j=Journal(path,{'source':'a'},reqs,21)
            j.append(dict(event='session_open',token='a'))
            j.debit(reqs[0])
            with self.assertRaises(ValueError):j.debit(reqs[0])
            reopened=Journal(path,{'source':'a'},reqs,21)
            self.assertEqual(len(reopened.starts),1)
            with self.assertRaises(ValueError):reopened.clean()
            write_new(path/'results'/(reqs[0]['id']+'.json'),dict(value=1))
            j.complete(reqs[0],1.)
            j.append(dict(event='session_close',token='a',elapsed_s=7.,status='CLOSED_BOUNDARY'))
            reopened=Journal(path,{'source':'a'},reqs,21);reopened.clean()
            self.assertEqual(reopened.wall_s,7.)
            with self.assertRaises(ValueError):Journal(path,{'source':'b'},reqs,21)
            with self.assertRaises(ValueError):reopened.debit(reqs[0])
            lines=j.path.read_bytes().splitlines(keepends=True)
            j.path.write_bytes(b''.join(lines[:-1]))
            with self.assertRaises(ValueError):Journal(path,{'source':'a'},reqs,21)

    def test_05_failure_survives_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            reqs=grid(fixture_cases()[:1],SEEDS[0],True);p=Path(temp)/'run'
            j=Journal(p,{},reqs,21);j.debit(reqs[0]);j.append(dict(event='request_error',id=reqs[0]['id'],message='injected'))
            loaded=Journal(p,{},reqs,21)
            self.assertEqual(len(loaded.starts),1);self.assertEqual(len(loaded.failures),1)
            with self.assertRaises(ValueError):loaded.clean()

    def test_06_input_and_authority_boundaries(self):
        fake=SimpleNamespace(scope='engineering')
        with self.assertRaises(ValueError):FormalInputs(fake,Path('unused'))
        with self.assertRaises((FileNotFoundError,ValueError,TypeError)):admit_formal('missing','missing')
        with self.assertRaises(ValueError):project_file('../outside')
        with self.assertRaises(ValueError):window_spec(dict(size=[4,3],position='corner',A_condition=0,endpoints='open_first_last_connected'))
        with self.assertRaises(ValueError):window_spec(dict(size=[9,9],position='corner',A_condition=0))
        with self.assertRaises(ValueError):window_spec(dict(size=[4,3],position='corner',A_condition=True))
        with self.assertRaises(ValueError):window_spec(dict(size=[4,3],position='corner',A_condition=3))

    def test_07_navigation_only_cache(self):
        # Actual pool operations, mocked producer. No real navigation computation.
        scene={'map':'a','tasks':[1]};p=NavigationPool('namespace');meter=Meter()
        bound=SimpleNamespace(geometry_key='a',source_sha256=p.sources,content_sha256='nav',nav={})
        with patch('rc3_s3_evaluator.inputs.map_key',side_effect=lambda s:s['map']),patch('rc3_s3_evaluator.inputs.navigation_hash',return_value='nav'),patch.object(p,'_build',return_value=bound) as build:
            with self.assertRaises(ValueError):p.get(scene,'hot',meter)
            p.prime(scene,meter);p.get(dict(scene,tasks=[2]),'hot',meter)
            self.assertEqual(build.call_count,1)
            p.get(scene,'cold',meter);p.get(scene,'quality',meter)
            self.assertEqual(build.call_count,3)
            with self.assertRaises(ValueError):p.get(dict(scene,map='b'),'hot',meter)
            p.namespace='other'
            with self.assertRaises(ValueError):p.get(scene,'hot',meter)
        self.assertEqual(set(vars(p)),{'namespace','sources','cache'})

    def test_08_independent_scalar_constraints(self):
        p,plan=scalar_fixture()
        with patch('rc3_global_local.validation.proxy_reachable',return_value=True):
            self.assertTrue(validate(p,plan)['valid_model_plan'])
            p.parent.A=np.array([[[0,1]]],bool)
            self.assertFalse(validate(p,plan)['valid_model_plan']) # F(left), A(right) cannot combine
            p.parent.A[:]=1
            for key,value in [('assignments',[]),('entry_global_node',1),('route_leg_costs_m',[float('inf'),1.]),('visit_order_candidate_ids',[99]),('route_polyline',[[0.,0.],[5.,0.]])]:
                bad=copy.deepcopy(plan);bad[key]=value
                self.assertFalse(validate(p,bad)['valid_model_plan'])
            p.parent.scene['obstacles']=[dict(xmin=.4,xmax=.6,ymin=-.1,ymax=.1)]
            self.assertFalse(validate(p,plan)['valid_model_plan'])

    def test_09_raw_pairs_and_route_stability(self):
        cases,rows=fixture_rows();verify_determinism(rows)
        pair=[r for r in rows if r['request']['method'] in ('RL_raw','RL_plus_H')]
        changed=copy.deepcopy(pair);changed[0]['raw_route_sha256']='different'
        with self.assertRaises(ValueError):verify_determinism(changed)
        with self.assertRaises(ValueError):verify_determinism(pair[1:])

    def test_10_fixed_endpoints_all_seeds_static(self):
        catalog=ModelCatalog()
        self.assertEqual(len(catalog.models),3)
        for s in SEEDS:
            for role in ('untrained','SL256','continuedSL512','RL512'):catalog.entry(s,role)
        with self.assertRaises(ValueError):catalog.entry(SEEDS[0],'RL256')
        with self.assertRaises(ValueError):catalog.entry(0,'RL512')
        catalog.models[str(SEEDS[0])]['RL512']['path']='rc3_s3_evaluator/engineering_fixture.pt'
        with self.assertRaises(ValueError):catalog.entry(SEEDS[0],'RL512')

    def test_11_original_geometry_attestation(self):
        assets=verified_geometry()
        self.assertEqual(len(assets),111)
        from .provenance import digest as real_digest
        def damaged(path):
            return 'bad' if str(path).endswith('GEOMETRY_GATE.json') else real_digest(path)
        with patch('rc3_s3_evaluator.provenance.digest',side_effect=damaged):
            with self.assertRaises(ValueError):verified_geometry()
        def damaged_asset(path):
            return 'bad' if str(path).endswith('train_00.json') else real_digest(path)
        with patch('rc3_s3_evaluator.provenance.digest',side_effect=damaged_asset):
            with self.assertRaises(ValueError):verified_geometry()


if __name__=='__main__':unittest.main(verbosity=2)
