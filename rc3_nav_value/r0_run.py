"""Registered R0 bounded deterministic rollout teacher. No policy or optimizer."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch
import numpy as np
from threadpoolctl import threadpool_limits
from rc3_fast import solver
from rc3_fast.scenes import make_complex_scene
from rc3_v3 import environment as old_env
from rc3_v4 import runtime
from rc3_baseline.validator import validate
from rc3_constructive.problem import Cache, resample_tasks
from rc3_constructive import baselines

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'r0'
BASE='8b3ad97cfa98109a46a3759f625f2c2ef7a04455'
TOL=1e-7
MODES=('regular','mixed','blocked')
LIMITS=dict(native=32,root=32,query=2048,teacher=32,other=160)


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def signature(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def append(p,d):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(d,allow_nan=False)+'\n');f.flush()


def source_hashes():
    names=['rc3_nav_value/r0_run.py','rc3_constructive/problem.py','rc3_constructive/env.py',
        'rc3_constructive/baselines.py','rc3_fast/solver.py','rc3_fast/scenes.py','rc3_fast/planner.py',
        'rc3_baseline/baseline.py','rc3_baseline/validator.py','rc3_v3/environment.py','rc3_v4/runtime.py']
    return {n:digest(ROOT/n) for n in names}


def register():
    if (OUT/'protocol.json').exists():raise FileExistsError('R0 already registered')
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==BASE
    audit=read(OUT/'seed_audit.json');assert audit['verdict']=='PASS_NO_RECORDED_PRIOR_USE' and not audit['collisions']
    seeds=[dict(case=i,map_seed=8300000000+i,task_seed=8400000000+i,mode=MODES[i%3]) for i in range(32)]
    write(OUT/'seeds.json',seeds)
    h_source={n:digest(ROOT/n) for n in ('rc3_constructive/baselines.py','rc3_fast/solver.py')}
    write(OUT/'protocol.json',dict(schema='rc3-nav-value-r0-v1',date='2026-11-05',base_commit=BASE,
        design_commit=BASE,proposal='RC3_研究模型与实验计划_20261105_评审修订版.docx',proposal_sha256=digest(ROOT/'RC3_研究模型与实验计划_20261105_评审修订版.docx'),
        primary_cases=32,all_cases_in_primary_summary=True,mode_counts=dict(regular=11,mixed=11,blocked=10),
        seeds_sha256=digest(OUT/'seeds.json'),seed_audit_sha256=digest(OUT/'seed_audit.json'),seed_boundary_sha256=digest(OUT/'seed_audit_boundary_sanity.json'),
        H='original H0: twice normalize+two_opt(passes=60), final normalize; then joint_search(passes=2), final normalize',
        H_sha256=signature(h_source),H_source_sha256=h_source,
        reference='g_B-v1: fixed native Fixed3 service sequence B, scan from beginning for first positive-gain legal node; serve all remaining covered tasks; auto depot return',
        primary_reference_cost='J(H(Complete_gB(empty prefix)))',native_Fixed3_separate=True,old_128_evaluation=False,
        objective='J=L+0.75K, closed tour, unchanged proxy reachability and navigation',
        checkpoints='Coverage thresholds0,1/8,...7/8, queried at first actual teacher state at or beyond threshold; consume all crossed thresholds once; no terminal/duplicate prefix',
        query_candidates='reference first; gain/(D_current,j+0.75) descending fill to4 unique; D_current,j ascending fill to6; independent fixed random permutation fill to8; all legal if fewer; numeric rule ties by originalID',
        exploration_seed='8500000000+case*8+checkpoint_index; independent of maps/tasks',
        winner_rule='minimum actual finalY; within absolute J tolerance1e-7 prefer reference, then originalID; no network scores',
        max_checkpoints_per_case=8,max_candidates_per_checkpoint=8,absolute_J_tolerance=TOL,
        top_level_caps=LIMITS,total_top_level_cap=2304,
        budget_semantics='One full nativeFixed3 call OR one Complete_gB+fullH/finalH call is a top-level evaluation. Internal recipe construction, two_opt and joint_search separately counted. Counts are not FLOPs or matched search effort.',
        cache_key='sceneSHA256, complete ordered prefix, forcedaction, referenceBhash, gBversion, Hhash, tolerance. Root uses forcedreference action so first root query can hit exact cache. Real calls versus uses/hits separately recorded.',
        calibration=dict(cases=[0,1,2],calls='first3 native and root allocations reused, no extra calibration H',
            wall_rule='Before main queries: min(1800,max(180,ceil(4*(32*max_prepare_s+32*max_native_s+2272*max_completeH_s+64*max_export_s)+60))) seconds. Soft checks before each terminal call and between cases.',
            main_timer='starts after calibration and source checks; includes all remaining scene preparation, teacher decisions, terminal calls, export and serialization; separate replay verification excluded'),
        audits='One forced uncached rootH repeat per case charged to other (32 of160); compare raw/final routes+metrics; no further H in independent saved-record verification',
        primary_gate=dict(mean_relative_gain_percent_min=.5,positive_scene_min=12,max_positive_relative_share=.25,
            feasible_all_32=True,complete_cases_32=True,strict_positive='rootJ-teacherJ>1e-7; q=100*(rootJ-teacherJ)/rootJ, thresholds unrounded',
            incomplete='inconclusive stop, no subset, replacement seed, expanded wall or calls'),
        bootstrap='exploratory scene-paired10000 seed114, no training-seed or deployment significance claim',
        fail_closed='Abort on invalid prefix, nonfinite/increase beyond numerical guard, certificate failure or source mismatch; preserve ledger/partial output, no implicit retries/resume',
        verification='Numeric replay of all saved unique completeH raw/final routes, candidate queues/checkpoints from saved data;96 exported plans and96 independent scalar geometry checks if all cases complete; zero verificationH',
        claims='R0 only tests queried action opportunities and bounded teacher improvement underH; no proof of learnability, missing-information necessity, RL contribution, neural deployment, or all-action optimality',
        neural_forwards=0,optimizer_updates=0,retained_test_queries=0,R1_started=False,R2_started=False,R3_started=False,
        cpu_threads=1,source_sha256=source_hashes()))
    print('Registered R0:32 fixed scenes;2304 top-level cap; no training',flush=True)


def check():
    p=read(OUT/'protocol.json');assert all(digest(ROOT/n)==h for n,h in p['source_sha256'].items())
    assert digest(OUT/'seed_audit.json')==p['seed_audit_sha256'] and digest(OUT/'seeds.json')==p['seeds_sha256']
    assert digest(ROOT/p['proposal'])==p['proposal_sha256']
    audit=read(OUT/'seed_audit.json')
    inventory=[json.loads(s) for s in (OUT/'seed_audit_inventory.jsonl').read_text(encoding='utf-8').splitlines()]
    canonical=''.join(f"{r['path']}\t{r['sha256']}\t{r['file_bytes']}\t{r['scanned_bytes']}\n" for r in sorted(inventory,key=lambda r:r['path']))
    assert hashlib.sha256(canonical.encode('utf-8')).hexdigest()==audit['scan']['inventory_sha256']
    assert digest(OUT/'seed_audit_boundary_sanity.json')==p['seed_boundary_sha256']
    assert digest(OUT/'seed_audit_inventory.jsonl')==read(OUT/'seed_audit_boundary_sanity.json')['frozen_files_sha256']['seed_audit_inventory.jsonl']
    return p


class Budget:
    def __init__(self,deadline=math.inf):
        ledger=OUT/'calls.jsonl';self.ledger=ledger;self.deadline=deadline
        rows=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
        self.started={k:sum(r['event']=='start' and r['kind']==k for r in rows) for k in LIMITS}
        self.completed={k:sum(r['event']=='complete' and r['kind']==k for r in rows) for k in LIMITS}
        if self.started!=self.completed:raise ValueError('Incomplete prior terminal call; no implicit retry')
    def call(self,kind,case,fn):
        if time.perf_counter()>=self.deadline:raise TimeoutError('Frozen wall limit reached')
        assert self.started[kind]<LIMITS[kind] and sum(self.started.values())<2304
        self.started[kind]+=1;ordinal=sum(self.started.values())
        append(self.ledger,dict(event='start',kind=kind,case=case,call=ordinal))
        tick=time.perf_counter();value=fn();elapsed=time.perf_counter()-tick
        self.completed[kind]+=1
        append(self.ledger,dict(event='complete',kind=kind,case=case,call=ordinal,elapsed_s=elapsed))
        return value,elapsed


@contextmanager
def instrument(count):
    def wrap(name,fn):
        def wrapped(*args,**kwargs):
            count[name]=count.get(name,0)+1
            return fn(*args,**kwargs)
        return wrapped
    with patch.object(old_env,'solve_route',wrap('solve_route',old_env.solve_route)),\
         patch.object(solver,'construct',wrap('construct',solver.construct)),\
         patch.object(solver,'two_opt',wrap('two_opt',solver.two_opt)),\
         patch.object(solver,'joint_search',wrap('joint_search',solver.joint_search)):
        yield


def scene_item(case,cache):
    seed=read(OUT/'seeds.json')[case];path=OUT/'scenes'/f'case{case:02}.json'
    if path.exists():scene=read(path)
    else:
        scene=resample_tasks(make_complex_scene(10,10,seed['map_seed'],seed['mode']),seed['task_seed']);write(path,scene)
    assert scene['seed']==seed['map_seed'] and scene['task_seed']==seed['task_seed']
    tick=time.perf_counter();item=cache.prepare(scene)
    return item,time.perf_counter()-tick,digest(path)


def state(item,prefix):
    left=np.ones(len(item.C),bool);current=0
    for node in prefix:
        assert type(node)==int and 1<=node<=item.C.shape[1]
        assert np.isfinite(item.D[current,node]) and np.isfinite(item.D[node,0])
        assert (left & item.C[:,node-1]).any(),'prefix zero-gain action'
        left &= ~item.C[:,node-1];current=node
    return left,current


def legal(item,left,current):
    return item.C[left].any(0)&np.isfinite(item.D[current,1:])&np.isfinite(item.D[1:,0])


def complete(item,B,prefix,action):
    prefix=list(prefix);left,current=state(item,prefix)
    if action is not None:
        assert left.any() and legal(item,left,current)[action-1]
        prefix.append(int(action));left &= ~item.C[:,action-1];current=action
    while left.any():
        mask=legal(item,left,current)
        node=next((n for n in B if mask[n-1]),None)
        if node is None:raise ValueError('g_B cannot complete legal prefix')
        prefix.append(int(node));left &= ~item.C[:,node-1];current=node
    return [0]+prefix+[0]


def metrics(item,route):
    nodes=route[1:-1];assert route[0]==route[-1]==0 and len(nodes)==len(set(nodes))
    assert all(1<=n<=item.C.shape[1] for n in nodes)
    assert item.C[:,np.asarray(nodes,dtype=int)-1].any(1).all()
    L=float(sum(item.D[a,b] for a,b in zip(route,route[1:])));assert np.isfinite(L)
    return dict(J=L+.75*len(nodes),L=L,K=len(nodes))


def candidates(item,B,prefix,case,checkpoint):
    left,current=state(item,prefix);mask=legal(item,left,current);ids=np.flatnonzero(mask)+1
    ref=next(n for n in B if mask[n-1]);gains=item.C[left].sum(0);score=gains/(item.D[current,1:]+.75)
    queue=[ref]
    def fill(order,target):
        for n in order:
            if len(queue)>=min(target,len(ids)):break
            n=int(n)
            if n not in queue:queue.append(n)
    fill(sorted(ids,key=lambda n:(-score[n-1],n)),4)
    fill(sorted(ids,key=lambda n:(item.D[current,n],n)),6)
    fill(np.random.default_rng(8500000000+case*8+checkpoint).permutation(ids),8)
    return queue,ref,score,len(ids)


class Evaluator:
    def __init__(self,item,B,case,scene_sha,budget,p):
        self.item,self.B,self.case,self.budget=item,B,case,budget
        self.ident=dict(scene=scene_sha,B=signature(B),g='g_B-v1',H=p['H_sha256'],tol=TOL)
        self.cache={};self.uses=0;self.hits=0
    def key(self,prefix,action):return signature(dict(**self.ident,prefix=prefix,action=action))
    def evaluate(self,prefix,action,kind,uncached=False):
        self.uses+=1;key=self.key(prefix,action)
        if key in self.cache and not uncached:
            self.hits+=1;return self.cache[key],True
        internal={}
        def calculate():
            raw=complete(self.item,self.B,prefix,action)
            with instrument(internal):
                h0=baselines.postprocess(self.item,raw)
                final=baselines.normalize(self.item,solver.joint_search(list(h0),self.item.C,self.item.D,passes=2))
            raw_m=metrics(self.item,raw);final_m=metrics(self.item,final)
            assert final_m['J']<=raw_m['J']+TOL
            return dict(case=self.case,kind=kind,key=key,prefix=list(prefix),action=action,
                raw_route=raw,raw=raw_m,route=final,final=final_m,internal_calls=internal)
        result,elapsed=self.budget.call(kind,self.case,calculate);result['terminal_s']=elapsed
        append(OUT/'evaluations.jsonl',result)
        if not uncached:self.cache[key]=result
        return result,False


def export(item,result,case,name):
    tick=time.perf_counter();plan=item.certify(result['route'],'R0_'+name)
    assert [0]+[n+1 for n in plan['visit_order_candidate_ids']]+[0]==result['route']
    assert abs(plan['objective_length_plus_stops_m']-result['final']['J'])<=TOL
    path=OUT/'plans'/f'case{case:02}_{name}.json';write(path,plan)
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path),valid=True,export_s=time.perf_counter()-tick)


def baseline(item,case,scene_sha,budget,p):
    internal={}
    def native():
        with instrument(internal):plan,info=runtime.run(item.base,'fixed3')
        assert not info['fallback_used'],info
        return plan,info
    (plan,info),elapsed=budget.call('native',case,native)
    path=OUT/'plans'/f'case{case:02}_native.json';write(path,plan)
    B=[int(n)+1 for n in plan['visit_order_candidate_ids']]
    left,current=state(item,[]);assert item.C[:,np.asarray(B)-1].any(1).all()
    assert all(np.isfinite(item.D[0,n]) and np.isfinite(item.D[n,0]) for n in B)
    ev=Evaluator(item,B,case,scene_sha,budget,p);ref=B[0]
    root,_=ev.evaluate([],ref,'root');root_export=export(item,root,case,'root')
    row=dict(case=case,B=B,B_hash=signature(B),scene_sha256=scene_sha,native_metrics=metrics(item,[0]+B+[0]),
        native_plan=dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path),valid=True),
        native_internal_calls=internal,native_info=info,native_s=elapsed,root=root,root_plan=root_export)
    write(OUT/'baselines'/f'case{case:02}.json',row)
    return row,ev


def calibrate():
    p=check()
    if (OUT/'calibration_started.json').exists():raise FileExistsError('No implicit calibration retry')
    write(OUT/'calibration_started.json',dict(protocol_sha256=digest(OUT/'protocol.json')))
    cache=Cache(1);budget=Budget();rows=[];started=time.perf_counter()
    for case in p['calibration']['cases']:
        item,prep,scene_sha=scene_item(case,cache);base,_=baseline(item,case,scene_sha,budget,p)
        rows.append(dict(case=case,prepare_s=prep,native_s=base['native_s'],completeH_s=base['root']['terminal_s'],export_s=base['root_plan']['export_s']))
    maximum=lambda k:max(r[k] for r in rows)
    cap=int(min(1800,max(180,math.ceil(4*(32*maximum('prepare_s')+32*maximum('native_s')+2272*maximum('completeH_s')+64*maximum('export_s'))+60))))
    write(OUT/'wall_protocol.json',dict(protocol_sha256=digest(OUT/'protocol.json'),wall_limit_s=cap,
        samples=rows,calibration_elapsed_s=time.perf_counter()-started,top_calls=budget.completed,
        frozen_before_main_queries=True,source_sha256=source_hashes(),hardware=dict(cpu=platform.processor(),python=platform.python_version(),numpy=np.__version__),
        scope='3 primary cases native/root results reused;6 actual terminal calls counted. No new scene outside32 and no uncharged H probe.'))
    print(json.dumps(read(OUT/'wall_protocol.json')),flush=True)


def run():
    p=check();wall=read(OUT/'wall_protocol.json');assert wall['protocol_sha256']==digest(OUT/'protocol.json')
    if (OUT/'run_started.json').exists():raise FileExistsError('No implicit R0 retry or resume')
    write(OUT/'run_started.json',dict(wall_sha256=digest(OUT/'wall_protocol.json')))
    started=time.perf_counter();budget=Budget(started+wall['wall_limit_s']);cache=Cache(1);summaries=[]
    try:
        for case in range(32):
            item,prep,scene_sha=scene_item(case,cache);path=OUT/'baselines'/f'case{case:02}.json'
            if path.exists():
                base=read(path);assert base['scene_sha256']==scene_sha
                ev=Evaluator(item,base['B'],case,scene_sha,budget,p);ev.cache[base['root']['key']]=base['root']
            else:base,ev=baseline(item,case,scene_sha,budget,p)
            B=base['B'];root=base['root'];prefix=[];threshold=0;checks=[];expected=root['final']['J']
            while True:
                left,current=state(item,prefix)
                if not left.any():break
                covered=len(left)-int(left.sum())
                if threshold<8 and 8*covered>=threshold*len(left):
                    crossed=[]
                    while threshold<8 and 8*covered>=threshold*len(left):crossed.append(threshold);threshold+=1
                    ci=len(checks);queue,ref,score,legal_count=candidates(item,B,prefix,case,ci)
                    values=[]
                    for node in queue:
                        result,hit=ev.evaluate(prefix,node,'query')
                        values.append(dict(node=node,Y=result['final']['J'],raw_J=result['raw']['J'],key=result['key'],cached=hit,route=result['route']))
                    reference=next(v for v in values if v['node']==ref)
                    assert abs(reference['Y']-expected)<=TOL,'reference continuity mismatch'
                    minimum=min(v['Y'] for v in values);ties=[v for v in values if v['Y']<=minimum+TOL]
                    winner=next((v for v in ties if v['node']==ref),min(ties,key=lambda v:v['node']))
                    assert winner['Y']<=expected+TOL;expected=winner['Y'];action=winner['node']
                    immediate=min(queue,key=lambda n:(-score[n-1],n))
                    raw_best=min(values,key=lambda v:(v['raw_J'],v['node']))['node']
                    cp=dict(case=case,checkpoint=ci,crossed_eighths=crossed,prefix=list(prefix),remaining_tasks=int(left.sum()),
                        legal_actions=legal_count,reference_action=ref,chosen_action=action,candidates=values,
                        immediate_action=immediate,raw_best_action=raw_best,reference_Y=reference['Y'],best_Y=minimum,
                        chosen_Y=expected,queried_opportunity_J=reference['Y']-minimum,
                        immediate_regret_J=next(v['Y'] for v in values if v['node']==immediate)-minimum,
                        immediate_vs_final_disagrees=next(v['Y'] for v in values if v['node']==immediate)>minimum+TOL)
                    append(OUT/'checkpoints.jsonl',cp);checks.append(cp)
                else:
                    mask=legal(item,left,current);action=next(n for n in B if mask[n-1])
                prefix.append(int(action))
            teacher,_=ev.evaluate(prefix,None,'teacher')
            assert abs(teacher['final']['J']-expected)<=TOL and teacher['final']['J']<=root['final']['J']+TOL
            teacher_export=export(item,teacher,case,'teacher')
            audit,_=ev.evaluate([],B[0],'other',uncached=True)
            assert audit['raw_route']==root['raw_route'] and audit['route']==root['route']
            assert abs(audit['final']['J']-root['final']['J'])<=TOL
            row=dict(**read(OUT/'seeds.json')[case],native=base['native_metrics'],root=root['final'],teacher=teacher['final'],
                gain_J=root['final']['J']-teacher['final']['J'],gain_percent=100*(root['final']['J']-teacher['final']['J'])/root['final']['J'],
                checkpoints=len(checks),candidate_uses=sum(len(c['candidates']) for c in checks),
                cache_uses=ev.uses,cache_hits=ev.hits,preparation_s=prep,
                plans=dict(native=base['native_plan'],root=base['root_plan'],teacher=teacher_export),
                B_hash=base['B_hash'],scene_sha256=scene_sha,valid=True)
            summaries.append(row);write(OUT/'case_results.json',summaries)
            print(json.dumps(dict(cases_done=case+1,top_level_calls=sum(budget.completed.values()),gain_percent=row['gain_percent'])),flush=True)
        check()
        write(OUT/'execution_summary.json',dict(complete=True,cases=32,terminal_started=budget.started,terminal_completed=budget.completed,
            total_terminal_calls=sum(budget.completed.values()),run_elapsed_s=time.perf_counter()-started,wall_limit_s=wall['wall_limit_s'],
            neural_forwards=0,optimizer_updates=0,new_policy_samples=0,retained_test_queries=0,R1_started=False,
            scope='Offline bounded teacher mechanism evidence, not neural deployment timing. Calibration calls included in cumulative ledger; main wall excludes earlier calibration.'))
    except Exception as error:
        write(OUT/'execution_failure.json',dict(error=repr(error),complete_cases=len(summaries),terminal_started=budget.started,terminal_completed=budget.completed,elapsed_s=time.perf_counter()-started))
        raise


def verify():
    check();assert read(OUT/'execution_summary.json')['complete']
    if (OUT/'verification_started.json').exists():raise FileExistsError('No duplicate verification')
    write(OUT/'verification_started.json',dict(scope='No H or native solver replay'))
    rows=[json.loads(s) for s in (OUT/'evaluations.jsonl').read_text().splitlines()]
    cache=Cache(1);certificates=[];numeric=0;steps=0;checkpoint_checks=0
    checkpoints=[json.loads(s) for s in (OUT/'checkpoints.jsonl').read_text().splitlines()]
    case_results=read(OUT/'case_results.json')
    from rc3_constructive.env import Env
    for case in range(32):
        scene=read(OUT/'scenes'/f'case{case:02}.json');item=cache.prepare(scene)
        base=read(OUT/'baselines'/f'case{case:02}.json');B=base['B']
        case_rows=[r for r in rows if r['case']==case];by_key={}
        for r in case_rows:
            by_key.setdefault(r['key'],r)
            nodes=r['raw_route'][1:-1]
            assert nodes[:len(r['prefix'])]==r['prefix']
            offset=len(r['prefix'])
            if r['action'] is not None:assert nodes[offset]==r['action'];offset+=1
            env=Env(item)
            for position,node in enumerate(nodes):
                if position>=offset:
                    mask=env.get_action_mask();assert node==next(n for n in B if mask[n-1])
                env.step(node-1)
            assert env.route==r['raw_route'] and abs(-sum(env.rewards)-r['raw']['J'])<=TOL
            for key in ('raw','final'):
                m=metrics(item,r['raw_route'] if key=='raw' else r['route'])
                for k in ('J','L','K'):assert abs(m[k]-r[key][k])<=TOL
            numeric+=1;steps+=len(r['raw_route'])-2
        cp_rows=[r for r in checkpoints if r['case']==case]
        teacher=next(r for r in case_rows if r['kind']=='teacher');prefix=[];threshold=0;ci=0;expected=base['root']['final']['J']
        for actual_action in teacher['raw_route'][1:-1]:
            left,current=state(item,prefix);covered=len(left)-int(left.sum())
            if threshold<8 and 8*covered>=threshold*len(left):
                crossed=[]
                while threshold<8 and 8*covered>=threshold*len(left):crossed.append(threshold);threshold+=1
                cp=cp_rows[ci];queue,ref,score,legal_count=candidates(item,B,prefix,case,ci)
                assert cp['prefix']==prefix and cp['checkpoint']==ci and cp['crossed_eighths']==crossed
                assert cp['legal_actions']==legal_count and cp['reference_action']==ref
                assert [v['node'] for v in cp['candidates']]==queue
                for v in cp['candidates']:
                    saved=by_key[v['key']]
                    assert saved['prefix']==prefix and saved['action']==v['node']
                    assert abs(saved['raw']['J']-v['raw_J'])<=TOL and abs(saved['final']['J']-v['Y'])<=TOL
                    assert saved['route']==v['route']
                ref_row=next(v for v in cp['candidates'] if v['node']==ref)
                assert abs(ref_row['Y']-expected)<=TOL
                best=min(v['Y'] for v in cp['candidates']);ties=[v for v in cp['candidates'] if v['Y']<=best+TOL]
                winner=next((v for v in ties if v['node']==ref),min(ties,key=lambda v:v['node']))
                assert cp['chosen_action']==winner['node']==actual_action
                assert abs(cp['chosen_Y']-winner['Y'])<=TOL
                expected=winner['Y'];ci+=1;checkpoint_checks+=1
            else:
                mask=legal(item,left,current);assert actual_action==next(n for n in B if mask[n-1])
            prefix.append(actual_action)
        assert ci==len(cp_rows)<=8 and abs(expected-teacher['final']['J'])<=TOL
        for name in ('native','root','teacher'):
            path=OUT/'plans'/f'case{case:02}_{name}.json';plan=read(path)
            assert digest(path)==case_results[case]['plans'][name]['sha256']
            expected_metrics=case_results[case]['native' if name=='native' else name]
            assert abs(plan['objective_length_plus_stops_m']-expected_metrics['J'])<=TOL
            cert=validate(scene,plan,item.base.sparse);assert cert['valid_model_plan'],cert
            certificates.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=digest(path),valid=True))
    ledger=[json.loads(s) for s in (OUT/'calls.jsonl').read_text().splitlines()]
    assert sum(r['event']=='start' for r in ledger)==sum(r['event']=='complete' for r in ledger)<=2304
    check();write(OUT/'verification.json',dict(all_passed=True,numeric_completeH_replays=numeric,raw_service_transitions=steps,
        separate_scalar_geometry_certificates=96,export_geometry_certificates=96,total_geometry_certificates=192,
        certificates=certificates,new_H_calls=0,new_native_solver_calls=0,new_model_forwards=0,
        new_completion_trajectories=0,reference_suffix_replay_checks=numeric,checkpoint_queue_winner_checks=checkpoint_checks,
        scope='Replay saved raw actions, check reference suffix order and all checkpoint queues/winners using saved costs; no new completion trajectory, H or solver calls. All96 saved plans scalar-validated; no IK.'))
    print('PASS: saved label numeric replay and96 scalar plan certificates; no H/solver/model calls',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['register','calibrate','run','verify']);a=parser.parse_args()
    with threadpool_limits(limits=1):globals()[a.phase]()
