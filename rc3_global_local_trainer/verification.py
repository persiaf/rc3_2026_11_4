"""Centrally budgeted real-optimizer engineering acceptance, never formal S2."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
from types import SimpleNamespace

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

from rc3_global_local.environment import Environment
from rc3_global_local.policy import GlobalLocalPolicy, tensor_hash
from rc3_global_local.problem import LocalProblem, signature
from rc3_global_local.terminal import TerminalCache
from rc3_global_local_impl_review.runtime_guards import load_saved_s0_parent, extract_snapshot
from rc3_nav_value.r1.oracle import _instrument, _empty_internal
from .engine import Trainer, make_optimizer, fork_from_supervised, InconclusiveBatch
from .checkpoint import save_checkpoint, load_checkpoint
from .rng import NamedStreams
from .verification_budget import AcceptanceBudget

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BINDING = None
BUDGET = None
OUT = None
TOL = None


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8', newline='\n')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def equal(a, b):
    if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray): return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict): return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def reject(fn, types=(ValueError, RuntimeError)):
    try: fn()
    except types: return True
    raise AssertionError('Invalid input was accepted')


def tiny(kind='normal'):
    # Hand-defined numerical unit fixture, not a scene-generator/nav invocation.
    nt, nc = (1, 1) if kind == 'one' else (3, 8)
    scene = dict(nx=1, ny=1, start=[-1., -1.], tables=[dict(id=0, ix=0, iy=0, cx=0., cy=0., xmin=-.5, xmax=.5, ymin=-.5, ymax=.5)],
      tasks=[dict(id=i, table_id=0, x=.1*i, y=.2, z=.3) for i in range(nt)],
      candidates=[dict(id=i, table_id=0, x=float(i), y=float(i%2), theta=.1*i, h=.5) for i in range(nc)], obstacles=[])
    C = np.ones((1, 1), bool) if kind == 'one' else np.array([[1,1,0,1,0,0,0,1],[0,1,1,0,1,0,0,0],[0,0,0,0,1,1,1,1]], bool)
    F = np.stack([C, np.zeros_like(C)], axis=2)
    xy = np.array([[-1., -1.]]+[[float(i),float(i%2)] for i in range(nc)])
    D = np.sqrt(((xy[:,None,:]-xy[None,:,:])**2).sum(2))
    parent = SimpleNamespace(scene=scene, F=F, A=np.ones_like(F), effective=F, D=D, identity=signature(dict(fixture=kind)))
    return LocalProblem(parent, [] if kind == 'empty' else list(range(nt)), [0])


class MockTerminal:
    def __init__(self): self.calls = 0
    def __call__(self, view, route):
        BUDGET.charge('mock_terminal_requests'); self.calls += 1
        return dict(**view.metrics(route), raw_J=view.metrics(route)['J'], route=list(route), cache_hit=False)
    def state_dict(self): return dict(schema='engineering-mock-terminal-v1', calls=self.calls)
    def validate_state_dict(self, state):
        if state['schema'] != 'engineering-mock-terminal-v1' or state['calls'] < 0: raise ValueError('Wrong fixture terminal')
    def load_state_dict(self, state): self.validate_state_dict(state); self.calls = state['calls']


def trainer(phase='RL', hidden=8, model=None, terminal=None, seed=11622026, batch=4, optimizer=None):
    if model is None:
        torch.manual_seed(41056); model = GlobalLocalPolicy(hidden)
    if optimizer is None: optimizer = make_optimizer(model)
    config = dict(phase=phase, batch_instances=batch, max_optimizer_updates=16,
      binding=dict(engineering_protocol=sha(HERE/'ENGINEERING_VERIFICATION_PROTOCOL.json'), execution_binding=BINDING,
                   dataset='hand-defined numerical unit fixtures', H='mock-terminal-no-H', formal_training=False))
    return Trainer(model, optimizer, NamedStreams(sha(HERE/'ENGINEERING_VERIFICATION_PROTOCOL.json'), seed), terminal or MockTerminal(), config=config)


def labels(view):
    env = Environment(view, lambda p,r:p.metrics(r)); actions=[]
    while not env.done:
        node=int(np.flatnonzero(env.mask())[0])+1; actions.append(node);env.step(node)
    return actions


def clean_diag(d):
    if isinstance(d, dict): return {k:clean_diag(v) for k,v in d.items() if k not in ('elapsed_s','elapsed_s_before_optimizer','H_s','export_s','validation_s')}
    if isinstance(d, list): return [clean_diag(v) for v in d]
    return d


def gradient_contract():
    v=tiny(); views=[v]*4; a,b=trainer(),trainer()
    ra=a.batch_gradients(views, 0, 'joint'); rb=b.batch_gradients(views, 0, 'accumulate')
    assert clean_diag(ra['instances']) == clean_diag(rb['instances'])
    assert np.isclose(ra['loss'],rb['loss'],rtol=TOL['loss_rtol'],atol=TOL['loss_atol'])
    max_abs=max_rel=0.
    for pa,pb in zip(a.model.parameters(),b.model.parameters()):
        if pa.grad is None: assert pb.grad is None;continue
        assert torch.allclose(pa.grad,pb.grad,rtol=TOL['gradient_rtol'],atol=TOL['gradient_atol'])
        delta=(pa.grad-pb.grad).abs();max_abs=max(max_abs,float(delta.max()));max_rel=max(max_rel,float((delta/pa.grad.abs().clamp_min(1e-12)).max()))
    assert ra['gradient_norm']>0
    for item in ra['instances']:
        rr=np.array([x['return_'] for x in item['rows']]); bas=(rr.sum()-rr)/2;adv=rr-bas
        assert np.allclose(item['baseline'],bas,rtol=0,atol=1e-12)
        assert np.allclose(item['advantages'],adv,rtol=0,atol=1e-12)
        ref=-sum(w*av*r['logprob_sum'] for w,av,r in zip([1,.25,.25],adv,item['rows']))/1.5
        assert np.isclose(ref,item['loss'],rtol=1e-7,atol=1e-7)
        assert len({r['sampling_seed'] for r in item['rows']})==3
        for r in item['rows']:
            assert abs(sum(r['rewards'])+r['J'])<TOL['reward_atol']
            assert all(n in menu for n,menu in zip(r['actions'],r['masks_local_nodes']))
            assert (r['logprobs'][0] is None)==(r['role']!='primary')
            assert sum(x or 0. for x in r['logprobs'])==r['logprob_sum'] or np.isclose(sum(x or 0. for x in r['logprobs']),r['logprob_sum'],atol=1e-6)
    write(OUT/'gradients.json',dict(joint=ra,accumulate=rb,max_absolute_gradient_difference=max_abs,max_relative_gradient_difference=max_rel))
    return dict(max_absolute_gradient_difference=max_abs,max_relative_gradient_difference=max_rel,gradient_norm=ra['gradient_norm'],optimizer_steps=0)


def rng_and_failures():
    v=tiny();t=trainer(); states=t.streams.state_dict()
    seeds=[t.streams.seed('RL',0,0,r) for r in ['primary','forced_low','forced_high']];assert len(set(seeds))==3
    data=t.streams.numpy('S2_data');data.integers(100,size=7); state=t.streams.state_dict(); expected=data.integers(100,size=20)
    q=NamedStreams(t.streams.protocol_sha256,t.streams.optimizer_seed);q.load_state_dict(state);assert np.array_equal(expected,q.numpy('S2_data').integers(100,size=20))
    reject(lambda:t.teacher_forcing_loss(v,[1.5]));reject(lambda:t.teacher_forcing_loss(v,[True]))
    reject(lambda:t.teacher_forcing_loss(v,[]));reject(lambda:t.teacher_forcing_loss(v,[999]))
    empty=trainer(); before=tensor_hash(empty.model);reject(lambda:empty.s2_step([tiny('empty')]*4),(InconclusiveBatch,));assert before==tensor_hash(empty.model) and empty.ledger.get('optimizer_updates',0)==0
    bad=trainer();next(bad.model.parameters()).data.fill_(float('nan'));reject(lambda:bad.s2_step([v]*4),(FloatingPointError,));assert bad.ledger.get('optimizer_updates',0)==0 and bad.stopped
    one=trainer();r=one.s2_step([tiny('one')]*4);assert r['zero_signal'];assert one.ledger['optimizer_updates']==1
    env=Environment(v,MockTerminal());reject(lambda:env.step(999))
    return dict(role_seeds=seeds,numpy_restore_exact=True,expected_rejections=7,one_first_zero_signal_optimizer_steps=1,all_empty_optimizer_steps=0)


def resume_case(phase, total, cut):
    v=tiny();pool=[(v,labels(v)),(v,list(reversed(labels(v))))]
    # The reverse label order need not cover legally with positive gain; use one
    # fixed complete route pool for this fixture, with independent index draws.
    pool=[pool[0],pool[0],pool[0]]
    continuous, split=trainer(phase),trainer(phase)
    def advance(t,n):
        rows=[]
        for _ in range(n):
            data=t.streams.numpy('S1_data' if phase=='S1' else 'S2_data')
            indices=data.integers(len(pool),size=4).tolist()
            out=t.s1_step([pool[i] for i in indices]) if phase=='S1' else t.s2_step([v]*4)
            rows.append(dict(indices=indices,diagnostics=clean_diag(out)))
        return rows
    expected=advance(continuous,total);first=advance(split,cut)
    path=OUT/'checkpoints'/f'{phase}_interrupted.pt';save_checkpoint(path,split,extra={'indices':first,'engineering_only':True})
    restored=trainer(phase);extra=load_checkpoint(path,restored)
    assert extra['indices']==first
    actual=first+advance(restored,total-cut)
    assert expected==actual
    assert equal(continuous.model.state_dict(),restored.model.state_dict())
    assert equal(continuous.optimizer.state_dict(),restored.optimizer.state_dict())
    assert equal(continuous.streams.state_dict(),restored.streams.state_dict())
    assert continuous.ledger==restored.ledger and continuous.cursors==restored.cursors
    assert continuous.weight_version==restored.weight_version and continuous.terminal.calls==restored.terminal.calls
    for name,t in [('continuous',continuous),('resumed',restored)]: save_checkpoint(OUT/'checkpoints'/f'{phase}_{name}.pt',t,extra={'engineering_only':True})
    # Wrong binding, wrong phase and mutated bytes must reject before touching weights.
    untouched=tensor_hash(restored.model)
    reject(lambda:load_checkpoint(path,restored,expected_binding={'wrong':'protocol'}))
    other=trainer('continued_SL' if phase=='S1' else 'S1');reject(lambda:load_checkpoint(path,other))
    corrupt=OUT/'checkpoints'/f'{phase}_corrupt.pt';shutil.copyfile(path,corrupt);shutil.copyfile(path.with_suffix('.pt.json'),corrupt.with_suffix('.pt.json'))
    data=bytearray(corrupt.read_bytes());data[len(data)//2]^=1;corrupt.write_bytes(data)
    reject(lambda:load_checkpoint(corrupt,restored));assert tensor_hash(restored.model)==untouched
    # Preserve failed bytes/receipt as an expected negative fixture.
    reject(lambda:save_checkpoint(path,restored),(FileExistsError,))
    write(OUT/f'{phase}_resume_trace.json',dict(continuous=expected,resumed=actual))
    return dict(exact_model=True,exact_Adam=True,exact_NamedStreams=True,exact_cursors_and_ledger=True,exact_sampling_and_actions=True,
      continuous_updates=total,interrupted_updates=cut,resumed_updates=total-cut,total_executed_optimizer_steps=2*total,
      checkpoint_paths=[str((OUT/'checkpoints'/f'{phase}_{x}.pt').relative_to(HERE)) for x in ['continuous','resumed']],expected_checkpoint_rejections=4)


def fork_case():
    v=tiny();warm=trainer('S1');untrained=copy.deepcopy(warm.model);untrained_hash=tensor_hash(untrained)
    warm.s1_step([(v,labels(v))]*4);start=tensor_hash(warm.model);assert start!=untrained_hash
    forks=fork_from_supervised(warm.model);assert all(x['tensor_sha256']==start for x in forks.values())
    models=[warm.model,untrained]+[x['model'] for x in forks.values()]
    for i,x in enumerate(models):
        for y in models[i+1:]: assert not {p.data_ptr() for p in x.parameters()} & {p.data_ptr() for p in y.parameters()}
    assert not forks['RL']['optimizer'].state and not forks['continued_SL']['optimizer'].state
    active=trainer(model=forks['RL']['model'],optimizer=forks['RL']['optimizer'])
    env=Environment(v,MockTerminal());encoded=active.model.encode(v)
    active.s2_step([v]*4);reject(lambda:active.model.score(env,encoded))
    fresh=active.model.encode(v);assert torch.isfinite(active.model.score(env,fresh)[torch.tensor(env.mask())]).all()
    cont=trainer('continued_SL',model=forks['continued_SL']['model'],optimizer=forks['continued_SL']['optimizer'])
    cont.s1_step([(v,labels(v))]*4)
    assert tensor_hash(warm.model)==start==tensor_hash(forks['frozen_SL']['model'])
    assert tensor_hash(untrained)==untrained_hash
    assert tensor_hash(active.model)!=start and tensor_hash(cont.model)!=start
    for name,t in [('fork_RL',active),('fork_continued_SL',cont),('warm_SL',warm)]: save_checkpoint(OUT/'checkpoints'/f'{name}.pt',t,extra={'engineering_only':True})
    return dict(weight_clones_initially_exact=True,storage_disjoint=True,fresh_Adam_states=True,untrained_and_frozen_unchanged=True,stale_cache_rejected=True,optimizer_steps=3)


def full_parent_case():
    stats={k:0 for k in ('terminal_requests','terminal_cache_hits','terminal_attempts','terminal_completed','terminal_failed','full_H_calls','export_attempts','exports','validation_attempts','validation_completed')}
    def record(r):
        if r['event']=='terminal_start': BUDGET.charge('real_full_terminal_attempts')
        from .verification_budget import append
        append(OUT/'real_terminal_journal.jsonl',r)
    H=json.loads((ROOT/'rc3_global_local/S0_protocol.json').read_text(encoding='utf8'))['H_sha256']
    terminal=TerminalCache(stats,record,H,24,time.perf_counter()+300)
    parents=[load_saved_s0_parent(i) for i in range(3)]
    views=[extract_snapshot(p,0,0,10,10) for p in parents];views.append(views[0])
    torch.manual_seed(11622026);model=GlobalLocalPolicy()
    ck=torch.load(ROOT/'rc3_global_local/initial_policy.pt',weights_only=False,map_location='cpu');model.load_state_dict(ck['state'])
    t=trainer(model=model,terminal=terminal);t.config['binding'].update(dataset='frozen S0 saved parents full10x10 batch4',H=H)
    before=tensor_hash(model);internal=_empty_internal();rss_before=psutil.Process().memory_info().rss
    BUDGET.full_parent=True
    try:
        with _instrument(internal): result=t.s2_step(views)
    finally: BUDGET.full_parent=False
    assert len(result['instances'])==4 and stats['terminal_requests']==12 and stats['terminal_failed']==0
    assert stats['validation_completed']==stats['terminal_attempts'] and tensor_hash(model)!=before
    save_checkpoint(OUT/'checkpoints'/'engineering_full_batch4.pt',t,extra={'engineering_only':True,'formal_S2_updates':0,'source':'saved S0 fixtures'})
    write(OUT/'full_parent_batch.json',dict(diagnostics=result,terminal_stats=stats,internal_search=internal,
      model_hash_before=before,model_hash_after=tensor_hash(model),RSS_before=rss_before,RSS_after=psutil.Process().memory_info().rss,
      parent_ids=[p.identity for p in parents],scope='Engineering optimizer acceptance only, no learned-performance claim'))
    return dict(batch_instances=4,complete_trajectories=12,optimizer_steps=1,formal_S2_updates=0,terminal_stats=stats,internal_search=internal,
      model_changed=True,gradient_norm_before_clip=result['gradient_norm_before_clip'],gradient_norm_after_clip=result['gradient_norm_after_clip'])


def formal_gate_case():
    from .approval import require_authority, AdmissionError, V2, V3, REGISTRY_V2, REGISTRY_V3, output_directory
    from .data import SceneProvider, TrainingRegistry
    before=BUDGET.counts.copy()
    for args in [(V2,REGISTRY_V2,None,None,'S1'),(V3,REGISTRY_V3,None,None,'S1'),
                 (V3,REGISTRY_V3,None,None,'S2'),(V3,REGISTRY_V3,None,None,'S3'),
                 (V3,REGISTRY_V2,None,None,'S1')]:
        reject(lambda:require_authority(*args),(AdmissionError,))
    registry=TrainingRegistry(REGISTRY_V3)
    reject(lambda:SceneProvider(registry,OUT/'forbidden_formal',{},None),(PermissionError,))
    reject(lambda:output_directory(ROOT/'rc3_global_local'),(AdmissionError,))
    from .runner import main
    assert main(['--stage','S2','--output',str(OUT/'forbidden_formal')])==2
    assert main(['--stage','S1','--protocol',str(V3),'--registry',str(REGISTRY_V3),'--output',str(OUT/'forbidden_formal')])==2
    assert not (OUT/'forbidden_formal').exists()
    assert before==BUDGET.counts
    return dict(no_approval_rejected=True,expected_rejections=9,optimizer_or_model_calls_added=0)


def orchestration_case():
    from .runner import TrainingRun, StageClock
    from .data import TrainingRegistry
    from .approval import REGISTRY_V3
    class FixtureProvider:
        def __init__(self): self.ledger={};self.requests=[]
        def get(self,spec): self.requests.append(copy.deepcopy(spec));return tiny()
        def state_dict(self): return dict(unit_fixture_only=True,requests=self.requests)
    class FixturePool:
        def example(self,index):
            v=tiny();return dict(view=v,actions=labels(v))
    # Low-level unit harness: never creates a formal ExecutionPermit or calls
    # real SceneProvider. It exercises the production phase/update/journal loop.
    run=object.__new__(TrainingRun);run.registry=TrainingRegistry(REGISTRY_V3);run.stop_requested=False
    rows=[]
    for phase in ['S1','RL']:
        t=trainer(phase);t.config['max_optimizer_updates']=1
        provider=FixtureProvider();clock=StageClock(60);folder=OUT/'orchestration'/phase;folder.mkdir(parents=True)
        assert run._loop(t,provider,clock,folder,FixturePool())
        assert t.cursors[phase]==1 and (folder/'update_0001.pt').exists()
        event=json.loads((folder/'updates.jsonl').read_text(encoding='utf8').splitlines()[0])
        assert event['phase']==phase and event['optimizer_updated']
        rows.append(dict(phase=phase,optimizer_steps=1,production_loop_and_checkpoint=True,unit_provider=True))
    return dict(phases=rows,new_scene_or_teacher_queries=0,total_optimizer_steps=2)


def sampling_case():
    from .data import TrainingRegistry, TrainingSchedule
    from .approval import REGISTRY_V3
    reg=TrainingRegistry(REGISTRY_V3)
    a=NamedStreams('engineering_schedule_unit',77); schedule=TrainingSchedule(reg,a)
    rows=[]
    for u in range(512): rows.extend(schedule.batch(u))
    full=[s for s in rows if s['width']==10];local=[s for s in rows if s['width']!=10]
    assert len(full)==len(local)==1024
    assert [sum((s['width'],s['height'])==size for s in local) for size in [(4,3),(6,6),(8,8)]]==[342,341,341]
    assert sum(s['endpoints']=='closed_depot' for s in local)==512
    assert all(0<=s['x0']<=10-s['width'] and 0<=s['y0']<=10-s['height'] for s in rows)
    for s in rows: reg.validate_spec(s)
    left=NamedStreams('engineering_schedule_unit',77);s1=TrainingSchedule(reg,left)
    prefix=[s1.batch(u) for u in range(7)];saved=left.state_dict()
    future=[s1.batch(u) for u in range(7,16)]
    right=NamedStreams('engineering_schedule_unit',77);right.load_state_dict(saved)
    assert future==[TrainingSchedule(reg,right).batch(u) for u in range(7,16)]
    bad=dict(rows[0],role='S3');reject(lambda:reg.validate_spec(bad))
    assert len(list(reg.teacher_specs()))==256 and len(list(reg.development_specs()))==18
    write(OUT/'sampling_schedule.json',dict(first_batches=prefix,following_after_restore=future,full=1024,local=1024,local_sizes=[342,341,341]))
    return dict(full_instances=1024,local_instances=1024,local_sizes=[342,341,341],schedule_restore_exact=True,S3_access_rejected=True,target_tasks_generated=0)


def teacher_partial_case():
    from .runner import TeacherPool, StageClock
    before=BUDGET.counts.copy()
    class Registry:
        sha256='engineering_fixture_not_a_formal_registry'
        def teacher_specs(self): return iter([{'fixture':True}])
    class Provider:
        permit=SimpleNamespace(binding={'engineering_fixture':True})
        def get(self,spec): raise RuntimeError('Expected mock provider failure, no generation or query')
    def pool(folder): return TeacherPool(folder,Registry(),Provider(),{},StageClock(60),{'fixture_H':'never_called'})
    for filename in ['events.jsonl','STARTED.json']:
        folder=OUT/'partial_teacher'/filename.replace('.','_');folder.mkdir(parents=True)
        (folder/filename).write_text('{}\n',encoding='utf8')
        reject(lambda:pool(folder).build())
    folder=OUT/'first_failed_teacher'
    reject(lambda:pool(folder).build())
    assert (folder/'STARTED.json').exists() and not (folder/'POOL_INDEX.json').exists()
    reject(lambda:pool(folder).build())
    assert before==BUDGET.counts
    return dict(preexisting_events_rejected=True,preexisting_start_marker_rejected=True,
      first_failure_persisted_before_provider=True,implicit_retry_rejected=True,
      optimizer_steps=0,teacher_queries=0,new_scenes_or_navigation=0,expected_rejections=4)


CASES={'gradient':gradient_contract,'rng_edges':rng_and_failures,'resume_S2':lambda:resume_case('RL',4,2),
       'resume_S1':lambda:resume_case('S1',3,1),'fork':fork_case,'full_parent_batch4':full_parent_case,'formal_gate':formal_gate_case,'sampling':sampling_case,'orchestration':orchestration_case,'teacher_partial':teacher_partial_case}


def run(binding_path, selected=None):
    global BINDING,BUDGET,OUT,TOL
    binding_path=Path(binding_path);binding=json.loads(binding_path.read_text(encoding='utf8'))
    assert binding['engineering_protocol_sha256']==sha(HERE/'ENGINEERING_VERIFICATION_PROTOCOL.json')
    for name,h in binding['source_sha256'].items(): assert sha(ROOT/name)==h,name
    for name,h in binding['input_sha256'].items(): assert sha(ROOT/name)==h,name
    assert binding['tolerance_sha256']==sha(HERE/'VERIFICATION_TOLERANCES.json')
    TOL=json.loads((HERE/'VERIFICATION_TOLERANCES.json').read_text(encoding='utf8'))
    BINDING=sha(binding_path);OUT=HERE/'verification_runs'/binding['run_id'];OUT.mkdir(parents=True,exist_ok=False)
    BUDGET=AcceptanceBudget(binding['run_id'],BINDING)
    started=time.perf_counter();result=dict(schema='rc3-trainer-verification-run-v1',binding=BINDING,run_id=binding['run_id'],status='RUNNING',source_sha256=binding['source_sha256'],cases={},error=None)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    try:
        with threadpool_limits(limits=1), BUDGET.instrument():
            for name in (selected or list(CASES)):
                BUDGET.record('case_start',case=name);tick=time.perf_counter()
                result['cases'][name]=dict(status='PASS',evidence=CASES[name](),wall_s=time.perf_counter()-tick)
                BUDGET.record('case_complete',case=name)
                write(OUT/'RESULT.json',result);gc.collect()
        result['status']='PASS'
    except BaseException as exc:
        result['status']='FAIL';result['error']=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
    finally:
        result['wall_s']=time.perf_counter()-started;result['cumulative_counters']=BUDGET.counts
        result['peak_RSS_os_bytes']=getattr(psutil.Process().memory_info(),'peak_wset',None)
        write(OUT/'RESULT.json',result);BUDGET.end(result['status'])
    print(json.dumps(result,ensure_ascii=True))
    if result['status']!='PASS': raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--binding',required=True);parser.add_argument('--case',action='append',choices=list(CASES));args=parser.parse_args()
    run(args.binding,args.case)
