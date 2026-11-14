"""Complete sampled plans, counted replays, exact policy-gradient and recovery."""
from __future__ import annotations
import copy,hashlib,os,random,time
from pathlib import Path
import numpy as np
import torch
from rc3_global_local.environment import Environment
from rc3_global_local.terminal import finish,export
from rc3_global_local.validation import validate
from rc3_relation_drl_20261113.actor import RelationActor,tensor_hash
from .common import ROOT,HOME,H_ID,put,read,sha,digest,replace_retry
from .runtime_boundary import persist_request

def setup(seed,device='cuda'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    return torch.device(device)
def sync(device):
    if torch.device(device).type=='cuda':torch.cuda.synchronize()
def memory(device):
    if torch.device(device).type!='cuda':return dict(peak_allocated=0,peak_reserved=0)
    sync(device);return dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
def instrument(actor,ledger):
    if getattr(actor,'_metered',False):return actor
    enc,score=actor.encode,actor.score
    def encoding(*args,**kwargs):ledger.check();ledger.counts['encode_calls']+=1;return enc(*args,**kwargs)
    def scoring(*args,**kwargs):ledger.check();ledger.counts['score_calls']+=1;return score(*args,**kwargs)
    actor.encode=encoding;actor.score=scoring;actor._metered=True;return actor
def request_record(ledger,rid,value,tick):
    value['full_request_s']=persist_request(ledger,rid,value,tick)
    if 'timing' in value:value['timing']['complete_request_s']=value['full_request_s']
    return value


def collect(actor,pool,tid,ledger,rid,seed=None,greedy=False):
    tick=ledger.begin('request',rid,task=tid,purpose='policy_sample' if not greedy else 'greedy')
    try:
        begin=time.perf_counter();p,source=pool.get(tid);input_s=time.perf_counter()-begin
        device=next(actor.parameters()).device;sync(device)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        generator=None if greedy else torch.Generator(device=device).manual_seed(int(seed))
        def terminal(problem,raw):
            ledger.check();ledger.counts['H_calls']+=int(bool(raw[1:-1]));return finish(problem,raw)
        env=Environment(p,terminal);actions=[];logps=[];mh=hashlib.sha256();before=tensor_hash(actor)
        decode=time.perf_counter()
        with torch.no_grad():
            encoded=actor.encode(p) if not env.done else None
            while not env.done:
                z=actor.score(env,encoded);mask=torch.tensor(env.mask(),device=device)
                if not torch.isfinite(z[mask]).all() or not torch.isneginf(z[~mask]).all():raise ValueError('Logit/mask contract')
                probabilities=torch.softmax(z,0)
                if bool(probabilities[~mask].any()):raise ValueError('Illegal action mass')
                a=int(z.argmax()) if greedy else int(torch.multinomial(probabilities,1,generator=generator).item())
                mh.update(np.packbits(env.mask()).tobytes());actions.append(a+1);logps.append(float(torch.log_softmax(z,0)[a]));env.step(a+1)
        sync(device);decode_s=time.perf_counter()-decode;r=env.result
        if abs(sum(env.rewards)+r['J'])>1e-7:raise ValueError('Complete reward identity')
        raw_plan=export(p,r['raw_route'],'relation_raw');raw_certificate=validate(p,raw_plan)
        if not raw_certificate['valid_model_plan']:raise ValueError(raw_certificate['errors'])
        raw_plan['validation']=raw_certificate
        if tensor_hash(actor)!=before:raise ValueError('Collection mutated weights')
        record=dict(request_id=rid,task_id=tid,purpose='policy',problem_identity=p.identity,parent_identity=p.parent.identity,
            physical_key=source['physical_key'],candidate_count=len(p.candidate_ids),task_count=len(p.task_ids),entry=p.entry,exit=p.exit,
            actor_tensor_sha256=before,sample_seed=seed,greedy=greedy,actions=actions,sample_log_prob=logps,mask_trace_sha256=mh.hexdigest(),
            metrics={k:r[k] for k in ('J','L','K','raw_J','raw_L','raw_K')},raw_route=r['raw_route'],route=r['route'],raw_plan=raw_plan,plan=r['plan'],
            reward_sum=sum(env.rewards),H_identity=H_ID,input_sources=source['input_source_files'],valid=True,
            timing=dict(input_s=input_s,decode_with_H_s=decode_s,H_s=r['H_s'],export_s=r['export_s'],validation_s=r['validation_s'],compute_before_output_s=time.perf_counter()-tick),
            memory=memory(device))
        return request_record(ledger,rid,record,tick)
    except BaseException as exc:ledger.fail('request',rid,exc);raise

def group_summary(records):
    costs=np.asarray([r['metrics']['J'] for r in records]);raw=[tuple(r['raw_route']) for r in records];post=[tuple(r['route']) for r in records]
    pairs=len(records)*(len(records)-1)//2
    return dict(tasks=[r['task_id'] for r in records],J_H=costs.tolist(),range=float(np.ptp(costs)),std=float(costs.std()),
        distinct_H_costs=len({round(float(j),7) for j in costs}),raw_route_duplicate_rate=1-len(set(raw))/len(raw),
        H_route_duplicate_rate=1-len(set(post))/len(post),equal_cost_pair_rate=sum(abs(costs[i]-costs[j])<=1e-7 for i in range(len(costs)) for j in range(i))/max(pairs,1))

def replay(actor,p,actions,ledger,rid,terminal_J=None,encoded=None,teacher_source=None,expected_logps=None):
    tick=ledger.begin('request',rid,task=p.identity,purpose='teacher_replay' if terminal_J is None else 'gradient_replay')
    try:
        terminal=(lambda problem,route:problem.metrics(route)) if terminal_J is None else (lambda problem,route:dict(J=float(terminal_J)))
        env=Environment(p,terminal);logps=[]
        cache=actor.encode(p) if encoded is None else actor.fork_cache(encoded)
        for n in actions:
            if env.done or not env.mask()[n-1]:raise ValueError('Replay action violates current positive-coverage mask')
            z=actor.score(env,cache);logps.append(torch.log_softmax(z,0)[n-1]);env.step(int(n))
        if not env.done:raise ValueError('Incomplete replay')
        if not logps:raise ValueError('Empty training trajectory cannot produce a policy update')
        lp=torch.stack(logps)
        maxdiff=None
        if expected_logps is not None:
            saved=lp.new_tensor(expected_logps);maxdiff=float((lp.detach()-saved).abs().max())
            if not torch.allclose(lp.detach(),saved,rtol=1e-5,atol=2e-5):raise ValueError('Recomputed on-policy log probability mismatch')
        certificate=None
        if teacher_source is not None:
            plan=export(p,[0,*map(int,actions),0],'saved_teacher_replay');certificate=validate(p,plan)
            if not certificate['valid_model_plan']:raise ValueError(certificate['errors'])
        ledger.counts['teacher_replays' if terminal_J is None else 'gradient_replays']+=1
        request_record(ledger,rid,dict(request_id=rid,purpose='teacher_replay' if terminal_J is None else 'gradient_replay',
            problem_identity=p.identity,actions=list(map(int,actions)),complete=True,new_H=False,new_teacher=False,
            J=float(env.result['J']),teacher_source=teacher_source,teacher_validation=certificate,max_logprob_error=maxdiff),tick)
        return lp
    except BaseException as exc:ledger.fail('request',rid,exc);raise

def optimizer(actor):return torch.optim.Adam(actor.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)

def apply_update(actor,opt,loss,ledger,rid,details,already_backward=False):
    before=tensor_hash(actor);versions=[p._version for p in actor.parameters()]
    if not already_backward:ledger.counts['backward_calls']+=1;loss.backward()
    grads={n:None if p.grad is None else float(p.grad.norm().detach().cpu()) for n,p in actor.named_parameters()}
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in actor.parameters()):raise ValueError('Missing/nonfinite relation gradient')
    norm=torch.nn.utils.clip_grad_norm_(actor.parameters(),1.,error_if_nonfinite=True)
    tick=ledger.begin('optimizer',rid)
    try:
        opt.step();sync(next(actor.parameters()).device)
        if any(not torch.isfinite(p).all() for p in actor.parameters()):raise ValueError('Nonfinite updated parameter')
        after=tensor_hash(actor);newversions=[p._version for p in actor.parameters()]
        result=dict(id=rid,loss=float(loss.detach().cpu()),gradient_norm=float(norm),gradient_norms=grads,before=before,after=after,
            weight_changed=before!=after,versions_before=versions,versions_after=newversions,details=details)
        put(ledger.folder/'updates'/(rid+'.json'),result);ledger.complete('optimizer',rid,tick,before=before,after=after)
        opt.zero_grad(set_to_none=True);return result
    except BaseException as exc:ledger.fail('optimizer',rid,exc);raise

def rl_update(actor,opt,groups,pool,ledger,rid):
    opt.zero_grad(set_to_none=True);total=0.;details=[];current=tensor_hash(actor)
    for gi,records in enumerate(groups):
        if len(records)!=4 or len({r['task_id'] for r in records})!=1 or any(r['actor_tensor_sha256']!=current for r in records):raise ValueError('Exact own-policy G4 group required')
        tid=records[0]['task_id'];p,_=pool.get(tid);scale=pool.reference(tid,p)
        costs=np.array([r['metrics']['J'] for r in records]);baseline=(costs.sum()-costs)/3.;weights=(costs-baseline)/scale
        cache=actor.encode(p);terms=[]
        for k,r in enumerate(records):
            lp=replay(actor,p,r['actions'],ledger,f'{rid}_replay_{gi}_{k}',r['metrics']['J'],cache,expected_logps=r['sample_log_prob'])
            terms.append(lp.sum()*float(weights[k])/4/len(groups))
        loss=torch.stack(terms).sum();total+=float(loss.detach());ledger.counts['backward_calls']+=1;loss.backward()
        details.append(dict(task_id=tid,reference_scale=scale,costs=costs.tolist(),loo_baselines=baseline.tolist(),weights=weights.tolist(),requests=[r['request_id'] for r in records]))
        del loss,terms,cache,lp
    return apply_update(actor,opt,torch.tensor(total),ledger,rid,details,already_backward=True)

def sl_update(actor,opt,tids,pool,ledger,rid):
    opt.zero_grad(set_to_none=True);total=0.;details=[]
    for i,tid in enumerate(tids):
        p,actions,source=pool.teacher(tid)
        lp=replay(actor,p,actions,ledger,f'recovery_v1_{rid}_teacher_{i}',teacher_source=source)
        loss=-lp.mean()/len(tids);total+=float(loss.detach());ledger.counts['backward_calls']+=1;loss.backward()
        details.append(dict(task_id=tid,teacher_source=source,actions=len(actions)));del loss,lp
    return apply_update(actor,opt,torch.tensor(total),ledger,rid,details,already_backward=True)

def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
def set_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])

def state_digest(value):
    h=hashlib.sha256()
    def walk(v):
        if torch.is_tensor(v):h.update(str((tuple(v.shape),v.dtype)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str((v.shape,v.dtype)).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());walk(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:walk(x)
        else:h.update(repr(v).encode())
    walk(value);return h.hexdigest()

def checkpoint(actor,opt,ledger,path,cursor,engineering_only=True):
    if ledger.pending:raise ValueError('Cannot checkpoint an incomplete resource transaction')
    payload=dict(schema='rc3-relation-checkpoint-v1',identity=ledger.identity,config=actor.config,engineering_only=engineering_only,
        model=copy.deepcopy(actor.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=rng_state(),cursor=cursor,
        counts=dict(ledger.counts),request_ids=sorted(ledger.ids),resource_elapsed_s=ledger.elapsed())
    payload['model_sha256']=tensor_hash(actor);payload['optimizer_sha256']=state_digest(payload['optimizer']);payload['rng_sha256']=state_digest(payload['rng'])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.pending')
    with tmp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    replace_retry(tmp,path);ref=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),cursor=cursor,model_sha256=payload['model_sha256'],optimizer_sha256=payload['optimizer_sha256'],rng_sha256=payload['rng_sha256'])
    put(path.with_suffix('.json'),ref);return ref

def restore(actor,opt,ledger,ref,engineering_only=True):
    path=ROOT/ref['path']
    if sha(path)!=ref['sha256']:raise ValueError('Checkpoint bytes')
    p=torch.load(path,map_location='cpu',weights_only=False)
    if p['schema']!='rc3-relation-checkpoint-v1' or p['identity']!=ledger.identity or p['engineering_only']!=engineering_only or p['config']!=actor.config:raise ValueError('Checkpoint identity/role mismatch')
    if ledger.pending or set(p['request_ids'])!=ledger.ids:raise ValueError('Restore requires exact closed query journal boundary')
    for key in ('request_attempts','optimizer_attempts','optimizer_completed'):
        if p['counts'].get(key,0)!=ledger.counts[key]:raise ValueError('Cannot reset/replay paid resources')
    actor.load_state_dict(p['model'],strict=True);opt.load_state_dict(p['optimizer']);set_rng(p['rng'])
    if tensor_hash(actor)!=p['model_sha256'] or state_digest(opt.state_dict())!=p['optimizer_sha256'] or state_digest(rng_state())!=p['rng_sha256']:raise ValueError('Restored model/Adam/RNG mismatch')
    return p

def export_actor(actor,path,engineering_only,identity):
    p=dict(schema='rc3-relation-actor-v1',config=actor.config,model={k:v.detach().cpu() for k,v in actor.state_dict().items()},
        model_sha256=tensor_hash(actor),engineering_only=engineering_only,identity=identity)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:torch.save(p,f)
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),model_sha256=p['model_sha256'],engineering_only=engineering_only)
def load_export(ref,device='cuda',allow_engineering=False):
    if sha(ROOT/ref['path'])!=ref['sha256']:raise ValueError('Export bytes')
    p=torch.load(ROOT/ref['path'],map_location='cpu',weights_only=False)
    if p['schema']!='rc3-relation-actor-v1' or (p['engineering_only'] and not allow_engineering):raise ValueError('Engineering models cannot initialize science')
    cfg={k:v for k,v in p['config'].items() if k!='model_id'};actor=RelationActor(**cfg).to(device);actor.load_state_dict(p['model'],strict=True)
    if tensor_hash(actor)!=p['model_sha256']:raise ValueError('Export tensor identity')
    return actor
