"""Preregistered alignment, two policy-evaluation rounds, and untouched test."""
import copy,gzip,json,pickle,time
import numpy as np
import torch
from rc3_rl_g2.adapter_g2 import ResearchAdapterG2
from rc3_rl_g2.train_ppo import configure
from rc3_fast.scenes import make_complex_scene
from rc3_fast.solver import prepare,route_objective
from rc3_v3.environment import Env as OriginalEnv
from rc3_improve.repair import repair_window,_local
from rc3_v4.runtime import run as original_run
from .core import ROOT,WS,Ledger,LimitReached,Env,SearchConfig,read,write,sha,jsonable,digest
from .learning import Value,observation,fit,save

def make(case,ledger,p):
    ledger.use('cold_plans');scene=make_complex_scene(case['nx'],case['ny'],case['seed'],case['mode'])
    adapter=ResearchAdapterG2.from_scene(scene,.75);adapter.problem_id='g23:'+adapter.problem_id
    return Env(adapter,ledger,case['seed'],SearchConfig(**p['config']))

def choose(env,policy):return env.rule() if policy is None else policy.select(env)
def stop(env):
    if not env.done:
        obs=env.observe();env.step(next(i for i,a in enumerate(obs['actions']) if a['kind']=='stop'))
    env.finalize();return env.total_reward
def follow(env,policy,deadline=float('inf')):
    while not env.done:
        if time.perf_counter()>=deadline:env.terminate('complete_soft_deadline');break
        a=choose(env,policy)
        if time.perf_counter()>=deadline:env.terminate('complete_soft_deadline');break
        env.step(a)
    env.finalize();return env.total_reward

def alignment(p,ledger):
    ledger.phase='alignment';traces=[]
    for case_id in p['alignment_cases']:
        source=read(WS/f'rc3_rl_g2/outputs/teacher_extra/case_{case_id:04d}/input.json')['source']['scene']
        ledger.use('cold_plans');adapter=ResearchAdapterG2.from_scene(source,.75);old=OriginalEnv(adapter.problem,16,adapter.anchor.route)
        original=[]
        for i in range(16):
            f,mask=old.observe()
            if not mask.any():break
            a=int(np.where(mask,f[:,8],-np.inf).argmax());window=dict(old.windows[a]);before=list(old.route)
            start,end,required,pool,_=_local(adapter.problem,before,window)
            ledger.use('nonstop',kind_detail='original_reference',case_id=case_id,window=window)
            old.step(a);original.append(dict(window=window,before=before,after=list(old.route),required=required.tolist(),pool=pool.tolist()))
        ledger.use('cold_plans');adapter2=ResearchAdapterG2.from_scene(source,.75);adapter2.problem_id='g23:'+adapter2.problem_id
        aligned=Env(adapter2,ledger,source['seed'],SearchConfig(**p['config']));compared=[]
        while not aligned.done and len(compared)<len(original):
            a=aligned.rule();o=aligned.observe()
            if o['actions'][a]['kind']=='stop':break
            reference=original[len(compared)];chosen=o['actions'][a]
            if list(aligned.current)!=reference['before'] or (chosen['start'],chosen['end'])!=(reference['window']['start'],reference['window']['end']):
                raise ValueError('Rule semantic operation mismatch')
            before_cost=aligned.current_cost;aligned.step(a)
            if list(aligned.current)!=reference['after']:raise ValueError('Original dense transition/prefix mismatch')
            compared.append(dict(action=chosen,route=list(aligned.current),before_J=before_cost,after_J=aligned.current_cost))
        if not compared:raise ValueError('No aligned rule prefix')
        stop(aligned)
        traces.append(dict(case_id=case_id,original=original,aligned=compared,matched_prefix=len(compared),work=aligned.work_used,
            valid=aligned.returned_certificate.plan['validation']['valid_model_plan']))
    write(ROOT/'outputs/alignment.json',dict(passed=True,cases=traces,scope='Exact semantic windows, required/pool records and original dense accepted routes; prefix limited by shared resources'))

def collect(p,ledger,cases,policy,phase,static=False):
    data=[];ledger.phase=phase;frozen=None if policy is None else copy.deepcopy(policy).eval()
    frozen_before=None if frozen is None else digest({k:v.detach().numpy() for k,v in frozen.state_dict().items()})
    for case in cases:
        parent=make(case,ledger,p)
        for state_index in range(2):
            if parent.done:break
            obs=parent.observe();raw=observation(parent)
            baseline=parent.fork();qstop=stop(baseline)-parent.total_reward
            rule=parent.rule();proposal=choose(parent,frozen)
            legal=[i for i,(a,yes) in enumerate(zip(obs['actions'],obs['mask'])) if yes and a['kind']!='stop']
            order=[]
            for a in [rule,proposal,*legal[::max(len(legal)//3,1)],*legal]:
                if a in legal and a not in order:order.append(a)
            for a in order[:3]:
                child=parent.fork();prefix=child.total_reward;_,r,done,_,info=child.step(a)
                one=child.fork();one_return=stop(one)-prefix
                if static:
                    control=child.fork();control_return=follow(control,None)-prefix
                else:control_return=None
                multi_return=follow(child,frozen)-prefix
                data.append(dict(observation=raw,action=a,one=one_return-qstop,multi=multi_return-qstop,
                    static=multi_return-qstop if control_return is None else control_return-qstop,
                    policy_id='rule' if frozen_before is None else frozen_before,remaining_h=obs['steps_remaining'],
                    remaining_work=obs['remaining_work'],qstop=qstop,step_reward=float(r),case=case,
                    final_correction=child.finalization['quality_correction'],termination=child.stop_reason))
            if state_index==0:
                for _ in range(2):
                    if parent.done:break
                    parent.step(choose(parent,frozen))
        stop(parent)
    if frozen is not None and digest({k:v.detach().numpy() for k,v in frozen.state_dict().items()})!=frozen_before:
        raise ValueError('Policy being evaluated drifted during collection')
    if not data:raise ValueError('No actual value labels collected')
    with gzip.open(ROOT/f'outputs/{phase}.pkl.gz','wb') as f:pickle.dump(data,f)
    return data

def test(p,ledger,models):
    ledger.phase='final_test';rows=[];methods=['original_fixed3','original_detour16','same_interface_rule',*models]
    for i,case in enumerate(p['splits']['test']):
        for rep in range(p['repetitions']):
            shift=(i+rep)%len(methods);order=methods[shift:]+methods[:shift]
            for method in order:
                started=time.perf_counter();deadline=started+p['complete_soft_s']
                if method.startswith('original_'):
                    ledger.use('cold_plans');scene=make_complex_scene(10,10,case['seed'],case['mode']);problem=prepare(scene,.75)
                    # Read-only original runtime, instrument each actual primitive via a local copied module namespace.
                    import types
                    from rc3_v3 import environment as v3
                    namespace=dict(vars(v3));base=v3.repair_window
                    def counted(*args,**kw):ledger.use('nonstop',kind_detail='original_test');return base(*args,**kw)
                    namespace['repair_window']=counted
                    Old=type('CountedOriginal',(v3.Env,),dict(step=types.FunctionType(v3.Env.step.__code__,namespace)))
                    from rc3_v4 import runtime
                    runspace=dict(vars(runtime));runspace['Env']=Old
                    localrun=types.FunctionType(runtime._run.__code__,runspace)
                    plan,diagnostic=localrun(problem,'fixed3' if method=='original_fixed3' else 'detour',None,16,max(0.,deadline-time.perf_counter()))
                    J=plan['objective_length_plus_stops_m'];R=None;work=None;calls=diagnostic['dp_calls'];reason=diagnostic['stop_reason'];fallback=diagnostic['fallback_used']
                else:
                    env=make(case,ledger,p);calls0=ledger.counts['nonstop'];follow(env,None if method=='same_interface_rule' else models[method],deadline)
                    plan=env.returned_certificate.plan;J=env.returned_certificate.cost;R=env.total_reward;work=env.work_used
                    calls=ledger.counts['nonstop']-calls0;reason=env.stop_reason;fallback=env.finalization['fallback_used']
                elapsed=time.perf_counter()-started
                row=dict(case_id=i,seed=case['seed'],mode=case['mode'],rep=rep,method=method,J=J,L=plan['route_length_m'],K=len(plan['service_stops']),
                    complete_s=elapsed,overrun=elapsed>p['complete_soft_s'],return_=R,work=work,calls=calls,reason=reason,fallback=fallback,valid=plan['validation']['valid_model_plan'])
                rows.append(row);write(ROOT/'outputs/test_results.json',rows)
                write(ROOT/f'outputs/plans/{i}_{rep}_{method}.json',plan)
                print(json.dumps(row),flush=True)
    return rows

def main():
    p=read(ROOT/'outputs/protocol.json')
    if sha(ROOT/'outputs/protocol.json')!=(ROOT/'outputs/protocol.sha256').read_text():raise ValueError('Protocol changed')
    if (ROOT/'outputs/resource.jsonl').exists():raise FileExistsError('No implicit resume/retry or budget expansion')
    for name,x in p['source_sha256'].items():
        if sha(ROOT/name)!=x:raise ValueError('Preregistered code changed')
    configure(2300);ledger=Ledger(p['limits'])
    try:
        alignment(p,ledger);models={}
        for i,seed in enumerate(p['seeds']):
            before=ledger.counts['nonstop'];ledger.seed_base=before;cases=p['splits']['train'][i*4:(i+1)*4];configure(seed)
            data0=collect(p,ledger,cases,None,f'seed{seed}_round0')
            initial=Value();one=copy.deepcopy(initial)
            mc=fit(initial,data0,'multi',256,ledger,ROOT/f'outputs/fit_{seed}_mc0.json');frozen=copy.deepcopy(mc).eval()
            one=fit(one,data0,'one',256,ledger,ROOT/f'outputs/fit_{seed}_one0.json')
            save(ROOT/f'outputs/model_{seed}_init.pt',frozen,seed=seed,policy_evaluation='rule',updates=256)
            data1=collect(p,ledger,cases,frozen,f'seed{seed}_round1',True)
            static=fit(copy.deepcopy(frozen),data1,'static',256,ledger,ROOT/f'outputs/fit_{seed}_static.json')
            multi=fit(copy.deepcopy(frozen),data1,'multi',256,ledger,ROOT/f'outputs/fit_{seed}_multi.json')
            one=fit(one,data1,'one',256,ledger,ROOT/f'outputs/fit_{seed}_one1.json')
            for name,model in [('frozen_init',frozen),('matched_static_rule_value',static),('multi_policy_iteration',multi),('one_step_repeated',one)]:
                model.eval();models[f'{seed}_{name}']=model
                save(ROOT/f'outputs/model_{seed}_{name}.pt',model,seed=seed,updates=256 if name=='frozen_init' else 512,threshold=0.)
            if ledger.counts['nonstop']-before>1536:raise ValueError('Per-seed train allocation exceeded')
            print('SEED_DONE',seed,ledger.counts,flush=True)
        test(p,ledger,models)
        write(ROOT/'outputs/run_receipt.json',dict(completed=True,counts=ledger.counts,elapsed_s=time.perf_counter()-ledger.started))
    except Exception as error:
        write(ROOT/'outputs/failure.json',dict(error=repr(error),counts=ledger.counts,elapsed_s=time.perf_counter()-ledger.started))
        raise

if __name__=='__main__':main()
