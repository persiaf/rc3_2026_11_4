"""Bounded pre-optimizer checks on saved S0 inputs only; no S1/S2 training."""
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits
from rc3_global_local.policy import GlobalLocalPolicy,tensor_hash
from rc3_global_local.environment import Environment
from rc3_global_local.terminal import TerminalCache
from rc3_nav_value.r1.oracle import _instrument,_empty_internal
from .runtime_guards import load_saved_s0_parent,extract_snapshot
from .endpoint_baselines import run_endpoint,closed_native_reference

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
def default(x):
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,np.ndarray):return x.tolist()
    raise TypeError(type(x).__name__)
def append(p,d):
    with p.open('a',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(d,ensure_ascii=False,allow_nan=False,default=default)+'\n');f.flush();os.fsync(f.fileno())


def register():
    target=OUT/'ENGINEERING_CHECK_PROTOCOL.json'
    if target.exists():raise FileExistsError('Engineering protocol already frozen')
    sources=['rc3_global_local_impl_review/check_engineering.py','rc3_global_local_impl_review/runtime_guards.py',
        'rc3_global_local_impl_review/endpoint_baselines.py','rc3_global_local/policy.py','rc3_global_local/environment.py',
        'rc3_global_local/problem.py','rc3_global_local/terminal.py','rc3_global_local/validation.py',
        'rc3_v4/runtime.py','rc3_v3/environment.py','rc3_improve/repair.py','rc3_fast/solver.py',
        'rc3_baseline/baseline.py','rc3_baseline/validator.py','rc3_nav_value/r1/oracle.py']
    for receipt in ('GUARD_TEST_RECEIPT.json','BASELINE_UNIT_RECEIPT.json'):
        r=read(OUT/receipt);assert r['failures']==r['errors']==0
        for n,h in r['source_sha256'].items():assert sha(ROOT/n if (ROOT/n).exists() else OUT/n)==h,n
    write(target,dict(schema='implementation-preflight-engineering-v1',base='057830d584b6c71c79ef5118087c681c98135fe2',
        source_sha256={n:sha(ROOT/n) for n in sources},
        initial_policy_sha256=sha(ROOT/'rc3_global_local/initial_policy.pt'),
        source_scope='Original saved S0 three parents and18 windows only; no future/S3 target instance creation',
        limits=dict(endpoint_baseline_attempts=36,closed_full_native_reference_attempts=18,
            full_parent_policy_trajectories=9,terminal_reward_attempts=9,backward_attempts=3,optimizer_updates=0,
            new_scene_generation=0,navigation_preparations=0,S3_task_queries=0,original60_reads=0,wall_s=1200),
        baseline='18 S0 windows x Fixed3/Detour16;9 closed windows additionally original native full-candidate reference, compare J/order',
        calibration='3 saved full10x10 S0 parents; each one full policy-sampled row including first logprob plus2 forced low/high first rows excluding forced logprob. Independent per-role torch.Generator SHA-derived streams. LOO detached returns, weights1/.25/.25 divided1.5, one backward/parent, no optimizer or saved updated weights.',
        scope='Engineering conformity/calibration, not trained DRL performance. Native full-reference call profiler overhead included; throughput from untrained saved fixtures is only an estimate, not a certified training budget.',
        failure='Keep charged attempts and partial data, stop on inconsistency; no seed/case replacements or automatic rerun'))


def run():
    p=read(OUT/'ENGINEERING_CHECK_PROTOCOL.json')
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,n
    assert sha(ROOT/'rc3_global_local/initial_policy.pt')==p['initial_policy_sha256']
    if (OUT/'engineering_started.json').exists():raise FileExistsError('One engineering attempt only')
    write(OUT/'engineering_started.json',dict(protocol_sha256=sha(OUT/'ENGINEERING_CHECK_PROTOCOL.json'),formal_optimizer_updates=0))
    start=time.perf_counter();deadline=start+1200
    counters=dict(parent_artifact_loads=0,endpoint_baseline_attempts=0,endpoint_baseline_completed=0,closed_reference_attempts=0,closed_reference_completed=0,
        model_instantiations=0,checkpoint_loads=0,policy_encode_calls=0,policy_score_calls=0,policy_trajectories=0,
        backward_attempts=0,backward_completed=0,optimizer_updates=0,new_scene_generation=0,navigation_preparations=0,S1_training_calls=0,S2_training_calls=0,S3_task_queries=0,original60_reads=0)
    native={};term={k:0 for k in ('terminal_requests','terminal_cache_hits','terminal_attempts','terminal_completed','terminal_failed','full_H_calls','export_attempts','exports','validation_attempts','validation_completed')}
    internal=_empty_internal();parents={};baseline_rows=[];calibration=[];comparisons=[];model=None;model_before=None
    process=psutil.Process();peak_rss=process.memory_info().rss;calibration_start_rss=None;calibration_peak_rss=None
    status='RUNNING';error=None
    def check():
        if time.perf_counter()>=deadline:raise TimeoutError('Engineering deadline')
    def sample_rss():
        nonlocal peak_rss,calibration_peak_rss
        current=process.memory_info().rss;peak_rss=max(peak_rss,current)
        if calibration_start_rss is not None:calibration_peak_rss=max(calibration_peak_rss or 0,current)
    def result_snapshot():
        return dict(schema='implementation-engineering-result-v1',status=status,error=error,counters=counters,native_counters=native,terminal_counters=term,
            internal_policy_terminal=internal,baseline_rows=baseline_rows,closed_native_comparisons=comparisons,calibration=calibration,
            model_hash_before=model_before,model_hash_after=tensor_hash(model) if model is not None else None,
            elapsed_wall_s=time.perf_counter()-start,peak_process_RSS_bytes=peak_rss,
            calibration_start_RSS_bytes=calibration_start_rss,calibration_peak_RSS_bytes=calibration_peak_rss,
            hardware=dict(cpu_count=os.cpu_count(),total_RAM_bytes=psutil.virtual_memory().total,available_RAM_bytes=psutil.virtual_memory().available,device='cpu',torch_threads=2,threadpool_limit=1),
            scope='Saved S0 engineering checks only; baseline observers/serialization and RSS sampling are instrumentation, not deployment timing; no formal optimization')
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        with threadpool_limits(limits=1):
            for case in range(3):check();parents[case]=load_saved_s0_parent(case);counters['parent_artifact_loads']+=1
            (OUT/'baseline_plans').mkdir(exist_ok=True);(OUT/'calibration_plans').mkdir(exist_ok=True)
            saved_views=sorted((ROOT/'rc3_global_local/views').glob('*.json'))
            for path in saved_views:
                check();v=read(path);w=v['window'];parent=parents[v['parent_case']]
                view=extract_snapshot(parent,w['x0'],w['y0'],w['width'],w['height'],v['entry_global_node'],v['exit_global_node'])
                assert view.task_ids.tolist()==v['task_ids_global'] and view.candidate_ids.tolist()==v['candidate_ids_global']
                write(OUT/f"bound_view_{v['case']:02}.json",dict(s0_view_identity=v['identity'],bound_view_identity=view.identity,**view.description(),source_view=str(path.relative_to(ROOT))))
                for method in ('fixed3','detour16'):
                    check();counters['endpoint_baseline_attempts']+=1;ordinal=counters['endpoint_baseline_attempts']
                    assert ordinal<=36
                    append(OUT/'engineering_journal.jsonl',dict(event='endpoint_start',case=v['case'],method=method,ordinal=ordinal))
                    plan,info=run_endpoint(view,method,counters=native)
                    counters['endpoint_baseline_completed']+=1
                    row=dict(case=v['case'],parent=v['parent_case'],method=method,kind='endpoint',J=plan['objective_length_plus_stops_m'],K=len(plan['visit_order_candidate_ids']),
                        global_order=plan['visit_order_candidate_ids'],valid=plan['validation']['valid_model_plan'],DP_calls=info['dp_calls'],elapsed_s=info['adapter_elapsed_s'])
                    baseline_rows.append(row);write(OUT/'baseline_plans'/f"{v['case']:02}_{method}_endpoint.json",dict(mapping=dict(target_task_ids_global=view.task_ids.tolist(),candidate_ids_global=view.candidate_ids.tolist()),plan=plan,info=info))
                    append(OUT/'engineering_journal.jsonl',dict(event='endpoint_complete',**row))
                    if view.entry==view.exit==0:
                        check();counters['closed_reference_attempts']+=1;assert counters['closed_reference_attempts']<=18
                        append(OUT/'engineering_journal.jsonl',dict(event='closed_reference_start',case=v['case'],method=method))
                        reference,refinfo=closed_native_reference(view,method,counters=native)
                        counters['closed_reference_completed']+=1
                        equal=dict(case=v['case'],method=method,objective_equal=abs(reference['objective_length_plus_stops_m']-row['J'])<=1e-7,
                            global_order_equal=reference['visit_order_candidate_ids']==row['global_order'])
                        comparisons.append(equal);assert equal['objective_equal'] and equal['global_order_equal'],equal
                        write(OUT/'baseline_plans'/f"{v['case']:02}_{method}_closed_reference.json",dict(mapping=dict(target_task_ids_global=view.task_ids.tolist()),plan=reference,info=refinfo))
                        append(OUT/'engineering_journal.jsonl',dict(event='closed_reference_complete',**equal))
                    sample_rss()
            check();counters['model_instantiations']+=1;model=GlobalLocalPolicy()
            checkpoint=torch.load(ROOT/'rc3_global_local/initial_policy.pt',map_location='cpu',weights_only=False);counters['checkpoint_loads']+=1
            model.load_state_dict(checkpoint['state']);model_before=tensor_hash(model);assert model_before==checkpoint['tensor_sha256']
            calibration_start_rss=process.memory_info().rss;calibration_peak_rss=calibration_start_rss
            terminal=TerminalCache(term,lambda r:append(OUT/'calibration_terminal_journal.jsonl',r),
                read(ROOT/'rc3_global_local/S0_protocol.json')['H_sha256'],9,deadline)
            for case,parent in parents.items():
                check();instance_start=time.perf_counter();view=extract_snapshot(parent,0,0,10,10,0,0)
                roles=('primary','forced_low','forced_high');rows=[];log_sums=[]
                with _instrument(internal):
                    for role in roles:
                        check();env=Environment(view,terminal);encoded=model.encode(view);counters['policy_encode_calls']+=1
                        legal=np.flatnonzero(env.mask())+1;forced=None if role=='primary' else int(legal[0] if role=='forced_low' else legal[-1])
                        seed=int.from_bytes(hashlib.sha256(f'preflight-v1/{case}/{role}/11632026'.encode()).digest()[:8],'little')%(2**63)
                        generator=torch.Generator(device='cpu').manual_seed(seed)
                        logsum=torch.zeros((),dtype=torch.float32);logps=[]
                        while not env.done:
                            check()
                            if forced is not None and not env.prefix:node=forced;lp=None
                            else:
                                scores=model.score(env,encoded);counters['policy_score_calls']+=1
                                chosen=torch.multinomial(torch.softmax(scores,0),1,generator=generator).squeeze(0)
                                node=int(chosen)+1;lp=torch.log_softmax(scores,0)[chosen];logsum=logsum+lp
                            env.step(node);logps.append(None if lp is None else float(lp.detach()));sample_rss()
                        counters['policy_trajectories']+=1;assert counters['policy_trajectories']<=9
                        row=dict(parent=case,role=role,sampling_seed=seed,forced_first=forced,task_count=len(view.task_ids),candidate_count=len(view.candidate_ids),
                            return_=sum(env.rewards),J=env.result['J'],raw_J=env.result['raw_J'],raw_route=[0,*env.prefix,0],final_route=env.result['route'],
                            rewards=env.rewards,trace=env.trace,logprobs=logps,logprob_sum=float(logsum.detach()))
                        assert abs(row['return_']+row['J'])<=1e-7
                        if role=='primary':assert row['logprobs'][0] is not None
                        else:assert row['logprobs'][0] is None
                        rows.append(row);log_sums.append(logsum)
                        write(OUT/'calibration_plans'/f'{case}_{role}.json',dict(problem=view.description(),row=row,plan=env.result['plan']))
                returns=torch.tensor([r['return_'] for r in rows],dtype=torch.float64)
                baseline=(returns.sum()-returns)/2;advantage=(returns-baseline).detach();weights=(1.,.25,.25)
                loss=sum(-w*a*lp for w,a,lp in zip(weights,advantage,log_sums))/1.5
                model.zero_grad(set_to_none=True);counters['backward_attempts']+=1;assert counters['backward_attempts']<=3
                loss.backward();counters['backward_completed']+=1
                norms={n:float(t.grad.norm()) if t.grad is not None else 0. for n,t in model.named_parameters()}
                assert all(np.isfinite(list(norms.values()))) and tensor_hash(model)==model_before
                sample_rss();instance=dict(parent=case,rows=rows,LOO_baseline=baseline.tolist(),advantages=advantage.tolist(),loss=float(loss.detach()),
                    gradient_norms=norms,elapsed_s=time.perf_counter()-instance_start,model_hash_unchanged=True)
                calibration.append(instance);append(OUT/'calibration_summary.jsonl',instance)
                del rows,log_sums,loss,encoded;gc.collect()
                write(OUT/'ENGINEERING_RESULT.json',result_snapshot())
            assert counters['endpoint_baseline_completed']==36 and counters['closed_reference_completed']==18
            assert counters['policy_trajectories']==9 and counters['backward_completed']==3 and counters['optimizer_updates']==0
            check();status='PASS'
    except Exception as exc:
        status='FAIL_STOP';error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());raise
    finally:
        write(OUT/'ENGINEERING_RESULT.json',result_snapshot())
    print(json.dumps(dict(status=status,counters=counters,native=native,terminal=term,elapsed_wall_s=time.perf_counter()-start,peak_RSS_bytes=peak_rss)))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true');args=parser.parse_args()
    register() if args.register else run()
