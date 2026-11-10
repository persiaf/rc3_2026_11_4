"""One authorized conditional P0--P5 execution, protected by a parent watchdog."""
import argparse
import subprocess
import sys
import time
import traceback
from .common import *


def seal_or_verify(path,value):
    if path.exists():require(read(path)==value,'Immutable completion differs')
    else:write_new(path,value)


def all_rows(store,kind=None,stage=None):
    out=[]
    for k,e in store.done.items():
        q=store.starts[k]['query']
        if (kind is None or q['kind']==kind) and (stage is None or q['stage']==stage):
            x=object_read(e['object']);x['_audit_time_s']=store.timings.get(k);out.append(x)
    return out


def p0(inputs,store,clock):
    from .verification import run
    if (HOME/'ENGINEERING_VERIFICATION.json').exists():return read(HOME/'ENGINEERING_VERIFICATION.json')
    require(not any(e['event']=='optimizer_attempt' for e in store.log.events),'Incomplete P0 optimizer transaction cannot be replayed')
    if not (HOME/'INPUT_FEASIBILITY.json').exists():inputs.preflight()
    return run(inputs,store,clock)


def execute():
    from .data import Inputs
    from .training import seed_run,read_complete
    from .eval import evaluate_set
    from .statistics import select_candidates,paired_effects
    prep=float(os.environ.get('RC3_INTEGRATED_LAUNCH_PERF',time.perf_counter()));configure();binding=verify_sources();log=Log();store=Store(log);clock=None
    try:
        if (HOME/'RESOURCE_CLOSEOUT.json').exists():
            closed=read(HOME/'RESOURCE_CLOSEOUT.json');n=closed['ledger_sequence']
            require(read(HOME/'RUN_COMPLETE.json')['source_binding_sha256']==digest(source_path()) and n<=len(log.events)
                and log.events[n-1]['sha256']==closed['ledger_tail'] and digest(HOME/'RESOURCE_ACCOUNTING.json')==closed['resource_accounting_sha256'],
                'Completed research closeout changed')
            print('ALREADY_COMPLETE_VERIFIED_NO_NEW_REQUEST',flush=True);return
        if not any(e['event']=='registration_cost' for e in log.events):
            r=read(HOME/'REGISTRATION_RESOURCE.json');log(dict(event='registration_cost',**r))
            log(dict(event='clock_close',name='P0',open_sequence=-1,elapsed_s=r['wall_s'],registration_preparation=True))
        if (HOME/'BOOTSTRAP_FAILURE_RESOURCE.json').exists() and not any(e['event']=='bootstrap_failure_cost' for e in log.events):
            r=read(HOME/'BOOTSTRAP_FAILURE_RESOURCE.json');log(dict(event='bootstrap_failure_cost',**r))
            log(dict(event='clock_close',name='P0',open_sequence=-1,elapsed_s=r['conservative_wall_s'],exact_wall_s=None,failed_import=True))
        log(dict(event='clock_close',name='P0',open_sequence=-1,elapsed_s=time.perf_counter()-prep,source_verification_and_startup=True))
        clock=Clock(log,'P0',1800);inputs=Inputs(log,clock);engineering=p0(inputs,store,clock);clock.close();clock=None
        require(engineering['status']=='PASS','P0 resource/engineering condition not passed; scientific training not started')
        path=HOME/'IMPLEMENTATION_REVIEW.md'
        if not path.exists():path.write_text('# 实施评审\n\n**GO：P0准确绑定、数学／恢复／角色检查及真实路线与优化器验收通过。**\n\n'
            f"执行源码绑定：`{digest(source_path())}`。P0保守单seed投影 `{engineering['benchmark']['conservative_seed_projection_s']:.1f}` 秒，硬上限14400秒。\n\n"
            '本GO允许按已授权的条件任务继续；P0工程模型不进入正式训练，不说明规划性能改善。原始奖励、H、mask、scope_r1、训练银行及封存边界保持。'
            '任务数归一改变实例加权，PPO使用episode平均的逐步损失之和；Critic独立；Actor与Adam在KL失败后精确回滚。\n\n'
            '恢复限于可证明的已关闭优化轮边界；中途已采样请求可从冻结旧Actor重建，未闭合优化器事务拒绝回放，失败与墙钟不退款。\n',encoding='utf8')
        choices={};seed_info={};p2=None
        # Native selection reference is deterministic and computed once, not once per seed.
        clock=Clock(log,'seed_11622026',14400);inputs.clock=clock
        evaluate_set(inputs.declared['selection'],[(None,'Fixed3',None),(None,'Detour16',None)],'selection','native_selection',0,store,inputs,clock)
        for seed in SEEDS:
            if seed!=SEEDS[0]:
                clock=Clock(log,'seed_'+str(seed),14400);inputs.clock=clock
            info=read_complete(seed,'main',log) or seed_run(seed,inputs,store,clock)
            seed_info[str(seed)]=info
            choice=select_candidates(info,all_rows(store,'selection'));choices[str(seed)]=choice
            cp=OUT/'seeds'/str(seed)/'SELECTION_DECISION.json'
            if not cp.exists():write_new(cp,choice)
            if seed==SEEDS[0]:
                p2=choice['P2_pass'] and info['status']=='COMPLETE';log(dict(event='P2_gate',pass_gate=p2,decision=choice))
            clock.close();clock=None
            if seed==SEEDS[0] and not p2:break
        if p2:
            clock=Clock(log,'confirmation_and_analysis',10800);inputs.clock=clock
            roles=[]
            for seed in SEEDS:
                roles.extend([(seed,'SL256',seed_info[str(seed)]['base_entry']),
                    (seed,'residual_SL',choices[str(seed)]['chosen']['residual_SL']['entry']),
                    (seed,'RL',choices[str(seed)]['chosen']['RL']['entry'])])
            roles.extend([(None,'Fixed3',None),(None,'Detour16',None)])
            promotion=evaluate_set(inputs.declared['promotion'],roles,'promotion','promotion',0,store,inputs,clock)
            comparisons={c:paired_effects(promotion,c) for c in ('SL256','residual_SL')}
            go=all(v['mean']>=.005 and v['CI95'][0]>0 and all(s>0 for s in v['seed_means'].values()) for v in comparisons.values())
            decision=dict(status='PASS' if go else 'NOT_PASSED',comparisons=comparisons,all_routes_valid=all(r['valid'] for r in promotion))
            if not (HOME/'PROMOTION_DECISION.json').exists():write_new(HOME/'PROMOTION_DECISION.json',decision)
            log(dict(event='P3_gate',**decision))
            if go:
                # All model choices now frozen; only now materialize final tasks.
                if not (HOME/'READY_SERVICE.json').exists():
                    env=os.environ.copy();env['RC3_READY_LAUNCH_PERF']=str(time.perf_counter())
                    measured=subprocess.check_output([sys.executable,'-B','-X','utf8','-m','rc3_integrated_rppo.ready'],cwd=ROOT,env=env,encoding='utf8')
                    write_new(HOME/'READY_SERVICE.json',json.loads(measured));clock.check()
                cases=inputs.declared['final_seen']+inputs.declared['final_unseen'];finalroles=[]
                for seed in SEEDS:
                    finalroles.extend([(seed,'SL256',seed_info[str(seed)]['base_entry']),
                        (seed,'residual_SL',choices[str(seed)]['chosen']['residual_SL']['entry']),
                        (seed,'RL_raw',choices[str(seed)]['chosen']['RL']['entry']),
                        (seed,'RL',choices[str(seed)]['chosen']['RL']['entry'])])
                finalroles.extend([(None,'Fixed3',None),(None,'Detour16',None)])
                quality=evaluate_set(cases,finalroles,'quality','final_quality',0,store,inputs,clock)
                lookup={(r['case_id'],r['seed'],r['study_method']):r for r in quality}
                for r in quality:
                    if r['study_method']=='RL':require(r['raw_route']==lookup[(r['case_id'],r['seed'],'RL_raw')]['raw_route'],'RLraw and RL+H raw routes differ')
                ids=set(inputs.declared['timing_case_ids']);timing=[c for c in cases if c['id'] in ids]
                for repeat in range(3):
                    for cache in ('cold','hot'):
                        evaluate_set(timing,finalroles,'timing','final_timing',0,store,inputs,clock,
                            order_seed=seed_for('timing',repeat,cache),cache=cache,repeat=repeat)
                seal_or_verify(HOME/'FINAL_CONFIRMATION_COMPLETE.json',dict(quality=3024,timing=3024,choices=choices,
                    effects={c:paired_effects(quality,c) for c in ('SL256','residual_SL','Fixed3','Detour16')},source_binding_sha256=digest(source_path())))
            clock.close();clock=None
            if go:
                clock=Clock(log,'seed_11622026',14400);inputs.clock=clock
                for variant,anchor,scale in [('no_anchor_KL',False,True),('no_task_scale',True,False)]:
                    info=read_complete(SEEDS[0],variant,log) or seed_run(SEEDS[0],inputs,store,clock,variant=variant,rounds=64,anchor=anchor,scale=scale)
                    evaluate_set(inputs.declared['selection'],[(SEEDS[0],variant,info['final_actor'])],'ablation',variant+'_selection',64,store,inputs,clock)
                clock.close();clock=None
        log(dict(event='conditional_task_complete',P2_pass=bool(p2),seed_choices=choices))
        seal_or_verify(HOME/'RUN_COMPLETE.json',dict(experiment=EXP,P2_pass=bool(p2),seeds_completed=list(seed_info),
            requests=store.counts,retries=store.retry,source_binding_sha256=digest(source_path())))
        from .report import summarize
        clock=Clock(log,'confirmation_and_analysis',10800);summarize(store,inputs);clock.close();clock=None
        # Seal exact closed analysis wall after derived-report IO, without
        # mutating any raw object, protocol, prior resource evidence or log.
        wall={}
        for e in log.events:
            if e['event']=='clock_close':wall[e['name']]=wall.get(e['name'],0.)+e['elapsed_s']
        seal_or_verify(HOME/'RESOURCE_CLOSEOUT.json',dict(resource_accounting_sha256=digest(HOME/'RESOURCE_ACCOUNTING.json'),
            exact_closed_stage_wall_s=wall,ledger_sequence=len(log.events),ledger_tail=log.tail,
            science_request_predebits=sum(store.counts.values()),optimizer_attempts=sum(e['event']=='optimizer_attempt' for e in log.events)))
    except BaseException as exc:
        payload=dict(error=repr(exc),traceback=traceback.format_exc(),requests=store.counts,source_binding_sha256=digest(source_path()),utc=now())
        independent_receipt([OUT/'failures',OUT/'emergency'],'WORKER_FAILURE',payload)
        print(payload['traceback'],file=sys.stderr,flush=True)
        try:
            if log.healthy:log(dict(event='worker_failure',**payload))
        finally:
            if clock is not None and log.healthy:clock.close()
        raise
    finally:log.close()


def supervise():
    require((source_path()).exists(),'Register exact protocol/source first');OUT.mkdir(parents=True,exist_ok=True)
    with Lease(OUT/'SUPERVISOR.lease',EXP):
        tag=now().replace(':','').replace(' ','_')+'_'+uuid.uuid4().hex[:6]
        folder=OUT/'process_logs';folder.mkdir(parents=True,exist_ok=True)
        with (folder/(tag+'.stdout.txt')).open('xb') as out,(folder/(tag+'.stderr.txt')).open('xb') as err:
            environment=os.environ.copy();environment['RC3_INTEGRATED_LAUNCH_PERF']=str(time.perf_counter())
            child=subprocess.Popen([sys.executable,'-B','-X','utf8','-m','rc3_integrated_rppo.runner','--worker'],cwd=ROOT,stdout=out,stderr=err,env=environment,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            while child.poll() is None:
                active=OUT/'ledger/ACTIVE_CLOCK.pt'
                if active.exists():
                    try:
                        state=torch.load(active,map_location='cpu',weights_only=True)
                        # A closed old stage is ignored; worker publishes a new stage before work.
                        if not state.get('closed') and state['carry_s']+time.perf_counter()-state['start_perf']>=state['cap_s']:
                            child.terminate();break
                    except (PermissionError,ValueError,RuntimeError):pass
                time.sleep(1.)
            code=child.wait();active=OUT/'ledger/ACTIVE_CLOCK.pt'
            if code and active.exists():
                state=torch.load(active,map_location='cpu',weights_only=True)
                elapsed=0. if state.get('closed') else time.perf_counter()-state['start_perf']
                write_new(OUT/'ledger/orphan_clocks'/(tag+'.json'),dict(experiment=EXP,open_sequence=state['open_sequence'],elapsed_s=elapsed,
                    child_returncode=code,utc=now(),boundary='Parent observes worker termination; no reset/refund'))
            print(json.dumps(dict(returncode=code,stdout=label(folder/(tag+'.stdout.txt')),stderr=label(folder/(tag+'.stderr.txt'))),ensure_ascii=False),flush=True)
            return code


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',action='store_true');a=parser.parse_args()
    if a.worker:execute()
    elif a.execute:sys.exit(supervise())
    else:parser.error('Use --execute for the already authorized conditional task')


if __name__=='__main__':main()
