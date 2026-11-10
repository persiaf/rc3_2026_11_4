"""Post-run raw evidence tables; a development gain is never a final claim."""
import csv
import collections
import io
from .common import *
from .runner import all_rows
from .statistics import paired_effects


def derived(path,data):
    """Draft summaries can be rebuilt; preserve any preceding different bytes."""
    path=Path(path)
    if path.exists():
        if path.read_bytes()==data:return
        old_sha=digest(path);archive=OUT/'derived_revisions'/path.name/(old_sha+'.bin')
        if not archive.exists():archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(path.read_bytes())
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.derived.'+uuid.uuid4().hex)
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def write_new(path,value):derived(path,json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False).encode('utf8')+b'\n')


def csv_new(name,rows,columns=None):
    rows=list(rows);columns=columns or list(dict.fromkeys(k for r in rows for k in r))
    f=io.StringIO(newline='');w=csv.DictWriter(f,fieldnames=columns or ['status']);w.writeheader()
    for row in rows:w.writerow({k:json.dumps(plain(v),ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in row.items()})
    derived(HOME/name,f.getvalue().encode('utf-8-sig'))


def summarize(store,inputs):
    events=store.log.events;rows=all_rows(store);evalrows=[r for r in rows if 'metrics' in r];trajectory=[r for r in rows if 'result' in r]
    choices={};complete={}
    for seed in SEEDS:
        p=OUT/'seeds'/str(seed)/'SELECTION_DECISION.json'
        if p.exists():choices[str(seed)]=read(p)
        p=OUT/'seeds'/str(seed)/'main/COMPLETE.json'
        if p.exists():complete[str(seed)]=read(p)
    quality=[r for r in evalrows if r['request']['kind']=='quality']
    scope='final_confirmation' if quality else ('promotion' if any(r['request']['kind']=='promotion' for r in evalrows) else 'selection_development')
    if not quality:
        kind='promotion' if scope=='promotion' else 'selection';quality=[r for r in evalrows if r['request']['kind']==kind]
        if kind=='selection':
            quality=[r for r in quality if r['study_method'] in ('SL256','Fixed3','Detour16') or (
                r['study_method'] in ('RL','residual_SL') and str(r['seed']) in choices and r['update']==choices[str(r['seed'])]['chosen'][r['study_method']]['round'])]
    flat=[]
    for r in evalrows:
        q=r['request'];flat.append(dict(scope=q['kind'],stage=q['stage'],case=r['case_id'],physical_parent=r['physical_key'],seed=r['seed'],
            role=r['study_method'],checkpoint_role=r['checkpoint_role'],update=r['update'],width=r['spec']['width'],height=r['spec']['height'],
            endpoint=r['spec']['endpoints'],mode=r['spec']['mode'],**r['metrics'],service_s=r['service_time_before_IO_s'],full_audit_s=r['_audit_time_s'],
            valid=r['valid'],actor_sha=r['actor_sha'],tensor_sha=r['tensor_sha'],input_identity=r['input_identity']))
    csv_new('ALL_ROUTE_RESULTS.csv',flat)
    grouped={}
    for r in quality:
        k=(r['physical_key'],r['seed'],r['study_method'],r['update']);grouped.setdefault(k,[]).append(r)
    parentrows=[]
    for (parent,seed,role,update),rr in grouped.items():parentrows.append(dict(scope=scope,group=rr[0]['spec']['role'],parent=parent,seed=seed,role=role,update=update,cases=len(rr),
        **{metric:float(np.mean([r['metrics'][metric] for r in rr])) for metric in ('J','L','K','raw_J','raw_L','raw_K')}))
    csv_new('QUALITY_BY_PARENT.csv',parentrows)
    grouped={}
    for r in quality:
        k=(r['spec']['width'],r['spec']['height'],r['spec']['endpoints'],r['study_method'],r['seed'],r['spec']['role']);grouped.setdefault(k,[]).append(r)
    csv_new('QUALITY_BY_SIZE_ENDPOINT.csv',[dict(scope=scope,group=k[5],width=k[0],height=k[1],endpoint=k[2],role=k[3],seed=k[4],cases=len(rr),
        **{m:float(np.mean([r['metrics'][m] for r in rr])) for m in ('J','L','K','raw_J','raw_L','raw_K')}) for k,rr in grouped.items()])
    training=[e for e in events if e['event']=='round_complete'];csv_new('TRAINING_CURVES.csv',[{k:v for k,v in e.items() if k not in ('previous','sha256','PPO')} for e in training])
    csv_new('POLICY_KL.csv',[{k:v for k,v in e.items() if k not in ('previous','sha256')} for e in events if e['event']=='PPO_epoch'])
    csv_new('VALUE_DIAGNOSTICS.csv',[dict(seed=e['seed'],variant=e['variant'],round=e['round'],epoch=i+1,**d) for e in training for i,d in enumerate(e['PPO']['critic'])])
    timings=[r for r in evalrows if r['request']['kind']=='timing']
    csv_new('TIMING_REQUESTS.csv',[dict(case=r['case_id'],seed=r['seed'],role=r['study_method'],cache=r['request']['cache'],repeat=r['request']['repeat'],
        service_s=r['service_time_before_IO_s'],full_audit_s=r['_audit_time_s'],valid=r['valid']) for r in timings])
    csv_new('TIMING_COMPONENTS.csv',[dict(case=r['case_id'],seed=r['seed'],role=r['study_method'],cache=r['request']['cache'],repeat=r['request']['repeat'],
        **r['components_s']) for r in timings])
    if timings:
        from .deployment import tables
        deploy,totals,contrasts=tables(quality,timings,read(HOME/'READY_SERVICE.json'))
        csv_new('DEPLOYMENT_BY_GROUP_SIZE.csv',deploy);csv_new('READY_AND_REQUEST_TOTALS.csv',totals)
        write_new(HOME/'FINAL_GROUP_CONTRASTS.json',contrasts)
    else:
        csv_new('DEPLOYMENT_BY_GROUP_SIZE.csv',[dict(status='NOT_EVALUATED_CONDITIONAL_STAGE_NOT_STARTED')])
        csv_new('READY_AND_REQUEST_TOTALS.csv',[dict(status='NOT_EVALUATED_CONDITIONAL_STAGE_NOT_STARTED')])
    actors=[]
    for p in sorted((OUT/'actors').rglob('*.pt.json')):
        e=read(p);require(digest(ROOT/e['path'])==e['actor_sha'],'Exported actor bytes changed');actors.append(e)
    write_new(HOME/'ACTOR_INDEX.json',dict(experiment=EXP,actors=actors,selected=choices,planning_default='Selected RL is runnable even if research gate fails; no performance endorsement'))
    write_new(HOME/'WEIGHT_LINEAGE.json',dict(base_models=inputs.ref['models'],teachers=inputs.ref['teacher_pool_index'],
        per_seed=complete,actors=actors,old_models_modified=False,engineering_weights_reused=False))
    wall={}
    for e in events:
        if e['event']=='clock_close':wall[e['name']]=wall.get(e['name'],0.)+e['elapsed_s']
    calls=collections.Counter((e.get('stage','matched_control'),e['role']) for e in events if e['event']=='optimizer_attempt')
    H=sum(r.get('counts',r.get('counters',{})).get('full_H_calls',0) for r in rows)
    internal=collections.Counter()
    for r in rows:
        for k,v in r.get('internal_search',{}).items():
            if isinstance(v,(int,float)):internal[k]+=v
    resource=dict(experiment=EXP,status='COMPLETE_CONDITIONAL_TASK',requests=store.counts,paid_requests=sum(store.counts.values()),
        infrastructure_retries=store.retry,failures=len(store.failed),successes=len(store.done),request_caps=CAPS,absolute_cap=50000,
        optimizer_attempts={str(k):v for k,v in calls.items()},scientific_actor_accepted_steps=sum(d['actor_accepted_steps'] for d in complete.values()),
        full_H_calls=H,native_requests=sum(r.get('study_method') in ('Fixed3','Detour16') for r in evalrows),
        policy_decisions=sum(r.get('policy_steps',0) for r in trajectory),internal_search=dict(internal),input_preparation=inputs.counts,
        cold_navigation_solves=sum(r.get('navigation_solves',0) for r in evalrows),terminal_result_cache_hits=0,
        immutable_completed_request_replay_hits='No extra terminal compute; request IDs and immutable ledger identify any resume replay',
        wall_s=wall,old_teacher_queries=0,original60_semantic_access=0,old_S3_missing_retried=0,source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
        timing_boundary='service includes shared input, D1 load, fresh task encoding, construction,H,export,validation; audit adds object/receipt/committed event IO. Timing-record IO stays in stage wall.',
        maintenance='Git, byte checksum audit and independent editorial review separate from scientific compute; no optimizer/plan claims')
    write_new(HOME/'RESOURCE_ACCOUNTING.json',resource)
    write_new(HOME/'FAILURE_AND_RECOVERY.json',dict(failed_requests=list(store.failed.values()),unclosed=[k for k in store.starts if k not in store.done and k not in store.failed],
        clock_recoveries=[e for e in events if e['event']=='clock_close' and e.get('interrupted_recovery')],
        policy='No refunds; first success; one registered transient-IO retry per slot/max32; dirty optimizer transaction rejects resume; old2missing slots preserved',
        old_failed_slots_reexecuted=0,source_binding_sha256=digest(HOME/'SOURCE_BINDING.json')))
    seedlist=tuple(int(s) for s in choices)
    effects={c:paired_effects(quality,c,seeds=seedlist) for c in ('SL256','residual_SL','Fixed3','Detour16')} if quality else {}
    write_new(HOME/'RESULT_SUMMARY.json',dict(scope=scope,choices=choices,effects=effects,request_counts=store.counts,
        final_confirmation_executed=bool(timings),final_quality_rows=len(quality) if scope=='final_confirmation' else 0,
        deployment='NOT_EVALUATED' if not timings else 'See deployment table; strict quality and time AND criteria required'))
    p2=all(v['P2_pass'] for v in choices.values()) if choices else False
    result=['# 实际实验报告','',f'实验 `{EXP}`。质量证据范围：**{scope}**。',
        f"实际付费请求 {sum(store.counts.values())}，成功 {len(store.done)}，失败 {len(store.failed)}；正式完成seed：{list(complete)}。",'',
        '| 比较：RL相对对照 | 父布局等权平均收益 | 95%区间 |','| --- | ---: | --- |']
    for c,e in effects.items():result.append(f"| {c} | {100*e['mean']:.4f}% | [{100*e['CI95'][0]:.4f}%, {100*e['CI95'][1]:.4f}%] |")
    result.extend(['','以上保留相同H后的未归一成本；raw/H后J、L、K在CSV逐例保存。选择集结果属于开发筛选；窗口／seed／重复计时没有被当作独立地图。',
        '监督对照只训练同规模残差头并匹配已接受Actor更新；Critic、回滚与实际计算另计。原60、旧失败槽和全部旧权重保持。',
        '训练回报、loss、梯度、H收益和工程速度均不能替代DRL独立增量。真实IK与机械臂关节轨迹不在代理验证范围。'])
    derived(HOME/'EXPERIMENT_REPORT.md',('\n'.join(result)+'\n').encode('utf8'))
    verdict='按预定P2条件停止；目标尚未达到' if not p2 else ('独立晋级门槛未通过' if not timings else '已完成确认，须逐格审查质量和时间AND门槛')
    derived(HOME/'NEXT_STEP_DECISION.md',('# 下一步判断\n\n'+verdict+'。\n\n不因本次loss或保住SL而宣称规划增益；未启动阶段标记为NOT_STARTED。固定DRL研究目标保持，原训练不追加。\n').encode('utf8'))
    derived(HOME/'ABLATION_REPORT.md',('# 条件消融\n\n'+('已按两项注册配置执行，见训练及逐例CSV。' if any(e['variant']!='main' for e in training) else 'NOT_STARTED：预定可复现增量条件未满足，未消费消融请求。')+'\n').encode('utf8'))
    derived(HOME/'RESULT_TO_CLAIM.md',('# 结果到主张\n\n'+verdict+'。\n\n当前自动汇总为待独立审查版本。实际证据见RESULT_SUMMARY、逐例CSV与不可变请求对象。禁止从开发筛选推论未见泛化或部署成功。\n').encode('utf8'))


if __name__=='__main__':raise SystemExit('Called only by the metered authorized runner')
