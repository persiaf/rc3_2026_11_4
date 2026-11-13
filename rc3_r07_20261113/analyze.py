"""Saved paired-route arithmetic only; no new Actor/H/planning calls."""
import csv,time,json
from collections import Counter,defaultdict
import numpy as np
from .common import ROOT,HOME,read,put,unarchive,sha

def table(name,rows):
    if not rows:return
    with (HOME/name).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def stats(x):
    return dict(mean=float(np.mean(x)),p50=float(np.median(x)),p95=float(np.quantile(x,.95)),min=float(np.min(x)),max=float(np.max(x))) if x else None
def summary(rr):
    if not rr:return dict(states=0)
    return dict(states=len(rr),physical_parents=len({x['physical_key'] for x in rr}),
        improved=sum(x['delta_J']<-1e-7 for x in rr)/len(rr),tied=sum(abs(x['delta_J'])<=1e-7 for x in rr)/len(rr),degraded=sum(x['delta_J']>1e-7 for x in rr)/len(rr),
        material_improved=sum(x['relative_gain']>=.005 for x in rr)/len(rr),material_different=sum(abs(x['relative_gain'])>=.005 for x in rr)/len(rr),
        delta_J=stats([x['delta_J'] for x in rr]),raw_delta_J=stats([x['raw_delta_J'] for x in rr]),delta_L=stats([x['delta_L'] for x in rr]),delta_K=stats([x['delta_K'] for x in rr]),
        H_identical_route_fraction=sum(x['H_identical_route'] for x in rr)/len(rr),H_same_cost_fraction=sum(abs(x['delta_J'])<=1e-7 for x in rr)/len(rr),
        paired_rollout_s=stats([x['pair_s'] for x in rr]),alternative_rollout_s=stats([x['alternative_s'] for x in rr]),
        baseline_H_s=stats([x['baseline_H_s'] for x in rr]),alternative_H_s=stats([x['alternative_H_s'] for x in rr]))

def main():
    start=time.time();rows=[];audit=0
    selection=read(HOME/'SELECTED_STATES.json');tasks=read(HOME/'TASKS.json')['records'];r=read(HOME/'RESOURCE_ACCOUNTING.json')
    if read(HOME/'controls/SUPERVISOR_RECEIPT.json')['status']!='COMPLETE':raise RuntimeError('Actual rollout execution incomplete')
    for f in sorted((HOME/'pairs').glob('*.json')):
        pair=read(f);s=pair['state'];a=unarchive(pair['baseline']['record']);b=unarchive(pair['alternative']['record'])
        assert a['state_fingerprint']==b['state_fingerprint']==s['fingerprint']
        assert a['prefix_local']==b['prefix_local']==s['prefix_local'] and a['actor_tensor_sha256']==b['actor_tensor_sha256'] and a['H_identity']==b['H_identity']
        assert a['forced_action']==s['top1'] and b['forced_action']==s['top2'] and s['top1']!=s['top2']
        for out in (a,b):
            assert out['raw_route']==[0,*s['prefix_local'],*out['suffix_actions_local'],0]
            for label,key,prefix in [('raw','raw_plan','raw_'),('H','plan','')]:
                plan=out[key];L=sum(plan['route_leg_costs_m']);K=len(plan['visit_order_candidate_ids'])
                assert plan['validation']['valid_model_plan'] and not plan['unserved_task_ids']
                assert abs(L-out['metrics'][prefix+'L'])<1e-7 and K==out['metrics'][prefix+'K'] and abs(L+.75*K-out['metrics'][prefix+'J'])<1e-7
                audit+=4
        row=dict(state_id=s['state_id'],task_id=s['task_id'],physical_key=s['physical_key'],size=s['size'],stage=s['stage'],progress=s['progress'],step=s['step'],top1=s['top1'],top2=s['top2'],
            baseline_J=a['metrics']['J'],alternative_J=b['metrics']['J'],relative_gain=(a['metrics']['J']-b['metrics']['J'])/a['metrics']['J'],
            **{'delta_'+m:b['metrics'][m]-a['metrics'][m] for m in ('J','L','K')},
            **{'raw_delta_'+m:b['metrics']['raw_'+m]-a['metrics']['raw_'+m] for m in ('J','L','K')},
            H_identical_route=a['route']==b['route'],H_identical_set=set(a['route'][1:-1])==set(b['route'][1:-1]),
            baseline_forced_survives=s['top1'] in a['route'][1:-1],alternative_forced_survives=s['top2'] in b['route'][1:-1],
            H_forced_action_distinction=(s['top1'] in a['route'][1:-1] and s['top1'] not in b['route'][1:-1]) or (s['top2'] in b['route'][1:-1] and s['top2'] not in a['route'][1:-1]),
            baseline_s=pair['baseline']['full_request_s'],alternative_s=pair['alternative']['full_request_s'],pair_s=pair['baseline']['full_request_s']+pair['alternative']['full_request_s'],
            baseline_H_s=a['H_s'],alternative_H_s=b['H_s'])
        rows.append(row)
    assert len(rows)==len(selection['states']) and read(HOME/'REPRODUCIBILITY.json')['status']=='PASS'
    groups={'overall':rows,'full':[x for x in rows if x['size']=='10x10'],'local':[x for x in rows if x['size']!='10x10']}
    for size in ('10x10','4x3','6x6','8x8'):
        groups[size]=[x for x in rows if x['size']==size]
        for stage in ('early','middle','late'):groups[size+':'+stage]=[x for x in rows if x['size']==size and x['stage']==stage]
    for stage in ('early','middle','late'):groups[stage]=[x for x in rows if x['stage']==stage]
    sums={g:summary(rr) for g,rr in groups.items()};parents=[]
    for task in tasks:
        rr=[x for x in rows if x['physical_key']==task['physical_key']]
        parents.append(dict(task_id=task['id'],physical_key=task['physical_key'],size=task['size'],states=len(rr),missing=3-len(rr),
            mean_signed_gain=float(np.mean([x['relative_gain'] for x in rr])) if rr else None,
            mean_abs_relative_delta=float(np.mean([abs(x['relative_gain']) for x in rr])) if rr else None,
            mean_oracle_per_state_gain=float(np.mean([max(0,x['relative_gain']) for x in rr])) if rr else None,
            any_material_improvement=any(x['relative_gain']>=.005 for x in rr)))
    good=[x for x in parents if x['states']];rng=np.random.default_rng(2026111307);draws={k:np.zeros(20000) for k in ('mean_signed_gain','mean_oracle_per_state_gain','mean_abs_relative_delta')}
    for size in ('10x10','4x3','6x6','8x8'):
        pp=[x for x in good if x['size']==size]
        if not pp:continue
        idx=rng.integers(0,len(pp),size=(20000,len(pp)))
        for key in draws:draws[key]+=np.asarray([x[key] for x in pp])[idx].sum(1)/len(good)
    intervals={key:dict(mean=float(np.mean([x[key] for x in good])),lower95=float(np.quantile(v,.025)),upper95=float(np.quantile(v,.975))) for key,v in draws.items()}
    full_parents=sum(x['size']=='10x10' and x['any_material_improvement'] for x in parents);local_parents=sum(x['size']!='10x10' and x['any_material_improvement'] for x in parents)
    gate=(sums['overall']['material_improved']>=.2 and sums['full']['material_improved']>=.15 and sums['local']['material_improved']>=.15 and full_parents>=8 and local_parents>=4 and intervals['mean_oracle_per_state_gain']['mean']>=.005)
    result=dict(status='OPPORTUNITY_SUFFICIENT' if gate else 'OPPORTUNITY_INSUFFICIENT',passed=bool(gate),summaries=sums,parent_intervals=intervals,
        full_parents_with_material_improvement=full_parents,local_parents_with_material_improvement=local_parents,
        actual_states=len(rows),missing_states=selection['missing'],hypothetical_oracle_not_trained_policy=True,
        runtime_scope='Resident fixed Actor, cached immutable input after selection; includes per-branch prefix restoration,encode,forced action,greedy suffix,H,export+validation and archive. No equal-cache deployment speed claim.',
        task_setup=selection['task_timings'],audit_checks=audit,new_forward_H_optimizer_or_planning_in_analysis=0,analysis_wall_s=time.time()-start)
    table('STATE_PAIRS.csv',rows);table('PARENT_OPPORTUNITIES.csv',parents)
    table('SIZE_PROGRESS_SUMMARY.csv',[dict(group=g,**{k:v for k,v in d.items() if not isinstance(v,dict)}) for g,d in sums.items()])
    put(HOME/'RESULTS.json',result);put(HOME/'INDEPENDENT_NUMERICAL_AUDIT.json',dict(status='PASS',checks=audit,primary_pairs=len(rows),actual_repeats=read(HOME/'REPRODUCIBILITY.json')['actual_repeated_requests'],all_prefix_and_policy_H_identical=True))
    lines=['# R07同状态动作反事实完整路线机会\n',f"结果：**{result['status']}**。固定48个不同训练物理父布局，实际{len(rows)}个状态，缺失{len(selection['missing'])}个，不补例。冻结SL256，两分支只有一次强制动作不同，之后均为同一冻结SL贪心，H版本/算法/最大pass相同。所有raw/H是实际完整执行与代理验证结果。\n",
        '|分组|状态/父布局|改善/持平/退化|≥0.5%改善|H路线完全相同|H-J差均值|配对耗时均值/p95秒|\n|---|---:|---:|---:|---:|---:|---:|']
    for key in ('overall','10x10','4x3','6x6','8x8','early','middle','late'):
        x=sums[key]
        if not x['states']:continue
        lines.append(f"|{key}|{x['states']}/{x['physical_parents']}|{100*x['improved']:.1f}%/{100*x['tied']:.1f}%/{100*x['degraded']:.1f}%|{100*x['material_improved']:.1f}%|{100*x['H_identical_route_fraction']:.1f}%|{x['delta_J']['mean']:.4f}|{x['paired_rollout_s']['mean']:.4f}/{x['paired_rollout_s']['p95']:.4f}|")
    lines+=['\n`delta_J=alternative_H_J−baseline_H_J`，负值表示改善；持平绝对容差1e-7，实质差异0.5%。raw/H的J/L/K、动作是否仍被H保留、每状态完整时间及H时间见STATE_PAIRS.csv，尺寸×进度见RESULTS.json和SIZE_PROGRESS_SUMMARY.csv。\n',
        f"按父布局等权的平均有符号收益为{100*intervals['mean_signed_gain']['mean']:.4f}%，描述性95%区间[{100*intervals['mean_signed_gain']['lower95']:.4f}%,{100*intervals['mean_signed_gain']['upper95']:.4f}%]。按预设候选对作有利选择的平均每状态假设潜力为{100*intervals['mean_oracle_per_state_gain']['mean']:.4f}%，区间[{100*intervals['mean_oracle_per_state_gain']['lower95']:.4f}%,{100*intervals['mean_oracle_per_state_gain']['upper95']:.4f}%]。该上界不能将三个不同前缀的收益相加，不能当作已训练策略收益。父布局是聚合/重抽样单位，不把状态当独立父布局。\n",
        f"至少一个实质改善状态的父布局：完整{full_parents}/32，局部{local_parents}/16。重复请求{read(HOME/'REPRODUCIBILITY.json')['actual_repeated_requests']}次，actions/mask/raw/H路线/成本逐字典相等；三个前缀的baseline还必须实际复现既有FrozenSL完整路线及H。\n",
        '动作价值仅是这个固定后续策略及H下的条件成本，不是最优Q。差异显著在本轮指预设0.5%实质差异，描述区间不构成训练外泛化确认。已有同任务强教师gap见TEACHER_GAPS.json，仅恢复既有标签，不新增教师或H。\n',
        '运行时间是本轮实际测量；两候选在每决策需两次完整续跑/H，若已有同状态baseline可复用，则新增alternative一次。输入恢复和状态选择另记task_setup，模型常驻加载计入总账。该额外规划代价没有被隐藏为单步距离计算。\n',
        '当前只有机会验证，Actor/Critic更新均0，无新teacher/scene/nav，不读取原60/旧S3/确认/最终集，不是实机IK或关节轨迹认证。\n']
    decision='值得另行审议基于动作相对价值的近似策略迭代；本轮不启动训练。' if gate else '机会不足：建议结束当前构造式学习方向，转向整体规划决策层次重设，不再继续修改回报基线。'
    lines.append('研究决定：'+decision);(HOME/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (HOME/'NEXT_STEP_DECISION.md').write_text('# R07研究决定\n\n'+decision+'\n\n'+f"预登记机会门槛：{result['status']}。这里只报告固定动作对的机会和成本，不是训练出的策略收益，不自动启动PG/Q或多种子。旧D5/D6负结果与所有模型保持。\n",encoding='utf-8')
    print(dict(status=result['status'],pairs=len(rows),audit_checks=audit,analysis_wall_s=time.time()-start),flush=True)
if __name__=='__main__':main()
