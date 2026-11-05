"""Saved R0 records -> statistics, resource account and report; no planning imports."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent/'r0'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def lines(p):return [json.loads(s) for s in Path(p).read_text(encoding='utf-8').splitlines()]


def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def bootstrap(a):
    a=np.asarray(a,float);rng=np.random.default_rng(114)
    means=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),values=a.tolist())


def main():
    p=read(OUT/'protocol.json');s=read(OUT/'execution_summary.json');v=read(OUT/'verification.json')
    assert s['complete'] and v['all_passed']
    rows=read(OUT/'case_results.json');cps=lines(OUT/'checkpoints.jsonl');evaluations=lines(OUT/'evaluations.jsonl');ledger=lines(OUT/'calls.jsonl')
    assert len(rows)==32 and [r['case'] for r in rows]==list(range(32))
    assert len(cps)<=256 and len({(r['case'],tuple(r['prefix'])) for r in cps})==len(cps)
    assert all(r['teacher']['J']<=r['root']['J']+1e-7 for r in rows)
    q=np.asarray([r['gain_percent'] for r in rows]);positive=np.asarray([r['gain_J']>1e-7 for r in rows])
    weights=np.where(positive,q,0.);share=float(weights.max()/weights.sum()) if weights.sum()>0 else None
    gate=dict(complete_cases=32,mean_gain_percent=float(q.mean()),mean_gain_pass=bool(q.mean()>=.5),
        positive_cases=int(positive.sum()),positive_count_pass=bool(positive.sum()>=12),max_positive_relative_share=share,
        concentration_pass=bool(share is not None and share<=.25),all_feasible=all(r['valid'] for r in rows) and v['all_passed'])
    gate['passed']=all(gate[k] for k in ('mean_gain_pass','positive_count_pass','concentration_pass','all_feasible'))
    quality={m:{k:float(np.mean([r[m][k] for r in rows])) for k in ('J','L','K')} for m in ('native','root','teacher')}
    paired=dict(teacher_vs_gBH=bootstrap(q),teacher_vs_native_Fixed3=bootstrap([100*(r['native']['J']-r['teacher']['J'])/r['native']['J'] for r in rows]),
        root_gBH_vs_native_Fixed3=bootstrap([100*(r['native']['J']-r['root']['J'])/r['native']['J'] for r in rows]))
    labels=sum(len(cp['candidates']) for cp in cps)
    raw_regret=[]
    for cp in cps:
        raw_choice=next(r for r in cp['candidates'] if r['node']==cp['raw_best_action'])
        raw_regret.append(raw_choice['Y']-cp['best_Y'])
    opportunities=dict(checkpoints=len(cps),query_label_uses=labels,
        checkpoints_with_final_opportunity=sum(cp['queried_opportunity_J']>1e-7 for cp in cps),
        mean_queried_opportunity_J=float(np.mean([cp['queried_opportunity_J'] for cp in cps])),
        immediate_rule_suboptimal_checkpoints=sum(cp['immediate_vs_final_disagrees'] for cp in cps),
        mean_immediate_rule_regret_J=float(np.mean([cp['immediate_regret_J'] for cp in cps])),
        raw_min_suboptimal_after_H_checkpoints=sum(r>1e-7 for r in raw_regret),
        mean_final_regret_of_raw_min_J=float(np.mean(raw_regret)),
        scope='Candidate-set-relative costs only; up to8 candidates, not all-action optimum; checkpoints nested within32 scenes, not independent statistical units')
    by_mode={mode:dict(cases=len(rr),mean_gain_percent=float(np.mean([r['gain_percent'] for r in rr])),positive=sum(r['gain_J']>1e-7 for r in rr))
             for mode in ('regular','mixed','blocked') if (rr:=[r for r in rows if r['mode']==mode])}
    write(OUT/'analysis.json',dict(gate=gate,quality=quality,paired=paired,opportunities=opportunities,by_mode=by_mode,
        scope='R0 mechanism gate only. No network learning, navigation necessity, reinforcement-learning contribution or deployment-speed conclusion.',
        next_action='R0 signal permits designing a separately authorized R1 protocol, not automatic training' if gate['passed'] else 'Stop R0 configuration; do not launchR1 or expandbudget'))
    starts=[r for r in ledger if r['event']=='start'];completes=[r for r in ledger if r['event']=='complete']
    assert len(starts)==len(completes)==s['total_terminal_calls']<=2304
    assert all(s['terminal_completed'][k]<=n for k,n in p['top_level_caps'].items())
    kind_counts={k:sum(r['kind']==k for r in evaluations) for k in ('root','query','teacher','other')}
    assert len(evaluations)==sum(s['terminal_completed'][k] for k in kind_counts)
    assert kind_counts=={k:s['terminal_completed'][k] for k in kind_counts}
    internal_H={k:sum(r['internal_calls'].get(k,0) for r in evaluations) for k in ('two_opt','joint_search')}
    bases=[read(OUT/'baselines'/f'case{i:02}.json') for i in range(32)]
    internal_native={k:sum(r['native_internal_calls'].get(k,0) for r in bases) for k in ('solve_route','construct','two_opt','joint_search')}
    times={k:sum(r['elapsed_s'] for r in completes if r['kind']==k) for k in p['top_level_caps']}
    resources=dict(top_level_caps=p['top_level_caps'],terminal_calls=s['terminal_completed'],total_terminal_calls=len(completes),
        full_H_calls=len(evaluations),native_Fixed3_complete_calls=s['terminal_completed']['native'],
        query_label_uses=labels,actual_query_terminal_calls=kind_counts['query'],query_cache_hits=sum(r['cached'] for cp in cps for r in cp['candidates']),
        total_fullH_result_uses=labels+32+32+32,fullH_cache_hits=sum(r['cached'] for cp in cps for r in cp['candidates']),
        determinism_root_rechecks=32,internal_fullH_calls=internal_H,internal_native_calls=internal_native,
        internal_count_scope='actual function invocations, including nestedtwo_opt counted once each; iteration/pass upper bounds not reported as actual counts',
        terminal_elapsed_s_by_kind=times,unique_scenes_generated=32,scene_preparations=dict(calibration=3,main=32,verification=32),
        exported_plans=96,export_geometry_certificates=96,separate_scalar_geometry_certificates=96,total_geometry_certificates=192,
        numeric_saved_fullH_route_replays=v['numeric_completeH_replays'],raw_service_transitions=v['raw_service_transitions'],
        verification_new_completion_trajectories=0,verification_new_H_calls=0,verification_new_native_solver_calls=0,
        new_model_forwards=0,optimizer_updates=0,policy_samples=0,retained_test_queries=0,
        R1_started=False,R2_started=False,R3_started=False,main_run_elapsed_s=s['run_elapsed_s'],wall_limit_s=s['wall_limit_s'],
        calibration_elapsed_s=read(OUT/'wall_protocol.json')['calibration_elapsed_s'],
        registration_amendment='Initial read-only inventory-digest check failed before all research calls; corrected canonical-vs-raw hash interpretation with archived firstregistration. No budget spent or seed/kernel/gate changed.',
        scope='New bounded offline planning research. Top-level count is not all internal search, equal compute, a neural inference metric, or full deployment timing.')
    assert resources['query_label_uses']-resources['query_cache_hits']==resources['actual_query_terminal_calls']
    write(OUT/'resources.json',resources)
    with (OUT/'per_scene.csv').open('w',encoding='utf-8-sig',newline='') as f:
        flat=[]
        for r in rows:
            out={k:r[k] for k in ('case','mode','map_seed','task_seed','gain_J','gain_percent','checkpoints','candidate_uses')}
            for m in quality:
                for k in ('J','L','K'):out[m+'_'+k]=r[m][k]
            flat.append(out)
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    table='\n'.join(f"| {m} | {x['J']:.4f} | {x['L']:.4f} | {x['K']:.4f} |" for m,x in quality.items())
    b=paired['teacher_vs_gBH'];verdict='通过本轮R0机会筛选' if gate['passed'] else '未通过本轮R0机会筛选，停止'
    report=f'''# R0 完整后处理后的有界教师机会检验

日期：2026-11-05；基于8b3ad97；依据已评审修订方案，先登记种子、完整H、账本与壁钟，再执行固定32例。本轮没有网络训练或推理。

**结果：{verdict}。** 相对主参照g_B＋H的平均场景配对改善为{b['mean']:.4f}%，探索性95%区间[{b['ci95'][0]:.4f}%,{b['ci95'][1]:.4f}%]；正收益{gate['positive_cases']}/32，最大单例正相对收益占比{100*share if share is not None else 0:.4f}%，全部最终方案有效。该结果只说明完整H后仍存在有界教师可利用的动作机会，不证明神经网络可学、导航信息必要、RL增量或部署优势。

## 1. 执行合同

32个新百桌实例全部纳入主汇总；regular/mixed/blocked为11/11/10例。map=8300000000+i，task=8400000000+i，i=0…31。已有文本、协议、gzip路线及文件名未发现已记录使用或原60保留测试重叠；二进制/ZIP内部及未登记外部运行不在该检查的声明范围。

g_B固定为每例原生Fixed3服务序列，从任意合法前缀扫描第一个正覆盖增益节点。原生Fixed3成本、g_B根补全后完整H成本、教师成本分别保存，不能混称。原update128没有加载或新增评价。

完整H严格复用原H0（两轮normalize/60-pass2-opt及末尾normalize），再joint_search(passes=2)，最后normalize；源码哈希在protocol.json中固定。环境使用完整未执行前缀、原C/D/逐臂代理约束，最大动作数受任务数限制。

覆盖阈值0、1/8、…、7/8沿教师实际路径首次跨过时查询，同一前缀跨多个阈值不重复。每点至多8唯一合法候选：参考动作、增益/距离规则补至4、距离补至6、独立固定seed探索补至8。成本容差1e-7，并列优先参考后原ID。没有学习网络、未增加部署搜索模式。

## 2. 统一质量矩阵

J=L+0.75K，越小越好。每例均采用相同原始输入。Fixed3为原生完整方法；root是g_B＋H，是本轮主参照；teacher是有界动作查询后的完整路线＋同一H。

| 方法 | 平均J | 平均L | 平均K |
| --- | ---: | ---: | ---: |
{table}

教师相对原生Fixed3平均配对改善{paired['teacher_vs_native_Fixed3']['mean']:.4f}%，但包含参考流程变化和额外动作补全搜索，不能算作模型或强化学习贡献。完整原始/最终路线、候选ID、任务分配和执行臂在r0/保存；本轮未调用Detour16，不能据此给出相对该强基线的部署结论。

## 3. 已固定的四项筛选条件

| 条件 | 登记门槛 | 实际 |
| --- | --- | --- |
| 平均相对主参照改善 | ≥0.5% | {gate['mean_gain_percent']:.4f}% |
| 正收益场景数 | ≥12/32 | {gate['positive_cases']}/32 |
| 最大单例正相对收益占比 | ≤25% | {100*share if share is not None else 0:.4f}% |
| 完成与可行 | 全部32例、全部有效 | 32例；全部有效 |

正收益以绝对J差>1e-7判断；分布占比使用正相对收益，所有阈值用未四舍五入值。主汇总没有挑选子集、补抽种子、放宽门槛或增加调用上限。bootstrap只作为32开发场景的探索性不确定性，非R1学习或最终独立验收。

## 4. 动作机会与排序

共{len(cps)}个实际前缀检查点、{labels}次候选标签使用；{opportunities['checkpoints_with_final_opportunity']}个检查点在查询集合中存在相对参考动作的严格H后改进。即时增益/距离规则在{opportunities['immediate_rule_suboptimal_checkpoints']}个检查点对真实最终成本非最优，平均集合内后悔值{opportunities['mean_immediate_rule_regret_J']:.4f}J。按raw成本选出的候选在{opportunities['raw_min_suboptimal_after_H_checkpoints']}个检查点对最终H成本非最优。

这些只在每点至多8个已查询动作中比较，不是全部合法动作最优；前缀嵌套于32场景，不能把检查点或标签数当作独立样本。逐点记录包括全部合法动作数、来源队列、参考动作、raw/final成本、选择及缓存来源。

每个查询点都核对参考价值与此前选定补全价值连续，并检查选择不增；非检查点沿同一g_B。终局成本与最后参考价值一致。此条件性保护相对g_B＋统一H，不直接等同原生Fixed3、未来近似网络或真实机器人保证。

## 5. 调用、时间与验证

实际顶层评价{resources['total_terminal_calls']}≤2304：原生Fixed3 {s['terminal_completed']['native']}，root {kind_counts['root']}，真实动作终局查询{kind_counts['query']}，教师终局{kind_counts['teacher']}，额外确定性复核{kind_counts['other']}。候选标签使用{labels}次，其中精确缓存命中{resources['query_cache_hits']}次；实际完整H共{len(evaluations)}次。未用额度保留，不因剩余预算追加实验。

原生Fixed3内部solve_route/construct分别为{internal_native['solve_route']}/{internal_native['construct']}次，two_opt {internal_native['two_opt']}次，joint_search {internal_native['joint_search']}次；完整H内部two_opt {internal_H['two_opt']}次、joint_search {internal_H['joint_search']}次，均为真实函数调用数，不冒充内部迭代次数或等算力比较。

前三个预定案例的3次原生和3次root作为已计账校准，并复用结果；主查询前按固定公式锁定壁钟{s['wall_limit_s']}秒。主阶段实际{s['run_elapsed_s']:.3f}秒，校准另{resources['calibration_elapsed_s']:.3f}秒。此为离线教师诊断，未做规范冷热完整部署测时，也没有神经推理时延。

唯一场景生成32例；场景准备为校准3、主阶段32、保存验证32。导出96份有效方案，另对96保存文件做独立标量几何/分配检查，共192证书。对全部{len(evaluations)}条保存完整H评估的raw/final成本、正增益服务和参考后缀数值重放；逐点检查阈值、候选队列、标签来源、最小/并列选择和连续性。验证不再运行H或原生求解，也不构造新补全轨迹，不涉及真实IK。

首次登记后的只读预检曾将清单的规范化TSV摘要误当成JSONL字节摘要，校准前即停止；当时场景、账本和调用数均为0。原登记与源码保存在registration_attempt1，登记修正同时检查规范摘要及独立字节摘要，不改变种子、H、教师或验收阈值。随后所有调用均在修正后的源哈希和壁钟记录下执行。

## 6. 结论与后续边界

R0支持的只是“经过完整后处理后，在本次固定候选查询与g_B补全下仍有分布合理的教师改进机会”。教师使用额外真实补全，其收益不能归为神经网络、导航编码或强化学习。

{'本轮通过只允许把R1作为下一份独立协议的设计对象；不自动开训。' if gate['passed'] else '本轮不启动R1，保留失败或不确定证据，不扩预算。'} R1/R2/R3、旧G1–G4严格信息见证、60保留测试均未启动/未查询。原同结构策略梯度支线保持收尾，原update128研究参照与默认Fixed3/Detour不变。下一阶段若研究有限网络归纳偏置，必须与严格信息缺口主张分开登记。

独立结果到主张审查位于paper/review-traces/experiment-result-to-claim/2026-11-05_run01/；paper claim audit不可用，因此论文级确认仍为provisional。

## 7. 复算入口

`python -B -m rc3_nav_value.r0_analyze`仅复算保存统计及报告；不会规划、推理或训练。r0_run的register/calibrate/run/verify是一次性入口，拒绝隐式重复，不要删除输出后重跑。输入/协议/ledger/evaluations/checkpoints/case_results/96plans/verification/resources构成完整证据链。本轮无整包ZIP备份。
'''
    (OUT.parent/'REPORT_R0.md').write_text(report,encoding='utf-8',newline='\n')
    print(json.dumps(dict(gate=gate,quality=quality,opportunities=opportunities,total_calls=len(completes))))


if __name__=='__main__':main()
