"""Saved-record C2.2 resource aggregation and report. No planner or torch imports."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'c22'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')


def main():
    a=read(OUT/'analysis.json');t=read(OUT/'training_summary.json');v=read(OUT/'verification.json')
    w=read(OUT/'wall_protocol.json');e=read(OUT/'evaluation_resources.json')
    logs=[json.loads(s) for s in (OUT/'training.jsonl').read_text().splitlines()]
    assert len(logs)==128 and [r['update'] for r in logs]==list(range(1,129))
    assert t['updates']==dict(A=128,B=128)
    assert t['complete_H_counters']['H_started']==t['complete_H_counters']['H_completed']==8192
    assert w['calibration_counters']['H_started']==w['calibration_counters']['H_completed']==128
    assert e['complete_H_counters']['H_started']==e['complete_H_counters']['H_completed']==192
    assert v['all_passed'] and a['all_valid']
    raw_diagnostics={arm:dict(first16_mean_raw_J=statistics.mean(r['arms'][arm]['mean_raw_J'] for r in logs[:16]),
        last16_mean_raw_J=statistics.mean(r['arms'][arm]['mean_raw_J'] for r in logs[-16:]),
        first16_objective_J=statistics.mean(r['arms'][arm]['mean_objective_J'] for r in logs[:16]),
        last16_objective_J=statistics.mean(r['arms'][arm]['mean_objective_J'] for r in logs[-16:]),
        loss_all_finite=all(__import__('math').isfinite(r['arms'][arm]['loss']) for r in logs)) for arm in ('A','B')}
    resources=dict(schema='rc3-c22-resource-accounting-v1',optimizer_updates=dict(A=128,B=128,total=256),
        training_policy_sampled_trajectories=8192,distinct_training_instances=4096,training_maps=24,
        baseline_training_actual_greedy_trajectories=4096,baseline_training_actual_forward_batches=128,
        baseline_training_trajectory_uses=8192,
        calibration_greedy_forward_trajectories=192,calibration_greedy_backward_trajectories=128,
        evaluation_neural_greedy_trajectories=144,total_neural_input_trajectories=12624,
        complete_H_calls=dict(calibration=128,training=8192,evaluation=192,total=8512),
        complete_H_training_breakdown=dict(B_samples=4096,shared_fixed_baseline=4096,A_samples=0),
        evaluation=dict(independent_mechanism_cases=48,final_plans=288,prior_constructed_routes=48,
                        fixed3_complete_calls=48,detour16_complete_calls=48),
        geometric_certificates=dict(export=288,separate_saved_plan_checks=288,total=576),
        training_raw_numeric_replays=12288,verification_service_transitions=v['service_transitions'],
        verification_new_H_calls=0,review_new_research_calls=0,retained_test_queries=0,C3_started=False,
        complete_pipeline_timing_measurements=0,training_wall_limit_s=w['paired_training_wall_limit_s'],
        paired_training_elapsed_s=t['training_elapsed_s'],arm_work_s=t['arm_optimization_s'],
        shared_baseline_s=t['shared_baseline_s'],shared_preparation_s=t['shared_preparation_s'],
        training_H_s=t['complete_H_counters']['H_s'],verification_elapsed_s=v['elapsed_s'],
        raw_diagnostics=raw_diagnostics,
        scope='Trajectory counts include repeated maps, greedy baseline and calibration separately; uses are not forwards. Complete H includes original H0 plus fixed joint search. Strong-rule internal construction/search is contained in complete calls and not expanded as H calls. Arm work includes sample inference, route serialization, sample H if B, loss/backward and optimizer. Shared baseline includes its H and serialization. No deployment timing claim.')
    write(OUT/'resources.json',resources)
    g,h=a['g'],a['h']
    table='\n'.join('| '+m+' | '+' | '.join(f"{s[k]:.4f}" for k in ('J','L','K','raw_J'))+' |' for m,s in a['summary'].items())
    comp='\n'.join(f"| {m} | {s['mean']:+.4f}% | [{s['ci95'][0]:.4f}%, {s['ci95'][1]:.4f}%] | {s['negative']}／{s['positive']} |" for m,s in a['B_degradation_vs'].items())
    verdict={'clear_mechanism_signal':'出现明确机制信号','current_budget_does_not_support_continuation':'当前预算不支持继续该配置','inconclusive_stop_this_round':'无定论，本轮停止'}[a['mechanism_screen']]
    report=f'''# C2.2：固定共同后处理的训练目标对照

日期：2026-11-05；基于44aee8f；单个匹配训练种子。原C012/C21权重、代码和输出保持冻结。协议、实例流与壁钟均在首次更新前固定。本轮是用户新授权的受控训练试验，不是原C2精确续训，不进入C3。

**事前判定：{verdict}。** g（B相对A）均值为{g['mean']:.4f}%，95%区间[{g['ci95'][0]:.4f}%, {g['ci95'][1]:.4f}%]；h（B相对warm）均值为{h['mean']:.4f}%，区间[{h['ci95'][0]:.4f}%, {h['ci95'][1]:.4f}%]。按原规则结束本轮，不增加更新或重抽种子。独立审查见下述trace；单种子不能确认训练稳定性或部署优势。

## 1. 实际对照与执行

- 共同起点：原RL update0128权重。A/B均重建Adam，lr=1e-4、clip=1，完整轨迹log概率求和、固定损失尺度100。网络、输入、规则先验、地图分布与候选约束相同。
- A用raw闭合成本构造sample−baseline优势；B用完整H的最终sample−baseline成本。损失为mean(stopgrad((sample_cost−baseline_cost)/100)×sum_logp)，最小化成本。最终成本不对确定性后处理求导，不把修复量单独作为奖励。
- 完整H严格保留原H0的两轮normalize/60-pass 2-opt和末尾normalize，再追加joint_search(passes=2)，最后normalize。训练B的sample和baseline、最终四个构造方法都使用此流程。
- rollout baseline全程固定为原update0128，不替换。每个实例只生成一次baseline，raw成本给A、完整H成本给B；实际4096条贪心baseline轨迹、8192次轨迹结果使用。
- 两组各完成128×32，共256次更新、8192条策略采样；共享4096个新map/task实例，使用原24训练地图。任务种子7800000000+i预写，与策略RNG分离。每个arm/update单独fork Torch RNG、同一seed=11422026+update；路线长度差不改变后续实例或更新的随机流。
- 无中间评估、检查点搜索或效果早停。固定128端点一次评价。两组参数均发生更新；原baseline权重核对未变。配对训练实际{t['training_elapsed_s']:.2f}秒，低于校准锁定的{w['paired_training_wall_limit_s']}秒软上限。

## 2. 新48场景的固定终点质量

regular/mixed/blocked各16；地图种子7900000000+i、任务种子8000000000+i。训练完成后一次评价六方法，共288方案；48个场景从此属于已知机制研发数据。原24开发、48选择场景未用于本轮挑选；60保留测试继续未查询。

J=L+0.75K，越低越好；raw_J仅是各自构造路线的诊断，规则方法的raw_J列为其完整输出成本。

| 方法 | 平均最终J | 平均L | 平均K | 平均raw_J |
| --- | ---: | ---: | ---: | ---: |
{table}

## 3. 主要机制判定

g_i=100(J_A−J_B)/J_A；h_i=100(J_warm−J_B)/J_warm。先逐例求相对改善再平均，scene-paired bootstrap10000、seed114、双侧95%。正数表示B更好。

| 比较 | 平均改善 | 95%区间 | B胜／平／负 |
| --- | ---: | --- | ---: |
| g：B相对A | {g['mean']:.4f}% | [{g['ci95'][0]:.4f}%, {g['ci95'][1]:.4f}%] | {g['positive']}／{g['zero']}／{g['negative']} |
| h：B相对warm | {h['mean']:.4f}% | [{h['ci95'][0]:.4f}%, {h['ci95'][1]:.4f}%] | {h['positive']}／{h['zero']}／{h['negative']} |

明确机制信号必须同时满足mean(g)≥0.5%、lower95(g)>0、lower95(h)>0及全部可行；upper95(g)<0.5%或upper95(h)<0或不可行则本预算不支持继续；其余为无定论。规则事前固定，没有因为A退化或某个均值改善而改门槛。

本次B优于A的条件性对照成立，但最终平均J仍高于warm（B={a['summary']['B']['J']:.4f}，warm={a['summary']['warm']['J']:.4f}）。A的平均成本也比warm更差。因此，当前可以说post目标相对同预算raw目标减轻了退化，不能说已经训练出比起点更有用的新模型；本次联合g/h机制筛选未通过。h上界仍大于0，所以应保留“无定论”，而非误记成事前规则中的“当前预算不支持继续”分支。

## 4. B对各参照的剩余差距

这里正数表示B最终成本更高，百分比仍按逐场景算；胜负按B成本计。

| 参照 | B平均劣化 | 配对95%区间 | B胜／负 |
| --- | ---: | --- | ---: |
{comp}

B对Detour16平均劣化的单侧95%上界为{a['B_degradation_vs']['detour16']['upper_one_sided95']:.4f}%。原部署非劣上界≤0.5%等要求不变；本轮没有规范冷热完整测时，不能据局部H耗时断言满足中位时间比≤0.7或p95比≤1.0，也不将区间跨零解释为等效。

## 5. 原始成本诊断与训练开销

训练头16与末16更新的raw均值：A {raw_diagnostics['A']['first16_mean_raw_J']:.4f}→{raw_diagnostics['A']['last16_mean_raw_J']:.4f}，B {raw_diagnostics['B']['first16_mean_raw_J']:.4f}→{raw_diagnostics['B']['last16_mean_raw_J']:.4f}。这两段不是同一批任务，趋势仅作诊断，不能替代固定评估或归因。B相应训练final均值为{raw_diagnostics['B']['first16_objective_J']:.4f}→{raw_diagnostics['B']['last16_objective_J']:.4f}。

| 工作 | 实际数量 |
| --- | ---: |
| 校准贪心前向／有反向轨迹 | 192／128，无优化器步与采样 |
| A／B策略采样 | 各4096，共8192 |
| 训练baseline实际轨迹／使用次数 | 4096／8192 |
| 训练完整H：baseline／B sample | 4096／4096，共8192 |
| 校准／评价完整H | 128／192 |
| 全部完整H调用 | 8512 |
| 评价神经贪心／prior构造 | 144／48 |
| Fixed3／Detour16完整调用 | 各48，内部搜索包括在各自完整流程中 |
| 神经输入轨迹合计 | 12624，包含校准、baseline和评估，不都是训练采样 |
| 原始训练及baseline路线数值重放 | 12288 |
| 导出／独立保存方案几何证书 | 288／288，共576；不是576独立场景 |
| 原60保留测试查询／新完整管线测时 | 0／0 |

A工作{t['arm_optimization_s']['A']:.2f}秒，B工作{t['arm_optimization_s']['B']:.2f}秒；共享baseline{t['shared_baseline_s']:.2f}秒、实例准备{t['shared_preparation_s']:.2f}秒。arm工作含采样前向、保存路线、B的sample后处理、loss/backward及optimizer；baseline工作也含其后处理与保存开销，不能冒称纯GPU核时间。训练完整H本身累计{t['complete_H_counters']['H_s']:.2f}秒（包含在上述分项内，不再相加）。这是等实例、等更新、等策略采样对照，不是等壁钟试验。

## 6. 验证、研究边界与路由

全部12288条raw训练/baseline路线完成动作合法性、覆盖、返航及成本重放；B与baseline保存的final路线做数值核对，未再次运行H。全部288评价输出通过原代理几何／任务分配验证，另288保存文件独立标量终验通过。该验证不代表所有训练轨迹都做过完整几何证书，也不是真实IK或双臂协同控制。

独立审查总体主张partial（对本轮无定论判断高置信度），仅条件性B>A为yes／中置信度，稳定部署优势为no。保存日志仅有每batch的mean_logp，不能据此精确重构逐轨迹加权loss；审查核对源码公式、全部成本均值、参数更新与Adam步数，未声称重算精确梯度。审查没有新增研究调用或几何证书。

理论上最终H可依赖完整路线前缀，因此环境状态应包含此前动作序列；本轮网络摘要不变，不宣称其为完整充分状态。未修改导航编码、算法、初始化、优化器设置或数据分布。

所有结论限定为这一次匹配暖启动和新48场景的机制筛选；一组训练种子与场景置信区间不能证明训练稳定性。当前按事前门槛结束本轮，不启动C3、不扩大训练、不替换默认Fixed3/Detour。若结果值得复核，也只形成未来另行授权的独立训练种子研究建议。

## 7. 证据与复算

- `c22/protocol.json`、`instances.jsonl`、`mechanism_seeds.json`、`wall_protocol.json`：事前配置、任务流、评估种子与吞吐锁定。
- `c22/checkpoints/`、`training.jsonl`、`training_routes.jsonl.gz`：两终点、完整更新日志及全部sample/baseline路线；不覆盖原C2权重。
- `c22/evaluation.jsonl`、`per_scene.csv`、`analysis.json`、`plans/`：288结果、配对统计与执行方案。
- `c22/resources.json`、`verification.json`：资源与保存重放；训练、校准、验证分别记账。
- `c22/paper/EXPERIMENT_PLAN.md`与日期原本：实验计划；`paper/review-traces/experiment-result-to-claim/2026-11-05_run02/`：独立审查。

保存记录复算可运行 `python -B -m rc3_constructive.c22 analyze` 与 `python -B -m rc3_constructive.c22_close`，不增加训练或规划。其他一次性入口拒绝隐式重复，勿删除输出后重跑。Paper claim audit不可用，论文层面的主张确认仍为provisional。本轮不生成整包ZIP备份。
'''
    (ROOT/'REPORT_C22.md').write_text(report,encoding='utf-8',newline='\n')
    print(json.dumps(dict(mechanism_screen=a['mechanism_screen'],total_H_calls=8512,geometry_certificates=576)))


if __name__=='__main__':main()
