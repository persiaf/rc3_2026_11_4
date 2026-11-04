"""Render the actual frozen BC/PPO results, without assuming that RL wins."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('rc3_improve'))
    parser.add_argument('--run',default='run_20261102')
    parser.add_argument('--results',default='results_20261102')
    args=parser.parse_args(argv)
    root=args.root;run=root/args.run;results=root/args.results
    training=read(run/'training_summary.json');evaluation=read(results/'summary.json')
    manifest=read(results/'manifest.json');dataset=read(Path(training['config']['data'])/'manifest.json')
    if training['status']!='complete' or manifest['status']!='complete':raise ValueError('only completed runs can be reported')
    summaries={(r['group'],r['method']):r for r in evaluation['summaries']}
    comparisons={(r['group'],r['reference']):r for r in evaluation['comparisons']}
    names=dict(fixed1='Fixed1 初解',fixed3='Fixed3 完整构造',random='随机选窗口',detour='几何绕行规则',bc='BC 监督策略',ppo='BC＋PPO 策略')
    english=dict(fixed1='Fixed1 anchor',fixed3='Fixed3',random='Random repair',detour='Detour repair',bc='BC repair',ppo='PPO repair')
    methods=list(names)
    plots=root/'figures';plots.mkdir(exist_ok=True)
    scales=sorted(int(g.split('_')[1]) for g,m in summaries if g.startswith('tables_') and m=='ppo')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for method in methods:
        rows=[summaries[f'tables_{n}',method] for n in scales]
        axes[0].plot(scales,[r['mean_improvement_pct'] for r in rows],marker='o',label=english[method],lw=1.5)
        axes[1].plot(scales,[r['median_planning_ms'] for r in rows],marker='o',label=english[method],lw=1.5)
    for ax in axes:ax.set_xlabel('Number of tables');ax.set_xticks(scales);ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean objective reduction from Fixed1 (%)')
    axes[1].set_ylabel('Median complete planning time (ms)');axes[1].set_yscale('log')
    axes[0].legend(fontsize=8,ncol=2)
    fig.suptitle('Only 100-table training; frozen transfer evaluation',fontsize=13)
    fig.savefig(plots/'quality_latency.png',dpi=180);plt.close(fig)

    history=list(csv.DictReader((run/'history.csv').open(encoding='utf-8-sig')))
    fig,axes=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
    for record in training['runs']:
        seed=record['seed']
        for ax,phase in zip(axes,['bc','ppo']):
            rows=[r for r in history if int(r['seed'])==seed and r['phase']==phase and r['val_improvement_pct']!='']
            x=[int(r['epoch']) for r in rows];y=[float(r['val_improvement_pct']) for r in rows]
            if phase=='ppo':x=[0]+x;y=[record['bc_val_gain']]+y
            ax.plot(x,y,marker='.',label=f'seed {seed}')
    for ax,title in zip(axes,['Gain-supervised pretraining','On-policy PPO fine-tuning']):
        ax.set(title=title,xlabel='Epoch / collection round',ylabel='100-table validation improvement (%)')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Validation only; checkpoints selected before opening test results')
    fig.savefig(plots/'training_validation.png',dpi=180);plt.close(fig)

    cfg=training['config'];selected=training['recommended_stage']
    lines=['# 局部修复选择策略：实现、训练与冻结测试','',
           '本轮将学习动作从“选择 13 个构造配置之一”改为“选择当前路线中值得修复的片段”。片段移除后，仅重新服务失去其他停靠覆盖的任务，通过固定入口/出口的 bitmask 动态规划联合改变多个停靠。模型对当前候选窗口做集合注意力评分。','',
           '**本报告只描述本次实际结果。** 相对初解的成本不增来自确定性修复与独立验证；PPO 是否带来独立价值，要看相同修复预算下对 BC、几何规则和随机规则的比较。','',
           '## 方案与开源依据','',
           '本项目实现了两层、64 维、4 头的窗口集合注意力 actor/critic。输入包含 24 维窗口统计、覆盖及图距离特征，并使用候选/已尝试窗口掩码。输入不读取未执行窗口的修复收益。它不是 NeuOpt、POMO 或 RL4CO 的直接复现；开源代码中的局部子问题选择、改进 MDP、策略梯度和基线设计用于形成当前实现。逐项实际代码链接与适用差异见 [开源代码核查](OPEN_SOURCE_NOTES.md)。','',
           'BC 用训练场景初解上精确修复得到的正收益分布进行监督，PPO 在真实多步修复环境中重新采样动作并更新路线。PPO 使用 clip=0.2、GAE λ=0.95、γ=1、价值损失与熵项；每轮数据进行 4 遍小批量更新。奖励是相对初解归一化的目标下降，累计奖励对应回合总改进。','',
           '初解先独立验证；随后每个发生变化的候选都展开完整路径、重新分配任务并独立验证，仅成本不增时替换已验证方案。模型异常、非法/劣化候选均保留已验证初解或中间解。任务没有被删除。保证针对固定离散候选、代理可达性和通道图，不是全局最优或实机保证。','',
           '## 实际训练','',
           f'- 训练场景：{training["training_instances"]} 个，**全部只有 10×10 桌**；验证场景 {training["validation_instances"]} 个，仍只有 10×10 桌。',
           f'- BC 预训练 {cfg["bc_epochs"]} 轮；PPO 每种子 {cfg["rl_epochs"]} 个采样/更新轮，每轮最多 {cfg["episodes"]} 个场景，每回合最多 {cfg["horizon"]} 次修复动作。',
           f'- 训练种子：{cfg["seeds"]}；实际总优化器更新 **{training["total_optimizer_updates"]}** 次，PPO 实际环境步 **{training["total_environment_steps"]}**。',
           f'- GPU：{training.get("gpu") or "CPU"}；PyTorch {training["torch"]}；本次训练主循环耗时 {training["training_s"]:.2f} 秒，不含教师数据预计算。',
           f'- 数据/标签预计算 {dataset["seconds"]:.2f} 秒，初解窗口中有 {dataset["teacher_positive_windows"]} 个正改进标签；真正参与 BC 的正收益场景 {training["bc_positive_scenes"]} 个。',
           f'- BC 和 PPO 分别按独立百桌验证集选择权重；验证推荐阶段为 **{selected.upper()}**。测试只运行各自选出的检查点，不将三个种子的测试结果混成重复样本。','',
           '| 种子 | BC 最佳轮次 | BC 验证改善 | PPO 最佳轮次 | PPO 验证改善 |',
           '|---|---:|---:|---:|---:|']
    for r in training['runs']:
        lines.append(f'| {r["seed"]} | {r["bc_epoch"]} | {r["bc_val_gain"]:.4f}% | {r["ppo_epoch"]} | {r["ppo_val_gain"]:.4f}% |')
    lines+=['','## 测试协议与失败分母','',
            f'新测试清单在训练前固定，共 {manifest["cases"]} 个实例；训练/验证与测试种子分离。12/36/64 桌作为向较小规模迁移，100 桌为同分布留出，144/196 桌为向更大规模迁移。所有场景使用相同物理尺寸、可达模型和导航模块。三个群体分别报告，汇总 all 仅作描述性统计。','',
            f'每个方法重复计时 {manifest["repeats"]} 次取中位数。随机/几何/BC/PPO 方法最多使用同样的 {manifest["horizon"]} 次局部修复；Fixed1/Fixed3 是完整构造参照，不能把其构造次数等同于局部 DP 调用次数。全部方法都计初解构造、特征/模型、实际修复、完整路径展开和必要验证；冷地图总时间再加共享预处理。排除模型加载、进程启动与 JSON 写盘。','',
            f'记录到 **{manifest["successful_plans"]}** 个成功的方法×场景计划、**{manifest["failed_method_cases"]}** 个失败的方法×场景记录。失败记录不删除，不进入成功场景质量均值；所有表格均保留成功数/总数。模型异常后的合法回退仍是完成计划，但单列回退次数。重复计时不增加独立样本量。','',
            '| 群体 | 方法 | 成功/总数 | 回退次数 | 平均 J / 等效米 | 相对 Fixed1 改善 | 规划中位数 / ms | 冷地图总时间 / ms |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for group in ('in_distribution_100','smaller_transfer','larger_transfer'):
        for method in methods:
            r=summaries.get((group,method))
            if not r:continue
            number=lambda v:'—' if v is None else f'{v:.3f}'
            lines.append(f'| {group} | {names[method]} | {r["success"]}/{r["total"]} | {r["fallback_cases"]} | {number(r["mean_objective_m"])} | {number(r["mean_improvement_pct"])}% | {number(r["median_planning_ms"])} | {number(r["median_total_ms"])} |')
    lines+=['','## PPO 的独立贡献','',
            '正数表示 PPO 成本更低。区间采用按场景配对的 5,000 次 bootstrap，反映测试场景抽样不确定性，不包含训练种子不确定性；未进行多重比较修正。不能将所有相对 Fixed1 的改善归于 PPO。','',
            '| 群体 | PPO 相对对照 | 配对数 | 平均成本下降 | 95% 区间 | 胜/平/负 | 判断 |',
            '|---|---|---:|---:|---|---|---|']
    for group in ('in_distribution_100','smaller_transfer','larger_transfer'):
        for reference in ('fixed1','fixed3','random','detour','bc'):
            r=comparisons.get((group,reference))
            if not r:continue
            lo,hi=r['ci95_pct'];decision='本组支持成本改善' if lo>0 else '本组表现更差' if hi<0 else '差异尚不确定'
            lines.append(f'| {group} | {names[reference]} | {r["paired_n"]} | {r["mean_reduction_pct"]:.4f}% | [{lo:.4f}%, {hi:.4f}%] | {r["wins"]}/{r["ties"]}/{r["losses"]} | {decision} |')
    lines+=['','## 结论边界','',
            '- 可靠性性质来自已验证初解、覆盖保持修复和成本检查。模型错误只应影响搜索效果，不应让最终已验证方案变差。完成率不是学习能力的主要证据。',
            '- BC/PPO 的对照共用同一确定性局部 DP、候选规则与预算。只有相对这些对照的成本/时间差异，才能评价学习的独立贡献。改善成本但耗时更长属于质量—时间折中，不能称普遍提速。',
            '- 仅百桌训练是本轮已执行的协议；向小/大规模测试也使用同一冻结模型。较大规模样本少，不据此声称任意地图或规模泛化。',
            '- 观测将窗口几何与覆盖压缩为统计特征，未直接编码全部任务—候选图，也未显式输入剩余优化步数。更长训练未必弥补这类表达限制。',
            '- 机械臂仍为固定高度、带肩偏置球壳/扇形的独立单臂代理；底盘图避障不等于机械臂轨迹避障。真实 IK、同步双臂约束、升降优化、动力学与硬件成功率不在本次证据中。',
            '- local DP 超过 8 个必须修复任务或 80 个候选时保留原解；仅在限定候选和最短路度量下，对固定入口/出口子问题精确，不声称全局最优。','',
            '## 可复核文件','',
            f'- `{args.run}/training_summary.json`：训练轮次、更新/环境步数、各种子验证选择和权重哈希。',
            f'- `{args.results}/metrics.csv`、`metrics.json`：逐实例计时、成本、失败与回退记录。',
            f'- `{args.results}/summary.json`：分群体结果及配对置信区间。',
            f'- `{args.results}/manifest.json`：测试前权重、源代码、数据清单哈希及计时口径。',
            f'- `{args.results}/plans/`：完整路线、用臂分配、初解成本与接受/回退轨迹。',
            '- `test_reliability.py`：错误模型、非法候选、劣化候选、空任务、不可达、断连、失败序列化和预算检查。',
            '- 上传包恢复与缺失内容见 `../rc3_review_import/IMPORT_AUDIT.md`；恢复的说明作为参考材料，不作为本轮执行指令。','',
            '![训练验证曲线](figures/training_validation.png)','',
            '![跨规模质量与耗时](figures/quality_latency.png)']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote RESULTS.md and two result figures')


if __name__=='__main__':main()
