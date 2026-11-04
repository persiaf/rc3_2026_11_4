"""Generate reproducible figures and a Chinese evidence report from frozen runs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('rc3_v3'))
    a=p.parse_args()
    training=read(a.run/'training_summary.json')
    summary=read(a.results/'summary.json')
    manifest=read(a.results/'manifest.json')
    data=read(Path(training['config']['data'])/'manifest.json')
    rows=summary['summaries'];pairs=summary['comparisons']
    chosen=training['selected']
    figures=a.output/'figures';figures.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    with (a.run/'history.csv').open(encoding='utf-8-sig',newline='') as f:history=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(7.5,4.2))
    for arch in ('attention','mlp'):
        for seed in training['config']['seeds']:
            selected=[r for r in history if r['architecture']==arch and int(r['seed'])==seed]
            ax.plot([int(r['epoch']) for r in selected],[float(r['val_gain_pct']) for r in selected],label=f'{arch}, seed {seed}',alpha=.85)
    ax.set(xlabel='Training epoch',ylabel='Validation reduction vs Fixed3 (%)',title='Actual sequential repair validation (100 tables)')
    ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(figures/'training.png');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.0))
    for method,color in [('random','#888888'),('detour','#e08a22'),('learned','#2472b6')]:
        sub=[r for r in rows if r['group']=='in_distribution_100' and r['method']==method and r['kind']=='calls']
        sub.sort(key=lambda r:r['budget'])
        axes[0].plot([r['budget'] for r in sub],[r['mean_improvement_pct'] for r in sub],marker='o',label=method,color=color)
        axes[1].plot([r['median_planning_ms'] for r in sub],[r['mean_improvement_pct'] for r in sub],marker='o',label=method,color=color)
    axes[0].set(xlabel='Maximum repair calls',ylabel='Cost reduction vs Fixed3 (%)',title='Equal call limits')
    axes[1].set(xlabel='Median complete planning time (ms)',ylabel='Cost reduction vs Fixed3 (%)',title='Includes reference and final verification')
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(figures/'quality_time.png');plt.close(fig)
    lines=['# v3 实际训练与冻结测试报告','',
        '本轮从较强的 Fixed3 已验证初解出发，学习选择局部覆盖修复片段。模型训练采用多状态收益监督，**不是新增 PPO 训练**。', '',
        '首轮6次训练仅用于验证诊断，发现STOP与窗口共用softmax会过早停止，保留在run_v3_20261102。正式轮将STOP改为独立全局二分类，固定保守阈值0.9；以下主结果来自重新训练的正式轮。两轮共23,040次参数更新；没有因测试结果重新调参。', '',
        f"训练：{training['training_instances']}个100桌场景，{training['validation_instances']}个独立100桌验证场景；{training['training_states']}个真实中间状态，其中{training['all_zero_states']}个全零收益状态。两个架构各3种子，每个至{training['config']['epochs']}轮，共{training['total_optimizer_updates']}次优化器更新，用时{training['training_s']:.1f}秒。",'',
        f"选中：{chosen['architecture']}，seed {chosen['seed']}，第{chosen['epoch']}轮，验证相对Fixed3平均降本{chosen['val_gain_pct']:.4f}%，{chosen['parameter_count']:,}个参数。选择依据是验证集真实连续修复结果；测试没有参与选模。",'',
        f"数据生成用时{data['generation_s']:.1f}秒，教师真实求解调用{data['statistics']['teacher_repair_calls']:,}次；训练状态覆盖0至7步。数据缓存{data['data_bytes']/1024**2:.2f} MiB，未重复存储大导航矩阵。",'',
        f"冻结测试{manifest['cases']}场景，方法×预算结果{manifest['method_budget_cases']}条，成功{manifest['successful_method_budget_cases']}条、失败{manifest['failed_method_budget_cases']}条；所有重复的最终方案共{manifest['saved_repeated_plans']}份保存供独立复验。",'',
        '## 次数预算结果','',
        '表中降本为逐场景相对Fixed3百分比的均值。完整规划包含初解、特征、模型、修复及最终验证；冷总时间还包含共享导航与可达性预处理。', '',
        '| 组别 | 方法/预算 | 场景数 | 降本% | 规划中位ms | 冷总中位ms | 实际调用均值 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for group in ('in_distribution_100','smaller_transfer','larger_transfer','stress_over_196'):
        for r in rows:
            if r['group']==group and (r['config']=='fixed3' or (r['kind']=='calls' and r['budget']==8)) and r['success']:
                lines.append(f"| {group} | {r['config']} | {r['success']}/{r['total']} | {r['mean_improvement_pct']:.3f} | {r['median_planning_ms']:.2f} | {r['median_total_ms']:.2f} | {r['mean_dp_calls']:.2f} |")
    lines+=['','## 学习选择的独立贡献','',
        '下表正数代表学习成本更低。区间是5000次场景配对bootstrap的95%区间，不包含训练种子不确定性，也未作多重比较校正。', '',
        '| 组别 | 学习预算 | 参照 | 配对数 | 降本% | 95%区间 |',
        '|---|---|---|---:|---:|---|']
    for r in pairs:
        if r['group'] in ('in_distribution_100','smaller_transfer','larger_transfer','stress_over_196') and r['config']=='learned_calls_8':
            lo,hi=r['ci95_pct']
            lines.append(f"| {r['group']} | 8 | {r['reference']} | {r['paired_n']} | {r['mean_reduction_pct']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines+=['','## 软完整时间预算','',
        '只对100桌9个冻结场景评估，时间档从验证集确定。安全调用上限256，超过训练的8步范围。新修复启动前检查时钟，但正在执行的DP和必要最终验证可能超时；因此这是相同软时间上限的实测对照，**不是严格等时或硬实时保证**。每场景重复3次，成本先取重复均值，耗时取中位数；不挑最好一次。','',
        '| 方法/时间档 | 降本% | 规划中位ms | 调用均值 | 超时比例% |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        if r['group']=='in_distribution_100' and r['kind']=='wall' and r['success']:
            lines.append(f"| {r['config']} | {r['mean_improvement_pct']:.3f} | {r['median_planning_ms']:.2f} | {r['mean_dp_calls']:.2f} | {100*r['mean_overrun_rate']:.1f} |")
    lines+=['','## 结论边界','',
        '是否采用学习选择，须同时看相同次数与时间下的降本及置信区间。更低成本但更长耗时不等于全面优于Fixed3；若区间跨零，不声明统计优势。最终模型只是由3种子验证选出的一个检查点，主表不是3种子测试均值。','',
        '可靠性来自已验证初解、覆盖保持修复、最终独立验证和回退；只保证所定义离散可达代理及导航图内的完整服务与成本不增加。实际双臂仍是球壳/扇形代理，固定升降；没有真实IK、双臂协作轨迹、容量约束或在线任务。','',
        '256/400桌各3例仅是压力测试；场景仍属于同一生成机制，不能推论任意布局。程序冷启动、模型加载、文件写入不在规划计时内；所有毫秒数只描述本机此次运行。','',
        '![训练验证曲线](figures/training.png)','',
        '![质量与完整规划时间](figures/quality_time.png)','',
        '逐例原始结果、重复方案、冻结哈希、独立复验见 results/ 与 verification/；开源方法取舍见 OPEN_SOURCE_NOTES.md。']
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(str(a.output/'RESULTS.md'))


if __name__=='__main__':main()
