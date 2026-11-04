"""Build figures and a Chinese evidence report from frozen raw experiment artifacts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from rc3_baseline.baseline import plot_scene


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('rc3_fast'))
    args=parser.parse_args(argv)
    root=args.root
    bench=root/'benchmark_20261102'
    data=read(bench/'summary.json')
    ood=read(root/'benchmark_ood_20261102'/'summary.json')
    plots=root/'figures';plots.mkdir(exist_ok=True)
    summary={(r['group'],r['method']):r for r in data['summaries']}
    comparisons={(r['group'],r['method'],r['reference']):r for r in data['comparisons']}
    methods=['legacy','legacy_nearest','vectorized','default_joint','fixed_top1','learned_top1','fixed_top3','learned_top3']
    names=dict(legacy='Original coverage + 2-opt',legacy_nearest='Original nearest + 2-opt',
               vectorized='Vectorized control',default_joint='Default + dock search',
               fixed_top1='Fixed 1',fixed_top3='Fixed 3',learned_top1='Learned 1',learned_top3='Learned 3')
    scales=[12,36,64,100]
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for method in ('legacy','fixed_top1','learned_top1','fixed_top3','learned_top3'):
        values=[summary[f'tables_{n}',method] for n in scales]
        style='--' if method.startswith('fixed') else '-'
        axes[0].plot(scales,[v['objective_mean_m'] for v in values],style,marker='o',label=names[method])
        axes[1].plot(scales,[v['planning_median_ms'] for v in values],style,marker='o',label=names[method])
        axes[2].plot(scales,[v['cold_total_median_ms'] for v in values],style,marker='o',label=names[method])
    for ax,title,ylabel in zip(axes,['Route quality','Planning, output and validation','Including a new navigation graph'],
                             ['Mean objective (equivalent m)','Median time (ms)','Median time (ms)']):
        ax.set(title=title,xlabel='Number of tables',ylabel=ylabel,xticks=scales)
        ax.grid(alpha=.2)
    axes[1].set_yscale('log');axes[2].set_yscale('log')
    axes[0].legend(fontsize=8)
    fig.suptitle('192 held-out scenes; CPU deployment; lower is better',fontsize=14)
    fig.savefig(plots/'quality_latency.png',dpi=180);plt.close(fig)

    history=list(csv.DictReader((root/'run_portfolio_20261102'/'history.csv').open(encoding='utf-8-sig')))
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for seed in (17,29,43):
        rows=[r for r in history if int(r['seed'])==seed]
        ax.plot([int(r['epoch']) for r in rows],[float(r['top3_regret_pct']) for r in rows],label=f'MLP seed {seed}',alpha=.8,lw=1)
    fixed=read(root/'run_portfolio_20261102'/'training_summary.json')['fixed_validation_regret_pct']['top3']
    ax.axhline(fixed,color='black',ls='--',label='Train-selected fixed 3')
    ax.set(title='More epochs did not consistently improve validation quality',xlabel='Training epoch',
           ylabel='Validation regret to 13-recipe oracle (%)')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(plots/'training_validation.png',dpi=180);plt.close(fig)

    path=sorted((bench/'plans').glob('scene_plan_10x10_blocked_*.json'))[0]
    example=read(path)
    fig,axes=plt.subplots(1,2,figsize=(15,8),layout='constrained')
    for ax,method in zip(axes,['legacy','learned_top3']):
        p=example['plans'][method]
        plot_scene(ax,example,f'{names[method]} | J={p["objective_length_plus_stops_m"]:.2f}',method)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=8,frameon=False)
    fig.suptitle(f'100 tables, {len(example["scene"]["tasks"])} tasks, {len(example["scene"]["obstacles"])} blockers | seed {example["scene"]["seed"]}',fontsize=14)
    fig.savefig(plots/'route_100_blocked.png',dpi=180);plt.close(fig)

    lines=['# 快速覆盖路由：实验结果与交付说明','',
           '本轮完成了可部署模型、共享求解器优化、扩大训练和独立留出测试。主要改善来自向量化和覆盖保持的停靠点优化；是否存在额外学习收益，需看同预算固定策略对照。','',
           '## 方法与数据','',
           '- 最终学习模型为 64 棵树的组合成本回归选择器，按验证集从两类 MLP 与随机森林中选出；一次预测后执行 1 或 3 个构造配置。不是端到端神经路径解码器。',
           '- 训练/验证/测试为 1,152/192/192 个场景，含 12、36、64、100 张桌和三种复杂度。划分种子完全不重叠；最终测试不曾用于训练或选模型。',
           '- 另有 24 个 144/196 桌外推场景，均在模型冻结前确定。外推样本较少，不用于证明一般跨尺寸能力。',
           '- MLP 两种目标各运行 3 个种子×500 轮，每种目标 27,000 次参数更新；另有初试 7,500 次更新。按验证集保留较早检查点，不将训练轮数增加等同于质量提升。',
           '- 路线目标 J=路程+0.75×停靠数。导航与可达性在所有方法间共用，不能把共同预处理优化误计为学习模型收益。','',
           '## 同分布留出测试','',
           '以下为 100 桌子集（48 个场景）的结果。规划耗时包含策略/特征、搜索、完整路径展开和独立校验；冷地图再包含导航预处理。均排除进程启动、模型加载和磁盘写入。','',
           '| 方法 | 平均 J / 等效米 | 规划中位数 / ms | 冷地图总时间 / ms | 缓存地图总时间 / ms |',
           '|---|---:|---:|---:|---:|']
    for method in methods:
        r=summary['tables_100',method]
        lines.append(f'| {names[method]} | {r["objective_mean_m"]:.2f} | {r["planning_median_ms"]:.2f} | {r["cold_total_median_ms"]:.2f} | {r["warm_total_median_ms"]:.2f} |')
    lines+=['','相对原 coverage+2-opt 的成本下降，按每个实例配对后平均；95% 区间为场景级 bootstrap，不是仅按聚合均值计算。','',
            '| 场景 | 模式 | 平均成本下降 | 95% 区间 | 规划速度比中位数 | 冷地图速度比中位数 |',
            '|---|---|---:|---|---:|---:|']
    for group in ('all','tables_100','100_complex','100_blocked'):
        for method in ('learned_top1','learned_top3'):
            r=comparisons[group,method,'legacy'];lo,hi=r['reduction_ci95_pct']
            lines.append(f'| {group} | {names[method]} | {r["mean_cost_reduction_pct"]:.2f}% | [{lo:.2f}%, {hi:.2f}%] | {r["median_planning_speedup"]:.2f}× | {r["median_cold_speedup"]:.2f}× |')
    lines+=['','## 学习的独立贡献','',
            '固定配置/组合均由训练集选出，与学习方法执行相同数量的候选、相同局部搜索。以下正值才代表学习选择优于固定策略。','',
            '| 场景 | 学习与固定对照 | 平均成本下降 | 95% 区间 | 胜/平/负 |',
            '|---|---|---:|---|---|']
    for group in ('all','tables_100'):
        for method,reference in (('learned_top1','fixed_top1'),('learned_top3','fixed_top3')):
            r=comparisons[group,method,reference];lo,hi=r['reduction_ci95_pct']
            lines.append(f'| {group} | {names[method]} vs {names[reference]} | {r["mean_cost_reduction_pct"]:.3f}% | [{lo:.3f}%, {hi:.3f}%] | {r["wins"]}/{r["ties"]}/{r["losses"]} |')
    lines+=['','选择器仍不能被解释为优于强无学习优化的证据。若上述区间跨过 0，独立学习收益不确定；若均值为负，则该测试中固定策略更好。部署时保留固定组合，是根据验证集作出的选择，不是测试后改选模型。','',
            '## 超出训练规模','',
            '| 桌子数 | 方法 | 平均 J | 规划中位数 / ms | 冷地图总时间 / ms |',
            '|---|---|---:|---:|---:|']
    for r in ood['summaries']:
        if r['group'] in ('tables_144','tables_196') and r['method'] in ('legacy','fixed_top1','learned_top1','fixed_top3','learned_top3'):
            lines.append(f'| {r["group"]} | {names[r["method"]]} | {r["objective_mean_m"]:.2f} | {r["planning_median_ms"]:.2f} | {r["cold_total_median_ms"]:.2f} |')
    counts=read(bench/'manifest.json')['valid_plans']+read(root/'benchmark_ood_20261102'/'manifest.json')['valid_plans']
    lines+=['',f'两组测试共 {counts} 个不同“场景×方法”计划通过覆盖、用臂分配、停靠顺序、障碍碰撞、路径和成本一致性检查。每个方法重复计时 3 次；重复计时不增加独立样本数。','',
            '## 训练效率与计时限制','',
            '标签生成只做一次，随后训练读取缓存特征与成本；MLP 数据常驻 GPU。推理导出不依赖 PyTorch/scikit-learn，避免逐停靠调用模型。每个场景特征提取、搜索和独立校验仍有实际成本，不能只报告模型前向时间。','',
            '| 运行 | 训练时间 / s | 更新数或树模型配置数 |', '|---|---:|---:|']
    for name in ('run_selector_20261102','run_extended_20261102','run_portfolio_20261102','run_forest_20261102'):
        s=read(root/name/'training_summary.json')
        lines.append(f'| {name} | {s["training_s"]:.2f} | {s.get("optimizer_updates",len(s["candidates"]))} |')
    lines+=['','CPU：AMD Ryzen 7 7745HX；GPU 训练：RTX 4070 Laptop 8 GB。时延来自本机单次实验中的三次重复中位数，受系统负载影响；未包含实机执行。模型文件和输入地图常驻时适合复用缓存，首次建图耗时不可忽略。','',
            '## 证据边界和后续方向','',
            '- C1 部分支持：百桌 top-1 相对原覆盖基线的平均成本下降 6.87%，配对规划速度比中位数 1.49×；首次建图后的速度比仅 1.10×。小规模场景和三策略模式并不普遍更快。',
            '- C2 未证实：相对同预算固定策略的成本差异置信区间包含 0。下一步需要更有空间结构的信息或直接学习局部改进，继续增加相同数据上的轮数不足以解决问题。',
            '- 仍是二维导航与代理可达性。固定升降、独立单臂任务；没有真实 IK、双臂同步协调、机械臂避障、动力学或在线任务到达。',
            '- 导航只保证指定通道图上的最短路；局部搜索不保证全局最优。复杂障碍分布是结构化生成，不代表任意拥堵环境。','',
            '## 文件','',
            '- `deployment/selector.npz`：验证集选出的最终可部署模型；`selector_neural.npz` 保留神经网络版本。',
            '- `deployment/selection.json`：测试前冻结的模型选择记录和哈希。',
            '- `benchmark_20261102/metrics.csv`、`summary.json`、`plans/`：192 个留出场景的原始结果。',
            '- `benchmark_ood_20261102/`：144/196 桌外推结果。',
            '- `data_extended_20261102/split_audit.json`：划分审计；`run_*/history.csv`：神经训练曲线。',
            '- 运行命令与 API 见 `README.md`。','',
            '![质量与延迟](figures/quality_latency.png)','',
            '![训练与验证](figures/training_validation.png)','',
            '![百桌路径示例](figures/route_100_blocked.png)','',
            f'路径图使用按文件名排序的第一个百桌 blocked 测试场景：`{path.name}`，没有按收益挑选。']
    (root/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote report and 3 figures')


if __name__=='__main__':
    main()
