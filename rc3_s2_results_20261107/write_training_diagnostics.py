"""Write descriptive tables from the closed S2 analysis, with zero new evaluation."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import time

HERE = Path(__file__).resolve().parent


def table(headers, rows):
    def fmt(value):
        return f'{value:.6f}' if isinstance(value, float) else str(value)
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(fmt(x) for x in row) + ' |' for row in rows])


def main():
    started = time.perf_counter()
    summary_path = HERE / 'analysis/ANALYSIS_SUMMARY.json'
    summary = json.loads(summary_path.read_text(encoding='utf8'))
    with (HERE / 'analysis/updates.csv').open(encoding='utf8', newline='') as f:
        updates = list(csv.DictReader(f))
    with (HERE / 'analysis/group_summary.csv').open(encoding='utf8', newline='') as f:
        groups = list(csv.DictReader(f))
    grad, signal, endpoints, costs, resources, windows = [], [], [], [], [], []
    for s in summary['seeds']:
        seed = s['seed']
        for phase in ('RL', 'continued_SL'):
            p = s['phase_diagnostics'][phase]
            grad.append([seed, phase, p['gradient_before']['mean'], p['gradient_before']['min'],
                         p['gradient_before']['max'], f"{p['clipped_updates_n']}/{p['completed_updates']}"])
            rows = [r for r in updates if int(r['seed']) == seed and r['phase'] == phase]
            if len(rows) != 512:
                raise ValueError('Diagnostics require the actual fixed 512-update endpoint')
            values = [float(r['loss']) for r in rows]
            endpoints.append([seed, phase, statistics.fmean(values[:32]), statistics.fmean(values[-32:])])
        a = s['advantage_distribution']
        signal.append([seed, a['std_population'], a['p05'], a['p95'], a['exact_zero_n'],
                       s['all_zero_advantage_instances_n'], s['phase_diagnostics']['RL']['all_zero_advantage_updates_n'],
                       s['entropy_decision_weighted_mean']])
        costs.append([seed, s['raw_mean_J'], s['H_mean_J'], s['raw_mean_L'], s['H_mean_L'],
                      s['raw_mean_K'], s['H_mean_K'], s['mean_H_gain']])
        l = s['resource_ledger']
        resources.append([seed, s['stage_wall_s'], l.get('task_scene_generations', 0),
                          l.get('navigation_completed', 0), l['environment_actions'],
                          l['terminal']['RL']['full_H_calls'] + l['terminal']['development']['full_H_calls']])
        for group in ('full10x10', 'local4x3', 'local6x6', 'local8x8'):
            matches = [r for r in groups if int(r['seed']) == seed and r['group_dimensions'] == 'seed|window_group'
                       and r['window_group'] == group and r['metric'] == 'H_J']
            if len(matches) != 1:
                raise ValueError('Missing or ambiguous window summary')
            r = matches[0]
            windows.append([seed, group, int(r['trajectory_n']), int(r['instance_n']),
                            float(r['mean']), float(r['p25']), float(r['p75'])])
    text = '# S2 保存数据训练诊断\n\n'
    text += ('三种子均为固定 RL512 / continued-SL512。这里使用保存训练抽样，不新增模型推理、优化器或路线评价。'
             '三个起点属于同一实例的相关轨迹；各 update 的任务变化，以下均值、分位和首末区间不能作为配对改善或泛化结论。\n\n')
    text += table(['seed', '训练raw J', '训练H后 J', 'raw L', 'H后 L', 'raw K', 'H后 K', 'H贡献raw J−H J'], costs)
    text += '\n\nH贡献全部归于原冻结后处理，不能归于神经网络。\n\n'
    text += table(['seed', 'phase', '裁剪前梯度范数均值', '最小', '最大', '实际裁剪次数/更新'], grad)
    text += ('\n\n所有1536次RL更新都触发范数1裁剪；continued-SL合计1534/1536次。'
             '梯度均有限，但RL原始范数较大且波动显著，说明更新长期受裁剪影响；数值有限不能证明学习有效或收敛。'
             '未据此调整学习率、奖励、算法、预算或裁剪阈值。\n\n')
    text += table(['seed', 'LOO优势总体SD', 'p05', 'p95', '精确零/6144轨迹', '全零/2048实例',
                   '全零/512更新', '按决策数加权熵(nats)'], signal)
    text += ('\n\n优势的均值因LOO结构接近零，不是学习失败判据；精确零和接近零分开记账。'
             '不存在整批全零优势更新。熵仅覆盖实际评分决策，不给强制首动作虚构评分。\n\n')
    text += table(['seed', 'phase', '前32次已记录loss均值', '后32次已记录loss均值'], endpoints)
    text += ('\n\nRL列为有符号REINFORCE目标，continued-SL列为masked CE，两者定义和尺度不同。'
             '这些是训练批次统计，不是固定起点和终点的独立模型评价，不能用loss大小比较规划质量。\n\n')
    text += table(['seed', '训练窗口', '轨迹数', '抽样实例次数', 'H后 J均值', 'p25', 'p75'], windows)
    text += ('\n\n窗口统计按实际训练抽样汇总；抽样次数不是唯一独立场景数。障碍模式、入口出口与起点角色的进一步交叉分组见'
             '[group_summary.csv](analysis/group_summary.csv)，未对未见规模或新布局作评价。\n\n')
    text += table(['seed', '累计阶段秒', '新增预登记训练/开发任务生成', '新增导航求解', 'RL环境动作', '实际完整H调用'], resources)
    text += ('\n\n任务生成均来自预登记train/dev银行，不是新增教师。S2每seed的一份累计时钟包括准备、导航、训练、H、开发、'
             '验证及检查点I/O，不能把phase快照再次相加；后处理分析与Git传输另账。全部保存终局计划18539/18539通过当时代理验证，'
             '失败0，缓存命中1；无新教师/native调用，无S3/原60语义消费。\n\n'
             '结论：固定训练阶段已真实完成且来源、保存数值和预算通过；跨seed回报差异仍受各自抽样影响。'
             '冻结模型相对frozen-SL、continued-SL、未训练和强规则的完整路径质量与时间优势，须由后续获准S3独立验证。\n')
    (HERE / 'TRAINING_DIAGNOSTICS.md').write_text(text, encoding='utf8', newline='\n')
    print(json.dumps(dict(status='DESCRIPTIVE_DIAGNOSTICS_WRITTEN', wall_s=time.perf_counter()-started,
                          summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(), scientific_calls=0)))


if __name__ == '__main__':
    main()
