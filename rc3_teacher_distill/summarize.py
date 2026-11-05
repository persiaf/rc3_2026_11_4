"""Write T-series reports solely from saved results. No planning or model calls."""
import json
from .register import OUT,read,write,digest


def main():
    status=read(OUT/'STATUS.json');resources=read(OUT/'resources.json')
    t1_complete=status.get('T1') in ('passed','failed_stop') and status.get('stage')!='T1'
    t2_complete=status.get('T2')=='complete' and status.get('status')!='inconclusive_stop'
    t2_started=bool(resources['ledger']['used'].get('T2_routes',0) or resources['ledger']['used'].get('T2_teacher',0)
                    or status.get('T2') in ('started','complete','inconclusive_stop'))
    lines=['# T0～T2 教师蒸馏与可靠动作决策报告','',
        '维护基准为5d5fe25，旧研究证据冻结于df8cf28。本轮位于独立本地分支codex/teacher-distill-t0-t2；未自动推送。',
        '', '## 事前登记与隔离','',
        f"协议SHA256：`{digest(OUT/'protocol.json')}`。全部新种子、源码/H哈希、监督配置、查询与时间上限在首个新场景和优化器步骤前登记；详见protocol.json及registration_receipt.json。",
        '固定教师为gB-v1（场景内原生Fixed3顺序），全部标签为完整前缀加动作再按gB补全、经冻结完整H的终局成本。没有混用R1后轮g0标签。',
        '原60保留测试继续封存；不执行PPO/API/T3，也不修改原约束、H或默认Fixed3／Detour。',
        '', '## T0：旧证据只读诊断','',
        '完整报告见T0_DIAGNOSTIC.md：R0存在大量H后并列，R1初始gB与后续暖网络g0为不同教师，后轮动作审计来自训练状态。T0不证明网络已经具备可学习性。',
        '', '## T1：统一菜单与匹配监督','',
        '同一K≤8菜单、相同38个部署可用特征、相同训练/校准/开发划分。规则为菜单内gain/(导航距离+0.75)，轻量模型为线性排序/价值模型，神经模型为两层64宽MLP；各学习模型3个固定种子、每种子1024步并取端点均值，不做开发集选轮或调参。',
        '完整前缀用于精确状态与终局标签，模型接收前缀统计及顺序加权摘要；这不是无损序列编码。没有安装LightGBM或声称运行LambdaMART。',
        '96个训练实例中80个拟合、16个仅校准；开发24例独立地图。主指标对场景均权，场景内对保存的不同前缀均权，配对bootstrap以场景为单位。',
        '菜单每个动作都实际查询并保存终局标签；未知选择为零来自有界接口，并不证明全合法动作菜单的外推能力。',
        '', '| 方法 | 并列最优命中% | 0.1%近优% | 0.5%近优% | 平均遗憾% | 相对规则遗憾改善%及95%区间 |',
        '|---|---:|---:|---:|---:|---|']
    if (OUT/'T1_GATE.json').exists():
        neural=read(OUT/'neural_actions.json');linear=read(OUT/'linear_actions.json')
        labels=[json.loads(s) for s in (OUT/'labels.jsonl').read_text(encoding='utf-8').splitlines() if s]
        import numpy as np
        dev=[r for r in labels if r['partition']=='development'];rule=[]
        for r in dev:
            y=r['Y'][r['menu'].index(r['rule_action'])];best=min(r['Y']);reg=100*(y-best)/best
            rule.append(dict(case=r['case'],top1=float(y<=best+1e-7),near=float(reg<=.1+1e-10),near5=float(reg<=.5+1e-10),reg=reg))
        def sm(k):return np.mean([np.mean([r[k] for r in rule if r['case']==c]) for c in sorted({r['case'] for r in rule})])
        lines.append(f"| 同菜单规则 | {100*sm('top1'):.4f} | {100*sm('near'):.4f} | {100*sm('near5'):.4f} | {sm('reg'):.6f} | 0（参照） |")
        for name,r in [('线性集成',linear),('MLP集成（主模型）',neural)]:
            lo,hi=r['regret_improvement_CI95']
            lines.append(f"| {name} | {r['top1_percent']:.4f} | {r['near_0_1_percent']:.4f} | {r['near_0_5_percent']:.4f} | {r['mean_regret_percent']:.6f} | {r['regret_improvement_percent']:.6f} [{lo:.6f}, {hi:.6f}] |")
        lines+=['','## 预测可靠性与停止判断','',
            '| 方法 | 选中动作乐观误差>1个百分点% | 选中动作区间覆盖% | 整场景所有菜单同时覆盖% | 校准半径（百分点） |',
            '|---|---:|---:|---:|---:|']
        for name,r in [('线性',linear),('MLP',neural)]:
            lines.append(f"| {name} | {r['selected_optimism_gt_1pp_percent']:.4f} | {r['selected_interval_coverage_percent']:.4f} | {r['scene_simultaneous_coverage_percent']:.4f} | {r['calibration_radius_pp']:.6f} |")
        gate=read(OUT/'T1_GATE.json')
        lines+=['','校准用16个校准地图的全菜单/全前缀最大绝对误差，不用开发标签拟合。小集成标准差不是可靠置信区间；宽半径的覆盖不能单独解释成准确预测或安全保证。',
            '',f"T1主门槛：**{('通过' if gate['passed'] else '未通过，停止') if t1_complete else '阶段未合格完成，仅列暂存计算结果，不作正式通过/失败判断'}**。所有保存门槛逐项如下：",'']
        for key,value in gate['checks'].items():lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
        write(OUT/'paper_claims.json',dict(T0='saved teacher opportunity and failure diagnostics only',
            menu='deterministic constrained interface; structural unknown avoidance, not all-action extrapolation',
            teacher_identification_supported=bool(t1_complete and gate['passed']),
            route_compression_supported=bool(t2_complete and (OUT/'T2_RESULTS.json').exists() and read(OUT/'T2_RESULTS.json')['compression_passed']),
            independent_RL_gain_supported=False,T3_authorized=False,
            generalization='single fixed dataset screen with three optimizer seeds; no population-level confirmation'))
    else:
        lines+=['','本轮在完成T1前出现事前预算/时间/有效性停止；不得用部分端点替代完整门槛。详见failure.json和STATUS.json。']
    lines+=['','## T2 完整路线','']
    if t2_complete and (OUT/'T2_RESULTS.json').exists():
        t2=read(OUT/'T2_RESULTS.json')
        lines+=['冻结模型已按事前协议在36个新场景验证，完整数据见quality_time.csv、T2_RESULTS.json、T2_requests.jsonl和T2_teacher.jsonl。',
            f"机制门槛={t2['mechanism_passed']}；新场景教师压缩门槛={t2['compression_passed']}；冷/热部署门槛={t2['deployment_passed']}。后处理收益、原生搜索与模型收益分别归因，不宣称DRL增量。"]
    elif t2_started:
        lines+=['**已启动但未合格完成，无定论并停止。** 保留已消耗调用和全部部分结果，不把结果文件存在视为完整通过；不以部分场景替换36例主汇总。详见STATUS.json、failure.json和资源账。']
    else:
        lines+=['**未启动。** T1未获得事前要求的动作识别证据；未运行T2完整路径或配对教师，不提供虚构的质量—时间比较。T2场景/调用实际数量以resources.json为准。evaluate_routes.py仅保留经静态审查的条件执行入口。']
    ledger=resources['ledger'];stats=resources['stats']
    lines+=['','## 独立资源账与论文结论','',
        f"实际顶层调用{ledger['total_started']}/9048；完成{ledger['total_completed']}，失败{ledger['total_failed']}，精确缓存命中{ledger['cache_hits']}。分类账：`{ledger['used']}`。",
        f"标签使用{stats['label_uses']}次，优化器实际更新{stats['optimizer_updates']}步；研究运行墙钟{resources['elapsed_wall_s']:.3f}秒。T0审计、事前审查和归档验收时间另记，不混作规划耗时。",
        '细分模型批次/行、菜单与特征、导航准备/命中、原始补全、内部搜索及双重验证见resources.json与run/；终局尝试先记账，失败不退款，日志IO不冒充计算收益。',
        '', '本轮可引用的结论必须同时受T1门槛、T2执行状态和独立审查约束。任何负结果仅限定于本次特征、菜单、监督配置和预算；不反证教师蒸馏或神经规划一般不可行。没有证据支持强化学习独立增量，也不授权T3。','']
    (OUT/'REPORT_T012.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    print(json.dumps(dict(report='rc3_teacher_distill/REPORT_T012.md',status=status,top_calls=ledger['total_started'],updates=stats['optimizer_updates'])))


if __name__=='__main__':main()
