"""Generate evidence-linked report and primary-gate plot; no environment use."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .core import ROOT,read,write,sha

def main():
    a=read(ROOT/'outputs/analysis.json');p=read(ROOT/'outputs/protocol.json');rows=read(ROOT/'outputs/test_results.json')
    names={'same_interface_rule':'同接口规则','one_step_repeated':'重复一步价值','frozen_init':'冻结初始化','matched_static_rule_value':'等更新静态规则价值','multi_policy_iteration':'多步策略改进','original_fixed3':'原Fixed3','original_detour16':'原Detour16'}
    grouped=defaultdict(list)
    for row in rows:
        key=row['method'].split('_',1)[1] if row['method'][0].isdigit() else row['method'];grouped[key].append(row)
    table={}
    for key,own in grouped.items():
        gains=[]
        for r in own:
            base=next(v['J'] for v in rows if v['case_id']==r['case_id'] and v['rep']==r['rep'] and v['method']=='original_fixed3')
            gains.append(100*(base-r['J'])/base)
        table[key]=dict(mean_gain=float(np.mean(gains)),median_ms=float(np.median([r['complete_s'] for r in own])*1000),
            overrun=sum(r['overrun'] for r in own),n=len(own))
    lines=['# G23 封顶试验结果（2026-11-04）','',
        '**本次未达到预登记主目标。按任务书停止把“Fixed3上的局部窗口选择＋STOP”继续作为第三部分DRL核心，不追加同方向训练、损失变体或跨规模测试。** 这是针对本次小数据、有限更新和仪表化实现的研究投入决定，不证明所有局部搜索强化学习都无效。默认部署继续使用冻结Fixed3／Detour。','',
        '## 已实际完成','',
        '- 规则可表达性：12个旧开发场景，原规则192次修复；新接口在资源允许的104步前缀上，语义窗口、操作前后路线逐项一致。不声称在共享W限制下完整复现全部16步。',
        '- 三训练种子2301／2302／2303，各4个不同的全新百桌训练场景。两轮真实策略评价与改进，累计3840次更新、132条实际查询的训练标签；不是只回归一次静态旧教师表。',
        '- 每次评价策略完整冻结。第一轮评价规则piD，得到pi1；第二轮实际按pi1到达状态并采集其后缀。匹配静态控制在相同首转移上继续规则piD；同一MC0初始化、同一追加256更新，只改变价值评价后缀策略。一步对照目标包含实际下一STOP费用，可以反复决策。未知动作未补零。',
        '- 六个独立新百桌测试场景，三模式各2个。十二份种子策略＋同接口规则／原Fixed3／原Detour共15个方法实例，每场景2次成对测时，180份输出全部通过保存方案的独立几何、覆盖、分臂、路程及成本复验。180不是180个独立场景。',
        '- 最终测试前登记了标签实际复核：再执行原训练分支，132条一步、多步、静态后缀及STOP基线全部复现；保留2067条后缀转移及终验账目。额外1497次修复已纳入总预算，不包装成零查询检查。','',
        '## 全局资源账','',
        '| 项目 | 实际 | 上限 |','| --- | ---: | ---: |']
    for key,n in a['resource_counts'].items():lines.append(f"| {key} | {n} | {p['limits'][key]} |")
    lines.extend(['',
        '非STOP共3694次＝规则对齐296＋训练1497＋最终测试404＋标签复核1497。252次冷问题／锚点构建包含审计、训练父状态、180次完整测试及复核父状态；952次终验也包含STOP和分支终点，不能计成952个新场景。8192是上限而非必须花满的配额；预登记的两轮和3840更新已经完成，不能因剩余额度临时加训。无PPO。','',
        '## 最终质量与完整时间','',
        '| 方法类 | 相对Fixed3平均降本 | 完整时间中位 ms | 超0.8s软预算／测量数 |',
        '| --- | ---: | ---: | ---: |'])
    for key in ['original_fixed3','original_detour16','same_interface_rule','frozen_init','matched_static_rule_value','one_step_repeated','multi_policy_iteration']:
        t=table[key];lines.append(f"| {names[key]} | {t['mean_gain']:.4f}% | {t['median_ms']:.2f} | {t['overrun']}/{t['n']} |")
    lines.extend(['',
        '学习类合并3种子的重复测量只是描述性汇总；各种子结果单独保存在analysis.json。多步模型三个种子的平均降本分别约0%、0.5521%、0%，两种子最终没有发起测试修复，显示明显不稳定。不能只保留2302。','',
        '主比较先对同场景的种子和重复成对求平均，再以6个场景做bootstrap95%区间。门槛是平均改善≥0.3%且区间下界>0，同时满足资源／验证要求；不能用R_W或速度替代失败的J目标。','',
        '| 多步模型相对 | 平均J改善 % | 场景配对95%区间 % | 通过 |','| --- | ---: | --- | --- |'])
    for key,item in a['comparisons'].items():
        lo,hi=item['ci95'];lines.append(f"| {names[key]} | {item['mean_gain_percent']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | 否 |")
    lines.extend(['',
        '相对初始化和等更新静态控制存在本轮有限的正增量，应当保留；但均未达0.3%，相对一步的区间跨零，相对同接口规则和原Detour平均更差。即使暂不考虑超时，质量门槛也没有通过。','',
        '105/180次超过0.8秒软预算；多步模型自身14/36次超限。当前修复及必要终验不能被硬中止，因此全部保留并按预登记规则判资源验收失败，不声称硬实时。完整时间从新场景生成／初始化开始，包含准备、锚点、菜单、图和数值头前向、修复及终验；模型常驻CPU，计数I/O也在计时内。原部署方法与同接口方法的额外扫描／认证不同，不能把延迟差全部归因神经网络。','',
        '## 语义及实现边界','',
        '本轮价值头输出Q^pi(s,a)−qSTOP(s)，STOP相对值固定0，选择正值最大的动作，否则停。qSTOP来自同状态实际STOP分支，保留待付菜单费用和终验修正；一步不是r本身。原两步教师、G22策略后缀、当前pi_k目标没有混用。左右臂可达边分开编码，预算、失败／尝试统计和规则标记参与评分；网络只使用历史的统计、掩码与局部图表示，不等于无损编码完整历史。','',
        'Dense修复复制原数值运算，仅插入计数；规则候选放在首位，必要时替换普通候选，仍最多16非STOP＋STOP。同接口成功复现保证可表达性，不保证模型一定选择规则或优于规则。现有可靠成本保底仍来自验证后的Fixed3锚点。','',
        'W是本轮明确的工作代理：额外规则扫描代理和Dense DP条目计数已加入，因此不能把它与旧G22相同数值W当成相同工作，也不能当墙钟时间。未做新缓存优化，更没有把工程收益记成RL收益。','',
        '132条训练标签中，2302第二轮仅有4个根状态的12行，因为其pi1很快STOP；不能声称每种子都采到8个状态或都进入新的后继区域。仅每种子4个训练场景、2轮更新、6个独立测试场景；这不是充分收敛或投稿级普遍性证明。','',
        '最终测试保存完整输出路径和每次调用资源账，但未保留所有中间神经图输入，不能声称测试逐决策前向被独立零误差重放。训练分支则有额外实际复核和完整后缀账。模型快照及源代码／协议哈希提供进一步追溯入口。','',
        '机器信息未写入事前protocol，是本轮预登记缺项。运行后补记：AMD Ryzen 7 7745HX、16逻辑处理器，CPU单线程，Python3.10.20／torch2.12.1+cu126／numpy2.2.6／scipy1.15.2；runtime_metadata.json明确标记posthoc，不倒改协议冒称已预登记型号。','',
        '## 后续决定与交付','',
        '按本次止损条件，关闭当前局部窗口＋停止的DRL核心路线。可保留其可靠规划平台和本次负面／次要正增量作为诊断证据；如果仍以DRL为论文核心，下一项应独立立项服务停靠点、任务分组与访问顺序的联合构造，并重新建立基线和资源协议。本轮不自动开展该新任务。','',
        '旧G2/G21/G22源码、权重、结果及归档已核对保持一致。标定与真实IK继续暂缓；全部有效性均限于二维静态代理模型。','',
        '复核入口：outputs/alignment.json、analysis.json、plan_validation.json、resource.jsonl、label_verification.json、verified_labels.json.gz；模型与全部180份JSON路径在outputs。独立结论见EXPERIMENT_RESULT_TO_CLAIM.md及paper/review-traces。','',
        '本轮读取了[Learning-to-Delegate官方仓库](https://github.com/mit-wu-lab/learning-to-delegate)、[NeuOpt官方仓库](https://github.com/yining043/NeuOpt)和[Bertsekas近似策略迭代综述](https://web.mit.edu/dimitrib/www/Bertsekas_ApproximatePolicyIteration_2011.pdf)，借鉴子问题委派、多步改进和固定策略评价／策略改进的结构。本轮未下载并复现这两个项目的完整训练；它们不证明本项目有效，也不提供近似神经策略必然改进的保证。'])
    (ROOT/'REPORT_G23.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');write(ROOT/'outputs/class_summary.json',table)
    # One reproducible primary-decision figure, data read from analysis.json.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=ROOT/'figures';directory.mkdir(exist_ok=True)
    labels=['Shared rule','Repeated one-step','Frozen init','Matched static value','Original Detour16']
    fig,ax=plt.subplots(figsize=(7.2,3.5),layout='constrained')
    for i,item in enumerate(a['comparisons'].values()):
        m=item['mean_gain_percent'];lo,hi=item['ci95'];ax.errorbar(m,i,xerr=[[m-lo],[hi-m]],fmt='o',color='#0072B2',capsize=3)
    ax.axvline(0,color='#777777',linewidth=.8);ax.axvline(.3,color='#D55E00',linestyle='--',label='Minimum mean gain: 0.3%')
    ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.set_xlabel('Multi-step relative cost improvement (%)')
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper left',fontsize=8,frameon=False)
    for ext in ('pdf','png'):fig.savefig(directory/f'primary_gate.{ext}',dpi=300)
    plt.close(fig)
    (directory/'CAPTION.md').write_text('六个未见场景；三种子、两重复先在场景内成对平均，横线为场景cluster bootstrap95%区间。正值表示多步较对应参照成本更低，虚线0.3%为预登记最小实用平均改善。所有对照均未通过质量门槛；105/180软时限超限另计。六场景的小样本不支持广泛无效性证明。图由python -B -m rc3_rl_g23.report从保存数据再生成，无新修复。\n',encoding='utf-8')

if __name__=='__main__':main()
