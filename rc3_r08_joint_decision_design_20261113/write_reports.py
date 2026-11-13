"""Render the saved-arithmetic evidence into the requested research materials."""
import csv,json
from .adapters import HOME

def main():
    v=json.loads((HOME/'ANALYSIS_SUMMARY.json').read_text(encoding='utf-8'));ss=v['summaries']
    rows=list(csv.DictReader((HOME/'ROUTE_GAP_RESULTS.csv').open(encoding='utf-8')))
    def select(cohort,source='FrozenSL',target='StrongTeacher',phase='H',size='ALL'):
        return next(x for x in ss if (x['cohort'],x['source_method'],x['target_method'],x['phase'],x['size'])==(cohort,source,target,phase,size))
    lines=['# 已保存完整路线的严格配对结构分析\n',
        '本轮无新策略前向、训练、规划、导航、native、教师或H；只按既有问题/父/端点/任务身份读取和重算保存路线。完整来源/字节hash见ANALYSIS_INPUTS.json；归档仅按已保存offset读取750个成员，不扫描旧32768轨迹或历史百万表。\n',
        '## 样本、统计单位与口径\n',
        '- 主体：96条完整10×10教师+256条局部教师，各与同问题准确FrozenSL参考配对；352问题来自96物理父，局部4×3/6×6/8×8为86/84/86。\n- D5只用固定4096 PG/SL终点及同期FrozenSL；D6只用固定256 A/B/SL及同期FrozenSL，每实验66暴露开发问题、11父，不跨实验合并均值。没有这两个终点对完整10×10的固定评价，所以不推测其父任务质量。\n- R07原144状态的保存baseline/alternative已逐对分析，3状态先父内平均，仍为48父；原48任务教师差距另列，不把状态当父。\n- 所有结构均值先父内再父等权；`relative_gain`为每任务相对差先父内均值，`gain_of_parent_mean_J`为旧门槛“每父两方法J先平均再相对差”，二者同时保存；不能互换。\n- delta=目标−来源，负值成本改善；gain=(来源−目标)/来源，正值改善。原生teacher raw是已完成其内部搜索的native-final，不能当作神经no-H。\n',
        '## 教师与FrozenSL：H后的实际差距\n',
        '|训练规模|问题/父|父等权任务相对收益|ΔJ|ΔL|ΔK|集合Jaccard|共同顺序逆序比例|任务配置归属变化|臂变化|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for size in ['10x10','4x3','6x6','8x8','ALL']:
        s=select('training352',size=size)
        lines.append(f"|{size}|{s['tasks']}/{s['parents']}|{100*s['relative_gain']:.4f}%|{s['delta_J']:.4f}|{s['delta_L']:.4f}|{s['delta_K']:.4f}|{s['set_jaccard']:.4f}|{100*s['common_order_disagreement']:.2f}%|{100*s['assignment_config_changed_fraction']:.2f}%|{100*s['assignment_arm_changed_fraction']:.2f}%|")
    t=[r for r in rows if r['cohort']=='training352' and r['phase']=='H']
    same=sum(r['same_set']=='True' for r in t);sm=[r for r in t if r['same_set']=='True' and r['same_order']=='False']
    lines += [f"\n教师H成本303/352更低、13持平、36更高，不是全局最优证书。集合相同仅{same}/352；其中不同顺序{len(sm)}对：L124 ΔL=−5.0033、L137 ΔL=0、L195 ΔL=−0.1115，均4×3且ΔK=0。不同顺序可以同成本，不能把逆序数当J。完整96对没有同集合配对，因而不能仅用它们识别固定集合排序效果。\n",
        '教师K更少217对、相同99对、更多36对；相同K中92对集合仍不同。教师L更短286对。这说明K标量相同不等于配置选择相同，K更少也不必然J更低。每个配置负责的任务/手臂、独占任务、共有/独有配置与共同配置顺序均在ROUTE_STRUCTURE_EVIDENCE.json，以原记录为事实来源。\n',
        f"全部352问题的父等权任务相对差为{100*select('training352')['relative_gain']:.6f}%，旧父均J配对口径为{100*select('training352')['gain_of_parent_mean_J']:.6f}%。R07固定48子集重现{100*select('R07_fixed48')['relative_gain']:.6f}%，保留原3.81%说法及样本范围，不用新范围改写旧结果。\n",
        '## raw到H：哪些变化已被规则完成\n',
        '|范围/方法|父等权raw K→H K|ΔJ|ΔL|ΔK|raw/H集合相同率|共同顺序逆序比例|无独占任务配置 raw→H均值|\n|---|---:|---:|---:|---:|---:|---:|']
    for method in ['FrozenSL','StrongTeacher']:
        for size in ['10x10','4x3','6x6','8x8','ALL']:
            s=select('training352_H_effect',method,method,'raw_to_H',size)
            lines.append(f"|{size}/{method}|{s['source_K']:.3f}→{s['target_K']:.3f}|{s['delta_J']:.4f}|{s['delta_L']:.4f}|{s['delta_K']:.4f}|{100*s['same_set']:.2f}%|{100*s['common_order_disagreement']:.2f}%|{s['source_noncritical_configs']:.4f}→{s['target_noncritical_configs']:.4f}|")
    lines += ['\n“无独占任务配置”指在已选集合中没有仅由它覆盖的目标任务；“冗余覆盖边”是Σ_t(覆盖次数−1)，不是重复执行次数。两者均从逐臂合法有效覆盖表观察，未实际删除、重排或再求解。H后FrozenSL每问题父等权只有0.03125个无独占任务配置，教师为0；残余差距不能简单解释成还留了大量可直接删的重复点。H还同时替换配置与改变顺序/归属，不能只把它当排序器。\n',
        '强教师的raw已包含Detour16内部优化，统一H额外效果很小；不能拿教师raw对神经raw的巨大差距冒充网络结构因果效果。\n',
        '## D5/D6固定主终点：确有结构变化，但没有任务增量\n',
        '|固定实验/目标对照|旧门槛H-J父配对收益|ΔL|ΔK|集合Jaccard|任务配置归属变化|\n|---|---:|---:|---:|---:|---:|']
    for cohort,source,target in [('D5_fixed4096','FrozenSL','D5-PG'),('D5_fixed4096','D5-SL-U','D5-PG'),('D6_fixed256','FrozenSL','D6-PG'),('D6_fixed256','D6-SL','D6-PG'),('D6_fixed256','D5-PG-A','D6-PG')]:
        s=select(cohort,source,target)
        lines.append(f"|{cohort}: {target} 对 {source}|{100*s['gain_of_parent_mean_J']:.6f}%|{s['delta_L']:.4f}|{s['delta_K']:.4f}|{s['set_jaccard']:.4f}|{100*s['assignment_config_changed_fraction']:.2f}%|")
    lines+=['\nD5/D6的停靠集合、相对顺序及任务归属确实变化。D6对FrozenSL虽少0.985配置，导航却多2.349m；不是“不能改动作”，也不能仅优化K或动作翻转数。固定32拟合率从未进入本表。完整分尺寸、raw/H和父值均在CSV；已有完整时间见原D5/D6报告，不能跨时期缓存条件混合为新计时优势。\n',
        '## R07：同可达前缀干预的保存证据\n']
    r=select('R07_counterfactual144','R07_baseline','R07_alternative')
    lines += [f"144状态/48父：H后{r['target_better']}改善、{r['tied']}持平、{r['target_worse']}退化；父等权有符号收益{100*r['relative_gain']:.6f}%，保留OPPORTUNITY_INSUFFICIENT。3550行总表中以独立cohort保存这组对照，不与352教师配对或D5/D6均值混合。\n",
        '原R07 H_EFFECTS.json记录raw有成本差121状态，其中59经H后持平；55/144最终H路线完全相同。H可改写构造前缀，所以这是离线完整路线反事实，不能当作机器人已执行前缀后的后缀Q。实际双rollout均值0.3496s、alternative0.1735s，仅其常驻/已恢复输入合同；本轮未重测或用它证明新动作速度。\n',
        '## 能观察、能推导与尚不能确定\n',
        '**已观察**：大多数H后教师/SL配对选择不同配置、重新分配了相当部分任务，同时共同配置顺序不同；导航项与配置数量项都变化。H已消去大量raw冗余并重新安排完整方案。两种训练终点可以明显改路线，但质量未过。\n',
        '**代码/数学推导**：当前任务责任与手臂选择由最早合法服务/L优先固定，单独“学分配”在不改转移时没有自由度；联合配置修改可以同时改变多个责任及连接，原单配置干预不保证触及它。K是配置次数，臂变化本身没有独立成本项。\n',
        '**需要新受控求解才能归因**：同集合固定顺序求最优、固定排序器比较集合、锁定H前后算子等。两方案集合不同的ΔL不能全归于排序；0.75ΔK也不等于“优化集合”的独立因果收益。长边/入口出口腿长度及共同边变化仅是观察，未求最短顺序或反事实入口。\n',
        '**当前不能回答**：新联合菜单是否有充分机会、25729参数头是否可学、是否优于Detour、未见父布局泛化、冷/热部署优势与真实IK/执行。均留给严格条件阶段，不补新求解以凑本轮结论。\n',
        f"数据完整性：{v['total_comparison_rows']}比较行、{v['stored_route_phases']}份已保存raw/H阶段；最大ΔJ−ΔL−0.75ΔK误差{v['maximum_component_identity_error']:.3g}。这是针对新增分析的核算，不重做历史百万项审计。固定训练池/暴露开发选择与每尺度不均衡限制保留，不作独立泛化统计。\n",
        '结论：残余差距主要表现为**不同覆盖配置组合与其访问连接/顺序共同变化**，不是遗漏覆盖或只多了若干可直接删点。现有记录不足以把集合、排序、任务分配拆成可相加的独立收益，也不足以证明某种新复杂网络必要。\n']
    (HOME/'ROUTE_GAP_ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    brief='''# 给导师：研究内容三下一版决策材料

**建议先审议一次有界“配置联合编辑机会”验证，暂不批准正式DRL训练。**固定目标仍是在复杂大规模父场景训练DRL，冻结后快速完成不同尺度局部完整约束规划；代理模型尚无真实IK/实机认证。

已经完成的基础是真实SL/全Actor与五张量DRL训练、96完整+256局部教师、完整环境/mask/同臂F&A/导航端点、H和独立验证、准确恢复与资源审计。S2/GL-v4/v5曾做SL初始化后的全Actor训练，不能再把“主Actor可训练”单独称为新机制。

D5固定4096对FrozenSL/匹配SL的H成本收益为−2.64%/−3.74%；D6固定256虽然比同轮旧PG少退化0.80%，仍比FrozenSL/匹配SL差1.23%/1.58%。两者路线实际改变，但没有建立完整任务增量。固定32拟合支线已封存，与任务质量不混用。

R07证明同前缀动作差异存在、可复现，但实质改善23/144、局部5/48，未过机会门槛；55对最终路线相同。单点Top2加同SL后续未充分利用整体差距。H会重写前缀，所以不能把其成本当已执行前缀的在线Q。

本轮实际重用全部352教师—FrozenSL配对/96物理父。H后教师有303条更好、13持平、36更差；配置集合只有7/352相同。父等权任务差距3.85%（R07原48子集仍3.81%）；约46.7%的任务配置归属改变，导航与停靠数量项均变化。H后的SL已很少保留可直接删的冗余配置。因此有依据把下一版决策对象从不可撤销下一点，改为相关配置组合及关联顺序；**没有证据把3.81%归给单独集合或排序因素**。

推荐唯一候选：上层Actor在完整方案上选择同时删换最多两个配置的动作（可含非相邻配置），下层按固定已有D有界插入/排序，终端同H并完整验证。先复用准确SL的64维表示与初解，只新建一个小菜单头；本轮未实现大型网络或训练器。它要学的是有限求解预算下先改哪些相关覆盖配置、怎样安排连续修改以及何时停止，而不是学习导航/合法性，也不是重做旧Detour连续窗口评分。

实施顺序：A已完成代码/路线分析、联合数学模型与12项接口单测；B先检验固定联合动作的真实完整机会；C再验真实转移/梯度/恢复/冻结推理；D只做单seed128更新及匹配监督/同菜单固定规则/单配置动作族对照；E必须有任务信号才做另外两seed、新父确认和一次最终评价及部署计时。若B无机会或D无收益就结束，不再加轮数/候选/网络/基线求正。

需要导师决定的是：是否采用这一有界联合决策方向并授权下一步B；不是宣告新DRL已经有效。本轮没有新规划、H、教师、场景、导航或Actor/Critic更新，B—E只是计划。完整预算与条件见R08_MASTER_PLAN.md；即使将来相对SL增量通过，是否胜过Fixed3/Detour仍须看完整独立对照，不能先写强规则优势。
'''
    (HOME/'TEACHER_DECISION_BRIEF.md').write_text(brief,encoding='utf-8')
    print(dict(status='REPORTS_WRITTEN',comparison_rows=v['total_comparison_rows']),flush=True)

if __name__=='__main__':main()
