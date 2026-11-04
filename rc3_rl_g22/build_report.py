"""Write the bounded frozen-policy report from audited saved results."""
from __future__ import annotations
from .common import ROOT,read,write

NAMES={'fixed3':'原 Fixed3','detour':'原 Detour16','stop':'共享环境 STOP','detour_early':'固定早停 Detour','continued_ce':'冻结继续 CE','ce_gain':'冻结 CE＋收益'}

def main():
    summary=read(ROOT/'outputs/summary.json');methods=summary['methods'];rows=read(ROOT/'outputs/evaluation/results.json')
    lines=['# G2.2：冻结模型完整规划反馈','',
        '日期：2026-11-04。已实际执行12个旧extra开发场景×6方法，共72份完整规划。所有计划均通过当前二维代理模型的独立验证；没有回退、运行异常或5秒软截止超限。模型文件及内存张量保持冻结，没有训练更新或PPO。',
        '',
        '**结果：已有模型确实能从部分未查询动作中取得有效改进，但尚未形成超过强规则的整体证据。CE＋收益在5/12场景的共享回报高于立即STOP；继续CE为1/12。固定早停Detour的平均降本、平均共享回报及本次完整时间表现仍更好。默认部署继续使用冻结Fixed3／Detour。**','',
        '## 1. 实际评估范围','',
        '从原teacher_extra按case_id固定选择各模式前4个，最终case_id为0～11：regular、mixed、blocked各4个。每场景100张桌、800个服务候选，实际任务数79～250。它们是已用过的开发数据，不是新独立测试集。没有按标签或模型表现挑选场景。',
        '',
        '新协议位于 `outputs/protocol.json`。G2.1机会门槛和失败结论保留；本轮以单独的evaluation-only协议运行，未创建其PPO授权或把passed改真。用户补充库链接无法读取，本轮基于完整粘贴正文固定实施参数，没有声称读取补充包中的具体上限。',
        '',
        '共享方法保持原H=16、工作量软上限2,000,000及原奖励。原Detour最多16次修复。早停规则固定为最多4次提案、连续1次没有已实现净进展就停止；没有调参网格。每份规划5秒软截止，当前最长实测约0.897秒；这不构成硬实时保证。每种方法每场景只执行一次，按场景轮换方法次序，CPU单线程顺序测时。','',
        '## 2. 六方法实测','',
        '| 方法 | 相对Fixed3平均降本 | 完整规划中位时间 ms | 平均回报增量vs STOP | 胜／平／负vs STOP | 提案总数 |',
        '| --- | ---: | ---: | ---: | --- | ---: |']
    for name,item in methods.items():
        delta='—' if item['mean_return_vs_stop'] is None else f"{item['mean_return_vs_stop']:+.6f}"
        wtl='—' if item['return_vs_stop_wins'] is None else f"{item['return_vs_stop_wins']}／{item['return_vs_stop_ties']}／{item['return_vs_stop_losses']}"
        lines.append(f"| {NAMES[name]} | {item['mean_gain_percent']:.4f}% | {item['median_complete_ms']:.2f} | {delta} | {wtl} | {item['total_proposals']} |")
    lines.extend(['',
        '降本为先逐场景计算100×(J_Fixed3−J)/J_Fixed3，再取均值；J=L+0.75K是等效路程成本。完整L、K、J和每场景结果见 `outputs/evaluation/results.csv`，没有将其写成真实机器人总服务时间。',
        '',
        '共享回报为路线质量百分比改善减去λ_work×work/work_scale。原Fixed3／Detour采用原算法动作空间，没有可比的共享环境回报，因此留空。共享STOP会支付菜单生成费用；它的回报均值约−0.139374，不能将0当作其默认回报。',
        '',
        '原Detour在12/12场景的J都低于两个学习模型。固定早停Detour相对CE＋收益，在J上3胜、7平、2负；本轮12/12场景的完整时间更短，但并非每场景质量均更优。CE＋收益相对继续CE在J上3胜9平，仍不足以证明优于规则。小样本、每组合一次计时和旧开发集限制了结论，不能写成稳定速度优势或泛化优势。',
        '',
        '## 3. 完整计时口径','',
        '主表从读取已知场景JSON计时，包含新建可达性／导航、一次已验证Fixed3初解、菜单、图特征、推理、修复和最终验证；没有prepared输入缓存。模型评估前常驻CPU，证据压缩和写盘在计时外，逐步反馈查表及记录构建开销在计时内。共享STOP比原Fixed3慢，反映共享适配器、菜单和终验等额外开销，不能仅凭神经网络推理时间判断部署速度。',
        '',
        '模型加载单独记录于 `outputs/evaluation/model_setup.json`：继续CE约2.236秒，随后加载CE＋收益约0.018秒。两者网络结构相同，加载包含实例化，且受到加载次序、库首次初始化和文件缓存影响；这不是严格冷盘比较，也不能据此认定其中一个模型结构启动更快。`complete_plus_model_load_s`仅为按本次实际加载费用相加的敏感性列，主表不使用它。',
        '',
        '## 4. 未查询动作的实际反馈','',
        '查表严格使用完整环境观察。相同路线、相同动作编号但工作预算或历史不同，都不能借用旧标签。动作分为“原教师精确状态上已查询”“精确状态上动作未查询”“原教师没有该状态”。未取得的Q值继续保持null。',
        '',
        '| 模型／非STOP动作类别 | 实际提案数 | 立即改善最好J | 即时净回报>0 | 实际策略后缀>0 | 策略后缀优于同状态STOP解析记账 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |'])
    category={'action_queried':'已查询动作','action_unqueried':'动作未查询','state_unseen':'状态未见'}
    for model,groups in summary['model_feedback'].items():
        for name,group in groups.items():
            lines.append(f"| {NAMES[model]}／{category[name]} | {group['n']} | {group['immediate_best_improvements']} | {group['positive_immediate_rewards']} | {group['positive_actual_policy_suffixes']} | {group['positive_vs_same_state_analytic_stop']} |")
    lines.extend(['',
        '这些动作来自同一批24条模型轨迹，存在回合内相关性，不能当作独立新场景。后缀为从动作发生状态出发、沿当前冻结策略实际执行至结束的回报，含后续费用和终验修正；既不是最优Q，也不是原教师两步标签。',
        '',
        '同状态STOP列使用零势函数下已知待付菜单费用及已认证最好路线作解析记账，假定该前缀终验没有回退，没有实际分叉执行STOP。它与实际策略后缀相减可帮助理解净收益；不能把它包装成新增的反事实实验。根状态另有真正执行的同场景STOP参照。',
        '',
        '具体案例可以定位现在缺失的训练信号：',
        '',
        '- **case 8，CE＋收益根动作原未查询。** 最好路线改善约0.2205%，即时净回报约+0.064258；冻结策略终局回报为−0.090335，但实际STOP为−0.155394，因此仍净胜+0.065059。负的后缀本身不能判定搜索无价值。',
        '- **case 1，CE＋收益首动作没有立即改善J。** 即时回报−0.098607，后续第二次提案才改进；整条冻结轨迹回报+0.001131，相对实际STOP增加+0.094047。一次动作没有立刻改善，不能直接标成长期负收益。',
        '- **case 0，继续CE取得改善后又付出额外费用。** 首动作已有约0.4548%的质量改善和+0.291123即时回报，但最后相对STOP为−0.038816。保存轨迹中可观察继续过久的迹象；事后选择最好停止前缀仍只是诊断，不是第七种已实现策略。',
        '',
        '因此，“未知全坏”和“更积极搜索就更好”都不符合这些记录。反馈支持同时改进继续动作的净价值排序及终止选择，而不是只调STOP阈值。',
        '',
        '## 5. 本轮后决定','',
        '本轮不再增加离线更新、PPO、场景或额外分支。继续CE保留为匹配对照；CE＋收益保留为反馈更积极的研究变体，不升级为部署默认。后续若另立实验，应先统一动作反馈的目标范围和监督深度，再对独立净收益头或联合排序／停止结构做相同数据、初始化和预算的对照，避免把actor分类置信度直接当数值收益。',
        '',
        '本轮28次模型提案反馈是偏向当前策略已选动作的样本，尚未覆盖完整17动作菜单；不能直接补满未知标签，不能将所有后缀回报混入原深度2教师Q。若进行策略价值监督，必须明确它估计的是哪个冻结策略及哪个STOP参照。单独冻结继续排序只训练停止头的局限仍保留；当前结果没有推翻此前上限诊断。',
        '',
        '## 6. 可复核记录','',
        '实际执行账目为72份规划：48条共享环境回合、95个共享动作、47次共享提案；原Detour另有192次修复，共239次两种协议下的提案／修复调用。它们的动作范围和工作定义不同，不能用调用数直接作等工作量排名。没有新教师树分支，训练更新0、PPO动作0。',
        '',
        '全部72份完整底盘折线、停靠及左右臂分配存于 `outputs/evaluation/case_*/plan.json`；逐步证据和模型输入存于对应 `evidence.json.gz`。`outputs/plan_audit.json`重算计划、成本、奖励、工作和精确教师范围；概率归一化按FP32误差界检查。`outputs/inference_replay.json`对实际52次模型输入做了零更新重放，所有动作相同、logit和概率最大差均为0。重放不产生新规划或奖励反馈。',
        '',
        '原版本前后完整性见 `outputs/freeze_before.json`、`freeze_after.json`；最终打包回执给出清单与依赖。二维代理模型内验证、真实双臂协作执行和学习有效性仍是不同结论；真实标定与IK继续暂缓。',
        '',
        '本轮可进入论文的方法诊断／开发反馈部分。它没有产生新的强化学习增量，不能写成DRL已优于强规划基线。独立结论复审见 `EXPERIMENT_RESULT_TO_CLAIM.md`；图表及生成脚本在 `figures/`。'])
    (ROOT/'REPORT_G22.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Also retain the main paper table as a numeric JSON artifact.
    write(ROOT/'outputs/report_table.json',{NAMES[name]:item for name,item in methods.items()})

if __name__=='__main__':main()
