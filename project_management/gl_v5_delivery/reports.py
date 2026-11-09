"""Actual GL-v5 evidence reports and narrowly scoped project status updates."""
import json
from pathlib import Path
from rc3_gl_v5_20261109.common import ROOT,HOME,read,write_new,digest,label,now

VERSION='2026.11.09-gl-v5-fixed-greedy-reference'


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def textfile(name,txt):
    p=HOME/name
    with p.open('x',encoding='utf8') as f:f.write(txt)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(v):return f'{v*100:.4f}%'


def reports():
    s=read(HOME/'results/SUMMARY.json');r=read(HOME/'RESOURCE_ACCOUNTING.json');a=read(HOME/'INDEPENDENT_AUDIT.json')
    require_pass=a['status']=='PASS'
    if not require_pass:raise ValueError('Independent numerical audit not passed')
    verdict=read(HOME/'CLAIM_REVIEW_VERDICT.json');decision='GO_SINGLE_SEED_CANDIDATE' if s['candidate_screen'] else 'REVISE'
    means=[[x['value'],x['study_method'],x['n'],*[f"{x[k]:.4f}" for k in ('raw_J','J','raw_L','L','raw_K','K','H_gain_J')]] for x in s['method_summary']]
    comparisons=[[x['metric'],x['control'],pct(x['mean']),f"[{pct(x['lower95'])}, {pct(x['upper95'])}]",x['positive']] for x in s['comparisons']]
    grad=[[x['size'],x['n'],f"{x['mean']:.5f}",f"{x['median']:.5f}",f"{x['p95']:.5f}",f"{x['max']:.5f}"] for x in s['gradient_by_size']]
    training=[[x['size'],x['n'],f"{x['mean_sample_J']:.4f}",f"{x['mean_reference_J']:.4f}",f"{x['mean_advantage']:.4f}",f"{x['std_advantage']:.4f}",x['negative_advantages'],f"{x['mean_H_gain_J']:.4f}"] for x in s['training_by_size']]
    adv=s['advantage'];done=read(HOME/'runs/SCIENCE/G128/COMPLETE.json');idx=read(HOME/'runs/SCIENCE/DEV432/ACTOR_INDEX.json')['models']['11622026']
    lineage={role:dict(study_method=role,checkpoint_role=role,actor_file=e['path'],actor_sha256=e['checkpoint_sha256'],tensor_sha256=e['model_tensor_sha256'],
        training_source=e['provenance'],fixed_update=256 if role=='SL256' else 128) for role,e in idx.items()}
    write_new(HOME/'WEIGHT_LINEAGE.json',dict(seed=11622026,models=lineage,common_start=done['start'],
        new_G_source_binding=digest(HOME/'SOURCE_BINDING.json'),old_B_C_sources_unmodified=True,old108_role_fields_unmodified=True))
    textfile('EXPERIMENT_REPORT.md',f'''# GL-v5固定贪心参考基线：实际小试

固定128终点完成，**{decision}**。仅新增G128，seed11622026，从精确SL256 fresh Adam；未重训A/B/C、重跑部署网格或填补旧S3缺失。采样1536、冻结参考512、终点评价432，共2480顶层请求，128优化器更新。共享墙钟{r['phase']['wall_s']:.6f}/7200秒，含注册准备{r['phase']['registration_carry_s']:.6f}秒、合成检查、输入准备、参考、训练、H、验证、检查点和IO。

## 数学和实施

三条自然起点轨迹保留原权重(1,.25,.25)/1.5、全首动作logprob、batch4及原512问题日程/随机流、50%full和50%local、网络/奖励/H、lr1e-4、clip1。唯一主要变量是用冻结SL256同输入贪心+H成本替代LOO。每实例损失为 `Σ_r w_r stopgrad(J_sample_r−J_SL_greedy_H) logP(full_trajectory_r)/1.5`，再对4实例平均。优势为`J_ref−J_sample`，负优势全部保留，不裁断、不筛获益样本。

冻结参考是输入的确定函数，不依赖当前采样动作，参考无梯度、无晋升、无策略RNG消费，模型/前缀/终局缓存与G分离。由`E[∇logP]=0`，替换该基线保持期望成本目标；有限采样、clip和Adam仍无贪心质量单调改进保证。合成枚举、真实训练函数的合成批梯度、baseline detach、首似然、RNG/cache及错误角色负向检查PASS，工程optimizer0，未使用新科学探针。

## 训练成本、优势与梯度

{table(['训练尺寸','采样数','采样H后J','冻结参考H后J','平均优势','优势标准差','负优势数','采样raw→H减少J'],training)}

优势全局：均值{adv['mean']:.6f}，标准差{adv['std']:.6f}，范围[{adv['min']:.6f},{adv['max']:.6f}]，正/负/零={adv['positive']}/{adv['negative']}/{adv['zero']}。这些是训练分布和采样诊断，不替代冻结贪心完整路线评价。

{table(['尺寸','实例数','batch缩放后梯度L2均值','中位','p95','最大'],grad)}

上述为各实例对实际4实例平均梯度的增量范数，未经额外反向；全批裁剪前/后范数与完整张量另存。裁剪更新{s['training']['clipped_updates']}/128，零梯度更新{s['training']['zero_gradient_updates']}，非有限值0。平均实际Adam参数步长L2={s['training']['Adam_step_l2_mean']:.8f}；每层变化、Adam内部step1—128、权重/Adam/RNG均保留。

G实际128次update内部耗时{s['training']['G_update_inclusive_s']:.6f}秒，既有B记录为{s['training']['old_B_update_inclusive_s']:.6f}秒；它们不是配对计时实验，不把两者差值视作精确部署/算法加速。明确增加的训练计算是512次D1冻结贪心参考请求，相对B1536采样终局请求增加33.3333%，另有相应参考模型读取、fresh编码、逐步打分和H/导出/验证；其真实调用及缓存命中见[资源账](RESOURCE_ACCOUNTING.json)。没有新增B/C训练或教师查询，推理G仍只需一次贪心+H。

参考512次请求全部收费，实际执行512次fresh贪心，486次完整H、26次精确终局缓存命中；G采样1536次均执行完整H、命中0。参考各不重叠计算子时钟合计82.666239秒（载入、fresh编码、构造、H/验证），原始结果写盘/Journal等费用另包含在外层总账，不能把这一子合计当作全部阶段耗时。科学阶段新增失败0；只读汇总脚本曾漏导入一个常量，故障与修复已在维护账记录，仅重读保存数据，未重跑科学请求。

## 72例固定终点评价

从B实际512训练日程出现的96物理布局，按map_key排序各选mixed/blocked6个；每布局一个新task draw、三尺寸两端点，优化前封存72例。新任务种子为61109050000＋布局序号，A_condition=序号模3，每模式内三A条件各两布局；所有方法完全匹配。输入可行432/432，完整任务覆盖与同臂F&A/导航/端点代理验证432/432。它们是已训练布局上的新任务，不称未见环境。没有使用旧S3或原60。

所有神经方法用相同D1轻量包，每请求fresh贪心+相同H；Fixed3/Detour16保持原生流程，无额外共同H。保存原始与H后成本；规则raw是内部搜索后的原生终局，不能与神经纯greedy的计算层次混同。仅导航共享同源保存D，第一次导航读取的实际时间照实记录，不据本网格宣称配对部署速度收益。

{table(['分组','模型','n','raw J','交付/H后J','raw L','L','raw K','K','H减少J'],means)}

## 事前统计与判定

每真实物理布局先平均全部6个尺寸/端点，再计算`(control−G)/control`；12布局等权，mixed6/blocked6分层父布局bootstrap20000次，PCG64种子11642026，线性分位数2.5/97.5。正值表示G成本较低。单优化种子，区间不提供跨种子或familywise确认；窗口不是独立环境重复。

{table(['层次','G对照','父布局平均收益','95%区间','平均及下界均>0'],comparisons)}

门槛在首次调用前固定：G的H后J相对共同SL、既有B、既有C，**三者各自平均改善严格>0且95%下界严格>0，并且全部432可行、有限、来源正确**。AND筛选实际为{decision}，raw独立报告，无Top1或额外事后性能门槛。H收益不记作学习收益，SL/规则仅作为辅助对照，固定DRL研究目标保持。

模型来源见[WEIGHT_LINEAGE](WEIGHT_LINEAGE.json)，2048份训练请求的明确调用者身份见[TRAINING_REQUEST_IDENTITIES](TRAINING_REQUEST_IDENTITIES.json)：G采样绑定实际优化前参数版本及完整张量artifact，训练中尚未导出actor的字段明确为null，不冒充固定终点；参考绑定冻结SL256 actor。432终点评价每行都含真实角色及actor/tensor/source/fixed_update。全部逐路线/逐更新见[CSV](TRAINING_AND_ROUTE_RESULTS.csv)，[采样—参考差和优势](SAMPLE_REFERENCE_COMPARISON.csv)、[训练曲线](TRAINING_CURVES.csv)、[独立审查](INDEPENDENT_REVIEW.md)。
''')
    textfile('INDEPENDENT_REVIEW.md',f'''# GL-v5独立审查

保存数组/张量的独立实现数值审计：**PASS**。另由独立paper architect/reviewer角色按experiment-result-to-claim技能评议，claim_supported=`{verdict['claim_supported']}`，confidence=`{verdict['confidence']}`；这是独立Codex角色审查，不冒充外部人类同行评审。

角色审查出具时全量机械审计仍在运行，原始轨迹据实保留该时点；本文的数值PASS引用之后完成的INDEPENDENT_AUDIT.json，不回写或冒充当时已通过。confidence限定本轮单种子、已训练布局开发筛选；缺少论文全稿claim audit，主张记录保持provisional，不进入论文确认写作。

审计核对{a['bound_sources']}源码身份、{a['old_frozen_files_unchanged']}旧冻结文件原字节、512个G/B同日程同view、1536采样请求、512冻结参考、432终点请求。128更新梯度clip、Adam实际参数步长、fresh Adam和固定SL来源一致；最大裁剪数组差{a['max_clipping_error']:.3g}，Adam步长差{a['max_Adam_step_error']:.3g}，父布局统计差{a['max_statistics_error']:.3g}。同臂服务检查{a['same_arm_assignments_checked']}条、实际greedy mask复核{a['greedy_masks_recomputed']}次，4个D1包与各完整checkpoint逐张量一致。

只读审计首次误把Python float64求和与训练原有float32顺序累加按1e-6比较，属于审计器口径错误。修正为独立精确重构float32原算术后，1536条完整似然和逐位一致，没有放宽数值等价阈值或修改训练。故障、原审计脚本SHA和处理记录见[审计器故障记录](READ_ONLY_AUDIT_FAILURE_01.json)；首次失败审计未写独立结束耗时收据，维护账如实为null。另有一次只读汇总缺失常量import的维护故障，均不增加科学请求或退款。

新终点结果显式study_method、checkpoint_role、actor/tensor/source和fixed_update；288个神经结果无旧默认角色混淆，144个native无神经来源或额外H。旧108个角色字段原文件及身份映射原字节保留，按真实参数和映射解释。

独立角色判断原文及结构化结论见[CLAIM_REVIEW_VERDICT](CLAIM_REVIEW_VERDICT.json)和[审查轨迹](paper/review-traces/experiment-result-to-claim/20261109_run01/)。审计程序见[脚本](../project_management/gl_v5_delivery/independent_audit.py)，[数值审计数据](INDEPENDENT_AUDIT.json)。审计耗时{a['wall_s']:.6f}秒，新增科学请求/策略实例化/optimizer/nav/native/H均0，按维护计费，不退款或重置科学账。

限制：一个优化种子；12个已训练布局上的新任务；开发筛选而非独立保留测试；仅代理F&A/导航/安全，不代表实机IK或关节轨迹认证。不得把训练采样变好、H收益或既有部署提速改写成独立RL泛化证据。
''')
    textfile('RESULT_TO_CLAIM.md',f'''# GL-v5结果到主张

intended_claim：冻结SL贪心成本基线的自然起点DRL可在固定新任务开发评价中，同时改善共同SL、已有自然起点LOO和继续SL的完整交付成本。

claim_supported：{verdict['claim_supported']}；confidence：{verdict['confidence']}；单种子candidate：{decision}。

{verdict['what_results_support']}

尚未支持：{verdict['what_results_dont_support']}

证据缺口：{verdict['missing_evidence']}

路由：{verdict['routing_action']}。不因单个成功对照、窗口或后处理收益改变事前AND判定，不自动追加训练/测试。DRL研究目标保持，旧S3与两个缺失、旧负结论保持。
''')
    continuation='值得另立独立多种子确认任务；本轮不自动执行，也不宣称部署成功。' if s['candidate_screen'] else '不建议以当前配方直接进入独立多种子确认；本轮收尾并保留负结果。后续DRL方法修订须新任务事前登记。'
    textfile('NEXT_STEP_DECISION.md',f'''# GL-v5下一步判断

固定128终点已实际完成，工程/来源数值审查PASS；机制筛选为 **{decision}**。{continuation}

G对共同SL、B、C的raw与H后父布局收益及区间见[实际报告](EXPERIMENT_REPORT.md)，必须三者同时满足原门槛；不能只因比某退化对照好而晋级。参考训练新增512付费请求，相对B增加33.3333%终局请求，128更新无扩展；H和固定基线的在线教师计算只发生在训练，G部署仍单次D1贪心+H。

研究目标保持大规模复杂父场景DRL训练—冻结策略快速局部完整约束规划。72例属于训练物理布局的新任务；单种子结论不支持最终泛化/部署成功。既有A/B/C和S1/S2/S3、scope_r1及全部旧角色原文件不覆盖，两个旧缺失不补，原60继续封存。

本轮GitHub独立分支codex/gl-v5-frozen-greedy-20261109，不合并main；实际同步见[GIT_VERSION_AUDIT](GIT_VERSION_AUDIT.json)。
''')
    textfile('README.md',f'''# GL-v5固定贪心参考基线自然起点DRL

仅G128真实训练128更新，1536采样＋512冻结参考＋432固定终点评价，共2480请求。机制筛选 **{decision}**。固定SL256/fresh Adam和原B512问题/自然随机流，唯一主要变量为冻结SL贪心+H成本基线。

- [完整条件、raw/H后质量及机制判断](EXPERIMENT_REPORT.md) · [逐更新/逐路线CSV](TRAINING_AND_ROUTE_RESULTS.csv)
- [训练曲线](TRAINING_CURVES.csv) · [采样/参考成本与优势](SAMPLE_REFERENCE_COMPARISON.csv) · [各实例梯度](GRADIENTS_BY_INSTANCE.csv)
- [432完整路线摘要](results/ENDPOINT_EVALUATION.csv) · [原始完整计划、梯度、权重/Adam/RNG](runs/SCIENCE/)
- [模型张量谱系与固定终点](WEIGHT_LINEAGE.json) · [准确源码绑定](SOURCE_BINDING.json)
- [2048训练请求的实际参数/参考actor来源](TRAINING_REQUEST_IDENTITIES.json)
- [事前协议](PROTOCOL.json) · [512日程与72新任务清单](INPUT_REGISTRY.json) · [优化前实际输入封存](runs/SCIENCE/NEW_TASK_BANK_FROZEN.json)
- [资源账](RESOURCE_ACCOUNTING.json) · [独立审查](INDEPENDENT_REVIEW.md) · [结果到主张](RESULT_TO_CLAIM.md) · [下一步](NEXT_STEP_DECISION.md)
- [实施前评审](IMPLEMENTATION_REVIEW.md) · [数学/RNG/角色工程检查](ENGINEERING_VERIFICATION.json) · [Git版本核对](GIT_VERSION_AUDIT.json)

12个已训练物理布局mixed/blocked各6，每布局1新task draw，三尺寸两端点。不是未见环境测试。旧证据、108原角色字段与追加映射、旧负结论和两缺失保持，未重训A/B/C、未重跑D0/D1/D2、未访问新S3或原60。SL/规则作辅助对照，最终研究方法保持DRL。PT/NPZ沿用Git LFS；工程提速/H收益不记作学习增量。
''')
    write_new(HOME/'STATUS.json',dict(status='COMPLETE_FIXED_G128',candidate_screen=decision,optimizer_updates=128,
        sample_requests=1536,reference_requests=512,evaluation_requests=432,old108_original_bytes_preserved=True,
        original60='SEALED',new_S3_access=0,source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),utc=now()))
    print(dict(reports='created',candidate=decision))


def metadata():
    s=read(HOME/'results/SUMMARY.json');verdict=read(HOME/'CLAIM_REVIEW_VERDICT.json');decision='GO_SINGLE_SEED_CANDIDATE' if s['candidate_screen'] else 'REVISE'
    p=ROOT/'PROJECT_VERSION.json';v=read(p);v['version']=VERSION;v['GL_v5']=dict(status='COMPLETE_FIXED_G128',report=label(HOME/'README.md'),
        branch='codex/gl-v5-frozen-greedy-20261109',candidate_screen=decision,source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),optimizer_updates=128,
        sample_requests=1536,reference_requests=512,evaluation_requests=432,original60='SEALED',main_merged=False);save(p,v)
    p=ROOT/'README.md';t=p.read_text(encoding='utf8');marker='<!-- GL_V4_CURRENT_BEGIN -->'
    t=t.replace('2026.11.09-gl-v4-deployment-natural-start',VERSION,1);idx=t.index(marker)
    block=f'''<!-- GL_V5_CURRENT_BEGIN -->
**GL-v5固定贪心参考小试完成：G128、128更新、1536采样＋512参考＋432终点评价；机制筛选{decision}。** 12个训练物理布局上的72个新任务窗口，mixed/blocked各6布局；固定参考为SL256同输入D1贪心＋H，三自然轨迹原权重、完整似然和原B512问题日程不变。见[本轮实际结果](rc3_gl_v5_20261109/README.md)与[结果到主张](rc3_gl_v5_20261109/RESULT_TO_CLAIM.md)。没有重训A/B/C或重新跑部署网格；旧108角色原文件、身份映射、负结论及两S3缺失原字节保持。以下为此前各阶段历史记录。
<!-- GL_V5_CURRENT_END -->

'''
    if '<!-- GL_V5_CURRENT_BEGIN -->' in t:raise ValueError('Duplicate GL-v5 metadata')
    p.write_text(t[:idx]+block+t[idx:],encoding='utf8')
    p=ROOT/'EXPERIMENT_RESULT_TO_CLAIM.md';t=p.read_text(encoding='utf8');entry=f'''## 2026-11-09：GL-v5固定贪心参考自然起点DRL

- intended_claim：在72个新任务窗口上同时改善共同SL、既有B和C的完整成本。claim_supported={verdict['claim_supported']}，confidence={verdict['confidence']}；事前AND筛选{decision}。
- 实际G128更新128次、1536采样终局＋512参考＋432评价，共2480请求；未重训旧对照，参考模型冻结、独立缓存/RNG，所有优势含负值保留。
- {verdict['what_results_support']}
- 缺口：{verdict['missing_evidence']}；路由：{verdict['routing_action']}。单优化种子、12已训练布局上的新任务；非最终泛化或部署证明，原研究DRL目标不变。
- [完整证据](rc3_gl_v5_20261109/EXPERIMENT_REPORT.md) · [独立角色审查及轨迹](rc3_gl_v5_20261109/INDEPENDENT_REVIEW.md)。旧证据和角色原文件保持。

'''
    p.write_text(t.replace('# 实验结果到主张记录\n\n','# 实验结果到主张记录\n\n'+entry,1),encoding='utf8')


if __name__=='__main__':
    import sys
    {'reports':reports,'metadata':metadata}[sys.argv[1]]()
