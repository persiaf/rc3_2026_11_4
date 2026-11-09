"""Actual GL-v4 delivery documents from sealed saved results, no experiments."""
import json
import csv
from pathlib import Path
from rc3_gl_v4_20261109.common import ROOT,HOME,read,write_new,digest,label,now

VERSION='2026.11.09-gl-v4-deployment-natural-start'


def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def textfile(name,txt):
    p=HOME/name
    if p.exists():raise FileExistsError(p)
    p.write_text(txt,encoding='utf8')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(x):return f'{100*x:.4f}%'


def reports():
    summary=read(HOME/'results/SUMMARY.json');resource=read(HOME/'RESOURCE_ACCOUNTING.json');audit=read(HOME/'INDEPENDENT_AUDIT.json')
    if audit['status']!='PASS':raise ValueError('Independent audit not passed')
    identities=read(HOME/'METHOD_IDENTITY_AUDIT.json')
    if identities['status']!='PASS_ACTUAL_MODEL_IDENTITIES_WITH_DISCLOSED_METADATA_DEFECT':raise ValueError('Method identity audit not passed')
    comparisons=summary['comparisons'];speed=summary['speedups'];candidate=summary['candidate_screen']
    index=read(ROOT/'rc3_s3_evaluator/S3_MODEL_INDEX.json')['models']['11622026']
    actors=read(HOME/'actors/ACTOR_INDEX.json')['models']['11622026']
    size_rows=[[role,index[role]['checkpoint_bytes'],actors[role]['checkpoint_bytes'],pct(1-actors[role]['checkpoint_bytes']/index[role]['checkpoint_bytes']),actors[role]['model_tensor_sha256']] for role in ('SL256','RL512')]
    speed_rows=[[v['role'],v['navigation'],v['mode'],v['paired_requests'],f"{v['D0_median_s']*1000:.3f}",f"{v['new_median_s']*1000:.3f}",
        pct(v['median_reduction']),f"{v['median_speedup']:.3f}",pct(v['paired_fraction_faster']),f"{v['p95_speedup']:.3f}"] for v in speed]
    startup=summary['deployment_startup'];eq=summary['deployment_equivalence']
    startup_rows=[[mode,f"{v['catalog_and_model_ready_s']:.6f}",v['resident_model_count'],v['model_loads'],eq['mode_model_loads'][mode],eq['resident_hits'][mode]] for mode,v in startup['modes'].items()]
    mean_rows=[]
    for size in ('ALL','4x3','6x6','8x8'):
        for role in ('SL256','A128','B128','C128','Fixed3','Detour16'):
            m=next(x for x in summary['method_summary'] if x['size']==size and x['model']==role)
            mean_rows.append([size,role,m['n'],*[f"{m[k]:.4f}" for k in ('raw_J','raw_L','raw_K','J','L','K','H_gain_J')]])
    comparison_rows=[[c['metric'],c['control'],pct(c['mean']),f"[{pct(c['lower95'])}, {pct(c['upper95'])}]",'/'.join(pct(x) for x in c['parent_gains']),
        '平均正且下界>0' if c['exploratory_positive'] else '未建立正改善'] for c in comparisons]
    arm_rows=[[arm,v['updates'],v['terminal_predebits'],v['terminal_success'],v['clipped_updates'],v['zero_gradient_updates'],v['initial_tensor'],v['final_tensor']] for arm,v in resource['science']['arms'].items()]
    decision='GO_SINGLE_SEED_CANDIDATE_ONLY' if candidate else 'REVISE'
    textfile('DEPLOYMENT_EQUIVALENCE.md',f'''# GL-v4部署拆分：实际等价与时间结果

**396/396配对请求完成，参数、实际mask/logits、贪心raw/final路线、服务分配和J/L/K逐位一致；等价工程GO。** 这只验证部署封装，路线质量完全未变，不构成RL学习增量，也不更改旧S3负结论。

## 来源与三个生命周期

固定seed11622026的原SL256与RL512，完整checkpoint保留作恢复证据。新策略包只含schema、策略张量、架构配置、张量身份和训练来源；无Adam、RNG、任务嵌入、路径或H缓存。架构GlobalLocalPolicy(hidden64)，global14/task5/candidate7，源码和参数名/形状/dtype/值均核对；未量化、编译或改变精度。5份包（此处2份及三组128终点的3份）已由独立只读脚本与各自完整checkpoint逐张量比较。

{table(['角色','完整checkpoint字节','策略包字节','尺寸减少','相同张量SHA256'],size_rows)}

D0每请求读取/核验完整checkpoint；D1每请求读取/核验轻量包；D2进程内常驻相同轻量参数。三者每次都重新准备任务特征、编码、构造、H、导出与验证。应用层导航冷热和模型是否常驻是独立因素；不将OS磁盘页缓存称为模型或导航冷状态。

## 准确网格、计时与启动

6个固定旧DEV案例，覆盖4×3/6×6/8×8和闭合/开放端点；2权重×3生命周期×(quality1+cold5+hot5)×6=396。每个case/cache/repeat区组内6种role/lifecycle组合按预登记PCG64种子6110911622交错。冷/quality各请求真实导航求解且不读取或填充热缓存；热模式仅预热导航，三个生命周期各有独立、同源的导航缓存。

共同worker从父进程启动到全部catalog就绪：{startup['process_to_all_catalogs_ready_s']:.6f}秒，其中策略包导出及张量往返核对{startup['export_and_tensor_roundtrip_s']:.6f}秒。下表catalog/model子计时与共同进程启动分开；没有把它冒充三个独立服务进程的冷启动测量，也不将启动开销免费摊除。

{table(['模式','catalog/model就绪秒','启动常驻模型数','启动模型加载数','含请求的总加载数','常驻命中'],startup_rows)}

阶段总墙钟 **{resource['phase_resources']['DEPLOYMENT']['wall_s']:.6f}秒／1800秒**，含数学检查、导出、源码核验、启动、导航预热、全部请求和IO。每请求时间从输入读取/验证到结果写盘/哈希，包含模型访问、完整父场景视图、fresh编码、贪心、H、导出/验证。最终Journal提交计入阶段墙钟。

{table(['权重','导航','轻量模式','配对数','D0中位ms','新中位ms','中位减少','中位加速倍数','配对更快比例','p95加速倍数'],speed_rows)}

上表是本机6旧DEV窗口的观察结果，cold/hot每role/mode各30次。是否减少完整开销须逐角色/缓存/模式判断，不能用文件尺寸减少替代真实时间收益。完整396行含J/L/K和来源见[DEPLOYMENT_BENCHMARK.csv](DEPLOYMENT_BENCHMARK.csv)，时间摘要见[DEPLOYMENT_TIMING](results/DEPLOYMENT_TIMING.csv)。模型常驻的启动费用单列，未复用任务编码、route或H结果。

直接回答：RL512轻量包每请求加载相对完整checkpoint，完整冷导航中位时间减少44.14%，热导航减少53.20%；常驻分别减少43.31%和53.24%。SL256冷导航反而慢约1.60%—1.97%，热导航减少3.83%—4.77%。常驻相对轻量包每次加载没有清楚的额外优势；不能宣称所有模型、所有导航状态均提速。

12个case/weight组合在三生命周期和全部重复中严格等价，fresh编码396次；H结果未缓存。全过程optimizer0、新教师0、新S3及原60读取0。独立审计见[INDEPENDENT_AUDIT](INDEPENDENT_AUDIT.json)。
''')
    textfile('START_OBJECTIVE_ABLATION.md',f'''# GL-v4自然起点目标对照：固定128终点

当前单种子机制筛选：**{decision}**。判定严格同时检查B相对A、共同SL256与C；只比A少退化不算通过。不据此声称最终泛化、部署成功或DRL研究目标已达到。

## 数学与匹配条件

三组从seed11622026同一SL256精确张量起点分别新建Adam，绝不继承S1动量或从旧RL512续训。A原primary/forced-low/forced-high；B只把两强制首动作改为自主采样，三条首动作log probability均计入；A/B均保持batch4、三轨迹、原权重w=(1,.25,.25)及总和1.5归一、detached LOO。

实例损失为 `−Σ_r w_r stopgrad(R_r−(Σ_(s≠r)R_s)/2) logP_r /1.5`，再对4实例平均。A的两辅助logP排除强制首动作；B三条logP包含全部动作。所有R均为负完整H后成本，网络、MDP、动作mask、奖励/终局修正、H、Adam1e-4、clip1、无熵项与无优势标准化不变。合成枚举的期望梯度与独立解析式相同；实际A/B函数在合成输入上的梯度也与独立公式一致，工程optimizer和科学policy forward均0。

A/B512个问题为原seed11622026首128 S2批，64批full10×10、64批local；parent/task/A/窗口/端点与实际view身份完全一致。冻结日程先于首调用，数据RNG固定使用原不可变v3SHA，不因新协议SHA变化。每个trajectory有独立生成器；两组分别持有自己的NamedStreams，用相同slot名实现配对随机数。B的forced_low/high仅是历史RNG槽名，不再强制动作。

C用原256条教师标签继续masked teacher-forcing CE，128更新、batch4，索引日程也预冻结。C无新教师、native或H请求。C是匹配更新次数的监督辅助对照，不是最终研究方法，也不是等FLOPs因果对照。

{table(['组','实际更新','终局预扣','终局完成','裁剪更新','零梯度更新','共同起点张量SHA','128终点张量SHA'],arm_rows)}

每32闭合更新保存完整参数、Adam、RNG、游标、资源和终局缓存；每更新额外保存梯度裁剪前/后及实际新权重。固定128，不评价或选择中间checkpoint。

## 新任务开发评价

36例来自3个既有DEV物理布局上的6个**新task draw**，每draw覆盖三尺寸和两端点；不来自旧S3或原60，不称新布局/未见测试。种子/A/窗口/端点在首次实际调用前预登记，实际scene/F/A及36个view身份在优化前完整封存。保留完整父地图和合法边界候选，scope_r1目标硬检查，不改写旧scope_r1。

全部6模型各36次，合计216份完整计划；神经模型用相同轻量包每请求加载、单次贪心+共同H，两个规则用原生端点适配且不额外加H。raw与H后成本分别报告。所有方法使用同一保存物理导航来源；DEV216首次导航产物加载发生在早期请求，完整耗时照实保留，因此这些时间只作费用/描述，不用作配对部署速度主张。工程速度结论仅来自396网格。

{table(['窗口','模型','案例数','raw J','raw L','raw K','H后/交付J','L','K','H减少J'],mean_rows)}

规则方法raw表示原生内部搜索后的交付路线，无额外共同H，不能与神经纯greedy原始路线混为同一计算层次。完整216例、raw/H后路线和384条更新见[TRAINING_AND_ROUTE_RESULTS.csv](TRAINING_AND_ROUTE_RESULTS.csv)，务必按row_type筛选，不把训练混合任务均值替代新开发完整路线结果。

保存结果存在一项已披露的角色元数据遗漏：A/B/C共108份结果的通用接口字段`training_weight_role`仍写SL256，`plan.method`也是通用adapter名称。实际加载路径、参数哈希、训练来源和`study_method`均对应正确的A/B/C，CSV分组及数值逐份一致；并未使用共同SL冒充三个终点。原始文件与冻结源码保持不动，严格追加[身份澄清](METHOD_IDENTITY_CLARIFICATION.json)及[216份身份映射审计](METHOD_IDENTITY_AUDIT.json)。后续读取须按已验证的study_method和精确参数身份分组，不按此默认角色字段分组。

## 事前统计及判断

以3个真实物理布局聚类，每布局先平均12子案例，再计算相对B收益；20000次父布局bootstrap，PCG64种子11632026。正值是B较低成本。36窗口/6任务draw不是36/6个独立物理布局，单优化种子不等于三种子复现；区间不宣称familywise同时覆盖。

{table(['成本层次','B对照','父布局平均收益','95%区间','三个布局收益','描述'],comparison_rows)}

事前candidate规则：B的H后成本对A、SL256、C**各自**平均收益>0且bootstrap95下界>0，并且全部计划/数值/来源有效；raw比较单列。实际结果为{decision}。其他尺度/端点数据只用于预登记的描述，不能事后挑选成功子集来替代整体筛选。

B相对A的raw成本平均改善8.6604%，但相对共同SL为−1.2147%，相对C为−0.3788%。H后相对A为+1.6183%、共同SL为+0.2321%，两者区间均跨零；相对C为−0.2829%。因此B减轻了A的原始路线退化，尚未获得可靠的独立学习收益，也未优于强规则。

H收益和轻量封装工程收益均不记为RL学习增量。即使candidate通过也只允许提出新的独立验证任务；本任务不自动追加训练或消费测试。
''')
    textfile('INDEPENDENT_REVIEW.md',f'''# 独立保存数据数值与来源审查

数值、实际模型来源与路线审查：**GO／PASS，附一项已披露元数据缺陷**。此处“独立”指独立实现并在独立进程执行的只读数值/来源核对；没有导入实验实现、实例化策略、运行autograd/optimizer或调用导航/规划器，不冒充外部人类同行评审。

实际核查：{audit['bound_source_files']}源码文件身份、{audit['old_frozen_files_unchanged']}旧冻结文件原字节、36份原模型来源；5份轻量包逐张量与完整来源参数/精度一致。612份实际计划逐例从原F/A和保存D重算合法候选、mask、同臂分配、raw/final J/L/K及导航polyline长度；实际服务分配{audit['same_arm_assignments_verified']}条，greedy mask {audit['greedy_masks_recomputed']}个。

384次更新的日志预扣/闭合、每次实际梯度范数、clip1前后张量、更新后权重哈希、共同SL起点和fresh Adam初态通过。A/B512问题与view相同；LOO、1/.25/.25、B首动作参与和3072终局预扣一致。梯度裁剪最大数组差{audit['max_clipping_array_error']:.3g}；父布局收益/区间与独立重算最大差{audit['max_cluster_stat_difference']:.3g}。

**GLV4_META_001：通用adapter默认角色未覆写。** A/B/C的108份原始结果仍有`training_weight_role=SL256`；`plan.method=frozen_supervised`表示共用执行管线，不能表示实际学习方法。代码根因是`complete_request`以通用method调用旧接口，同时通过Catalog.override_role正确载入各组模型，随后未重标角色。独立追加审计对全部216份原始artifact和Journal哈希、声明study_method、actor文件/张量哈希、固定128来源及CSV分组/成本进行严格匹配，144份神经计划和72份native均符合准确来源。没有改动任何原始结果或科学源码，也未重跑、补值或隐藏此字段错误；[澄清](METHOD_IDENTITY_CLARIFICATION.json)和[逐份映射](METHOD_IDENTITY_AUDIT.json)供复核。数值结论不变，不声称原始角色元数据全部正确。

部署等价工程GO，时间减幅只能按实际role/cache表报告。自然起点的单种子candidate判定为{decision}，不能用B仅优于A、后处理或工程提速替代学习收益。完整成本与限制见[目标对照](START_OBJECTIVE_ABLATION.md)。

限制：新任务仅3个已暴露DEV物理布局；未验证新的未见布局、跨优化种子或实机IK。DEV216请求计时含第一次保存导航产物加载，不能拿它作严格配对部署速度比较；396网格的冷热和模型状态独立、可比较。原S3两个缺失和负结论保留，不回填、不换任务或改A。

审计墙钟{audit['wall_s']:.6f}秒，新增科学请求/策略forward/optimizer/nav/native/H均0。[完整审计数据](INDEPENDENT_AUDIT.json)及[独立脚本](../project_management/gl_v4_delivery/independent_audit.py)。
''')
    textfile('NEXT_STEP_DECISION.md',f'''# GL-v4下一步决定

部署封装：等价工程GO；是否减少完整开销按[实际配对时间](DEPLOYMENT_EQUIVALENCE.md)逐角色/缓存回答，不赋予路线质量收益。

自然起点DRL：**{decision}**。384更新及3288训练/评价请求已完成；独立来源、数学和路线审查PASS。该判定只面向本轮单种子机制筛选，不能解释为最终泛化/部署成功。若未通过，不凭B优于A或只比另一训练少退化宣布学习改进。

部署RL512完整开销下降已有配对工程证据，SL冷导航未改善，常驻没有清楚额外优势。B对A的raw改善未转化为相对共同SL和继续SL的可靠完整路线收益。停止本轮固定预算实验，保留负结果。原结果108个通用角色默认值的遗漏已用[精确身份映射](METHOD_IDENTITY_AUDIT.json)澄清；实际权重来源核查通过，原始证据保持。

固定研究目标保持大规模复杂父场景DRL训练—冻结策略快速局部完整规划，不以SL或规则替代。当前任务到固定128终点评价停止，不自动追加训练、挑中间模型、补旧S3失败或读取原60。后续改变方法/预算/分布或开展独立确认须新任务及事前协议，不覆盖旧证据。

GitHub交付在codex/gl-v4-natural-start-20261109，发布事实见[GIT_VERSION_AUDIT](GIT_VERSION_AUDIT.json)，不合并main。科学结论、资源和完整路线分别见[目标对照](START_OBJECTIVE_ABLATION.md)、[资源账](RESOURCE_ACCOUNTING.json)、[合并CSV](TRAINING_AND_ROUTE_RESULTS.csv)。
''')
    textfile('EXPERIMENT_REPORT.md',f'''# GL-v4实际实验报告

部署396/396完成，等价PASS；三组各128更新，共384；RL请求3072、固定终点评价216，共3288，新增失败0。部署阶段{resource['phase_resources']['DEPLOYMENT']['wall_s']:.6f}/1800秒，训练与评价共享阶段{resource['phase_resources']['SCIENCE']['wall_s']:.6f}/7200秒。

模型包仅改变部署生命周期，保留参数/实际mask/logits/路线/J/L/K一致；完整开销是否降低及启动费用见[DEPLOYMENT_EQUIVALENCE](DEPLOYMENT_EQUIVALENCE.md)。自然起点仅改变强制首动作的采样/计分，保留原权重与detached LOO；真实新36例的B对A/SL/C及强规则完整结果、raw/H归因和单种子筛选{decision}见[START_OBJECTIVE_ABLATION](START_OBJECTIVE_ABLATION.md)。

新任务6draw/3旧DEV几何在优化前封存；不宣称新布局泛化。原S1/S2/S3权重、scope_r1、两旧缺失及负结论原字节保持，未追加旧训练或消费新S3/原60。新协议、实例日程、源码绑定在首实际调用前已保存并提交，运行源码SHA为`{digest(HOME/'SOURCE_BINDING.json')}`。

实际模型与路线来源通过独立复核。发现并追加披露108份A/B/C结果的通用默认角色字段标注遗漏；实际actor/tensor/source和CSV分组全部正确，原始结果字节与冻结源码不改写，见[身份澄清](METHOD_IDENTITY_CLARIFICATION.json)。

[全部逐请求/逐更新CSV](TRAINING_AND_ROUTE_RESULTS.csv)、[部署CSV](DEPLOYMENT_BENCHMARK.csv)、[资源账](RESOURCE_ACCOUNTING.json)、[独立审查](INDEPENDENT_REVIEW.md)、[下一步](NEXT_STEP_DECISION.md)。
''')
    textfile('RESULT_TO_CLAIM.md',f'''# GL-v4结果到主张

支持：两原冻结模型的轻量部署策略包与完整checkpoint参数及实际规划输出等价；具体工程时间收益以396配对观测为限。三fresh Adam匹配起点、原权重自然起点与继续SL的128终点按预算实际完成。

自然起点candidate判定：{decision}；B相对原配方、共同SL和继续SL的raw/H后成本区间分别见[完整对照](START_OBJECTIVE_ABLATION.md)。单种子、3个已暴露DEV物理布局的新任务，只能用于机制筛选，不支持最终泛化/部署成功。不能把封装工程提速、H收益或仅B优于A描述为独立RL质量增益。

原S3负结论及两历史缺失不变，研究目标仍以DRL为核心。[独立核对](INDEPENDENT_REVIEW.md)是独立数值程序审计，不是人类同行审稿或全文论文claim审计。
''')
    textfile('README.md',f'''# GL-v4部署拆分与自然起点DRL对照

实际完成396部署请求、三组384更新、3072训练终局请求和216新任务终点评价。部署严格等价PASS；自然起点单种子筛选：**{decision}**。旧S3及两缺失原样保留。

- [部署等价、大小、启动和完整时间](DEPLOYMENT_EQUIVALENCE.md) · [396逐请求CSV](DEPLOYMENT_BENCHMARK.csv)
- [起点目标完整对照及raw/H成本](START_OBJECTIVE_ABLATION.md) · [384更新＋216路线CSV](TRAINING_AND_ROUTE_RESULTS.csv)
- [完整路线216行](results/ENDPOINT_EVALUATION.csv) · [训练曲线](results/TRAINING_CURVES.csv) · [匹配比较](results/MECHANISM_COMPARISONS.json)
- [资源账](RESOURCE_ACCOUNTING.json) · [独立数值审查](INDEPENDENT_REVIEW.md) · [下一步](NEXT_STEP_DECISION.md)
- [原始角色字段遗漏的追加澄清](METHOD_IDENTITY_CLARIFICATION.json) · [216份精确模型身份映射](METHOD_IDENTITY_AUDIT.json)
- [预登记协议](PROTOCOL.json) · [512共同日程及36新任务声明](INPUT_REGISTRY.json) · [准确源码绑定](SOURCE_BINDING.json) · [实施前评审](IMPLEMENTATION_REVIEW.md)
- [部署策略包及来源](actors/ACTOR_INDEX.json) · [固定128终点策略包](runs/SCIENCE/DEV216/ACTOR_INDEX.json) · [实际完整checkpoint/Adam/RNG/梯度与计划](runs/)
- [Git版本与远端核验](GIT_VERSION_AUDIT.json)

新任务使用3个已有DEV物理布局上的6个新task draw，非未见S3。单种子不能证明最终泛化。精度/网络/奖励/H未变；工程速度、后处理与学习收益分开。所有PT/NPZ按现有Git LFS管理，检出提交后执行`git lfs pull`。不生成额外整包备份、不合并main。
''')
    write_new(HOME/'STATUS.json',dict(status='COMPLETE_BOUNDED_GL_V4',experiment_id=resource['experiment_id'],baseline='925a0a1a8957299e3e849b29ac501a99212aa902',
        source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),deployment_equivalence='PASS',deployment_requests=396,
        optimizer_updates=384,training_terminal_requests=3072,endpoint_evaluation_requests=216,candidate_screen=decision,
        original_S3_unchanged=True,original60='SEALED',new_S3_access=0,utc=now()))
    print(dict(documents='created',candidate=decision))


def metadata():
    summary=read(HOME/'results/SUMMARY.json');r=read(HOME/'RESOURCE_ACCOUNTING.json');p=ROOT/'PROJECT_VERSION.json';v=read(p)
    v['version']=VERSION;v['current_status_date']='2026-11-09';v['GL_v4']=dict(status='COMPLETE_BOUNDED',report=label(HOME/'README.md'),
        source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),deployment_requests=396,optimizer_updates=384,RL_requests=3072,evaluations=216,
        natural_start_candidate=summary['candidate_screen'],old_S3_results_exposure_and_gaps_preserved=True,
        branch='codex/gl-v4-natural-start-20261109',main_merged=False)
    save(p,v)
    p=ROOT/'README.md';txt=p.read_text(encoding='utf8');txt=txt.replace('2026.11.09-s3-scope-r1-formal-results',VERSION,1)
    block='''<!-- GL_V4_CURRENT_BEGIN -->
**GL-v4独立实验已完成：396部署配对请求、三组384更新、3072训练终局及216新任务完整评价。** 轻量部署与原完整checkpoint参数、mask/logits、路线和J/L/K严格一致，具体完整开销与启动费用逐条件报告；自然起点筛选为REVISE，B未可靠改善共同SL及继续SL。B保留原1/.25/.25，仅改变两辅助首动作的采样与计分。新36例来自3个旧DEV几何上的6个新任务draw，单种子机制筛选不能宣称最终泛化/部署成功。见[本轮全部结果](rc3_gl_v4_20261109/README.md)、[部署对照](rc3_gl_v4_20261109/DEPLOYMENT_EQUIVALENCE.md)、[自然起点结果](rc3_gl_v4_20261109/START_OBJECTIVE_ABLATION.md)和[资源/审查](rc3_gl_v4_20261109/INDEPENDENT_REVIEW.md)。原S3负结论、暴露状态及两个缺失原字节保留；以下是此前各阶段历史记录。
<!-- GL_V4_CURRENT_END -->

'''
    marker='<!-- S3_SCOPE_RESULTS_CURRENT_BEGIN -->'
    if '<!-- GL_V4_CURRENT_BEGIN -->' in txt:raise ValueError('GL-v4 block exists')
    idx=txt.index(marker);p.write_text(txt[:idx]+block+txt[idx:],encoding='utf8')
    p=ROOT/'EXPERIMENT_RESULT_TO_CLAIM.md';txt=p.read_text(encoding='utf8');entry=f'''## 2026-11-09：GL-v4部署拆分与自然起点对照

- 实际396/396部署请求、三组384优化器更新、3072终局请求和216新任务方案；独立数值/来源审计PASS。
- 部署3生命周期的参数、mask/logits、路线、J/L/K一致。工程完整开销按role/cache报告，不算RL学习收益，H单列。
- B只改变两强制首动作，保留原1/.25/.25与detached LOO；同起点、同512问题日程、fresh Adam，必须对A/SL/C全部检查。单种子candidate={summary['candidate_screen']}，具体方向和区间见[完整结果](rc3_gl_v4_20261109/START_OBJECTIVE_ABLATION.md)。
- 新36任务来自3旧DEV布局6新draw，不是新布局泛化；不把旧S3变回未见测试、不填补其缺失、不自动扩训，固定DRL研究目标保持。

'''
    p.write_text(txt.replace('# 实验结果到主张记录\n\n','# 实验结果到主张记录\n\n'+entry,1),encoding='utf8')
    print(VERSION)


if __name__=='__main__':
    import sys
    {'reports':reports,'metadata':metadata}[sys.argv[1]]()
