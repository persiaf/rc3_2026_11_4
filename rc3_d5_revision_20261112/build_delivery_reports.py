"""Describe the completed campaign from saved tables only; no model execution."""
from pathlib import Path
import csv,json,hashlib,datetime

H=Path(__file__).resolve().parent
ROOT=H.parent
TRACE=ROOT/'paper/review-traces/experiment-result-to-claim/20261112_d5_revision_run01'
def read(name):return json.loads((H/name).read_text(encoding='utf-8-sig'))
def write(name,s):(H/name).write_text(s.rstrip()+'\n',encoding='utf-8')
rows=list(csv.DictReader((H/'tables/AGGREGATE_RESULTS.csv').open(encoding='utf-8-sig',newline='')))
methods=['FrozenSL','D5-SL-U','D5-PG','Fixed3','Detour16']
def selected(group='overall'):
    return [next(r for r in rows if r['method']==m and r['endpoint']==('4096' if m.startswith('D5-') else '0') and r['group']==group and r['aggregation']=='parent_equal_mean') for m in methods]
def table(group='overall'):
    lines=['| 方法 | raw J | raw L | raw K | H后 J | H后 L | H后 K | 成功 | 完整时间p50/p95 (s) |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in selected(group):
        vals=' | '.join(f"{float(r[k]):.4f}" for k in ['raw_J','raw_L','raw_K','J','L','K'])
        lines.append(f"| {r['method']} | {vals} | {r['valid']}/{r['planned']} | {float(r['actual_time_p50_s']):.4f}/{float(r['actual_time_p95_s']):.4f} |")
    return '\n'.join(lines)
d=read('P1_DECISION.json')
g=d['comparisons']
gains='| 比较：PG4096对照 | raw J父配对收益 | H后J父配对收益 | SEL8 H后收益 | DEV3 H后收益 |\n|---|---:|---:|---:|---:|\n'
for name in ['FrozenSL','SL-U']:
    x=g[name];groups=x['groups']
    gains+=f"| {name} | {100*x['raw_J']['mean']:.4f}% | {100*x['J']['mean']:.4f}% | {100*groups['trained_parent_new_tasks_exposed_selection']['mean']:.4f}% | {100*groups['independent_parent_exposed_development']['mean']:.4f}% |\n"

write('EXPERIMENT_REPORT.md',f'''# D5完整教师修订：P1已完成，固定终点无开发收益

本次run为`d5-data-revision-20261112-r1`，基准`20478a4dde2888f86abb9526c9d39ab71a27ee52`，独立分支`codex/rc3-d5-data-revision-20261112`。数据、工程和科学驱动通过后，seed2026111101的D5-PG与匹配D5-SL-U均实际完成4096次更新。4096主终点未过原P1门槛，进入P6。P2 Full-PG/SL-C、P3另外两seed、P4部署优化、P5确认与最终评价均NOT_RUN，指标为null；剩余预算不是追加实验许可。

## 数据和日程的实际完成情况

新增96条完整10×10强教师覆盖96个训练物理父布局。先按允许训练元数据选每父最小原标签index，再锁定完整窗口和端点，不看路线成本选例。未找到可直接绑定的完整标签，因此实际调用原Detour16强教师96次、固定统一H96次；额外8次技术恢复未使用。无新场景、导航求解或替代seed。

每个完整任务实际有70—252个任务、208—732个候选；mixed/blocked各48，A条件0/1/2各32，96个均为closed_depot。10×10指完整父窗口大小，不表示每个父布局均有1000个任务。来源、F/A/D、候选、窗口端点、方法/H、raw/H动作、服务分配、成本和证书见[FULL_TEACHER_MANIFEST](FULL_TEACHER_MANIFEST.json)及其逐条归档。完整动作均经当前Environment逐步mask/step重放并完成全部任务。

旧256条局部教师及原索引逐字节保持。联合352个唯一任务全部通过准入。三seed日程各4096×8槽位，每批4完整+4局部，局部尺寸原轮换不变；每seed完整16384、局部16384次，三个局部尺寸分别5462/5461/5461次。重复使用精确标签，未声称32768个唯一问题。同seed D5-PG、Full-PG、D5-SL直接读取同一个不可变日程；本轮实际PG和SL的全部4096批逐任务一致，Full-PG未执行。PG只用自己的on-policy动作，教师不作PG动作提示。352条同问题FrozenSL参考绑定模型/问题/H/环境/成本身份，冻结复用。

## 训练与恢复

正式起点为准确base-only SL256，文件SHA256 `60f5aa4e59983e92d565f0599823b710965cf2c96a041e7eb483d23a8eb0556c`，tensor SHA256 `1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e`。两流独立从该起点克隆，未使用P0或固定32诊断权重。五张量25152/65088标量可训练，其余39936冻结；Adam1e-4、clip1、batch8、熵0、完整gamma=1回报、H/mask/J不变，无Critic/A1幅度/永久贴近SL guard。原输入和评分路径可微，重建受影响key/cache。

PG真实在线采样32768条完整轨迹；SL-U使用同问题的32768条已有标签重放。更新共8192次。0/1024/2048/4096均保留实际Actor导出，0点质量复用准确SL256同期基线，没有挑中间模型。

工程初次计数器用途/阶段重名失败、P1 step740 Windows文件替换失败、step1771后的系统睡眠时钟维护均保留。最终工程为累计59请求、10次Actor尝试（包含旧15/5），未超原60/24。恢复精确继承模型、Adam、RNG、采样流、游标和预算，共复用11条同Actor完整轨迹；无跨Actor轨迹共享。真实多扣的失败请求与睡眠2363.847163秒不退款。R02/R03/R04B仅修计账、I/O和时钟接口，科学公式、数据与门槛未变。细节见[来源与计账口径](PROVENANCE_AND_ACCOUNTING_NOTES.md)、[740恢复](R03_ACTUAL_RECOVERY_RECEIPT.json)、[1771恢复](R04_ACTUAL_RECOVERY_RECEIPT.json)。

PG含中断累计训练15895.0366秒、分配参考244.2980秒，合计16139.3346秒，低于8h；其活跃训练13530.7694秒。SL-U训练累计4055.0010秒、活跃4054.3501秒。SL-C潜在候选位置的时间比0.294325，尚未匹配，也没有以SL-C名义评价；P1失败后不延长该流。

## 同期完整任务主结果

同一66个已暴露开发请求，11个物理父布局，每父6请求，3尺寸×2端点。SEL8是训练父上的新任务，DEV3是已暴露的非训练父；二者均用于开发筛选。J=L+0.75K，L为导航长度、K为停靠次数。先父内平均再父等权；由于每父均6请求，本表绝对均值与请求等权均值一致。全部两种聚合均另存CSV，不能把均值的比值冒充父配对平均收益。

{table()}

两学习方法为4096终点，其他为同期冻结基线。Fixed3/Detour16为原生最终路线，raw/H两栏保留同一native-final口径，不将其解释为神经策略的无H消融。成功率是当前Environment及保存独立计划证书的代理约束范围，不是实机IK或机械臂关节轨迹认证。

实际完整请求时间包括本研究的压缩归档写入，截止于随后pointer/event/resource日志尾部；尾部仍计入作业总墙钟。各开发方法运行时期和输入缓存冷热不同，因此表中实际计时不能支持部署加速或公平冷热速度比较；P4没有执行。

## 原门槛与父布局判断

收益定义为每父`(对照父均J−PG父均J)/对照父均J`，再父等权；正值才改善。

{gains}

原门槛是：4096全部请求有效，H后J对FrozenSL收益≥1%、对SL-U≥0.5%，SEL与DEV分别不退化。有效性66/66通过，但两个主比较及两个分组均负；两对照各有10/11父退化。P1为`NO_DEVELOPMENT_SIGNAL`，不是技术未达到4096。完整父值见[PARENT_PAIRED_RESULTS.csv](tables/PARENT_PAIRED_RESULTS.csv)，未把66个任务当成66个独立父布局，也未新设事后置信门槛。

PG1024/2048对FrozenSL的H后J父配对收益为−2.8983%/−4.8236%；它们只作固定曲线，不用于选权重。4096时PG相对FrozenSL和SL-U的raw/H停靠集合及访问顺序均在66/66任务改变；动作与路线确实可变，但这没有产生总体成本收益。相对FrozenSL逐任务H-J为22改善、0严格持平、44退化。说明性路线仅取每类case ID字典序首项，不以个例替代总表。

## 分尺寸与暴露层结果

以下仍是同一4096终点与同期基线，不混入旧实验均值。

### 4×3（22请求、11父）

{table('4x3')}

### 6×6（22请求、11父）

{table('6x6')}

### 8×8（22请求、11父）

{table('8x8')}

### SEL8（48请求，训练父新任务，已暴露）

{table('trained_parent_new_tasks_exposed_selection')}

### DEV3（18请求，非训练父，已暴露）

{table('independent_parent_exposed_development')}

本轮端点评价为4×3/6×6/8×8局部完整任务；没有新增完整10×10评价网格或未暴露父确认。因此不能据本表确认完整父任务部署性能或独立泛化。

## 审查、资源与交付

保存输出数值复核PASS，共449149项，覆盖96新增教师、旧256字节、三日程、全部32768条PG记录、两流4096连续权重/优化器链、594条评价和8个导出；复核本身不创建模型、不前向、不规划。最初复核脚本list/tuple类型比较失败已保存，修正仅作用于核算脚本，其276.950322秒保守上界保留计账。独立角色再次从逐请求重算主表及门槛，结论`claim_supported=no`。图表复核通过，英文图注的小修已落实。

累计完整请求33954（33952完成、2失败），包含旧15请求；本次新增33939。Actor尝试累计8202=8192科学+10工程，Critic0。teacher192、reference352、train32769（32768成功+1失败）、eval594、purpose engineering47；用途和阶段不是可相加的两套请求。native228=96教师+132规则评价；统一H/其他H实际合计33725。总预算96h/300000不变，数据335.3188秒<3600秒。最终墙钟（含后处理、审查和发布）以[RESOURCE_ACCOUNTING.json](RESOURCE_ACCOUNTING.json)及最终结账/发布收据为准，科学结束原账另存[RESOURCE_SCIENTIFIC_FINAL_RAW](RESOURCE_SCIENTIFIC_FINAL_RAW.json)，未重置。

未持久化的维护批次反向/前向尾部、旧前置搜索、未暴露原生DP细分及最早零散读取时间仍标UNKNOWN，不补零、不重跑补日志。新增encode/score及四类内部搜索时间另记实际测量，继承的feature/forward字段不能当成本次全量。

- [逐请求及分组表](tables/COMPLETE_TASK_RESULTS.csv)、[两种聚合](tables/AGGREGATE_RESULTS.csv)、[停靠/顺序变化](tables/STOP_SET_ORDER_CHANGES.csv)
- [模型索引](MODEL_INDEX.json)、[实际PG4096权重](streams/2026111101/D5-PG/step04096.actor.pt)、[实际SL-U4096权重](streams/2026111101/D5-SL/step04096.actor.pt)
- [方法图](METHOD_DIAGRAM.md)、[固定终点曲线](figures/fixed_endpoint_quality.pdf)、[父配对图](figures/parent_paired_HJ.pdf)、[保存路线](figures/saved_route_examples.pdf)
- [结果到主张](RESULT_TO_CLAIM.md)、[下一步](NEXT_STEP_DECISION.md)、[独立审查](../paper/review-traces/experiment-result-to-claim/20261112_d5_revision_run01/REVIEWER_RAW.md)

旧v1 `P6_TECHNICAL_DATA_INCOMPLETE`、P1—P5 NOT_RUN及全部旧教师/源码/权重/53行表/固定32封存保持原字节；本次是新执行身份的真实P1结果。原60、旧S3、确认/最终集语义读取0。用户Word和无关文件不提交，不合并main。
''')

write('NEXT_STEP_DECISION.md','''# P6：本配方完成并停止

**固定4096 P1已完成，NO_DEVELOPMENT_SIGNAL。** 数据与科学驱动的技术目标完成；五张量完整回报PG的任务收益门槛没有通过。

PG相对FrozenSL、匹配更新/任务暴露SL-U的父等权H-J收益为−2.6431%、−3.7371%；SEL8/DEV3均负，均有10/11父退化。继续执行条件不成立。因此Full-PG/时间匹配SL-C、另外两seed、部署优化、新父确认和最终评价均NOT_RUN。已有潜在SL-C候选不是已匹配或已评价的C基线。

本执行身份封闭，continue/resume只返回封闭状态；不延长、不换seed、不增加算法/损失/学习率/网络/幅度，不恢复固定32/A1/Critic/旧停止支线。实际模型只作本次科学证据归档，未提升为生产默认，也不自动成为后续初始化。

可靠基础是准确SL起点、原Actor五张量可微路径、合法mask/完整任务转移、强教师来源、同任务监督对照、真实on-policy采样和故障恢复。尚未达到的研究核心是通过DRL改善完整任务规划，并在冻结后获得经同期确认的快速部署。当前结果不能唯一归因于冻结表示、PG噪声或某个优化细节；Full-PG等归因阶段未获门槛准入，缺项保留。

本轮只完成负结果分析、模型/逐请求/资源/独立审查及GitHub交付。任何新机制需要另一个明确批准的任务；本文件不生成新的实验预算。
''')
write('RESULT_TO_CLAIM.md','''# 结果到主张

`claim_supported = no`，对固定seed、固定4096、已暴露66任务的P1负结果置信度高。论文写作状态为**provisional**：独立实验审查已完成，`pending_external_review=false`；仓库无`paper/PAPER_CLAIM_AUDIT.json`，尚无论文级数字审计许可收据。

| 可核验结果 | 支持的陈述 | 不支持的延伸 |
|---|---|---|
| 96完整强教师/96父、256旧标签字节不变、352问题重放PASS | 本修订补齐完整训练监督和共同日程 | 新场景生成、更大规模任务数量或独立测试 |
| 同seed两流4096、32768同任务槽位、真实在线采样及恢复 | 科学CLI、固定任务匹配、原Actor五张量实际训练跑通 | 工程PASS即任务质量提升 |
| H-J父配对收益对FrozenSL−2.6431%、对SL-U−3.7371%，SEL/DEV负 | 本配方未过原P1主门槛，按原条件P6收口 | DRL普遍无效、冻结表示是唯一根因或Full-PG更优 |
| 66/66 raw/H停靠集合和顺序都改变，22/0/44逐任务H-J改善/平/退 | 评分训练实际改变完整路线 | 动作变化、参数变化或损失下降可以替代任务收益 |
| 全594开发请求保存证书有效 | 本次代理环境中的完整任务约束满足 | 实机机械臂IK、安全扫掠或关节轨迹认证 |
| 完整请求真实计时，缓存状态不同 | 成本账可复核 | 部署加速或公平冷热对照；P4未执行 |

研究内容三的固定目标保持：大规模复杂父场景中DRL训练、冻结Actor快速完成不同规模局部完整约束规划。旧S2、GL-v4、GL-v5已实施SL初始化后的全Actor训练，不能把“主Actor可训练”本身作为本轮新机制；本轮明确范围是冻结表示后的原评分五张量。

数据暴露边界：SEL8是训练父新任务，DEV3是已暴露非训练父；单seed/66开发问题仅作机制筛选。确认/最终及另外两seed均未执行。没有独立泛化、跨seed稳定性或部署速度结论。原生Fixed3/Detour16为native-final，不是神经无H消融。

独立数值复核449149项PASS，另有新角色逐请求重算主结论。审查原文、解析verdict、输入prompt和路由见[审查trace](../paper/review-traces/experiment-result-to-claim/20261112_d5_revision_run01/TRACE_MANIFEST.json)。缺失的共享skill附件已按内联契约留痕，不伪造论文审查。门槛失败不触发辅助诊断或补充实验。

建议用于报告的主张：**补齐完整强教师并严格匹配任务暴露后，原Actor五张量完整回报PG完成第一seed的固定试验，但在已暴露完整任务开发网格上劣于FrozenSL和匹配继续SL；原P1条件因此终止本配方。**
''')
write('TEACHER_DECISION_BRIEF.md','''# 供老师决策：数据补齐，D5主配方P1负结果

本轮已新增96条完整10×10强教师覆盖96个训练父布局，保留旧256标签，完成真实生产训练/评价/恢复CLI。PG和五张量继续SL从同一准确SL256分别完成4096更新，逐任务暴露一致。

主要任务结果是：PG的H后J相对FrozenSL父等权退化2.64%，相对匹配继续SL退化3.74%；SEL8与暴露DEV3均退化。完整有效性通过，raw/H停靠集合与顺序确实改变，但不能形成质量收益。原定P1门槛没有通过，后续匹配全Actor/时间SL-C、三seed、部署和独立确认均未执行。

可继续依赖的是原Actor表示与评分的真实可微路径、SL初始化、任务与导航/覆盖代理约束、强教师来源、共同日程以及故障恢复。尚未实现的是DRL对完整任务的可靠增益，以及该增益在冻结部署时同时满足质量与速度要求。本轮不能唯一判定应改表示还是优化；按门槛未执行的Full-PG等缺项必须保留。

旧S2、GL-v4、GL-v5已经做过SL初始化后的全Actor训练，不能再把“让主Actor可训练”当作未尝试的新机制。固定32、A1、Critic和旧停止实验继续封存，本轮不追加辅助诊断、不自动改算法或预算。下一版如要改变机制，须以完整任务质量及匹配监督/计算对照为问题重新审批；本材料不自行提出或启动新试验。

主表、raw/H分量、尺寸/父布局、实际时间和证据限制见[完整报告](EXPERIMENT_REPORT.md)。本结果是一个seed、已暴露开发数据上的机制筛选，不是新的独立泛化确认。
''')
write('findings.md','''# Completed D5 revision findings

- `claim_supported: no`; `confidence: high` for the fixed first-seed P1 negative result.
- Both PG and update/exposure-matched SL-U completed4096. Fixed-endpoint parent-paired H-J gains are−2.6430806085% versusFrozenSL and−3.7370868961% versusSL-U. SEL8 and exposedDEV3 are negative.
- Route validity and actual trainability pass; the task-quality condition fails. P6 closeout; P2—P5 NOT_RUN. No additional experiments are requested or authorized by this analysis.
- Writing verdict: **provisional**, negative-result/engineering evidence only. External-role review completed; `pending_external_review=false`. No `paper/PAPER_CLAIM_AUDIT.json` exists.
- Trace: [prompt/raw/parsed/routing](../paper/review-traces/experiment-result-to-claim/20261112_d5_revision_run01/TRACE_MANIFEST.json). Figure review passed with caption-only presentation revisions, implemented without new computation.
''')

readme=(H/'README.md').read_text(encoding='utf-8-sig')
readme=readme.replace('正式P1正在运行，完整结果以本目录最终报告和实际状态为准。','**P1已完成：PG4096、SL-U4096；原固定终点门槛未通过，NO_DEVELOPMENT_SIGNAL→P6。** H后J父配对收益对FrozenSL为−2.6431%、对匹配SL-U为−3.7371%，SEL8/DEV3均负。P2—P5均NOT_RUN，本身份已封闭。')
readme=readme.replace('## 来源与参数','[完整报告](EXPERIMENT_REPORT.md) · [给老师的短材料](TEACHER_DECISION_BRIEF.md) · [结果到主张](RESULT_TO_CLAIM.md) · [阶段状态](STAGE_RESULTS.json) · [模型索引](MODEL_INDEX.json) · [逐请求表](tables/COMPLETE_TASK_RESULTS.csv) · [资源账](RESOURCE_ACCOUNTING.json) · [独立复核](INDEPENDENT_NUMERICAL_AUDIT.json)\n\n## 来源与参数')
write('README.md',readme)
note=read('EXECUTION_SESSION_NOTE.json')
note.update(active_controller_exec_session=None,active_job=None,status='P6_CLOSED_NO_DEVELOPMENT_SIGNAL',remaining=['Finalize reports/resource/publication only; no scientific rerun'],scientific_completed=True)
(H/'EXECUTION_SESSION_NOTE.json').write_text(json.dumps(note,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

caption_files=['figures/CAPTIONS.md','figures/latex_includes.tex']
receipt=dict(status='IMPLEMENTED_CAPTION_ONLY_MINOR_REVISIONS',review='paper/review-traces/experiment-result-to-claim/20261112_d5_revision_run01/FIGURE_REVIEW.json',
            new_model_or_planning_calls=0,rendered_figure_bytes_unchanged=True,
            source_sha256={p:hashlib.sha256((H/p).read_bytes()).hexdigest() for p in caption_files},
            changes=['English caption aggregation, J formula and units','Route4096/control/tolerance22-0-44 counts and actual case IDs'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(H/'FIGURE_CAPTION_REVISION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status='SAVED_EVIDENCE_REPORTS_WRITTEN',new_forward_optimizer_or_planning=0,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
