"""Closeout documentation from saved S3 results; never run scientific pipelines."""
import csv
import json
from pathlib import Path
from rc3_s3_evaluator.common import ROOT,read,digest,write_new,now,require,label

HOME=ROOT/'rc3_s3_scope_results_20261109'
RES=HOME/'results'
BASE='6a7ff5dbbf26df2ec47532f402c8f18f10b7198a'
VERSION='2026.11.09-s3-scope-r1-formal-results'
SOURCE='311a58bd867e95c852ec3246c54a398468eb526db391fbb6d12239c6a3379949'


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def markdown(name,content):
    p=HOME/name;require(not p.exists(),'Do not overwrite closeout '+name);p.write_text(content,encoding='utf8')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def pct(v):return f'{100*v:.4f}%'
def strata(group,size):return ('已训练布局' if group=='seen' else '未训练布局')+' '+('×'.join(map(str,size)) if not isinstance(size,str) else size.replace('x','×'))


def reports():
    a=read(RES/'SCIENTIFIC_ANALYSIS.json');r=read(RES/'RESOURCE_ACCOUNTING.json')
    audit=read(RES/'CONSTRAINT_SOURCE_AUDIT.json');review=read(HOME/'INDEPENDENT_REVIEW.json')
    comp=read(RES/'MATCHED_COMPARISONS.json')['comparisons']
    with (RES/'METHOD_QUALITY_TIME.csv').open(encoding='utf8',newline='') as f:methods=list(csv.DictReader(f))
    lookup={(v['group'],v['size'],v['method']):v for v in methods}
    qcomplete=[v for v in a['quality_cells'] if v['status']=='COMPLETE']
    dcomplete=[v for v in a['deployment_cells'] if v['status']=='COMPLETE']
    require(r['new_success']==13817 and r['new_failures']==0 and r['total_success']==15874,'Unexpected execution result')
    require(len(qcomplete)==5 and len(dcomplete)==9 and not any(c['passed'] for c in qcomplete+dcomplete),'Reassess report narrative against actual gates')
    qtable=[];gtable=[];dtable=[];ctable=[];htable=[]
    for cell in a['quality_cells']:
        group=cell['group'];size='x'.join(map(str,cell['size']));key=(group,size)
        vals=[float(lookup[key+(m,)]['J_mean']) for m in ('Fixed3_endpoint_adapter','Detour16_endpoint_adapter','untrained','frozen_supervised','continued_SL','RL_raw','RL_plus_H')]
        qtable.append([strata(group,cell['size']),*map(lambda x:f'{x:.4f}',vals),cell['status']])
        cc=[v for v in comp if (v['group'],v['size'])==key]
        gtable.append([strata(group,cell['size']),*[pct(next(x for x in cc if x['control']==m)['registered_parent_gain_mean']) if cell['status']=='COMPLETE' else '缺失／不判定' for m in ('frozen_supervised','continued_SL','Fixed3_endpoint_adapter','Detour16_endpoint_adapter')],
            '未通过' if cell['passed'] is False else 'INCOMPLETE'])
        for control in cell['controls']:
            ctable.append([strata(group,cell['size']),control['control'],pct(control['mean']),f"[{pct(control['lower95'])}, {pct(control['upper95'])}]",'/'.join(pct(g) for g in control['seed_gains']),'未通过'])
        hc=next(x for x in cc if x['control']=='RL_raw')
        htable.append([strata(group,cell['size']),pct(hc['registered_parent_gain_mean']) if cell['status']=='COMPLETE' else '仅71/72配对观察，不作正式格判定',hc['matched_quality_pairs']])
    for cell in a['deployment_cells']:
        rl=cell['timing']['RL_plus_H'];det=cell['timing']['Detour16_endpoint_adapter']
        dtable.append([strata(cell['group'],cell['size']),cell['cache'],f"{rl['median_s']*1000:.2f}",f"{rl['p95_s']*1000:.2f}",
            f"{det['median_s']*1000:.2f}",f"{det['p95_s']*1000:.2f}",
            f"{cell['median_ratio']:.4f}" if cell.get('median_ratio') is not None else '不判定',
            f"{cell['p95_ratio']:.4f}" if cell.get('p95_ratio') is not None else '不判定',
            '是' if cell.get('Fixed3_dominates') else ('不判定' if cell.get('Fixed3_dominates') is None else '否'),
            '未通过' if cell['passed'] is False else 'INCOMPLETE'])
    seedtable=[[s,z['new_debits'],z['new_success'],z['total_debits'],z['total_success'],len(z['historical_failure_classes']),
                f"{z['measured_new_wall_s']:.6f}",f"{z['charged_wall_s']:.6f}",f"{z['remaining_wall_s']:.6f}"] for s,z in r['actual_by_seed'].items()]
    old=read(ROOT/'rc3_s3_recovery_20261108/results/RESOURCE_ACCOUNTING.json')
    top=dict(r)
    top.update(scope='All S3 science plus separately identified scope_r1 continuation and read-only analysis',
        this_turn=dict(request_debits=13817,successful_requests=13817,failed_requests=0,
            measured_stage_wall_s=sum(z['measured_new_wall_s'] for z in r['actual_by_seed'].values()),
            request_operation_counts={k:v-old['request_operation_counts'].get(k,0) for k,v in r['request_operation_counts'].items()},
            hot_prime_counts={k:v-old['hot_prime_counts'].get(k,0) for k,v in r['hot_prime_counts'].items()},
            optimizer_updates=0,teacher_queries=0,original60_semantic_reads=0),
        inherited=dict(predebits=2059,success=2057,failed=2,first_seed_charged_wall_s=2896.9277158999903,exact_original_stage_wall_s=None),
        separately_paid_prior_diagnostic=dict(navigation_calls=1,wall_s=read(ROOT/'rc3_s3_connectivity_20261109/RESOURCE_ACCOUNTING.json')['diagnostic']['all_read_compute_save_wall_s'],report='rc3_s3_connectivity_20261109/RESOURCE_ACCOUNTING.json',included_in_current_scientific_navigation_total=False),
        read_only_postprocessing=dict(production_analysis_wall_s=read(ROOT/'rc3_s3_connectivity_20261109/runs/rc3_v3_global_local_20261106_01/scope_r1/SCIENTIFIC_ANALYSIS_SCOPE_R1.json')['postprocessing_wall_s'],
            closeout=read(RES/'READ_ONLY_CLOSEOUT.json'),matched_comparisons_wall_s=read(RES/'MATCHED_COMPARISONS.json')['read_only_wall_s'],
            component_profile_wall_s=read(RES/'SAVED_COMPONENT_PROFILE.json')['wall_s'],new_scientific_calls=0,
            independent_review_accounting='INDEPENDENT_REVIEW.json and paper/review-traces; Git/publication maintenance separately timestamped, not training or planning'))
    write_new(HOME/'RESOURCE_ACCOUNTING.json',top)
    identities={
        'protocol':'rc3_global_local_execution_v3/V3_EXECUTION_CANDIDATE.json',
        'registry':'rc3_global_local_execution_v3/REGISTRY.json',
        'training_source':'rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json',
        'evaluation_source':'rc3_s3_connectivity_20261109/SOURCE_BINDING.json',
        'original_evaluation_source':'rc3_s3_evaluator/S3_SOURCE_BINDING.json',
        'model_index':'rc3_s3_evaluator/S3_MODEL_INDEX.json',
        'original_S3_authorization':'rc3_s3_results_20261108/authorizations/S3_AUTHORIZATION.json',
        'scope_recovery_decision':'rc3_s3_connectivity_20261109/RECOVERY_DECISION.json',
        'actual_execution_instruction':'rc3_s3_scope_results_20261109/EXECUTION_INSTRUCTION_RECEIPT.json'}
    write_new(HOME/'EVIDENCE_IDENTITY.json',dict(experiment_id='rc3_v3_global_local_20261106_01',baseline=BASE,
        source_binding_sha256=SOURCE,H_identity='5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14',
        identities={k:dict(path=p,sha256=digest(ROOT/p)) for k,p in identities.items()},
        runtime_source_changed_this_turn=False,analysis_helpers_are_not_production_evaluation_source=True,
        model_source_audit='results/MODEL_SOURCE_AUDIT.json',frozen_prior_files_verified=2515))
    markdown('IMPLEMENTATION_REVIEW.md',f'''# S3 scope_r1 实际执行复核

技术执行 **GO／已执行完成**。本轮直接使用准确绑定 `{SOURCE}` 的既有生产入口，没有修改正式评估器、训练实现、模型、奖励、H或门槛。真实执行指令和原阶段授权分别保留在[EVIDENCE_IDENTITY](EVIDENCE_IDENTITY.json)，不重复批准v3。

三种子的未执行请求按原注册顺序全部处理；Journal、HEAD、请求结果哈希、逐阶段时钟与调度封口通过只读复核。旧2059预扣和2896.9277158999903秒首种子预算结转保留；不退款、不重跑两旧失败。只读分析代码位于`project_management/s3_scope_execution_delivery`，与194个冻结生产源码文件及12固定模型来源分开，不能作为新评估绑定冒充原311a绑定。

108个声明输入使用v3+scope_r1：完整父地图/障碍/通道/全局上下文、同臂F&A、全部合法边界候选与端点保留，实际目标可服务性仍是硬检查。旧全父不可服务性记录为诊断。这是明确的输入接纳修订，不能称为未修订v3的完整科学网格。

完整质量格5个和完整部署格9个均未通过原数值门槛；另1质量／3部署格不完整。技术运行成功不等于性能命题成立。详见[实验报告](EXPERIMENT_REPORT.md)、[独立审查](INDEPENDENT_REVIEW.md)和[下一步](NEXT_STEP_DECISION.md)。
''')
    markdown('EXPERIMENT_REPORT.md',f'''# S3 scope_r1 冻结策略正式评价结果

本轮实际完成 **13817/13817** 个获准未执行请求，新增失败0；累计15876预扣、15874有效结果和2历史缺失。按既定规则，5个完整质量格均未通过，9个完整部署格均未通过；其余1质量格和3部署格保持INCOMPLETE。没有获得DRL相对监督模型或强规则的实用优势。

## 执行与来源

实验`rc3_v3_global_local_20261106_01`，基准`{BASE}`，分支`codex/global-local-preflight`，准确执行源码绑定`{SOURCE}`。冻结原v3、Registry、H及untrained／SL256／continuedSL512／RL512三个seed终点均不变。原60测试语义读取0，训练更新0，新教师查询0。正式入口仅启动一次并正常退出0。

{table(['Seed','本轮预扣','本轮成功','累计预扣','累计成功','历史失败','本轮阶段秒','累计预算秒','剩余预算秒'],seedtable)}

三段新阶段墙钟合计{top['this_turn']['measured_stage_wall_s']:.6f}秒，继承后预算账合计{r['charged_total_stage_wall_s']:.6f}秒。原最早阶段准确墙钟仍为null；592秒是明示的保守预算结转，不伪称精确测时。上述计时包含阶段核验、准备、导航、推理、H、导出、验证和IO；封口后跨种子统计及独立审查另列只读分析账，不伪装为零耗时。各seed均未超过14400秒或5292请求。

## 输入与约束

108/108声明局部输入在scope_r1下可服务；保存的15874份成功计划生产验证均通过，逐例覆盖证书及端点/碰撞证书已核；独立只读重核{audit['task_assignments_verified']}条服务分配，全部覆盖目标且满足同一臂F&A。该比例以成功返回的计划为分母，不能写成15876/15876请求成功。原全父范围拒绝和持久化失败仍在原账。

108输入的scene/F/A原始receipt、结果来源D摘要、全局任务/候选ID、parent/view identity及重复间路线稳定性通过核对；12模型共36来源文件哈希不变，2515个冻结旧文件原字节保持。RL_raw与RL_plus_H的2267个完整匹配请求对原始路线一致，另1个质量配对因历史缺失不补值；2268个case/seed/method组合的可用重复路线和成本确定性一致。约束是导航、覆盖和逐臂可达的代理模型验证，不是实际机器人标定、IK或关节轨迹认证。

## 七方法完整路线成本

下表为独立quality请求的观察均值，J=L+0.75K，越低越好。Fixed3/Detour16按原生开放/闭合端点适配，不额外加共同H；其余模型除RL_raw均单次贪心后加共同H。untrained也加同一H。未训练布局4×3的RL_plus_H仅71/72，其余方法72/72；该行非平衡描述不能用于正式比较，严格配对观察另见CSV。

{table(['布局与窗口','Fixed3','Detour16','untrained+H','frozenSL+H','continuedSL+H','RL_raw','RL+H','质量格'],qtable)}

完整逐请求J/L/K、raw J/L/K、输入身份、路线哈希、耗时及组件时间见[REQUEST_METRICS.csv](results/REQUEST_METRICS.csv)，完整结果和路线见[RESULT_ARTIFACT_INDEX](results/RESULT_ARTIFACT_INDEX.json)。全部方法的L/K、冷/热中位数和p95见[METHOD_QUALITY_TIME.csv](results/METHOD_QUALITY_TIME.csv)，三个模型seed分开见[QUALITY_BY_MODEL_SEED.csv](results/QUALITY_BY_MODEL_SEED.csv)。不使用训练期间J替代这些冻结结果。

## DRL增量与H归因

原聚合顺序：同一物理父布局内先平均两端点、匹配三个模型seed，再对每父布局计算(control−RL)/control；正值代表RL改善。已训练6个物理布局，未训练12个，不能将15874个重复请求当独立样本。20000次PCG64父布局重采样，固定seed11632026；区间不宣称familywise同时覆盖。

每个质量格须同时相对frozenSL与continuedSL满足：平均收益≥0.5%、95%下界>0、三个固定seed各自收益>0；全局须全部六格完整且通过。部署每格要求相对Detour成本退化单侧95%上界≤0.5%、中位时间比≤0.7、p95比≤1，且不被Fixed3在J/中位/p95同时支配；全局须全部十二格完整且通过。

{table(['布局与窗口','对frozenSL收益','对continuedSL收益','对Fixed3收益','对Detour16收益','原质量门槛'],gtable)}

{table(['布局与窗口','监督对照','平均相对收益','95%区间','三个seed收益','判定'],ctable)}

五个完整格相对frozenSL平均收益均负，三seed对应收益也均负；对continuedSL同样没有格通过。相比强规则，完整格的RL+H成本也更高。这是本配置的冻结评价负结果，不是对DRL一般能力的否定。未训练4×3格保留缺失，不计算原正式区间或门槛。

{table(['布局与窗口','H相对同一RL_raw的收益','质量配对数'],htable)}

上表改善属于确定性H，只能说明后处理改进了相同神经原始路线，不能归于RL学习。共同H累计9071次（untrained2268、frozenSL2268、continuedSL2268、RL_plus_H2267）；RL_raw及两个原生规则方法的额外共同H均0。原生算法内部确定性搜索另记账。

## 十二部署格与完整时间

所有时间保留加载/核验、准备、特征/编码、构造、H、导出、验证与IO。热模式仅预热导航，特征、模型构造与最终路线没有跨请求免费复用。表内时间单位ms；不完整格虽保留实际观察时间，不计算正式比值/门槛。

{table(['布局与窗口','缓存','RL中位','RL p95','Detour中位','Detour p95','中位比','p95比','Fixed3支配RL','原部署门槛'],dtable)}

9个完整部署格中位比1.7648–2.7157、p95比1.5736–2.6444，均未达到中位≤0.7、p95≤1；同时Detour质量退化上侧界未达≤0.5%，Fixed3在J/中位/p95上支配RL+H。未获得任何完整规模/布局部署格优势。

保存组件诊断显示，所有hot请求汇总的RL+H模型加载/核验中位约0.550730秒，编码约0.013669秒、贪心约0.017332秒；frozenSL加载/核验中位约0.012778秒，continuedSL约0.206846秒。RL完整checkpoint约37.8–38.0MB，frozenSL约0.84MB。加载阶段包含完整身份核验及checkpoint读取；这是可观测开销，不是已验证的因果消融。未测量剥离checkpoint或常驻模型的反事实速度，不修改协议计时，也不能用单独前向速度替代完整请求结果。详见[SAVED_COMPONENT_PROFILE](results/SAVED_COMPONENT_PROFILE.json)。组件时间可能嵌套，各组件中位数不能相加当成总时间。

## 失败、缺失与版本限制

quality2267/2268、cold6803/6804、hot6804/6804。

- 历史槽163：seed11622026，S3_003／Fixed3／cold重复1，持久化基础设施失败。它使已训练6×6冷部署格缺失。
- 历史S3_042：同seed，RL_plus_H／quality重复0，原全父范围拒绝。它使未训练4×3质量格及冷/热部署格缺失。

两个失败都未退款、未补跑，未用其他重复/seed/均值补值，最终全局independent_RL_pass、deployment_pass和combined_pass严格为null。完整格的明确未通过结论仍可报告，不能把请求调度完成写成完整科学网格通过，也不能把null改成false或true。

旧2057结果与新scope_r1结果保持各自版本。scope_r1实际改变输入接纳范围，原先已通过全父检查的局部view语义保持；统一地图、任务、A、候选、模型和H未变。混合计时保留持久化与scope标签，[TIMING_BY_REVISION.csv](results/TIMING_BY_REVISION.csv)逐版本给出观察统计。请求计时边界一致但新增范围证据及修复IO开销不保证完全相等，合并时间是获准网格下的观察结果，不宣称版本无开销差异或纯神经前向速度。

## 资源、审查与结论

累计实际导航求解{r['navigation_solves']}次（请求内{r['request_operation_counts']['navigation_solves']}、导航预热55），导航热命中6804，原生适配请求4535，神经贪心11339，共同H9071，新增优化器0、教师0；先前1次收费诊断导航独立列示，未冒充原D恢复或本轮免费缓存。内部construct/two_opt/joint_search/recipe、验证、IO和组件秒详见[RESOURCE_ACCOUNTING](RESOURCE_ACCOUNTING.json)。

独立审查见[INDEPENDENT_REVIEW](INDEPENDENT_REVIEW.md)，完整来源见[EVIDENCE_IDENTITY](EVIDENCE_IDENTITY.json)。本轮得到的是：固定模型可以在这些局部输入上产生代理约束有效的完整计划，但当前RL512配置未建立对监督/强规则的质量与完整速度优势。不存在本轮支持的规模优势，既定快速局部规划性能目标尚未达到。研究目标保持DRL核心；不自动追加训练、换模型、补缺失或启动新研究。
''')
    markdown('RESULT_TO_CLAIM.md',f'''# S3结果到科研主张

独立角色的原始裁决与审计见[INDEPENDENT_REVIEW.json](INDEPENDENT_REVIEW.json)，不将执行者判断冒充外部人类评审。

支持：13817个未执行请求实际完成且旧账继承；108个输入、15874个返回计划和816657条分配的保存证据成立；12固定模型来源不变；完整质量格5/5未通过、完整部署格9/9未通过原门槛。

不支持：当前DRL优于frozenSL或continuedSL、优于Fixed3/Detour16、某个完整规模达到实用质量—时间优势、全局S3 AND通过、一般DRL无效或真实机械臂轨迹认证。H改善不能归于网络。

缺失：两历史失败槽不补值，未训练4×3质量及其两部署格、已训练6×6冷部署格仍不完整；全局质量/部署/AND保持null。混合输入/持久化版本计时不等价于无开销差异。论文数字审计可用性及provisional状态以独立审查记录为准。

四项直接回答：DRL相对监督未建立优势；相对强规则未建立优势；没有完整窗口规模展示通过原部署门槛的实际优势；当前冻结模型完整请求尚未达到既定快速局部规划目标。对本配置的负结果不得外推为DRL研究目标不可行。

路由：归档实际负结果，固定研究目标不变，不擅自补跑失败、追加S1/S2、挑选中间checkpoint或读取原60。未来实现/科学变化需独立任务及事前协议，不能把本轮工程执行成功改写成性能创新成立。
''')
    markdown('NEXT_STEP_DECISION.md',f'''# 下一步决定

**技术执行GO／已完成原顺序剩余13817请求；科学性能结论REVISE。** 本轮无新的运行阻塞，不再需要恢复任何未执行请求（剩余0）。三seed均已封口，但两历史失败使全科学网格保持缺失。

5完整质量格、9完整部署格均未通过；1质量及3部署格INCOMPLETE，全局AND为null。当前不宣布冻结DRL实用优势，不推荐依据已见结果重新挑checkpoint、场景、A或种子。

不自动追加训练、方法试验或补齐历史失败。原60继续封存，研究目标继续固定为大规模父场景DRL训练—冻结策略快速局部完整规划。若后续获独立任务，可根据保存的加载/核验开销与RL相对同起点的质量退化诊断设计新研究；本报告不授予执行权限，也不将纯监督或规则替代固定研究目标。

完整结果见[实验报告](EXPERIMENT_REPORT.md)、[逐请求数据](results/REQUEST_METRICS.csv)、[六/十二格](results/SCIENTIFIC_ANALYSIS.json)、[独立审查](INDEPENDENT_REVIEW.md)和[累计资源](RESOURCE_ACCOUNTING.json)。Git推送与远端事实记录见[GIT_VERSION_AUDIT](GIT_VERSION_AUDIT.json)及仓库内发布收据；不合并main。
''')
    markdown('README.md',f'''# S3 scope_r1 正式恢复实际交付

**13817/13817新增请求成功，新增失败0；累计15874份有效结果、2个历史缺失。** 三seed按序封账，未追加训练或读取原60。5个完整质量格及9个完整部署格全部未通过原门槛；另外1质量格／3部署格INCOMPLETE，全局AND为null。当前RL没有建立对监督或强规则的质量与时间优势。

- [正式结果与全部表格](EXPERIMENT_REPORT.md)、[结果到主张](RESULT_TO_CLAIM.md)、[下一步决定](NEXT_STEP_DECISION.md)
- [逐请求J/L/K、raw/H成本与完整时间CSV](results/REQUEST_METRICS.csv)、[七方法质量—时间](results/METHOD_QUALITY_TIME.csv)、[三seed质量](results/QUALITY_BY_MODEL_SEED.csv)
- [六个质量格和十二个部署格](results/SCIENTIFIC_ANALYSIS.json)、[匹配比较与父布局区间](results/MATCHED_COMPARISONS.json)、[计时版本](results/TIMING_BY_REVISION.csv)
- [全部原始结果索引与哈希](results/RESULT_ARTIFACT_INDEX.json)、[请求状态与历史失败](results/REQUEST_STATE_INDEX.json)、[只读覆盖/同臂审计](results/CONSTRAINT_SOURCE_AUDIT.json)
- [累计及本轮资源](RESOURCE_ACCOUNTING.json)、[完整模型来源](results/MODEL_SOURCE_AUDIT.json)、[协议/源代码/授权身份](EVIDENCE_IDENTITY.json)
- [独立审查](INDEPENDENT_REVIEW.md)、[技术执行复核](IMPLEMENTATION_REVIEW.md)、[Git版本审计](GIT_VERSION_AUDIT.json)

生产运行完整输入、来源修订、Journal、时钟、封口和13817份新结果在[scope_r1运行目录](../rc3_s3_connectivity_20261109/runs/rc3_v3_global_local_20261106_01/scope_r1/)。模型/NPZ/大Journal遵循Git LFS，请检出提交后执行`git lfs pull`。不生成额外整包备份，不覆写S1-A/S1-B/S2及旧S3失败证据。

评估源码SHA256 `{SOURCE}`，与训练源码身份分开。输入范围为v3+scope_r1，不能称未修订原v3完整网格通过。原阶段精确历史墙钟仍null，592保守预算结转保留；两失败不填补。科研结果限于静态代理约束和本配置，不声称实机IK认证或DRL一般无效。
''')
    print(dict(reports='created',review_verdict=review.get('claim_supported'),new_success=13817))


def metadata():
    a=read(RES/'SCIENTIFIC_ANALYSIS.json');r=read(RES/'RESOURCE_ACCOUNTING.json')
    p=ROOT/'rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json';s=read(p)
    s.update(status='APPROVED_V3_S3_SCOPE_R1_SCHEDULE_DISPOSED_WITH_HISTORICAL_GAPS',
        S3_runtime='ALL_REMAINING_REQUESTS_FINISHED_RETAINED_TWO_MISSING',S3_execution_readiness='NO_UNEXECUTED_REQUESTS_REMAIN',
        S3_target_tasks='ALL108_AUTHORIZED_CASES_MATERIALIZED',S3_target_semantic_consumption=108,S3_unique_materialized_target_cases=108,
        S3_scientific_decision='COMPLETE_CELLS_NOT_PASSED_GLOBAL_GATES_NULL_INCOMPLETE',
        S3_requests_debited=15876,S3_requests_completed=15874,S3_requests_failed=2,S3_requests_unexecuted=0,
        S3_seeds_scheduled=[11622026,11622027,11622028],S3_seeds_completed=[11622026,11622027,11622028],S3_seeds_not_started=[],
        S3_seeds_completed_scope='Request schedule disposed; first seed has2 retained missing slots, not a complete successful scientific grid',
        S3_quality_gate=None,S3_deployment_gate=None,S3_combined_gate=None,
        S3_quality_cells_complete=5,S3_quality_cells_failed=5,S3_quality_cells_incomplete=1,
        S3_deployment_cells_complete=9,S3_deployment_cells_failed=9,S3_deployment_cells_incomplete=3,
        S3_report=label(HOME/'README.md'),S3_current_report=label(HOME/'README.md'),result_report=label(HOME/'README.md'),
        S3_scope_current_report=label(HOME/'README.md'),S3_scope_formal_grid_executed_this_turn=13817,
        S3_scope_new_success=13817,S3_scope_new_failures=0,S3_scope_remaining_requests=[0,0,0],
        S3_scope_remaining_wall_s=[z['remaining_wall_s'] for z in r['actual_by_seed'].values()],
        S3_scope_prior_failure_disposition='Two original failures retained missing; all other originally unexecuted requests completed once',
        S3_scope_current_counter_scope='Current cumulative15876/15874/2; older S3_recovery_* fields preserve prior recovery-turn snapshots',
        S3_counter_snapshot_scope='Current closed original+recovery+scope_r1 journals; all15876 slots disposed,2 retained missing',
        S3_scope_revision_basis='User explicitly authorized exact311a58bd binding and remaining13817 requests; execution receipt preserved',
        S3_scope_measured_new_wall_s=sum(z['measured_new_wall_s'] for z in r['actual_by_seed'].values()),
        S3_scope_charged_wall_by_seed={k:z['charged_wall_s'] for k,z in r['actual_by_seed'].items()},
        S3_scope_exact_original_stage_wall_s=None,S3_scope_old_failed_slots_refilled=False,
        S3_current_status_closed_utc=now(),S1_B_S2_S3_authorized_scope='S1-B/S2 complete; authorized S3 request schedule closed with two historical gaps; no additional training or replacement requests',
        working_tree_execution_sources='All194 scope runtime binding files unchanged; analysis/delivery helpers are separately identified, no training/model/H changes')
    save(p,s)
    p=ROOT/'PROJECT_VERSION.json';v=read(p);v['version']=VERSION;v['current_status_date']='2026-11-09'
    v['S3_scope_r1_formal_results']=dict(status='REQUEST_SCHEDULE_DISPOSED_WITH_TWO_HISTORICAL_GAPS',
        report=label(HOME/'README.md'),baseline=BASE,source_binding_sha256=SOURCE,new_debits=13817,new_success=13817,new_failure=0,
        cumulative_debits=15876,cumulative_success=15874,cumulative_failures=2,unexecuted=0,
        quality_coverage='2267/2268',cold_coverage='6803/6804',hot_coverage='6804/6804',
        quality_cells_failed=5,quality_cells_incomplete=1,deployment_cells_failed=9,deployment_cells_incomplete=3,
        global_quality_pass=None,global_deployment_pass=None,combined_pass=None,
        optimizer_updates=0,original60='SEALED',models_unchanged=12,old_frozen_files_unchanged=2515,
        main_merged=False,branch='codex/global-local-preflight',historical_objects_preserved=True)
    save(p,v)
    p=ROOT/'README.md';txt=p.read_text(encoding='utf8')
    txt=txt.replace('截至 **2026-11-08**','截至 **2026-11-09**',1).replace('2026.11.09-s3-connectivity-target-scope-ready',VERSION,1)
    marker='<!-- S3_SCOPE_RESULTS_CURRENT_BEGIN -->'
    require(marker not in txt,'Current result block already exists')
    block='''<!-- S3_SCOPE_RESULTS_CURRENT_BEGIN -->
**S3 scope_r1已实际完成剩余13817/13817请求，新增失败0；当前配置未建立DRL质量或速度优势。** 累计15876预扣、15874有效计划与2个原历史缺失；三seed封账，108输入和816657条同臂F&A服务分配核验通过，12固定模型及2515旧文件未改。5完整质量格／9完整部署格均未通过，另1质量／3部署格INCOMPLETE，全局AND为null。完整质量格RL+H相对frozenSL成本高1.10%–1.96%、相对Detour高3.72%–7.52%；完整部署格中位耗时为Detour的1.76–2.72倍。H收益单列，不归于学习；不追加训练、补缺失或读原60。详见[本轮全部交付](rc3_s3_scope_results_20261109/README.md)、[实际报告](rc3_s3_scope_results_20261109/EXPERIMENT_REPORT.md)、[逐请求CSV](rc3_s3_scope_results_20261109/results/REQUEST_METRICS.csv)、[独立审查](rc3_s3_scope_results_20261109/INDEPENDENT_REVIEW.md)和[下一步](rc3_s3_scope_results_20261109/NEXT_STEP_DECISION.md)。以下记录保留各历史阶段当时状态，当前以上述结果和活动状态为准。
<!-- S3_SCOPE_RESULTS_CURRENT_END -->

'''
    idx=txt.index('<!-- S3_CONNECTIVITY_CURRENT_BEGIN -->');txt=txt[:idx]+block+txt[idx:];p.write_text(txt,encoding='utf8')
    p=ROOT/'EXPERIMENT_RESULT_TO_CLAIM.md';txt=p.read_text(encoding='utf8');review=read(HOME/'INDEPENDENT_REVIEW.json')
    entry=f'''## 2026-11-09：S3 scope_r1三种子冻结正式结果

- 实际新13817/13817成功、失败0，累计15874有效/15876预扣，两个旧缺失未补值，全部未执行请求已处理。108局部输入、816657条服务分配、固定模型与2515旧证据复核通过。
- 独立reviewer `claim_supported={review.get('claim_supported')}`，`confidence={review.get('confidence')}`，具体审计命题与性能命题区分见[独立审查](rc3_s3_scope_results_20261109/INDEPENDENT_REVIEW.md)。完整质量5格与完整部署9格均未通过；另1质量／3部署格缺失，全局AND为null。
- 当前RL+H未建立相对监督或强规则的质量/完整速度优势，无完整规模通过原部署门槛；H改善属于后处理，不能记为RL学习收益。不外推DRL一般无效，不更改固定研究目标。
- 路由：归档实际负结果、窄化论文主张；不自动补齐失败、追加训练、挑中间模型或读取原60。论文数字审计/provisional边界按独立审查记录，不能冒充完整论文审计。

依据：[实际结果](rc3_s3_scope_results_20261109/EXPERIMENT_REPORT.md)、[结果到主张](rc3_s3_scope_results_20261109/RESULT_TO_CLAIM.md)、[原始数据](rc3_s3_scope_results_20261109/results/REQUEST_METRICS.csv)。

'''
    require('## 2026-11-09：S3 scope_r1三种子冻结正式结果' not in txt,'Claim record already exists')
    p.write_text(txt.replace('# 实验结果到主张记录\n\n','# 实验结果到主张记录\n\n'+entry,1),encoding='utf8')
    print(VERSION)


if __name__=='__main__':
    import sys
    {'reports':reports,'metadata':metadata}[sys.argv[1]]()
