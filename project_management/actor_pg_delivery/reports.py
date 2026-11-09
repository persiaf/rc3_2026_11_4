"""Human-readable report and mutable project metadata from completed saved runs."""
import csv
import json
from rc3_actor_pg_20261109.common import *

VERSION='2026.11.09-actor-deployment-matched-pg-screen'


def save(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def doc(name,text):
    p=HOME/name;require(not p.exists(),'Do not overwrite report '+name);p.write_text(text,encoding='utf8')
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def csv_rows(name):
    with (HOME/'results'/name).open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def pct(x):return f'{100*x:.4f}%'


def reports():
    summary=read(HOME/'results/SUMMARY.json');resource=read(HOME/'RESOURCE_ACCOUNTING.json')
    review=read(HOME/'INDEPENDENT_REVIEW.json');models=csv_rows('DEV_MODEL_SUMMARY.csv');timing=csv_rows('DEPLOYMENT_TIMING.csv')
    curves=read(HOME/'results/GRADIENT_AND_WEIGHT_SUMMARY.json');comp=read(HOME/'results/MECHANISM_COMPARISONS.json')
    lookup={(r['method'],r['mode'],r['navigation']):r for r in timing};dtable=[]
    for cache in ('cold','hot'):
        for mode in ('D0','D1','D2'):
            r=lookup['RL_plus_H',mode,cache];d0=lookup['RL_plus_H','D0',cache]
            ratio=float(d0['median_s'])/float(r['median_s'])
            dtable.append([cache,mode,r['n'],f"{float(r['median_s'])*1000:.3f}",f"{float(r['p95_s'])*1000:.3f}",f'{ratio:.4f}',pct(1-float(r['median_s'])/float(d0['median_s']))])
    st=[[m,f"{r['process_to_ready_s']:.6f}",r['resident_models']] for m,r in summary['deployment_startup'].items()]
    mt=[[r['model'],r['n'],*[f"{float(r[k]):.6f}" for k in ('raw_J','raw_L','raw_K','J','L','K','H_gain_J')]] for r in models]
    ct=[[c['metric'],c['control'],pct(c['mean']),f"[{pct(c['lower95'])}, {pct(c['upper95'])}]",'/'.join(pct(v) for v in c['parent_gains']),
         '正向raw信号' if c['metric']=='raw_J' and c['exploratory_positive'] else ('未通过postH筛选' if c['metric']=='J' else '区间跨零')] for c in comp['comparisons']]
    ptable=[[p,v['status'],f"{v['wall_s']:.6f}",v['cap_s']] for p,v in resource['phase_resources'].items()]
    grad=[[r['arm'],r['updates'],f"{r['gradient_norm_min']:.4f}",f"{r['gradient_norm_mean']:.4f}",f"{r['gradient_norm_max']:.4f}",r['clipped'],r['zero_signal'],r['nonfinite']] for r in curves['training']]
    source=digest(HOME/'SOURCE_BINDING.json')
    doc('EXPERIMENT_REPORT.md',f'''# 推理制品提速与匹配策略梯度小试实际结果

**部署改造有效且质量精确保持；本次B128未通过预登记的完整H后收益筛选。** 全部756部署请求、A/B各128更新及1536训练终局请求、54固定DEV请求实际完成，无科学失败。部署与机制两个额度分别使用，没有沿用旧S3余额或补齐旧失败。

## 固定目标、来源与执行

研究目标仍是大规模复杂父场景DRL训练、冻结策略快速生成满足代理约束的不同规模局部完整路径。监督只提供共同SL256初始化，规则和H只作对照/固定后处理。本轮是研发机制筛选，不是S3确认。

基准`{BASE}`，独立实验`{EXPERIMENT}`，分支`codex/actor-pg-screen-20261109`。准确新源码绑定`{source}`。协议、请求顺序、512共同实例声明、预算和停止规则均在首次科学请求前登记；用户本条任务同时明确授权实施和有界实验，不冒充其事先读过生成的SHA。见[协议](PROTOCOL.json)、[授权](AUTHORIZATION.json)、[实施GO](IMPLEMENTATION_REVIEW.md)和[执行验收收据](EXECUTION_ACCEPTANCE.json)。原v3和旧S3活动状态原字节保留。

{table(['阶段','实际状态','累计阶段秒','秒上限'],ptable)}

阶段时间包含子进程启动/import、来源核验、准备、导航/H、验证、日志、梯度及checkpoint IO。包含/排他组件与外层墙钟不能相加。注册、零科学单测、结果分析、独立审计和Git维护另账，不伪称零耗时。工程8测试PASS，真实工程optimizer和GlobalLocalPolicy推理均0。

## 十二个actor-only制品

三个seed的untrained、SL256、continuedSL512、RL512均导出。每个25个参数张量，合计300个name/shape/dtype/value及tensor hash逐项完全一致；没有量化、剪枝、蒸馏或网络变化。制品只含参数、原hidden64及输入必要架构、来源hash。

每个actor为268504字节。seed11622026的RL512完整checkpoint为37950130字节，actor约为其1/141.339，减少99.2925%；其他角色和seed逐项见[ACTOR_SIZES.csv](results/ACTOR_SIZES.csv)。这去除了训练Adam/RNG/训练cache等负载，不是减少网络参数。完整checkpoint及其来源文件不写。

全部可用制品见[ACTOR_INDEX.json](actors/ACTOR_INDEX.json)，每制品旁的.pt.json保留25项校验。另存A128/B128新固定终点的精确推理制品，不能用合成坏文件夹具充当科学模型。模型谱系见[WEIGHT_LINEAGE.json](WEIGHT_LINEAGE.json)。

## D0、D1、D2完整请求速度与启动

固定DEV_000—DEV_005，覆盖4×3/6×6/8×8及闭合/开放端点；它们来自同一个旧DEV父布局。固定seed11622026、七方法、三加载模式、两导航状态、每格三重复，共756请求。D0/D1/D2各为独立进程；D2启动一次加载4个不同actor角色，RL_raw和RL_plus_H共享同一RL512参数。

下表RL_plus_H每格18次观察（6输入×3重复），单位ms；完整请求保留模型访问、解析/核验、导航、特征、构造、完整H、导出、验证和持久化。倍率为D0中位数/当前中位数。

{table(['导航','模式','n','完整中位ms','完整p95 ms','中位提速倍数','中位时间减少'],dtable)}

{table(['模式','进程启动至ready秒','常驻actor数'],st)}

启动span从父进程启动子Python之前计至目录/模型准备完毕，包含import和准入。热模式只预热导航，预热3次实际导航也计入部署阶段。D2不免费复用任务编码或最终路线。启动与首个请求各自测量span见[STARTUP_AND_FIRST_RESPONSE](results/STARTUP_AND_FIRST_RESPONSE.json)；两span之和不包含中间receipt/debit IO，不能冒称精确进程到首响应时长，中间IO仍在阶段账。

较大的中位差出现在D0改成小制品D1时：RL+H冷/热中位减少47.71%/53.05%；D2相对D0减少48.03%/57.91%。D2相对D1的冷中位仅略减，且冷p95为834.0ms，高于D1的754.0ms；不能声称常驻在所有尾延迟上均优于逐请求actor加载。

原生方法没有模型加载，但其尾延迟也随顺序变化：Fixed3热p95在D0/D1/D2为733.21/392.18/410.96ms，Detour16冷p95为1206.40/880.63/803.31ms。因此本机观察到的差异包含系统/顺序/前序内存状态等未分离影响，尤其不能把全部p95差值因果归于actor加载。模型制品确实缩小且路线精确相同；严格隔离加载生命周期的因果幅度仍需另行授权的随机交叉复核，本轮不增加请求。

RL_raw同样有实际提速：D0冷/热中位1322.04/1123.17ms，D1为696.61/487.90ms，D2为689.99/445.24ms。七方法全部median/p95见[DEPLOYMENT_TIMING.csv](results/DEPLOYMENT_TIMING.csv)，全756行见[DEPLOYMENT_REQUESTS.csv](results/DEPLOYMENT_REQUESTS.csv)。模式顺序事前固定D0→D1→D2，没有随机交叉/负载追踪；这些是本机六例的观察结果，不是跨硬件或S3部署判定。

## 质量是否完全保持

**是：本轮756/756请求通过精确等价检查。** 同case/method的raw/final路线、J/L/K（含raw J/L/K）、view/D身份及全部服务分配在三模式、冷热和重复间一致，覆盖42个case/method组合；RL_raw/RL_plus_H共108个匹配对的原始路线完全相同。540次神经请求都有新的编码，D2仅有180次参数驻留命中，没有跨任务动态编码缓存。

各方法共同H次数符合定义：部署432次完整H，两个原生规则方法和RL_raw均不额外加共同H。所有计划保留原导航、覆盖、同臂F&A、端点和独立代理验证，原native端点适配拒绝隐藏回退。见[等价检查](runs/DEPLOYMENT/EQUIVALENCE.json)和独立审查。十二个制品均参数等价，但运行质量/时间实验只覆盖第一seed六旧DEV例，不能改写旧S3结论。

## A128与B128：真实DRL更新

两组从seed11622026同一SL256 tensor `1817f58aa6506456ff12f0c5fe0a914e156aef8af319f4687dde90782f55a63e`开始，各自fresh Adam，不继承S1动量；每组64全父批次与64局部批次、batch4、同512个输入view身份。A按原primary/forced-low/high、1/.25/.25；B三条独立策略采样、所有首动作计logprob、等权detached LOO。没有PPO、熵/KL项、优势归一、学习率/奖励/网络变化。

{table(['组','实际更新','clip前norm最小','均值','最大','裁剪次数','零信号批','非有限'],grad)}

两组每次更新均发生有效参数变化并保存完整动作、mask、logprob、奖励、baseline、advantage、完整clip前后梯度和结果参数；0/32/64/96/128保存完整checkpoint、Adam与RNG。A共1024个强制首动作未计首logprob，B的1536个轨迹首动作均由策略采样并计入。所有3072训练终局请求实际执行H、导出和验证，缓存命中0、失败0。

128更新后的参数L2相对变化A约1.3561%、B约1.2803%，仅说明更新真实发生，不代表规划性能。完整训练曲线见[TRAINING_CURVES.csv](results/TRAINING_CURVES.csv)、[梯度/权重摘要](results/GRADIENT_AND_WEIGHT_SUMMARY.json)；逐更新原始记录与梯度在[runs/A128](runs/A128/)和[runs/B128](runs/B128/)。没有按DEV或训练loss选中间权重。

## 固定终点18 DEV的raw/H完整路线

共同SL256、A128、B128各18次单次贪心+完整H，共54次。先冻结终点，再进行共同评价，不使用S3或原60。下表是等权18案例的绝对成本均值。

{table(['模型','n','raw J','raw L','raw K','H后J','H后L','H后K','H绝对改善'],mt)}

B的H后绝对平均J较A低0.13006%、较SL低0.18698%。这点必须报告，不能写成绝对平均J也劣化。但是，事前主统计是先在每个真实父布局平均其6个案例，再算(control−B)/control，最后对3个父布局等权平均。它与全体绝对成本均值的相对变化不同，后者隐含更高成本父布局更大权重；不得看到结果后切换筛选口径。

{table(['指标','B相对对照','等权父相对收益','95%父簇区间','三个父布局收益','解释'],ct)}

原定postH两对照AND筛选结果：**REVISE**。B对A的raw收益7.9989%（三父布局均正），是有价值的原始策略行为信号；B对共同SL的raw收益1.2686%区间跨零。H后，B对A为−0.1927%、对SL为−0.0561%，区间都跨零，没有建立稳定的完整交付成本改善。不能只因B raw比退化的A更好就宣布DRL改善共同起点，也不能用极小绝对均值下降替代预登记判定。

各案例全J/L/K、原始/最终路线hash、来源和完整时间见[DEV54_RESULTS.csv](results/DEV54_RESULTS.csv)；所有匹配差异见[DEV_MATCHED_PER_CASE.csv](results/DEV_MATCHED_PER_CASE.csv)，三尺寸/两端点的全部结果见[DEV_STRATA.csv](results/DEV_STRATA.csv)。没有删例、补抽、改A或增加更新。

区间按3个真实DEV父布局重采样20000次，PCG64 seed11632026；18例不是18个独立环境，只有一个优化seed，区间不包含优化seed不确定性，不能当作S3确认或128步足以收敛的证据。

## 现有证据排除了什么，仍未确定什么

已澄清：原完整训练checkpoint的逐请求加载/身份核验负担，是可大幅减少的端到端开销；同权重的制品/驻留改变没有改变本轮规划质量。因此不能将旧时间全部归因于网络算子，也不能用提速声称神经策略质量提高。

本次指定的“去强制首动作+等权三采样”128更新配方，未产生按预登记口径可确认的H后改进；不能将“只要移除强制首动作就一定获得稳定交付收益”写成结论。

仍不能单独归因：B同时改了首动作采样与混合权重，未分离二者；一个seed/128步/三个旧DEV布局不能排除更长训练的潜力、随机种子变化、表示/信用分配/样本效率问题，或H造成的raw与最终排名差异。全部更新发生裁剪是观察事实，不是裁剪导致负结果的因果证明。没有证据否定DRL一般能力，也不允许据此将固定目标改为纯监督或规则。

## 完整资源与独立审查

部署756请求；机制3072训练+54 DEV=3126请求，两个独立额度合计3882，未互借。实际新导航435次（部署cold378+hot预热3+DEV54），训练仅加载已保存导航1002次、内存命中22次；native216，完整共同H3558，optimizer256，teacher0，任务/FA新生成0，原60读取0。内部搜索、验证、缓存、checkpoint和IO见[RESOURCE_ACCOUNTING](RESOURCE_ACCOUNTING.json)。维护和只读审计不是新的训练或性能试验。

独立审查见[INDEPENDENT_REVIEW.md](INDEPENDENT_REVIEW.md)及完整trace；科研主张见[RESULT_TO_CLAIM](RESULT_TO_CLAIM.md)。这些有效性仅限代理约束，不是实际机械臂IK/关节轨迹认证。全部旧S0—S3代码、模型、结果和两个历史缺失槽保持，旧S3结论不改。后续判断为**保留部署改造GO；机制配方REVISE**，不自动扩训、追加探针或读取保留测试。
''')
    doc('RESULT_TO_CLAIM.md',f'''# 本轮结果到主张

独立裁决见[INDEPENDENT_REVIEW.json](INDEPENDENT_REVIEW.json)，执行者不代替reviewer判定。

支持的窄结论：三个seed十二actor逐参数完全相等；第一seed六旧DEV输入的756请求精确保持质量。RL+H冷/热中位从D0的1342.77/1104.74ms降为D1的702.11/518.72ms、D2的697.89/465.00ms；启动成本单列，D2冷p95不优于D1。属于加载生命周期工程收益，不是DRL学习收益或旧S3门槛翻转。

机制事实：两组从同一SL256分别freshAdam，各128真实更新和1536实际终局评价，无免费cache/失败替换。B raw相对A的父布局平均收益7.9989%，但相对SL raw区间跨零。B postH绝对平均J156.3909低于SL156.6838和A156.5945；预登记等权父相对收益对A−0.1927%、对SL−0.0561%，区间均跨零，双对照筛选REVISE。

不支持：B已稳定改善共同SL起点；移除forcing已成为充分补救；当前DRL已优于强规则、跨布局泛化或满足S3部署AND；128更新已收敛；所有NN时间来自算子；实际机器人运动学/关节安全认证。

边界：一个优化seed、三个已使用DEV父布局，B联合改变forcing与权重；模式顺序固定，六部署输入仅一个父布局。未做新增S3测试，旧60封存；不以全体绝对均值替换预登记父相对指标。论文数字全审计可用性/provisional状态按独立审查记录。

路由：确认窄部署工程结果，保留机制负结果和raw行为信号；固定DRL研究目标不变，不自动加轮数/换seed/换奖励/换网络或引入PPO。后续先利用已保存日志分析raw到H后及训练采样到贪心终点差异，任何新科学试验另立协议和任务。
''')
    doc('NEXT_STEP_DECISION.md','''# 下一步决定

**部署制品工程GO；当前匹配机制筛选REVISE。任务规定的有界实验均已执行完成，没有未执行请求。**

D1/D2可作为之后明确授权评价的推理实现候选。它们在当前六个旧DEV输入保持精确质量并实测提速，但不回写旧S3时间、不改变旧S3负结果。D2有启动成本，且本轮冷p95高于D1，不声称常驻全面支配逐请求actor加载。

B128原始路线相对A有正向信号，完整H后未通过B对A及共同SL的预登记AND筛选。绝对均值微降和父相对收益方向不同，二者均公开；不以少退化、训练loss或已见结果重新选择统计量/模型。

下一项若获授权，应先只读分析保存的逐轨迹/梯度与raw/H映射，再形成能分离forcing与权重、并检验实际H后收益的新DRL机制协议。当前不自动增加训练、加入新方法、重测S3或读原60。DRL大规模父训练—冻结策略局部完整规划的目标固定，不转为纯监督或规则。

全部数据和来源见[实际报告](EXPERIMENT_REPORT.md)、[资源账](RESOURCE_ACCOUNTING.json)、[模型谱系](WEIGHT_LINEAGE.json)和[独立审查](INDEPENDENT_REVIEW.md)。Git远端及LFS核对在[GIT_VERSION_AUDIT](GIT_VERSION_AUDIT.json)与仓库内发布收据；不合并main。
''')
    doc('README.md',f'''# actor-only部署与匹配策略梯度小试

**已实际完成：756部署请求、A/B各128更新、3072训练终局请求和54固定DEV评价。** 十二actor逐参数精确相等，部署756请求raw/final路线与J/L/K完全保持。RL+H D2相对完整checkpoint的冷/热中位提速1.924/2.376倍，启动成本另列。B128在H后未通过预登记双对照筛选，不能宣布稳定改善SL起点；极小绝对均值下降也完整公开。

- [实际报告和四项回答](EXPERIMENT_REPORT.md)、[结果到主张](RESULT_TO_CLAIM.md)、[下一步](NEXT_STEP_DECISION.md)
- [12推理制品及来源](actors/ACTOR_INDEX.json)、[每模型制品大小](results/ACTOR_SIZES.csv)、[全部权重谱系](WEIGHT_LINEAGE.json)
- [756请求](results/DEPLOYMENT_REQUESTS.csv)、[七方法42格完整median/p95](results/DEPLOYMENT_TIMING.csv)、[提速与削减比例](results/DEPLOYMENT_SPEEDUP.csv)、[启动记录](results/STARTUP_AND_FIRST_RESPONSE.json)
- [18×3完整raw/H成本](results/DEV54_RESULTS.csv)、[匹配逐案例收益](results/DEV_MATCHED_PER_CASE.csv)、[绝对均值](results/DEV_MODEL_SUMMARY.csv)、[预登记父布局相对收益与区间](results/MECHANISM_COMPARISONS.json)
- [训练曲线](results/TRAINING_CURVES.csv)、[完整A记录](runs/A128/)、[完整B记录](runs/B128/)、[梯度与参数变化](results/GRADIENT_AND_WEIGHT_SUMMARY.json)
- [协议](PROTOCOL.json)、[用户授权范围](AUTHORIZATION.json)、[执行源码](SOURCE_BINDING.json)、[实例日程](INPUT_REGISTRY.json)、[零科学工程验收](ENGINEERING_VERIFICATION.json)
- [全部资源](RESOURCE_ACCOUNTING.json)、[运行环境](ENVIRONMENT_RECORD.json)、[独立审查](INDEPENDENT_REVIEW.md)、[Git记录](GIT_VERSION_AUDIT.json)

制品可通过`rc3_actor_pg_20261109.artifacts.load_actor`校验后加载；`Catalog('D2', models, seed=...)`仅驻留该seed四个actor，不保存编码。完整科学入口`python -B -X utf8 -m rc3_actor_pg_20261109.run --execute`已消费本轮固定日程；重复运行只识别已完成阶段，不提供额外免费请求。不要把合成负测夹具当模型。

独立实验`{EXPERIMENT}`，基准`{BASE}`，源码SHA256`{source}`。请检出Git提交后`git lfs pull`取得.pt等对象。S0—S3及旧失败原字节保留、旧v3指针不改、原60不读，不新增整包备份，不合并main。
''')
    write_new(HOME/'STATUS.json',dict(experiment_id=EXPERIMENT,status='ALL_BOUNDED_PHASES_COMPLETE',
        deployment='GO_MEASURED_SPEED_AND_EXACT_QUALITY',mechanism='REVISE_NO_REGISTERED_POSTH_INCREMENT',
        scientific_requests=3882,optimizer_updates=256,old_S3_changed=False,old60='SEALED',
        reviewer_claim_supported=review.get('claim_supported'),next_step='No automatic new science; preserve fixed DRL research goal'))
    print('Actual reports saved')


def metadata():
    review=read(HOME/'INDEPENDENT_REVIEW.json')
    p=ROOT/'PROJECT_VERSION.json';v=read(p);v['version']=VERSION
    v['actor_deployment_matched_pg_screen']=dict(status='COMPLETE_DEPLOYMENT_GO_MECHANISM_REVISE',
        experiment_id=EXPERIMENT,baseline=BASE,branch='codex/actor-pg-screen-20261109',source_binding_sha256=digest(HOME/'SOURCE_BINDING.json'),
        report=label(HOME/'README.md'),deployment_requests=756,exact_quality_equivalence=True,actor_exports=12,
        RL_plus_H_D2_speedup_cold=1.9240417825380483,RL_plus_H_D2_speedup_hot=2.3757626816161515,
        A_updates=128,B_updates=128,training_terminal=3072,development_terminal=54,mechanism_total=3126,
        B_postH_parent_gain_vs_A=-.001927368060246723,B_postH_parent_gain_vs_SL=-.0005612725774716351,
        absolute_J_SL=156.68383188345558,absolute_J_A=156.59453580882226,absolute_J_B=156.39086788831113,
        registered_screen='REVISE',old_scientific_results_modified=False,old_budgets_reused=False,old60='SEALED',main_merged=False)
    save(p,v)
    p=ROOT/'README.md';text=p.read_text(encoding='utf8');marker='<!-- ACTOR_PG_SCREEN_CURRENT_BEGIN -->';require(marker not in text,'Already updated')
    block='''<!-- ACTOR_PG_SCREEN_CURRENT_BEGIN -->
**独立actor-only部署/匹配PG小试已实际完成：部署工程GO，机制筛选REVISE。** 756部署请求保持raw/final路线及J/L/K精确一致，12制品不改网络/参数；RL+H D2冷/热中位697.89/465.00ms，相对D0提速1.924/2.376倍，启动成本另报。两组各128真实更新、共3072训练终局及54固定DEV请求。B绝对平均J156.3909略低于SL156.6838/A156.5945，但预登记等权父相对收益对SL−0.0561%、对A−0.1927%，区间跨零，未建立稳定H后增量。B raw对A改善约8%不能充当改善共同起点或完整交付的证据。旧S0—S3、两失败槽与旧结论保持原字节；不使用旧余额、不读原60、不合并main。见[实际结果与全部交付](rc3_actor_pg_20261109/README.md)、[报告](rc3_actor_pg_20261109/EXPERIMENT_REPORT.md)、[独立审查](rc3_actor_pg_20261109/INDEPENDENT_REVIEW.md)。下列S3及更早记录保留其当时状态，不因本次加载优化回写。
<!-- ACTOR_PG_SCREEN_CURRENT_END -->

'''
    idx=text.index('<!-- S3_SCOPE_RESULTS_CURRENT_BEGIN -->');text=text[:idx]+block+text[idx:]
    text=text.replace('2026.11.09-s3-scope-r1-formal-results',VERSION,1);p.write_text(text,encoding='utf8')
    p=ROOT/'EXPERIMENT_RESULT_TO_CLAIM.md';text=p.read_text(encoding='utf8')
    entry=f'''## 2026-11-09：actor-only部署与匹配PG独立小试

- 完成756部署请求、A/B各128更新和3072训练+54DEV终局；十二actor参数精确相等、756 raw/final/JLK等价；D2 RL+H冷/热中位相对D0提速1.924/2.376倍，启动与尾延迟另列，不改旧S3。
- B postH绝对均值低于A/SL，但预登记等权父相对收益对A−0.1927%、对SL−0.0561%，95%区间均跨零，双对照筛选REVISE。raw对A正信号不代表稳定改善SL或H后质量。
- 独立审查claim_supported={review.get('claim_supported')}，confidence={review.get('confidence')}，详见[独立审查](rc3_actor_pg_20261109/INDEPENDENT_REVIEW.md)。窄工程提速与DRL科学收益分开，完整论文claim audit/provisional边界保留。
- 路由：确认部署工程、保留机制不足及raw信号，固定DRL目标不变；不自动扩训/更换方法/补旧失败或读原60。

依据：[实际报告](rc3_actor_pg_20261109/EXPERIMENT_REPORT.md)、[全部保存数据](rc3_actor_pg_20261109/results/)、[下一步](rc3_actor_pg_20261109/NEXT_STEP_DECISION.md)。

'''
    require('## 2026-11-09：actor-only部署与匹配PG独立小试' not in text,'Already recorded')
    p.write_text(text.replace('# 实验结果到主张记录\n\n','# 实验结果到主张记录\n\n'+entry,1),encoding='utf8')
    print(VERSION)


if __name__=='__main__':
    import sys
    {'reports':reports,'metadata':metadata}[sys.argv[1]]()
