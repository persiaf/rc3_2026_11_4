"""Finalize reviewed claims and additive accounting; never execute science."""
import csv
from collections import defaultdict
from rc3_integrated_rppo.common import *
from rc3_integrated_rppo.report import derived


def text_file(name,text):derived(HOME/name,(text.rstrip()+'\n').encode('utf8'))


def main():
    require((HOME/'POSTPROCESS_COMPLETE.json').exists(),'Wait for actual postprocess verification')
    require(read(HOME/'CLAIM_REVIEW.json')['claim_supported']=='no','This closeout text expects observed no-claim result')
    require(not (HOME/'FINAL_REPORT_COMPLETE.json').exists(),'Reviewed report already closed')
    write_new(HOME/'FINAL_REPORT_SOURCE_BINDING.json',dict(source_file=label(Path(__file__)),source_sha256=digest(Path(__file__)),
        scientific_source_sha256=digest(source_path()),protocol_sha256=digest(protocol_path()),new_scientific_requests=0,
        scope='Derived reviewed reports, exact loaded result statistics and final resource carry; no model/solver/H/optimizer'))
    log=Log();clock=Clock(log,'confirmation_and_analysis',10800)
    try:
        summary=read(HOME/'RESULT_SUMMARY.json');res=read(HOME/'RESOURCE_ACCOUNTING_FINAL.json');review=read(HOME/'CLAIM_REVIEW.json')
        decision=read(OUT/'seeds/11622026/SELECTION_DECISION.json');done=read(OUT/'seeds/11622026/main/COMPLETE.json')
        with (HOME/'QUALITY_BY_PARENT.csv').open(encoding='utf-8-sig') as f:parent=list(csv.DictReader(f))
        means={}
        for role in ('SL256','residual_SL','RL','Fixed3','Detour16'):
            rr=[x for x in parent if x['role']==role];means[role]={k:float(np.mean([float(r[k]) for r in rr])) for k in ('J','L','K','raw_J','raw_L','raw_K')}
        lines=['# Integrated-RPPO-v1-small256 实际实验报告','',
            '**工程实现 GO；科学判定 REVISE：P2 未通过，当前固定配置收尾，研究目标尚未达到。**','',
            '固定目标仍为大规模复杂父场景DRL训练—冻结策略快速完成局部完整约束规划。实测发生在单种子开发筛选；没有将监督或规则替代研究目标。','',
            '## 执行版本与范围','',
            f"- 活动协议：`{label(protocol_path())}`，SHA256 `{digest(protocol_path())}`。",
            f"- 科学源码：`{label(source_path())}`，SHA256 `{digest(source_path())}`。",
            '- 原512方案及全部P0报告原字节保留。CPU／GPU完整轮保守投影超出14400秒后，按任务书第7节在首次科学请求前一次缩为所有seed256轮、候选128／256；原日程前256轮、阈值和其他数据划分保持。不能据此推论512轮后期收益。',
            '- 残差头CUDA FP32训练，冻结骨干／Critic和D1推理在CPU；不使用TF32、量化或编译。P0初始零残差logits／mask精确一致及CPU/CUDA梯度容差通过；不声称训练后跨设备贪心路线逐位一致。','',
            '## 实际训练与权重','',
            f"seed11622026完成{done['rounds']}轮、6144条自然采样episode：3072条全父10×10、4×3／6×6／8×8各1024条，覆盖96个实际物理布局与768个父任务变体。",
            f"实际Actor Adam接受{done['actor_accepted_steps']}步、匹配残差SL {done['matched_SL_steps']}步、Critic {done['critic_steps']}步（预热236步）。阶段精确墙钟{res['wall_s']['seed_11622026']:.6f}秒，上限14400秒。",
            '骨干65088参数冻结，残差17217参数可训练；Critic独立，不打包部署。Actor文件包含完整推理参数，恢复Checkpoint保存残差／监督／Critic、Adam、RNG、游标，并绑定精确外部冻结骨干。','',
            '依据同一开发选择规则独立选中RL128和残差SL256。选中RL为512个Actor步，选中监督为1024步；两条完整训练流最大预算／参数／候选机会匹配，不能称选中步数或FLOPs相等。RL256及全部固定端点仍完整保留。',
            '详见[Actor索引](ACTOR_INDEX.json)、[权重谱系](WEIGHT_LINEAGE.json)、[精确更新数映射](MODEL_ENDPOINT_IDENTITY_MAP.json)和[可部署模型清单](DEPLOYMENT_PACKAGE_MANIFEST.json)。','',
            '## 选择集实际质量','',
            '48例来自8个训练过的真实物理父布局，每布局三尺寸×两端点，mixed／blocked各4个父布局。先平均各父布局六例成本，再计算相对收益；不是48个独立环境。区间为20000次分层父布局bootstrap。选择和汇总使用同一开发集，区间属于描述性统计，不能充作独立确认。','',
            '| 方法（选中端点） | 平均J | 平均L | 平均K | raw J |', '| --- | ---: | ---: | ---: | ---: |']
        for role in ('SL256','residual_SL','RL','Fixed3','Detour16'):
            m=means[role];lines.append(f"| {role} | {m['J']:.6f} | {m['L']:.6f} | {m['K']:.3f} | {m['raw_J']:.6f} |")
        lines+=['','| RL128相对参照 | 父布局平均收益 | 描述性95%区间 |','| --- | ---: | --- |']
        for role,e in summary['effects'].items():lines.append(f"| {role} | {100*e['mean']:+.6f}% | [{100*e['CI95'][0]:+.6f}%, {100*e['CI95'][1]:+.6f}%] |")
        lines+=['',
            '神经比较使用同一冻结H。Fixed3／Detour16保持原生端点适配与内部搜索流程，未额外添加共同H；规则的raw字段也是其原生返回成本。不能将神经raw路线与原生返回路线的差值解释为纯网络能力的消融。',
            'P2要求RL对自身SL与独立选中的匹配监督平均收益均至少0.5%；实际+0.223271%、−0.868168%，均未通过。没有额外增加Top1或强规则前置门槛。',
            '同更新数补充也未显示优势：128轮各512 Adam步时RL相对监督为−0.231194%；256轮各1024步时为−1.086436%。这些仅复用已有开发数据，不改变选择规则。','',
            '## raw、H与耗时的结论边界','',
            '选中RL的raw-J相对SL为−0.094853%，相对选中监督为−2.295364%；小幅正向平均变化只见于共同H之后，未构成可靠DRL增量。绝对H降本不得归因于网络。逐例算术分解见[raw／H对照](RAW_AND_H_CONTRASTS.csv)和[路线变化与H分解](ROUTE_CHANGE_AND_H_ATTRIBUTION.csv)。相对SL，48例raw路线改变6例、H后改变5例。',
            '完整开发请求服务／审计时间保存在ALL_ROUTE_RESULTS.csv，包含实际准备、D1载入、动态特征、构造、H、导出验证与规定IO。开发质量请求的顺序／缓存状态不是最终配对冷／热计时，不能从这些时间宣布部署通过。',
            'P2停止后，另外两个seed、独立晋级48、最终216、冷／热三重复计时和P5均为NOT_STARTED。时间及最终网格空表／状态文件表示未执行，不表示0耗时、失败或通过。原60和旧S3两缺失未消费／补跑。','',
            '## 资源、异常及审查','',
            f"累计{res['paid_requests']}次预扣、{res['successes']}次成功、{res['failures']}次请求失败，新增基础设施付费重试{res['infrastructure_retries']}次；H共{res['full_H_calls']}次，原生规则96次。P0为66完整请求、32工程optimizer调用，工程权重未晋升。",
            '首次导入错误保留原日志，以明确标记的60秒保守启动费用结转；CPU／GPU资源投影原报告和一次缩减收据保留。中途Git/LFS只读diff孤儿进程造成内存压力，已限定清理，全部慢轮耗时保留，没有退款、换seed、重抽、恢复计数清零或事后改门槛。',
            '缓存的历史进程合计按冻结LRU／真实调用序列作显式追溯重建，并与P0事前记录及R03实测计数逐项一致；不是伪称当时已完整仪表记录。见CACHE_ACCOUNTING_RECONSTRUCTION.json。新task/F&A生成17，导航求解0，教师查询0；旧银行仅缺一条按原种子在新目录补齐，原输入和教师不覆盖。',
            '独立委派角色已从6834个对象、432条CSV、212份源码、14份绑定及7份Actor复核，未发现科学数值／来源不一致；结论claim_supported=no。这不是外部人类同行评审，PAPER_CLAIM_AUDIT缺失，论文写作判定保持provisional。',
            '已完成57764个旧文件字节核对，未发现未声明变化。离线约束与动作奖励复核见POSTPROCESS_COMPLETE.json；数值后处理资源见RESOURCE_ACCOUNTING_FINAL.json，追加缓存汇总和报告收尾见RESOURCE_ACCOUNTING_CLOSEOUT.json及FINAL_REPORT_COMPLETE.json。代理验证不等于真实IK、关节轨迹或机器人实机安全认证。','',
            '## 结论','',
            '**本配置已实际训练并交付可运行冻结Actor，但未达到预登记的独立DRL增量和快速规划研究目标。按P2收尾，不进入后续条件阶段，不追加旧实验。**',
            '图表：[训练诊断](TRAINING_DIAGNOSTICS.png) · [开发对比](QUALITY_CONTRASTS.png)；逐例、参数、日志和完整恢复证据均保留。']
        text_file('EXPERIMENT_REPORT.md','\n'.join(lines))
        text_file('RESULT_TO_CLAIM.md','''# 结果到主张

**claim_supported=no；confidence=high；工程GO／科学REVISE。**

支持真实PPO训练、合法完整代理路线、可运行冻结Actor及本轮P2停止。所选RL128相对SL的H后收益+0.223271%，95%描述区间[-0.034122%,+0.556194%]；对选中监督为−0.868168%。raw对SL为−0.094853%。这些来自8个训练布局的开发选择，不支持可靠独立DRL增量、三seed复现、未见环境或快速部署通过。

1024个已接受Actor步、梯度／损失、P0检查和H降本不替代规划性能。选中RL512步、选中监督1024步；同轮512／1024步补充对照也未显正点收益。强规则开发质量更好；最终匹配计时未启动，联合部署门槛未评价。

其余seed、独立晋级、最终216、三重复冷／热和P5为NOT_STARTED。当前固定配置保留负结果收尾，研究目标不变，不在本任务追加科学实验。完整判断见INDEPENDENT_REVIEW.md／CLAIM_REVIEW.json与paper/review-traces；该独立委派不等于外部人类评审，论文主张审计缺失，写作仍provisional。
''')
        text_file('findings.md','''# 研究发现

实验rc3_integrated_rppo_v1_20261110_01，活动版本small256：可靠DRL增量主张未获支持。数值稳定和全部代理路线有效，选择结果RL128对SL仅+0.2233%、对残差SL256为−0.8682%；raw未改善SL。选中模型来自已见物理布局的开发数据，不能升格独立泛化。KL变化很小是诊断观察，不能在无因果消融时认定唯一失败根因。

保留本次源码／固定端点／完整负结果，按原P2停止；不借剩余额度扩训，不改变DRL核心目标。未来任何正主张仍需真实增量及独立确认，这不是新实验授权。路由narrow_claim_and_close_predeclared_P2，claim_supported=no，confidence=high，论文写作provisional。
''')
        text_file('NEXT_STEP_DECISION.md','''# 下一步决定

**工程GO；科研REVISE：本配置按P2收尾，目标未达到。**

已完成seed11622026固定256轮、1024 PPO Actor／1024监督／1260 Critic更新。独立选择RL128与监督256，前两项0.5%主门槛均未通过。完整计划按事前资源条款一次缩减，512轮后期未评价。

另外两seed、晋级、最终216、冷／热正式计时、P5均不启动。保存可运行Actor和所有固定端点；不追加训练、不改变研究定位、不将监督或规则替代DRL目标。原60、旧S3缺失与旧成果保持。GitHub独立研究分支codex/rc3-integrated-rppo-v1；实际同步以GIT_VERSION_AUDIT／PUBLICATION_RECEIPT的远端核验为准，不合并main。
''')
        state=dict(experiment=EXP,technical='GO',scientific='REVISE_P2_NOT_PASSED',goal_attained=False,claim_supported='no',
            first_seed='COMPLETE_256_ROUNDS',other_seeds='NOT_STARTED',promotion='NOT_STARTED',final216='NOT_STARTED',formal_timing='NOT_STARTED',P5='NOT_STARTED',
            selected_RL_round=128,selected_control_round=256,source_binding_sha256=digest(source_path()),protocol_sha256=digest(protocol_path()))
        write_new(HOME/'STAGE_STATUS.json',state)
        readme=(HOME/'README.md').read_text(encoding='utf8')
        readme+='\n\n## 已完成结果\n\n**工程GO／科研REVISE，P2未通过。** 首seed256轮已真实完成，所选RL128相对SL仅+0.2233%、相对监督256为−0.8682%；其余条件阶段未启动。活动版本[small256协议](PROTOCOL_SMALL256.json)及[R03绑定](SOURCE_BINDING_R03.json)，保留原512版本。\n\n[实际报告](EXPERIMENT_REPORT.md) · [Actor索引](ACTOR_INDEX.json) · [部署清单](DEPLOYMENT_PACKAGE_MANIFEST.json) · [完整资源](RESOURCE_ACCOUNTING_FINAL.json) · [独立审查](INDEPENDENT_REVIEW.md) · [下一步](NEXT_STEP_DECISION.md)。上述规划命令调用已实际用于评估的同一冻结Actor实现；显式消费者新推理不能混入本轮科学结果。\n'
        derived(HOME/'README.md',readme.encode('utf8'))
        root=ROOT/'EXPERIMENT_RESULT_TO_CLAIM.md';body=root.read_text(encoding='utf8');marker='## Integrated-RPPO-v1-small256（2026-11-10）'
        if marker not in body:
            body+='\n\n'+marker+'\n\n固定DRL目标保持；真实完成首seed256轮，P2未通过，claim_supported=no／confidence=high，写作provisional。所选RL128对SL H后+0.223271%、对匹配监督256为−0.868168%；raw对SL−0.094853%。只属8个已训练物理布局的开发选择，未触发其余seed／独立晋级／最终216／计时／P5。不以工程、loss、H收益替代DRL增量。\n\n[实际证据](rc3_integrated_rppo/EXPERIMENT_REPORT.md) · [独立审查](rc3_integrated_rppo/INDEPENDENT_REVIEW.md) · [研究发现与收尾路由](rc3_integrated_rppo/findings.md)。原S0—S3、GL-v4/v5和旧失败保留。\n'
            if b'\r\n' in root.read_bytes():body=body.replace('\n','\r\n')
            root.write_bytes(body.encode('utf8'))
        log(dict(event='reviewed_reports_finalized',claim_review_sha256=digest(HOME/'CLAIM_REVIEW.json'),status=state))
    finally:clock.close();log.close()
    events,tail=read_chain(OUT/'ledger/events.jsonl');wall={}
    for e in events:
        if e['event']=='clock_close':wall[e['name']]=wall.get(e['name'],0.)+e['elapsed_s']
    res=dict(res,wall_s=wall,closed_ledger_tail=tail,cumulative_loading=read(HOME/'CACHE_ACCOUNTING_RECONSTRUCTION.json'),
        administrative_byte_audit=read(HOME/'CLOSEOUT_INTEGRITY_AUDIT.json'),claim_review_sha256=digest(HOME/'CLAIM_REVIEW.json'))
    write_new(HOME/'RESOURCE_ACCOUNTING_CLOSEOUT.json',res)
    write_new(HOME/'FINAL_REPORT_COMPLETE.json',dict(status='COMPLETE_REVIEWED_P2_CLOSEOUT',state=state,resource_final_sha256=digest(HOME/'RESOURCE_ACCOUNTING_FINAL.json'),
        resource_closeout_sha256=digest(HOME/'RESOURCE_ACCOUNTING_CLOSEOUT.json'),closed_stage_wall_s=wall,closed_ledger_tail=tail,new_scientific_calls_after_core=0,utc=now()))
    print(json.dumps(state))


if __name__=='__main__':main()
