"""Build descriptive S1-B reports exclusively from saved scientific evidence."""
from pathlib import Path
import csv
import json
import sys
import time
import statistics

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rc3_s1b_results_20261106.audit_s1b import HERE,RUN,SEEDS,read,digest,PH,SH,PR,AH

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')

def md(name,text):
    (HERE/name).write_text(text.strip()+'\n',encoding='utf8',newline='\n')

def main():
    start=time.perf_counter();a=read(HERE/'MECHANICAL_AUDIT.json');d=read(HERE/'TEACHER_STATE_DIAGNOSTICS.json')
    assert a['status']==d['status']=='PASS'
    summary=[]
    for s in a['seeds']:
        initial=next(x for x in d['groups'] if x['seed']==s['seed'] and x['endpoint']=='initial')
        final=next(x for x in d['groups'] if x['seed']==s['seed'] and x['endpoint']=='SL256')
        summary.append(dict(seed=s['seed'],updates=s['updates'],training_wall_s=s['scientific_wall_s'],
            initial_route_macro_CE=initial['route_macro_CE'],final_route_macro_CE=final['route_macro_CE'],
            initial_top1=initial['top1'],final_top1=final['top1'],final_top3=final['top3'],final_top5=final['top5'],
            final_nontrivial_top1=final['nontrivial_top1'],nontrivial_states=final['nontrivial_decisions'],
            total_teacher_states=final['decisions'],first32_batch_mean_CE=s['first32_mean_loss'],last32_batch_mean_CE=s['last32_mean_loss'],
            gradient_before_min=s['gradient_before_min'],gradient_before_max=s['gradient_before_max'],gradient_after_max=s['gradient_after_max'],
            clipped_updates=s['clipped_updates'],abnormal_updates=s['nonfinite_or_invalid_updates'],
            unique_teachers_sampled=s['unique_teachers_sampled'],relative_parameter_L2_change=s['relative_L2_change'],
            conservative_training_plus_all_teacher_diagnostics_wall_s=s['scientific_wall_s']+d['wall_s']))
    write(HERE/'SEED_SUMMARY.json',summary)
    with (HERE/'seed_summary.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    table='| Seed | 更新 | 全教师初始CE | SL256 CE | 初始Top1 | SL256 Top1 | 训练阶段秒 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for s in summary:
        table+=f"| {s['seed']} | {s['updates']} | {s['initial_route_macro_CE']:.6f} | {s['final_route_macro_CE']:.6f} | {s['initial_top1']:.2%} | {s['final_top1']:.2%} | {s['training_wall_s']:.3f} |\n"
    detail='| Seed | 前32批平均CE | 后32批平均CE | 梯度裁剪前范围 | 裁剪后最大范数 | 裁剪次数 | 异常更新 | 抽样覆盖教师数 |\n|---|---:|---:|---|---:|---:|---:|---:|\n'
    for s in summary:
        detail+=f"| {s['seed']} | {s['first32_batch_mean_CE']:.6f} | {s['last32_batch_mean_CE']:.6f} | {s['gradient_before_min']:.6f}–{s['gradient_before_max']:.6f} | {s['gradient_after_max']:.9f} | {s['clipped_updates']} | {s['abnormal_updates']} | {s['unique_teachers_sampled']}/256 |\n"
    spread=(f"SL256全教师CE三seed均值{statistics.mean(s['final_route_macro_CE'] for s in summary):.6f}，样本标准差{statistics.stdev(s['final_route_macro_CE'] for s in summary):.6f}；"
            f"Top1均值{statistics.mean(s['final_top1'] for s in summary):.2%}，样本标准差{100*statistics.stdev(s['final_top1'] for s in summary):.4f}个百分点，"
            f"最大seed间差{100*(max(s['final_top1'] for s in summary)-min(s['final_top1'] for s in summary)):.4f}个百分点。这里只描述三个固定seed，不构成独立场景显著性检验。")
    binding=f"协议 `{PH}`；批准执行源码清单 `{SH}`；独立源码批准收据 `{PR}`；三种子阶段授权 `{AH}`。Registry `75d110bef7e9954e9ace14198d262a940af6aacb17abad71cf83771ace8b363a`；H `5db67cacc1128b6143c7a04b9b84a70c37bdc0ab917a0eebae0b8aca132c4d14`。"
    md('README.md',f'''# S1-B 三种子监督初始化：实际执行结果

**三种子全部完成，各256次真实更新，共768次。S2技术准入GO；S2/S3未授权、未执行。**

{table}

CE为固定256条训练教师的路线宏平均；Top1为全部保存前缀的决策微平均。每个端点覆盖{summary[0]['total_teacher_states']}个状态，模型按教师动作推进。各seed多合法动作状态的命中率及Top3/5详见[诊断结果](TEACHER_STATE_DIAGNOSTICS.json)。这些数据反映训练教师状态上的部分模仿能力，不证明完整路线质量、未见场景泛化或DRL增量。

![三种子实际训练曲线](figures/s1b_training_curves.png)

![固定教师状态初末对照](figures/s1b_teacher_agreement.png)

全部权重及Adam/RNG在[原实验目录](../rc3_global_local_execution_v3/runs/rc3_v3_global_local_20261106_01/)，每seed有untrained及32/64/96/128/160/192/224/256检查点。固定端点及哈希见[权重谱系](WEIGHT_LINEAGE.json)。

- [实验报告](EXPERIMENT_REPORT.md)、[技术复核](IMPLEMENTATION_REVIEW.md)、[机械审计](MECHANICAL_AUDIT.json)
- [独立审查](INDEPENDENT_REVIEW.md)、[结论边界](RESULT_TO_CLAIM.md)、[下一步决定](NEXT_STEP_DECISION.md)
- [资源账](RESOURCE_ACCOUNTING.json)、[Git版本核对](GIT_VERSION_AUDIT.json)、[发布收据](PUBLICATION_RECEIPT.json)
- [逐状态指标CSV](teacher_state_metrics.csv)、[三种子汇总CSV](seed_summary.csv)、[曲线数据和脚本](figures/)

新增教师、场景生成、导航求解、native、H均为0；原S1-A {a['original_S1A_files_byte_checked']}份文件字节保持一致。S3及原60保留测试未读取/评价。
''')
    md('EXPERIMENT_REPORT.md',f'''# S1-B实际科学实验报告

实验`rc3_v3_global_local_20261106_01`；分支`codex/global-local-preflight`；执行基准`ffcfcabc4ccc0b071a368b5e6e49684be3de4d64`。研究目标继续为大规模复杂父场景DRL训练—冻结策略局部快速约束规划，S1仅提供辅助初始化。

{binding}

按11622026、11622027、11622028顺序运行同一生产入口；后续seed在前序COMPLETE校验后才启动。每seed 256次更新、batch4，Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，梯度全局范数裁剪1。teacher forcing masked CE先对单路线决策平均，再对4条路线平均；未改变网络、MDP、奖励、mask、H或样本分布。三个seed共享冻结256教师，数据流按各自S1_data/PCG64有放回采样；每seed1024条路线读取。

## 固定端点配对诊断

{table}

所有256教师、每条全部保存前缀、固定untrained与SL256端点均纳入。每端点{summary[0]['total_teacher_states']}状态；共6端点、{d['ledger']['model_scores']}次动作打分。只执行保存的教师动作，预测替代动作不进入环境；终局仅计算保存路线的静态metrics，无新规划/H查询。诊断前登记DIAGNOSTIC_PLAN.json，未选择最佳检查点、seed或子集。

初始与最终的全教师CE及Top1同时改善，支持已保存教师状态上的监督学习；最终Top1仍明显低于100%，不能描述为已完整掌握教师决策。标签内训练表现不构成独立验证。并列按最小局部索引处理；Top1可能把可等价的替代动作计为未命中，所以它也不能直接解释为实际成本遗憾。

## 曲线、梯度与权重

{detail}

{spread}

网络共有65,088个参数、25个参数张量，各seed的25个张量均变化；实际变化参数分别61,695/61,927/63,354，参数相对L2变化约7.19%/7.16%/7.20%。767/768次更新触发原定范数裁剪；这是固定优化配方的正常操作，非数值异常。裁剪后最大范数与1的微小偏差小于1.2e-7，审计容差1e-5。三seed分别抽到253/253/251条不同教师，符合原定有放回采样，没有补齐或替换未抽到的样本。

曲线原始256批次与最多16次尾随均值全部保留；批次变化导致噪声，主要初末比较使用同一256教师集。三个seed的起点/终点张量哈希、Torch初始化RNG及命名数据RNG均不同；各32次检查点的Adam步数、RNG游标、日志前缀哈希和权重版本相符。参数改变量、Top3/5和多合法动作子集见SEED_SUMMARY.json及TEACHER_STATE_DIAGNOSTICS.json。未按训练曲线挑选模型。

## 资源与不足

真实科学更新768；模型科学初始化3；新增scene/nav/native/H/teacher均0；失败和重试0。已逐字节核对原S1-A {a['original_S1A_files_byte_checked']}文件/{a['original_S1A_bytes']}字节，全部未变。分析另外构造6个仅装载既有权重的推理实例，优化器更新0，单独收费。生产阶段时钟、共享诊断时钟以及保守逐seed合并上界分别列在RESOURCE_ACCOUNTING.json；不将重叠时钟重复求和。

没有执行S2强化学习、S3局部完整路线评估或原60测试。没有新的J/L/K、冻结模型完整规划时间、约束可行率或独立泛化比较。学习率/预算/科学停止门槛未作事后调整。S2在线H费用、尚未访问训练任务的可行性和RL方差仍是需要正式运行回答的问题。
''')
    md('IMPLEMENTATION_REVIEW.md',f'''# S1-B实施闭环复核

**GO：已获明确授权的S1-B完成；S2仅技术准入GO，等待独立阶段授权。**

本文件为实际执行后的回顾，不伪称新事前审查。开跑依据是基准ffcfcab中已经完成的兼容修复/256教师回归、冻结工程验收、用户针对准确源码0f0ffd2…的新批准及本轮覆盖三seed的独立S1-B授权。没有重跑前轮256教师工程回归，也没有重建S1-A。

{binding}

生产authority逐次验证34份源码、H、环境、Registry、原S1-A完成和教师池来源，再启动优化器。现有严格兼容加载器消费原始full/reduced导航表示，原输入/NPZ/receipt/教师计划字节均未变。原协议、旧v2指针与旧435b源码清单均保留。

实际批次梯度、合法动作mask、终局标签来源、初始化/Adam/RNG谱系、每32更新检查点及顺序完成链均已由MECHANICAL_AUDIT.json核验。正式S1科学初始化3、更新768；本轮没有算法/奖励/样本分布/预算变更。

训练教师状态上部分动作模仿已经建立；完整策略闭环、未见任务质量、RL独立增量仍未测试。剩余风险包括共享小教师池上的过拟合、teacher forcing与自主轨迹分布差异、任务和开放端点难度、S2回报/优势方差及在线H时间。不得用SL指标替代DRL收益证据。
''')
    md('NEXT_STEP_DECISION.md',f'''# 下一阶段决定

**S1-B COMPLETE / PASS；S2技术准入GO，科学执行等待S2独立授权。S3和原60继续封存。**

三seed固定256端点全部完成，更新768/768；数值异常0，来源与优化器/RNG/日志闭合，原教师证据未变。全教师配对CE和动作命中均改善，说明辅助监督初始化发生有效学习；这不构成完整路线性能门槛或DRL科学结论，也没有新增一个事后SL性能准入阈值。

正式S2沿用已批准同一v3和0f0ffd2…源码版本：每seed从其匹配SL256（不得用工程夹具或别seed权重）分叉RL和continuedSL，两臂从相同模型张量开始，分别fresh Adam，冻结SL保留为对照。S2阶段授权应引用本轮三个完成收据及准确源/协议/Registry/H，并在执行中按既定seed顺序核上游。S1-B授权不涵盖S2。

建议预算就是原批准配方：每seed RL512+continuedSL512次更新，batch4；RL终局尝试最多6144，development/other最多2048，合计8192；两臂与数据准备、导航、完整H、开发检查及IO共用7200秒。三seed最多3072更新、24576终局尝试、21600累计阶段秒；这些是上限，不是已经消费。不能增加预算补救失败。

样本仍为96×8预登记父场景/任务银行，50%整10×10、50%原几何中的4×3/6×6/8×8任务窗。多起点三条轨迹、独立采样随机流、精确mask、负完整JH回报与固定加权LOO策略梯度保持不变；continuedSL只消费原教师。18个开发任务只在256/512处检查可行性与资源，不选最佳端点，不读取S3。

每次访问的训练任务必须先通过原F/A/导航与来源检查；768声明不代表已全部生成或证明可行。失败、非有限、全空批或超限必须保存已扣费部分结果并停；安全边界恢复仅用同授权/精确checkpoint+Adam+RNG+日志游标+累计账，不静默重跑、换seed或清零资源。

技术风险仍包括自主轨迹分布偏移、教师模仿不完全、策略梯度方差和H在线耗时。S2是否在7200秒内完成与是否获得优于冻结SL/continuedSL/规则的独立RL增量均未知；现有S1结果不能预先保证。没有启动S2或S3。
''')
    write(HERE/'WEIGHT_LINEAGE.json',dict(schema='rc3-s1b-weight-lineage-v1',experiment_id='rc3_v3_global_local_20261106_01',
        protocol_sha256=PH,execution_source_manifest_sha256=SH,source_approval_sha256=PR,S1_B_authorization_sha256=AH,
        teacher_pool_index_sha256=digest(RUN/'teachers256/POOL_INDEX.json'),
        seeds=[dict(seed=s['seed'],checkpoints=s['checkpoints'],completion_sha256=s['completion_sha256'],
            future_RL_and_continuedSL_source=s['checkpoints'][-1],future_optimizer='fresh Adam independently in each fork; not executed') for s in a['seeds']],
        S2_executed=False,S3_executed=False))
    write(HERE/'STAGE_STATUS.json',dict(schema='rc3-s1b-stage-status-v1',S1_B='COMPLETE',seeds=list(SEEDS),
        scientific_optimizer_updates=768,technical_S2='GO',S2_authorized=False,S2_executed=False,S3_authorized=False,S3_executed=False,
        old60='SEALED',S3='SEALED',source_revision_approved=True,protocol_changed=False,
        source_approval_sha256=PR,stage_authorization_sha256=AH,result_scope='S1 training-teacher-prefix imitation only'))
    write(HERE/'RESOURCE_ACCOUNTING.json',dict(schema='rc3-s1b-resource-accounting-v1',experiment_id='rc3_v3_global_local_20261106_01',
        scientific_training=dict(seeds=[dict(seed=s['seed'],optimizer_updates=s['updates'],stage_wall_s=s['scientific_wall_s'],
            stage_cap_s=1800,ledger=s['ledger']) for s in a['seeds']],optimizer_updates_total=768,
            policy_initializations=3,stage_wall_sum_s=sum(s['scientific_wall_s'] for s in a['seeds']),
            new_scene_generation=0,new_navigation_solves=0,new_teacher_queries=0,native_calls=0,H_calls=0,failures=0,retries=0),
        training_teacher_diagnostics=dict(wall_s=d['wall_s'],wall_cap_s=d['wall_cap_s'],ledger=d['ledger'],provider_ledger=d['provider_ledger'],
            allocation='Shared once in total resource consumption; conservatively charge the entire shared diagnostic duration to EACH seed for the1800s compliance bound',
            per_seed_conservative_s={str(s['seed']):s['scientific_wall_s']+d['wall_s'] for s in a['seeds']}),
        maintenance_byte_audit=dict(wall_s=a['audit_wall_s'],files=a['original_S1A_files_byte_checked'],bytes=a['original_S1A_bytes'],optimizer_updates=0,teacher_queries=0),
        broad_frozen_evidence_byte_audit=read(HERE/'FROZEN_EVIDENCE_AUDIT.json') if (HERE/'FROZEN_EVIDENCE_AUDIT.json').exists() else None,
        time_scope='Production elapsed_stage includes authorization/source/data/cache/initialization/backward/checkpoint IO through completion resources; diagnostic timer includes load/score/CSV. Last tiny receipt cannot self-time. Process startup before main was not separately instrumented. Do not sum nested clocks or multiply shared diagnostic wall for actual consumption.',
        extra_generation_or_training_in_analysis=0,diagnostic_policy_instances=6,diagnostic_optimizer_updates=0,
        S1_A_repeated=False,S1_A_bytes_preserved=True,S3_task_reads=0,original60_semantic_reads=0,
        S2_optimizer_updates=0,S3_evaluations=0,git_and_publication='Maintenance, not scientific training or planning; recorded separately in Git audit',
        report_build_wall_s=time.perf_counter()-start))
    print(json.dumps(dict(status='REPORTS_BUILT',seeds=summary),ensure_ascii=False))

if __name__=='__main__':main()
