"""Failure reporting/Git metadata maintenance ONLY; never imports science code.

This helper does not recover, fit, infer, evaluate, or reset the exhausted M1
scientific budget. It documents missing outcomes rather than fabricating zeros.
"""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def read(name): return json.loads((HERE/name).read_text(encoding='utf-8-sig'))
def put(name,value):
    with (HERE/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def md(name,text):
    with (HERE/name).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    account=read('RESOURCE_ACCOUNTING.json');registry=read('M1_DATA_AND_SOURCE_IDENTITY.json')
    assert account['status']=='FAILED_PRESERVED' and account['counts']['critic_optimizer_attempts']==0
    wall=account['wall_s'];over=wall-1800
    missing='NOT_RUN_WALL_LIMIT_BEFORE_ENGINEERING_AND_FIT'
    checks=[]
    for agg in ('state_pooled','parent_equal'):
        for item in ('value_vs_conditional','paired_parent_CI','value_vs_C0','normalized_L_vs_C0','normalized_K_vs_C0','normalized_H_vs_C0'):
            checks.append(dict(check=item+'_'+agg,status='INSUFFICIENT',value=None,reason=missing))
        for size in ('10x10','4x3','6x6','8x8'):
            w,h=map(int,size.split('x'))
            coverage=next(r for r in registry['coverage'] if r['split']=='validation' and r['width']==w and r['height']==h)
            checks.append(dict(check='size_'+size+'_'+agg,status='INSUFFICIENT',value=None,
                registered_parents=coverage['parents'],metadata_coverage_status='GO',evaluated_parents=0,reason=missing))
    decision=dict(schema='rc3-r04-m1-g3-execution-failure-v1',status='INSUFFICIENT',execution_status='FAIL',
        endpoint=None,checkpoints_executed=[],checks=checks,C1_scientific_conclusion='NOT_EVALUATED',
        initialization_metrics='NOT_RUN',midpoint_rule_evaluated=False,endpoint256='NOT_RUN',
        reason='Cumulative wall cap detected in source preflight; no optimizer/test/feature restoration began',
        cumulative_wall_s=wall,cap_s=1800,overrun_s=over,cap_enforcement='FAILED_TO_INTERRUPT_BETWEEN_COOPERATIVE_CHECKS',
        budget_reset=False,automatic_retry=False,G2='NOT_PASSED',M2='NOT_RUN',M3='NOT_RUN',
        next_step='STOP_CURRENT_EXECUTION; repair hard-timeout/environment before any separately authorized recovery; no G2 admission',
        interpretation='Missing M1 outcomes neither reject nor confirm C1 or Actor centering. No route-benefit claim.')
    put('M1_GATE_DECISION.json',decision)
    rows=[]
    for model in ('C0','C1'):
        for step in (0,128,256):
            for comp in ('value','normalized_L','normalized_K','normalized_H','raw_remaining_L','raw_remaining_K','terminal_H_credit'):
                for agg in ('state_pooled','parent_equal'):
                    rows.append(dict(configuration=model,endpoint=step,split='validation',window='ALL',source_stage='ALL',
                        component=comp,aggregation=agg,status='INSUFFICIENT',mse='',conditional_mse='',global_mse='',
                        ratio_conditional='',ci95_low='',ci95_high='',evaluated_states=0,evaluated_parents=0,
                        registered_parents=16,reason=missing))
    with (HERE/'M1_METRICS.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    put('STAGE_STATUS.json',dict(status='FAIL',scientific_gate='INSUFFICIENT',source_commit='9b2a743ca83f454aa15439e03eac3232848871e5',
        protocol_sha256=sha(HERE/'M1_EXECUTION_PROTOCOL.json'),data_sha256=sha(HERE/'M1_DATA_AND_SOURCE_IDENTITY.json'),
        actor_optimizer_attempts=0,critic_optimizer_attempts=0,scientific_endpoints=[],
        old_R03_256='NOT_RESUMED',original60_and_oldS3='SEALED',next_step=decision['next_step']))
    put('ACTIVE_EXECUTION_STATUS.json',dict(schema='rc3-r04-m1-closed-execution-v1',status='STOPPED_FAILED_WALL_LIMIT',
        protocol='M1_EXECUTION_PROTOCOL.json',protocol_sha256=sha(HERE/'M1_EXECUTION_PROTOCOL.json'),
        stage_authorization='STAGE_AUTHORIZATION.json',gate='M1_GATE_DECISION.json',resource='RESOURCE_ACCOUNTING.json',
        authorization_budget_exhausted=True,old_global_pointers_modified=False,M2_M3_authorized=False))
    put('MISSING_RESULTS.json',dict(reason=missing,unit_tests_run=0,unit_tests_implemented=14,
        restored_trajectories=0,trained_models=0,checkpoint_exports=0,
        absent=['UNIT_TEST_RESULTS.json','ACTUAL_ENVIRONMENT.json','RESTORED_INPUT_INDEX.json','M1_RESTORED_DATASET.pt',
                'M1_STATE_PREDICTIONS.csv.gz','TRAINING_LOG.csv','FIT_ONLY_REFERENCES.json','PAIRED_PARENT_INTERVALS.json',
                'CHECKPOINT_INTERFACE_RESULTS.json','models/'],
        metric_table='84 explicit missing rows; blank numeric errors are not zeros or failed quality measurements'))
    report=f'''# R04-M1 实际执行报告

**执行 FAIL；G3 INSUFFICIENT。未获得 C0/C1 校准结果，不能声称完成256步拟合或通过工程验收。**

本轮已从 `ec4f5fafe788dc27c592db765cc87c3c855ca891` 建立独立分支，固化M1执行协议、独立批准/阶段授权、数据索引、共同批次、typed Critic及保存恢复接口，并完成静态独立审查。源码与登记在首次拟合入口前提交为 `9b2a743ca83f454aa15439e03eac3232848871e5`。随后真实运行因墙钟超限停止，没有继续或补跑。

## 实际消费与失败

| 项目 | 实际结果 |
|---|---:|
| 累计科学执行墙钟 | {wall:.9f}秒 |
| 原上限 | 1800秒 |
| 超出上限 | {over:.9f}秒 |
| Critic optimizer尝试/完成 | 0 / 0 |
| Actor optimizer | 0 |
| 恢复/语义读取的轨迹 | 0 |
| 新场景、导航、教师、native、H请求 | 全部0 |
| 原60及旧S3任务访问 | 0 |
| 工程测试、step0/128/256评价 | 全部NOT_RUN |
| 新权重及校准预测 | 未生成 |

首次metadata登记发生重复 `phase` 日志参数错误，尚未读取元数据或开始优化器。保留原错误与源码hash，按外部工具实测1.3070723秒保守扣1.5秒；修复仅事件写入，不改变协议、数据选择或预算。元数据随后成功锁定，累计账本没有清零。

正式入口在源码hash预检期间出现长时间停顿。等待调用请求1000毫秒，工具最终报告约2843.45秒；程序下一次预算检查发现超限，退出码1并保存FAILURE及关闭账本。**合作式检查未能在两次检查之间强制中断，因此1800秒硬上限未被严格守住。** 2859.26秒是真实保留消费，不能截断到1800、排除失败或写成预算内成功。现有收据只有检测栈和82次已插桩读取/hash计数，不能定位停顿发生于磁盘、系统暂停或其他环节；不能擅自归因。

## 元数据锁定结果

原33个fit物理父保留。从96个历史R03训练父中扣除前轮50父，46个候选按 `SHA256(11622026|R04-validation|physical_key)` 取前16，未看value误差或路线成本。实际锁定116条：64fit、52holdout。holdout是新的离线校准折，但属于历史SL/DRL见过的父布局，不能称未知场景泛化。

| holdout尺寸 | 固定物理父数 | 固定轨迹数 |
|---|---:|---:|
| 10×10 | 16 | 16 |
| 4×3 | 11 | 11 |
| 6×6 | 12 | 12 |
| 8×8 | 13 | 13 |

12个固定父×尺寸单元无可选已存记录，保留缺失，不换父或补任务。每尺寸均>=3父，元数据覆盖GO；这不等于完成评价。256×8个不同fit父的共同批次已冻结。

## 单因素与接口完成范围

C0/C1均为268-128-128-3、同seed11622026和同初始净张量。只改变输出物理单位；C0直接输出raw L/K/H，C1输出L/n、K/n、H/n。L/K softplus，H有符号，组合value为 `-(L/n+0.75K/n-H/n)`；H保留完整终端路径对历史前缀的修正信用。相同净张量的初始物理输出不同，不能归结为纯梯度尺度因果。

损失保持三分量0.5平方误差，state SUM/trajectory再episode MEAN；Adam1e-4、clip1、分量权重、批次和网络一致。没有C2 value项、条件残差或新特征。typed C1通过显式raw适配避免二次除n。Checkpoint源码绑定语义、来源、Adam/RNG/游标/资源；Critic导出明确不能替代Actor部署。

这些是已实现并静态审查的接口。14个测试尚未运行；真实保存恢复、导出、净张量相等及来源校验全量完成均未获本轮实测确认。保存状态没有原生268张量，计划的准确冻结SL前缀恢复亦未开始。

## G3逐项实际判断

`M1_GATE_DECISION.json`逐项为INSUFFICIENT：两聚合C1/条件常数、C1/C0、20,000次配对父区间、归一L/K/H、各尺寸误差、初始化与终点差异均没有数值。`M1_METRICS.csv`的84行明确记录缺项，MSE与区间为空，绝非零误差。128中期停止规则也没有被评价；本次停止原因是运行墙钟，不能冒充已触发C1质量停止规则。

独立审查仅复核实际失败账本、5条事件hash链、零更新与缺失产物，见 `INDEPENDENT_FAILURE_REVIEW.json`；未伪造未执行的数值复核，也未重跑历史完整审计。

## 下一步

**本次执行停止，G3未通过，不进入G2最终验收或M3。** 这不是C1科学否定，也不否证Actor居中假设。当前授权墙钟已耗尽，不通过改目录、清账或重跑获得结果。下一次任何科学恢复都必须保留本次失败与消费，先解决硬超时和环境停顿，并登记明确的新恢复授权；本轮没有启动该恢复。

R03仍在128停止，256未恢复；旧校准、R04原提案、旧模型、用户Word及封存边界保留。没有路线收益或真实IK/实机结论。

## 已实际发出的入口

```powershell
.\\rc3_learning\\.venv\\Scripts\\python.exe -B -X utf8 -m rc3_r04_m1_20261111.register
.\\rc3_learning\\.venv\\Scripts\\python.exe -B -X utf8 -m rc3_r04_m1_20261111.run
```

登记最终退出0；执行退出1。现目录有不可重置账本，会拒绝直接再次执行。注册目标环境为Python3.10.20/Torch2.12.1+cu126/NumPy2.2.6/CPU1线程，但本轮在环境确认前退出，不能把注册值写成实际环境验收PASS。
'''
    md('M1_CALIBRATION_REPORT.md',report)
    md('EXPERIMENT_REPORT.md','# R04-M1结果\n\n实际执行失败、G3证据不足；未发生Critic拟合。完整结果见[M1_CALIBRATION_REPORT.md](M1_CALIBRATION_REPORT.md)。\n')
    md('RESULT_TO_CLAIM.md','# 本轮结果可支持的结论\n\n支持：准确M1数据、协议和代码接口已锁定；入口已实际执行并在预检发现墙钟超限，失败及零更新证据保留。\n\n不支持：C1改善/失败、条件常数比较、父布局区间、路线收益、G2通过、Actor居中无效、工程测试通过。\n\nG3为INSUFFICIENT，不能用旧C1数值填入新留出折，也不能将静态GO当作执行成功。\n')
    md('NEXT_STEP_DECISION.md','# 下一步：停止本次执行\n\n执行FAIL、G3 INSUFFICIENT。累计2859.26秒超过1800秒，不能自动补跑。先定位停顿并建立能中断阻塞的外部硬超时；任何科学恢复须新明确授权并保留失败与累计消费。\n\n尚无C1科学判决，不进入G2最终验收或M3；不启动M2或额外配置。原R03的256终点继续未运行。\n')
    md('README.md','# R04-M1有界离线校准：实际执行失败\n\n**执行FAIL，G3 INSUFFICIENT。** 数据/接口已登记，实际入口在源码预检时发现墙钟超限：2859.26秒，Critic/Actor更新均0。未生成校准权重或预测，不能声称校准完成。\n\n- [实际报告](M1_CALIBRATION_REPORT.md)\n- [G3逐项缺项](M1_GATE_DECISION.json)与[指标表](M1_METRICS.csv)\n- [数据来源](M1_DATA_AND_SOURCE_IDENTITY.json)与[执行协议](M1_EXECUTION_PROTOCOL.json)\n- [真实资源账](RESOURCE_ACCOUNTING.json)与[独立失败复核](INDEPENDENT_FAILURE_REVIEW.md)\n- [生产迁移状态](PRODUCTION_MIGRATION_STATUS.md)与[下一步停止决定](NEXT_STEP_DECISION.md)\n\n旧R03/校准/R04提案未改；原60/旧S3继续封存；未合并main。\n')
    md('DELIVERY_INDEX.md','# 交付索引\n\n优先阅读：\n\n1. [M1_CALIBRATION_REPORT.md](M1_CALIBRATION_REPORT.md)：实际失败、数据锁定和结果缺项。\n2. [M1_GATE_DECISION.json](M1_GATE_DECISION.json)：G3逐项INSUFFICIENT。\n3. [RESOURCE_ACCOUNTING.json](RESOURCE_ACCOUNTING.json)：2859.26秒及两个保留失败，零优化器。\n4. [PRODUCTION_MIGRATION_STATUS.md](PRODUCTION_MIGRATION_STATUS.md)：已实现接口与未测试/G2缺口。\n5. [INDEPENDENT_FAILURE_REVIEW.md](INDEPENDENT_FAILURE_REVIEW.md)：独立失败证据审查。\n\n另见M1_METRICS.csv（空数值为明确缺项）、M1_DATA_AND_SOURCE_IDENTITY.json、M1_EXECUTION_PROTOCOL.json、SOURCE_BINDING.json、PROTOCOL_APPROVAL.json、STAGE_AUTHORIZATION.json和IMPLEMENTATION_REVIEW.json。\n')
    original=(ROOT/'README.md').read_bytes();at=original.find(b'\n')+1
    insertion=('\n## 2026-11-11：R04-M1执行失败，未开始拟合\n\n'
        '[本次M1实际报告](rc3_r04_m1_20261111/M1_CALIBRATION_REPORT.md)：已锁定33fit/16holdout父、116条轨迹及C0/C1生产接口源码；真实入口在源码预检发现墙钟超限，累计2859.26秒，Critic/Actor更新及轨迹恢复均0。**执行FAIL、G3 INSUFFICIENT；无新校准模型或性能结果，不进入G2/M3。** 时间硬上限未被严格守住，失败和全部消费保留，不自动补跑。[资源账](rc3_r04_m1_20261111/RESOURCE_ACCOUNTING.json)、[独立失败复核](rc3_r04_m1_20261111/INDEPENDENT_FAILURE_REVIEW.md)。下文历史原文保留。\n').encode()
    updated=original[:at]+insertion+original[at:]
    assert updated[:at]+updated[at+len(insertion):]==original
    (ROOT/'README.md').write_bytes(updated)
    put('README_UPDATE_RECEIPT.json',dict(prior_sha256=hashlib.sha256(original).hexdigest(),updated_sha256=sha(ROOT/'README.md'),
        insertion_at=at,insertion_bytes=len(insertion),prior_bytes_recoverable_by_removing_only_this_insertion=True))
    manifest_path=ROOT/'project_management/GIT_MANIFEST.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    manifest['version']='2026.11.11-r04-m1-execution-failed-budget-exhausted'
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            manifest['files'][path.relative_to(ROOT).as_posix()]=dict(sha256=sha(path),bytes=path.stat().st_size)
    manifest['files']['README.md']=dict(sha256=sha(ROOT/'README.md'),bytes=(ROOT/'README.md').stat().st_size)
    manifest['R04_M1_execution']=dict(base_commit='ec4f5fafe788dc27c592db765cc87c3c855ca891',
        branch='codex/rc3-r04-m1-calibration-20261111',status='FAIL',G3='INSUFFICIENT',
        report='rc3_r04_m1_20261111/M1_CALIBRATION_REPORT.md',wall_s=wall,optimizer_attempts=0,
        publication='PENDING_PUSH',old_entries_retained=True,user_Word_hashes_not_refreshed=True)
    manifest_path.write_bytes((json.dumps(manifest,ensure_ascii=False,indent=2)+'\n').replace('\n','\r\n').encode())
    print(json.dumps(dict(execution='FAIL',G3='INSUFFICIENT',wall_s=wall,missing_metric_rows=len(rows),new_models=0)))

if __name__=='__main__':main()
