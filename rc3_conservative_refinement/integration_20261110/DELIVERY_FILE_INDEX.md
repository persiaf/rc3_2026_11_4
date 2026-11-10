# 本轮交付文件索引

基准为`3f7507611b85dac757f126555258f32e0888dcb2`；真实训练器源码已先随`de700981ca6cb261bfdb7190b065af34c31aaead`发布。本次在独立`codex/rc3-conservative-paired-ppo-20261110`继续提交结果，不合并main。完整逐文件SHA见项目[GIT_MANIFEST.json](../../project_management/GIT_MANIFEST.json)。

| 类别 | 文件/证据 |
| --- | --- |
| 新206维执行代码 | [runtime.py](../runtime.py)、[trainer.py](../trainer.py)、[saved_inputs.py](../saved_inputs.py)、[storage.py](../storage.py)、[runner.py](../runner.py)、[analysis.py](../analysis.py)、[register.py](../register.py)、[engineering.py](../engineering.py) |
| 新工程检查 | test_runtime.py、test_trainer.py、test_saved_inputs.py、test_execution.py；90测试与20真实请求、接受/拒绝/恢复/部署、原生基线在engineering_R03；R01/R02失败保留 |
| 批准与绑定 | SOURCE_BINDING_R03.json、SCHEDULE_R03.json、PROTOCOL_DRAFT_R03.json及authorizations/实际回复、独立协议批准、阶段授权；冻结原字节 |
| 正式事实汇总 | [FORMAL_EXPERIMENT_REPORT.md](FORMAL_EXPERIMENT_REPORT.md)、[FORMAL_RESOURCE_ACCOUNTING.json](FORMAL_RESOURCE_ACCOUNTING.json) |
| 完整实际运行 | runs/conservative_paired_11622026_v1：2376请求JSON、138完整checkpoint、6 actor-only包、64/128两终点评价表与完整hash链 |
| 保存后审计 | analysis/逐请求raw/H路线、全部时间、2304条guard、149920条G/V记录、父簇对照、分量校准、停止汇总及AUDIT；无新规划/优化器 |
| 部署 | [DEPLOYMENT_ARTIFACTS.json](DEPLOYMENT_ARTIFACTS.json)、[DEPLOYMENT_AND_REPRODUCTION.md](DEPLOYMENT_AND_REPRODUCTION.md) |
| 审查与主张 | [独立正式审查](INDEPENDENT_FORMAL_REVIEW.md)、RESULT_TO_CLAIM.md、findings.md、paper/review-traces/；工程与科学结论分开 |
| 失败源码追溯 | source_revisions/：R01/R02精确SHA的事后恢复对象与索引，不冒充当时Git快照，不重新执行失败版本 |
| 维护与发布 | README、阶段状态、NEXT_STEP_DECISION、项目manifest与发布核对收据；闭合日志目录内LFS规则，根.gitattributes不改 |

原候选lookahead/adapter/model/costs/guard及旧测试、SL/H/导航/训练历史源码和冻结制品不改。最终字节复核见[FROZEN_BINDING_CLOSEOUT_CHECK.json](FROZEN_BINDING_CLOSEOUT_CHECK.json)：600份执行源码和17项锚点全部一致。

用户已有Word修改、四份新增论文Word、研究情况ZIP及目录均未纳入提交。没有新整包备份、分支删除、强推、main合并或封存任务读取。

老师优先阅读：正式报告 → 独立正式审查 → trainer.py → runtime.py → NEXT_STEP_DECISION.md。三项可立即开展的研究任务详见最后一份文件。
