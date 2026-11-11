# 供老师决定下一版主方案的材料

本轮仅整理已有证据和核对实现。新训练、模型前向、规划、导航、教师、native、H均0；不生成新的实验预算或执行授权。

**优先阅读：[老师决策短材料](TEACHER_DECISION_BRIEF.md)。**

| 附件 | 用途 |
|---|---|
| [完整任务统一表](TASK_LEVEL_COMPARISON.md) / [CSV](TASK_LEVEL_COMPARISON.csv) | 53行，实验/端点/cohort分开，父布局等权raw/H J/L/K与各自完整时间 |
| [原父布局配对区间](SAVED_PARENT_CONTRASTS.csv) | 已存112行区间转录，不重新bootstrap |
| [最新PG792评价逐例转录](LATEST_PG_EVALUATION_ROWS.csv) | 从原索引筛选已有evaluation行，不含在线训练请求 |
| [可学习模块实际作用](LEARNABLE_MODULE_AUDIT.md) | 基座、残差、mask、幅度、guard、H、服务分配及代理验证边界 |
| [32状态封存与旧账](FIXED32_CLOSURE.md) / [JSON](FIXED32_CLOSURE.json) | 共享PASS及CE14/Adam18/QP13 FAIL，预算与真实消费按原任务保留 |
| [结果到主张](RESULT_TO_CLAIM.md) / [独立审查](paper/review-traces/experiment-result-to-claim/2026-11-11_run01/) | 材料事实与下一版建议分开，不把建议写成已验证收益 |
| [维护资源账](RESOURCE_ACCOUNTING.json) / [下一步决定](NEXT_STEP_DECISION.md) | 本轮科学调用0、历史账不清零、无自动续跑 |
| [转录脚本](build_existing_tables.py) / [输入SHA与构表收据](TABLE_BUILD_RECEIPT.json) | 仅标准库读取已有表格，不导入模型或规划器 |
| [来源与历史字节核对](SOURCE_EVIDENCE_MANIFEST.json) / [Git远端核对](GIT_VERIFICATION.json) | 基准及实际交付身份，旧文件不覆盖 |

这些完整任务评价均是已暴露开发数据，不能升级为新的独立泛化确认。现有导航/F&A代理验证不等于真实IK或关节轨迹认证。未读原60、旧S3；未修改历史文件、合并main或提交用户Word。

已推送材料提交`4c320356775c253d97f603ca4b7d7aec8aa22fd7`，GitHub远端ref实际核对一致；新增LFS对象0，模型/历史证据保持。核对收据随后以纯元数据子提交保存，最终SHA另核对。
