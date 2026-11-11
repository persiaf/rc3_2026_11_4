# R04交付清单

独立分支`codex/rc3-r04-value-relative-review-20261111`，基准853eb6b016df3aee08c233894ab6c24beb3765d8，不合并main。

| 要求文件 | 交付内容 |
| --- | --- |
| [R04_RESEARCH_DESIGN.md](R04_RESEARCH_DESIGN.md) | 固定目标、历史结论、两主假设、条件先验可删除性、依赖及四项最终回答 |
| [R04_MATHEMATICAL_MODEL.md](R04_MATHEMATICAL_MODEL.md) | C1单位、H整路径信用、Vcond+分量残差、合法pre-tanh均值、非零postmean反例和幅度/间隔界 |
| [R04_IMPLEMENTATION_PLAN.md](R04_IMPLEMENTATION_PLAN.md) | typed loss、原生产trainer/adapter/export/load/checkpoint迁移与真实事务验收缺口 |
| [R04_ABLATION_PROTOCOL.json](R04_ABLATION_PROTOCOL.json) | 未批准的离线门槛、可选条件锚点及四臂pilot，精确预算/公平控制/停止规则 |
| [R04_ACCEPTANCE_CRITERIA.md](R04_ACCEPTANCE_CRITERIA.md) | G0～G4、当前G2/G3阻断及何时允许申请新科学执行 |
| [R04_CODE_REVIEW.md](R04_CODE_REVIEW.md) | 逐项诊断代码/CSV检查、真实schema风险与实际工程结论 |

实际代码：relative_actor.py、critic_interfaces.py；各自独立测试18＋19项；offline_interfaces.py检查原stateless PPO loss、C1 raw适配、冻结wrapper及固定8个保存logit状态。新Actor包为严格mapping原型，不加载真实SL，也未接旧生产导出器。

身份与实账：ENGINEERING_REGISTRATION、ENGINEERING_AUTHORIZATION、IMPLEMENTATION_REVIEW、ACTUAL_ENVIRONMENT、UNIT_TEST_RESULTS、OFFLINE_INTERFACE_RESULTS、EVIDENCE_AUDIT、RESOURCE_ACCOUNTING、FROZEN_BYTES_CHECK和ENGINEERING_EVENTS。当前实际150.839秒、256模块调用/4次反向、全部optimizer与新科学请求0。

独立审查：STATIC_REVIEW保留初始REVISE和首算前修复；FINAL_INDEPENDENT_REVIEW判定设计/工程GO，科学NOT_READY/NOT_AUTHORIZED，不新增执行。保存CSV全量审计41,624,938项PASS，窗口/阶段矩统计使用新流式实现，继承旧算术函数的部分不伪称独立新实现。

路线图：paper/日期版与固定EXPERIMENT_PLAN、EXPERIMENT_TRACKER及MANIFEST，只是方案文档。所有M1/M2/M3科学阶段均未执行；不恢复R03，不加训练seed，不读原60/旧S3，不替换默认Fixed3/Detour。

维护：根README只插入R04入口，旧全文字节保持顺序；项目GIT_MANIFEST只追加本轮所有文件并更新本轮README，用户Word及无关ZIP/目录不入提交。GIT_LFS_VERIFICATION记录继承7对象本地fsck及一次独立空缓存远端样本hash一致，本轮新增LFS对象0、没有新模型或整包备份。最终远端结果另见PUBLICATION_RECEIPT。
