# 实验历史与数值追溯

本轮没有重跑历史科研实验。表中提交是报告/结果形成或归档版本；实际执行源码以对应SOURCE_BINDING、协议、权重谱系和批准收据为准，不能用整理提交替代。

| 阶段与概述使用的数值 | 报告、明细和资源 | 固定版本/执行来源 |
|---|---|---|
| T0/T1完成；3960评价、6144监督更新；T1未通过，T2未运行 | [报告](../rc3_teacher_distill/REPORT_T012.md)、[resources](../rc3_teacher_distill/resources.json)、[labels](../rc3_teacher_distill/labels.jsonl)、[门槛](../rc3_teacher_distill/T1_GATE.json) | [6f595cfa8](https://github.com/persiaf/rc3_2026_11_4/tree/6f595cfa89f74d7f7a0e39faaadf5b8655d83f33)、[登记收据](../rc3_teacher_distill/registration_receipt.json) |
| S1-A完成：256教师、512 native/H调用、optimizer0 | [报告](../rc3_s1a_results_20261106/EXPERIMENT_REPORT.md)、[逐教师CSV](../rc3_s1a_results_20261106/PER_TEACHER.csv)、[资源](../rc3_s1a_results_20261106/RESOURCE_ACCOUNTING.json) | [3da8929f1](https://github.com/persiaf/rc3_2026_11_4/tree/3da8929f18357f2f35dd1816c3bff8a7bcf9fb67)、[协议批准](../rc3_global_local_execution_v3/authorizations/20261106_S1_A/PROTOCOL_APPROVAL.json) |
| S1-B完成：三种子768监督更新 | [报告](../rc3_s1b_results_20261106/EXPERIMENT_REPORT.md)、[seed_summary](../rc3_s1b_results_20261106/seed_summary.csv)、[资源](../rc3_s1b_results_20261106/RESOURCE_ACCOUNTING.json)、[谱系](../rc3_s1b_results_20261106/WEIGHT_LINEAGE.json) | [9babb93e9](https://github.com/persiaf/rc3_2026_11_4/tree/9babb93e92a560c3c3f5ab2c852d928ab5fae155)、[源码绑定](../rc3_navigation_compat_20261106/SOURCE_BINDING_UPDATED.json) |
| S2完成：三种子共3072正式更新，包含RL与continued-SL | [报告](../rc3_s2_results_20261107/EXPERIMENT_REPORT.md)、[资源](../rc3_s2_results_20261107/RESOURCE_ACCOUNTING.json)、[原日志索引](../rc3_s2_results_20261107/ARTIFACT_INDEX.md)、[谱系](../rc3_s2_results_20261107/WEIGHT_LINEAGE.json) | [1d4bc80fa](https://github.com/persiaf/rc3_2026_11_4/tree/1d4bc80fa160c50c0f733d7276a8ad6ed96bc975)、同一批准源码绑定 |
| S3：15876预扣、15874有效、2缺失；5质量/9部署完整格未过；另1/3格不完整，全局null | [报告](../rc3_s3_scope_results_20261109/EXPERIMENT_REPORT.md)、[逐请求CSV](../rc3_s3_scope_results_20261109/results/REQUEST_METRICS.csv)、[资源](../rc3_s3_scope_results_20261109/RESOURCE_ACCOUNTING.json)、[审查](../rc3_s3_scope_results_20261109/INDEPENDENT_REVIEW.json) | [1ca6659a4](https://github.com/persiaf/rc3_2026_11_4/tree/1ca6659a4c437cdf51dacd6dbcca9529dc8f9efe)、[scope源码](../rc3_s3_connectivity_20261109/SOURCE_BINDING.json)、[指令](../rc3_s3_scope_results_20261109/EXECUTION_INSTRUCTION_RECEIPT.json) |
| GL-v4完成：三组共384更新；封装等价，机制REVISE | [报告](../rc3_gl_v4_20261109/EXPERIMENT_REPORT.md)、[训练路线CSV](../rc3_gl_v4_20261109/TRAINING_AND_ROUTE_RESULTS.csv)、[资源](../rc3_gl_v4_20261109/RESOURCE_ACCOUNTING.json)、[角色澄清](../rc3_gl_v4_20261109/METHOD_IDENTITY_AUDIT.json) | [4dcd45103](https://github.com/persiaf/rc3_2026_11_4/tree/4dcd45103ea9d5758eb4bc5c056b962179ca2096)、[源码](../rc3_gl_v4_20261109/SOURCE_BINDING.json) |
| GL-v5完成：G128共128更新、2480请求；机制REVISE | [报告](../rc3_gl_v5_20261109/EXPERIMENT_REPORT.md)、[训练路线CSV](../rc3_gl_v5_20261109/TRAINING_AND_ROUTE_RESULTS.csv)、[资源](../rc3_gl_v5_20261109/RESOURCE_ACCOUNTING.json)、[审查](../rc3_gl_v5_20261109/INDEPENDENT_AUDIT.json) | [998f02b6c](https://github.com/persiaf/rc3_2026_11_4/tree/998f02b6c54a25a98a8bc9fbddec309a8cdb920c)、[源码](../rc3_gl_v5_20261109/SOURCE_BINDING.json) |
| RPPO部分运行：文件名closed16，JSON实际closed_rounds=20；不推断最终更新/晋级 | [部分收据](../rc3_integrated_rppo/PARTIAL_PUBLICATION_seed11622026_closed16.json)、[执行指针](../rc3_integrated_rppo/EXECUTION_PROTOCOL.json)、[R03绑定](../rc3_integrated_rppo/SOURCE_BINDING_R03.json) | 本地`0c268426f3760f4d06596a29f65b3d1a5309987f`；审计时不在origin，无可用远端提交链接 |

S3明细链接仅供获授权后的审查，本轮只读汇总报告、状态与来源元数据，没有打开目标任务/路线明细重算性能。原60封存。窗口、计时重复和优化种子均不等于独立物理环境数。

## 更早研究

| 研究线 | 原始报告与当前解释 |
|---|---|
| v4主实验 | [v4结果](../rc3_v4/RESULTS.md)、[冻结报告](../rc3_v4/results/RESULTS.md)：监督结构模型的结果不升级为DRL成功 |
| v5/v5.1维护 | [v5结果](../rc3_v5/RESULTS.md)、[证据地图](../rc3_v5/EVIDENCE_MAP.md)、[v5.1](../rc3_v5_1/README.md)：工程维护，真实IK未完成 |
| G2～G2.3 | [收尾](../rc3_g23_closeout/README.md)：旧窗口＋STOP支线保持冻结 |
| C0～C2.2 | [C012](../rc3_constructive/REPORT_C012.md)、[C21](../rc3_constructive/REPORT_C21.md)、[C22](../rc3_constructive/REPORT_C22.md)、[收尾](../rc3_constructive/REPORT_CLOSEOUT.md)：未通过/无定论，C3未运行 |
| R0/R1 | [R0](../rc3_nav_value/REPORT_R0.md)、[R1](../rc3_nav_value/REPORT_R1.md)：教师机会与学习收益分开，R2/R3未运行 |
| S0/工程准备 | [S0](../rc3_global_local/REPORT_S0.md)、[训练器工程](../rc3_global_local_trainer/TRAINER_VERIFICATION.md)：工程更新与正式训练分账 |

旧首页所有详细数字、文字和原链接按原字节保留在[README_HISTORY.md](../README_HISTORY.md)，SHA256=`7349b1d190d91f9549f1f007f517cac39f9524863594dfc9fdb994a203eff758`。其中“当前/未执行”均有历史时点，统一现状看[进度](RESEARCH_PROGRESS.md)。本轮没有新增无来源的历史数字。
