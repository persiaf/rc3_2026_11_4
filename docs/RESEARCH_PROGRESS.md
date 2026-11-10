# 真实研究进度

状态截止于整理所用科研快照`0c268426f3760f4d06596a29f65b3d1a5309987f`。下表的“完成”指相应运行或工程工作完成，不能替换“通过科学门槛”。本轮未重新执行历史实验。

| 研究线 | 已完成内容 | 科学判断与未完成内容 | 权威记录 |
|---|---|---|---|
| v4/v5/v5.1可靠规划 | Fixed3/Detour、代理验证、导出、外部服务表与输入预检 | 规则仍为默认；真实IK与实机验证未完成 | [v4结果](../rc3_v4/RESULTS.md)、[v5证据地图](../rc3_v5/EVIDENCE_MAP.md)、[v5.1](../rc3_v5_1/README.md) |
| G2～G2.3旧DRL | 历史训练、诊断和收尾均保留 | 局部窗口选择＋STOP路线保持收尾 | [G2.3收尾](../rc3_g23_closeout/README.md) |
| C构造式支线 | C0/C1/C2、C2.1后处理、C2.2对照完成 | 当前同结构继续策略梯度路线收尾；C3未运行 | [收尾](../rc3_constructive/REPORT_CLOSEOUT.md) |
| R价值支线 | R0教师机会和R1有界筛选完成 | R1未建立可靠质量/部署优势；R2/R3未运行 | [R0](../rc3_nav_value/REPORT_R0.md)、[R1](../rc3_nav_value/REPORT_R1.md) |
| T教师蒸馏 | T0诊断、T1监督筛选完成；T1共3960次评价、6144监督更新 | T1主门槛未通过；T2完整路线、T3未运行；不能写成RL收益 | [报告](../rc3_teacher_distill/REPORT_T012.md)、[账本](../rc3_teacher_distill/resources.json) |
| S0全局—局部环境 | 一致性与重放工程完成 | 软件一致性不是学习性能证据 | [S0](../rc3_global_local/REPORT_S0.md) |
| S1-A教师 | 256教师、512次native/H评价、无优化器更新 | 教师收益不是网络收益；原导航兼容修复另有来源链 | [报告](../rc3_s1a_results_20261106/EXPERIMENT_REPORT.md) |
| S1-B监督 | 三种子共768次正式监督更新 | 模仿能力不能当作完整DRL规划改进 | [报告](../rc3_s1b_results_20261106/EXPERIMENT_REPORT.md) |
| S2强化学习与继续监督 | 三种子共3072次正式更新，包含两种训练角色 | 训练执行完成；未凭训练曲线证明部署优势 | [报告](../rc3_s2_results_20261107/EXPERIMENT_REPORT.md) |
| S3 scope_r1 | 15876预扣槽处置完成，15874有效计划、2历史缺失保留 | 5完整质量格和9完整部署格未通过；另1/3格不完整，全局AND为null；不补缺失 | [最终报告](../rc3_s3_scope_results_20261109/EXPERIMENT_REPORT.md)、[决定](../rc3_s3_scope_results_20261109/NEXT_STEP_DECISION.md) |
| GL-v4自然起点 | 部署封装等价、三组共384次更新及固定终点评价完成 | 封装工程GO；自然起点机制筛选REVISE | [报告](../rc3_gl_v4_20261109/EXPERIMENT_REPORT.md)、[决定](../rc3_gl_v4_20261109/NEXT_STEP_DECISION.md) |
| GL-v5固定贪心参考 | G128共128更新、2480请求 | 机制筛选REVISE，未改善主要学习对照 | [报告](../rc3_gl_v5_20261109/EXPERIMENT_REPORT.md)、[决定](../rc3_gl_v5_20261109/NEXT_STEP_DECISION.md) |
| Integrated RPPO | 独立模块、预登记、工程/源码修订和部分不可变运行证据已提交 | `PARTIAL_RUN_NOT_CLOSED_NO_PERFORMANCE_CLAIM`；本快照没有闭合终报告，不能给最终晋级/多种子/性能结论 | [部分收据](../rc3_integrated_rppo/PARTIAL_PUBLICATION_seed11622026_closed16.json)、[执行协议指针](../rc3_integrated_rppo/EXECUTION_PROTOCOL.json) |

## 当前状态应从哪里读

S1～S3的准确执行状态为[ACTIVE_EXECUTION_STATUS](../rc3_global_local_execution_v3/ACTIVE_EXECUTION_STATUS.json)及其独立批准/阶段收据。旧[FUTURE_DESIGN_STATUS](../rc3_global_local/FUTURE_DESIGN_STATUS.json)保留v2当时“未开始”的原字节，不是新的全局总进度。各历史README中的“本轮未执行”均有时间范围。

RPPO是后续独立协议线，不通过回写旧v3指针更改旧实验。其部分收据文件名/提交标题含`closed16`，但JSON实际记录`closed_rounds=20`，这里只按字段说明部分证据，不据文件名推断更新数。原工作区后续未提交账本与checkpoint不在本轮冻结快照；不扫描它们来宣布新阶段完成。

交接包固定于GL-v5，因此其中“残差PPO未执行”仅适用于交接包来源时点。仓库后续已有RPPO模块和部分运行证据，也仍不能写成成功的最终模型。[版本差异说明](HANDOFF_PACKAGE.md)

## 下一步研究与授权边界

固定DRL目标继续保持；下一步先审查表示、信用分配、更新约束与完整耗时，再在新协议下决定实验。方法、奖励、分布、预算或门槛实质变化须绑定源码/H/环境/种子并保留旧证据。协议批准和阶段执行授权分别记录。本轮授权仅覆盖文档与两个新工程规划示例，训练、S3重跑及原60保留测试均未执行。
