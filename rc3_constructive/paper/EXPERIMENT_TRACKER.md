# C0–C2 执行表

| ID | 工作 | 输入 | 状态 | 门槛 |
| --- | --- | --- | --- | --- |
| C1-contract | 关键合同检查 | 新环境边界例 | DONE：最终7/7 | 回报、ID、padding、逐臂交集、梯度、有效性、无穷掩码梯度 |
| C0-exact | 12例完整及受限全局DP | 6／10／14任务各4例 | DONE：各12/12最优、6例严格集合外价值 | 离散原图范围 |
| C0-time | 完整冷热分解 | 24新百桌开发例，2重复 | DONE：480测量 | 强规则前沿及公共开销 |
| C2-calibrate | 无optimizer贪心吞吐 | 32纯诊断例 | DONE：锁定5885秒软上限 | 64贪心反向探针及2条延时探针分别记录 |
| C2-train | 一种子REINFORCE | 24训练地图、独立任务流 | DONE：512更新／16,384优化采样 | 实际约798.3秒，选128检查点 |
| C2-assess | 原始及统一后处理评价 | 24开发＋48选择例 | DONE：另672全方法测量 | 主目标与继续信号均未通过 |
| C2-verify | 全路线数值重放及保存方案几何检查 | 16,384路线、1,584,734转移 | DONE：最终544证书；首轮543另计 | UTF8失败来源、恢复和重复工作明示 |
| C2-review | 独立结果及图表审查 | 完整保存证据 | DONE：主张no、有限信号partial；图表PASS | 下游写作provisional |
| C3 | 三种子、学习对照、测试及泛化 | 60保留新测试＋后续跨规模 | NOT STARTED | 不属于本轮第一份结果 |

## C2.1（独立于原封顶训练，2026-11-05）

| ID | 工作 | 状态与边界 |
| --- | --- | --- |
| C21-register | 保存路线与 H1 参数事前固定 | DONE：基于0a5fa02，五个检查点及输入哈希锁定 |
| C21-H1 | 24×3 次共同后处理 | DONE：72/72，无重复或失败重试；零新网络路线、零训练 |
| C21-verify | 输出独立几何／分配与指标核对 | DONE：72导出＋72独立终验，216次保存路线数值核对 |
| C21-attribution | ΔH、相同H1学习增量及Fixed3差距 | DONE：RL平均J=273.9585；相对Fixed3平均配对差+0.8170%；学习增量区间跨零 |
| C21-review | 独立结果到主张审查 | 见2026-11-05_run01 trace与EXPERIMENT_RESULT_TO_CLAIM.md |
| Post-aware-training | 单变量目标机制试验 | NOT STARTED：需未来明确任务，不自动扩训 |
| Reserved-test | 60个保留场景 | UNQUERIED：本轮仍为0 |

## C2.2（2026-11-05）

| ID | 工作 | 状态与判定 |
| --- | --- | --- |
| C22-register/calibrate | 固定实例、共同H与壁钟 | DONE：4096新任务、48机制种子、2401秒上限；校准零优化 |
| C22-A/B | 匹配暖启动raw/post目标训练 | DONE：各128更新×32，8192策略采样，固定baseline |
| C22-evaluate | 六方法固定终点评估 | DONE：新48场景、288方案；原60测试查询0 |
| C22-mechanism | g/h联合筛选 | INCONCLUSIVE STOP：g通过，h区间跨零且均值负；不自动追加 |
| C22-audit | 完整路线重放与独立结果审查 | 见verification.json与2026-11-05_run02 trace |
| C3／独立训练种子复核 | 超出本轮范围 | NOT STARTED |

## 当前支线收尾（2026-11-05）

| 项目 | 状态 |
| --- | --- |
| 同结构继续策略梯度训练 | CLOSED：接受C22无定论并停止，不从A/B继续 |
| 原update0128 | FROZEN RESEARCH REFERENCE：不是部署提升 |
| C012／C21／C22原始证据 | FROZEN，全部保留 |
| 收尾补算／字节检查 | SAVED-RECORD ONLY：新训练／规划等调用均0 |
| 导航关系信息价值验证 | PROPOSED NOT STARTED：先独立立项，完整H后的决策价值与场景实现性均待验 |
| 新模型／C3／原60保留测试 | NOT STARTED／UNQUERIED |
