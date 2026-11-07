# S2正式DRL科学训练闭环

状态：`COMPLETE_ALL3_FIXED_ENDPOINTS`。实际完成更新 `3072`；S3 模型谱系就绪 `GO`，执行就绪 `REVISE_PENDING_FROZEN_PROTOCOL_EVALUATOR_AND_SEPARATE_AUTHORIZATION`，执行授权仍为无。

| seed | 实际状态 | RL updates | continued-SL updates | 共享秒数 | RL终局尝试 | 其他终局尝试 | 总终局尝试 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 11622026 | COMPLETE | 512 | 512 | 2954.54 | 6143 | 36 | 6179 |
| 11622027 | COMPLETE | 512 | 512 | 3029.52 | 6144 | 36 | 6180 |
| 11622028 | COMPLETE | 512 | 512 | 3253.56 | 6144 | 36 | 6180 |

固定协议 `0c5bc7d7a9557edf2af76f51a9f418550f0b4ce1ec6a8b47ddea95c6256e78bb`，已批准源码绑定 `0f0ffd2d2dda2fefefcfcb5cb26dc5197b5d11bd1332b0a5546992409049bfe0`；原真实源码修订批准 `9a45d6e7a6b7ccbc8d43b614646ecefe417ba10eaabfe2055c4e896f1e6aa7c4` 保持。

本轮真实用户 S2 授权 `4dba652b2156b88848c4c1c49826952414a5b6d08baa7ce250d710b3ea312423`（训练前提交 `13a1495`）一次覆盖三固定 seed；它不授权 S3。

开发可行性分母：当前保存记录对应 `108` 次重复 DEV 评价，其中可行 `108` 次。原声明 DEV 任务只有 18 个；三 seed 全完成时为 `18个DEV任务 × 2个固定checkpoint(256/512) × 3个seed = 108次重复评价`。即使汇总为 108/108，也只表示这些重复开发评价全部可行，绝非 108 个唯一独立案例，更不是封存 S3 的 108 个目标案例。本轮 S3 目标语义消费为 `0`。这些数据只支持开发可行性／资源检查，不作质量筛选或泛化主张。

报告：[实验](EXPERIMENT_REPORT.md)、[主张边界](RESULT_TO_CLAIM.md)、[真实资源账](RESOURCE_ACCOUNTING.json)、[权重谱系](WEIGHT_LINEAGE.json)、[阶段状态](STAGE_STATUS.json)、[下一步](NEXT_STEP_DECISION.md)。

[完整权重与产物入口](ARTIFACT_INDEX.md)、[梯度、优势、熵、分窗口与首末区间诊断](TRAINING_DIAGNOSTICS.md)、[独立审查](INDEPENDENT_REVIEW.md)、[单独记账的报告与传输异常](MAINTENANCE_NOTES.json)。

[分析方法](ANALYSIS_METHOD.md)、[保存数据分析](analysis/ANALYSIS_SUMMARY.json)、[图注](FIGURE_CAPTIONS.md) 与图 01–09 的独立脚本保留。图形仅用于训练诊断。

原 S1-A/S1-B 与历史证据经单次独立字节审计保留；`.gitattributes` 仅为新 S2 RL journal 增加路径专属 LFS 规则，审计中的该文件检查注明为规则更新前。

- 训练输入与采样分布随 update 改变；return/J/L/K 曲线及首末区间仅描述实际训练抽样，不能证明 DRL 质量提升或新规模泛化。
- H_gain=raw_J-H_J 归属于冻结 H 后处理；不得解释为独立网络收益。
- RL 的 REINFORCE objective 与 continued-SL 的交叉熵定义和尺度不同，不以损失大小或下降幅度比较规划质量。
- 256/512 固定开发检查复用原 18 个声明 DEV 任务；三 seed 全完成时 18×2×3=108 次重复开发评价，绝非 108 个唯一独立案例，也不是封存 S3 的 108 个目标案例。仅作可行性与资源检查，不作质量筛选、最佳 checkpoint 选择或泛化判断；本轮 S3 语义消费为 0。
- 未新增 frozen-SL、未训练策略的路线评价，未执行 SL 对 RL 的冻结路线比较；不据此宣称优于原生基线、规则或教师。
- 保存的模型计划验证只覆盖路线与代理可达约束；真实物理标定、IK 与机械臂关节轨迹认证尚未完成。
- S3 与原 60 历史保留任务继续封存；模型谱系就绪 GO 只覆盖 S2 权重证据，不代表 S3 执行器就绪或阶段授权。未启动与缺失终点不写成失败或成功。
- ledger 的动态计数覆盖有限：缺失模型初始化键不代表实际构造为零；source policy/source Adam 重建与三分支复制在阶段时钟内。没有新增科学未训练起点，原 S1 端点继续复用。
