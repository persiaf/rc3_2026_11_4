# 注册前独立源码审查快照

日期：2026-11-10。角色：独立 paper architect/reviewer，继续 [集成预审](INTEGRATION_PRE_REVIEW.md)。本次只读检查 `training.py`、`ppo.py`、`runner.py`、`verification.py`、`register.py`，并追踪必要的 `model.py`、`rollout.py`、`data.py`、`eval.py`、`common.py`、`statistics.py`、`report.py` 接口。实现正在由主执行者同步修正；本文件记录观察时点，不是最终冻结 SHA 或执行验收。

**快照判定：REVISE，剩余事项均为具体恢复／交付合同，不构成算法或研究方向否定。** 本 reviewer 未运行模型实例化、forward、backward、optimizer、导航、H、教师、新输入或科学任务；全部新增研究调用为 0。未修改源文件、Git或冻结历史证据。

## 已通报并在随后源码中观察到关闭的事项

| 事项 | 初始问题 | 随后观察到的修改 | 判定范围 |
| --- | --- | --- | --- |
| 动态 D 身份 | 旧 problem identity 不含 D，原动态缓存 key 无 D | `model.encode/features/pack_states` 显式加入 endpoint D hash，并拒绝混用 | 静态关闭；实际负向测试归 P0。 |
| 128 轮早停恢复 | 128 检查点保存后中断，恢复从 129 开始，可能绕过64/128双坏早停 | `training.seed_run` 的恢复分支重建选择事件，在128边界重算同一条件，满足则保持 STOPPED_PREDECLARED_EARLY | 静态关闭，不新增性能条件。 |
| checkpoint 不可变身份 | `restore` 只信任 CURRENT 内部字段，未核闭合事件所封 SHA | 读取前要求 CURRENT 精确 SHA、seed/stage、round/cursor 对应唯一 `checkpoint_closed` 事件 | 静态关闭；未锚定 CURRENT 拒绝，而非脏优化器重放。 |
| P5 已完成阶段 | 第二消融中断后重启，先恢复已完成第一消融，因全账已有第二消融更新而被拒绝 | `read_complete` 核 completion 文件与 `seed_stage_complete` SHA，runner 对主阶段／消融均先复用完成收据 | 静态关闭，完成阶段不重新训练。 |
| P5 配对 RNG | 协议称 paired seeds，但动作随机流把 variant stage 纳入 seed | 独立请求身份保留 variant，另设 `sampling_stage='main'` 与主线前64轮配对 | 静态关闭；各方法仍各自产生 on-policy 动作。 |
| unseen 物理重复 | 只排除历史固定 key，未排除本轮先前另一 unseen seed 的物理 key | `data.parent` 新增对本轮 index 的跨 map_seed 去重，`register` 纳入历史几何元数据根 | 静态关闭；不以 seed 不同替代物理唯一性，也不读取旧S3目标语义。 |
| 孤立墙钟尾段 | 原实现只累计闭合钟，漏掉中断尾段 | 已加入父终止收据和无收据时的尾段补扣 | 同一系统单调钟原点下已改善；跨机器重启的证明条件见下一表。 |

以上“关闭”只表示本次实际读到相应源码，不声称 P0 已运行或通过。具体行号在同步修改中会变化，可按函数和事件名定位。

## 本快照尚待关闭的具体事项

| 文件／定位 | 可触发的错误 | 执行前所需处理 |
| --- | --- | --- |
| `runner.execute`，FINAL_CONFIRMATION_COMPLETE 的 `write_new`（审查时约94行） | P4已经完成、P5期间中断后重新进入，会重读合法缓存结果，然后因已存在最终确认收据触发 FileExistsError，无法到达P5恢复 | 完成收据存在时核其源码、选择、结果身份和计数后复用；不覆盖、不再次生成科学请求。 |
| `runner.execute`，RUN_COMPLETE 的 `write_new`（约104行）及 `report.csv_new/summarize` | 汇总或IO在中途失败后，已有RUN_COMPLETE／前几个CSV／JSON阻止再次分析；`open('x')`首个已存在文件就再失败 | 将只读分析交付做成可恢复事务／内容确定的存在核验，或独立完成收据；保持原请求对象与研究结果不可变，不重放训练或评价。 |
| `report.summarize` 的尺寸表、effects与timing输出 | 最终尺寸表未以seen/unseen分组，聚合effects混合两分布；仅有逐请求计时，无所引用的deployment table或严格成本／median／p95判定 | 补任务书要求的seen/unseen分列、尺寸／端点质量表与(group,size,cache)部署比较；完整格按已定门槛判读，缺失保持INCOMPLETE。无需改变门槛。 |
| `common.Clock.__init__` 无父终止收据的 `time.perf_counter()-old_start_perf`（约111行） | 机器重启后单调钟原点改变；正差值也不能证明旧尾段，因此仅检查delta>=0不足以支持跨重启预算继承 | 使用可核boot身份证明同一原点；无法证明时拒绝该差值，或按原UTC至当前UTC作明确保守扣账。父终止收据有精确尾段则照实用收据。 |

这些问题已逐项即时通知主执行者。若其后已修改，应对最终准确源码复核并追加关闭记录，不把本快照的旧行号当作仍然存在的错误。

## 未发现矛盾的核心执行语义

- 新 `PPO` 使用各有效动作的概率比、固定采样 old_logp/V_old、完整 reward-to-go 与 detached 统一 RMS 尺度；Actor 为每episode步和再对episode平均，Critic独立。
- 精确KL在相同有效动作mask上求全分布；越界恢复失败epoch的Actor与Adam，真实尝试不退账，LR减半次数与连续失败批停止有明确代码。
- 匹配 residual-SL 只用原教师，优化残差头，按接受的Actor epoch数匹配；实际额外Critic、失败回滚与数据访问另计，不声称等FLOPs。
- P0计划42完整请求及不超过32真实工程optimizer尝试，工程模型与正式seed模型分离；实际P0仍需在注册冻结后运行。
- 选择候选128/256/512、64仅诊断，P2研究晋级与强规则部署判断分开；未新增Top1或额外质量门槛。
- 新最终目标任务仅在三seed选择与晋级完成后生成；原60、旧失败槽和旧S3语义边界保持。源码阅读不构成重新消费目标任务。

最终源码绑定、注册时间与 P0 真实收据仍由主执行者形成。本文件不提前授予科学成功主张，不替代 `IMPLEMENTATION_REVIEW.md` 的最终 GO/REVISE/BLOCKED，也不要求对已授权条件阶段重复申请许可。
