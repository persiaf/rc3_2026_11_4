# 本轮实际结果独立复核

结论：执行与来源工程完成，G3 **数值 FAIL**。这两个结论必须同时保留；`COMPLETE_FAIL` 表示已完成获准离线实验但科学门槛未通过。

## 核对范围与依据

仅核本轮 attempts/01、02 的资源/失败/绑定收据、02 的新指标小表、门槛、初始化、checkpoint 收据和既有独立数值复核收据。没有重新前向、反向、拟合、bootstrap 或旧大表审计。独立算式已在 worker 内完成，2679495 项检查 PASS，覆盖 223776 条 state-component 记录、1680 个汇总组、fit33/holdout16、116 条轨迹与端点 0/128。

01 的 heartbeat 文件替换 WinError5 发生在任何轨迹恢复、step0 或优化器尝试之前。失败原文及自动 eligibility=false 均保留；独立补充审查根据具体堆栈将其认定为获准的记录接口修复。02 仅增加该文件替换的有限重试；锁持有人或底层原因仍为 UNKNOWN。

## 实际完成与停止

- 真实 Torch 环境分类、原 14 项 Critic 接口测试和精确 SL256 入口完成；116 条原定轨迹、7992 个状态全部恢复。
- C0=128、C1=128，Critic 尝试/完成总计 256；Actor 和全部禁止请求/封存语义读取计数为 0。
- step0 和 128 实际保存并评价。两种聚合的 C1/条件常数分别 2.863347、3.382871，且各自 95% 配对区间完全大于 1；128 中期规则已触发。step256 状态为 `NOT_RUN_REGISTERED_MIDPOINT_STOP`，不以零或缺失 MSE 代填。
- Value 对 C0 和归一化 L/K/H 对 C0 的六项分量比较通过；条件常数优势、配对区间上界与全部尺寸误差未通过。四尺寸各有 16、11、12、13 个验证父，覆盖不构成 INSUFFICIENT。
- G2 未执行，M2/M3 未获授权且未运行。没有规划收益证据。

## 原始接口收据的精确解释

`CHECKPOINT_INTERFACE_RESULTS.json` 的全部四项 roundtrip 确认模型/Adam/RNG/游标/资源恢复与导出预测逐位一致。其原模板 `resume_scope` 写有 “Each actual endpoint ... continued once”，并将 step128 的 `loaded_as_next_step_state` 记为 true；该文字不能被解释为 128 之后又执行了优化器更新。

实际日志只到 128。step0 恢复后继续了训练；step128 恢复/导出通过后触发中期停止，没有 step129。保留原收据原字节，并以本说明限定其含义。它也不是从头连续训练与完整暂停恢复两次独立运行的等价实验，更不是 G2 Actor/Adam 事务验收。

## 资源与原账

02 监督器确认根进程退出、Job 活跃进程数 0，无墙钟违规。监督结束时本轮累计 149.47272907448104 秒，包含 AST 预检查、两次注册、01 失败与02完整运行；旧科学/工程原账 2885.2303396420666 秒继续保留，总研究消费此时为 3034.7030687165475 秒。该数值不包含之后单独登记的最终小表审查工具消费，最终总账应再加 `FINAL_REVIEW_ACCOUNTING.json`。

worker 的 `wall_s` 已含 prior，不能把它与 prior 再相加。外层使用 registration wall 与每次 ENTRY_CONSUMPTION 的独立 wall 求和。复核时 EVENTS/resource 的运行中 snapshot SHA 与最后 close 后的 SHA 不同是已记录的正常收尾；不可要求它们与最终收据相同，也不可将其误报为数据篡改。

独立最终判定：执行审查 PASS；本 C1 配置 G3 FAIL；下一步停止当前配置。更窄研究主张及限制见 [RESULT_TO_CLAIM.md](RESULT_TO_CLAIM.md)。
