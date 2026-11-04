# PPO 接口准备状态

本轮前置机会门槛未通过，**未创建 PPO 授权文件，未启动 PPO 或新的环境交互**。`train_ppo_pilot.py`、`collector_pilot.py`、`worker_pilot.py` 仅为准备代码；`outputs/ppo_prepared_receipt.json` 记录 10 项纯张量/模拟 collector 单测通过。没有执行新 multiprocessing worker 与真实环境的端到端验证，不能声称接口已通过实际训练验收。

入口读取 G2.1 自身的 `outputs/ppo_authorization.json`，要求 `authorized` 为 true、协议哈希匹配、单一预注册 seed、动作上限为 1024/2048/4096、选定 BC 路径和哈希、原 G2 协议哈希及前置证据文件哈希。`gates` 中 toy、opportunity、early_stop_rules_frozen、one_step_control_ready 必须全部为 true；运行前还会对选定 BC 用原教师输入复核完整合法动作机会门槛。

若未来具备独立授权，代码拟采用冻结 G2 的数值 PPO 超参数和 8×32 采样，最多 4096 次提交动作；1024/2048/4096 保存检查。G2 的自动环境 validation 不会调用。prepared cache、提交步骤日志、失败日志及训练输出均限于 G2.1，禁止自动恢复和自动扩预算。

日志区分原始未标准化 STOP/继续 GAE、critic MSE/解释方差、真实 proposal 计数增量、至少两次 proposal 的完成回合，以及相同场景相对初始 STOP 的回报。初始 STOP 使用零势函数的冻结语义计算已构建菜单费用，不额外执行 STOP 回合；若实际首动作 STOP，则交叉核对该解析参照。未知质量不能填零。训练回合墙钟包含优化器及工作进程等待，不能当作部署完整规划耗时。

授权格式尚未实例化；下一轮若获准，仍需先进行范围明确、计数完整的真实接口验证。当前不因准备代码存在而绕过失败门槛。
