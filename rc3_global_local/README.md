# 全局—局部条件化DRL：S0

研究目标固定见[RESEARCH_OBJECTIVE.md](RESEARCH_OBJECTIVE.md)。本轮代码和研究记录位于独立分支`codex/global-local-s0`，基准`6f595cf`；默认Fixed3／Detour及旧研究证据保留，按既有用户要求不自动推送。

主要入口：

- [S0报告](REPORT_S0.md)、[实验计划](paper/EXPERIMENT_PLAN.md)、[状态](STATUS.json)。
- [原约束内核核查](KERNEL_REVIEW.md)、[文献核实](LITERATURE_VERIFICATION.md)、[事前审查](PREREG_REVIEW.md)。
- [独立保存结果审查](FINAL_REVIEW.md)确认S0工程一致性，未证明DRL性能；`audit_saved.json`单列保存几何验收与v2静态审查，未重跑H或模型。
- `S0_protocol.json`登记当前S0；未来正式执行采用`FUTURE_DESIGN_STATUS.json`指向的`S2_S3_protocol_v2.json`及`future_seeds_v2.json`。原v1设计与S0登记原字节保留，v1不再具有正式执行资格。
- `parents/`保存原父场景和F/A/D/导航前驱数组；`views/`保存窗口与全局ID映射；`plans/`保存完整端点轨迹、停靠位姿和逐臂任务分配；`episodes.jsonl`保存每步动作/奖励/终止与重放证据。
- `gradient_probe.json`、`resources.json`与`terminal_journal.jsonl`保存无更新梯度探针及独立资源账；两类合成单测单独记账。

这是原全局地图中的**局部任务窗口**：导航继续使用全部100桌和障碍；所有服务目标任务的允许候选（含窗口外候选）保留。不同入口/出口的成本、导航折线和端点独立验证，不省略连接代价。

有效服务必须同一只臂同时满足F与A。代理模式只支持原固定升降高度、独立单臂服务语义；不同高度需原外部F/A接口，耦合双臂任务需相应模型。当前轨迹导出不虚构关节角：`arm_joint_trajectories=null`，真实关节轨迹认证未完成。

使用项目解释器`rc3_learning/.venv/Scripts/python.exe -B`：

1. `-m unittest rc3_global_local.test_environment -v`：独立合成环境测试。
2. `-m unittest rc3_global_local.test_policy -v`：独立合成神经契约测试；会执行结构backward但不优化权重。
3. `-m rc3_global_local.register`：只允许首次登记，检查独立审查、单测和源码，再核对种子。
4. `-m rc3_global_local.run_s0`：只允许一次真实S0工程探针；有run_started即拒绝隐式重跑。
5. `-m rc3_global_local.report_s0`：仅整理保存结果，无新增研究调用。

登记后不要原地重写unit receipts、源码或协议；保存证据复核使用独立审查入口。训练/基线完整runner尚须未来执行任务绑定，当前没有S2 optimizer入口，不会自动训练。

设计v2修正了regular生成器“不同seed但相同物理几何”的划分风险。未来父图均用mixed/blocked变化模式；任何S1教师或optimizer之前须按canonical map_key核对实际唯一性和训练/开发/未见布局不交，碰撞则停机而非换seed。当前只登记标签与规则，没有生成或查询S3任务，不能提前声称18个未来布局的几何独立性已被验证。

S0已完成：3父图18窗口，63条完整轨迹、63次精确缓存重放；约束验证和奖励一致性通过，1次真实backward探针后权重不变、optimizer0。原始完整父输入数组和轨迹均已保存。正式S1/S2/S3保持未启动，后续沿固定DRL目标进入执行绑定与训练，不将本软件验收替换为纯监督或规则研究成果。
