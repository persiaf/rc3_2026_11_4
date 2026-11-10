# 算法、约束与接口设计

本页说明实际代码与候选方案，不修改任何算法。默认规则入口与科研DRL环境是两套调用链，应按各自schema调用。

## 问题定义

父场景保存几何、桌/任务/候选全局ID和导航证据。局部窗口改变目标任务集合；底盘可利用完整父环境和窗口外合法候选。每个任务必须由一个候选和某一只臂完成。

科研接口中F和A均为`任务×候选×2臂`布尔表，覆盖关系为`C[i,j] = any_arm(F[i,j,arm] and A[i,j,arm])`。不能使用`any(F) and any(A)`让左右臂交叉满足。D保存绕障导航距离，不可达为正无穷，路径前驱用于展开折线。输入身份包含来源hash、几何和必要导航字段，兼容旧schema时不能丢弃`rejected_edge_count`或重算教师。[问题实现](../rc3_global_local/problem.py)、[导航修复](../rc3_navigation_compat_20261106/README.md)

覆盖访问序列含指定入口和出口。历史可比目标为`J = L + 0.75 K`，L是绕障路径长度，K是有效服务停靠次数。0.75为等效距离罚项，不能当作实测秒数、能耗或机械臂动作耗时。

## Fixed3与Detour16

[rc3_v4/runtime.py](../rc3_v4/runtime.py)构造recipe `(0,1,7)`的候选初解，按成本选取能通过独立验证的Fixed3锚点。Detour在合法局部修复动作中按`detour_excess`选择；每次修复须保持覆盖与成本不增，最终方案独立认证，失败保留已认证锚点。

Detour16对应`--method detour --budget 16`，即最多16次修复调用；可能因没有合法动作提前结束。它不是另一套网络，也不是16秒。默认CLI预算仍为8，本轮显式给16，不改默认值。软时间预算允许当前原子修复和最终验证越界，不能宣称硬实时。[规划入口](../rc3_v5/plan.py)

默认代理入口以起点闭环回基地，F由球壳/扇区代理产生；不等同外部F/A证书入口，也不能用本轮两个闭环示例证明科研环境的非同端点局部任务性能。[外部入口契约](../rc3_v5/EXTERNAL_SCHEMA.md)

## 全局—局部构造式DRL

[Environment](../rc3_global_local/environment.py)状态含剩余目标、当前候选、入口出口与有序前缀；合法动作须有正目标覆盖增益、从当前点可达且能够到出口。全部任务完成才终止，不允许用STOP提前结束；空任务也应返回合法入口到出口路径。

每步奖励为移动与停靠负成本，终局补出口及冻结H修正；原实现检查`sum(rewards) = -J_H`。稠密成本奖励已经存在，不能重复包装为新增贡献。H为冻结完整后处理，须单列其收益；学习方法raw与H后结果分别报告。

[GlobalLocalPolicy](../rc3_global_local/policy.py)使用父场景表格token、任务—候选覆盖聚合、动态导航特征和有序前缀GRU。它并不是参考论文中global/local两路CNN的直接复现。[训练引擎](../rc3_global_local_trainer/engine.py)保存原监督/策略梯度路径与恢复谱系。

## 后续Integrated RPPO

[模块说明](../rc3_integrated_rppo/README.md)给出冻结SL256基座、零输出残差Actor、独立Critic与逐动作clipped PPO。准确活动小预算协议由[EXECUTION_PROTOCOL](../rc3_integrated_rppo/EXECUTION_PROTOCOL.json)指向，源码修订须结合其绑定阅读，不能只按最早PROTOCOL推断正在执行的版本。

本轮只查阅已提交来源和部分收据；不调用runner、infer或恢复入口，不补跑未完成阶段。现有模块和部分运行证据不证明该方法已改善规划质量，最终结论仍缺闭合终报告。

## 交接示例的区别

ZIP中的`src/rc3_handoff`为新建教学接口：0-based局部动作、raw回报、无科研冻结H；科研环境动作节点从1起，0为符号端点。两者的candidate ID、service node、entry/exit映射必须显式转换。[逐项接口说明](HANDOFF_PACKAGE.md)
