# G0/G1 模块合同（实现前固定）

本阶段不实现或训练PPO。所有新代码在本目录；运行产物在`outputs/`。旧v4/v5/v5.1只读。

## 适配层（根代理）

`adapter.py` 提供 `ResearchAdapter.from_scene(scene, penalty=.75)`、`ResearchAdapter.from_external(bundle, penalty=.75, service_checker=None)`；实例属性 `.problem`为独立复制的旧Problem、`.problem_id`为输入身份、`.anchor`为独立认证的Fixed3 `Certificate`、`.source`为可保存的场景或外部输入。F与D采用不可写数组，不修改调用者输入。`adapter.certify(route, method='rl_search') -> Certificate`会规范化、独立终验并核对抽象/导出成本；失败抛异常。`adapter.payload()`提供重建所需源数据与身份；`ResearchAdapter.from_payload(payload)`重建并核对身份。

## 动作与修复（proposal代理）

公共类型位于`types.py`。函数：

`build_actions(problem, route, *, remaining_work, steps_remaining, history=(), config=SearchConfig(), problem_id='') -> ActionMenu`

history为已经尝试的完整action_context_key序列；菜单只做廉价覆盖/候选统计，不能执行所有修复。actions包含最多16个非STOP动作，再追加STOP；按廉价绕行量排序后交错improve/replace，tie确定。mask列表与actions等长；每项features及feature_names固定。菜单返回本次构造的`work_units`，相同状态一次observe构造后必须缓存菜单供step使用，不能重复扣费或更新动作次序。env负责在执行所选动作时统一扣菜单构造+proposal工作；若已仅够菜单开销，不能再启动修复，直接以预算终止并照实计费。诊断单独调用构造菜单也必须累计其工作。

`propose(problem, route, action, *, config=SearchConfig()) -> ProposalResult`

improve保留旧小窗严格改善语义；replace禁用本次被移除的唯一服务node，可返回目标变差但完整可行的proposal。node 0是depot、候选i对应node i+1。required按整条retained route计算。全局canonical后仍检查覆盖、唯一停靠和有限边。超过任务/候选上限、无完整覆盖等明确失败，禁止以原路由冒充已完成的强制替换。F/A/D保持不变。

proposal计数：attempts=1，candidate_checks=本次扫描的候选配置数，dp_states=实际扩展的有限(mask,last)状态数（含full），dp_transitions=评估的入口连接、有限前驱到新增覆盖目标及出口连接；work_units为这些项之和，失败至少1。menu工作以廉价窗口上下文评价次数及候选扫描计数单列（具体定义固定在action_builder）。时间不是训练工作计数。本阶段不启用proposal缓存，避免未登记查询或陈旧结果；context key仍绑定Problem、完整route、operator/window/banned、剩余预算及caps，为重放和未来缓存使用。

## 环境与运行（environment代理）

`SearchEnv(adapter, config=SearchConfig(), seed=0, *, proposer=None, action_builder=None)`，提供 `observe()`、`step(action_id)`、`state_dict()`、`load_state_dict(state)`、`finalize()` 和可重放工具。可注入proposer/certifier用于针对性故障测试；默认仅使用共同生产接口。

anchor/current/best均独立保存。合法proposal经adapter.certify后，只有J<=best+epsilon*anchor才接受current；best只在更低认证cost时更新。STOP始终为未终止状态的普通合法动作。已真正终止时不再决策（观察可保留STOP但mask不得暗示继续搜索）；预算/horizon/STOP真正terminated，不bootstrap。collector rollout cut标为truncated但不改变环境搜索状态，应bootstrap。observe返回有序actions/mask/features和完整动态上下文，禁止后续重枚举替代采样时记录。

主奖励 `100*(old_best-new_best)/max(anchor,1e-8) - lambda_work*work_units/work_scale`；可选phi=`100*(best-current)/max(anchor,1e-8)`，加potential_weight*(phi_next-phi_old)，真正终止phi_next=0。完整回合望远镜恒等式必须覆盖失败、无变化、变差、恢复、STOP。最终终验失败回退认证anchor时，以实际返回cost修正已累计质量奖励，不能报告未返回改善。记录menu/proposal/verification等工作和各阶段耗时，不当作部署速度结论。

新runtime使用本环境，不能再次调用旧runtime拒绝current变差。提供random、greedy规则和简化ALNS式自适应算子策略入口，均只使用相同有限菜单/修复器/预算；本阶段不把它们称为已调优强基线。禁止隐藏试遍全菜单求真实收益。

## G0 诊断（根代理）

新开发32个10×10场景先固定清单，状态为认证Fixed3初始状态；最多8分支（含STOP）。最终预注册 protocol.json 将早期“先2后3”草案具体固定为**全部场景直接深度3**，没有依据观察到的结果追加深度。从共享有序菜单取前7个合法非STOP条目（improve/replace交错）和STOP；保存全部成功/失败查询、menu/DP/验证开销和状态转移。完整树不是公平策略性能评测。重复前缀重放的费用也记录，不能免费查询后只报告最后一步。

每份case保存输入payload、初态和所有action表/日志，可从state_dict重放；完整性/成本/预算/奖励复核后才讨论G0后效。找不到真实后效则明确负结果，不训练或合成冒充真实正例。
