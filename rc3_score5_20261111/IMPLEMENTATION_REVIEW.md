# 实施审查：GO（仅登记工程验收）

准确SL256来源沿用既有完整字节/上游S1谱系检查。冻结39936参数，仅允许key.weight、dynamic.weight、nav.weight、query.weight、query.bias共25152参数。新适配层拒绝全Actor/重复/外部optimizer；旧全参数Trainer保持原字节，不强行复用其全参数校验。

使用原GlobalLocalPolicy static_data与原score路径。仅冻结表示和prefix GRU可no_grad/detach；key必须在grad score中重投影，禁止拿旧detached206残差输入替代。数值keys只在无grad采样/推理的同权重缓存中复用；实际score验证problem/model/tensor版本/revision/epoch，更新、回滚、恢复后重新encode。

合法mask、转移、覆盖、入出口、J=L+.75K及gamma=lambda1完整终端回报/H/导航来源/贪心解码不变。新协议统一规定PG/SL相同增量KL与概率步幅、clip和Adam；没有旧A1残差±.25、base-TV永久SL边界或base-KL罚项。原score内置10*tanh保留。

复用真实Torch来源分类、严格已保存输入恢复及Windows Job监督器。在首个实际模型请求/optimizer前冻结源码/数据日程/数值容差、24请求/12Actor尝试/600秒工程上限。计划14完整请求、5真实Actor尝试（PG/SL、注入拒绝、恢复后双实例下一更新），无Critic、新任务、新导航、新教师或native。工程失败消费保存，不免费换数据或清零。

工程目标是初始等价、五模块梯度、冻结不变、白名单、版本/缓存、真实更新回滚恢复和导出输出一致；没有教师Top1或固定32拟合门槛。全部工程权重禁止正式初始化。

首轮正式方案五张量PG/同五张量继续SL，66已暴露案例同期FrozenSL/Fixed3/Detour16，固定64/128。PG50%full，现有监督标签只有local，不能冒充同数据/FLOPs。正式科学初值始终重新从合法SL256加载。Science授权false，controller及worker双入口先验拒绝未授权调用，独立保留集也不授权。工程GO不是正式执行许可。
