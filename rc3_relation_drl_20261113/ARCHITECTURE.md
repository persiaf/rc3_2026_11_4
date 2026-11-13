# 关系Actor实际结构

模型身份：`rc3-navigation-remaining-coverage-relation-v1`。实现入口为[actor.py](actor.py)的`RelationActor`。默认隐藏64维、2个关系块、4头注意力、query分块64；无dropout、Critic、任务分配头、独立机械臂动作或STOP改变。

本次唯一机制假设是**显式候选间导航关系与动态剩余覆盖关系是否改善构造决策**。旧Actor已能选择停靠集合与访问顺序，原环境也已累计到完整H后成本。全参数训练、完整回报或组合决策本身均不作为新增贡献。

## 数据与关系

沿用历史`static_data`的任务5维、候选7维和父场景14维输入，从`LocalProblem.F`读取已经同臂相与的有效`E[task,candidate,arm]`。不修改原问题、环境mask、H、J、导航求解器或独立完整验证。

每个关系块顺序为：任务从合法E边聚合候选；候选从合法E边聚合任务；候选以真实`D[1:,1:]`进行导航关系自注意力。E的左右臂标记进入每头边偏置；导航偏置使用`D/10`和`log(1+D/10)`。无限导航边被完全屏蔽。所有合法候选保留，不进行Top-k或邻居剪枝。

当前RC3父导航仍由原`Parent`约束为无向，局部符号节点0保持既有入口/出口语义。Actor读取这份D；本实现没有把MATLAB有向导航接入RC3求解链。

动态剩余覆盖采用候选到任务的可学习支持注意力。静态编码缓存其初始支持权重与value，解码每步乘剩余任务mask并重新归一，再聚合支持特征；在可表示数值范围内等价于对同一attention logits重新掩去已完成任务。它不同于覆盖均值聚合，同时免去每步重做query/key投影。当前/出口真实距离和覆盖增益保留在解码输入，前缀GRU保留访问顺序信息。

## 接口与缓存

```python
actor = RelationActor()                    # 独立新结构；不加载旧SL256
encoded = actor.encode(problem)
cache = actor.fork_cache(encoded)          # 每条轨迹一个prefix，静态图可共享
logits = actor.score(env, cache)           # 非法动作严格为 -Inf
```

缓存绑定问题对象及identity、模型对象、参数名称/对象/`_version`/dtype/device/requires_grad、加载epoch、梯度模式和train/eval模式。正常optimizer更新、参数替换、freeze、设备变更或`load_state_dict`后旧缓存拒绝使用。代码不使用绕过PyTorch版本计数的`parameter.data`写入。

同题四轨迹可共享一次可微静态编码并在一次backward合并损失；跨不同backward不能复用已经消耗的计算图。每条轨迹独立`fork_cache`，避免前缀相互污染；共享原cache交错调用时也会按前缀一致性重建，但推荐显式fork。空任务可编码为空候选张量，`score`拒绝终止状态；空任务的零成本/出口处理仍由环境负责。

分块控制attention临时query分配，不改变全边图；训练保留的计算图仍有二次边数量成本。最大规模显存、耗时及吞吐以M1实测为准，不以此结构说明替代验收。

`test_actor.py`仅含人工构造的接口/梯度测试，无历史或保留数据、完整规划、H或optimizer。测试是否已执行及次数以本轮统一工程记录为准。
