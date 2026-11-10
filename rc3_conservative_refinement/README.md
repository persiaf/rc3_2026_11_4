# 保守修正与停靠后通行信息：独立工程候选

**最新进展：[真实206维训练器集成与工程验收](integration_20261110/README.md)。** 90测试、20真实工程请求与接受/拒绝/恢复/部署均已验收；用户已明确批准准确R03参数及阶段，正式配对已在128按预定停止条件结束，未见ON/OFF路线收益；64/128证据和判读见上方入口。新入口为`runner.py`，不调用旧202维训练循环。

以下为**上一轮零优化器候选的设计及历史状态**；58项结果、原协议和源码绑定保留原证据，不能代替本轮真实验收。原SL、H、mask、奖励、旧研究结果与阶段授权保持不变。研究目标仍是大规模父场景DRL训练、冻结后快速规划不同规模局部完整任务；当前仍为二维代理，非真实IK/实机认证。

## 老师意见与真实现状

最新[small256报告](../rc3_integrated_rppo/EXPERIMENT_REPORT.md)已实现并训练冻结SL、零残差、训练专用Critic及受控PPO，且6144条episode中3072条为完整10×10。P2未通过。旧训练完整RTG、gamma=lambda=1已经存在，不能重新宣称“首次加入长期回报”。

旧主训练1024个Actor步均接受，已保存的平均KL极小；1024个Critic主更新均梯度裁剪。这些观察不证明学习率过大、Critic不准或任一失败因果。[本轮只读诊断](evidence/HISTORICAL_DIAGNOSIS.json)。因此不以减小步长或多训练自动解决，也不重启旧实验。

本候选落实两次反馈：保留原监督基座和启发式H，以零初始化小修正逐步学习；新增用户明确提出的**假设停靠后的剩余任务通行距离摘要**。改变的信息是否有用，须同结构配对试验验证。

## 后续通行特征

设R为当前未完成目标，C为同臂`F & A`形成的任务—候选覆盖关系，D为已有绕障距离。对于当前合法候选j：

1. 先扣除本次服务：`R_after(j) = R \ {t: C[t,j]}`。
2. 对每个剩余任务t，求`d(t|j) = min_{k:C[t,k]} D[j,k]`；有出口约束时只采用能到出口的服务k。k遍历问题的完整服务候选集合，不以窗口边界或当前动作mask擅自删掉未来provider。
3. 对**任务的最近距离**求均值和最大值；不是对候选距离取均值，也不是求解完整剩余路线。所有任务已完成时二者为0。
4. 不可达任务保留在目标集合，raw均值/最大值为inf；网络有限编码以独立不可达比例标记，不能把数值占位0解释为可达。当前mask不变，逐任务可达也不构成完整覆盖路线存在性证明。

新增四维为`mean/10m, max/10m, n_after/n_current_remaining, n_unreachable/n_after`。后两维是明确的计数/缺失语义。策略输入从旧202维扩为206维；Critic状态仍268维。

`lookahead.py`只复用传入D/C，不运行导航。候选到各任务provider的最近距离可在一次请求编码时预计算，动态状态只更新任务集合并汇总。编码与动态汇总的开销都必须计入最终完整请求时间，不能免费缓存跨请求数据制造提速。

## 小修正与评估员

`SmallCorrection`只将末层置零，因此初始合法logits、mask和选择与SL严格相同；幅度乘数为正，避免把梯度永久关掉。候选幅度±0.25为待评审设定，不是已批准训练超参数。小logit变化仍可翻转近并列候选，不保证路线不变或成本下降。

`guard.py`逐个已检查状态限制old→new KL、每候选概率变化和相对SL的TV；超界全部拒绝，可恢复修正头和Adam快照。它目前是独立检查/回滚工具，**没有接入旧PPO循环**，不保证未知状态。旧PPO clip只是损失剪裁，不是所有动作概率的硬界。[PPO原论文](https://arxiv.org/abs/1707.06347v2)

训练专用`RemainingCostCritic`分开预测原始轨迹的剩余L、剩余K和signed H信用，组合`V=−(L+.75K−H_credit)/N_initial`。目标来自完整后续轨迹，不截断远期成本。H可能重写先前前缀，故其信用不能假装成“当前位置以后物理剩余路程”。`G−V_old`为正表示实际完整成本优于预计；优势detach。新增分解损失是候选变化，不声称已经校准。Actor导出内存包排除Critic/Adam；真实部署加载与身份验收尚未实现。

## 同条件对照

主要比较“新增距离摘要开/关”，两组均206维、相同初始化、冻结基座、H、Critic、guard、训练日程和预算。关闭组只把均值/最大两维置零，保留两项计数语义；它仍有原202维导航信息，不能叫“完全没有导航”。两组不能共享更新后的离策略轨迹；同初始策略预热数据若复用，须单独登记来源与费用。

同时保留frozenSL、匹配继续监督和原生Fixed3/Detour16参考，raw与H后J/L/K分别报告，按真实父布局聚类。完整耗时须含新增特征、模型、导航、H、导出与独立验证。详见[实验计划](paper/EXPERIMENT_PLAN.md)及[执行状态](paper/EXPERIMENT_TRACKER.md)；正式训练没有获本候选自动授权。

## 代码与运行边界

| 文件 | 作用 |
|---|---|
| `lookahead.py` | 扣除当前服务、provider距离预计算、动态均值/最大值与不可达语义 |
| `adapter.py` | 固定旧feature encoder，206维输入、距离开/关身份与新packer |
| `model.py` | 零输出修正、分解Critic、完整回报优势与训练用损失 |
| `costs.py` | 完整轨迹L/K/H目标与回报守恒，空任务无伪动作 |
| `guard.py` | 逐状态检查及拒绝恢复，不执行optimizer.step |
| `verify_zero_update.py`、`test_*.py` | 合成张量/数学/接口验收，不读取历史权重或任务 |

不要用旧runner/旧`batch_logits`训练新头：旧代码硬编码202维和残差幅度2，旧D1 schema也不接受新包。当前仅验CPU工程路径，没有CUDA训练、真实规划或部署速度结论。

复核工程用例可使用已有Python3.10/Torch环境，选择不存在的输出目录：

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -X utf8 -m rc3_conservative_refinement.verify_zero_update --output rc3_conservative_refinement/evidence/new_engineering_check
```

上述只运行合成测试。当前绑定、累计验收与限制见[SOURCE_BINDING](SOURCE_BINDING.json)、[工程报告](EXPERIMENT_REPORT.md)、[资源账](RESOURCE_ACCOUNTING.json)和[下一步](NEXT_STEP_DECISION.md)。
