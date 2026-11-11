# R04实现与迁移审查

本轮实现仅是可测试的接口原型，不是已集成的科学训练器。旧R03及前轮诊断代码全部冻结；没有新训练入口、恢复命令或自动执行候选协议的路径。

| 接口 | 原行为/风险 | 本轮已完成 | 将来科学集成前仍需完成 |
| --- | --- | --- | --- |
| Actor输出 | `SmallCorrection=.25*tanh(f)` | `RelativeCorrection=.25*tanh(f-legal_mean(f))`；非法项先屏蔽，mean保持梯度，单合法输出0 | adapter注入新类型并在Actor父对象记录semantic_id；禁止靠shape猜语义 |
| C1输出 | raw分量dict、旧loss再除n | typed `C1Prediction`；归一单位明确，`as_training_dict()`恢复raw n倍，等价loss检查 | `_admit/_critic_step/rollout`显式适配，不将normalized字段伪装成raw；旧tensor日志循环与typed metadata分开 |
| 状态归约 | 旧PPO为每trajectory state SUM，再episode均值；前轮离线为state MEAN | loss函数返回per-state，调用者必须显式选择归约 | 新离线门槛拟合与核心pilot都固定R03 SUM/episode均值；MSE评价仍是各口径的状态均值，不能把评价均值误作拟合归约 |
| 条件先验 | 前轮仅强常数对照 | fit-only条件表、typed可观测键、fallback和分量一致合成；负L/K硬拒绝 | 尚未实现可训练条件残差网络；正性参数化、baseline版本和冻结时点须独立注册后才能研究 |
| PPO/guard | 旧训练器在constructor创建Adam | 只调用stateless PPO loss与原guard；fake状态精确回滚测试 | 新科学runner双optimizer、每候选前Actor/Adam深拷贝、guard后严格回滚、实际日志、源/预算绑定；本轮禁止实例化旧trainer |
| 冻结SL | wrapper.train需持续保持encoder.eval、requires_grad=False | 人工冻结encoder与旧adapter模式检查；保存SL logits上零头严格一致 | 准确SL256及上游S1谱系必须真实加载核对；真实任务等价/完整验证须另行授权 |
| 部署 | 旧export仅看206/hidden/amplitude，可能错标absolute schema | 新mapping package/loader严格拒绝旧schema、未知字段和Critic/Adam；只检查correction | **旧export与load必须同时升级**，增加correction语义、特征及base身份；真实文件hash/SL恢复/路线等价/冷启动计时尚未运行 |
| checkpoint | 原tensor形状无法识别C0/C1或absolute/relative | 新语义与单位标识已定义 | 新checkpoint schema必须记录Actor/Critic语义、初始n规范、归约、条件表hash、RNG/Adam/游标与累计账；不得加载R03残差当作相对头继续训练 |
| 记录 | old ratio常为更新前on-policy1 | 原始f、legal mean、precenter f、bounded delta、post mean、mask及pre/post ratio/rank接口 | 实际更新与推理逐请求计全部开销；将zero-init、actual accepted/rejected、raw/H成本与时间分别保存 |

## 新schema边界

Actor为`relative_pre_tanh_centered_v1`，C1为`rc3-r04-c1-per-initial-n-v1`，条件表为`rc3-r04-fit-only-conditional-components-v1`。同形状旧权重不代表同语义：旧absolute head权重通过居中后策略会改变，旧raw Critic权重改成normalized解释也会改变单位。新科学pilot必须同SL起点、重新零初始化残差并显式初始化新Critic；不回写旧模型。

新Actor-only映射仍只含correction与冻结base路径/hash；不包含Critic、条件表、Adam或训练器。映射测试不等于生产loader/完整路径验证。真实文件序列化、来源hash、base谱系和完整请求计时是后续集成门槛，不得由本轮人工fixture代替。

旧`LookaheadResidualActor.actor_only_package()`还会调用correction自身的旧同名方法；新RelativeCorrection未实现该旧方法。人工wrapper仅检查train冻结和logits，不走这个旧导出路径。将来必须一并迁移父adapter与export/loader，不能将standalone mapping测试说成现成的drop-in兼容。

## 本轮工程的精确覆盖

- Actor18项、Critic19项独立单元测试：人工张量/人工训练折统计，只有前向/梯度检查，没有optimizer。
- 保存logit检查按旧候选CSV字节顺序固定前8个状态，使用人工零206维特征和人工冻结encoder；不重建任务，不调用真实SL。验证对任意有限合法base logits的零头等价、mask、mapping加载、排序及ratio。
- 人工2状态PPO/C1接口验证loss有限、Actor/Critic梯度隔离、C1 raw适配与原loss等价；不是训练、完整checkpoint恢复或Adam真实更新验收。
- 旧CSV全量只读复核写新R04收据，旧NUMERICAL_RECHECK不覆盖；窗口/阶段表另由新streaming矩统计复算。

## 未来预算与风险处置

当前工程登记900秒，最多2048个nn.Module调用、64次backward/64次autograd.grad、37个unit case、1组离线接口与1次CSV审计，全部optimizer/科学请求为0。失败先记账，不以换目录重放。若工程失败，只修工程实现并在原累计预算下登记修订；不通过放宽数值或科学阈值“修好”。

未来离线校准及四臂pilot分别列为PENDING授权；预算不能从R03剩余额度继承。代码/初始化/数据/奖励/H/安全规则若有实质改变，必须新增协议版本、说明与源码身份；不能改写R04冻结提案或原活动指针。
