# R04独立静态审查

审查对象：R04设计、数学定义、待批消融协议、验收要求和新工程源码。基准身份为 `853eb6b016df3aee08c233894ab6c24beb3765d8`。本审查只读取源码与协议，不导入torch、不实例化模型、不运行测试、数值扫描、CSV算术、optimizer或规划请求。具体执行成绩应由根代理登记后唯一一次工程验收给出。

## 判定

**最终静态判定：GO_FOR_REGISTERED_ZERO_OPTIMIZER_ENGINEERING_ONLY。** 初审REVISE的S01已由根代理与算术审查代理修复，见下方修复复阅；可以绑定精确源码后执行本轮一次工程验收。当前37个单元用例、8组保存logit和人工loss接口的调用关系中，未发现静态可证明必然失败的错误。模块调用2048、backward64、autograd.grad64的上限从有限调用结构看有余量，但本审查不把该判断作为已消费实账或PASS。

设计主体可作为未激活候选继续审查；**联合PPO仍为NOT_READY / NOT_AUTHORIZED**。设计GO或工程PASS都不能升级候选协议、解封数据或许可真实更新。

## 登记前应修复的问题

**R04-S01，阻断当前有界执行：`run_engineering.Meter.charge`只在charge入口检查壁钟。** 旧算术函数的单次完整扫描和新streaming循环内部没有周期壁钟检查；最后一次charge之后，哈希、写盘及最终收据仍可能消耗时间。`status`目前可以在实际壁钟超过900秒后写为PASS。应增加独立`check_wall()`，在长CSV读取期间周期调用、在设置PASS前再次调用，并在最终账中明确cap满足情况。超限须失败、保留原事件与消费，不登记新目录重跑。最终检查仅防止错误PASS，周期检查才限制长扫描中的继续消费。

## 已核对正确的核心语义

- Actor实现是精确的`.25*tanh(f-legal_mean(f))`，非法项先排除，mean未detach，单合法动作归零，冻结base/features不传播梯度。文档与测试明确保留post-tanh均值反例和post-center越界反例；没有错误宣布共同偏移已完全消失，也没有放松`.5`的gap充分锁定界。
- C1采用typed归一分量对象；`as_training_dict()`先恢复raw单位，旧loss再除固定初始n。新loss保留逐状态向量，与旧PPO的trajectory内SUM/离线MEAN分别说明；没有把两种归约混为一个因素。
- 条件键只使用已有宽、高、当前剩余比例。拟合入口硬拒绝非fit、未登记父和重复状态；先条件内state均值、再trajectory等权、最后父等权。unknown条件回退fit-global；三个分量与value使用同一线性组合。当前未实现可训练条件残差网络，未把表估计当成在线训练。
- 工程runner不实例化旧trainer；unit中的参数赋值与fake Adam形状状态属于fixture。不能据此宣称真实optimizer候选接受/拒绝、Adam精确回滚、checkpoint恢复或生产部署验收完成。
- 旧算术复核复用旧`independent_recheck`纯函数已明确标示，不伪称第二套独立实现；新增window/phase矩统计独立实现，输出通过旧`_json_safe`处理NumPy标量、tuple和非有限值后再严格JSON写入。
- 科学协议全部保留PENDING/NOT_AUTHORIZED；原60/S3与真实SL反序列化均不属于本轮工程。

## 科学集成的明确阻断

**R04-S02，生产导出仍不兼容。** 把`RelativeCorrection`赋给旧`LookaheadResidualActor.correction`后，旧wrapper的`actor_only_package()`会调用新对象不存在的同名方法。当前8组检查调用standalone `actor_only_package()`，所以没有覆盖wrapper导出路径。旧production export/loader还可能把相同形状新头标成absolute schema。当前原型范围可以明确保留此阻断；未来必须同时迁移wrapper、export、loader、checkpoint，并在Actor父对象记录semantic_id和特征/基座身份后才可生产使用。当前mapping的严格schema/hash检查不能代替真实SL加载与完整请求验证。

**R04-S03，真实optimizer与恢复尚未验收。** 双optimizer快照、拒绝回滚、累计RNG/游标/请求账及暂停恢复必须另行登记并获阶段授权。当前禁止通过实例化旧trainer来补这个缺口。

**R04-S04，现有科学证据不满足G3。** C1已有改善仍未同时胜条件常数；C2总value改善伴随H信用退化。当前工程不产生新的校准、路线、时间或泛化证据。

## 待批协议的可识别性与门槛

四臂C0/C1×A0/A1能区分Critic参数化与Actor输出语义，且将条件锚点、C2和新增特征排除在核心pilot外，优于把两个因素同时开启后与历史终点比较。同张量初始化仍对应不同的C0/C1物理预测；step0披露与端点记录必要，但不能将结果归因为纯梯度尺度。条件残差候选比较anchor自身、global-anchor残差和conditional-anchor残差，能暴露“只是不同先验初值”的收益；floor改变锚点时仍须新协议精确记录。

G3同时要求两全局口径、父配对95%上界、逐尺寸不劣和分量不退化，是保守的进入决策门槛；它不是校准过的总体5%假设检验。固定16父若任一尺寸少于3父，应按方案报INSUFFICIENT，不能换父或用state数填补。该门槛可能使pilot长期无法进入，这应作为预设取舍披露，不得结果后放宽。

H2的数学结构与有用排序学习假设不逻辑依赖C1先击败lookup。现方案选择用G3阻断整个2×2属于项目范围决策；G3失败只能写“本pilot未获进入条件”，不能写“A1或H2已被否证”。单seed完整成本和时间结果即使未来正向，也只支持该pilot的机制线索。

M3请求预算表达中“66 saved windows, both endpoint conventions”应在精确注册时明确66是否已经包含两种端点。当前`2*66*7=924`将2用于32/64两次评估；若66只是窗口数且每窗另有两种端点，请求数还需乘2。当前不得读取任务或执行请求来消除此文案歧义。

## 非阻断硬化建议

- 初审曾发现`rank_response`未验证log-softmax及ratio最终有限。登记前已增加两类非有限结果的硬拒绝，并在既有nonfinite用例内加入极端finite logit和ratio overflow fixture；没有新增case数量。本项静态关闭，数值结果仍须唯一一次实跑确认。
- `ConditionalComponentBaseline.__init__`只核schema与自带content hash，不能单靠hash证明完整字段、fit来源或分量域正确。当前正常构造入口受限，未来文件loader应额外核expected hash、字段schema、声明父集合与外部精确来源。
- `C1Critic`允许自定义正hidden宽度，但设计固定128；本轮使用默认128没有混杂，未来生产schema应固定或显式登记hidden。

## 后续审查边界

根代理可在修复S01后登记源码身份并执行本轮唯一一次工程验收。独立终审只复阅新输出、实账和源码diff，不另行复跑测试或CSV审计。任何失败按原累计账处理；本静态意见不授予科学阶段权限。

## 登记前修复复阅

仅静态复阅，没有导入torch或运行数值。`Meter.check_wall()`已由每次charge及最终PASS前调用，audit增加每100000项check和每10000条新streaming预测行heartbeat，并在算术结束时检查。该修复关闭S01对当前工程的阻断；最终receipt写盘属于必要收尾，实际wall及cap满足仍以工程实账为准，不能预先保证900秒内成功。

M1拟合归约改为state SUM/episode均值，与拟议PPO和R03一致；报告MSE仍保留两种评价聚合。这是未来未批准的新门槛，不改写前轮MEAN离线证据。G3已明确为保守项目决策及非H2逻辑依赖。M3明确66是已含两端点约定的总case数，924评估请求的歧义已消除。旧wrapper导出缺方法已在实现计划明确列为未来G2迁移阻断。

复阅源码SHA256：`run_engineering.py=28d1dcc7e9a5e40d26a103141592a15dbc0ae9e7a9b3b0d8f8d50869ba479b74`；`evidence_audit.py=d8253042b1a85328f26fdf594cb766af20debcbf0af2c9ec1ac1a179127e758e`。其余复阅文件hash见`STATIC_REVIEW.json`；最终登记仍须核对本机真实字节。
