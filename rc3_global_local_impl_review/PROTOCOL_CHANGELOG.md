# v2到v3提案的训练前变更说明

当前冻结依据仍为S0目录中原v2；`PROPOSED_PROTOCOL_V3.json`与`PROPOSED_SEEDS_V3.json`是本次实施评审形成的未激活提案。旧v1/v2、S0和原future pointer均保留原字节。当前没有正式训练或S3目标结果。

| 项目 | v2问题 | v3提案及变化类型 |
|---|---|---|
| S1/continuedSL | 损失、batch、optimizer细则不全 | 同结构teacher-forcing CE，route内动作均值再batch4均值；Adam1e-4、固定betas/eps/clip；实质配方固定 |
| 权重血缘 | 起点/动量继承不明确 | 每seed untrained→SL256→冻结SL/RL/continuedSL完全相同权重分叉，后两者fresh Adam；保存各哈希/RNG/optimizer；实质比较条件固定 |
| 采样/退化 | RNG、first不足2、空实例/空batch未冻 | phase与row命名独立流、主first参与、forced first排除；空batch停机而不补样，单first保留两个独立suffix辅助；实质随机分布/边界固定 |
| 宏batch执行 | 完整12轨迹图峰值未测 | 固定batch4，逐实例loss/4梯度累积，batch末clip/step一次，执行前必须梯度等价回归；数值语义保持工程选择并明确版本 |
| 更新/耗时账 | 512只覆盖RL，不含控制 | 每seed S1_256+RL512+continuedSL512=1280，三seed3840；S2原7200秒明确定义包括RL/控制/开发，原6144+2048终局不扩大；S1此前缺失墙钟边界另列 |
| S1分布 | mixed只closed、blocked只open；每父图只有一种size | 保留256/原seed bank，mode×endpoint四格各64，64父3size/32父2size；**实质样本分布修订** |
| S3 A分布 | A与size混杂 | 保留108及全部map/task/窗口，A按parent+position+size循环，同group/size均衡；**实质样本分布修订** |
| S3判定 | 未冻结主分组/三seed聚合 | parent内2endpoint、再3seed均值；6个group×size全部过两SL门，12个cache strata过部署门；数值阈值保留，但**决策逻辑明确化属于门槛修订** |
| 管线 | native和额外commonH边界模糊 | S1 teacher nativeDetour+额外H；S3 native不再额外H；神经raw/+H分开，不能把H/搜索收益记成NN；实质管线定义固定 |

保持不变：固定DRL研究目标、原约束/代价J=L+.75K、终局-JH、主/辅助权重1/.25/.25、γ=1、无熵/优势标准化、96/3/6/12实际布局bank、S2主RL512及6144奖励/8192总上限、S3完整请求上限及质量/时间数值阈值。v3不增加预算以追求通过。

这不是已批准的正式执行协议。先完成正式训练runner与权重/RNG/退化/资源回归、固定源码，再由后续获准执行任务明确采用新版本；不能按当前v2缺省条件直接启动optimizer。

提案文稿修订02修正了继承的v2教师实例文件名及“几何门待执行”过期文字，指向本轮已经通过的物理布局证据；未改变提案01的算法、分布、预算或门槛。原提案01与原收据另存于drafts/，修订链及哈希保留，正式训练仍为0。
