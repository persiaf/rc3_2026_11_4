# R04代码与CSV审查

结论分层：**设计和独立接口原型可进入本轮零优化器工程验收；科学训练器/生产部署迁移仍为REVISE，不允许联合PPO。** 最终独立审查及实际测试收据另列，不由本文预填测试PASS。

## 历史诊断逐项检查

| 对象 | 审查结果与保留限制 |
| --- | --- |
| 前轮data.py与split | physical_key整父隔离，ON/OFF pair同折，50父33/17；holdout仍是历史DRL见过父布局；不能把8150状态作独立样本 |
| 前轮actor.py与ACTOR_RANK_RESPONSE | before/after准确检查点与collection概率核对；pre/post ratio独立；2426/8144 strict gap>.5、6单合法、92局部排序变化而top1=0保留 |
| ACTOR_CANDIDATES.csv.gz | 全合法候选和6个权重/特征条件可复算；非法候选缺原缓存，原mask/logits经保存前缀重建；并非当时原始完整张量直接落盘 |
| 前轮critic.py与CRITIC_MATCHED_CALIBRATION | C0/C1/C2同净张量但物理初值不同；离线state mean区别于旧PPO sum；C1改善但未胜条件常数，C2有H代价 |
| CRITIC_STATE_PREDICTIONS与STRATIFIED | 7分量、两聚合、窗口/阶段与常数都保留；原在线Critic为混合checkpoint描述，不能伪装第四个新拟合配置；分层只有1父时不作稳定结论 |
| 前轮run.py/独立recheck/TRAINING_LOG | 固定共同256批次、768真实更新和失败累计证据保留；本轮只读算术审计，原闭合账不重开、不覆盖原NUMERICAL文件 |

本轮EVIDENCE_AUDIT对全部指定CSV和源码做hash及完整算术核对，复用旧纯审计函数的部分明确标为维护复核，另用新的streaming矩统计复算全部窗口/阶段表。实际结果以收据为准，不把软件PASS解释为科研成功。

## 生产级阻断

1. **P1：Actor导出语义错标。** 旧`runtime.export_actor`仅依据206维/head尺寸等构造旧absolute schema。将RelativeCorrection直接塞旧adapter后调用它，可能得到同形状但错误语义的权重包；父adapter的旧`actor_only_package`还会调用新head不存在的旧同名方法。必须同时升级adapter、export、load、父Actor semantic_id与checkpoint，拒绝跨schema。新mapping原型严格拒绝旧包，但未修改或运行生产loader。
2. **P1：Critic单位与typed返回值。** 旧trainer直接按raw字典遍历tensor并再次除n；C1若把normalized字段放入raw键会二次归一。新typed对象和raw n倍适配只解决接口契约，未来`_admit/_critic_step/rollout/logging`仍须显式改造；不能直接运行旧入口。
3. **P1：同形状checkpoint不可互换。** 旧restore仅靠维度/张量hash不能确认absolute/relative和raw/normalized。新schema必须把语义、单位、归约、条件表来源、冻结SL/H、RNG/Adam/游标与累计资源一起绑定；不恢复R03。
4. **P1：真实guard事务未在新结构下运行。** 本轮仅手工fixture和fake-state恢复。真实Adam接受/拒绝精确回滚、暂停恢复、真实路线actor-only等价须另行登记、授权和验收，不能用37单元测试代替。
5. **P1（仅可选分支）：条件残差网络未实现。** 本轮条件表/lookup/分量合成可检查，但可训练残差网络、正性link和实际文件来源边界仍是候选设计。不能把纯组合接口当已训练长期Critic。

## 数学与科研风险

- **P2：precenter不等于postmean0。** tanh非线性会留下共同分量；额外post-center可能使某动作超±.25，不能偷偷加。中心化不改变最大pairdiff .5，也未证明学到有用排序。
- **P2：guard不等于路径证书。** 近tie首选翻转可以满足KL/prob/TV；仍须完整覆盖、导航、代理可达、安全及raw/H验证，不宣称真实IK。
- **P2：初始化和归约混杂。** C1与C0同净张量不等于同物理初值；本轮不追加拟合来“控制”它。未来比较保留初始行并明确归约，不能将mean/sum与Actor/Critic因素混在一起归因。
- **P2：条件先验信息边界。** train-fold最终标签可估计先验输出，但最终路径长度/H不能作为运行时condition；validation不能参与估计、fallback或类别选择。输入已包含window/progress，旧误差尚不能证明必须增加新条件信息。
- **P2：置信范围。** “C1明显更好”是本数据数值结论，尚无对应统计显著性与稳定泛化证明；本轮没有新增训练或路线结果。

## 本轮可确认的实现范围

新增代码位于本目录：相对head、严格mapping loader、typed C1、fit-only条件表、排序/ratio接口及独立人工测试。没有改旧科学源码、历史模型、用户Word或根.gitattributes。源码和输入在首次工程计算前绑定，全部optimizer与科学请求cap为0；实际消费及失败保留在本目录RESOURCE_ACCOUNTING。

未完成的生产阻断会在R04_ACCEPTANCE_CRITERIA的G2保持未通过；工程成功只允许记录“原型工程通过”，不得升级为“联合PPO已获许可”或“R04性能改善”。
