# 训练器工程验收

**PASS（工程）；Formal S1/S2 REVISE。** 所有optimizer均在`ENGINEERING_VERIFICATION_PROTOCOL.json`与执行绑定登记后运行，预算32次、24真实终局/H、900秒。v3未批准，原60和S3目标任务未消费。

## 条件与实际结果

CPU，torch2.12.1+cu126，torch threads2/threadpool1，deterministic algorithms开启。小fixture采用手写C/D/F及mock终局；真实probe仅加载原三个S0父图和初始64维模型，未新生成场景、任务draw、导航或教师标签。

| 验收组 | 结果 |
|---|---|
| 批次联合/累积梯度、LOO公式、首动作、合法mask和完整回报 | PASS；max abs7.1525574e-7、max relative4.3053528e-6；容差事前冻结 |
| 随机流、坏标签、空批、非有限、单合法首动作 | PASS；空批/非有限0更新，单首动作零信号1次登记更新 |
| S2连续4 vs中断2+恢复2 | PASS；模型/Adam/RNG/游标/ledger/动作精确；实际共8次更新 |
| S1连续3 vs中断1+恢复2 | PASS；同上；实际共6次更新 |
| 权重分叉、fresh Adam、缓存版本失效 | PASS；共3次工程更新，未训练/冻结权重不变 |
| 真实完整父场景batch4 | PASS；1次工程RL更新，12次真实H及完整导出/验证，失败0 |
| v2缺配方/v3无批准/S3及错误输出路径 | PASS；正式入口在任何模型/查询前拒绝 |
| 混合抽样及恢复元数据 | PASS；1024 full+1024 local，size342/341/341，无目标任务生成 |
| 生产S1/RL更新循环、journal与checkpoint | PASS；单位provider/pool各1次更新，非真实数据生成链 |
| 教师池partial/首失败/STARTED后重启拒绝 | PASS；4个预期负例，0optimizer/教师/场景/导航 |

前9组为`acceptance_01`，21次Adam，保存墙钟14.202022秒；末组为`acceptance_02`，零更新、1.531342秒。原始RESULT、完整journal、检查点及预期损坏检查点均保留。累计20次小fixture/mock终局更新和1次真实百桌更新；**正式S1/S2科学更新0**。全H内部搜索单列，不算网络收益。

真实batch4的梯度norm从888.473763裁剪至0.999999860，模型参数发生变化；这验证真正执行optimizer，没有证明有用学习。12次H/验证均完成，缓存命中0。进程峰值RSS931,483,648bytes，属于本CPU fixture测量，不代表未来所有任务峰值或GPU需求。

## 版本覆盖与独立复核

两次执行绑定源hash、输入和数值容差。01之后只修TeacherPool残留拒绝/STARTED分支并新增对应case；engine/rng/checkpoint/data/approval与01一致，生产更新循环也未改变。`source_revisions/binding01`保存旧runner/verification；02没有重复已通过优化或消耗新H。`VERIFICATION_RESULT.json`明确该覆盖关系并绑定当前源码。

`RESUME_PAIRS.json`提供连续/恢复检查点对，独立审查直接比较保存tensor、Adam、流状态、计数和全局RNG，并核冻结协议/registry/pointer原字节；不重复模型或优化器。审查结论与耗时见`INDEPENDENT_AUDIT.json`及`INDEPENDENT_REVIEW.md`。

## 测量边界

计时包括验收主体及夹具载入/检查点IO，始于Python/torch导入后。首次进程启动/导入、代码编写与静态审阅未全部单独精确计时，资源账标为未计量，不虚构总任务墙钟。正式阶段累计计时、checkpoint关闭和失败账已有实现；完整正式数据/教师/开发链尚未实际消耗。

本轮证明训练系统工程通路可用；不证明新任务银行全可行、S1拟合效果、DRL独立质量增量、冻结S3速度或真实机器人物理安全。未执行阶段不作成败判断。
