# R04-M1 实际离线校准报告

本配置结论为 **FAIL；在 step128 按原规则停止**。116 条固定轨迹已全部恢复，step0 与 step128 已实际评价，C0/C1 各完成 128 次更新，共 256 次 Critic optimizer 尝试及完成。step256 状态为 `NOT_RUN_REGISTERED_MIDPOINT_STOP`，没有续跑、替换样本或追加配置。

报告对象是 [attempt02](attempts/02/ENTRY_CONSUMPTION.json)，run_id `rc3-r04-m1-real-20261111-02`。本轮用户基准为 `992981414239da51c662b5060921ceea2312b7a1`；实际执行源码提交为 `3b915f61e954637cf71bdf0dd59a81aa3ddda14a`，绑定 SHA256 为 `8806070b0c8937ceb6fb8e44bf2b1f1c788905ac4c1ad6b6945d837235a78dbf`。原科学协议继续保留其自身历史基准和原字节；不把新入口身份回写为旧协议身份。

**实际环境、输入与口径**

Python 3.10.20、NumPy 2.2.6、Torch 2.12.1+cu126，CPU 单线程，确定性算法启用，Windows-10-10.0.26100-SP0。实际 `torch.ops`、`torch.classes` 被识别为虚拟命名空间，普通项目模块仍按真实源码 SHA 核验，正式使用标志为 true；14 项原 Critic 接口测试全部通过。

第一条固定输入为 `train:OFF:74:2`，任务数 226，状态为 149×268；完整 L/K/H/RTG 标签和来源通过后，同一受监督进程继续恢复余下 115 条。最终共 7992 个状态：33 个 fit 物理父对应 64 条轨迹；16 个离线 holdout 物理父对应 52 条轨迹、3426 个状态。holdout 是历史 R03 训练父中的新离线校准划分，不能称为未见场景泛化测试。尺寸缺失单元保留，没有换父布局或补样。依据：[真实预检](attempts/02/REAL_PREFLIGHT.json)、[完整恢复索引](attempts/02/RESTORED_INPUT_INDEX.json)、[固定数据索引](attempts/02/M1_DATA_AND_SOURCE_IDENTITY.json)。

两配置均为 268–128–128–3，同 seed11622026、初始净张量、共同 8 条父布局平衡批次、Adam 学习率 0.0001、betas=(0.9,0.999)、eps=1e-8、梯度 clip=1。分量损失保持 state SUM/trajectory，再 episode MEAN；评价仍按各自聚合口径计算 MSE，不能把训练归约改写为 state MEAN。没有 C2 value 主损失、条件残差或新增特征。

常数参照仅由 fit 折估计；条件只用窗口尺寸与可观测剩余比例分箱，unknown 条件回退 fit-global。下表中的“条件常数 MSE”均指该 fit-only 参照在同一 holdout 上的误差，未用 holdout 目标拟合。

**Value：初始化与 step128**

| 终点 | 聚合 | C0 MSE | C1 MSE | 条件常数 MSE | C1/条件 | C1/条件配对95%CI | C1/C0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | state_pooled | 1.371889042 | 0.897467012 | 0.247656663 | 3.623835526 | [2.755151410, 5.055881769] | 0.654183382 |
| 0 | parent_equal | 1.595315924 | 0.956957194 | 0.230493981 | 4.151766513 | [3.158996170, 5.424231129] | 0.599854348 |
| 128 | state_pooled | 1.377307810 | 0.709127008 | 0.247656663 | 2.863347186 | [2.083510817, 4.180973505] | 0.514864581 |
| 128 | parent_equal | 1.602134103 | 0.779731488 | 0.230493981 | 3.382871356 | [2.446599088, 4.576223660] | 0.486683035 |

区间直接摘自预登记的物理父布局配对重采样结果，20000 draws、PCG64 seed11622026。state-pooled 保留状态数权重，parent-equal 先对每个物理父的状态误差求均值再等权聚合；两者均报告，未互相替换。来源：[M1_METRICS.csv](attempts/02/M1_METRICS.csv)、[配对区间](attempts/02/PAIRED_PARENT_INTERVALS.json)。

C1 在 step128 优于 C0，但条件参照比值为 2.863347186 / 3.382871356，两种 95% 区间整体高于 1。不能将 C1 相对 C0 的下降解释为满足 G3，更不能解释为路线收益。C0 的组合 value MSE 从 step0 到 step128 略升；没有因该结果改变拟合步数、参照或目标。

**归一 L/K/H 分量**

| 终点 | 聚合 | 分量 | C0 MSE | C1 MSE | 条件常数 MSE | C1/条件 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | state_pooled | normalized_L | 3.052442707 | 1.540444545 | 0.290010337 | 5.311688398 |
| 0 | state_pooled | normalized_K | 0.179192383 | 0.092881783 | 0.005796013 | 16.025117829 |
| 0 | state_pooled | normalized_H | 1.272276326 | 1.063532705 | 0.045900226 | 23.170533233 |
| 0 | parent_equal | normalized_L | 3.427146915 | 1.806511378 | 0.273631160 | 6.601994371 |
| 0 | parent_equal | normalized_K | 0.188436478 | 0.091141707 | 0.005840693 | 15.604606023 |
| 0 | parent_equal | normalized_H | 1.300579712 | 1.088480442 | 0.046043898 | 23.640058534 |
| 128 | state_pooled | normalized_L | 2.856830000 | 0.558707466 | 0.290010337 | 1.926508795 |
| 128 | state_pooled | normalized_K | 0.168219398 | 0.042487020 | 0.005796013 | 7.330387920 |
| 128 | state_pooled | normalized_H | 1.126781055 | 0.060634717 | 0.045900226 | 1.321011306 |
| 128 | parent_equal | normalized_L | 3.189135298 | 0.612840788 | 0.273631160 | 2.239660088 |
| 128 | parent_equal | normalized_K | 0.175521521 | 0.044714422 | 0.005840693 | 7.655671110 |
| 128 | parent_equal | normalized_H | 1.130738862 | 0.058450053 | 0.046043898 | 1.269441893 |

step128 的三个归一分量均超过 C0 的改善要求，且都没有全面胜过条件常数。H/n 是包括历史路径前缀修正在内的终端全路径成本信用，不是物理剩余距离；本次只使用保存的完整 H 标签，没有产生新 H。

**不同尺寸的 step128 holdout 误差**

| 窗口 | 聚合 | 物理父数 | 状态数 | C0 value MSE | C1 value MSE | 条件常数 MSE | C1/条件 | 误差门槛 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10x10 | state_pooled | 16 | 1827 | 1.140738942 | 0.675209308 | 0.251116263 | 2.688831455 | FAIL |
| 10x10 | parent_equal | 16 | 1827 | 1.328736571 | 0.714533425 | 0.218459303 | 3.270785068 | FAIL |
| 4x3 | state_pooled | 11 | 161 | 3.697362031 | 1.678965714 | 0.732132248 | 2.293254694 | FAIL |
| 4x3 | parent_equal | 11 | 161 | 4.114046376 | 1.920293220 | 0.724128774 | 2.651867026 | FAIL |
| 6x6 | state_pooled | 12 | 469 | 1.726718188 | 0.714365645 | 0.173635811 | 4.114160786 | FAIL |
| 6x6 | parent_equal | 12 | 469 | 2.010065229 | 0.832615371 | 0.213340450 | 3.902754353 | FAIL |
| 8x8 | state_pooled | 13 | 969 | 1.268751695 | 0.609402226 | 0.196464217 | 3.101848454 | FAIL |
| 8x8 | parent_equal | 13 | 969 | 1.460562280 | 0.672940106 | 0.201227210 | 3.344180469 | FAIL |

四个尺寸分别有 16、11、12、13 个验证父，均满足至少 3 个父的覆盖要求，覆盖判定 GO；八个尺寸×聚合误差比值均大于 1，误差判定 FAIL。本次 G3 失败不是由缺少尺寸覆盖造成的。

**G3 逐项决定**

| 原检查项 | 状态 | 实际值 | 原门槛 |
| --- | --- | --- | --- |
| value_vs_condition_state_pooled | FAIL | 2.863347186 | <= 0.95 |
| paired_upper_state_pooled | FAIL | 4.180973505 | < 1.0 |
| value_vs_C0_state_pooled | GO | 0.514864581 | < 1.0 |
| normalized_L_vs_C0_state_pooled | GO | 0.195569028 | <= 1.1 |
| normalized_K_vs_C0_state_pooled | GO | 0.252569090 | <= 1.1 |
| normalized_H_vs_C0_state_pooled | GO | 0.053812333 | <= 1.1 |
| size_10x10_state_pooled | FAIL | 2.688831455；父数 16 | <= 1.0 |
| size_4x3_state_pooled | FAIL | 2.293254694；父数 11 | <= 1.0 |
| size_6x6_state_pooled | FAIL | 4.114160786；父数 12 | <= 1.0 |
| size_8x8_state_pooled | FAIL | 3.101848454；父数 13 | <= 1.0 |
| value_vs_condition_parent_equal | FAIL | 3.382871356 | <= 0.95 |
| paired_upper_parent_equal | FAIL | 4.576223660 | < 1.0 |
| value_vs_C0_parent_equal | GO | 0.486683035 | < 1.0 |
| normalized_L_vs_C0_parent_equal | GO | 0.192165189 | <= 1.1 |
| normalized_K_vs_C0_parent_equal | GO | 0.254751791 | <= 1.1 |
| normalized_H_vs_C0_parent_equal | GO | 0.051691911 | <= 1.1 |
| size_10x10_parent_equal | FAIL | 3.270785068；父数 16 | <= 1.0 |
| size_4x3_parent_equal | FAIL | 2.651867026；父数 11 | <= 1.0 |
| size_6x6_parent_equal | FAIL | 3.902754353；父数 12 | <= 1.0 |
| size_8x8_parent_equal | FAIL | 3.344180469；父数 13 | <= 1.0 |

以上全部来自 [M1_GATE_DECISION.json](attempts/02/M1_GATE_DECISION.json)。两个聚合的 C1/条件常数比值同时大于 1.25，触发原 step128 停止规则；两组一起停止。G3 总状态为 **FAIL**，不是未评价，也不是 INSUFFICIENT。step256 没有数值，不能填零或外推。下一步为 **STOP_THIS_C1_CONFIGURATION**，不进入 G2 或 M3。

**相同张量不等于相同初始物理输出**

C0/C1 初始净张量 SHA256 均为 `6ab9370a6754be4cd1fdfc218279c92626dd36f60db3dcfcdb74ae841d6b09a2`。全部恢复结束后，正式初始化前显式重置 Torch、NumPy、Python 随机流；没有用验证目标设偏置。

C0 直接解释为原始 L/K/H，组合 value 为 `-(L+0.75K-H)/n`；C1 直接解释为 L/n、K/n、H/n，value 为 `-(L/n+0.75K/n-H/n)`。相同净输出张量在 C1 下恢复的原始物理分量是 C0 的 n 倍，因此 step0 物理预测和 MSE 本就不同。当前比较不能声称是排除初始化影响后的纯梯度尺度因果识别。C1 typed 输出不会再除一次 n。依据：[初始化记录](attempts/02/INITIALIZATION.json)。

**独立复核与保存接口**

[独立数值复核](attempts/02/INDEPENDENT_NUMERICAL_RECHECK.json)为 PASS：对本次保存表完成 2679495 项检查、223776 条状态分量行和 1680 个汇总组核对；独立确认 C0/C1 各 128 步、最后终点 128、G3=FAIL。本报告只摘录已核实小表，没有重复扫描状态大表或重跑模型。

step0/128 的 C0/C1 均保存了真实 checkpoint 与 export；恢复时核对角色、单位、来源、模型/Adam/RNG/游标/资源，恢复与导出预测逐位一致。[接口结果](attempts/02/CHECKPOINT_INTERFACE_RESULTS.json)的通用 `loaded_as_next_step_state=true` 字段及 `continued once` 描述须限定理解：step0 恢复后确实继续拟合；step128 只完成恢复/导出检查，随后停止，没有 step129。该检查不能替代真实 Actor/Adam 事务 G2，也不证明独立连续运行完全等价。

**主要阶段耗时与资源**

| 阶段 | 实测秒 | 范围 |
| --- | --- | --- |
| heavy_imports | 2.544808 | 阶段起止记录 |
| preflight_and_imports | 3.624704 | 包含子阶段，不能与其重复相加 |
| zero_update_critic_interface_tests | 0.393724 | 阶段起止记录 |
| saved_input_loader | 1.444501 | 阶段起止记录 |
| frozen_SL_loading | 0.402841 | 阶段起止记录 |
| restore_116_saved_trajectories | 96.001371 | 阶段起止记录 |
| dataset_save_and_fit_references | 0.171525 | 阶段起止记录 |
| endpoint_0 | 4.535641 | 阶段起止记录 |
| matched_fit_1_128 | 5.221978 | 阶段起止记录 |
| endpoint_128 | 5.343855 | 阶段起止记录 |
| saved_result_curves | 0.022048 | 阶段起止记录 |
| save_summaries | 0.091008 | 包含子阶段，不能与其重复相加 |
| independent_numerical_review | 8.827779 | 阶段起止记录 |

主要开销是 116 条保存轨迹的来源读取与 268 维状态恢复，96.001371 秒；C0/C1 配对拟合 1–128 共 5.221978 秒，独立数值复核 8.827779 秒。资源记录另给出 saved_view_restore=62.843991 秒、base_encoder_forward=2.812073 秒、base_state_features=23.986169 秒；这些是恢复阶段内的分项，不能与恢复总时重复相加。前后未命名开销、监督器启动/退出等由完整墙钟收据保留，阶段表不是对总时的强制分摊。依据：[阶段 TRACE](attempts/02/PREFLIGHT_TRACE.jsonl)、[工作进程资源](attempts/02/RESOURCE_ACCOUNTING.json)。

attempt02 监督器完整实测 135.797000000 秒，包含导入、读取、恢复、拟合、评价、保存与进程退出；本轮截至该入口结束累计 149.472729074 秒。随后仅报告小表读取及整理另记 [REPORT_ACCOUNTING.json](REPORT_ACCOUNTING.json)，**最终新增消费以根 RESOURCE_ACCOUNTING.json 为准**。旧消费 2885.230339642 秒保持原账，包括旧科学失败 2859.2566759 秒、11.586754084 秒及工程 14.386909658 秒；本轮 attempt01 的失败也保留并计入新增累计。没有用目录或 run_id 重置额度。

Actor optimizer、新场景、导航、教师、native、H、原60、旧S3全部为 0；本次真实 Critic 完成 256 次反向与更新。监督器确认工作进程退出、Job 内存活进程为 0，无执行墙钟违规。R03 的 128 停止和 256 禁止续跑、既有 ON/OFF/匹配监督负结论保持不变。

**下一步结论**

本 C1 配置停止。当前证据支持“相同预算下 C1 的离线归一分量及 value 优于 C0”，不支持“已胜过可观测条件常数”“长期价值足以进入联合 PPO”或“路线已改善”。本次 M1 失败不否证居中 Actor 假设；A0/A1 生产适配与 G2 仍是未验收事项，本轮不自动开展。[生产迁移状态](PRODUCTION_MIGRATION_STATUS.md)列出剩余接口边界。
