# 注意力策略＋模仿预训练＋强化学习：可运行原型

本项目在相邻的 `rc3_baseline` 上增加可训练策略。**本次已经实际训练了神经网络**，并完成检查点保存、重新加载、独立测试和路径校验。交付范围是小规模端到端验证，不是完整百桌训练或论文性能结论。

环境继承 RL4CO 的 `RL4COEnvBase`，使用 TensorDict 表示批量状态；策略、行为克隆和 REINFORCE 主循环由本项目实现，没有直接套用原版 TSP 策略，也没有使用 Lightning Trainer。接口依据 [RL4CO 自定义环境教程](https://rl4co.ai4co.org/examples/3-creating-new-env-model/)，训练思路参考 [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475)。这里的距离偏置、覆盖聚合和动作状态是针对本问题的改写，不是对原论文性能的复现。

## 运行方法

**以下命令均从包含 `rc3_baseline` 和 `rc3_learning` 的工作目录执行。** 本机虚拟环境已安装并验证，可直接使用，不需要激活脚本：

```powershell
& ".\rc3_learning\.venv\Scripts\python.exe" -B -m rc3_learning.train
```

默认使用 128 维嵌入、2 层编码器、4 个注意力头、批量 8、Adam 学习率 1e-4；生成 64 个训练实例、16 个验证实例，完成 15 轮 BC 和 3 轮 RL，并在 12 个独立测试实例上评估。训练/验证规模为 2×2 和 3×2；测试规模额外包含 4×3。输出到新的带时间戳目录，已有结果不会被覆盖。

复现本次交付配置并另存结果：

```powershell
& ".\rc3_learning\.venv\Scripts\python.exe" -B -m rc3_learning.train --train-instances 64 --val-instances 16 --test-per-scale 4 --bc-epochs 15 --rl-epochs 3 --output rc3_learning/run_repeat
```

自定义规模和训练轮数，例如：

```powershell
& ".\rc3_learning\.venv\Scripts\python.exe" -B -m rc3_learning.train --train-scales 3x2 4x3 --eval-scales 3x2 4x3 6x6 --train-instances 128 --val-instances 32 --bc-epochs 30 --rl-epochs 10 --batch-size 8 --output rc3_learning/run_custom
```

该命令是接口示例，本次没有执行它。可使用 `--device cpu` 强制 CPU，`--no-plots` 省略绘图，`--config rc3_baseline/demo_config.json` 设置演示几何参数。`--init-checkpoint` 加载已有权重继续训练，但会建立新优化器；它不是恢复随机数、批次位置和优化器状态的精确断点续训。

独立加载交付的检查点，在新种子上推理和比较基线：

```powershell
& ".\rc3_learning\.venv\Scripts\python.exe" -B -m rc3_learning.evaluate --checkpoint rc3_learning/run_verified_20261102_v2/best.pt --output rc3_learning/eval_new --scales 2x2 3x2 4x3 --seeds 4 --seed-start 1000000
```

评估不会更新权重。它同时运行四种原有无学习方法和重放教师，使用同一场景、F 和导航图；学习策略本身采用单次贪心解码，没有附加 2-opt 或修复来掩盖原始策略表现。

## 环境与依赖

本机环境：Python 3.10.20、PyTorch 2.12.1+cu126、RL4CO 0.7.0、TorchRL 0.14.1、TensorDict 0.14.3、NumPy 2.2.6、SciPy 1.15.2、Matplotlib 3.10.9；GPU 为 RTX 4070 Laptop 8 GB。实际版本也记录在每次运行的 `environment.json` 中。

项目 `.venv` 复用 `mobile_dual_arm_planning` Conda 环境的 PyTorch，新增依赖安装在项目内部。发现继承的 Conda 数值库在虚拟环境中出现 LAPACK DLL 加载问题后，已经在项目 `.venv` 内安装独立的 NumPy/SciPy/Matplotlib wheels，并重新完成训练和绘图；原 Conda 环境没有被修改。

迁移到其他机器时不要复制 `.venv`。先建立 Python 环境，依据 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 安装适合该机器的 CPU/CUDA 版本，再按 `requirements.txt` 和 `requirements-extra-lock.txt` 安装相应依赖。本机的完整核心版本固定在这些文件中；更换 PyTorch/TorchRL 版本后需要重新运行测试。`rc3_baseline` 必须保持为相邻目录。

## 实现与数据流

| 文件 | 功能 |
|---|---|
| `data.py` | 复用场景/F/导航图，重放教师，补齐不同规模批次，导出独立可验方案 |
| `env.py` | `ReachabilityCoverRoutingEnv`：重置、动作掩码、自动服务、成本及完成/失败状态 |
| `model.py` | 距离感知注意力编码器、逐候选评分器和教师/采样/贪心解码 |
| `train.py` | 行为克隆、REINFORCE、冻结 rollout baseline、验证集选择检查点 |
| `evaluate.py` | 检查点推理、公平对照、路径展开、服务清单和指标 |
| `plots.py` | 学习策略路径图 |
| `test_learning.py` | 状态转移、掩码、成本、梯度和重新加载测试 |

任务特征包含位置、桌中心位置及当前唯一的任务类型标记；服务节点包含位置、朝向正余弦、固定高度、左右臂覆盖比例及代理距离质量。真实任务类型和真实操作质量尚未接入。

任务特征通过覆盖关系聚合到候选点。候选间真实通道距离作为每个注意力头的可学习距离偏置；当前点到候选的通道距离、返回起点距离和剩余左右臂覆盖数进入动态评分。模型没有固定为 800 维的输出层。

任务补齐、候选补齐、图池化和动作输出均有掩码。层归一化仅作用于特征维度，测试验证了改变补齐区域数值不会改变真实候选的输出。终止样本在批量 softmax 中有数值占位，但不会产生服务动作或额外费用。

动作仅能选择有正新增覆盖、当前可达且能返回起点的真实候选。执行时自动完成其覆盖的全部剩余独立任务。若还有任务而所有动作被屏蔽，环境设置 `failed=true`、`completed=false`；训练拒绝将这种轨迹视作成功样本，也不会删除或重采样不可行任务。

每步奖励为 `-(移动距离 + 停靠费)/max(教师目标值, 1)`，最后一步另外加入返回起点距离，不再收取服务停靠费；空任务成本为 0。训练使用完整回合、gamma=1。环境成本与导出折线及独立校验器再次核对。

教师来自原有覆盖贪心加 2-opt，但标签必须按最终访问顺序重新计算覆盖和用臂分配，删除零新增覆盖停靠。本轮随机样本恰好没有可删除的停靠，该分支由定向测试覆盖；实现不会依赖这一巧合。

教师标签与参考成本只供监督损失和奖励归一化使用，不作为网络输入。推理可用 `make_instance(..., with_teacher=False)` 构建实例；测试已验证有/无教师信息产生相同动作和真实成本。评估脚本生成教师是为了报告对照，不是学习策略的必要推理步骤。

## 训练与检查点选择

BC 最小化教师动作的交叉熵，并按有效服务决策数归一化。RL 使用：

```text
loss = mean(stop_gradient(J_sample/J_ref - J_frozen_greedy/J_ref)
            * sum(log policy_probability_of_sampled_actions))
```

这是成本最小化方向的 REINFORCE。冻结策略来自最佳 BC 验证检查点；每轮在固定验证集上进行单侧配对检验，只有平均成本更低且 p<0.05 时才替换 rollout baseline。验证集至少需要两个样本才能替换。本次 3 轮 RL 没有触发 baseline 替换。

`best.pt` 按验证集平均实际目标成本选择，`bc.pt` 保存进入 RL 前的最佳验证权重，`last.pt` 保存最后训练权重和优化器状态。最佳检查点可以早于最后一轮，甚至在未产生改进时保留初始模型，不会人为保证 RL 胜出。本次选中的是 RL 第 2 轮。

训练、验证、测试种子互不重叠，均在 `dataset_manifest.json` 中列出。测试模型预测只发生在选定检查点之后，不用测试结果调参或选择检查点。场景和教师全部预生成并保存在内存及 JSON 中；大规模正式训练还需要评估内存并考虑分片或缓存，当前没有实现流式百桌训练。

## 本次验证结果

最终完整交付运行位于 **`run_verified_20261102_v2/`**：

- 64 个训练实例、16 个验证实例、12 个测试实例，随机种子不重叠。
- BC 120 次更新，RL 24 次更新，两阶段都实际改变了网络权重。
- BC 每决策交叉熵从第 1 轮的 2.2865 降至第 15 轮的 1.1949。
- 初始模型验证集平均目标成本为 25.5813，最终选定检查点为 20.9906。
- 12 个学习策略测试方案均完成全部任务；包括四种无学习对照和重放教师，共 72 份方案全部通过从磁盘重新读取的独立校验。
- 15 项训练专项测试通过，包含 CUDA 反向传播；独立加载检查点的评估命令也已实跑。

测试集平均目标值 `路程 + 0.75×停靠数`，单位为等效米；每个规模只有 4 个样本：

| 桌阵列 | 重放教师 | 学习策略贪心推理 | 解释 |
|---|---:|---:|---|
| 2×2 | 16.2625 | 15.4875 | 这四个样本上学习策略均值较低 |
| 3×2 | 20.7625 | 23.8250 | 学习策略均值更高 |
| 4×3 | 42.8250 | 61.4125 | 未参与训练的规模，性能明显下降 |

这证明训练、推理和校验链路可运行，**不证明优于启发式，更不证明跨规模性能已经解决**。当前逐实例 GPU 推理在这些小场景中也比简单启发式慢。完成率主要由合法动作掩码与可行实例保证，不能单独作为学到优质规划的证据。

原始结果、分项耗时、停靠数和完整路径均已保存。计时先预热网络并同步 CUDA；分别报告共享预处理、特征/输入传输、推理、输出转移/路径展开/校验。学习策略的计时不计生成教师的时间，因为其推理不需要教师；也不含模型加载、导入、绘图或写盘。`total_s` 为已列阶段之和，不是整个命令的墙钟时间。

## 结果文件与复验

| 文件/目录 | 内容 |
|---|---|
| `best.pt` / `bc.pt` / `last.pt` | 可重新加载的模型检查点 |
| `history.json` / `history.csv` | 每轮损失、验证成本、梯度范数、更新数及耗时 |
| `training_curves.png` | BC 损失与验证集成本曲线 |
| `dataset_manifest.json` | 三个数据划分、全部种子及教师信息 |
| `teachers/{train,validation,test}/` | 各实例完整场景及可独立检查的教师方案 |
| `evaluation/metrics.csv` / `summary.json` | 12 个独立测试场景上的逐例指标和汇总 |
| `evaluation/service_schedule.csv` | 全部方法的逐任务服务位置、顺序和用臂 |
| `evaluation/scene_plan_*.json` | 每个场景六种方法的完整路径与分配 |
| `evaluation/scales.png` | 各测试规模一个代表场景的学习策略路径 |
| `run_manifest.json` / `environment.json` | 配置、实际依赖版本、源码哈希和检查点选择 |

独立复验不需要 PyTorch：

```powershell
python rc3_baseline/validator.py rc3_learning/run_verified_20261102_v2/evaluation
```

专项测试：

```powershell
& ".\rc3_learning\.venv\Scripts\python.exe" -B -m unittest rc3_learning.test_learning -v
```

开发过程中的 `run_integration`、`run_smoke_20261102`、`run_verified_20261102` 等目录留作诊断记录。`run_smoke_20261102` 曾在绘图阶段中断，不能当作完整成功运行；最终报告以 `run_verified_20261102_v2` 为准。

机械臂仍是带肩部偏置的球壳/扇形代理，高度固定，任务按单臂独立顺序执行。尚未实现真实 IK、双臂同步协作、升降优化、关节轨迹、转向动力学、在线任务到达或容量约束。
