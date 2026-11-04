# v3：Fixed3 初解上的快速覆盖修复学习

本目录是新增实现，保留原 `rc3_improve` v2 及其数据。当前版本用真实中间状态的局部收益监督训练模型，模型负责选择值得修复的窗口或停止；覆盖DP负责修复，独立验证器检查最终任务、左右臂分配、底盘折线路径与成本。

当前目标为 `J = 底盘路程 + 0.75 × 服务停靠次数`，单位为等效米；报告中的百分比下降指这个联合目标，不一定等于纯路程下降。

**默认入口使用 Fixed3。** 学习模式在已验证 Fixed3 基础上尝试降低成本，增加的运行时间与实际收益见 `RESULTS.md`。不预设注意力优于MLP，也不将此版监督训练称为PPO。

## 直接运行

在包含 `rc3_v3`、`rc3_fast`、`rc3_improve` 和 `rc3_baseline` 的项目根目录执行。便携NPZ模型推理不需要Torch：

```powershell
python -m rc3_v3.plan --method fixed3 --scale 10x10 --mode blocked --seed 771901 --output demo/fixed3.json
python -m rc3_v3.plan --method learned --budget 8 --scale 10x10 --mode blocked --seed 771901 --output demo/learned.json
python -m rc3_v3.plan --method learned --scene your_scene.json --output demo/custom.json
```

输出JSON包含任务、服务位置、左右臂分配、访问序列、完整避障折线路径、独立验证及计时；同名CSV给出任务服务清单。失败时返回明确原因和全部未完成任务，不通过删任务或重采样伪造成功。

```python
from rc3_v3.plan import Planner
from rc3_fast.scenes import make_complex_scene
planner = Planner(method='learned', horizon=8)
result = planner.plan(make_complex_scene(10, 10, 771901, 'blocked'))
assert result['status'] == 'complete'
```

同一个Planner缓存静态导航图，任务变化仍重新计算可达关系。`--time-limit-ms 200 --budget 256`可设软规划时间上限：预算包含初解、模型、修复及最终验证，排除导航预处理、模型加载和文件写入。原子修复和最终验证可能超时；这不是硬实时控制。256调用上限超出训练8步范围，仅作为时间实验的保护上限。

## 重新生成、训练与验证

本机Torch环境为 `rc3_learning/.venv/Scripts/python.exe`，训练需GPU或显式`--device cpu`。以下训练及完整复验命令应在Torch环境执行（完整复验含Torch/NPZ对照测试）。便携推理依赖`numpy scipy threadpoolctl`，图表额外使用`matplotlib`；依赖版本见requirements文件及真实运行记录。

```powershell
python -m rc3_v3.data --output rc3_v3/new_data --seed 6241237
python -m rc3_v3.train --data rc3_v3/new_data --output rc3_v3/new_run --epochs 80 --device cuda
python -m rc3_v3.evaluate --calibrate --data rc3_v3/new_data --output rc3_v3/new_wall_budget.json
python -m rc3_v3.evaluate --data rc3_v3/new_data --model rc3_v3/new_run/best.npz --wall-config rc3_v3/new_wall_budget.json --output rc3_v3/new_results
# 以下两步在用于重训的独立项目副本中执行：部署新模型后检查默认入口一致性。
Copy-Item -LiteralPath rc3_v3/new_run/best.npz -Destination rc3_v3/deployment/policy.npz
python -m rc3_v3.verify --results rc3_v3/new_results --model rc3_v3/deployment/policy.npz --output rc3_v3/new_verification
python -m unittest discover -s rc3_v3 -p "test_*.py" -v
```

`verify`是交付检查：除检查显式传入模型与评估哈希一致，还要求默认入口的`deployment/policy.npz`与该模型一致。因此，重训和部署建议在完整项目的另一个副本中进行，保留本次冻结交付。若在当前目录替换部署模型，应先备份原`deployment`目录及其清单；复制新模型会使原冻结部署清单不再代表当前文件，需保留对应训练、评估与新复验记录，不能继续引用原清单声称复现。

历史种子排除会随本地既有产物改变；若要逐例复现实验，使用随包冻结的`data_v3_20261102`，而不是重新抽取同seed的清单。`data.pkl`是本项目本机生成的训练缓存；不要用pickle打开不可信来源文件。新运行目录须为空，不覆盖已冻结结果。

## 方法与范围

- Fixed3三配方 `(0,1,7)` 按便宜组合成本排序，只对可输出候选进行完整独立认证；搜索期间保留已验证初解。
- 窗口步长1，宽度最大4，根据必要任务数缩小，DP上限8任务/80候选。三个选择方法共享同一修复器及邻域。
- 失败缓存包含局部节点、必要任务和候选池，仅缓存确定性无改善；外部覆盖改变后重新尝试。
- 28维观察包含24维路线/覆盖/导航信息，以及剩余预算比例、执行比例、当前/初始成本比与累计改善。预算是相对比例，不能据此声称模型识别绝对调用数或观察为充分Markov状态。
- 监督数据覆盖0至7步，混合oracle、规则、随机轨迹。收益回归、正收益分类与窗口排序联合训练；全零收益状态还训练独立全局停止二分类头，固定保守阈值0.9。它不是校准正确率承诺。没有当前学习策略访问状态后的重标注轮次，不称为DAgger。
- Attention与MLP共享数据和轮数，但参数量和计算量不同；对照是实用模型选择，不是严格参数匹配的架构消融。
- 96训练和18验证全为100桌；54主测试覆盖12至196桌，另6压力测试为256/400桌。同一冻结模型、相同物理参数跨规模测试。

实际机械臂仍是带肩偏置的球壳/扇形可达代理，升降固定。没有真实机械臂关节轨迹、协作抓取、容量约束或持续任务到达。输出可靠性只针对当前离散代理与静态导航图，不等价于真实机器人安全认证。

开源项目与代码核查见 `OPEN_SOURCE_NOTES.md`；原始训练/测试/独立复验和结论边界见 `RESULTS.md` 及对应JSON/CSV。

首轮 `run_v3_20261102` 发现STOP同窗口共用softmax会提前停止，已保留诊断和修改前源码快照。正式版为 `run_v3_stopfix_20261102`，使用独立二分类停止判断重新训练。全部修订在冻结测试之前完成；不得把首轮和正式轮的指标混写。
