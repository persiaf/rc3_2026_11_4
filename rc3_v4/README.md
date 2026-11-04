# 第四版：覆盖规划、质量参照与结构输入研究

默认规划器仍为 **Fixed3**，需要降低路线成本时选择 **Detour**。二者不需要训练或 Torch。第四版保留第三版全部冻结结果，新增外部服务约束接口、小规模全局精确参照、同网络结构输入消融，以及严格审计和推理优化。

成本为 `J = 底盘折线路程 + 0.75 × 服务配置数`，单位是等效米。当前普通场景仍使用演示代理可达性；输出的“通过验证”指给定候选、导航图和服务约束模型内的完整方案。

## 运行稳定规划器

在包含 `rc3_v4`、`rc3_v3`、`rc3_improve`、`rc3_fast`、`rc3_baseline` 的项目根目录执行：

```powershell
python -m pip install -r rc3_v4/requirements-runtime.txt
python -B -m rc3_v4.plan --method fixed3 --scale 10x10 --mode blocked --seed 771902 --output demo/fixed3.json
python -B -m rc3_v4.plan --method detour --budget 8 --scale 10x10 --mode blocked --seed 771902 --output demo/detour.json
```

JSON 包含任务分配、左右臂、服务配置、访问顺序、完整底盘折线、独立校验与耗时；同名 CSV 是任务服务清单。`--scene` 可读取原场景 JSON 或含 `scene` 的已有输出。

```python
from rc3_v4.plan import Planner
planner = Planner(method="detour", horizon=8)
result = planner.plan(scene)
assert result["status"] == "complete"
```

规划器可缓存同一地图的导航预处理。`--time-limit-ms 400` 是包括初解的软预算，正在运行的修复和强制终验可能超时。不能用这个参数承诺硬实时。失败结果保留全部待服务任务，不宣布物理不可行。

## 外部可达关系与离散高度

```powershell
python -B -m rc3_v4.external_example --output demo/external
python -B -m rc3_v4.external demo/external/external_demo.json --method detour --budget 8 --output demo/external_plan.json
python -B -m rc3_v4.external demo/external/external_demo.json --verify-plan demo/external_plan.json --output demo/external_check.json
```

外部模式支持不同服务高度，分别使用规划 F 与审计 F，并绑定任务、候选、模型资产版本和见证记录。可接入调用者提供的独立 IK/碰撞检查回调。若停靠罚项不是默认0.75，规划与独立复验均须提供相同 `--penalty`。

详细 JSON/NPZ/MATLAB 交换格式见 [EXTERNAL_SCHEMA.md](EXTERNAL_SCHEMA.md)。示例是人工指定的 `synthetic_demo`，只证明接口、数据校验和失败处理可运行；不证明真实升降、双臂或 IK 性能。现阶段不验证机械臂运动全程、升降过渡和双臂同步协作。

## 学习研究及复现

结构网络接收局部必需任务、候选、两臂 F、入口/出口距离、候选间导航 D，以及原28维统计。control 与 structured 同为38,438参数，两层32维距离偏置注意力；control屏蔽结构数值，训练预算/标签/种子完全相同。此结构仍没有完整编码候选对窗口外任务的覆盖，不是普遍充分状态。

本轮是**真实修复收益监督学习**，没有新增 PPO 或其他强化学习训练。参考 Learning to Delegate 等项目选择子问题的思路，具体来源与未复现部分见 [OPEN_SOURCE_NOTES.md](OPEN_SOURCE_NOTES.md)。

```powershell
# 仅训练和完整单元测试需要Torch；本机使用既有CUDA环境。
python -m pip install -r rc3_v4/requirements-train.txt
python -B -m rc3_v4.train --data rc3_v4/data --output new_training --epochs 80 --batch-size 8
python -B -m rc3_v4.parity --training new_training
python -B -m rc3_v4.evaluate --training new_training --data rc3_v4/data --output new_results
```

上述命令在同一冻结测试集上是**复现**，不能称为新的独立测试。进一步研究须另冻结未见种子；不要利用本轮39场景调参。训练使用96场景、384状态和原18验证场景；两种输入、三个种子、80轮，共23,040次更新。每10轮按验证集实际路线成本选模，最终轮不一定最好。

已训练研究模型可用纯NumPy执行：

```powershell
python -B -m rc3_v4.plan --method learned --checkpoint rc3_v4/training/structured.npz --scale 10x10 --mode blocked --seed 771902 --output demo/research.json
```

此命令只是显式选择研究模型，默认入口不会自动切换。采用判据与新测试结论见 [results/RESULTS.md](results/RESULTS.md)。原第三版MLP的比较使用相同冻结权重及新的线程控制适配器，无重新训练；微测不能直接解释成完整规划同比提速。

约60KB的旧MLP可以通过 `python -B -m rc3_v4.plan --method legacy --scale 10x10 --mode blocked --seed 771902 --output demo/legacy.json` 使用优化后的推理入口。它与第四版结构模型分开命名，均为可选研究策略。

## 精确参照与审计

- [small_results/READOUT.md](small_results/READOUT.md)：27场景、45个场景—罚项组合、225方案；44组合得到离散全局最优，1组合任务数超限，未记为最优。
- `rc3_v4.exact.solve(problem)`：非负有向最短路度量和非负停靠罚项，任务bitmask全局动态规划；默认最多14任务、128候选，超时返回界与可行解。
- `python -B -m unittest discover -s rc3_v4 -t . -p "test_*.py" -q`：全模块回归；需要Torch。
- `python -B -m rc3_v4.report --results rc3_v4/results`：从所有原始重复方案重建结果并复验，哈希错误或方案失败退出非零。
- [maintenance.md](maintenance.md)：旧哈希汇总漏洞、STOP脚本适用范围和审查附件记录。

正式分工保持 `prepare → plan → verify → evaluate`：数据绑定与导航准备、规划、独立约束检查、最后才统计性能。初解认证失败时报告失败；改善方案未通过终验则返回已认证初解。这个保证不依赖网络预测准确，也不意味着全局最优或实机认证。

结果、权重、源代码、种子、完整方案和运行清单均保留。下载包不包含虚拟环境；需要旧版全部原始结果时继续使用保留的 `rc3_v3_delivery.zip`。
