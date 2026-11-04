# 可达约束下的快速覆盖路由：学习选择器＋局部搜索

本目录保留原有 `rc3_baseline`、`rc3_learning`，增加用于复杂百桌场景的 CPU 规划器。
学习模型每个场景只调用一次，选择构造策略或三策略组合，然后由共享求解器生成服务顺序和避障折线路径。
这是**学习选择启发式的混合规划器**，不是直接逐点输出路线的端到端神经网络。

最终模型、留出测试数据和结论见 [实验结果报告](EXPERIMENT_REPORT.md)。模型与优化代码的收益分开评价：固定 top-1、固定 top-3、随机 top-3 使用完全相同的局部搜索。

本轮实际结论：百桌学习 top-1 比原 coverage+2-opt 平均降低目标成本 6.87%，规划中位数 39.17 ms，含首次建图 227.79 ms。**学习相对同预算固定策略的额外收益尚不确定**；追求稳定低开销时可选下面的 `--policy fixed`。这不是优于强优化算法的一般性结论。

## 已完成的工程改动

- 批量计算左右臂可达矩阵，与原标量实现对照。
- 向量化最优改进 2-opt，与原实现保持相同停止规则和并列次序。
- 在不丢失覆盖的条件下替换停靠点、删除冗余停靠；随后重新优化顺序。
- 所有输出按最终访问顺序重新分配任务和用臂，删除零新增覆盖停靠，返回起点成本计入目标。
- 导出 NumPy 权重；部署只依赖 NumPy、SciPy，不加载 PyTorch、RL4CO 或 scikit-learn。
- `Planner` 提供有限大小的导航缓存；任务改变时重新计算可达性，地图/障碍/停靠点/机器人参数改变时重新建图。
- 原导航与独立验证器同时支持额外矩形障碍，保留旧默认行为。

## 在当前工作目录运行

以下命令的工作目录是包含 `rc3_baseline`、`rc3_learning`、`rc3_fast` 的父目录。

```powershell
# 最终冻结的可部署模型；完整 JSON 和任务服务 CSV
python -B -m rc3_fast.plan --model rc3_fast/deployment/selector.npz --scale 10x10 --mode blocked --top-k 1 --output rc3_fast/my_result/scene_plan_100.json

# 三策略预算：通常提高质量，耗时增加
python -B -m rc3_fast.plan --model rc3_fast/deployment/selector.npz --scale 10x10 --mode blocked --top-k 3 --output rc3_fast/my_result/scene_plan_100_top3.json

# 验证集支持的固定组合：省掉模型推理和特征提取；使用相同优化器
python -B -m rc3_fast.plan --model rc3_fast/deployment/selector.npz --policy fixed --top-k 1 --scale 10x10 --mode blocked --output rc3_fast/my_result/scene_plan_fixed.json

# 保留的神经选择器也可直接部署（本轮验证结果弱于树模型）
python -B -m rc3_fast.plan --model rc3_fast/deployment/selector_neural.npz --top-k 3 --scale 10x10 --mode blocked --output rc3_fast/my_result/scene_plan_neural.json

# 输入自己修改的 scene JSON（也支持包含 scene 的现有输出文件）
python -B -m rc3_fast.plan --model rc3_fast/deployment/selector.npz --scene rc3_fast/my_result/scene_plan_100.json --top-k 1 --output rc3_fast/my_result/scene_plan_reloaded.json

# 重新读取落盘文件，独立检查，不信任保存的 valid 标记
python -B rc3_baseline/validator.py rc3_fast/my_result
```

不传 `--model` 时运行默认策略＋局部搜索，作为无需模型的回退方案。输出路径允许覆盖同名文件，实验数据/训练/基准目录则要求全新或为空。

Python 集成与同地图多批次任务：

```python
from rc3_fast.planner import Planner
from rc3_fast.scenes import make_complex_scene

planner = Planner("rc3_fast/deployment/selector.npz", top_k=1, cache_size=2)
scene = make_complex_scene(10, 10, 20261102, "blocked")
result = planner.plan(scene)
plan = next(iter(result["plans"].values()))
path_xy = plan["route_polyline"]
service_sequence = plan["service_stops"]

# 修改 scene['tasks'] 后，位置不变的地图可继续复用导航缓存。
# 任务 id 必须从 0 连续编号，table_id 对应 tables 的连续编号。
next_result = planner.plan(scene)
assert next_result["timing"]["cache_hit"]
```

## 模型与训练

输入为 38 维场景统计：任务密度、左右臂覆盖、每项任务的服务选择数、候选覆盖、图上距离和绕行比例、桌子与障碍、任务空间分布等。不会读取 seed、教师标签或测试集成本。空间信息被统计量压缩，这也是学习能力受限的可能原因。

构造配置共 13 种：最近可服务点，以及 `新增覆盖数**alpha/(图上距离+beta)`，其中 alpha∈{0.5,1,2,4}，beta∈{0.75,2,6}。模型选择后，每种配置使用相同 2-opt、两轮覆盖保持替换/删除。目标为 `J=路程+0.75×服务停靠数`，单位为等效米。

实际比较两种 MLP 训练目标：预测 13 个配置成本；预测 13 个单配置与 286 个三配置组合的 best-of-k 成本。MLP 为 38→64→64→输出，AdamW，批量 64，学习率 1e-3 余弦下降，三个种子各 500 轮，按验证集选检查点，不默认最后一轮最好。

另比较随机森林的组合成本回归：64 棵树，最大深度 10，最小叶样本数 {8,24,64}，三个随机种子。固定组合来自训练集穷举 13 选 3，最终模型家族与检查点只依据验证集决定。

训练数据按场景划分，13 个策略的成本矩阵预先缓存；训练不重复求导航和局部搜索。神经模型将训练张量常驻 GPU。部署权重转为 NumPy，并校验与原模型预测一致。

```powershell
# 重建扩大版数据；已有目录不会被覆盖
python -B -m rc3_fast.dataset --output rc3_fast/data_repeat --train-per-group 96 --val-per-group 16 --test-per-group 16 --seed 20261103

# 需要本机已验证的 PyTorch 虚拟环境
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_fast.train_selector --data rc3_fast/data_repeat --output rc3_fast/train_repeat --epochs 500 --portfolio-targets

# 可选树模型，使用已安装 scikit-learn 的 Python
python -B -m rc3_fast.train_forest --data rc3_fast/data_repeat --output rc3_fast/forest_repeat

# 独立留出测试，包含全部路径输出和校验
python -B -m rc3_fast.benchmark --data rc3_fast/data_extended_20261102 --model rc3_fast/deployment/selector.npz --output rc3_fast/benchmark_repeat --repeats 3

# 144/196 桌外推，不参与训练和选模型
python -B -m rc3_fast.benchmark --data rc3_fast/data_extended_20261102 --cases rc3_fast/ood_cases_20261102.json --model rc3_fast/deployment/selector.npz --output rc3_fast/ood_repeat --repeats 3

python -B -m unittest rc3_fast.test_fast -v
python -B -m unittest discover -s rc3_baseline -p test_baseline.py
```

## 场景、计时与适用范围

训练/验证/同分布测试均包含 12、36、64、100 张桌及 regular/mixed/blocked 三种生成分布。每桌 0～3 个静态任务。mixed 加入稀疏/均衡/密集任务分布、桌子尺寸与位置变化、通道宽度变化；blocked 额外阻断部分横向通道，保留纵向通道。场景不可行时直接报错，不重新抽任务来隐藏失败。

冷地图时间包含可达性、建图和最短路、模型/搜索、路径展开和独立校验；复用地图时间仍包含当前任务可达性和地图键检查。基准所有方法共用同一可达性和导航实现，计时排除 Python 启动、模型加载与 JSON 磁盘写入。单独的网络前向时间不能代表完整规划时间。

机械臂仍是肩部偏置球壳与扇形代理，高度固定。底盘用圆形外接尺寸膨胀为矩形障碍检查，允许原地转向，不模拟动力学。额外障碍参与底盘路径检查，**没有纳入机械臂轨迹碰撞**。模型只支持独立单臂任务；不声称同步双臂协作、真实 IK 可行、连续空间全局最短路或在线到达任务已解决。要接真实机器人，应替换可达性/服务方案接口，并补实际控制与碰撞验证。

## 研究参考

思路参考 [Learning Heuristic Selection](https://arxiv.org/abs/2006.08246) 与 [Learning to Delegate](https://arxiv.org/abs/2107.04139)，均不是本桌面问题的性能证据。树模型使用本机 scikit-learn 1.6.1 的 [RandomForestRegressor](https://scikit-learn.org/1.6/modules/generated/sklearn.ensemble.RandomForestRegressor.html)。本轮不主张新的基础网络架构。
