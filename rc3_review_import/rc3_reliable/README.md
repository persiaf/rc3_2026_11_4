# 可校验的多桌任务覆盖路径规划

研究判断、数学保证、学习模型诊断和后续实验设计见 [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md)。原始日志重算、独立检查及小规模精确参照放在 `audits/`；本适配层新增验证单独放在 `validation_results/`。

本交付在现有固定高度几何代理与通道图求解器外增加一层结果保留和认证逻辑。先生成并完整验证一个参考方案，再尝试其他构造策略；只有任务、用臂、导航折线、停靠顺序和成本全部通过校验，而且目标值确实降低时，才替换当前方案。任何后续建议错误或候选求解失败都会保留已经验证的参考方案。

默认快速模式使用已固定的配方 1。质量模式依次执行 `[1, 0, 7]`，返回其中通过认证且成本最低的方案。没有新训练，不需要模型检查点、GPU、PyTorch 或 scikit-learn。

**保证只针对声明的二维导航和固定高度操作空间代理。它不证明真实 IK 可解、机械臂避障、双臂同步动作、升降优化、实机成功、全局最优或硬实时。**

## 直接使用

在本目录创建环境并安装依赖：

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

快速模式，生成百桌堵塞场景示例及完整可验结果：

```bash
python -B reliable.py --scale 10x10 --scene-mode blocked --seed 3700000000 --mode fast --output my_results/scene_plan_fast.json
```

质量模式，依次尝试三个固定配方：

```bash
python -B reliable.py --scale 10x10 --scene-mode blocked --seed 3700000000 --mode quality --output my_results/scene_plan_quality.json
```

需要对原覆盖贪心加 2-opt 基线提供成本兜底时：

```bash
python -B reliable.py --scale 10x10 --scene-mode blocked --seed 3700000000 --mode quality --reference legacy --output my_results/scene_plan_legacy_guard.json
```

导入已有场景。输入可以是原始 scene JSON，也可以是含有 `scene` 字段的历史输出：

```bash
python -B reliable.py --scene my_scene.json --mode quality --output my_results/scene_plan_imported.json
```

从磁盘独立重验输出，不信任保存的 `valid` 标记：

```bash
python -B rc3_baseline/validator.py my_results
```

API 用法：

```python
from reliable import plan_reliably
from rc3_fast.scenes import make_complex_scene

scene = make_complex_scene(10, 10, 3700000000, 'blocked')
result = plan_reliably(scene, mode='quality', reference='fixed1')

if result['status'] == 'complete':
    plan = result['plans']['reliable']
    path_xy = plan['route_polyline']
    service_stops = plan['service_stops']
    certificate = result['certificate']
else:
    print(result['code'], result['precheck'], result['pending_task_ids'])
```

## 参考方案和成本保证

目标始终是 `J = 路程 + 0.75 × 服务停靠数`，单位为等效米。每项原始任务保留在完成率分母中。

| 设置 | 默认候选尝试顺序 | 默认尝试数 | 返回成本的确定性比较对象 |
|---|---|---:|---|
| `mode=fast, reference=fixed1` | 配方 1 | 1 | 本场景已认证的 Fixed1 |
| `mode=quality, reference=fixed1` | 配方 1、0、7 | 3 | 本场景已认证的 Fixed1 |
| `mode=fast, reference=legacy` | 原覆盖基线、配方 4、1 | 3 | 本场景已认证的原覆盖贪心加 2-opt |
| `mode=quality, reference=legacy` | 原覆盖基线、配方 4、1、0、7 | 5 | 本场景已认证的原覆盖贪心加 2-opt |

配方 0 为最近可行候选；配方 1、4、7 分别使用 `新增覆盖数**alpha / (距离 + 0.75)`，其中 `alpha=0.5、1、2`，随后使用共同的保覆盖局部优化。legacy 参考采用原基线流程及已验证等价的向量化 2-opt。

这里的 legacy 专指本次上传 Python 项目 `rc3_baseline` 的 `coverage_cost_2opt`，不表示它与论文中其他 MATLAB 场景、参数和算法的结果逐例等价。

证书会明确写入 `reference`、`reference_objective_m`、`returned_objective_m` 和 `cost_not_worse_than_reference`。**默认 Fixed1 兜底不能解释为不劣于 legacy。**参考对象、场景、候选集合、导航图和成本定义必须相同。

如果可选建议或候选失败后保留已验证方案，`certificate.fallback_used=true`，具体原因记录在 `trace`。没有已认证锚点时直接返回失败，不把失败称作成功回退。

记通过完整校验的初始参考为 `P0`。每次仅在新候选通过校验且 `J(Pnew) < J(Pbest) - 1e-9` 时替换，因此返回值满足 `J(Preturn) <= J(P0)`。这是相对已计算参考方案的成本保证，不是最优性或近似比保证。

## 候选数量预算与可选学习建议

`max_recipes` 包含参考方案及失败的候选尝试。比如质量模式设为 1，仅计算并返回参考方案：

```bash
python -B reliable.py --scale 10x10 --mode quality --max-recipes 1 --output my_results/scene_plan_budget1.json
```

可选建议只能追加到固定序列后面，且只有剩余预算足够执行追加候选时才读取建议：

```python
result = plan_reliably(scene, mode='quality', max_recipes=5,
                       suggested_recipes=[3, 10])
```

也可以传入接收 scene 副本的回调，由外部选择器返回配方 ID。无效 ID、异常回调、候选求解器异常或未通过认证的候选均写入 `trace`，不会覆盖已认证参考。这里没有打包或重新训练选择器；现有模型若接入，只能建议追加候选。

候选数量预算不等于时间截止保证。每个候选内部仍需完成路径求解、展开和校验。本适配层没有硬实时承诺，也没有中途打断一个未认证候选后将其交付。

## 预检和结构化失败

预检分别计算三项数量：原始任务数、至少有一个机械臂代理候选的任务数、至少有一个可从 depot 到达并返回的代理候选的任务数。

| 返回代码 | 含义 |
|---|---|
| `no_service_candidate` | 至少一项任务没有当前有限候选集合下的机械臂代理可行候选 |
| `disconnected` | 机械臂代理存在可行候选，但这些候选在当前通道图中不能从 depot 往返 |
| `mixed_task_precheck_failure` | 两类任务同时存在，分别列出 ID |
| `input_rejected` | 输入字段、ID、数值或固定高度约定不受支持 |
| `preprocessing_failed` | 例如 depot 落入膨胀障碍内，或导航模型构造失败 |
| `anchor_failed` | 初始参考未能完成求解和认证，没有可返回的已认证方案 |
| `empty_task_set` | 任务集合为空，返回显式的零成本有效闭合路线 |

失败保留 `scene`、全部 `task_ids` 和 `pending_task_ids`，不删除不可服务任务，不将失败改成完成。`no_service_candidate` 与 `disconnected` 仅说明当前离散候选、操作空间代理或通道图不足，**不是物理世界无解证明**；可通过补充候选、修正真实可达图或改进导航图再次求解。

若静态场景中每项任务至少有一个与 depot 双向连通的可服务候选，而且不存在容量、时窗、共享资源或双臂同时动作约束，则每次构造动作至少减少一个未覆盖任务，至多 N 次服务选择即可覆盖 N 项任务。这个条件性完备性来自候选和动作约束，不来自学习模型。

## 固定高度和双臂边界

运行依赖沿用演示参数：桌面高度与肩部高度为 0.75 m；候选 `h=0.40 m` 只是固定升降元数据。实际可达性使用独立的 `shoulder_height`，没有建立 `shoulder_height(h)` 标定关系。适配层主动拒绝候选高度与固定元数据不一致的输入，避免把改变 `h` 误当作已经进行了升降规划。

左右臂由肩部偏置、球壳距离和水平扇区分别判断，一个任务选一只可行臂顺序完成。没有真实关节模型、末端姿态约束、臂与环境碰撞、自碰撞或同时双臂调度。

底盘在障碍膨胀后的静态无向通道图中行驶；独立验证器重新检查整条折线的碰撞、导航腿终点、服务位姿、任务分配和成本。导航最短路仅针对该通道图。扰动和阻塞场景仍是结构化静态场景，不等价于任意动态环境。

输入任务、桌子、候选 ID 必须分别从 0 连续编号。机器人参数及任务坐标均需有限；保留原始 schema v2 服务清单。可达性真实数据的接入需要同步替换规划端和验证端的代理规则，不能只替换矩阵而继续用球壳验证器认证真实机械臂。

## 本次新增验证

本交付实际完成了 20 个新种子场景，每种规模 5 个：规则布局 1 个、几何扰动 2 个、通道阻塞 2 个，覆盖 4×3、6×6、8×8、10×10。

- 快速和质量模式共导出 40 份方案；从磁盘重新读取后，全部通过当前独立验证，全部任务完成。
- 20 个场景的质量模式返回值均不高于同场景 Fixed1 的目标值。
- 8 项针对性测试覆盖空任务、无服务候选、图断连、异常建议、求解器异常及非法候选、命名参考的成本单调性、单候选预算和不受支持的变升降输入。
- legacy 参考的成本兜底由针对性测试覆盖；20 场景批次比较使用默认 Fixed1 参考。

这些是本适配层的正确性抽样，不是新的学习性能结论或充分的统计性能比较。新 CPU 时间保存在 `validation_results/metrics_new_cpu.csv`；它来自本次 Linux 环境，每例运行一次，没有导航缓存，不能与历史 Windows 实验的时延直接合并。原始实验、训练权重和原性能指标均未修改。

运行针对性测试：

```bash
python -B -m unittest test_reliable -v
```

另存一轮相同确定性验证，不覆盖已有结果：

```bash
python -B validate_examples.py --output validation_repeat
```

## 文件

| 文件 | 用途 |
|---|---|
| `reliable.py` | 可靠性适配 API 和 CLI |
| `test_reliable.py` | 8 项针对性测试 |
| `validate_examples.py` | 20 场景、40 方案的保存与重验流程 |
| `validation_results/validation_report.json` | 新环境、抽样范围、独立校验及源码哈希 |
| `validation_results/metrics_new_cpu.csv` | 本适配层新 CPU 测量 |
| `validation_results/scene_plan_*.json` | 完整场景、方案、证书和候选尝试轨迹 |
| `source_provenance.json` | 上传来源哈希、交付副本哈希与唯一副本修改说明 |
| `rc3_baseline/`、`rc3_fast/` | 必要运行模块副本 |

为避免重复校验，交付副本的 `solver.export_plan` 仅增加 `validate_output` 开关，默认仍为 `True`，保留原接口行为。适配层设为 `False`，随后由 `_certify` 对完整输出统一校验一次。原上传源码未修改；该副本变化与前后哈希均记录在 provenance 中。

## 小规模精确参考

当前离散覆盖路由模型的精确动态规划仅用于少任务参考，不用于百桌全任务精确求解。脚本自动寻找本包运行模块，不需要训练模型或原始大压缩包。

```bash
python -B audits/exact/exact_cover_dp.py --check-only
python -B audits/exact/exact_cover_dp.py --cases 12 --output exact_repeat
```

`audits/exact/run_12/` 是本次已运行的历史参照，包含原始场景、四方法计划和成本差距。新运行请使用新输出目录。`audits/source_metrics/` 是原上传性能 CSV 的未改动副本，不能与 `metrics_new_cpu.csv` 混合作为一次计时实验。

论文可将此层表述为“以已认证参考方案为锚点的候选择优与成本兜底”。如果默认参考是 Fixed1，应写“不劣于同实例 Fixed1”；选择 legacy 参考时，才可写“不劣于同实例原覆盖贪心加 2-opt”。
