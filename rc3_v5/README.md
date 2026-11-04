# v5：第四版验收后的维护与真实机器人输入准备

本版保留 **Fixed3 快速默认、Detour 规则质量模式、冻结旧 MLP 可选质量模式、Structured 研究模式**。没有新增训练或改变 v4 的选模、测试集与采用门槛。旧版本文件和权重保持原样，全部改动在 `rc3_v5`。

已完成共同可达约束修复、图嵌入缓存、本机旧 MLP 同预算补评、固定版本 AlohaMini1 几何资产与配置预检。真实 IK/碰撞闭环仍需有效标定配置，不能把当前代理可达性实验当作真实机械臂执行认证。数字和边界见 [RESULTS.md](RESULTS.md)，三部分论文证据见 [EVIDENCE_MAP.md](EVIDENCE_MAP.md)。

## 直接生成完整路径

在交付包根目录运行，推理只需 NumPy、SciPy、threadpoolctl；不需 Torch/GPU。运行依赖在 `rc3_v5/requirements-runtime.txt`。本轮 CPU 推理与集成检查使用 Python 3.13.5；原版含 Torch 的回归在已有 Python 3.10 环境通过。

```powershell
python -m pip install -r rc3_v5/requirements-runtime.txt
python -m rc3_v5.plan --method fixed3 --scale 10x10 --mode blocked --seed 771903 --output outputs/fixed3.json
python -m rc3_v5.plan --method detour --budget 8 --scale 10x10 --mode blocked --seed 771903 --output outputs/detour.json
python -m rc3_v5.plan --method legacy --budget 8 --scale 10x10 --mode blocked --seed 771903 --output outputs/mlp.json
python -m rc3_v5.plan --method learned --budget 8 --scale 10x10 --mode blocked --seed 771903 --output outputs/structured_cached.json
```

`legacy` 自动加载 `rc3_v3/deployment/policy.npz`（59,744 bytes）；`learned` 加载同一冻结的 `rc3_v4/training/structured.npz`，通过新缓存适配器运行。输出 JSON 包含场景、停靠顺序、任务/用臂分配、完整折线路径、成本与诊断，同时输出服务表 CSV。

包内 `examples/mlp_100tables.json` 及同名 CSV 是已运行的百桌输出示例，用于查看数据格式；其单次耗时不作为统计性能结论。

`--scene scene.json` 可代替随机场景。当前代理入口为固定升降高度的静态批次问题。外部 F/A 证书必须使用下列外部入口，避免被代理可达性覆盖。

持续使用时保留 `Planner` 对象，以复用未改变地图的导航预处理：

```python
from rc3_v5.plan import Planner
planner = Planner(method="legacy", horizon=8)
result = planner.plan(scene)
assert result["status"] == "complete"
```

若指定 `--budget 256 --time-limit-ms 400`，含义是 **400 ms 软预算**。已开始的修复和最终验证可能超时；本轮两方法都 54/54 超时。完整端到端使用还需加上模型加载、场景/可达性/导航预处理及 I/O。默认固定 8 次模式不承诺硬截止时间。

## 外部服务约束与机器人输入

```powershell
python -m rc3_v5.external_example --output outputs/external_demo
python -m rc3_v5.external --help
python -m rc3_v5.robot_input inventory --manifest rc3_v5/robot_assets/manifest.json --output outputs/robot_inventory.json
python -m rc3_v5.robot_input check --manifest rc3_v5/robot_assets/manifest.json --config rc3_v5/robot_assets/runtime_config.template.json --output outputs/robot_preflight.json
```

演示 F/A 是合成表。外部构造、修复、路线规范化及分配均使用 `F & A`；最终分别检查两表并执行提供的 `service_checker`。这叫双表一致性检查，A 已参与决策，不能再称为未见标签盲测。[EXTERNAL_SCHEMA.md](EXTERNAL_SCHEMA.md) 给出文件格式与回调契约。

真实资产已完整取得：两份固定提交 URDF、19 个引用 STL、版本与 SHA 清单。`inventory` 应成功；未填配置模板的 `check` 应返回 `blocked_inputs`、退出码 2。先复制模板再填写有效限位、零位、TCP、升降、夹爪、停放与碰撞配置并附来源；详见 [ROBOT_INPUT_STATUS.md](ROBOT_INPUT_STATUS.md)。输入通过后仍须执行下游 IK、碰撞与服务见证生成，本版没有虚构这一步的结果。

## 复现与核对

```powershell
python -B -m rc3_v5.verify
python -B -m rc3_v5.external_results.reproduce
python -B -m rc3_v5.audit_supplement --input rc3_v5/mlp_budget_windows --output outputs/native_cost_audit.json
python -B -m rc3_v5.cache_diagnostic --help
python -B -m rc3_v5.budget_supplement --help
```

已有计时结果仅运行一次完整预声明协议，未挑最快或最好的一轮。重跑性能时另选空输出目录，并串行运行；不要覆盖原记录。核对上传补评时，将 `--input` 改为 `rc3_v4_review_import/rc3_v4_review_evidence/mlp_budget_supplement`。几何/成本和协议/统计审核分开保存在 `*_supplement_audit.json`、`*_protocol_audit.json`。

缓存仅用于一个静态 Env，路线/窗口布局不变时复用图嵌入，动态头和掩码仍重新计算。权重不可变；F、D 或任务/候选语义变更必须创建新 Env。`invalidate()` 只清本适配器缓存，不能更新环境自身的旧缓存。完整限制在 `cache_results/CACHE_CONTRACT_AND_LIMITATIONS.txt`。
