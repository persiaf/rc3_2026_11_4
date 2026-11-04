# 外部服务约束接口 v1：v5 实现

已实现 `prepare_external → plan_external → validate_external`。高层仍在二维导航图上计算；每个服务候选保留 `(x,y,theta,h)`，允许不同离散升降高度。旧版代理规划和验证代码保持原样。

**这里交付的是可运行的数据接口、双表一致性检查和验证适配器，并未因此完成真实 IK 或双臂物理实验。** 随包示例明确标为 `synthetic_demo`，不能用它报告真实升降收益、真实双臂优势或代理误判率。

## 数据与单位

顶层包含 `schema = "rc3-external-reachability-v1"`、`scene`、`planning`、`audit`。

| 对象 | 约定 |
|---|---|
| `scene` | 原导航场景格式；距离和高度为米，角度为弧度；包括任务、桌、障碍、起点、候选和通道线 |
| 任务、候选、桌 ID | 各自从 0 开始的连续整数；数组顺序必须与 ID 一致 |
| `task.service_mode` | 必须显式为 `independent_single_arm`；双臂同步、搬运容量等任务不在该格式内 |
| `candidate.h` | 服务时的离散升降高度；无需等于旧字段 `robot.fixed_lift_height` |
| `F` | `N × M × 2`，轴顺序为任务、候选、手臂；最后一轴固定 `[L,R]`；值为布尔或整数 0/1 |
| `planning.F` | 原始规划可达关系 F；v5 与 A 求交后供构造、修复、规范化和分配使用 |
| `audit.F` | 外部复核关系 A；v5 已把它用于规划，终验仍分别检查 F、A；不能用预测置信度代替 |
| `witnesses` | 可选原始见证记录，例如 task_id、candidate_id、arm、q、检查日志；完整内容纳入证书哈希 |

导航继续使用已有通道图。候选需要 `corridor_anchor`，场景需要有序 `x_lines`、`y_lines`；这不是任意地图的自动建图器。基座采用 `base_radius + clearance` 的圆包络检查矩形障碍膨胀，输入方必须声明该包络对所有候选高度都保守。接口不推导升降时的底盘/机械臂包络。

目标保持 `J = L + lambda K`，默认 `lambda = 0.75 m/stop`。同一平面位置、不同高度视为不同服务配置，可以形成零底盘距离的连续服务段；每个配置仍计一次停靠罚项。这里没有升降时间、升降路径碰撞或转向时间，不能将 J 当作实际执行总时间。

## v5 的共同约束与可靠出口

本版有效关系是 `B[i,j,a] = F[i,j,a] & A[i,j,a]`。`prepare_external` 返回的 `Problem.F` 就是 B；原始 F 和 A 留在输入证书中。Fixed3 选点、Detour 修复、删除无新增覆盖停靠以及到站分配都使用 B。每到一站，只服务尚未完成且存在共同允许手臂的任务；两臂均允许时固定优先 L，只有 R 共同允许则用 R，无共同手臂则留给后续停靠。整条路由仍无法覆盖时明确拒绝。

这比上传的“只改出口用臂”补丁多改了一处必要语义：若先依据 F 删除停靠，再依据 F∩A 分配，可能把唯一受 A 允许的后站提前删掉。`external_results/canonicalization_counterexample.json` 给出 4 任务合成反例：F 称候选 0 能完成所有任务，但 A 只允许任务 0 在候选 23 完成。原路由 `[depot,0,23,depot]` 在共同约束下合法；F-only 规范化却删掉 23。v5 保留两站，底盘路程 11.3 m，2 次停靠，`J=12.8`（默认罚项 0.75）。上传的仅分配补丁仍拒绝该例，v5 的出口与规划均通过。

`export_external` 也兼容旧调用方传入的 F-only `Problem`：先验证其场景与 bundle 一致、F 不启用证书禁止的服务，再用浅副本求共同关系，不修改原 Problem。调用方提供的更严格掩码会保留，不会被 bundle 自动放宽。此步骤不改变导航和请求罚项。

**A 已参与选点、修复和分配，因此这是双表约束一致性验证，不是把 A 当盲测标签的泛化实验。** 不同 producer_id 也不能证明生成器独立。需要物理独立证据时，应使用独立实现和真实标定资产配置 `service_checker`，并保留原始复算日志；当前接口不内置物理检查器。

保底初解须先通过出口验证，再通过抽象路线与实际输出成本一致性检查，之后才能保存为 anchor。Detour 提案还须通过同样成本一致性及成本不增检查。缺失/无效初解一律报告规划或认证失败，绝不报告物理不可行。回调拒绝已选 L 不会被暗中当作改用 R 成功；最终提案回调拒绝或异常时保留已认证初解。成本不增只相对于同一 F∩A 约束下已认证的初解，不承诺优于 F-only 路线或全局最优。

数据 schema 仍是 v1，Python 函数签名与 v4 兼容；`prepare_external().F` 的语义有上述收紧。F=A 的合成 demo 最终计划（含验证字段）与 v4 字典完全相等；诊断新增 `service_relation`、`audit_table_used_for_planning` 和共同真值项数。冻结 v4 源码、数据、历史结果均未修改。

## 版本与证据来源

每份证书包含 `role`、任务/候选 ID 顺序、轴顺序、shape、`scene_sha256`、`F_sha256`、`certificate_sha256` 和 `source`。整个场景（包括任务姿态附加信息）、F、来源及见证都被绑定；改任务、候选、模型来源或 F 后须重新生成证书。两份证书必须绑定同一场景和同一模型/约束资产版本。

`source` 的必填字段如下：

```json
{
  "producer_id": "your-independent-checker-id",
  "generator": "your-IK-and-collision-checker",
  "generator_version": "your-version-or-commit",
  "robot_model_id": "your-robot-revision",
  "generated_utc": "2026-11-03T00:00:00Z",
  "evidence_level": "externally_attested",
  "asset_sha256": {
    "kinematics": "64 lowercase hexadecimal digits",
    "joint_limits": "64 lowercase hexadecimal digits",
    "collision_geometry": "64 lowercase hexadecimal digits",
    "task_pose_spec": "64 lowercase hexadecimal digits",
    "lift_spec": "64 lowercase hexadecimal digits"
  },
  "constraint_checks": {
    "joint_limits": "checked",
    "end_effector_pose": "checked",
    "self_collision": "checked",
    "environment_collision": "checked",
    "lift_configuration": "checked"
  },
  "navigation_footprint_scope": "conservative_circular_bound_all_candidate_heights"
}
```

上面的 SHA 字符串只是格式说明，不能直接运行。`externally_attested` 要求全部检查状态为 `checked` 且提供列出的资产哈希。它的含义仅为“外部生产者声明做过检查”；程序不会因为这几个字符串就确认物理真实性。`synthetic_demo` 可记录 `synthetic` / `not_checked`，其证据级别会随验证结果输出。

规划和审计必须使用不同 `producer_id`。这个检查只能避免误将同一个标识作为两份数据，**不能证明生成代码、模型误差或实验人员相互独立**。哈希证明内容绑定和变化检测，不证明来源真实性，也不自动核验磁盘上的机器人资产。真实接入时应另外记录所用检查器、资产文件、软件版本和复核日志。

## Python / MATLAB 交换

JSON 是 MATLAB 的交换格式；NPZ 是 Python 的紧凑格式，使用 `allow_pickle=False`。NPZ 内只有 `metadata_json`、`planning_F`、`audit_F`，无对象数组。

为避免两种语言的 JSON 浮点序列化影响哈希，MATLAB 只输出未封装草稿：

```text
draft.scene
draft.planning.F
draft.planning.source
draft.planning.witnesses  (optional)
draft.audit.F
draft.audit.source
draft.audit.witnesses    (optional)
```

`F` 在 JSON 中应明确嵌套为 `[task][candidate][arm]`。若 MATLAB 的多维数组 JSON 行为不符合预期，用 cell 明确轴顺序：

```matlab
F_cells = cell(N, 1);
for i = 1:N
    row = cell(M, 1);
    for j = 1:M
        row{j} = double(reshape(F(i,j,:), 1, 2));
    end
    F_cells{i} = row;
end
draft.planning.F = F_cells;
% audit.F 必须由审计检查结果独立生成，并采用相同轴顺序。
% 用 jsonencode(draft) 写 UTF-8 JSON；所有 ID 使用 0 起始。
```

Python 封装命令不执行物理验证：

```powershell
python -B -m rc3_v5.external matlab_draft.json --seal-draft --output external_bundle.json
python -B -m rc3_v5.external external_bundle.json --method fixed3 --output external_plan.json
python -B -m rc3_v5.external external_bundle.json --method detour --budget 8 --output external_repaired.json
python -B -m rc3_v5.external external_bundle.json --verify-plan external_repaired.json --output external_check.json
```

示例命令从交付包的项目根目录执行。若需 NPZ，封装时将输出扩展名设为 `.npz`。独立验证命令读取原始输入，重新计算证书及方案检查，忽略保存的 `validation` 布尔结果。

## 可靠性范围

新验证器检查完整任务分配、合法用臂、审计 F、场景/证书绑定、服务姿态（含 h）、停靠顺序、各段终点、闭合折线、膨胀障碍碰撞、路程和 `L+lambda K`。它没有调用旧球壳扇形可达性函数。

Fixed3 三个候选路线按成本排序后逐个认证，至少一个认证成功才建立保底初解。Detour 的最终方案须通过同一出口且成本不增；修复异常、审计拒绝或成本异常都会回退已认证初解。全部初解都被拒绝时返回 `status=failed`、未完成任务 ID 和 `physical_infeasibility_claim=false`，不把“输入不足/审计不一致”说成物理不可行。规划和审计的全部 F 差异数也会输出。

可选择接入真实复算回调：

```python
def my_service_checker(task, candidate, arm, audit_certificate):
    # 由你自己的机器人模块加载版本对应的资产、见证及姿态要求；
    # 在此重新求解/验证 IK、关节限位和碰撞，返回 Python bool。
    return actual_check_result

result = plan_external(bundle, method='detour', service_checker=my_service_checker)
```

回调拒绝或抛出异常均导致相应方案认证失败。回调正确性、独立性和真实检查范围由外部实现负责；本接口仍输出 `built_in_physical_ik_recheck=false`，并记录调用次数。可达表/单点 IK 都不能单独证明机械臂从前一状态到服务姿态的整段无碰轨迹。

## 已完成的验证

`python -B -m unittest rc3_v5.test_external -v`：28 项测试通过，即 9 项 v4 原测试、11 项上传补丁的新增回归、8 项 v5 共同覆盖与成本一致性回归。测试直接导入 v5，未替换 `sys.modules["rc3_v4.external"]`。分母是测试方法数，参数化子案例不另外计数。日志与证据见 `external_results/tests.log`、`external_results/SUMMARY.json`。

`python -B -m rc3_v5.external_example --output rc3_v5/external_results/demo`：生成 2 桌 4 任务的 synthetic 示例及诊断。完整表支持低/高两个高度、左右臂，Fixed3 与 Detour 均得到独立记录验证通过的方案；限制为固定低高度或仅左臂时，该人工约束表无法覆盖全部任务；审计拒绝其中一个任务时不输出完成方案。这证明接口能传递这些约束，不能证明实际机器人有相同收益。

下一项真实验证所需输入：机器人模型及版本、升降范围/坐标变换、两臂关节限位、末端姿态/任务容差、碰撞几何与环境模型、离线可达/审计结果和代表性服务点见证。接口与合成测试不替代这些物理输入；资产盘点另见本版机器人资产报告。

验证非默认停靠罚项时必须独立提供同一请求值，例如 --penalty 1.5。验证器不信任计划自报的罚项；该绑定在本版保留，并纳入上面的 9 项原测试。
