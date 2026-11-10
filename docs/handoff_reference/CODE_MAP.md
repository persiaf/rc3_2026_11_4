# 代码地图与实际阅读路径

## 1. 新交接接口（本次新增，不是最新科研模型）

| 文件 | 输入／输出 | 作用及边界 |
|---|---|---|
| contracts.py | scene/F/A/D/导航证据 → Parent/TaskView | 验证形状、单位约定、连通性、目标窗口；数组只读，身份可追溯 |
| environment.py | TaskView＋局部动作 → state/reward/plan | 正覆盖增益构造；0-based动作；当前raw奖励，无科研H |
| validation.py | view/plan → 验证证书 | 标量重核覆盖、同臂F&A、路径长度、膨胀矩形碰撞 |
| io.py | Parent ↔ JSON/NPZ | 禁止pickle，内容哈希，拒绝覆盖 |
| repository_bridge.py | 已准备RC3 parent或旧归档scene → Parent | 新身份转换，不冒充旧view identity |
| maps.py | HWC地图与机器人像素位置 → global/local | 新实现的观察处理示例；不能代替精确几何约束 |
| learning_math.py | logits/mask/advantages等 → 可微损失 | 数学助手；DDQN/PPO非完整训练器 |
| cli.py | 参数/现有场景包 → demo计划 | 新regular百桌工程示例，不属于科研测试 |

## 2. 当前研究仓库（按固定commit获取）

数据层：rc3_fast/scenes.py、solver.py，rc3_baseline/baseline.py，rc3_global_local_impl_review/runtime_guards.py。
问题层：rc3_global_local/problem.py、environment.py。
模型层：rc3_global_local/policy.py。
交付层：rc3_global_local/terminal.py、validation.py。
训练层：rc3_global_local_trainer/engine.py、checkpoint.py、rng.py、data.py。
评价层：rc3_s3_evaluator/，rc3_s3_connectivity_20261109/scope.py。
最新已完成机制：rc3_gl_v4_20261109/、rc3_gl_v5_20261109/。

建议完整调用顺序：
scene + task draw → F/A + navigation → immutable parent → target window → legal-mask environment → frozen actor / training rollout → complete return/H → export → independent validate → aggregate statistics。

注意：当前原生环境动作node从1起，0兼作局部符号入口/出口；新交接接口动作从0起且显式存entry/exit。适配器必须显式转换。

## 3. 原论文代码对应位置

src/DDQN/Agent.py：在线/目标Q、global/local卷积、采样与训练。
src/DDQN/ReplayMemory.py：回放存储。
src/DDQN/Trainer.py：交互训练调度。
src/CPP/State.py / Physics.py / Grid.py / Rewards.py：相机覆盖任务、运动预算与终止。
src/base/GridActions.py：通用六动作、CPP使用NoHover五动作。
main.py / main_mc.py / main_scenario.py：训练、随机评价、固定场景。

不要直接将CPP的camera.computeView作为机械臂可达函数。不得把LAND成功、视觉覆盖率原封不动当成本项目任务成功。DDQN的固定维度输出也不能直接表示可变数量停靠候选。

## 4. 旧MATLAB接口

本包不包含旧main_mobile_manipulation_planning.m。先由原项目输出scene/F/A/D/路径前驱，再使用matlab/export_rc3_handoff.m导出明确schema。调用python scripts/convert_matlab.py input.mat output_dir转换。若只有D、没有导航前驱，不能宣称生成了绕障完整折线，应补原规划器路径产物。
