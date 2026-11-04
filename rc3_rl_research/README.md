# G0/G1：可达性约束下的预算感知多步搜索

这是独立新增的研究模块。当前交付的是可重放环境、非单调服务配置替换和有限前瞻诊断；**没有训练 PPO，没有新神经网络权重**。原 v4/v5/v5.1 及旧模型、结果保留。最终证据和阶段判断见 `REPORT_G01.md`、`EXPERIMENT_RESULT_TO_CLAIM.md`。

## 安装位置与依赖

将本目录放在冻结 v5 解压根目录中，与 `rc3_baseline/`、`rc3_fast/`、`rc3_improve/`、`rc3_v3/`、`rc3_v5/` 并列。补充包依赖这些原有模块，不是独立替代整套 v5。Python ≥3.10、NumPy、SciPy；重画机制曲线另需 matplotlib。当前环境验证使用 Python 3.13.5、NumPy 2.1.3、SciPy 1.15.3。G0/G1 不需要 PyTorch 或 GPU。

在上述根目录执行：

```powershell
python -B -m unittest rc3_rl_research.test_proposal rc3_rl_research.test_env rc3_rl_research.test_adapter -v

python -B -m rc3_rl_research.runtime_rl --scene rc3_rl_research/outputs/g0/case_01/scene.json --method greedy --horizon 16 --work-budget 2000000 --output rc3_rl_research/outputs/my_run.json

python -B -m rc3_rl_research.runtime_rl --replay rc3_rl_research/outputs/my_run.json

python -B -m rc3_rl_research.runtime_rl --replay rc3_rl_research/outputs/g0/case_01/selected_replay.json
```

`--method random` 为同菜单均匀随机（含 STOP）；`--method alns` 为简化的自适应算子权重参考，**尚不是充分调优的强 ALNS**。`greedy` 以廉价 detour 特征排序，不免费预求所有动作 DP。三者共享动作构建、掩码、候选上限、修复器、接受界、费用和验证。外部约束输入须加 `--external`，规划全过程使用 F∩A。CLI 不内置真实 IK。

## 新环境做了什么

`adapter.py` 独立复制输入，F 和 D 为不可写数组，先获取经验证的 Fixed3 初解。`repair_proposal.py` 的 improve 保留旧连续窗口即时改进语义；replace 移除一个停靠，仅在本次局部候选池禁用它，允许产生较差但可行的方案。必须重做的任务来自**整个保留路线**的覆盖计数。全局规范化和用臂分配继续使用同一有效可达关系。

`rl_env.py` 分离 anchor、current、best。候选必须经过独立方案验证，且满足 `J(candidate) ≤ J(best) + 0.02 J(anchor)` 才能成为 current；best 只在新纪录时更新。终止重新认证 best，失败返回已认证 anchor，并修正最终质量奖励。该认证针对给定模型，不代表实机执行验证。

每步主奖励为 `100 × (old_best − new_best) / max(anchor, 1e-8) − 0.01 × work / 10000`。可选势函数只作为差分项，真正终止的势取 0。STOP、horizon 和预算结束是真终止；rollout 收集截断不结束搜索，保留后继值 bootstrap。不存在“恢复原来的好解”就重复领取质量奖励的机制。

完整状态记录当前/最好/初解、预算、全部尝试历史、RNG、抽样时的有序菜单与 mask。`state_dict/load_state_dict` 支持中间状态；`rollout_record` 保存抽样动作的原始概率、价值和菜单，并检查其与实际 transition 一致。当前没有图策略网络及 PPO 更新器，G1 的观测是明确可用的工程接口，并不声称其压缩特征是完备 Markov 表示。

## 工作量与时间

菜单工作计数是实际评估的局部上下文数加候选扫描数；提案计数是尝试、候选扫描、实际有限 DP 状态和入口/转移/出口条目数。单位用于可复核代理计费，不是 FLOPs，也不等于秒。菜单只构建一次，重复 observe 幂等，其费用归下一 step 或终止事件。失败查询照常计费。

**这是原子操作软工作预算**：菜单或一个已启动的修复可能越过剩余额度，之后立即结束并报告 `work_overrun`。可选 `--time-limit-ms` 也是软期限，初解、原子修复和强制终验可能使完整时间超过上限。完整场景到方案时间与分阶段耗时有记录，本轮不据诊断 oracle 的耗时宣称部署加速。

没有 proposal 缓存或禁忌屏蔽；历史用于状态、上下文指纹和重放。贪心可能重复选无改进窗口，本轮将其明确定位为接口参考。后续构建强规则基线时，需要在所有比较策略中统一处理重复尝试和候选裁剪，重新声明公平协议。

## 重放或重做诊断

现有协议为 32 个新百桌场景（11 regular、11 mixed、10 blocked），每层至多 8 个分支（含 STOP）、全部深度 3。种子在生成场景前排除全部可发现历史种子记录；算法实现另有 SHA256 绑定。所有树查询和完整菜单分别在各 case 的 `queries.jsonl.gz` 与 `menus.jsonl.gz`，可用 Python 标准库 gzip 读取。

```powershell
# 已完成记录会复用；汇总不会重新执行已存在案例。
python -B -m rc3_rl_research.diagnose_g0

# 对全部 32 条选定轨迹真正重新执行与核对。
python -B -m rc3_rl_research.tools.verify_delivery
```

若要从头重做同一批诊断，创建新的输出目录，仅复制已交付的 `protocol.json`、`protocol.sha256` 和 `implementation.json`，再传 `--directory 新目录`。**不要运行 prepare_g0 重新抽样来声称复现原协议**。可用相同 `--shard-count 4`、互异 `--shard-index 0..3` 并行各自案例，最后用不分片的命令汇总。并行诊断耗时不是单策略性能证据。

协议只查询共享菜单的前 7 个合法非 STOP 条目及 STOP，属于有明确计算账本的有限 oracle。它既不是全动作最优解，也不是部署策略。32 个场景是开发样本，后续网络选择后还需独立确认集。

## 产物与范围

- `OPEN_SOURCE_NOTES.md`：固定提交的 NeuOpt、CleanRL、RL4CO 代码依据及不能直接照搬的语义。
- `INTERFACES.md`、`design/`：实现合同、用户方案提取、实施计划。
- `outputs/g0/`：预注册协议、源场景、全部查询、逐场景结果和可执行重放。
- `outputs/integration/`：真实小场景、共同约束反例与三个共享接口策略重放；这些不是 G0 新场景。
- `outputs/test_env_artifacts/`：故障注入及合成夹具；其中故意篡改的文件用于拒绝测试，不应作为有效输出。
- `outputs/development_case00_before_replay_fixes/`：最终冻结前的首次 case00 调试运行，完整保留；正式统计只用 `outputs/g0/` 的 32 个场景。其查询和重放成本在报告中另列，不计为独立新场景。
- `outputs/` 中原版本完整性记录、回归日志、独立审阅及验证回执。

后续训练应先明确共享动作强规则、工作量惩罚的尺度及候选菜单质量，再开展同初始化冻结 BC、继续 BC、BC＋PPO 的控制比较。本阶段不改变 Fixed3／Detour 的正式默认地位。
