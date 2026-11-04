# G2.2 独立复核与下一轮研究规格

## 结论

G2.2 的主表、动作账本与保存方案内部算术复算一致。冻结学习策略找到了部分有效操作，但还没有证明整体优于强规则，也没有新增 PPO 收益证据。下一轮建议只保留一次有明确止损条件的“规则引导、预算条件化的多步价值策略改进”试验。

本包是独立补充，不包含新模型或新路径。本次新增训练、环境查询、模型前向、规划、性能测时均为 0。原始 G2.2 包与历史冻结结论未修改。下一轮规格是研究建议，不是已经执行的登记协议，也不改变原 G2.1 门禁的失败状态。

## 新发现

1. 原 Detour16 在 12/12 场景取得比两个模型更低的 J，但其 192 次修复和学习策略的 13/15 次提案不属于相同动作与工作协议，不能称为等工作量优势。
2. CE＋收益相对实际根 STOP 的平均净回报增量为 +0.045764，但 7/12 场景为负，中位数为 −0.090410。早停 Detour 也存在均值为正、8/12 场景为负的情况。
3. 在固定已访问动作序列上，提前 STOP 只能保持或减弱最终路径质量。计入待付菜单、假设前缀终验不回退的最佳已访问 STOP 是事后诊断：CE、收益、早停规则的平均 ΔR 分别为 0.077053、0.142247、0.258430，不能作为模型成绩。
4. case 1 的收益模型第一步 current/best 均不变；第二步有效动作在根菜单中已存在。该例未证明首步必要，更未证明成功越谷。G2.2 真正 current 变差的三次操作均未随后改善 best。
5. 菜单占已记工作代理的 96%–99%，但完整时间还大量花在准备、Fixed3 锚点和验证。R_W 改善不等于毫秒改善。
6. 原 Detour 保存证据只有内部操作索引，尚不足判定其有效操作是否处于学习菜单中。首先需要核对候选可表达性与相同转移语义。

## 文件导航

- `results_audit/RESULTS_AUDIT.md`：72 方案、六方法、成对比较、时间边界与论文措辞。
- `results_audit/RESULTS_RECOMPUTED.json`：所有逐场景复算、配对统计与输入哈希。
- `trajectory_audit/TRAJECTORY_AUDIT.md`：48 条共享轨迹、最佳已访问 STOP、case 0/1/7/8 分析。
- `trajectory_audit/trajectory_audit.json`：逐步证据、事后诊断与边界。
- `timing_audit/RECORDED_TIMING.md`、`RECORDED_TIMING.json`：原计时字段和工作代理的分别汇总。
- `direction_review/RECOMMENDATION.md`：优先方向、价值定义、时序目标和止损条件。
- `NEXT_RUN_SPEC.md`：面向后续代码实现的有限研究规格及验收逻辑。
- `REFERENCES.md`：已查看的原始论文与官方仓库、可以借鉴的部分。
- `REPRODUCIBILITY.json`：本补充包从新目录运行三个只读脚本的复验结果。
- `MANIFEST.sha256.json`：本补充包文件校验。

## 复算方式

需要 Python 3.10 或以上，仅依赖标准库。原 G2.2 包 SHA256：

`e99f9f8d1a8d46f1d7bc728391237f1000e652f1bcb1f11f79b57415eeb627db`

把原始 `rc3_rl_g22_delivery.zip` 解压得到 `rc3_rl_g22/`；本包解压得到 `rc3_g22_review_and_g23_spec/`。在本包目录执行下列命令，将路径替换成实际位置；不需要旧 G2/G2.1 的 Python 环境或模型。

```bash
python results_audit/audit_g22_results.py --source /path/to/rc3_rl_g22 --output /path/to/recomputed/results.json
python trajectory_audit/audit_trajectories.py --source /path/to/rc3_rl_g22 --output /path/to/recomputed/trajectories.json
python timing_audit/audit_recorded_timing.py --source /path/to/rc3_rl_g22 --output /path/to/recomputed/timing.json
```

这些脚本读取既有证据，不调用规划器或环境。没有独立恢复旧 teacher.pkl 重建“已查询/未查询/未见状态”分类，也没有重新运行几何碰撞、真实可达性或 IK 检查。72 是方案文件数、12 才是场景数。

## 对第三部分的定位

建议暂定研究题目为“面向移动双臂任务路径规划的约束关联图与预算感知深度价值策略”。可复用现有候选停靠、分臂可达关系、局部修复和验证模块；创新有效性必须来自多步强化学习相对强规则、同结构一步学习、初始化模型的独立增量。独立验证保证的是声明代理模型内的输出约束，不能代替真实机器人验证；已有工程优化不能全部归为深度强化学习贡献。
