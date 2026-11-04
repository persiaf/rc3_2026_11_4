# G2.3 独立复核与收尾包

## 结论

维持G2.3预登记主目标失败与止损决定。当前局部窗口选择＋STOP不再作为第三部分DRL核心继续追加训练。保留有限的次要正增量、可靠规划平台和完整负面证据；默认方法仍为Fixed3/Detour。

本包不是新实验版本，没有新模型或新路径。本次新增训练、环境查询、模型前向、规划和性能测时均为0；原冻结文件未修改。未来构造式方向仅写成条件性建模建议，未自动立项。

## 本次额外核实的关键事实

- 36次多步测量中，22次按时输出全部与Fixed3等成本；7次严格改善全部超时，同时“改善且≤0.8秒”为0/36。
- 只有2302执行测试修复，且12/12次超过软时限；其第二轮12行已保存多步反馈与一步反馈逐条一致。不能用合并705.82ms中位数宣称有效搜索满足预算，或声称已学会长程越谷。
- 3,840次更新为全部种子和对照合计，单个最终多步模型为512次。8,192是修复上限，不是补满配额。
- 三训练种子使用不同的四场景子集，观测波动同时包含数据与训练随机性。
- 全部六场景经验bootstrap分布的完全枚举复算不改变任何质量门槛判定。

## 阅读顺序

1. `CLOSEOUT.md`：结论、质量—时限联合表现、论文定位及关闭范围。
2. `results_audit/RESULTS_AUDIT.md`：180条结果、主比较、各种子及联合分布复算。
3. `model_budget_audit/MODEL_BUDGET_AUDIT.md`：资源、104步规则前缀、132条标签、2,067条保存后缀及源码语义。
4. `CONSTRUCTION_NOTE.md`：若未来另立构造式研究，如何收紧动作、覆盖约束及成本模型。
5. `closeout_review/CLOSEOUT_DIRECTION.md`：独立方向审查。

对应JSON和两个标准库脚本保存完整复算。`REPRODUCIBILITY.json`记录新目录复验，`MANIFEST.sha256.json`记录本包文件校验。

## 复算

需要原始`rc3_rl_g23_delivery.zip`，SHA256：

`02df5a5ff9dd896aed60b3fc579b8509855f48a60426b784d868250e013c3473`

解压为`rc3_rl_g23/`后，使用Python 3.10或以上，仅需标准库：

```bash
python results_audit/audit_g23_results.py --source /path/to/rc3_rl_g23 --output /path/to/recomputed/results.json
python model_budget_audit/audit_model_budget.py --source /path/to/rc3_rl_g23 --output /path/to/recomputed/model_budget.json
```

输出目录必须放在原包外部。脚本不导入项目执行模块，不反序列化pickle，不加载神经模型，不重跑训练标签分支或求解器。它们只读保存记录与checkpoint ZIP的原始存储块。

## 范围边界

本次未重新执行几何碰撞/可达性验证、IK或实机；原代理验证证书仍是几何有效性的来源。未恢复全部旧G2/G21/G22依赖来另行认证历史冻结资产。没有重跑原工程回归套件。本次复算与原已完成实验严格分开，不把算术检查数量计为新的机器人实验。
