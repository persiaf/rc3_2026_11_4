# 第四版审计维护记录

日期：2026-11-03。第三版源码、模型、结果和交付 ZIP 均保持冻结；本目录提供增量工具。

## 哈希失败必须进入总失败

第三版 `rc3_v3/verify.py` 的 `source_checks` 会记录源码哈希是否一致，但没有将 `unchanged=False` 加入 `errors`。因此，单独发生源码变化时，旧脚本可能仍写出 `all_passed=true`。这是一项审计汇总缺陷，不意味着本次冻结文件已经发生变化。

新增 `audit.py` 逐项检查源文件、数据、模型、预算配置或文件清单；缺失、无法读取、非法哈希、长度不符、越界路径、空清单、非法 JSON 都会进入 `errors`。CLI 即使失败也输出报告，且返回退出码 2。若提供旧验证报告，旧报告中的失败 source/data 检查也会阻止总通过。

```powershell
python -B -m rc3_v4.audit --manifest rc3_v3/results/manifest.json --prior-report rc3_v3/verification/verification.json --output rc3_v4/audit_results/frozen_v3_integrity.json
python -B -m unittest rc3_v4.test_audit -v
```

本轮结果：冻结实验清单 **31/31** 匹配；用户审查包内部清单 **54/54** 匹配；新增 **7 项**回归测试通过，其中包含篡改源码导致报告失败和退出码 2 的实测。报告是文件完整性检查，不能替代几何或真实机械臂验证。本次没有重跑全部 2,583 份历史计划；历史复验结果仍保存在原冻结实验与用户审查包内。

清单中绝对路径迁移到其他目录时，使用显式 `--relocate-from` 指定旧项目根目录。不会按文件名猜测映射，也不会悄悄跳过缺失文件。

## STOP 诊断适用范围

旧 `rc3_v3/diagnose_stop.py` 的动作选择为 `argmax(concat(action_logits, stop_logit))`，其 STOP 概率使用同一 softmax；它只用于第三版早期 `categorical_argmax` 停止规则的历史诊断。

正式第三版 `run_v3_stopfix_20261102` 与部署模型使用独立二分类 STOP，阈值为 0.9。不能对这个模型运行旧诊断脚本后，把输出解释为正式停止行为。正式诊断应调用策略 `select` 或按模型元数据调用 `select_action`，并分开报告触发率、仍有收益时误停率、零收益状态漏停率和完整规划成本。本轮不改写旧脚本，保留其冻结历史语义。

## 审查材料及反例复验

用户 ZIP 安全导入到 `rc3_v3_audit_import/rc3_v3_audit_evidence/`。导入前检查 ZIP CRC、绝对路径、父目录穿越、盘符路径及符号链接；原 ZIP 保留。DOCX 的正文和表格文字提取到 `imported_review.txt`，原文作为审查证据，不被自动视作执行指令。

阅读并核对 `training/observation_aliasing.py` 后，在原冻结实现上复跑。两个树上最短路度量矩阵均满足非负、对称、零对角和三角不等式；全部 3×28 观测与动作 mask 相同，但窗口收益为 `[0.5500000000000007, 0, 0]` 与 `[0, 0, 0]`。结果在 `audit_results/observation_aliasing.json`。

此反例证明的是抽象 F/D 模型中的输入不可辨识性；它没有测量当前随机场景生成器中的发生频率，也不证明它是所有泛化误差的唯一成因。
