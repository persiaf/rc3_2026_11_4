第三版独立复核与研究收敛证据包
审计日期 2026-11-03

一 本包是什么

本包配合 rc3_v3_research_review.docx，保存本轮新增的实验统计、训练诊断、独立方案复验、观察信息反例及一个验证集规则试验。
没有修改原 rc3_v3_delivery.zip 的源码、数据或权重，没有重新训练模型，没有将追加诊断并入原 60 个测试场景。
本包不是新的部署版本；继续部署原 v3 的 Fixed3。需要更低成本时可选择原有有预算的 Detour 修复。新增 two-stop 规则未优于现有方案，不建议部署。

二 关键结论

1. 60 场景、861 组方法预算记录、2,583 份重复计划；全量独立复验通过，全部最终成本不高于认证后的落盘 Fixed3。
2. 工程提速成立：18 验证场景 141.56 -> 77.36 ms，成本不变；这是工程优化的贡献。
3. 主 54 例、8 次修复，MLP 相对 Fixed3 降本 0.78645%，相对 Detour 仅 0.06015%，配对区间跨零。
4. 部署模型为收益监督 MLP，14,596 参数，59,744 字节，不是新一轮 PPO。
5. 训练与验证初解排序差距明显；现有 28 维观察可以将真实收益不同的两个抽象 F/D 问题映射成完全相同输入。
6. 单/双候选评分在原 18 验证例、同机一次诊断中没有得到更好的质量—时间表现，完整保留负结果。
7. 真实 IK、升降耦合、双臂同步与互碰仍未验证；规划与独立验证器需要同步升级可达依据。

三 目录

experiments/             按原始逐次记录独立重算的统计与图表
training/                训练、标签、STOP、一步排序与信息反例
reliability/             全量落盘方案检查与便携推理复演
two_stop_diagnostic/     18 原验证场景上的新规则探索性诊断
VERSION_SCOPE.json       源 ZIP 与本轮工作范围、环境、文件数量
MANIFEST.json            包内文件 SHA-256（不包含自身）

每个目录内的 *_REVIEW.md / *_AUDIT.md 或 README.md 为详细中文说明。
部分历史 JSON 中保留了审查当时的路径，代表证据来源，不要求用户拥有同名路径。
profile_policy_context.py 是本机微测记录脚本，仍保留审查工作目录路径；不是跨平台复验主入口，不能用其局部速度比代替完整规划加速。

四 复现入口

先将原 rc3_v3_delivery.zip 解压，定位包含 rc3_v3、rc3_fast、rc3_improve、rc3_baseline 的目录。
以下 Windows 示例假设原工程在 D:\rc3_v3_project，证据包在 D:\rc3_v3_audit_evidence。
请替换成实际路径，不要将 source-root 指向 rc3_v3 子目录。

需要 NumPy、SciPy；画图另需 Matplotlib，模型线程控制可用 threadpoolctl。审查使用 Python 3.12.14、NumPy 2.3.5；不需要 Torch 完成以下 NumPy 诊断与复验。

最快复现新信息反例：

python D:\rc3_v3_audit_evidence\training\observation_aliasing.py --source-root "D:\rc3_v3_project" --output-dir "D:\rc3_review_output\aliasing"

重算原始统计，再重算工程及分量，最后画图：

python D:\rc3_v3_audit_evidence\experiments\audit_statistics.py --source-root "D:\rc3_v3_project" --output-dir "D:\rc3_review_output\statistics"
python D:\rc3_v3_audit_evidence\experiments\audit_engineering.py --source-root "D:\rc3_v3_project" --output-dir "D:\rc3_review_output\statistics"
python D:\rc3_v3_audit_evidence\experiments\draw_audit.py --data-dir "D:\rc3_review_output\statistics"

重新独立验证原始全部落盘计划（不会重新规划所有路线）：

python D:\rc3_v3_audit_evidence\reliability\audit_saved_plans.py --source-root "D:\rc3_v3_project" --output "D:\rc3_review_output\plans"

抽样便携复演与失败处理：

python D:\rc3_v3_audit_evidence\reliability\replay_portable.py --source-root "D:\rc3_v3_project" --output "D:\rc3_review_output\replay"

完整训练与验证状态诊断（额外枚举标签，计算比单次规划多）：

python D:\rc3_v3_audit_evidence\training\audit_training.py --source-root "D:\rc3_v3_project" --output-dir "D:\rc3_review_output\training"

重跑 three-method 验证集规则诊断：

python D:\rc3_v3_audit_evidence\two_stop_diagnostic\run_diagnostic.py --source-root "D:\rc3_v3_project" --output "D:\rc3_review_output\two_stop"

注意：最后一个脚本中的 rule_coverage_diagnostic.json 为审查后对 details.json 的派生汇总，主脚本生成 details.json、case_metrics.csv、summary.json；原派生汇总随本包保留。

五 统计与可靠性的边界

成本 J = 路程 + 0.75 × 停靠数，不等于执行时间。
原完整规划时间包括参考构造、特征、推理、修复和最终认证，预处理、加载和写盘单独考虑。
原时间来自上传 Windows 记录；新增规则诊断和微测来自本轮 Linux 环境，不能相互拼接成速度改善结论。
2,583 份是重复方案，独立场景为 60；400 桌只有 3 例。20,000 次配对 bootstrap 以场景为单位，未包含训练种子不确定性或多重比较修正。
旧 categorical STOP 诊断脚本不可直接用于新的 binary-threshold 模型。
当前独立验证仍验证球壳/扇形代理约束。高覆盖率不等于真实机器人关节轨迹可行。
本轮实跑 16 项非 Torch 单测，不声称复跑原交付中全部 Torch 训练测试。

六 后续优先级

第一：真实机械臂服务约束与独立验证依据接通，保持在线高层二维。
第二：小规模精确解或下界、强覆盖路线对照，明确目标函数权重的意义。
第三：只做一次加入候选覆盖与导航距离的结构消融，在新冻结测试上按预设质量/时间门槛决定是否继续学习。
不要将本轮追加的负结果重新命名为已验证优越方法，也不要把旧测试反复用于挑模型。
