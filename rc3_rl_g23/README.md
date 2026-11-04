# G23 封顶试验交付

已实际完成规则接口复现、三个种子两轮价值策略改进、180次完整测试和132条训练标签的实际分支复核。累计3694/8192次真实修复调用、3840次优化更新。主目标未过，按任务书关闭当前局部窗口＋STOP作为DRL核心，不追加同方向训练。

完整结果见REPORT_G23.md；原始规格已从用户ZIP读取并保存在inputs，未使用库链接替代本地完整文件。所有原G2/G21/G22继续冻结。

## 复核命令（不新增修复）

```powershell
.\rc3_learning\.venv\Scripts\python.exe -B -m unittest rc3_rl_g23.test_contract -v
.\rc3_learning\.venv\Scripts\python.exe -B rc3_rl_g23/review_saved_evidence.py
.\rc3_learning\.venv\Scripts\python.exe -B -m rc3_rl_g23.analyze
```

第二项仅反序列化可信本地张量和保存数据，不构造模型或前向，不调用规划器。第三项以已有场景种子重建输入F并检查保存路径，不进行导航求解／修复。历史测试不混计为本轮重新执行。

**trial.py和verify_labels.py都会执行真实修复，不是零查询检查。当前已有完整记录，禁止自动重跑。** 其中标签复核新增1497次调用已计入3694总账。主程序拒绝隐式恢复／覆写原资源账；复现新一轮必须另存工作副本并重新计数，不能删除历史记录假装首次运行。

## 数据及模型入口

- outputs/protocol.json及sha：先于真实调用固定的范围和资源；EXPERIMENT_PLAN.md解释统计和止损。
- outputs/resource.jsonl、resources.json：所有原规则、对齐、训练、测试和复核调用；模型之外的查询同样计数。
- outputs/alignment.json：12场景192次原修复、104步完全匹配前缀。
- outputs/seed*_round*.pkl.gz：只对实际查询首动作的训练数据，明确pi和h／W，未知动作不补值。
- outputs/verified_labels.json.gz：额外实际复核的132条目标、2067条后缀转移和STOP／终验账。
- outputs/model_*：每种子四类策略共12份权重，另有3份MC0初始化副本；不计成15个独立训练模型。Value类位于learning.py，加载torch payload的state即可；均未推荐部署。
- outputs/test_results.json、plans/、analysis.json：180份最终路径及每种子结果、场景级区间和超时；只有6个独立测试场景。
- outputs/runtime_metadata.json：运行后补记机器型号／软件，明确其没有在原协议预登记。
- EXPERIMENT_RESULT_TO_CLAIM.md、paper/review-traces：独立审查和只读重算；figure为预登记门槛示意。

增量包需与原G2、G21、G22解压目录并列使用；依赖文件和归档SHA在outputs/freeze_before.json。不会替换旧文件或旧门禁。模型常驻CPU完整计时，软截止不构成硬实时保证；真实标定和IK仍暂缓。

策略推理使用 `Value.select(env)`：该接口将STOP相对优势精确设为0，只在最大非STOP值>0时继续。`forward`用于查询动作的回归，STOP槽没有监督，不能直接对其未经覆盖的原始输出做argmax来替代策略接口。
